"""Frozen predictor artifacts and reproducibility metadata for notebook handoff."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "xgboost",
    "lightgbm",
    "PyYAML",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_commit(project_root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(project_root or ".").resolve()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment(project_root: str | Path | None = None) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    torch_details: dict[str, Any] = {}
    try:
        import torch

        # Some repaired or vendor-managed environments can import torch while
        # lacking usable wheel metadata.  Preserve the runtime version in that
        # case so the provenance record remains complete.
        packages["torch"] = packages.get("torch") or str(torch.__version__)
        torch_details = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_details = {"cuda_available": False, "cuda_version": None, "gpu_name": None}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": current_git_commit(project_root),
        "packages": packages,
        "torch": torch_details,
    }


def write_frozen_reference_artifact(
    destination: str | Path,
    *,
    manifest: dict[str, Any],
    y_validation: np.ndarray,
    y_test: np.ndarray,
    validation_probability: np.ndarray,
    test_probability: np.ndarray,
    validation_raw_probability: np.ndarray,
    test_raw_probability: np.ndarray,
) -> tuple[Path, Path]:
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / "frozen_reference_artifact.npz"
    np.savez_compressed(
        artifact_path,
        y_validation=np.asarray(y_validation, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        validation_probability=np.asarray(validation_probability, dtype=float),
        test_probability=np.asarray(test_probability, dtype=float),
        validation_raw_probability=np.asarray(validation_raw_probability, dtype=float),
        test_raw_probability=np.asarray(test_raw_probability, dtype=float),
    )
    completed_manifest = dict(manifest)
    completed_manifest.update(
        {
            "artifact_file": artifact_path.name,
            "artifact_sha256": sha256_file(artifact_path),
            "y_validation_sha256": sha256_array(np.asarray(y_validation, dtype=np.int64)),
            "y_test_sha256": sha256_array(np.asarray(y_test, dtype=np.int64)),
        }
    )
    manifest_path = directory / "frozen_reference_manifest.json"
    manifest_path.write_text(json.dumps(completed_manifest, indent=2, default=str), encoding="utf-8")
    return artifact_path, manifest_path


def _candidate_manifest_paths(
    dataset_name: str,
    search_roots: Iterable[str | Path] | None,
) -> list[Path]:
    explicit = os.getenv(f"THESIS_{dataset_name.upper()}_ARTIFACT_DIR")
    roots = [Path(path) for path in (search_roots or [])]
    if explicit:
        roots.insert(0, Path(explicit))
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))
    candidates: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "frozen_reference_manifest.json":
            candidates.append(root)
        elif root.is_dir():
            direct = root / "frozen_reference_manifest.json"
            if direct.exists():
                candidates.append(direct)
            candidates.extend(root.glob("**/frozen_reference_manifest.json"))
    unique = {path.resolve() for path in candidates if path.exists()}
    matching: list[Path] = []
    for path in sorted(unique):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(manifest.get("dataset_name", "")).lower() == dataset_name.lower():
            matching.append(path)
    return matching


def load_frozen_reference_artifact(
    dataset_name: str,
    *,
    expected_config: dict[str, Any] | None = None,
    search_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    manifests = _candidate_manifest_paths(dataset_name, search_roots)
    if not manifests:
        raise FileNotFoundError(
            f"No frozen reference artifact found for {dataset_name}. Attach the corresponding "
            "benchmark notebook output or set THESIS_{dataset_name.upper()}_ARTIFACT_DIR."
        )
    if len(manifests) > 1:
        raise RuntimeError(
            f"Multiple frozen artifacts found for {dataset_name}; set the explicit artifact directory: "
            f"{[str(path.parent) for path in manifests]}"
        )
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_config is not None and manifest.get("config_sha256") != stable_config_hash(expected_config):
        raise ValueError("Frozen artifact config hash does not match the current dataset config")
    artifact_path = manifest_path.parent / str(manifest["artifact_file"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"Frozen artifact file is missing: {artifact_path}")
    if sha256_file(artifact_path) != manifest.get("artifact_sha256"):
        raise ValueError("Frozen artifact checksum mismatch")
    with np.load(artifact_path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if sha256_array(arrays["y_validation"].astype(np.int64)) != manifest.get("y_validation_sha256"):
        raise ValueError("Frozen validation-label checksum mismatch")
    if sha256_array(arrays["y_test"].astype(np.int64)) != manifest.get("y_test_sha256"):
        raise ValueError("Frozen test-label checksum mismatch")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
        **arrays,
    }


def assert_frozen_alignment(
    artifact: dict[str, Any],
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> None:
    expected_validation = np.asarray(y_validation, dtype=np.int64)
    expected_test = np.asarray(y_test, dtype=np.int64)
    if not np.array_equal(artifact["y_validation"], expected_validation):
        raise ValueError("Current validation split does not align with the frozen artifact")
    if not np.array_equal(artifact["y_test"], expected_test):
        raise ValueError("Current test split does not align with the frozen artifact")


def find_result_file(
    filename: str,
    search_roots: Iterable[str | Path],
    parent_hint: str | None = None,
) -> Path:
    matches: set[Path] = set()
    for root_value in search_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        direct = root / filename
        if direct.exists():
            matches.add(direct.resolve())
        if root.is_dir():
            matches.update(path.resolve() for path in root.glob(f"**/{filename}"))
    if parent_hint:
        matches = {path for path in matches if parent_hint.lower() in str(path.parent).lower()}
    if not matches:
        raise FileNotFoundError(f"Could not find required result file {filename!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files named {filename!r} found: {sorted(str(path) for path in matches)}")
    return next(iter(matches))
