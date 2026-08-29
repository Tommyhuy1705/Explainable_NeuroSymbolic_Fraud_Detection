"""Provenance, checksum and lineage contracts for stress-test notebooks."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.artifacts import capture_environment, sha256_file, stable_config_hash


STRESS_SOURCE_FILES: tuple[str, ...] = (
    "scripts/generate_stress_test_notebooks.py",
    "scripts/run_stress_test.py",
    "src/artifacts.py",
    "src/data/preprocessing.py",
    "src/logic/fraud_rules.py",
    "src/logic/predicates.py",
    "src/stress_testing/artifacts.py",
    "src/stress_testing/attribution.py",
    "src/stress_testing/candidate_rules.py",
    "src/stress_testing/contrastive.py",
    "src/stress_testing/data.py",
    "src/stress_testing/experiment.py",
    "src/stress_testing/matched_risk.py",
    "src/stress_testing/metrics.py",
    "src/stress_testing/predictors.py",
    "src/stress_testing/rule_audit.py",
    "src/stress_testing/selective_policy.py",
)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON ``null``.

    Python's default ``NaN``/``Infinity`` tokens are not valid JSON. Stress
    artifacts are intended for independent lineage validation, so they use the
    portable JSON grammar even when an estimator reports an undefined metric.
    """

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def current_commit(project_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(project_root).resolve()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stress_pipeline_fingerprint(
    project_root: str | Path,
    config_path: str | Path,
    *,
    source_files: Sequence[str] = STRESS_SOURCE_FILES,
) -> str:
    """Hash the exact stress source/config snapshot without touching 01-08."""

    root = Path(project_root).resolve()
    config = Path(config_path).resolve()
    paths = [root / relative for relative in source_files] + [config]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stress fingerprint inputs are missing: {missing}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stress_source_fingerprint(
    project_root: str | Path,
    *,
    source_files: Sequence[str] = STRESS_SOURCE_FILES,
) -> str:
    """Hash executable stress source independently of dataset configuration."""

    root = Path(project_root).resolve()
    paths = [root / relative for relative in source_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stress source fingerprint inputs are missing: {missing}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def output_checksums(
    destination: str | Path,
    output_files: Iterable[str | Path],
) -> dict[str, str]:
    root = Path(destination).resolve()
    checksums: dict[str, str] = {}
    for value in output_files:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Declared stress output is missing: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Declared output lies outside destination: {path}") from exc
        checksums[relative] = sha256_file(path)
    return dict(sorted(checksums.items()))


def write_stress_manifest(
    destination: str | Path,
    *,
    dataset: str,
    config_path: str | Path,
    project_root: str | Path,
    claim_eligible: bool,
    claim_blockers: Sequence[str],
    data_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_files: Iterable[str | Path],
) -> Path:
    root = Path(destination).resolve()
    config = load_yaml_config(config_path)
    payload = {
        "artifact_schema_version": 1,
        "artifact_type": "thesis_stress_test",
        "dataset": str(dataset),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(project_root),
        "config_path": Path(config_path).resolve().relative_to(Path(project_root).resolve()).as_posix(),
        "config_sha256": stable_config_hash(config),
        "stress_pipeline_fingerprint": stress_pipeline_fingerprint(project_root, config_path),
        "stress_source_fingerprint": stress_source_fingerprint(project_root),
        "claim_eligible": bool(claim_eligible),
        "claim_blockers": list(claim_blockers),
        "data_manifest": dict(data_manifest),
        "summary": dict(summary),
        "outputs": output_checksums(root, output_files),
        "environment": capture_environment(project_root),
    }
    return write_json(root / "stress_test_manifest.json", payload)


def write_stress_lineage(
    destination: str | Path,
    *,
    notebook_id: str,
    config_path: str | Path,
    project_root: str | Path,
    output_files: Iterable[str | Path],
) -> Path:
    root = Path(destination).resolve()
    config = load_yaml_config(config_path)
    payload = {
        "lineage_schema_version": 1,
        "notebook_id": str(notebook_id),
        "git_commit": current_commit(project_root),
        "config_sha256": stable_config_hash(config),
        "stress_pipeline_fingerprint": stress_pipeline_fingerprint(project_root, config_path),
        "stress_source_fingerprint": stress_source_fingerprint(project_root),
        "outputs": output_checksums(root, output_files),
    }
    return write_json(root / "stress_lineage.json", payload)


def validate_stress_lineage(
    lineage_path: str | Path,
    *,
    expected_dataset_manifest: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(lineage_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("lineage_schema_version", -1)) != 1:
        raise ValueError(f"Unsupported stress lineage schema: {path}")
    root = path.parent
    declared = payload.get("outputs")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"Stress lineage has no declared outputs: {path}")
    for relative, expected in declared.items():
        output = (root / str(relative)).resolve()
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Lineage output escapes its artifact root: {relative}") from exc
        if not output.is_file():
            raise FileNotFoundError(f"Lineage output is missing: {output}")
        if sha256_file(output) != str(expected):
            raise ValueError(f"Lineage checksum mismatch: {output}")
    if expected_dataset_manifest is not None:
        manifest = Path(expected_dataset_manifest).resolve()
        if manifest.parent != root:
            raise ValueError("Stress manifest and lineage must share one artifact root")
    return payload


def find_unique_stress_manifest(
    dataset: str,
    search_roots: Iterable[str | Path],
) -> Path:
    matches: set[Path] = set()
    for root_value in search_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        candidates = (
            [root]
            if root.is_file() and root.name == "stress_test_manifest.json"
            else root.glob("**/stress_test_manifest.json")
        )
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("dataset", "")).lower() == str(dataset).lower():
                matches.add(candidate.resolve())
    if not matches:
        raise FileNotFoundError(f"No stress-test manifest found for {dataset}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple stress-test manifests found for {dataset}: {sorted(str(path) for path in matches)}"
        )
    return next(iter(matches))
