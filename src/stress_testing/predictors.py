"""Versioned stress-test predictors with deterministic, auditable training.

This module intentionally does not reuse or rename the thesis project's
existing ``TabularResNet``.  ``TabularResNetV2`` is a separately versioned
architecture with learnable input gates and gated residual transformations.
Tree backends are exact in full runs: a missing XGBoost or LightGBM package is
an error, never an implicit change of model family.  A sklearn fallback exists
only for explicitly requested quick fixture tests.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import random
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Sequence

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when the requested scientific backend is not installed."""


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        raise BackendUnavailableError("TabularResNetV2 requires PyTorch; install the declared project dependency")
    import torch

    return torch


torch = _require_torch()
from torch import nn  # noqa: E402  (guarded optional import)
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class DeviceResolution:
    requested: str
    selected: str
    reason: str
    device_capability: str | None = None
    supported_architectures: tuple[str, ...] = ()


def resolve_safe_torch_device(requested: str | None = None) -> DeviceResolution:
    """Resolve a usable torch device, rejecting unsupported CUDA wheel/device pairs.

    Kaggle can expose a GPU (for example, an ``sm_60`` P100) while its installed
    wheel only contains kernels for newer architectures. ``is_available()`` is
    then true although the first model kernel fails.  We compare the device
    capability with the wheel's compiled architecture list before allocating a
    model and explicitly document a CPU fallback in provenance.
    """

    requested_name = "auto" if requested is None else str(requested).lower()
    if requested_name == "cpu":
        return DeviceResolution(requested=requested_name, selected="cpu", reason="cpu explicitly requested")
    if requested_name != "auto" and not requested_name.startswith("cuda"):
        raise ValueError("device must be None/'auto', 'cpu', or a CUDA device such as 'cuda:0'")
    if not torch.cuda.is_available():
        return DeviceResolution(requested=requested_name, selected="cpu", reason="CUDA is not available")

    try:
        device_index = 0
        if requested_name.startswith("cuda:"):
            device_index = int(requested_name.split(":", maxsplit=1)[1])
        major, minor = torch.cuda.get_device_capability(device_index)
        capability = f"sm_{major}{minor}"
        supported = tuple(str(value) for value in torch.cuda.get_arch_list())
        compiled_sm = tuple(value for value in supported if value.startswith("sm_"))
        if compiled_sm and capability not in compiled_sm:
            return DeviceResolution(
                requested=requested_name,
                selected="cpu",
                reason=f"installed torch wheel has no kernel image for {capability}",
                device_capability=capability,
                supported_architectures=supported,
            )
        selected = f"cuda:{device_index}"
        # A one-element allocation catches driver/runtime failures not exposed
        # by architecture metadata, before model state is moved.
        torch.zeros(1, device=selected)
        return DeviceResolution(
            requested=requested_name,
            selected=selected,
            reason="CUDA device and installed torch wheel are compatible",
            device_capability=capability,
            supported_architectures=supported,
        )
    except (AssertionError, RuntimeError, ValueError) as error:
        return DeviceResolution(
            requested=requested_name,
            selected="cpu",
            reason=f"CUDA preflight failed: {type(error).__name__}: {error}",
        )


class GatedResidualBlockV2(nn.Module):
    """Pre-normalized gated residual block with learnable residual strength."""

    def __init__(self, hidden_dim: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        expanded = int(hidden_dim) * int(expansion_factor)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.value_and_gate = nn.Linear(hidden_dim, expanded * 2)
        self.output_projection = nn.Linear(expanded, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization(inputs)
        values, gates = self.value_and_gate(normalized).chunk(2, dim=-1)
        transformed = torch.nn.functional.silu(values) * torch.sigmoid(gates)
        transformed = self.output_projection(self.dropout(transformed))
        return inputs + self.residual_scale * self.dropout(transformed)


class TabularResNetV2(nn.Module):
    """Tabular residual network v2 for a single binary fraud logit.

    Differences from the existing v1 model are structural rather than a class
    alias: per-feature learnable gates, GLU-like residual branches, learnable
    residual scaling, SiLU activations, and a direct normalized prediction
    head.  The architecture identifier is persisted with every trained model.
    """

    architecture_version = "2.0"

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 192,
        num_blocks: int = 4,
        expansion_factor: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1 or int(hidden_dim) < 4 or int(num_blocks) < 1:
            raise ValueError("input_dim, hidden_dim, and num_blocks must be positive (hidden_dim >= 4)")
        if int(expansion_factor) < 1 or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("expansion_factor must be positive and dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.input_gate_logits = nn.Parameter(torch.zeros(self.input_dim))
        self.input_projection = nn.Linear(self.input_dim, int(hidden_dim))
        self.input_normalization = nn.LayerNorm(int(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlockV2(int(hidden_dim), int(expansion_factor), float(dropout))
                for _ in range(int(num_blocks))
            ]
        )
        self.head_normalization = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.input_gate_logits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"expected a [rows, {self.input_dim}] feature tensor")
        # Multiplication by two initializes the sigmoid gates at identity scale.
        gated = features * (2.0 * torch.sigmoid(self.input_gate_logits))
        hidden = torch.nn.functional.silu(self.input_normalization(self.input_projection(gated)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.head_normalization(hidden)).squeeze(-1)


@dataclass
class PredictorBundle:
    """A trained but not yet calibrated/frozen stress-test predictor."""

    family: str
    backend: str
    model: Any
    feature_names: tuple[str, ...]
    seed: int
    provenance: dict[str, Any]
    training_history: tuple[dict[str, float], ...] = ()
    frozen: bool = False


@dataclass(frozen=True)
class FrozenPredictor:
    """A model, calibrator, and threshold frozen before explanation."""

    family: str
    backend: str
    model: Any
    feature_names: tuple[str, ...]
    seed: int
    calibrator: Any
    threshold: float
    provenance: dict[str, Any]
    frozen: bool = True

    def predict_raw_proba(self, features: np.ndarray, *, device: str | None = None) -> np.ndarray:
        return _raw_predict_probabilities(self, features, device=device)

    def predict_proba(self, features: np.ndarray, *, device: str | None = None) -> np.ndarray:
        raw = self.predict_raw_proba(features, device=device)
        calibrated = np.asarray(self.calibrator.transform(raw), dtype=float)
        if calibrated.shape != raw.shape or not np.isfinite(calibrated).all():
            raise RuntimeError("frozen calibrator returned an invalid probability vector")
        return np.clip(calibrated, 0.0, 1.0)

    def predict(self, features: np.ndarray, *, device: str | None = None) -> np.ndarray:
        return (self.predict_proba(features, device=device) >= self.threshold).astype(np.int8)


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _data_fingerprint(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("training features must be a non-empty two-dimensional matrix")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("training labels must be one-dimensional and row-aligned")
    if not np.isfinite(X).all() or not np.isfinite(y.astype(float)).all():
        raise ValueError("training data contains non-finite values; preprocess it before model fitting")
    if not set(np.unique(y).tolist()).issubset({0, 1, False, True}) or np.unique(y).size != 2:
        raise ValueError("imbalance-aware binary training requires both 0 and 1 labels")
    names = (
        tuple(f"feature_{index:04d}" for index in range(X.shape[1]))
        if feature_names is None
        else tuple(str(value) for value in feature_names)
    )
    if len(names) != X.shape[1] or len(set(names)) != len(names) or any(not value for value in names):
        raise ValueError("feature_names must be non-empty, unique, and column-aligned")
    return X, y.astype(np.int8, copy=False), names


def _resolve_tree_backend(family: str, *, quick_run: bool, allow_quick_fallback: bool) -> str:
    normalized = family.lower().replace("-", "_")
    package = {"xgboost": "xgboost", "lightgbm": "lightgbm"}.get(normalized)
    if package is None:
        raise ValueError("tree family must be 'xgboost' or 'lightgbm'")
    if importlib.util.find_spec(package) is not None:
        return package
    if bool(quick_run) and bool(allow_quick_fallback):
        return "hist_gradient_boosting_quick_fixture"
    raise BackendUnavailableError(
        f"requested backend '{package}' is unavailable; full stress runs never silently substitute a model"
    )


def _tree_defaults(backend: str, *, quick_run: bool) -> dict[str, Any]:
    estimators = 25 if quick_run else 600
    if backend == "xgboost":
        return {
            "n_estimators": estimators,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 2.0,
            "reg_lambda": 1.0,
            "n_jobs": -1,
        }
    if backend == "lightgbm":
        return {
            "n_estimators": estimators,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "n_jobs": -1,
        }
    return {"max_iter": 20 if quick_run else 300, "learning_rate": 0.08, "max_depth": 6}


def _train_tree(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
    quick_run: bool,
    allow_quick_fallback: bool,
) -> tuple[Any, str, dict[str, Any]]:
    backend = _resolve_tree_backend(family, quick_run=quick_run, allow_quick_fallback=allow_quick_fallback)
    settings = _tree_defaults(backend, quick_run=quick_run)
    settings.update(dict(config))
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    imbalance_weight = float(negatives / positives)
    if backend == "xgboost":
        from xgboost import XGBClassifier

        settings.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "random_state": int(seed),
                "scale_pos_weight": imbalance_weight,
            }
        )
        model = XGBClassifier(**settings)
        model.fit(X, y, verbose=False)
    elif backend == "lightgbm":
        from lightgbm import LGBMClassifier

        settings.update(
            {
                "objective": "binary",
                "random_state": int(seed),
                "bagging_seed": int(seed),
                "feature_fraction_seed": int(seed),
                "data_random_seed": int(seed),
                "deterministic": True,
                "force_col_wise": True,
                "scale_pos_weight": imbalance_weight,
                "verbosity": -1,
            }
        )
        model = LGBMClassifier(**settings)
        model.fit(X, y)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Translate the two common boosting controls so an explicit fixture
        # fallback remains usable with an XGBoost/LightGBM-style quick config.
        if "n_estimators" in settings and "max_iter" not in settings:
            settings["max_iter"] = settings.pop("n_estimators")
        allowed = set(HistGradientBoostingClassifier().get_params()) - {"random_state"}
        settings = {key: value for key, value in settings.items() if key in allowed}
        model = HistGradientBoostingClassifier(random_state=int(seed), **settings)
        sample_weight = np.where(y == 1, imbalance_weight, 1.0)
        model.fit(X, y, sample_weight=sample_weight)
    audit_settings = {
        key: value
        for key, value in settings.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return model, backend, {
        "imbalance_strategy": "scale_pos_weight" if backend != "hist_gradient_boosting_quick_fixture" else "sample_weight",
        "positive_class_weight": imbalance_weight,
        "training_parameters": audit_settings,
        "quick_fallback_used": backend == "hist_gradient_boosting_quick_fixture",
    }


def _neural_defaults(*, quick_run: bool) -> dict[str, Any]:
    return {
        "hidden_dim": 48 if quick_run else 192,
        "num_blocks": 2 if quick_run else 4,
        "expansion_factor": 2,
        "dropout": 0.15,
        "epochs": 2 if quick_run else 30,
        "batch_size": 128 if quick_run else 2048,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 5.0,
    }


def _train_neural(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
    quick_run: bool,
    requested_device: str | None,
) -> tuple[TabularResNetV2, str, tuple[dict[str, float], ...], dict[str, Any]]:
    settings = _neural_defaults(quick_run=quick_run)
    settings.update(dict(config))
    device = resolve_safe_torch_device(requested_device)
    set_deterministic_seed(seed)
    model = TabularResNetV2(
        input_dim=X.shape[1],
        hidden_dim=int(settings["hidden_dim"]),
        num_blocks=int(settings["num_blocks"]),
        expansion_factor=int(settings["expansion_factor"]),
        dropout=float(settings["dropout"]),
    ).to(device.selected)
    features_tensor = torch.from_numpy(np.array(X, dtype=np.float32, copy=True))
    labels_tensor = torch.from_numpy(np.array(y, dtype=np.float32, copy=True))
    dataset = TensorDataset(features_tensor, labels_tensor)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=min(int(settings["batch_size"]), len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.selected.startswith("cuda"),
    )
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    positive_weight = float(negatives / positives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device.selected)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_rows = 0.0, 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device.selected, non_blocking=True)
            batch_labels = batch_labels.to(device.selected, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            total_rows += len(batch_labels)
        history.append({"epoch": float(epoch), "train_loss": total_loss / max(total_rows, 1)})
    model.eval()
    audit_settings = {
        key: value
        for key, value in settings.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return model, device.selected, tuple(history), {
        "architecture": "TabularResNetV2",
        "architecture_version": TabularResNetV2.architecture_version,
        "imbalance_strategy": "BCEWithLogitsLoss.pos_weight",
        "positive_class_weight": positive_weight,
        "training_parameters": audit_settings,
        "device_resolution": {
            "requested": device.requested,
            "selected": device.selected,
            "reason": device.reason,
            "device_capability": device.device_capability,
            "supported_architectures": list(device.supported_architectures),
        },
    }


def train_stress_predictor(
    family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int = 42,
    quick_run: bool = False,
    allow_quick_fallback: bool = False,
    device: str | None = None,
) -> PredictorBundle:
    """Fit an exact model family on train rows only.

    Validation and test arrays are intentionally absent from the signature so
    callers cannot accidentally use them during model fitting. Calibration and
    F2 threshold selection are handled by :func:`fit_validation_protocol` in
    ``metrics.py`` after raw validation probabilities have been produced.
    """

    X, y, names = _validate_training_data(X_train, y_train, feature_names)
    normalized = str(family).lower().replace("-", "_")
    set_deterministic_seed(seed)
    settings = {} if config is None else dict(config)
    history: tuple[dict[str, float], ...] = ()
    if normalized in {"xgboost", "lightgbm"}:
        model, backend, training_audit = _train_tree(
            normalized,
            X,
            y,
            seed=seed,
            config=settings,
            quick_run=quick_run,
            allow_quick_fallback=allow_quick_fallback,
        )
        selected_device = "cpu"
    elif normalized in {"tabular_resnet_v2", "tabularresnetv2", "resnet_v2"}:
        normalized = "tabular_resnet_v2"
        model, selected_device, history, training_audit = _train_neural(
            X,
            y,
            seed=seed,
            config=settings,
            quick_run=quick_run,
            requested_device=device,
        )
        backend = "torch"
    else:
        raise ValueError("family must be xgboost, lightgbm, or tabular_resnet_v2")
    provenance: dict[str, Any] = {
        "predictor_protocol_version": "stress-predictor-v1",
        "family": normalized,
        "backend": backend,
        "seed": int(seed),
        "quick_run": bool(quick_run),
        "allow_quick_fallback": bool(allow_quick_fallback),
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "feature_names": list(names),
        "training_prevalence": float(y.mean()),
        "training_fingerprint_sha256": _data_fingerprint(X, y),
        "device": selected_device,
        "library_versions": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": _package_version("torch"),
            "xgboost": _package_version("xgboost"),
            "lightgbm": _package_version("lightgbm"),
        },
        **training_audit,
    }
    return PredictorBundle(
        family=normalized,
        backend=backend,
        model=model,
        feature_names=names,
        seed=int(seed),
        provenance=provenance,
        training_history=history,
    )


def _validate_prediction_features(predictor: PredictorBundle | FrozenPredictor, features: np.ndarray) -> np.ndarray:
    X = np.asarray(features, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(predictor.feature_names):
        raise ValueError(
            f"prediction features must have shape [rows, {len(predictor.feature_names)}] in fitted feature order"
        )
    if not np.isfinite(X).all():
        raise ValueError("prediction features contain non-finite values")
    return X


def _raw_predict_probabilities(
    predictor: PredictorBundle | FrozenPredictor,
    features: np.ndarray,
    *,
    batch_size: int = 8192,
    device: str | None = None,
) -> np.ndarray:
    X = _validate_prediction_features(predictor, features)
    if len(X) == 0:
        raise ValueError("at least one prediction row is required")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if predictor.backend == "torch":
        resolution = resolve_safe_torch_device(device)
        model = predictor.model.to(resolution.selected)
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), int(batch_size)):
                batch = torch.from_numpy(np.array(X[start : start + int(batch_size)], copy=True)).to(
                    resolution.selected
                )
                outputs.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
        probabilities = np.concatenate(outputs) if outputs else np.empty(0, dtype=float)
    else:
        probabilities = np.asarray(predictor.model.predict_proba(X)[:, 1], dtype=float)
    if probabilities.shape != (len(X),) or not np.isfinite(probabilities).all():
        raise RuntimeError("model returned invalid or misaligned probabilities")
    return np.clip(probabilities, 0.0, 1.0)


def predict_probabilities(
    predictor: PredictorBundle | FrozenPredictor,
    features: np.ndarray,
    *,
    batch_size: int = 8192,
    device: str | None = None,
    calibrated: bool = True,
) -> np.ndarray:
    """Predict aligned probabilities; apply calibration only for frozen models."""

    raw = _raw_predict_probabilities(predictor, features, batch_size=batch_size, device=device)
    if isinstance(predictor, FrozenPredictor) and calibrated:
        transformed = np.asarray(predictor.calibrator.transform(raw), dtype=float)
        if transformed.shape != raw.shape or not np.isfinite(transformed).all():
            raise RuntimeError("calibrator returned invalid or misaligned probabilities")
        return np.clip(transformed, 0.0, 1.0)
    return raw


def freeze_predictor(
    predictor: PredictorBundle,
    calibrator: Any,
    threshold: float | None = None,
    *,
    protocol_metadata: Mapping[str, Any] | None = None,
) -> FrozenPredictor:
    """Deep-copy and freeze a fitted predictor, calibrator, and F2 threshold.

    ``calibrator`` can be either a fitted object exposing ``transform`` or a
    frozen validation-protocol result exposing both ``transform`` and
    ``threshold``.  Attribution code accepts only the returned object.
    """

    if predictor.frozen:
        raise ValueError("predictor is already frozen")
    if not hasattr(calibrator, "transform"):
        raise TypeError("calibrator must expose a transform(probabilities) method")
    inferred_threshold = getattr(calibrator, "threshold", None)
    selected_threshold = inferred_threshold if threshold is None else threshold
    if selected_threshold is None or not np.isfinite(float(selected_threshold)):
        raise ValueError("a finite validation-selected threshold is required")
    if not 0.0 <= float(selected_threshold) <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if hasattr(calibrator, "frozen") and not bool(calibrator.frozen):
        raise ValueError("calibration protocol must be frozen before freezing the predictor")
    frozen_model = copy.deepcopy(predictor.model)
    if predictor.backend == "torch":
        frozen_model = frozen_model.cpu().eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
    frozen_calibrator = copy.deepcopy(calibrator)
    provenance = copy.deepcopy(predictor.provenance)
    provenance.update(
        {
            "frozen": True,
            "frozen_threshold": float(selected_threshold),
            "calibrator_type": type(calibrator).__name__,
            "validation_protocol": dict(protocol_metadata or getattr(calibrator, "provenance", {})),
        }
    )
    return FrozenPredictor(
        family=predictor.family,
        backend=predictor.backend,
        model=frozen_model,
        feature_names=predictor.feature_names,
        seed=predictor.seed,
        calibrator=frozen_calibrator,
        threshold=float(selected_threshold),
        provenance=provenance,
    )
