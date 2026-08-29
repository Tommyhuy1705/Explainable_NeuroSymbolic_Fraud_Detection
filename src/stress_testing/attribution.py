"""Finite, feature-aligned attributions for frozen stress-test predictors.

Tree explanations use the backends' native ``pred_contribs`` implementation.
Neural explanations use gradient times input for the uncalibrated fraud logit.
Both deliberately explain the fixed model score, while calibration and the F2
threshold remain frozen decision-layer metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .predictors import FrozenPredictor, resolve_safe_torch_device, torch


@dataclass(frozen=True)
class AttributionResult:
    """One attribution value per row and fitted feature."""

    values: np.ndarray
    base_values: np.ndarray
    raw_outputs: np.ndarray
    feature_names: tuple[str, ...]
    method: str
    provenance: dict[str, Any]

    def absolute_feature_importance(self) -> dict[str, float]:
        importance = np.abs(self.values).mean(axis=0)
        return {name: float(value) for name, value in zip(self.feature_names, importance, strict=True)}


def _matrix_and_names(
    predictor: FrozenPredictor,
    features: Any,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    inferred_names: tuple[str, ...] | None = None
    if hasattr(features, "columns"):
        inferred_names = tuple(str(value) for value in features.columns)
    X = np.asarray(features, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("attribution features must be a two-dimensional matrix")
    if X.shape[0] < 1:
        raise ValueError("attribution requires at least one row")
    supplied = tuple(str(value) for value in feature_names) if feature_names is not None else inferred_names
    effective_names = predictor.feature_names if supplied is None else supplied
    if effective_names != predictor.feature_names:
        raise ValueError(
            "attribution feature names/order do not match the frozen predictor; reorder using its feature_names"
        )
    if X.shape[1] != len(predictor.feature_names):
        raise ValueError("attribution feature count does not match the frozen predictor")
    if not np.isfinite(X).all():
        raise ValueError("attribution features contain non-finite values")
    return X, effective_names


def _fingerprint(features: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(features)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_result(
    values: np.ndarray,
    base_values: np.ndarray,
    raw_outputs: np.ndarray,
    *,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    attributions = np.asarray(values, dtype=float)
    bases = np.asarray(base_values, dtype=float).reshape(-1)
    outputs = np.asarray(raw_outputs, dtype=float).reshape(-1)
    if attributions.shape != (rows, columns):
        raise RuntimeError(
            f"attribution backend returned {attributions.shape}; expected {(rows, columns)}"
        )
    if bases.shape != (rows,) or outputs.shape != (rows,):
        raise RuntimeError("attribution base values/raw outputs are not row-aligned")
    if not (np.isfinite(attributions).all() and np.isfinite(bases).all() and np.isfinite(outputs).all()):
        raise RuntimeError("attribution backend returned NaN or infinite values")
    return attributions, bases, outputs


def _xgboost_contributions(predictor: FrozenPredictor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if predictor.backend != "xgboost":
        raise RuntimeError(
            "native XGBoost pred_contribs are unavailable because this is not an exact XGBoost backend"
        )
    import xgboost as xgb

    booster = predictor.model.get_booster()
    matrix = xgb.DMatrix(X)
    contributions = np.asarray(booster.predict(matrix, pred_contribs=True), dtype=float)
    if contributions.ndim != 2 or contributions.shape[1] != X.shape[1] + 1:
        raise RuntimeError("XGBoost pred_contribs did not return features plus one bias column")
    values = contributions[:, :-1]
    base_values = contributions[:, -1]
    raw_outputs = values.sum(axis=1) + base_values
    return values, base_values, raw_outputs


def _lightgbm_contributions(predictor: FrozenPredictor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if predictor.backend != "lightgbm":
        raise RuntimeError(
            "native LightGBM pred_contribs are unavailable because this is not an exact LightGBM backend"
        )
    contributions = np.asarray(predictor.model.predict(X, pred_contrib=True), dtype=float)
    if contributions.ndim != 2 or contributions.shape[1] != X.shape[1] + 1:
        raise RuntimeError("LightGBM pred_contribs did not return features plus one bias column")
    values = contributions[:, :-1]
    base_values = contributions[:, -1]
    raw_outputs = values.sum(axis=1) + base_values
    return values, base_values, raw_outputs


def _neural_gradient_times_input(
    predictor: FrozenPredictor,
    X: np.ndarray,
    *,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if predictor.backend != "torch":
        raise RuntimeError("gradient-times-input requires a torch-backed frozen predictor")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    resolution = resolve_safe_torch_device(device)
    model = predictor.model.to(resolution.selected).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("neural model parameters must be frozen before attribution")
    values: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    zero = torch.zeros((1, X.shape[1]), dtype=torch.float32, device=resolution.selected)
    with torch.no_grad():
        base_logit = float(model(zero).detach().cpu().item())
    for start in range(0, len(X), int(batch_size)):
        batch = torch.tensor(
            X[start : start + int(batch_size)],
            dtype=torch.float32,
            device=resolution.selected,
            requires_grad=True,
        )
        logits = model(batch)
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=batch,
            grad_outputs=torch.ones_like(logits),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        values.append((gradients * batch).detach().cpu().numpy())
        raw_outputs.append(logits.detach().cpu().numpy())
    attribution_values = np.concatenate(values, axis=0) if values else np.empty_like(X)
    outputs = np.concatenate(raw_outputs) if raw_outputs else np.empty(0, dtype=float)
    bases = np.full(len(X), base_logit, dtype=float)
    return attribution_values, bases, outputs, resolution.selected


def compute_attributions(
    predictor: FrozenPredictor,
    features: Any,
    *,
    feature_names: Sequence[str] | None = None,
    batch_size: int = 1024,
    device: str | None = None,
) -> AttributionResult:
    """Compute native-tree or neural gradient-times-input attributions.

    Only :class:`FrozenPredictor` is accepted.  This guards the experimental
    order: train -> validation-only calibration/threshold -> freeze -> explain.
    """

    if not isinstance(predictor, FrozenPredictor) or not predictor.frozen:
        raise TypeError("attributions require a FrozenPredictor")
    X, names = _matrix_and_names(predictor, features, feature_names)
    if predictor.family == "xgboost":
        values, bases, outputs = _xgboost_contributions(predictor, X)
        method = "xgboost_pred_contribs"
        selected_device = "cpu"
    elif predictor.family == "lightgbm":
        values, bases, outputs = _lightgbm_contributions(predictor, X)
        method = "lightgbm_pred_contribs"
        selected_device = "cpu"
    elif predictor.family == "tabular_resnet_v2":
        values, bases, outputs, selected_device = _neural_gradient_times_input(
            predictor,
            X,
            batch_size=batch_size,
            device=device,
        )
        method = "gradient_times_input_raw_logit"
    else:  # pragma: no cover - FrozenPredictor constructors are controlled
        raise ValueError(f"unsupported frozen predictor family: {predictor.family}")
    values, bases, outputs = _validate_result(
        values,
        bases,
        outputs,
        rows=len(X),
        columns=X.shape[1],
    )
    return AttributionResult(
        values=values,
        base_values=bases,
        raw_outputs=outputs,
        feature_names=names,
        method=method,
        provenance={
            "attribution_protocol_version": "stress-attribution-v1",
            "method": method,
            "explained_score": "raw_model_logit",
            "model_family": predictor.family,
            "model_backend": predictor.backend,
            "model_seed": int(predictor.seed),
            "rows": int(len(X)),
            "feature_count": int(X.shape[1]),
            "feature_names": list(names),
            "input_fingerprint_sha256": _fingerprint(X),
            "device": selected_device,
            "model_calibrator_threshold_frozen": True,
            "frozen_threshold": float(predictor.threshold),
        },
    )
