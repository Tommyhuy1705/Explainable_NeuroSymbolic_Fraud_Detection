"""Leakage-safe calibration, threshold selection, and stress-test metrics.

The stress-test protocol deliberately gives each validation row exactly one
role: calibrator fitting, calibrator selection, or decision-threshold
selection.  The selected calibrator is *not* refit after selection.  Test
labels therefore never influence calibration or the operating point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


_EPSILON = 1e-7


def _as_binary_labels(values: np.ndarray | Iterable[int], *, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if labels.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(labels.astype(float)).all():
        raise ValueError(f"{name} contains non-finite values")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1, False, True}):
        raise ValueError(f"{name} must contain binary labels encoded as 0/1")
    return labels.astype(np.int8, copy=False)


def _as_probabilities(
    values: np.ndarray | Iterable[float],
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != expected_rows:
        raise ValueError(f"{name} must be one-dimensional and aligned with labels")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} contains non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{name} must be in the closed interval [0, 1]")
    return np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)


def expected_calibration_error(
    y_true: np.ndarray | Iterable[int],
    probabilities: np.ndarray | Iterable[float],
    *,
    n_bins: int = 15,
) -> float:
    """Return equal-width expected calibration error (ECE)."""

    labels = _as_binary_labels(y_true, name="y_true")
    scores = _as_probabilities(probabilities, expected_rows=len(labels), name="probabilities")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least two")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    value = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & (scores <= upper if index == int(n_bins) - 1 else scores < upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(value)


def recall_at_fpr(
    y_true: np.ndarray | Iterable[int],
    probabilities: np.ndarray | Iterable[float],
    *,
    target_fpr: float = 0.01,
) -> float:
    """Return the greatest empirical recall at or below ``target_fpr``.

    This definition includes the ROC origin and never interpolates to an
    unattained operating point.  Both classes are required.
    """

    labels = _as_binary_labels(y_true, name="y_true")
    scores = _as_probabilities(probabilities, expected_rows=len(labels), name="probabilities")
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("target_fpr must be between zero and one")
    if np.unique(labels).size != 2:
        return float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores, drop_intermediate=False)
    admissible = true_positive_rate[false_positive_rate <= float(target_fpr) + 1e-12]
    return float(admissible.max(initial=0.0))


def _null_metrics(labels: np.ndarray, *, beta: float, n_bins: int) -> dict[str, float]:
    prevalence = float(labels.mean())
    constant = np.full(len(labels), np.clip(prevalence, _EPSILON, 1.0 - _EPSILON), dtype=float)
    all_positive = np.ones(len(labels), dtype=np.int8)
    both_classes = np.unique(labels).size == 2
    return {
        "null_prevalence": prevalence,
        "null_pr_auc": prevalence,
        "null_roc_auc": 0.5 if both_classes else float("nan"),
        "null_f2": float(fbeta_score(labels, all_positive, beta=beta, zero_division=0)),
        "null_precision": float(precision_score(labels, all_positive, zero_division=0)),
        "null_recall": float(recall_score(labels, all_positive, zero_division=0)),
        "null_recall_at_1pct_fpr": 0.0 if both_classes else float("nan"),
        "null_brier": float(brier_score_loss(labels, constant)),
        "null_ece": expected_calibration_error(labels, constant, n_bins=n_bins),
        "null_nll": float(log_loss(labels, constant, labels=[0, 1])),
    }


def evaluate_stress_predictions(
    y_true: np.ndarray | Iterable[int],
    probabilities: np.ndarray | Iterable[float],
    *,
    threshold: float,
    beta: float = 2.0,
    n_calibration_bins: int = 15,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """Evaluate one frozen test prediction vector against prevalence baselines."""

    labels = _as_binary_labels(y_true, name="y_true")
    scores = _as_probabilities(probabilities, expected_rows=len(labels), name="probabilities")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between zero and one")
    decisions = (scores >= float(threshold)).astype(np.int8)
    both_classes = np.unique(labels).size == 2
    metrics = {
        "n_rows": float(len(labels)),
        "n_positive": float(labels.sum()),
        "prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "f2": float(fbeta_score(labels, decisions, beta=beta, zero_division=0)),
        "precision": float(precision_score(labels, decisions, zero_division=0)),
        "recall": float(recall_score(labels, decisions, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(labels, scores, target_fpr=target_fpr),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": expected_calibration_error(labels, scores, n_bins=n_calibration_bins),
        "nll": float(log_loss(labels, scores, labels=[0, 1])),
        "threshold": float(threshold),
        "predicted_positive_rate": float(decisions.mean()),
        "target_fpr": float(target_fpr),
    }
    metrics.update(_null_metrics(labels, beta=beta, n_bins=n_calibration_bins))
    metrics.update(
        {
            "pr_auc_lift_over_null": metrics["pr_auc"] - metrics["null_pr_auc"],
            "f2_lift_over_null": metrics["f2"] - metrics["null_f2"],
            "brier_improvement_over_null": metrics["null_brier"] - metrics["brier"],
            "nll_improvement_over_null": metrics["null_nll"] - metrics["nll"],
        }
    )
    return metrics


@dataclass(frozen=True)
class ValidationPartitions:
    """Ordered, disjoint validation partitions with explicit statistical roles."""

    calibrator_fit: np.ndarray
    calibrator_selection: np.ndarray
    threshold_selection: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "calibrator_fit": self.calibrator_fit.astype(int).tolist(),
            "calibrator_selection": self.calibrator_selection.astype(int).tolist(),
            "threshold_selection": self.threshold_selection.astype(int).tolist(),
        }


def make_validation_partitions(
    y_validation: np.ndarray | Iterable[int],
    *,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> ValidationPartitions:
    """Create three chronological partitions without shuffling validation rows."""

    labels = _as_binary_labels(y_validation, name="y_validation")
    weights = np.asarray(fractions, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("fractions must contain three positive finite values")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("validation partition fractions must sum to one")
    if len(labels) < 6:
        raise ValueError("at least six validation rows are required for three partitions")
    first_end = max(2, int(np.floor(len(labels) * weights[0])))
    second_end = max(first_end + 2, int(np.floor(len(labels) * (weights[0] + weights[1]))))
    second_end = min(second_end, len(labels) - 2)
    partitions = ValidationPartitions(
        calibrator_fit=np.arange(0, first_end, dtype=int),
        calibrator_selection=np.arange(first_end, second_end, dtype=int),
        threshold_selection=np.arange(second_end, len(labels), dtype=int),
    )
    combined = np.concatenate(
        [partitions.calibrator_fit, partitions.calibrator_selection, partitions.threshold_selection]
    )
    if len(np.unique(combined)) != len(labels) or not np.array_equal(np.sort(combined), np.arange(len(labels))):
        raise RuntimeError("validation partitions must be disjoint and exhaustive")
    for role, indices in partitions.as_dict().items():
        if np.unique(labels[np.asarray(indices, dtype=int)]).size != 2:
            raise ValueError(
                f"{role} partition needs both classes; revise the upstream temporal split or partition fractions"
            )
    return partitions


@dataclass(frozen=True)
class FrozenCalibrator:
    """An immutable fitted probability calibrator."""

    method: str
    model: Any

    def transform(self, probabilities: np.ndarray | Iterable[float]) -> np.ndarray:
        scores = np.asarray(probabilities, dtype=float)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("calibrator input must be a finite one-dimensional array")
        scores = np.clip(scores, _EPSILON, 1.0 - _EPSILON)
        if self.method == "none":
            calibrated = scores
        elif self.method == "platt":
            calibrated = self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.model.predict(scores)
        else:  # pragma: no cover - constructor is internal and validates methods
            raise RuntimeError(f"Unknown fitted calibration method: {self.method}")
        calibrated = np.asarray(calibrated, dtype=float)
        if calibrated.shape != scores.shape or not np.isfinite(calibrated).all():
            raise RuntimeError("calibrator returned invalid probabilities")
        return np.clip(calibrated, _EPSILON, 1.0 - _EPSILON)


def _fit_calibrator(method: str, probabilities: np.ndarray, labels: np.ndarray, *, seed: int) -> FrozenCalibrator:
    normalized = str(method).lower()
    if normalized == "none":
        return FrozenCalibrator(method="none", model=None)
    if normalized == "platt":
        model = LogisticRegression(random_state=int(seed), solver="lbfgs", max_iter=1000)
        model.fit(probabilities.reshape(-1, 1), labels)
        return FrozenCalibrator(method="platt", model=model)
    if normalized == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probabilities, labels)
        return FrozenCalibrator(method="isotonic", model=model)
    raise ValueError(f"unsupported calibration method: {method}")


def _selection_loss(labels: np.ndarray, probabilities: np.ndarray, *, metric: str, n_bins: int) -> float:
    if metric == "brier":
        return float(brier_score_loss(labels, probabilities))
    if metric == "nll":
        return float(log_loss(labels, probabilities, labels=[0, 1]))
    if metric == "ece":
        return expected_calibration_error(labels, probabilities, n_bins=n_bins)
    raise ValueError("selection_metric must be one of: brier, nll, ece")


def _select_f2_threshold(labels: np.ndarray, probabilities: np.ndarray, *, beta: float, grid_size: int) -> tuple[float, float]:
    if int(grid_size) < 3:
        raise ValueError("threshold_grid_size must be at least three")
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(probabilities, np.linspace(0.0, 1.0, int(grid_size))), [1.0])
        )
    )
    best_threshold, best_score = 0.5, -np.inf
    for candidate in candidates:
        decisions = (probabilities >= candidate).astype(np.int8)
        score = float(fbeta_score(labels, decisions, beta=beta, zero_division=0))
        if score > best_score or (np.isclose(score, best_score) and float(candidate) > best_threshold):
            best_threshold, best_score = float(candidate), score
    return best_threshold, best_score


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationProtocolResult:
    """Frozen validation-only calibrator and F2 operating point."""

    calibrator: FrozenCalibrator
    threshold: float
    threshold_f2: float
    comparison: tuple[dict[str, float | str], ...]
    partitions: ValidationPartitions | None
    provenance: dict[str, Any]
    frozen: bool = True

    def transform(self, probabilities: np.ndarray | Iterable[float]) -> np.ndarray:
        return self.calibrator.transform(probabilities)


def fit_validation_protocol(
    y_validation: np.ndarray | Iterable[int],
    validation_probabilities: np.ndarray | Iterable[float],
    *,
    calibration_methods: tuple[str, ...] = ("none", "platt", "isotonic"),
    selection_metric: str = "brier",
    partition_fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    beta: float = 2.0,
    n_calibration_bins: int = 15,
    threshold_grid_size: int = 501,
    seed: int = 42,
) -> ValidationProtocolResult:
    """Fit, select, and freeze calibration/threshold using validation only."""

    labels = _as_binary_labels(y_validation, name="y_validation")
    raw = _as_probabilities(
        validation_probabilities,
        expected_rows=len(labels),
        name="validation_probabilities",
    )
    methods = tuple(dict.fromkeys(str(method).lower() for method in calibration_methods))
    if not methods:
        raise ValueError("at least one calibration method is required")
    if selection_metric not in {"brier", "nll", "ece"}:
        raise ValueError("selection_metric must be one of: brier, nll, ece")
    partitions = make_validation_partitions(labels, fractions=partition_fractions)
    fit_indices = partitions.calibrator_fit
    selection_indices = partitions.calibrator_selection
    fitted: dict[str, FrozenCalibrator] = {}
    rows: list[dict[str, float | str]] = []
    for method in methods:
        calibrator = _fit_calibrator(method, raw[fit_indices], labels[fit_indices], seed=seed)
        calibrated = calibrator.transform(raw[selection_indices])
        row: dict[str, float | str] = {
            "method": method,
            "brier": float(brier_score_loss(labels[selection_indices], calibrated)),
            "nll": float(log_loss(labels[selection_indices], calibrated, labels=[0, 1])),
            "ece": expected_calibration_error(
                labels[selection_indices], calibrated, n_bins=n_calibration_bins
            ),
            "selection_loss": _selection_loss(
                labels[selection_indices], calibrated, metric=selection_metric, n_bins=n_calibration_bins
            ),
        }
        fitted[method] = calibrator
        rows.append(row)
    best = min(rows, key=lambda row: (float(row["selection_loss"]), methods.index(str(row["method"]))))
    selected_method = str(best["method"])
    selected = fitted[selected_method]
    threshold_probabilities = selected.transform(raw[partitions.threshold_selection])
    threshold, threshold_f2 = _select_f2_threshold(
        labels[partitions.threshold_selection],
        threshold_probabilities,
        beta=beta,
        grid_size=threshold_grid_size,
    )
    provenance: dict[str, Any] = {
        "protocol_version": "stress-validation-v1",
        "seed": int(seed),
        "validation_rows": int(len(labels)),
        "validation_prevalence": float(labels.mean()),
        "partition_fractions": [float(value) for value in partition_fractions],
        "partition_sizes": {key: len(value) for key, value in partitions.as_dict().items()},
        "selected_calibration_method": selected_method,
        "calibration_selection_metric": selection_metric,
        "threshold_objective": f"f{beta:g}",
        "threshold": float(threshold),
        "threshold_objective_value": float(threshold_f2),
        "validation_fingerprint_sha256": _array_hash(labels, raw),
    }
    return ValidationProtocolResult(
        calibrator=selected,
        threshold=float(threshold),
        threshold_f2=float(threshold_f2),
        comparison=tuple(rows),
        partitions=partitions,
        provenance=provenance,
    )


def fit_reserved_validation_protocol(
    y_calibration_fit: np.ndarray | Iterable[int],
    calibration_fit_probabilities: np.ndarray | Iterable[float],
    y_calibration_select: np.ndarray | Iterable[int],
    calibration_select_probabilities: np.ndarray | Iterable[float],
    *,
    calibration_methods: tuple[str, ...] = ("none", "platt", "isotonic"),
    selection_metric: str = "brier",
    beta: float = 2.0,
    n_calibration_bins: int = 15,
    threshold_grid_size: int = 501,
    seed: int = 42,
) -> ValidationProtocolResult:
    """Fit and lock calibration using the two pre-reserved outer roles.

    The calibrator is fitted only on ``calibration_fit``.  Method comparison
    and the F-beta operating point are locked on ``calibration_select``.  This
    avoids silently splitting the already small calibration-fit role a second
    time and makes the executed roles match the stress-test protocol.
    """

    fit_labels = _as_binary_labels(y_calibration_fit, name="y_calibration_fit")
    select_labels = _as_binary_labels(y_calibration_select, name="y_calibration_select")
    fit_raw = _as_probabilities(
        calibration_fit_probabilities,
        expected_rows=len(fit_labels),
        name="calibration_fit_probabilities",
    )
    select_raw = _as_probabilities(
        calibration_select_probabilities,
        expected_rows=len(select_labels),
        name="calibration_select_probabilities",
    )
    methods = tuple(dict.fromkeys(str(method).lower() for method in calibration_methods))
    if not methods:
        raise ValueError("at least one calibration method is required")
    if selection_metric not in {"brier", "nll", "ece"}:
        raise ValueError("selection_metric must be one of: brier, nll, ece")

    fitted: dict[str, FrozenCalibrator] = {}
    rows: list[dict[str, float | str]] = []
    for method in methods:
        calibrator = _fit_calibrator(method, fit_raw, fit_labels, seed=seed)
        calibrated = calibrator.transform(select_raw)
        rows.append(
            {
                "method": method,
                "brier": float(brier_score_loss(select_labels, calibrated)),
                "nll": float(log_loss(select_labels, calibrated, labels=[0, 1])),
                "ece": expected_calibration_error(
                    select_labels, calibrated, n_bins=n_calibration_bins
                ),
                "selection_loss": _selection_loss(
                    select_labels,
                    calibrated,
                    metric=selection_metric,
                    n_bins=n_calibration_bins,
                ),
            }
        )
        fitted[method] = calibrator

    best = min(
        rows,
        key=lambda row: (float(row["selection_loss"]), methods.index(str(row["method"]))),
    )
    selected_method = str(best["method"])
    selected = fitted[selected_method]
    select_calibrated = selected.transform(select_raw)
    threshold, threshold_f2 = _select_f2_threshold(
        select_labels,
        select_calibrated,
        beta=beta,
        grid_size=threshold_grid_size,
    )
    provenance: dict[str, Any] = {
        "protocol_version": "stress-reserved-roles-v2",
        "seed": int(seed),
        "calibrator_fit_rows": int(len(fit_labels)),
        "calibrator_fit_prevalence": float(fit_labels.mean()),
        "calibration_select_rows": int(len(select_labels)),
        "calibration_select_prevalence": float(select_labels.mean()),
        "role_contract": {
            "calibrator_fit": "calibration_fit",
            "calibrator_selection": "calibration_select",
            "threshold_selection": "calibration_select",
        },
        "outer_roles_used_without_resplitting": True,
        "selected_calibration_method": selected_method,
        "calibration_selection_metric": selection_metric,
        "threshold_objective": f"f{beta:g}",
        "threshold": float(threshold),
        "threshold_objective_value": float(threshold_f2),
        "reserved_role_fingerprint_sha256": _array_hash(
            fit_labels, fit_raw, select_labels, select_raw
        ),
    }
    return ValidationProtocolResult(
        calibrator=selected,
        threshold=float(threshold),
        threshold_f2=float(threshold_f2),
        comparison=tuple(rows),
        partitions=None,
        provenance=provenance,
    )
