"""Validation-only calibration and decision-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, fbeta_score, log_loss

from .prediction_metrics import expected_calibration_error


class ProbabilityCalibrator:
    """Fit calibration on validation predictions and reuse it unchanged on test."""

    def __init__(self, method: Literal["none", "platt", "isotonic"] = "none") -> None:
        self.method = method
        self.model = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        probabilities = np.asarray(probabilities, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if self.method == "none":
            self.model = None
        elif self.method == "platt":
            self.model = LogisticRegression(random_state=42)
            self.model.fit(probabilities.reshape(-1, 1), y_true)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(probabilities, y_true)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if self.model is None:
            return probabilities
        if self.method == "platt":
            return self.model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(probabilities), dtype=float)


@dataclass
class CalibrationSelection:
    """Validation-only calibration selection and its auditable comparison table."""

    method: str
    calibrator: ProbabilityCalibrator
    comparison: list[dict[str, float | str]]
    fit_indices: np.ndarray
    selection_indices: np.ndarray


def select_probability_calibrator(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    methods: list[str] | tuple[str, ...] = ("none", "platt", "isotonic"),
    selection_metric: str = "brier",
    fit_fraction: float = 0.5,
    n_calibration_bins: int = 15,
) -> CalibrationSelection:
    """Fit calibrators on early validation rows and select on later validation rows."""
    labels = np.asarray(y_validation, dtype=int)
    probabilities = np.asarray(validation_probabilities, dtype=float)
    if len(labels) != len(probabilities) or len(labels) < 4:
        raise ValueError("Calibration selection requires aligned validation arrays with at least four rows")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("calibration_fit_fraction must be strictly between 0 and 1")
    supported_metrics = {"brier", "ece", "nll"}
    if selection_metric not in supported_metrics:
        raise ValueError(f"Unsupported calibration selection metric: {selection_metric}")
    unique_methods = list(dict.fromkeys(str(method).lower() for method in methods))
    if not unique_methods:
        raise ValueError("At least one calibration method is required")

    split_index = min(max(int(len(labels) * float(fit_fraction)), 2), len(labels) - 2)
    fit_indices = np.arange(split_index)
    selection_indices = np.arange(split_index, len(labels))
    if np.unique(labels[fit_indices]).size < 2 or np.unique(labels[selection_indices]).size < 2:
        raise ValueError("Both calibration-fit and calibration-selection partitions need both classes")

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, ProbabilityCalibrator] = {}
    for method in unique_methods:
        calibrator = ProbabilityCalibrator(method).fit(probabilities[fit_indices], labels[fit_indices])
        selected_probabilities = np.clip(
            calibrator.transform(probabilities[selection_indices]), 1e-7, 1.0 - 1e-7
        )
        row: dict[str, float | str] = {
            "method": method,
            "brier": float(brier_score_loss(labels[selection_indices], selected_probabilities)),
            "ece": expected_calibration_error(
                labels[selection_indices], selected_probabilities, n_calibration_bins
            ),
            "nll": float(log_loss(labels[selection_indices], selected_probabilities, labels=[0, 1])),
            "fit_rows": float(len(fit_indices)),
            "selection_rows": float(len(selection_indices)),
        }
        rows.append(row)
        fitted[method] = calibrator

    best_row = min(rows, key=lambda row: (float(row[selection_metric]), unique_methods.index(str(row["method"]))))
    best_method = str(best_row["method"])
    return CalibrationSelection(
        method=best_method,
        calibrator=fitted[best_method],
        comparison=rows,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
    )


def select_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    objective: str = "f2",
    beta: float = 2.0,
    grid_size: int = 501,
) -> tuple[float, float]:
    """Select a threshold from validation only and return threshold and objective value."""
    y_validation = np.asarray(y_validation, dtype=int)
    probabilities = np.asarray(validation_probabilities, dtype=float)
    quantile_grid = np.quantile(probabilities, np.linspace(0.0, 1.0, grid_size))
    thresholds = np.unique(np.concatenate(([0.0], quantile_grid, [1.0])))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        labels = (probabilities >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_validation, labels, zero_division=0)
        elif objective in {"f2", "fbeta"}:
            effective_beta = 2.0 if objective == "f2" else beta
            score = fbeta_score(y_validation, labels, beta=effective_beta, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score
