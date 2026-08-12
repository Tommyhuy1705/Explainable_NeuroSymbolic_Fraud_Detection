"""Validation-only calibration and decision-threshold selection."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score


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
