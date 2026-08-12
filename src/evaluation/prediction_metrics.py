"""Metrics for imbalanced binary fraud detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _clip(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == n_bins - 1 else probabilities < upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(probabilities[mask].mean()))
    return float(ece)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    beta: float = 2.0,
    n_calibration_bins: int = 15,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _clip(probabilities)
    labels = (probabilities >= threshold).astype(int)
    has_both_classes = np.unique(y_true).size == 2
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else float("nan"),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "fbeta": float(fbeta_score(y_true, labels, beta=beta, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities, n_calibration_bins),
        "nll": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "positive_rate": float(labels.mean()),
    }
