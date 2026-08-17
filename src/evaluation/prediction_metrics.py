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


def paired_bootstrap_pr_auc_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    n_bootstrap: int = 300,
    seed: int = 42,
    max_rows: int | None = None,
) -> dict[str, float]:
    """Paired stratified bootstrap CI for PR-AUC(A) minus PR-AUC(B)."""
    labels = np.asarray(y_true, dtype=int)
    first = np.asarray(probabilities_a, dtype=float)
    second = np.asarray(probabilities_b, dtype=float)
    if not (len(labels) == len(first) == len(second)):
        raise ValueError("Labels and paired prediction arrays must align")
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("Paired bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    if max_rows is not None and len(labels) > int(max_rows):
        positive_rows = max(1, int(max_rows * len(positive_indices) / len(labels)))
        negative_rows = max(1, int(max_rows) - positive_rows)
    else:
        positive_rows = len(positive_indices)
        negative_rows = len(negative_indices)
    differences = np.empty(int(n_bootstrap), dtype=float)
    for iteration in range(int(n_bootstrap)):
        sampled = np.concatenate(
            [
                rng.choice(positive_indices, size=positive_rows, replace=True),
                rng.choice(negative_indices, size=negative_rows, replace=True),
            ]
        )
        rng.shuffle(sampled)
        differences[iteration] = average_precision_score(labels[sampled], first[sampled]) - average_precision_score(
            labels[sampled], second[sampled]
        )
    observed = average_precision_score(labels, first) - average_precision_score(labels, second)
    return {
        "observed_difference": float(observed),
        "bootstrap_mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_a_better": float((differences > 0.0).mean()),
        "bootstrap_iterations": float(len(differences)),
        "bootstrap_rows": float(positive_rows + negative_rows),
    }
