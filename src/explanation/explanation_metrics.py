"""Rule-level and system-level explanation-quality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rule_quality_table(
    truth_values: pd.DataFrame,
    y_true: np.ndarray,
    activation_threshold: float = 0.6,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    evaluated_rows = len(y_true)
    if len(truth_values) != evaluated_rows:
        raise ValueError("Rule truth values and labels must align")
    base_rate = float(y_true.mean()) if evaluated_rows else float("nan")
    rows = []
    for rule in truth_values.columns:
        scores = truth_values[rule].to_numpy(float)
        active = scores >= activation_threshold
        active_count = int(active.sum())
        precision = float(y_true[active].mean()) if active_count else float("nan")
        rows.append(
            {
                "rule": rule,
                "evaluated_rows": evaluated_rows,
                "coverage": float(active_count / evaluated_rows) if evaluated_rows else float("nan"),
                "active_count": active_count,
                "fraud_precision": precision,
                "lift": precision / base_rate if active_count and base_rate > 0 else float("nan"),
                "mean_truth": float(scores.mean()) if evaluated_rows else float("nan"),
                "rule_auc": float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["lift", "coverage"], ascending=False).reset_index(drop=True)


def explanation_quality_metrics(
    explanations: pd.DataFrame,
    y_true: np.ndarray,
    predicted_probabilities: np.ndarray,
    decision_threshold: float,
) -> dict[str, float | int]:
    """Summarize rule evidence with explicit alert and non-alert denominators.

    ``prediction_rule_consistency`` is retained for backward compatibility. Its
    balanced counterpart averages alert support with the rate of non-alerts that
    have no rule evidence, so class imbalance cannot dominate the score.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    test_rows = len(y_true)
    if not (test_rows == len(probabilities) == len(explanations)):
        raise ValueError("Explanation, label, and probability arrays must align")
    predicted_alert = probabilities >= decision_threshold
    explained = explanations["explained"].to_numpy(bool)
    # `explained` already reflects the explainer's configured activation threshold.
    rule_alert = explained
    alert_count = int(predicted_alert.sum())
    non_alert_count = test_rows - alert_count
    explained_count = int(explained.sum())
    zero_rule_count = test_rows - explained_count
    explained_alert = explained & predicted_alert
    explained_alert_count = int(explained_alert.sum())
    unsupported_alert_count = alert_count - explained_alert_count
    rule_evidence_without_alert = explained & ~predicted_alert
    rule_evidence_without_alert_count = int(rule_evidence_without_alert.sum())
    predicted_alert_fraud_count = int(y_true[predicted_alert].sum())
    explained_alert_fraud_count = int(y_true[explained_alert].sum())
    contradiction = predicted_alert != rule_alert
    explained_precision = (
        float(explained_alert_fraud_count / explained_alert_count)
        if explained_alert_count
        else float("nan")
    )
    all_alert_precision = (
        float(predicted_alert_fraud_count / alert_count)
        if alert_count
        else float("nan")
    )
    precision_gain = (
        explained_precision - all_alert_precision
        if explained_alert_count and alert_count
        else float("nan")
    )
    alert_support_rate = float(explained_alert_count / alert_count) if alert_count else float("nan")
    unsupported_alert_rate = float(unsupported_alert_count / alert_count) if alert_count else float("nan")
    rule_evidence_without_alert_rate = (
        float(rule_evidence_without_alert_count / non_alert_count)
        if non_alert_count
        else float("nan")
    )
    # Give alert support and non-alert rejection equal weight. Both denominators
    # must exist; otherwise a balanced two-group summary is undefined.
    balanced_consistency = (
        float((alert_support_rate + (1.0 - rule_evidence_without_alert_rate)) / 2.0)
        if alert_count and non_alert_count
        else float("nan")
    )
    active_count_column = "active_rule_count" if "active_rule_count" in explanations else "rule_count"
    displayed_count_column = (
        "displayed_rule_count" if "displayed_rule_count" in explanations else active_count_column
    )
    mean_active_rule_count = (
        float(explanations[active_count_column].mean()) if test_rows else float("nan")
    )
    mean_displayed_rule_count = (
        float(explanations[displayed_count_column].mean()) if test_rows else float("nan")
    )
    if test_rows and "available_rule_count" in explanations:
        available = explanations["available_rule_count"].to_numpy(float)
        active_counts = explanations[active_count_column].to_numpy(float)
        valid = available > 0
        mean_active_rule_fraction = (
            float(np.mean(active_counts[valid] / available[valid])) if valid.any() else float("nan")
        )
    else:
        mean_active_rule_fraction = float("nan")
    rule_sparsity = (
        1.0 - mean_active_rule_fraction if np.isfinite(mean_active_rule_fraction) else float("nan")
    )
    return {
        "test_rows": test_rows,
        "predicted_alert_count": alert_count,
        "non_alert_count": non_alert_count,
        "predicted_alert_fraud_count": predicted_alert_fraud_count,
        "explained_count": explained_count,
        "zero_rule_count": zero_rule_count,
        "explained_alert_count": explained_alert_count,
        "explained_alert_fraud_count": explained_alert_fraud_count,
        "unsupported_alert_count": unsupported_alert_count,
        "rule_evidence_without_alert_count": rule_evidence_without_alert_count,
        "explanation_coverage_all": float(explained_count / test_rows) if test_rows else float("nan"),
        "zero_rule_fraction": float(zero_rule_count / test_rows) if test_rows else float("nan"),
        "explanation_coverage_alerts": alert_support_rate,
        "unsupported_alert_rate": unsupported_alert_rate,
        "rule_evidence_without_alert_rate": rule_evidence_without_alert_rate,
        "mean_rule_count": mean_active_rule_count,
        "mean_active_rule_count": mean_active_rule_count,
        "mean_displayed_rule_count": mean_displayed_rule_count,
        "mean_active_rule_fraction": mean_active_rule_fraction,
        "sparsity": rule_sparsity,
        "rule_sparsity": rule_sparsity,
        "prediction_rule_consistency": (
            float((predicted_alert == rule_alert).mean()) if test_rows else float("nan")
        ),
        "balanced_prediction_rule_consistency": balanced_consistency,
        "contradiction_rate": float(contradiction.mean()) if test_rows else float("nan"),
        "explained_alert_precision": explained_precision,
        "all_alert_precision": all_alert_precision,
        "explained_alert_precision_gain": precision_gain,
    }


def bootstrap_explanation_precision_gain(
    explanations: pd.DataFrame,
    y_true: np.ndarray,
    predicted_probabilities: np.ndarray,
    decision_threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a percentile CI for precision gain among explained alerts."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    explained = explanations["explained"].to_numpy(bool)
    if not (len(y_true) == len(probabilities) == len(explained)):
        raise ValueError("Explanation, label, and probability arrays must align")
    predicted_alert = probabilities >= decision_threshold
    explained_alert = predicted_alert & explained
    if len(y_true) == 0 or not predicted_alert.any() or not explained_alert.any():
        return {
            "precision_gain_bootstrap_mean": float("nan"),
            "precision_gain_ci_low": float("nan"),
            "precision_gain_ci_high": float("nan"),
            "bootstrap_valid_iterations": 0.0,
        }
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_alert = predicted_alert[indices]
        sampled_explained_alert = sampled_alert & explained[indices]
        if not sampled_alert.any() or not sampled_explained_alert.any():
            continue
        base_precision = float(y_true[indices][sampled_alert].mean())
        explained_precision = float(y_true[indices][sampled_explained_alert].mean())
        gains.append(explained_precision - base_precision)
    if not gains:
        return {
            "precision_gain_bootstrap_mean": float("nan"),
            "precision_gain_ci_low": float("nan"),
            "precision_gain_ci_high": float("nan"),
            "bootstrap_valid_iterations": 0.0,
        }
    values = np.asarray(gains, dtype=float)
    return {
        "precision_gain_bootstrap_mean": float(values.mean()),
        "precision_gain_ci_low": float(np.quantile(values, 0.025)),
        "precision_gain_ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_valid_iterations": float(len(values)),
    }
