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
    base_rate = float(y_true.mean())
    rows = []
    for rule in truth_values.columns:
        scores = truth_values[rule].to_numpy(float)
        active = scores >= activation_threshold
        precision = float(y_true[active].mean()) if active.any() else 0.0
        rows.append(
            {
                "rule": rule,
                "coverage": float(active.mean()),
                "active_count": int(active.sum()),
                "fraud_precision": precision,
                "lift": precision / base_rate if base_rate > 0 else float("nan"),
                "mean_truth": float(scores.mean()),
                "rule_auc": float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["lift", "coverage"], ascending=False).reset_index(drop=True)


def explanation_quality_metrics(
    explanations: pd.DataFrame,
    y_true: np.ndarray,
    predicted_probabilities: np.ndarray,
    decision_threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    predicted_alert = probabilities >= decision_threshold
    explained = explanations["explained"].to_numpy(bool)
    # `explained` already reflects the explainer's configured activation threshold.
    rule_alert = explained
    alert_count = int(predicted_alert.sum())
    explained_alert = explained & predicted_alert
    contradiction = predicted_alert != rule_alert
    explained_precision = float(y_true[explained_alert].mean()) if explained_alert.any() else 0.0
    all_alert_precision = float(y_true[predicted_alert].mean()) if predicted_alert.any() else 0.0
    return {
        "explanation_coverage_all": float(explained.mean()),
        "explanation_coverage_alerts": float(explained_alert.sum() / max(alert_count, 1)),
        "mean_rule_count": float(explanations["rule_count"].mean()),
        "sparsity": float(1.0 / (1.0 + explanations["rule_count"].mean())),
        "prediction_rule_consistency": float((predicted_alert == rule_alert).mean()),
        "contradiction_rate": float(contradiction.mean()),
        "explained_alert_precision": explained_precision,
        "all_alert_precision": all_alert_precision,
        "explained_alert_precision_gain": explained_precision - all_alert_precision,
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
