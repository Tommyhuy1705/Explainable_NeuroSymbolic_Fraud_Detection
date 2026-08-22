import numpy as np
import pandas as pd
import pytest

from src.explanation import (
    bootstrap_explanation_precision_gain,
    explanation_quality_metrics,
    rule_quality_table,
)


def _explanations(explained: list[bool], rule_count: list[int] | None = None) -> pd.DataFrame:
    counts = rule_count if rule_count is not None else [int(value) for value in explained]
    return pd.DataFrame({
        "explained": explained,
        "rule_count": counts,
        "active_rule_count": counts,
        "displayed_rule_count": [min(value, 1) for value in counts],
        "available_rule_count": [4] * len(counts),
    })


def test_rule_quality_uses_nan_for_a_rule_without_active_rows():
    truth_values = pd.DataFrame(
        {
            "inactive_rule": [0.1, 0.2, 0.3, 0.4],
            "active_rule": [0.7, 0.8, 0.1, 0.2],
        }
    )
    table = rule_quality_table(truth_values, np.array([0, 1, 0, 1]), activation_threshold=0.6)

    inactive = table.loc[table["rule"] == "inactive_rule"].iloc[0]
    active = table.loc[table["rule"] == "active_rule"].iloc[0]
    assert inactive["evaluated_rows"] == 4
    assert inactive["active_count"] == 0
    assert inactive["coverage"] == 0.0
    assert np.isnan(inactive["fraud_precision"])
    assert np.isnan(inactive["lift"])
    assert active["active_count"] == 2
    assert active["fraud_precision"] == pytest.approx(0.5)
    assert active["lift"] == pytest.approx(1.0)


def test_explanation_metrics_preserve_valid_metrics_and_report_denominators():
    explanations = _explanations([True, True, False, False], [2, 1, 0, 0])
    metrics = explanation_quality_metrics(
        explanations,
        y_true=np.array([1, 0, 1, 0]),
        predicted_probabilities=np.array([0.9, 0.8, 0.7, 0.1]),
        decision_threshold=0.5,
    )

    assert metrics["test_rows"] == 4
    assert metrics["predicted_alert_count"] == 3
    assert metrics["non_alert_count"] == 1
    assert metrics["predicted_alert_fraud_count"] == 2
    assert metrics["explained_count"] == 2
    assert metrics["zero_rule_count"] == 2
    assert metrics["explained_alert_count"] == 2
    assert metrics["explained_alert_fraud_count"] == 1
    assert metrics["unsupported_alert_count"] == 1
    assert metrics["rule_evidence_without_alert_count"] == 0
    assert metrics["explanation_coverage_all"] == pytest.approx(0.5)
    assert metrics["zero_rule_fraction"] == pytest.approx(0.5)
    assert metrics["explanation_coverage_alerts"] == pytest.approx(2 / 3)
    assert metrics["unsupported_alert_rate"] == pytest.approx(1 / 3)
    assert metrics["rule_evidence_without_alert_rate"] == 0.0
    assert metrics["prediction_rule_consistency"] == pytest.approx(0.75)
    assert metrics["balanced_prediction_rule_consistency"] == pytest.approx(5 / 6)
    assert metrics["all_alert_precision"] == pytest.approx(2 / 3)
    assert metrics["explained_alert_precision"] == pytest.approx(0.5)
    assert metrics["explained_alert_precision_gain"] == pytest.approx(-1 / 6)
    assert metrics["mean_active_rule_count"] == pytest.approx(0.75)
    assert metrics["mean_displayed_rule_count"] == pytest.approx(0.5)
    assert metrics["mean_active_rule_fraction"] == pytest.approx(0.1875)
    assert metrics["rule_sparsity"] == pytest.approx(0.8125)


def test_explanation_metrics_use_nan_when_there_are_no_predicted_alerts():
    metrics = explanation_quality_metrics(
        _explanations([True, False, False]),
        y_true=np.array([1, 0, 1]),
        predicted_probabilities=np.array([0.2, 0.3, 0.4]),
        decision_threshold=0.5,
    )

    assert metrics["predicted_alert_count"] == 0
    assert metrics["explained_count"] == 1
    assert metrics["explained_alert_count"] == 0
    assert np.isnan(metrics["explanation_coverage_alerts"])
    assert np.isnan(metrics["unsupported_alert_rate"])
    assert metrics["rule_evidence_without_alert_rate"] == pytest.approx(1 / 3)
    assert np.isnan(metrics["balanced_prediction_rule_consistency"])
    assert np.isnan(metrics["all_alert_precision"])
    assert np.isnan(metrics["explained_alert_precision"])
    assert np.isnan(metrics["explained_alert_precision_gain"])


def test_explanation_metrics_use_nan_when_alerts_have_no_rule_explanation():
    metrics = explanation_quality_metrics(
        _explanations([False, False, False]),
        y_true=np.array([1, 0, 1]),
        predicted_probabilities=np.array([0.9, 0.8, 0.1]),
        decision_threshold=0.5,
    )

    assert metrics["predicted_alert_count"] == 2
    assert metrics["explained_alert_count"] == 0
    assert metrics["explanation_coverage_alerts"] == 0.0
    assert metrics["unsupported_alert_rate"] == 1.0
    assert metrics["rule_evidence_without_alert_rate"] == 0.0
    assert metrics["balanced_prediction_rule_consistency"] == pytest.approx(0.5)
    assert metrics["all_alert_precision"] == pytest.approx(0.5)
    assert np.isnan(metrics["explained_alert_precision"])
    assert np.isnan(metrics["explained_alert_precision_gain"])


def test_balanced_consistency_weights_alert_and_non_alert_agreement_equally():
    # Overall agreement is inflated by nine non-alerts; balanced consistency gives
    # alert support and non-alert rejection equal weight.
    explanations = _explanations([False] + [False] * 8 + [True])
    metrics = explanation_quality_metrics(
        explanations,
        y_true=np.zeros(10, dtype=int),
        predicted_probabilities=np.array([0.9] + [0.1] * 9),
        decision_threshold=0.5,
    )

    assert metrics["prediction_rule_consistency"] == pytest.approx(0.8)
    assert metrics["explanation_coverage_alerts"] == 0.0
    assert metrics["unsupported_alert_rate"] == 1.0
    assert metrics["rule_evidence_without_alert_rate"] == pytest.approx(1 / 9)
    assert metrics["balanced_prediction_rule_consistency"] == pytest.approx(4 / 9)


@pytest.mark.parametrize(
    ("probabilities", "explained"),
    [
        (np.array([0.1, 0.2, 0.3]), [True, False, False]),
        (np.array([0.9, 0.8, 0.1]), [False, False, False]),
        (np.array([], dtype=float), []),
    ],
)
def test_bootstrap_returns_undefined_interval_when_gain_has_no_denominator(probabilities, explained):
    y_true = np.array([1, 0, 1])[: len(probabilities)]
    interval = bootstrap_explanation_precision_gain(
        _explanations(explained),
        y_true=y_true,
        predicted_probabilities=probabilities,
        decision_threshold=0.5,
        n_bootstrap=25,
        seed=7,
    )

    assert interval["bootstrap_valid_iterations"] == 0.0
    assert np.isnan(interval["precision_gain_bootstrap_mean"])
    assert np.isnan(interval["precision_gain_ci_low"])
    assert np.isnan(interval["precision_gain_ci_high"])


def test_metric_functions_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        rule_quality_table(pd.DataFrame({"rule": [0.1, 0.2]}), np.array([1]))

    with pytest.raises(ValueError, match="must align"):
        explanation_quality_metrics(
            _explanations([True, False]),
            y_true=np.array([1]),
            predicted_probabilities=np.array([0.9, 0.1]),
            decision_threshold=0.5,
        )
