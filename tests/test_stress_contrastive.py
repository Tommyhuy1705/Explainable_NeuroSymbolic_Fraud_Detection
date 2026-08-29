from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.stress_testing.contrastive import (
    ContrastiveMetaEvidence,
    fit_contrastive_meta_evidence,
)


def _audit_fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Rows 0-3 are below the frozen predictor threshold and must be irrelevant
    # to fitting. Rows 4-15 contain six TP and six FP predictor alerts.
    truth = pd.DataFrame(
        {
            "amount_rule": [0.2, 0.9, 0.1, 0.8, 0.92, 0.88, 0.85, 0.80, 0.76, 0.72, 0.12, 0.18, 0.22, 0.28, 0.31, 0.35],
            "velocity_rule": [0.1, 0.2, 0.9, 0.8, 0.86, 0.82, 0.79, 0.75, 0.70, 0.68, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15],
        }
    )
    labels = np.asarray([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    probabilities = np.asarray(
        [0.10, 0.20, 0.30, 0.49, 0.82, 0.80, 0.77, 0.74, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56],
        dtype=float,
    )
    return truth, labels, probabilities


def test_contrastive_fit_is_deterministic_and_uses_alert_subset_only() -> None:
    truth, labels, probabilities = _audit_fixture()
    settings = dict(
        decision_threshold=0.50,
        activation_threshold=0.60,
        seed=1705,
        min_alert_rows=8,
        min_class_rows=3,
    )
    first = fit_contrastive_meta_evidence(truth, labels, probabilities, **settings)
    second = fit_contrastive_meta_evidence(truth, labels, probabilities, **settings)

    assert isinstance(first, ContrastiveMetaEvidence)
    assert first.available
    assert first.provenance == second.provenance
    assert_frame_equal(first.coefficient_table(), second.coefficient_table())
    np.testing.assert_allclose(
        first.transform(truth, probabilities),
        second.transform(truth, probabilities),
        atol=0.0,
        rtol=0.0,
    )
    assert first.provenance["fit_partition"] == "rule_audit"
    assert first.provenance["fit_scope"] == "frozen_predictor_alerts_only"
    assert first.provenance["class_weight"] == "balanced"
    assert first.provenance["test_data_used_for_fit"] is False
    assert first.feature_schema == (
        "amount_rule",
        "velocity_rule",
        "frozen_calibrated_risk_z",
    )
    assert first.frozen_risk_mean == first.provenance["frozen_risk_mean"]
    assert first.frozen_risk_std == first.provenance["frozen_risk_std"]
    assert first.frozen_risk_coefficient == first.provenance["frozen_risk_coefficient"]
    coefficient_table = first.coefficient_table()
    assert coefficient_table["feature_type"].tolist() == [
        "audited_rule",
        "audited_rule",
        "frozen_risk",
    ]
    assert isinstance(json.dumps(first.provenance, sort_keys=True), str)

    # Mutating every non-alert label and rule value cannot change the fit or its
    # provenance fingerprint because only frozen-predictor alerts are consumed.
    changed_truth = truth.copy()
    changed_labels = labels.copy()
    changed_probabilities = probabilities.copy()
    non_alerts = probabilities < settings["decision_threshold"]
    changed_truth.loc[non_alerts, :] = 1.0 - changed_truth.loc[non_alerts, :]
    changed_labels[non_alerts] = 1 - changed_labels[non_alerts]
    changed_probabilities[non_alerts] = np.asarray([0.11, 0.22, 0.33, 0.44])
    changed = fit_contrastive_meta_evidence(
        changed_truth, changed_labels, changed_probabilities, **settings
    )
    assert changed.provenance == first.provenance
    assert_frame_equal(changed.coefficient_table(), first.coefficient_table())


def test_transform_is_label_free_finite_and_zeroes_unsupported_rows() -> None:
    truth, labels, probabilities = _audit_fixture()
    fitted = fit_contrastive_meta_evidence(
        truth,
        labels,
        probabilities,
        0.50,
        activation_threshold=0.60,
        min_alert_rows=8,
        min_class_rows=3,
    )
    signature = inspect.signature(fitted.transform)
    assert "labels" not in signature.parameters
    assert "frozen_probabilities" in signature.parameters

    later = pd.DataFrame(
        {
            "amount_rule": [0.10, 0.90, 0.25, 0.75],
            "velocity_rule": [0.20, 0.10, 0.95, 0.80],
        }
    )
    later_probabilities = np.asarray([0.52, 0.58, 0.76, 0.91], dtype=float)
    evidence = fitted.transform(later, later_probabilities)
    assert evidence.shape == (4,)
    assert np.isfinite(evidence).all()
    assert ((0.0 <= evidence) & (evidence <= 1.0)).all()
    assert evidence[0] == 0.0
    assert (evidence[1:] > 0.0).all()

    # Arbitrary policy/test outcomes cannot affect a transform API that never
    # accepts them.
    future_labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
    before = fitted.transform(later, later_probabilities)
    future_labels[:] = 1 - future_labels
    after = fitted.transform(later, later_probabilities)
    assert future_labels.sum() == 2
    np.testing.assert_array_equal(before, after)


def test_frozen_risk_is_an_effective_probability_aware_feature() -> None:
    truth, labels, probabilities = _audit_fixture()
    fitted = fit_contrastive_meta_evidence(
        truth,
        labels,
        probabilities,
        0.50,
        activation_threshold=0.60,
        min_alert_rows=8,
        min_class_rows=3,
    )
    assert fitted.available
    assert abs(fitted.frozen_risk_coefficient) > 0.0

    identical_rule_truth = pd.DataFrame(
        {
            "amount_rule": [0.80, 0.80],
            "velocity_rule": [0.70, 0.70],
        }
    )
    evidence = fitted.transform(
        identical_rule_truth,
        np.asarray([0.56, 0.82], dtype=float),
    )
    assert evidence[0] != evidence[1]


@pytest.mark.parametrize(
    ("truth", "labels", "probabilities", "expected_reason"),
    [
        (
            pd.DataFrame(index=range(6)),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.full(6, 0.9),
            "no_audited_rules",
        ),
        (
            pd.DataFrame({"rule": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]}),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.full(6, 0.1),
            "insufficient_predictor_alerts",
        ),
        (
            pd.DataFrame({"rule": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]}),
            np.asarray([1, 1, 1, 1, 1, 1]),
            np.full(6, 0.9),
            "insufficient_tp_or_fp_alerts",
        ),
        (
            pd.DataFrame({"rule": [0.1, 0.2, 0.3, 0.4, 0.5, 0.55]}),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.full(6, 0.9),
            "no_active_rule_support_among_alerts",
        ),
        (
            pd.DataFrame({"rule": [0.70, 0.80, 0.75, 0.85, 0.72, 0.82]}),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.full(6, 0.9),
            "non_varying_frozen_risk_among_alerts",
        ),
        (
            pd.DataFrame({"rule": np.full(6, 0.8)}),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.asarray([0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),
            "non_varying_rule_evidence_among_alerts",
        ),
    ],
)
def test_insufficient_audit_evidence_returns_serializable_zero_scorer(
    truth: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    expected_reason: str,
) -> None:
    result = fit_contrastive_meta_evidence(
        truth,
        labels,
        probabilities,
        0.50,
        activation_threshold=0.60,
        min_alert_rows=4,
        min_class_rows=2,
    )
    assert not result.available
    assert result.fallback_reason == expected_reason
    assert result.provenance["fail_closed"] is True
    assert result.provenance["test_data_used_for_fit"] is False
    scored = result.transform(truth, probabilities)
    np.testing.assert_array_equal(scored, np.zeros(len(truth)))
    table = result.coefficient_table()
    assert set(table.columns) >= {
        "rule",
        "feature",
        "feature_type",
        "coefficient",
        "odds_ratio",
        "available",
        "fallback_reason",
    }
    # The table is directly CSV-serializable even for a zero-rule fallback.
    assert isinstance(table.to_csv(index=False), str)


def test_contrastive_schema_and_numerical_contract_fail_loudly() -> None:
    truth, labels, probabilities = _audit_fixture()
    fitted = fit_contrastive_meta_evidence(
        truth,
        labels,
        probabilities,
        0.50,
        min_alert_rows=8,
        min_class_rows=3,
    )
    with pytest.raises(KeyError, match="missing fitted rules"):
        fitted.transform(truth.drop(columns="velocity_rule"), probabilities)
    with pytest.raises(ValueError, match="aligned"):
        fitted.transform(truth, probabilities[:-1])
    invalid = truth.copy()
    invalid.loc[0, "amount_rule"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_contrastive_meta_evidence(invalid, labels, probabilities, 0.50)
    with pytest.raises(ValueError, match="aligned"):
        fit_contrastive_meta_evidence(truth, labels[:-1], probabilities, 0.50)
