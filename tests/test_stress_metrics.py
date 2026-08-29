from __future__ import annotations

import numpy as np
import pytest

from src.stress_testing.metrics import (
    evaluate_stress_predictions,
    fit_reserved_validation_protocol,
    fit_validation_protocol,
    make_validation_partitions,
    recall_at_fpr,
)


def _ordered_validation_fixture(rows: int = 60) -> tuple[np.ndarray, np.ndarray]:
    labels = np.tile(np.array([0, 0, 0, 1, 0, 1], dtype=int), rows // 6)
    probabilities = np.where(labels == 1, 0.78, 0.12).astype(float)
    probabilities += np.linspace(-0.03, 0.03, len(labels))
    return labels, np.clip(probabilities, 0.01, 0.99)


def test_metrics_report_required_scores_and_prevalence_baselines():
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    probabilities = np.array([0.01, 0.08, 0.20, 0.40, 0.75, 0.95])

    metrics = evaluate_stress_predictions(labels, probabilities, threshold=0.5)

    required = {
        "pr_auc",
        "roc_auc",
        "f2",
        "precision",
        "recall",
        "recall_at_1pct_fpr",
        "brier",
        "ece",
        "nll",
        "null_prevalence",
        "null_pr_auc",
        "null_roc_auc",
        "null_f2",
        "null_brier",
        "null_ece",
        "null_nll",
    }
    assert required.issubset(metrics)
    assert metrics["null_prevalence"] == pytest.approx(1 / 3)
    assert metrics["null_pr_auc"] == pytest.approx(1 / 3)
    assert metrics["null_roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc_lift_over_null"] > 0.0
    assert metrics["brier_improvement_over_null"] > 0.0
    assert all(np.isfinite(metrics[key]) for key in required)


def test_recall_at_one_percent_fpr_uses_attained_roc_points():
    labels = np.array([0] * 100 + [1] * 4)
    perfect = np.array([0.01] * 100 + [0.99] * 4)
    reversed_scores = np.array([0.99] * 100 + [0.01] * 4)

    assert recall_at_fpr(labels, perfect, target_fpr=0.01) == pytest.approx(1.0)
    assert recall_at_fpr(labels, reversed_scores, target_fpr=0.01) == pytest.approx(0.0)


def test_validation_partitions_are_ordered_disjoint_and_exhaustive():
    labels, _ = _ordered_validation_fixture()
    partitions = make_validation_partitions(labels)

    assert np.array_equal(partitions.calibrator_fit, np.arange(0, 24))
    assert np.array_equal(partitions.calibrator_selection, np.arange(24, 42))
    assert np.array_equal(partitions.threshold_selection, np.arange(42, 60))
    combined = np.concatenate(
        [partitions.calibrator_fit, partitions.calibrator_selection, partitions.threshold_selection]
    )
    assert np.array_equal(np.sort(combined), np.arange(60))
    assert len(np.unique(combined)) == 60


def test_validation_protocol_freezes_calibrator_and_f2_threshold_with_provenance():
    labels, raw = _ordered_validation_fixture()

    protocol = fit_validation_protocol(labels, raw, seed=17, threshold_grid_size=51)

    assert protocol.frozen is True
    assert protocol.calibrator.method in {"none", "platt", "isotonic"}
    assert 0.0 <= protocol.threshold <= 1.0
    assert 0.0 <= protocol.threshold_f2 <= 1.0
    assert len(protocol.comparison) == 3
    assert protocol.provenance["threshold_objective"] == "f2"
    assert protocol.provenance["validation_rows"] == 60
    assert len(protocol.provenance["validation_fingerprint_sha256"]) == 64
    calibrated = protocol.transform(raw)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()


def test_reserved_validation_protocol_uses_declared_outer_roles_without_resplitting():
    fit_labels, fit_raw = _ordered_validation_fixture(60)
    select_labels, select_raw = _ordered_validation_fixture(72)

    protocol = fit_reserved_validation_protocol(
        fit_labels,
        fit_raw,
        select_labels,
        select_raw,
        seed=17,
        threshold_grid_size=51,
    )

    assert protocol.partitions is None
    assert protocol.provenance["outer_roles_used_without_resplitting"] is True
    assert protocol.provenance["role_contract"] == {
        "calibrator_fit": "calibration_fit",
        "calibrator_selection": "calibration_select",
        "threshold_selection": "calibration_select",
    }
    assert protocol.provenance["calibrator_fit_rows"] == 60
    assert protocol.provenance["calibration_select_rows"] == 72
    assert len(protocol.provenance["reserved_role_fingerprint_sha256"]) == 64


def test_validation_partition_fails_loudly_when_a_role_has_one_class():
    labels = np.array([0] * 20 + [1] * 10)
    with pytest.raises(ValueError, match="needs both classes"):
        make_validation_partitions(labels)


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, np.nan])
def test_metrics_reject_invalid_thresholds(bad_threshold: float):
    labels, probabilities = _ordered_validation_fixture()
    with pytest.raises(ValueError, match="threshold"):
        evaluate_stress_predictions(labels, probabilities, threshold=bad_threshold)
