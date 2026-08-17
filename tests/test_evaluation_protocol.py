import numpy as np

from src.evaluation import (
    ProbabilityCalibrator,
    evaluate_binary_predictions,
    paired_bootstrap_pr_auc_difference,
    select_probability_calibrator,
    select_threshold,
)


def test_threshold_selection_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.30, 0.55, 0.80, 0.95])
    threshold, score = select_threshold(y, probabilities, objective="f2")
    metrics = evaluate_binary_predictions(y, probabilities, threshold, beta=2.0)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= score <= 1.0
    assert metrics["pr_auc"] > 0.9


def test_none_calibration_is_identity():
    probabilities = np.array([0.1, 0.4, 0.8])
    calibrator = ProbabilityCalibrator("none").fit(probabilities, np.array([0, 0, 1]))
    assert np.allclose(calibrator.transform(probabilities), probabilities)


def test_calibration_selection_uses_disjoint_validation_partitions():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = np.array([0.05, 0.10, 0.70, 0.20, 0.65, 0.15, 0.30, 0.80, 0.25, 0.75])
    selection = select_probability_calibrator(y, probabilities, fit_fraction=0.5)
    assert not np.intersect1d(selection.fit_indices, selection.selection_indices).size
    assert set(selection.comparison[0]) >= {"method", "brier", "ece", "nll"}
    assert selection.method in {"none", "platt", "isotonic"}


def test_paired_bootstrap_detects_better_ranking():
    y = np.array([0] * 80 + [1] * 20)
    better = np.linspace(0.0, 1.0, 100)
    worse = np.full(100, 0.5)
    result = paired_bootstrap_pr_auc_difference(y, better, worse, n_bootstrap=30, seed=4)
    assert result["observed_difference"] > 0
    assert result["ci_high"] > 0
