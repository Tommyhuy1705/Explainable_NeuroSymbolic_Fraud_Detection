import numpy as np

from src.evaluation import ProbabilityCalibrator, evaluate_binary_predictions, select_threshold


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
