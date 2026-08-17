"""Prediction metrics and leakage-safe evaluation protocol."""

from .prediction_metrics import (
    evaluate_binary_predictions,
    expected_calibration_error,
    paired_bootstrap_pr_auc_difference,
)
from .protocol import (
    CalibrationSelection,
    ProbabilityCalibrator,
    select_probability_calibrator,
    select_threshold,
)

__all__ = [
    "ProbabilityCalibrator",
    "CalibrationSelection",
    "evaluate_binary_predictions",
    "expected_calibration_error",
    "paired_bootstrap_pr_auc_difference",
    "select_probability_calibrator",
    "select_threshold",
]
