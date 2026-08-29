from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.stress_testing.attribution import compute_attributions
from src.stress_testing.metrics import fit_validation_protocol
from src.stress_testing.predictors import freeze_predictor, train_stress_predictor


def _fixture() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    features, labels = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=9,
    )
    return features.astype(np.float32), labels.astype(int), tuple(f"f{index}" for index in range(5))


def _protocol():
    labels = np.tile(np.array([0, 0, 0, 1, 0, 1]), 10)
    probabilities = np.where(labels == 1, 0.82, 0.14).astype(float)
    return fit_validation_protocol(labels, probabilities, threshold_grid_size=31)


def test_neural_gradient_times_input_is_finite_and_feature_aligned():
    X, y, names = _fixture()
    trained = train_stress_predictor(
        "tabular_resnet_v2",
        X,
        y,
        feature_names=names,
        config={"hidden_dim": 12, "num_blocks": 1, "epochs": 1, "dropout": 0.0},
        seed=13,
        quick_run=True,
        device="cpu",
    )
    frozen = freeze_predictor(trained, _protocol())

    result = compute_attributions(frozen, X[:7], feature_names=names, batch_size=3, device="cpu")

    assert result.values.shape == (7, 5)
    assert result.base_values.shape == (7,)
    assert result.raw_outputs.shape == (7,)
    assert np.isfinite(result.values).all()
    assert result.feature_names == names
    assert result.method == "gradient_times_input_raw_logit"
    assert result.provenance["model_calibrator_threshold_frozen"] is True
    assert set(result.absolute_feature_importance()) == set(names)


def test_attribution_rejects_unfrozen_model_and_wrong_feature_order():
    X, y, names = _fixture()
    trained = train_stress_predictor(
        "tabular_resnet_v2",
        X,
        y,
        feature_names=names,
        config={"hidden_dim": 12, "num_blocks": 1, "epochs": 1},
        quick_run=True,
        device="cpu",
    )
    with pytest.raises(TypeError, match="FrozenPredictor"):
        compute_attributions(trained, X[:2], feature_names=names)  # type: ignore[arg-type]

    frozen = freeze_predictor(trained, _protocol())
    with pytest.raises(ValueError, match="feature names/order"):
        compute_attributions(frozen, X[:2], feature_names=tuple(reversed(names)))


def test_xgboost_uses_native_pred_contribs_with_bias_column():
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("xgboost is not installed in this test environment")
    X, y, names = _fixture()
    trained = train_stress_predictor(
        "xgboost",
        X,
        y,
        feature_names=names,
        config={"n_estimators": 6, "max_depth": 2, "n_jobs": 1},
        seed=7,
        quick_run=True,
    )
    frozen = freeze_predictor(trained, _protocol())

    result = compute_attributions(frozen, X[:8], feature_names=names)

    assert result.method == "xgboost_pred_contribs"
    assert result.values.shape == (8, len(names))
    assert np.allclose(result.values.sum(axis=1) + result.base_values, result.raw_outputs)
    assert np.isfinite(result.values).all()
