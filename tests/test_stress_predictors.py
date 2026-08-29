from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.stress_testing import predictors
from src.stress_testing.metrics import fit_validation_protocol
from src.stress_testing.predictors import (
    BackendUnavailableError,
    FrozenPredictor,
    TabularResNetV2,
    freeze_predictor,
    predict_probabilities,
    resolve_safe_torch_device,
    train_stress_predictor,
)


def _training_fixture() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    features, labels = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.82, 0.18],
        random_state=11,
    )
    names = tuple(f"x{index}" for index in range(features.shape[1]))
    return features.astype(np.float32), labels.astype(int), names


def _protocol_fixture():
    labels = np.tile(np.array([0, 0, 1, 0, 1, 0]), 10)
    probabilities = np.where(labels == 1, 0.8, 0.15).astype(float)
    return fit_validation_protocol(labels, probabilities, threshold_grid_size=31)


def test_tabular_resnet_v2_is_a_distinct_versioned_architecture():
    from src.models.tabular_resnet import TabularResNet

    model = TabularResNetV2(input_dim=6, hidden_dim=16, num_blocks=1)

    assert type(model) is not TabularResNet
    assert model.architecture_version == "2.0"
    assert hasattr(model, "input_gate_logits")
    output = model(predictors.torch.zeros((4, 6)))
    assert output.shape == (4,)


def test_training_api_cannot_receive_validation_or_test_arrays():
    parameters = inspect.signature(train_stress_predictor).parameters
    assert "X_validation" not in parameters
    assert "y_validation" not in parameters
    assert "X_test" not in parameters
    assert "y_test" not in parameters


def test_full_tree_run_never_silently_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(predictors.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(BackendUnavailableError, match="never silently substitute"):
        predictors._resolve_tree_backend("xgboost", quick_run=False, allow_quick_fallback=True)
    assert (
        predictors._resolve_tree_backend("xgboost", quick_run=True, allow_quick_fallback=True)
        == "hist_gradient_boosting_quick_fixture"
    )
    with pytest.raises(BackendUnavailableError):
        predictors._resolve_tree_backend("xgboost", quick_run=True, allow_quick_fallback=False)


def test_explicit_quick_fixture_fallback_is_trained_and_recorded(monkeypatch: pytest.MonkeyPatch):
    X, y, names = _training_fixture()
    monkeypatch.setattr(predictors.importlib.util, "find_spec", lambda _name: None)

    bundle = train_stress_predictor(
        "xgboost",
        X,
        y,
        feature_names=names,
        config={"n_estimators": 4, "max_depth": 2},
        quick_run=True,
        allow_quick_fallback=True,
    )

    assert bundle.backend == "hist_gradient_boosting_quick_fixture"
    assert bundle.provenance["quick_fallback_used"] is True
    assert predict_probabilities(bundle, X[:4]).shape == (4,)


def test_explicit_cpu_device_resolution_is_auditable():
    resolution = resolve_safe_torch_device("cpu")
    assert resolution.selected == "cpu"
    assert "explicitly requested" in resolution.reason


def test_incompatible_cuda_architecture_falls_back_safely_to_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(predictors.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(predictors.torch.cuda, "get_device_capability", lambda _index: (6, 0))
    monkeypatch.setattr(predictors.torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_80"])

    resolution = resolve_safe_torch_device("cuda:0")

    assert resolution.selected == "cpu"
    assert resolution.device_capability == "sm_60"
    assert "no kernel image" in resolution.reason


def test_neural_training_is_deterministic_and_imbalance_aware_on_cpu():
    X, y, names = _training_fixture()
    config = {
        "hidden_dim": 16,
        "num_blocks": 1,
        "expansion_factor": 1,
        "dropout": 0.0,
        "epochs": 2,
        "batch_size": 64,
    }
    first = train_stress_predictor(
        "tabular_resnet_v2", X, y, feature_names=names, config=config, seed=23, quick_run=True, device="cpu"
    )
    second = train_stress_predictor(
        "tabular_resnet_v2", X, y, feature_names=names, config=config, seed=23, quick_run=True, device="cpu"
    )

    first_probabilities = predict_probabilities(first, X)
    second_probabilities = predict_probabilities(second, X)
    assert np.allclose(first_probabilities, second_probabilities, atol=1e-7)
    assert first.provenance["architecture"] == "TabularResNetV2"
    assert first.provenance["architecture_version"] == "2.0"
    assert first.provenance["imbalance_strategy"] == "BCEWithLogitsLoss.pos_weight"
    assert first.provenance["positive_class_weight"] > 1.0
    assert len(first.provenance["training_fingerprint_sha256"]) == 64


@pytest.mark.parametrize("family,package", [("xgboost", "xgboost"), ("lightgbm", "lightgbm")])
def test_exact_tree_families_train_without_backend_substitution(family: str, package: str):
    if predictors.importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed in this test environment")
    X, y, names = _training_fixture()
    config = {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}
    if family == "lightgbm":
        config["num_leaves"] = 7
    bundle = train_stress_predictor(
        family,
        X,
        y,
        feature_names=names,
        config=config,
        seed=31,
        quick_run=True,
    )

    assert bundle.backend == package
    assert bundle.provenance["quick_fallback_used"] is False
    assert bundle.provenance["positive_class_weight"] > 1.0
    probabilities = predict_probabilities(bundle, X[:9])
    assert probabilities.shape == (9,)
    assert np.isfinite(probabilities).all()


def test_freeze_binds_model_calibrator_and_threshold_before_explanations():
    X, y, names = _training_fixture()
    bundle = train_stress_predictor(
        "tabular_resnet_v2",
        X,
        y,
        feature_names=names,
        config={"hidden_dim": 12, "num_blocks": 1, "epochs": 1, "dropout": 0.0},
        seed=3,
        quick_run=True,
        device="cpu",
    )
    protocol = _protocol_fixture()

    frozen = freeze_predictor(bundle, protocol)

    assert isinstance(frozen, FrozenPredictor)
    assert frozen.frozen is True
    assert frozen.threshold == pytest.approx(protocol.threshold)
    assert frozen.provenance["frozen"] is True
    assert all(not parameter.requires_grad for parameter in frozen.model.parameters())
    assert np.isfinite(frozen.predict_proba(X[:5])).all()
