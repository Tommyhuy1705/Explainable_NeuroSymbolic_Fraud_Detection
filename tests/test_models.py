import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.models import FraudMLP, TabularResNet
from src.training import predict_torch_proba, train_torch_model


@pytest.mark.parametrize(
    "model",
    [FraudMLP(input_dim=6, hidden_dims=(12, 8), dropout=0.0), TabularResNet(6, 12, 2, 0.0)],
)
def test_model_output_shape(model):
    features = torch.randn(7, 6)
    assert model(features).shape == (7,)


def test_short_training_run_is_finite():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 1.0).astype(np.int64)
    model = FraudMLP(input_dim=6, hidden_dims=(12,), dropout=0.0)
    result = train_torch_model(
        model,
        X[:96],
        y[:96],
        X[96:],
        y[96:],
        {"epochs": 2, "batch_size": 31, "patience": 2, "learning_rate": 0.001},
        seed=42,
        device="cpu",
    )
    probabilities = predict_torch_proba(result.model, X[96:], device="cpu")
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
