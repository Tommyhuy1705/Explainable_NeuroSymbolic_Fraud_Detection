import numpy as np

from src.models import build_tree_classifier


def test_tree_classifier_produces_binary_probabilities():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 6))
    labels = (features[:, 0] + features[:, 1] > 1.0).astype(int)
    model = build_tree_classifier(
        {
            "backend": "hist_gradient_boosting",
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        random_state=42,
    )
    model.fit(features[:90], labels[:90])
    probabilities = model.predict_proba(features[90:])[:, 1]
    assert probabilities.shape == (30,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
