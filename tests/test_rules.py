import numpy as np

from src.data import load_config, make_synthetic_fraud_data
from src.explanation import (
    RuleExplainer,
    bootstrap_explanation_precision_gain,
    explanation_quality_metrics,
    rule_quality_table,
)
from src.logic import FraudRuleEngine


def test_rule_engine_and_explainer_outputs():
    config = load_config("configs/ieee_cis.yaml")
    frame = make_synthetic_fraud_data(1200, seed=11)
    train = frame.iloc[:800]
    test = frame.iloc[800:]
    engine = FraudRuleEngine(config["logic"]["rules"]).fit(train, "isFraud")
    truth = engine.evaluate(test)
    assert len(truth) == len(test)
    assert truth.shape[1] >= 3
    assert ((truth >= 0.0) & (truth <= 1.0)).all().all()

    probabilities = np.clip(0.05 + 0.9 * truth.max(axis=1).to_numpy(), 0, 1)
    explainer = RuleExplainer(engine, activation_threshold=0.6, top_k_rules=2)
    explanations = explainer.explain(test, probabilities, decision_threshold=0.5)
    assert (explanations["active_rule_count"] >= explanations["displayed_rule_count"]).all()
    assert (explanations["displayed_rule_count"] <= 2).all()
    assert (explanations["available_rule_count"] == truth.shape[1]).all()
    metrics = explanation_quality_metrics(explanations, test["isFraud"], probabilities, 0.5)
    rule_table = rule_quality_table(truth, test["isFraud"], 0.6)
    assert 0.0 <= metrics["explanation_coverage_all"] <= 1.0
    assert "explained_alert_precision_gain" in metrics
    assert metrics["mean_rule_count"] == metrics["mean_active_rule_count"]
    assert 0.0 <= metrics["rule_sparsity"] <= 1.0
    assert set(["coverage", "fraud_precision", "lift"]).issubset(rule_table.columns)
    interval = bootstrap_explanation_precision_gain(
        explanations,
        test["isFraud"],
        probabilities,
        0.5,
        n_bootstrap=50,
        seed=42,
    )
    assert interval["bootstrap_valid_iterations"] > 0
    assert interval["precision_gain_ci_low"] <= interval["precision_gain_ci_high"]
