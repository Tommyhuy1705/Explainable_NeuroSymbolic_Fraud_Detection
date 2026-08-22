import numpy as np
import pandas as pd
import pytest

from src.data import load_config
from src.logic import FraudRuleEngine
from src.logic.predicates import greater_than, less_than, outside_range


ACTIVATION_THRESHOLD = 0.60


def _rule_definition(config_path: str, rule_name: str) -> dict:
    config = load_config(config_path)
    return next(rule for rule in config["logic"]["rules"] if rule["name"] == rule_name)


def test_predicate_default_preserves_relative_softness_behaviour():
    values = np.array([80.0, 100.0, 120.0])
    expected = 1.0 / (1.0 + np.exp(-((values - 100.0) / 20.0)))

    assert np.allclose(greater_than(values, 100.0, 0.20), expected)


def test_absolute_softness_uses_native_feature_units():
    values = np.array([18.0, 23.0, 24.0, 25.0, 30.0])
    truth = less_than(values, 25.0, 3.0, softness_mode="absolute")

    assert truth[2] < ACTIVATION_THRESHOLD
    assert truth[3] == pytest.approx(0.5)
    assert np.array_equal(values[truth >= ACTIVATION_THRESHOLD], np.array([18.0, 23.0]))


def test_ieee_unusual_hour_rule_activates_configured_early_hours():
    definition = _rule_definition(
        "configs/ieee_cis.yaml",
        "unusual_transaction_hour",
    )
    hours = np.arange(24, dtype=float)
    train = pd.DataFrame({"transaction_hour": hours, "isFraud": hours % 7 == 0})
    engine = FraudRuleEngine([definition]).fit(train, "isFraud")

    condition = engine.rules[0].conditions[0]
    truth = engine.evaluate(train)[definition["name"]].to_numpy()

    assert condition.softness_mode == "absolute"
    assert np.array_equal(hours[truth >= ACTIVATION_THRESHOLD], np.arange(6, dtype=float))
    assert truth[5] >= ACTIVATION_THRESHOLD
    assert truth[6] == pytest.approx(0.5, abs=1e-5)
    assert truth[23] == pytest.approx(0.5, abs=1e-5)


def test_baf_young_high_income_rule_can_activate_at_configured_threshold():
    definition = _rule_definition("configs/baf.yaml", "young_high_income_request")
    train = pd.DataFrame(
        {
            "customer_age": [18, 20, 22, 24, 26, 30, 35, 40, 50, 60],
            "income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "fraud_bool": [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "customer_age": [20, 24, 30],
            "income": [140, 140, 140],
            "fraud_bool": [1, 0, 0],
        }
    )
    engine = FraudRuleEngine([definition]).fit(train, "fraud_bool")

    age_condition, income_condition = engine.rules[0].conditions
    truth = engine.evaluate(test)[definition["name"]].to_numpy()

    assert age_condition.softness_mode == "absolute"
    assert income_condition.softness_mode == "relative"
    assert truth[0] >= ACTIVATION_THRESHOLD
    assert truth[1] < ACTIVATION_THRESHOLD
    assert truth[2] < ACTIVATION_THRESHOLD


def test_invalid_softness_configuration_fails_fast():
    with pytest.raises(ValueError, match="finite, positive"):
        outside_range(np.array([5.0]), 6.0, 23.0, softness=0.0, softness_mode="absolute")
    with pytest.raises(ValueError, match="Unsupported softness mode"):
        greater_than(np.array([1.0]), 0.0, softness=1.0, softness_mode="unknown")


def test_numeric_missing_values_reuse_training_median_on_later_splits():
    rule = {
        "name": "large_value",
        "description": "Large numeric value",
        "conditions": [
            {"feature": "amount", "operator": "greater", "value": 50.0, "softness": 10.0}
        ],
    }
    train = pd.DataFrame({"amount": [0.0, 10.0, np.nan], "isFraud": [0, 1, 0]})
    test = pd.DataFrame({"amount": [np.nan, 100.0, 200.0], "isFraud": [0, 1, 1]})
    engine = FraudRuleEngine([rule]).fit(train, "isFraud")

    condition = engine.rules[0].conditions[0]
    truth = engine.evaluate(test)["large_value"].to_numpy()

    assert condition.numeric_fill_value == pytest.approx(5.0)
    assert truth[0] == pytest.approx(greater_than(np.array([5.0]), 50.0, 10.0, "absolute")[0])
    assert truth[0] < 0.02
    assert engine.fitted_thresholds().loc[0, "fitted_missing_value"] == pytest.approx(5.0)
