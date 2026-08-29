from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from src.stress_testing.candidate_rules import (
    NON_INTERVENABLE_HISTORY_DERIVED_FEATURES,
    combine_candidate_registries,
    fit_cart_surrogate_candidates,
    fit_configured_candidates,
    generate_counterfactual_numeric_candidates,
    score_only_baseline_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _IdentityPreprocessor:
    def __init__(self, columns: tuple[str, ...]) -> None:
        self.selected_features = list(columns)
        self.numeric_features = list(columns)
        self.columns = columns

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, self.columns].to_numpy(dtype=np.float32)


class _RecordingIdentityPreprocessor(_IdentityPreprocessor):
    def __init__(self, columns: tuple[str, ...]) -> None:
        super().__init__(columns)
        self.transformed_frames: list[pd.DataFrame] = []

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.transformed_frames.append(frame.loc[:, self.columns].copy())
        return super().transform(frame)


class _FrozenLinearPredictor:
    frozen = True
    family = "fixture_linear"
    backend = "fixture"
    threshold = 0.55

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names

    def predict_raw_proba(self, matrix: np.ndarray, *, device: str | None = None) -> np.ndarray:
        del device
        X = np.asarray(matrix, dtype=float)
        logits = 1.8 * X[:, 0] + 0.05 * X[:, 1]
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, matrix: np.ndarray, *, device: str | None = None) -> np.ndarray:
        raw = self.predict_raw_proba(matrix, device=device)
        return np.clip(0.05 + 0.90 * raw, 0.0, 1.0)


class _WeightedFrozenPredictor:
    frozen = True
    family = "fixture_weighted_linear"
    backend = "fixture"
    threshold = 0.50

    def __init__(
        self,
        feature_names: tuple[str, ...],
        weights: tuple[float, ...],
        *,
        intercept: float = -2.0,
    ) -> None:
        self.feature_names = feature_names
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)

    def predict_raw_proba(self, matrix: np.ndarray, *, device: str | None = None) -> np.ndarray:
        del device
        X = np.asarray(matrix, dtype=float)
        logits = np.clip(X @ self.weights + self.intercept, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, matrix: np.ndarray, *, device: str | None = None) -> np.ndarray:
        return self.predict_raw_proba(matrix, device=device)


def _counterfactual_fixture() -> tuple[pd.DataFrame, _IdentityPreprocessor, _FrozenLinearPredictor]:
    rows = 240
    frame = pd.DataFrame(
        {
            "risk_amount": np.linspace(-2.5, 3.5, rows),
            "nuisance": np.sin(np.linspace(0.0, 9.0, rows)),
        }
    )
    names = ("risk_amount", "nuisance")
    return frame, _IdentityPreprocessor(names), _FrozenLinearPredictor(names)


def test_configured_candidates_fit_thresholds_on_train_only():
    train = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0],
            "channel": ["web", "web", "atm", "web", "atm", "branch", "web", "atm"],
            "label": [0, 0, 0, 0, 1, 0, 1, 1],
        }
    )
    definitions = [
        {
            "name": "domain_high_amount",
            "tier": "A",
            "description": "Training-tail amount rule.",
            "conditions": [
                {
                    "feature": "amount",
                    "operator": "greater_quantile",
                    "value": 0.75,
                    "softness": 0.1,
                }
            ],
        },
        {
            "name": "train_channel_risk",
            "tier": "B",
            "conditions": [
                {"feature": "channel", "operator": "category_risk", "value": 0.5, "softness": 0.1}
            ],
        },
    ]

    result = fit_configured_candidates(train, "label", definitions)
    threshold = result.registry.set_index("rule").loc["domain_high_amount", "fitted_conditions"][0][
        "fitted_threshold"
    ]
    expected = float(train["amount"].quantile(0.75))
    assert threshold == expected
    assert set(result.registry["tier"]) == {"A", "B"}
    assert set(result.registry["fit_partition"]) == {"train"}
    assert result.provenance["validation_or_test_data_used"] is False

    # Evaluation on later extreme rows must not refit or mutate the threshold.
    later = pd.DataFrame(
        {
            "amount": [1_000_000.0, 2_000_000.0],
            "channel": ["unknown", "web"],
            "label": [1, 0],
        }
    )
    result.evaluate(later)
    threshold_after = result.engine.fitted_thresholds().query(
        "rule == 'domain_high_amount'"
    )["fitted_threshold"].iloc[0]
    assert threshold_after == expected


def test_counterfactual_candidates_are_train_only_and_test_label_invariant():
    frame, preprocessor, predictor = _counterfactual_fixture()
    parameters = inspect.signature(generate_counterfactual_numeric_candidates).parameters
    assert "y_validation" not in parameters
    assert "y_test" not in parameters
    assert "test_frame" not in parameters

    settings = dict(
        numeric_features=["risk_amount", "nuisance"],
        quantile=0.85,
        min_mean_score_drop=0.03,
        min_affected_fraction=0.40,
        per_row_material_drop=0.02,
        min_tail_rows=20,
        max_candidates=3,
        max_rows=None,
        score_mode="raw",
    )
    first = generate_counterfactual_numeric_candidates(
        frame, preprocessor, predictor, **settings
    )
    # Arbitrary later labels can change without entering candidate generation.
    y_test = np.zeros(500, dtype=int)
    y_test[::7] = 1
    y_test[:] = 1 - y_test
    second = generate_counterfactual_numeric_candidates(
        frame, preprocessor, predictor, **settings
    )

    assert y_test.sum() > 0  # the mutation above is real but irrelevant by design
    assert first.definitions == second.definitions
    assert_frame_equal(first.registry, second.registry)
    assert_frame_equal(first.diagnostics, second.diagnostics)
    assert first.provenance == second.provenance
    assert first.provenance["validation_or_test_labels_used"] is False
    assert first.definitions
    selected = first.definitions[0]
    assert selected["tier"] == "B"
    assert selected["conditions"][0]["feature"] == "risk_amount"
    assert selected["conditions"][0]["operator"] == "greater_quantile"


def test_counterfactual_quantile_and_median_are_fitted_from_training_frame():
    frame, preprocessor, predictor = _counterfactual_fixture()
    result = generate_counterfactual_numeric_candidates(
        frame,
        preprocessor,
        predictor,
        numeric_features=["risk_amount"],
        quantile=0.80,
        min_mean_score_drop=0.01,
        min_affected_fraction=0.10,
        min_tail_rows=10,
        max_rows=101,
    )

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["train_median"] == frame["risk_amount"].median()
    assert diagnostic["fitted_threshold"] == frame["risk_amount"].quantile(0.80)
    assert result.definitions[0]["conditions"][0]["value"] == 0.80
    assert result.provenance["sampling"] == "deterministic_systematic_train_sample"
    assert result.provenance["counterfactual_sample_rows"] == 101


def test_transxion_counterfactuals_recompute_semantic_features_and_ignore_stale_values():
    rows = 180
    paid = np.linspace(2.0, 92.0, rows)
    received = np.linspace(7.0, 35.0, rows) + np.sin(np.linspace(0.0, 5.0, rows))
    denominator = np.abs(paid) + np.abs(received) + np.finfo("float64").eps
    clean = pd.DataFrame(
        {
            "amount_paid": paid,
            "amount_received": received,
            "amount_paid_log": np.log1p(paid),
            "amount_received_log": np.log1p(received),
            "amount_relative_difference": np.abs(paid - received) / denominator,
        }
    )
    stale = clean.copy()
    stale["amount_paid_log"] = -1000.0
    stale["amount_received_log"] = 1000.0
    stale["amount_relative_difference"] = 0.12345
    names = tuple(clean.columns)
    contract = {
        "amount_paid": [
            {
                "output": "amount_paid_log",
                "formula": "log1p_nonnegative",
                "inputs": ["amount_paid"],
            },
            {
                "output": "amount_relative_difference",
                "formula": "absolute_relative_difference",
                "inputs": ["amount_paid", "amount_received"],
            },
        ],
        "amount_received": [
            {
                "output": "amount_received_log",
                "formula": "log1p_nonnegative",
                "inputs": ["amount_received"],
            },
            {
                "output": "amount_relative_difference",
                "formula": "absolute_relative_difference",
                "inputs": ["amount_paid", "amount_received"],
            },
        ],
    }
    predictor = _WeightedFrozenPredictor(
        names, (0.003, 0.002, 0.40, 0.10, 0.70), intercept=-2.2
    )
    settings = dict(
        numeric_features=["amount_paid", "amount_received"],
        quantile=0.80,
        min_mean_score_drop=0.0,
        min_affected_fraction=0.0,
        per_row_material_drop=0.0,
        min_tail_rows=10,
        max_candidates=2,
        max_rows=None,
        derived_feature_contract=contract,
    )
    clean_preprocessor = _RecordingIdentityPreprocessor(names)
    stale_preprocessor = _RecordingIdentityPreprocessor(names)
    clean_result = generate_counterfactual_numeric_candidates(
        clean, clean_preprocessor, predictor, **settings
    )
    stale_result = generate_counterfactual_numeric_candidates(
        stale, stale_preprocessor, predictor, **settings
    )

    assert clean_result.definitions == stale_result.definitions
    assert_frame_equal(clean_result.registry, stale_result.registry)
    assert_frame_equal(clean_result.diagnostics, stale_result.diagnostics)
    assert clean_result.provenance == stale_result.provenance
    assert clean_result.provenance["baseline_derived_features_recomputed"] is True
    assert clean_result.provenance["arbitrary_code_execution_used"] is False
    assert set(clean_result.provenance["derived_feature_recomputed_outputs"]) == {
        "amount_paid_log",
        "amount_received_log",
        "amount_relative_difference",
    }

    baseline, paid_intervention, received_intervention = (
        stale_preprocessor.transformed_frames
    )
    np.testing.assert_allclose(
        baseline["amount_paid_log"], np.log1p(baseline["amount_paid"])
    )
    np.testing.assert_allclose(
        baseline["amount_received_log"], np.log1p(baseline["amount_received"])
    )
    for transformed in (baseline, paid_intervention, received_intervention):
        expected = np.abs(
            transformed["amount_paid"] - transformed["amount_received"]
        ) / (
            np.abs(transformed["amount_paid"])
            + np.abs(transformed["amount_received"])
            + np.finfo("float64").eps
        )
        np.testing.assert_allclose(transformed["amount_relative_difference"], expected)
    assert paid_intervention["amount_paid"].nunique() == 1
    assert received_intervention["amount_received"].nunique() == 1


def test_amlnet_counterfactuals_recompute_amount_and_balance_semantics():
    rows = 160
    amount = np.linspace(1.0, 65.0, rows)
    balance = np.linspace(180.0, 45.0, rows)
    clean = pd.DataFrame(
        {
            "amount": amount,
            "sender_balance_before": balance,
            "amount_log": np.log1p(amount),
            "amount_to_sender_balance_fraction": np.clip(
                amount / np.abs(balance), 0.0, 10.0
            ),
        }
    )
    stale = clean.copy()
    stale["amount_log"] = -777.0
    stale["amount_to_sender_balance_fraction"] = 9.75
    names = tuple(clean.columns)
    contract = {
        "amount": [
            {
                "output": "amount_log",
                "formula": "log1p_nonnegative",
                "inputs": ["amount"],
            },
            {
                "output": "amount_to_sender_balance_fraction",
                "formula": "amount_to_absolute_balance_fraction",
                "inputs": ["amount", "sender_balance_before"],
            },
        ],
        "sender_balance_before": [
            {
                "output": "amount_to_sender_balance_fraction",
                "formula": "amount_to_absolute_balance_fraction",
                "inputs": ["amount", "sender_balance_before"],
            }
        ],
    }
    preprocessor = _RecordingIdentityPreprocessor(names)
    predictor = _WeightedFrozenPredictor(
        names, (0.006, -0.001, 0.35, 0.80), intercept=-1.8
    )
    result = generate_counterfactual_numeric_candidates(
        stale,
        preprocessor,
        predictor,
        numeric_features=["amount", "sender_balance_before"],
        quantile=0.80,
        min_mean_score_drop=0.0,
        min_affected_fraction=0.0,
        per_row_material_drop=0.0,
        min_tail_rows=10,
        max_candidates=2,
        max_rows=None,
        derived_feature_contract=contract,
    )

    baseline, amount_intervention, balance_intervention = preprocessor.transformed_frames
    for transformed in (baseline, amount_intervention, balance_intervention):
        np.testing.assert_allclose(
            transformed["amount_log"], np.log1p(transformed["amount"])
        )
        expected_ratio = np.clip(
            transformed["amount"]
            / np.abs(transformed["sender_balance_before"]).clip(
                lower=np.finfo("float64").eps
            ),
            0.0,
            10.0,
        )
        np.testing.assert_allclose(
            transformed["amount_to_sender_balance_fraction"], expected_ratio
        )
    assert amount_intervention["amount"].nunique() == 1
    assert balance_intervention["sender_balance_before"].nunique() == 1
    diagnostics = result.diagnostics.set_index("feature")
    assert diagnostics.loc["amount", "derived_recomputation_count"] == 2
    assert diagnostics.loc["sender_balance_before", "derived_recomputation_count"] == 1


def test_counterfactual_derived_contract_fails_closed_on_invalid_or_incomplete_specs():
    frame = pd.DataFrame(
        {
            "amount_paid": np.linspace(1.0, 20.0, 40),
            "amount_received": np.linspace(2.0, 15.0, 40),
            "amount_relative_difference": np.zeros(40),
        }
    )
    names = tuple(frame.columns)
    preprocessor = _IdentityPreprocessor(names)
    predictor = _WeightedFrozenPredictor(names, (0.1, 0.1, 0.1))
    common = dict(
        numeric_features=["amount_paid", "amount_received"],
        min_tail_rows=2,
    )

    with pytest.raises(ValueError, match="unsupported derived-feature formula"):
        generate_counterfactual_numeric_candidates(
            frame,
            preprocessor,
            predictor,
            derived_feature_contract={
                "amount_paid": [
                    {
                        "output": "amount_relative_difference",
                        "formula": "eval_user_expression",
                        "inputs": ["amount_paid", "amount_received"],
                    }
                ]
            },
            **common,
        )

    with pytest.raises(KeyError, match="columns missing from training frame"):
        generate_counterfactual_numeric_candidates(
            frame,
            preprocessor,
            predictor,
            derived_feature_contract={
                "amount_paid": [
                    {
                        "output": "missing_log_column",
                        "formula": "log1p_nonnegative",
                        "inputs": ["amount_paid"],
                    }
                ]
            },
            **common,
        )

    incomplete = {
        "amount_paid": [
            {
                "output": "amount_relative_difference",
                "formula": "absolute_relative_difference",
                "inputs": ["amount_paid", "amount_received"],
            }
        ]
    }
    with pytest.raises(ValueError, match="candidate intervention 'amount_received'"):
        generate_counterfactual_numeric_candidates(
            frame,
            preprocessor,
            predictor,
            derived_feature_contract=incomplete,
            **common,
        )


@pytest.mark.parametrize(
    "history_feature",
    sorted(NON_INTERVENABLE_HISTORY_DERIVED_FEATURES),
)
def test_history_derived_fields_cannot_be_direct_counterfactual_interventions(
    history_feature: str,
):
    rows = 80
    frame = pd.DataFrame(
        {
            "amount": np.linspace(1.0, 100.0, rows),
            history_feature: np.arange(rows, dtype=float),
        }
    )
    names = tuple(frame.columns)
    with pytest.raises(ValueError, match="history-derived features is forbidden"):
        generate_counterfactual_numeric_candidates(
            frame,
            _IdentityPreprocessor(names),
            _WeightedFrozenPredictor(names, (0.01, 0.01)),
            numeric_features=[history_feature],
            min_tail_rows=5,
        )


def test_automatic_counterfactual_discovery_excludes_history_derived_fields():
    rows = 120
    frame = pd.DataFrame(
        {
            "risk_amount": np.linspace(-2.0, 4.0, rows),
            "sender_seconds_since_previous": np.linspace(0.0, 900.0, rows),
        }
    )
    names = tuple(frame.columns)
    result = generate_counterfactual_numeric_candidates(
        frame,
        _IdentityPreprocessor(names),
        _WeightedFrozenPredictor(names, (1.5, 0.001), intercept=-1.0),
        numeric_features=None,
        quantile=0.80,
        min_mean_score_drop=0.0,
        min_affected_fraction=0.0,
        per_row_material_drop=0.0,
        min_tail_rows=10,
    )

    assert set(result.diagnostics["feature"]) == {"risk_amount"}
    assert result.provenance["automatically_excluded_derived_features"] == [
        "sender_seconds_since_previous"
    ]
    assert result.provenance["direct_derived_feature_intervention_used"] is False
    if not result.registry.empty:
        assert set(result.registry["feature_space"]) == {"raw_causal_features"}
        assert not result.registry["direct_intervention_is_derived"].astype(bool).any()


def test_declared_semantic_derived_output_cannot_be_directly_intervened():
    frame = pd.DataFrame(
        {
            "amount_paid": np.linspace(1.0, 20.0, 40),
            "amount_received": np.linspace(2.0, 15.0, 40),
            "amount_relative_difference": np.linspace(0.0, 1.0, 40),
        }
    )
    names = tuple(frame.columns)
    shared_spec = {
        "output": "amount_relative_difference",
        "formula": "absolute_relative_difference",
        "inputs": ["amount_paid", "amount_received"],
    }
    with pytest.raises(ValueError, match="derived"):
        generate_counterfactual_numeric_candidates(
            frame,
            _IdentityPreprocessor(names),
            _WeightedFrozenPredictor(names, (0.1, 0.1, 0.1)),
            numeric_features=["amount_relative_difference"],
            derived_feature_contract={
                "amount_paid": [shared_spec],
                "amount_received": [shared_spec],
            },
            min_tail_rows=2,
        )


def test_configured_registry_distinguishes_history_derived_from_raw_features():
    frame = pd.DataFrame(
        {
            "amount": np.arange(1.0, 13.0),
            "sender_seconds_since_previous": np.arange(12.0),
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    result = fit_configured_candidates(
        frame,
        "label",
        [
            {
                "name": "rapid_sender_high_amount",
                "tier": "A",
                "conditions": [
                    {
                        "feature": "sender_seconds_since_previous",
                        "operator": "less_quantile",
                        "value": 0.25,
                    },
                    {
                        "feature": "amount",
                        "operator": "greater_quantile",
                        "value": 0.75,
                    },
                ],
            }
        ],
    )
    row = result.registry.iloc[0]
    assert row["feature_space"] == "mixed_raw_and_causally_derived_features"
    assert row["raw_features"] == ["amount"]
    assert row["derived_features"] == ["sender_seconds_since_previous"]
    assert result.provenance["configured_history_features_are_direct_interventions"] is False


def test_stress_configs_remove_history_fields_from_direct_intervention_candidates():
    required = set(NON_INTERVENABLE_HISTORY_DERIVED_FEATURES)
    for filename in ("transxion_v2.yaml", "amlnet_v1.yaml"):
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "stress" / filename).read_text(encoding="utf-8")
        )
        counterfactual = config["counterfactual_candidates"]
        direct = set(counterfactual["numeric_features"])
        declared_derived = set(counterfactual["non_intervenable_derived_features"])
        assert required <= declared_derived
        assert direct.isdisjoint(declared_derived)


def test_cart_surrogate_paths_are_deterministic_and_exact_leaf_activations():
    frame, preprocessor, predictor = _counterfactual_fixture()
    X_train = preprocessor.transform(frame)
    parameters = inspect.signature(fit_cart_surrogate_candidates).parameters
    assert "y_train" not in parameters
    assert "y_validation" not in parameters
    assert "y_test" not in parameters

    settings = dict(
        feature_names=predictor.feature_names,
        surrogate_target="score",
        score_mode="raw",
        max_depth=3,
        min_samples_leaf=15,
        max_rules=5,
        random_state=17,
    )
    first = fit_cart_surrogate_candidates(X_train, predictor, **settings)
    second = fit_cart_surrogate_candidates(X_train, predictor, **settings)

    assert_frame_equal(first.registry, second.registry)
    assert first.provenance == second.provenance
    assert first.provenance["ground_truth_labels_used"] is False
    assert first.provenance["validation_or_test_data_used"] is False
    truth = first.evaluate(X_train, feature_names=predictor.feature_names)
    applied_leaf = first.model.apply(X_train)
    for rule in first.rules:
        assert np.array_equal(
            truth[rule.name].to_numpy(dtype=bool), applied_leaf == rule.leaf_id
        )
        assert all(np.isfinite(condition.threshold) for condition in rule.conditions)


def test_score_only_baseline_is_explicit_and_registry_combination_is_stable():
    frame, preprocessor, predictor = _counterfactual_fixture()
    generated = generate_counterfactual_numeric_candidates(
        frame,
        preprocessor,
        predictor,
        numeric_features=["risk_amount"],
        quantile=0.80,
        min_mean_score_drop=0.01,
        min_affected_fraction=0.10,
        min_tail_rows=10,
    )
    baseline = score_only_baseline_registry(predictor)
    combined = combine_candidate_registries(generated.registry, baseline)

    baseline_row = combined.set_index("rule").loc["predictor_score_only"]
    assert baseline_row["tier"] == "baseline"
    assert bool(baseline_row["requires_rule_evidence"]) is False
    assert baseline_row["decision_threshold"] == predictor.threshold
    assert combined.iloc[-1]["rule"] == "predictor_score_only"
