"""Protocol-order and end-to-end tests for the thesis stress orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.stress_testing.experiment import (
    PROTOCOL_PHASES,
    _ProtocolLedger,
    _assert_no_leakage_features,
    _conservative_stress_outcome,
    _independent_ablation_rule_pool,
    _negative_control_rows,
    _add_guarded_ensemble,
    _positive_top_k_attribution_evidence,
    _selection_signature,
    _validate_protocol_config,
    run_stress_test,
)
from src.stress_testing.artifacts import load_yaml_config
from src.stress_testing.selective_policy import LockedSelectivePolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _locked_policy() -> LockedSelectivePolicy:
    return LockedSelectivePolicy(
        dataset="fixture",
        predictor="xgboost:seed=42",
        coverage=0.50,
        method="audit_weighted",
        abstain=False,
        reason="validation_guardrails_passed",
        validation_alert_count=4,
        validation_selected_count=2,
        validation_all_alert_precision=0.50,
        validation_score_only_precision=0.50,
        validation_selected_precision=0.75,
        validation_delta_vs_score_only=0.25,
        validation_evidence_cutoff=0.60,
        minimum_score_only_delta=0.0,
    )


def test_protocol_ledger_is_strict_and_complete() -> None:
    ledger = _ProtocolLedger()
    with pytest.raises(RuntimeError, match="expected 'data_loaded'"):
        ledger.advance("temporal_split_locked")
    for phase in PROTOCOL_PHASES:
        ledger.advance(phase)
    assert ledger.complete
    assert [row["sequence"] for row in ledger.events] == list(range(1, len(PROTOCOL_PHASES) + 1))


def test_leakage_denylist_is_case_insensitive_and_fail_closed() -> None:
    _assert_no_leakage_features(["amount", "event_hour"], ["label", "fraud_probability"])
    with pytest.raises(AssertionError, match="Leakage-denied"):
        _assert_no_leakage_features(
            ["amount", "Fraud_Probability"],
            ["label", "fraud_probability"],
        )


def test_locked_test_selection_is_invariant_to_any_test_label_permutation() -> None:
    probabilities = np.asarray([0.91, 0.88, 0.73, 0.65, 0.20, 0.10])
    evidence = {"audit_weighted": np.asarray([0.10, 0.90, 0.80, 0.20, 0.50, 0.40])}
    policy = _locked_policy()
    signature, masks = _selection_signature([policy], probabilities, 0.60, evidence)
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)

    for seed in (1, 42, 2026):
        permuted_labels = np.random.default_rng(seed).permutation(labels)
        # The policy application API has no label argument.  Outcome
        # permutation therefore cannot affect a previously locked mask.
        repeated, repeated_masks = _selection_signature([policy], probabilities, 0.60, evidence)
        assert len(permuted_labels) == len(probabilities)
        assert repeated == signature
        np.testing.assert_array_equal(repeated_masks[0.50], masks[0.50])


def test_negative_control_rankings_preserve_the_locked_alert_budget() -> None:
    policy = _locked_policy()
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray([0.95, 0.90, 0.85, 0.80, 0.75, 0.70])
    evidence = np.asarray([0.90, 0.80, 0.70, 0.0, 0.0, 0.0])
    selected = np.asarray([True, True, True, False, False, False])
    score_only = np.asarray([True, True, True, False, False, False])
    rows = _negative_control_rows(
        policy,
        labels,
        probabilities,
        0.50,
        selected,
        score_only,
        evidence,
        seed=42,
    )
    assert {row["requested_selected_count"] for row in rows} == {3}
    assert {row["selected_count"] for row in rows} == {3}
    assert {row["achieved_alert_coverage"] for row in rows} == {0.5}


def test_top_k_attribution_baseline_uses_signed_positive_local_evidence() -> None:
    values = np.asarray(
        [
            [4.0, -9.0, 3.0, 0.1, 0.2, 0.3],
            [-8.0, -7.0, -6.0, -5.0, -4.0, 0.1],
        ]
    )
    evidence = _positive_top_k_attribution_evidence(values, top_k=5)
    # Row one keeps 4, 3, 0.3 and 0.2 among its five largest absolute effects;
    # row two's top-five effects all point away from the fraud logit.
    np.testing.assert_allclose(evidence, np.asarray([7.5, 0.0]))


def test_guarded_ensemble_is_label_free_aligned_and_fail_closed() -> None:
    evidence = _add_guarded_ensemble(
        {
            "fp_penalized": np.asarray([0.0, 0.6, 0.9]),
            "attribution_gated": np.asarray([0.0, 0.3, 0.6]),
            "contrastive_meta": np.asarray([0.0, 0.9, 0.3]),
        },
        ["fp_penalized", "attribution_gated", "contrastive_meta"],
    )
    np.testing.assert_allclose(evidence["guarded_ensemble"], [0.0, 0.6, 0.6])
    with pytest.raises(KeyError, match="unavailable"):
        _add_guarded_ensemble({"fp_penalized": np.ones(2)}, ["fp_penalized", "missing"])


def test_ablation_rule_pool_reselects_from_complete_pre_dedup_audit() -> None:
    audit = pd.DataFrame(
        {
            "rule": ["tier_b_global_first", "tier_a_first", "tier_a_budgeted_globally"],
            "audit_pass": [True, True, True],
            "audit_alert_score": [0.95, 0.85, 0.75],
            "audit_alert_tp_activation": [0.90, 0.80, 0.70],
            "audit_alert_fp_activation": [0.10, 0.10, 0.10],
        }
    )
    truth = pd.DataFrame(
        {
            "tier_b_global_first": [1.0, 1.0, 0.0, 0.0],
            "tier_a_first": [1.0, 0.0, 1.0, 0.0],
            "tier_a_budgeted_globally": [0.0, 1.0, 0.0, 1.0],
        }
    )

    completed, _, weights, provenance = _independent_ablation_rule_pool(
        audit,
        truth,
        {"tier_a_first", "tier_a_budgeted_globally"},
        variant="domain_tier_a_only",
        pool_definition="all_tier_a_rules",
        audit_alert_mask=np.ones(len(truth), dtype=bool),
        activation_threshold=0.60,
        max_jaccard=0.85,
        max_rules=2,
        fp_penalty=1.0,
    )

    # A global max-rules=2 pass would spend one slot on the higher-scoring Tier
    # B rule and discard ``tier_a_budgeted_globally``.  Independent Tier-A
    # selection must recover both valid Tier-A rules from the complete audit.
    assert set(weights["rule"]) == {"tier_a_first", "tier_a_budgeted_globally"}
    assert completed["selected"].astype(bool).all()
    assert provenance["candidate_source_table"] == "rule_audit_before_primary_deduplication"
    assert provenance["deduplication_recomputed_within_pool"] is True
    assert provenance["weights_recomputed_within_pool"] is True
    assert provenance["primary_selected_weights_filtered"] is False
    assert provenance["test_labels_used"] is False


def test_low_score_headroom_alone_is_not_called_saturation() -> None:
    outcome, diagnostics = _conservative_stress_outcome(
        {"abstain": True, "score_only_headroom": 0.01},
        {"ci_high": float("nan")},
        {"matched_ci_high": float("nan")},
        {"risk_conditioned_tp_fp_gap": float("nan")},
        [],
        pd.DataFrame(),
        saturation_headroom=0.05,
    )
    assert outcome == "low_score_only_headroom_unconfirmed"
    assert diagnostics["saturation_confirmed"] is False


def test_saturation_label_is_restricted_to_eligible_amlnet_full_runs() -> None:
    outcome, diagnostics = _conservative_stress_outcome(
        {
            "abstain": True,
            "score_only_headroom": 0.01,
            "alert_positive_count": 25,
        },
        {"ci_high": 0.01, "valid_iterations": 500},
        {"matched_ci_high": 0.01, "matched_pairs": 25},
        {"risk_conditioned_tp_fp_gap": 0.01},
        [],
        pd.DataFrame(),
        saturation_headroom=0.05,
        saturation_scope_eligible=False,
    )
    assert outcome == "low_score_only_headroom_unconfirmed"
    assert diagnostics["saturation_scope_eligible"] is False
    assert diagnostics["saturation_sample_adequate"] is True


def test_outcome_thresholds_must_be_explicit_and_bounded() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs/stress/transxion_v2.yaml")
    _validate_protocol_config(config)

    missing = load_yaml_config(PROJECT_ROOT / "configs/stress/transxion_v2.yaml")
    del missing["evaluation"]["saturation_headroom"]
    with pytest.raises(ValueError, match="pre-register outcome thresholds"):
        _validate_protocol_config(missing)

    invalid = load_yaml_config(PROJECT_ROOT / "configs/stress/transxion_v2.yaml")
    invalid["evaluation"]["minimum_positive_stability_fraction"] = 1.1
    with pytest.raises(ValueError, match="minimum_positive_stability_fraction"):
        _validate_protocol_config(invalid)

    missing_contrastive = load_yaml_config(
        PROJECT_ROOT / "configs/stress/transxion_v2.yaml"
    )
    missing_contrastive["logic"]["evidence_methods"].remove("contrastive_meta")
    with pytest.raises(ValueError, match="contrastive_meta"):
        _validate_protocol_config(missing_contrastive)

    missing_guarded = load_yaml_config(PROJECT_ROOT / "configs/stress/transxion_v2.yaml")
    del missing_guarded["logic"]["guarded_ensemble"]
    with pytest.raises(ValueError, match="guarded ensemble"):
        _validate_protocol_config(missing_guarded)

    derived_intervention = load_yaml_config(
        PROJECT_ROOT / "configs/stress/amlnet_v1.yaml"
    )
    derived_intervention["counterfactual_candidates"]["numeric_features"].append(
        "amount_log"
    )
    with pytest.raises(ValueError, match="derived outputs"):
        _validate_protocol_config(derived_intervention)

    history_intervention = load_yaml_config(
        PROJECT_ROOT / "configs/stress/transxion_v2.yaml"
    )
    history_intervention["counterfactual_candidates"]["numeric_features"].append(
        "sender_seconds_since_previous"
    )
    with pytest.raises(ValueError, match="history-derived fields"):
        _validate_protocol_config(history_intervention)

    undeclared_history = load_yaml_config(
        PROJECT_ROOT / "configs/stress/transxion_v2.yaml"
    )
    undeclared_history["counterfactual_candidates"][
        "non_intervenable_derived_features"
    ].remove("pair_prior_transaction_count")
    with pytest.raises(ValueError, match="declare all non-intervenable"):
        _validate_protocol_config(undeclared_history)


def test_positive_incremental_evidence_requires_convergent_diagnostics() -> None:
    controls = [
        {"control": "locked_vasre", "selected_precision": 0.80, "selected_count": 10},
        {"control": "predictor_score_only", "selected_precision": 0.60, "selected_count": 10},
        {"control": "row_permuted_evidence", "selected_precision": 0.55, "selected_count": 10},
        {"control": "random_evidence", "selected_precision": 0.50, "selected_count": 10},
    ]
    shuffled_controls = [
        {
            "variant": "validation_label_shuffle_control",
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.62,
            "delta_vs_score_only": 0.02,
            "abstain": False,
        },
        {
            "variant": "validation_weight_shuffle_control",
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.65,
            "delta_vs_score_only": 0.01,
            "abstain": False,
        },
    ]
    stable = pd.DataFrame(
        {
            "full_validation_delta_vs_score_only": [0.08] * 5,
            "method_matches_reference": [True] * 5,
            "abstain": [False] * 5,
        }
    )
    outcome, diagnostics = _conservative_stress_outcome(
        {
            "abstain": False,
            "score_only_headroom": 0.30,
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.80,
            "equal_count_score_only_comparison": True,
        },
        {"ci_low": 0.01, "ci_high": 0.10},
        {"matched_ci_low": 0.01, "matched_ci_high": 0.08},
        {"risk_conditioned_tp_fp_gap": 0.06},
        controls,
        stable,
        shuffled_controls=shuffled_controls,
        saturation_headroom=0.05,
        minimum_positive_stability_fraction=0.80,
    )
    assert outcome == "positive_incremental_evidence"
    assert diagnostics["positive_incremental_evidence_confirmed"] is True
    assert diagnostics["positive_stability_fraction"] == 1.0

    contradictory = stable.copy()
    contradictory["method_matches_reference"] = False
    outcome, diagnostics = _conservative_stress_outcome(
        {
            "abstain": False,
            "score_only_headroom": 0.30,
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.80,
            "equal_count_score_only_comparison": True,
        },
        {"ci_low": 0.01, "ci_high": 0.10},
        {"matched_ci_low": -0.01, "matched_ci_high": 0.08},
        {"risk_conditioned_tp_fp_gap": 0.06},
        controls,
        contradictory,
        shuffled_controls=shuffled_controls,
        saturation_headroom=0.05,
        minimum_positive_stability_fraction=0.80,
    )
    assert outcome == "inconclusive_incremental_evidence"
    assert diagnostics["positive_incremental_evidence_confirmed"] is False
    assert diagnostics["matched_risk_supports_positive"] is False
    assert diagnostics["validation_stability_supports_positive"] is False

    equivalent_controls = [dict(row) for row in controls]
    equivalent_controls[-1]["selected_precision"] = 0.80
    outcome, diagnostics = _conservative_stress_outcome(
        {
            "abstain": False,
            "score_only_headroom": 0.30,
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.80,
            "equal_count_score_only_comparison": True,
        },
        {"ci_low": 0.01, "ci_high": 0.10},
        {"matched_ci_low": 0.01, "matched_ci_high": 0.08},
        {"risk_conditioned_tp_fp_gap": 0.06},
        equivalent_controls,
        stable,
        shuffled_controls=shuffled_controls,
        saturation_headroom=0.05,
        minimum_positive_stability_fraction=0.80,
    )
    assert outcome == "inconclusive_incremental_evidence"
    assert diagnostics["negative_controls_support_positive"] is False


def test_positive_and_saturation_claims_fail_closed_on_shuffled_controls() -> None:
    coverage = {
        "abstain": False,
        "score_only_headroom": 0.01,
        "alert_positive_count": 25,
        "selected_count": 10,
        "score_only_selected_count": 10,
        "selected_alert_precision": 0.99,
        "equal_count_score_only_comparison": True,
    }
    controls = [
        {"control": "locked_vasre", "selected_precision": 0.99, "selected_count": 10},
        {"control": "predictor_score_only", "selected_precision": 0.99, "selected_count": 10},
        {"control": "row_permuted_evidence", "selected_precision": 0.98, "selected_count": 10},
        {"control": "random_evidence", "selected_precision": 0.98, "selected_count": 10},
    ]
    stable = pd.DataFrame(
        {"validation_delta_vs_score_only": [0.0, 0.01], "abstain": [False, False]}
    )
    valid_shuffles = [
        {
            "variant": "validation_label_shuffle_control",
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.98,
            "delta_vs_score_only": 0.0,
            "abstain": False,
        },
        {
            "variant": "validation_weight_shuffle_control",
            "selected_count": 10,
            "score_only_selected_count": 10,
            "selected_alert_precision": 0.98,
            "delta_vs_score_only": 0.01,
            "abstain": False,
        },
    ]
    common = {
        "bootstrap_result": {"ci_high": 0.01, "valid_iterations": 500},
        "matched_result": {"matched_ci_high": 0.01, "matched_pairs": 25},
        "residual_result": {"risk_conditioned_tp_fp_gap": 0.01},
        "negative_controls": controls,
        "stability_rows": stable,
        "saturation_headroom": 0.05,
    }

    confirmed, diagnostics = _conservative_stress_outcome(
        coverage,
        shuffled_controls=valid_shuffles,
        **common,
    )
    assert confirmed == "confirmed_score_only_saturation"
    assert diagnostics["mandatory_shuffled_controls_consistent_with_no_gain"] is True

    unfair = [dict(row) for row in valid_shuffles]
    unfair[0]["selected_count"] = 9
    blocked, diagnostics = _conservative_stress_outcome(
        coverage,
        shuffled_controls=unfair,
        **common,
    )
    assert blocked == "low_score_only_headroom_unconfirmed"
    assert diagnostics["shuffled_control_counts_match_primary"] is False

    misleading_null = [dict(row) for row in valid_shuffles]
    misleading_null[1]["delta_vs_score_only"] = 0.03
    blocked, diagnostics = _conservative_stress_outcome(
        coverage,
        shuffled_controls=misleading_null,
        **common,
    )
    assert blocked == "low_score_only_headroom_unconfirmed"
    assert diagnostics["shuffled_controls_no_material_gain"] is False


@pytest.mark.parametrize(
    ("config_name", "output_name"),
    [
        ("transxion_v2.yaml", "transxion_fixture"),
        ("amlnet_v1.yaml", "amlnet_fixture"),
    ],
)
def test_quick_stress_fixtures_write_canonical_artifact_contract(
    tmp_path: Path,
    config_name: str,
    output_name: str,
) -> None:
    result = run_stress_test(
        PROJECT_ROOT / "configs/stress" / config_name,
        output_dir=tmp_path / output_name,
        quick_run=True,
        use_test_fixture=True,
        device="cpu",
    )
    output = Path(result["output_dir"])
    required = {
        "data_manifest.json",
        "split_integrity.csv",
        "protocol_ledger.json",
        "predictor_manifest.json",
        "predictive_metrics.csv",
        "predictor_explanation_sensitivity.csv",
        "predictor_attribution_sensitivity.csv",
        "predictor_contrastive_sensitivity.csv",
        "predictor_contrastive_sensitivity_provenance.json",
        "calibration_comparison.csv",
        "rule_registry.csv",
        "counterfactual_diagnostics.csv",
        "surrogate_provenance.json",
        "rule_audit.csv",
        "contrastive_meta_coefficients.csv",
        "contrastive_meta_provenance.json",
        "contrastive_meta_artifact.joblib",
        "locked_policies.json",
        "coverage_results.csv",
        "matched_risk_results.csv",
        "residual_evidence.csv",
        "paired_bootstrap.csv",
        "negative_controls.csv",
        "ablation_validation_results.csv",
        "ablation_validation_stability.csv",
        "ablation_rule_selection.csv",
        "ablation_rule_pool_provenance.json",
        "ablation_results.csv",
        "stress_summary.json",
        "stress_test_manifest.json",
        "stress_lineage.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert result["claim_eligible"] is False
    assert "generated_test_fixture_not_for_thesis_claims" in result["claim_blockers"]

    ledger = json.loads((output / "protocol_ledger.json").read_text(encoding="utf-8"))
    assert [event["phase"] for event in ledger] == list(PROTOCOL_PHASES)
    selection_event = next(
        event for event in ledger if event["phase"] == "test_selections_materialized_without_outcome_use"
    )
    assert selection_event["test_outcomes_used_for_selection"] is False
    outcome_event = next(
        event for event in ledger if event["phase"] == "test_outcomes_released_for_evaluation"
    )
    assert outcome_event["test_label_permutation_invariance_asserted"] is True

    split = pd.read_csv(output / "split_integrity.csv")
    assert split["both_classes"].all()
    assert split["chronologically_disjoint"].all()
    assert {"requested_proportion", "observed_proportion"}.issubset(split.columns)

    predictor = json.loads((output / "predictor_manifest.json").read_text(encoding="utf-8"))
    assert predictor["frozen_before_explanation"] is True
    assert predictor["selection_partition"] == "calibration_select"
    assert 0.0 <= float(predictor["threshold"]) <= 1.0
    assert predictor["training_provenance"]["validation_protocol"][
        "outer_roles_used_without_resplitting"
    ] is True
    portable = joblib.load(output / "frozen_reference_artifact.joblib")
    if portable["predictor"].backend == "torch":
        assert {
            parameter.device.type for parameter in portable["predictor"].model.parameters()
        } == {"cpu"}
        assert predictor["serializations"]["torch_tensor_device"] == "cpu"

    predictive = pd.read_csv(output / "predictive_metrics.csv")
    assert {"family", "selection_split_pr_auc", "pr_auc", "recall_at_1pct_fpr"}.issubset(
        predictive.columns
    )
    registry = pd.read_csv(output / "rule_registry.csv")
    score_only = registry.loc[registry["rule"] == "predictor_score_only"]
    assert len(score_only) == 1
    assert not bool(score_only.iloc[0]["requires_rule_evidence"])
    audit = pd.read_csv(output / "rule_audit.csv")
    assert "predictor_score_only" not in set(audit["rule"])
    assert not set(registry.loc[registry["tier"] == "C", "rule"]) & set(audit["rule"])

    ablation = pd.read_csv(output / "ablation_results.csv")
    assert {
        "full_primary_ab",
        "domain_tier_a_only",
        "counterfactual_only",
        "train_derived_tier_b_only",
        "method_unweighted",
        "method_audit_weighted_no_attribution",
        "method_fp_penalized_no_attribution",
        "method_attribution_gated",
        "method_contrastive_meta",
        "method_guarded_ensemble",
        "cart_tier_c_baseline",
        "attribution_top_k_baseline",
        "validation_label_shuffle_control",
        "validation_weight_shuffle_control",
    } <= set(ablation["variant"])
    assert not ablation["test_labels_used_for_selection"].astype(bool).any()
    ablation_validation = pd.read_csv(output / "ablation_validation_results.csv")
    assert ablation_validation["stability_guardrail_applied"].astype(bool).all()
    ablation_stability = pd.read_csv(output / "ablation_validation_stability.csv")
    assert set(ablation_stability["variant"]) == set(ablation["variant"])
    independent_variants = {
        "domain_tier_a_only",
        "counterfactual_only",
        "train_derived_tier_b_only",
    }
    independent_validation = ablation_validation.loc[
        ablation_validation["variant"].isin(independent_variants)
    ]
    assert set(independent_validation["variant"]) == independent_variants
    assert set(independent_validation["rule_pool_selection_scope"]) == {
        "independent_complete_pre_dedup_audit_pool"
    }
    assert independent_validation["rule_pool_dedup_recomputed"].astype(bool).all()
    assert independent_validation["rule_pool_weights_recomputed"].astype(bool).all()
    assert not independent_validation["rule_pool_primary_weights_filtered"].astype(bool).any()
    assert not independent_validation["rule_pool_test_labels_used"].astype(bool).any()
    independent_test = ablation.loc[ablation["variant"].isin(independent_variants)]
    assert set(independent_test["rule_pool_selection_scope"]) == {
        "independent_complete_pre_dedup_audit_pool"
    }
    rule_selection = pd.read_csv(output / "ablation_rule_selection.csv")
    assert set(rule_selection["variant"]) <= independent_variants
    assert not rule_selection["test_labels_used"].astype(bool).any()
    pool_provenance = json.loads(
        (output / "ablation_rule_pool_provenance.json").read_text(encoding="utf-8")
    )
    assert {row["variant"] for row in pool_provenance} == independent_variants
    assert all(row["primary_selected_weights_filtered"] is False for row in pool_provenance)
    assert all(row["test_labels_used"] is False for row in pool_provenance)

    sensitivity = pd.read_csv(output / "predictor_explanation_sensitivity.csv")
    assert {"xgboost", "lightgbm", "tabular_resnet_v2"} == set(sensitivity["family"])
    assert not sensitivity["test_labels_used_for_selection"].astype(bool).any()
    assert set(sensitivity["rule_set_origin"]) == {
        "reference_predictor_train_and_rule_audit"
    }
    assert sensitivity["diagnostic_only"].astype(bool).all()
    assert not sensitivity["policy_stability_guardrail_applied"].astype(bool).any()

    contrastive_provenance = json.loads(
        (output / "contrastive_meta_provenance.json").read_text(encoding="utf-8")
    )
    assert contrastive_provenance["fit_partition"] == "rule_audit"
    assert contrastive_provenance["policy_select_data_used_for_fit"] is False
    assert contrastive_provenance["test_data_used_for_fit"] is False
    assert contrastive_provenance["frozen_risk_feature"] == "frozen_calibrated_risk_z"
    coefficients = pd.read_csv(output / "contrastive_meta_coefficients.csv")
    assert "frozen_risk" in set(coefficients["feature_type"])
    assert "contrastive_meta" in set(
        pd.read_csv(output / "policy_candidates_validation.csv")["method"]
    )

    coverage = pd.read_csv(output / "coverage_results.csv")
    expected_quality = {
        "supported_alert_rate",
        "unsupported_alert_rate",
        "rule_sparsity_alerts",
        "contradiction_rate",
        "evidence_without_alert_rate",
        "positive_region_fidelity",
        "alert_conditional_rank_fidelity",
    }
    assert expected_quality.issubset(coverage.columns)
    assert {
        "requested_selected_count",
        "realized_selected_count",
        "score_only_selected_count",
        "score_only_requested_budget_selected_count",
        "realized_explanation_coverage",
        "score_only_realized_coverage",
        "equal_count_score_only_comparison",
        "selected_positive_count",
        "test_positive_count",
        "alert_positive_count",
        "unexplained_alert_count",
        "selected_recall_test_positives",
        "selected_recall_alert_positives",
    }.issubset(coverage.columns)
    assert coverage["equal_count_score_only_comparison"].astype(bool).all()
    assert (
        coverage["selected_count"].astype(int)
        == coverage["score_only_selected_count"].astype(int)
    ).all()
    np.testing.assert_allclose(
        coverage["abstention_rate"],
        1.0 - coverage["explanation_coverage"],
        equal_nan=True,
    )
    assert set(np.round(coverage["coverage_budget"], 2)) == {0.05, 0.10, 0.25, 0.50}

    summary = json.loads((output / "stress_summary.json").read_text(encoding="utf-8"))
    assert "contrastive_meta" in summary["evidence_methods"]
    assert "guarded_ensemble" in summary["evidence_methods"]
    assert summary["contrastive_meta"]["fit_partition"] == "rule_audit"
    assert summary["contrastive_meta"]["test_data_used_for_fit"] is False
    assert summary["guarded_ensemble"]["components"] == [
        "fp_penalized",
        "attribution_gated",
        "contrastive_meta",
    ]
    assert {
        row["variant"] for row in summary["independent_ablation_rule_pools"]
    } == independent_variants
    assert all(
        row["selection_scope"] == "independent_complete_pre_dedup_audit_pool"
        for row in summary["independent_ablation_rule_pools"]
    )
    assert summary["attribution_top_k_baseline"]["formula"] == (
        "sum_positive_among_top5_absolute_native_attributions"
    )
    assert summary["rule_audit_scope"]["scope"] == "frozen_predictor_alerts_only"
    assert summary["rule_audit_scope"]["non_alert_rows_used_for_rule_selection_or_weighting"] is False
    assert summary["fixed_budget_fairness"] == {
        "requested_budget_reported_separately": True,
        "core_score_only_comparator": "matched_realized_selected_count",
        "unsupported_alerts_used_to_fill_budget": False,
        "mandatory_shuffled_controls": [
            "validation_label_shuffle_control",
            "validation_weight_shuffle_control",
        ],
        "claim_gate_requires_equal_realized_counts": True,
    }
    assert summary["outcome_decision_thresholds"] == {
        "saturation_headroom": 0.05,
        "saturation_precision_margin": 0.02,
        "saturation_evidence_margin": 0.05,
        "minimum_positive_stability_fraction": 0.8,
        "minimum_saturation_alert_positives": 20,
        "minimum_saturation_matched_pairs": 20,
        "minimum_saturation_bootstrap_replicates": 200,
    }
    outcomes = pd.read_csv(output / "stress_outcomes.csv")
    assert set(outcomes["saturation_headroom_threshold"]) == {0.05}
    assert set(outcomes["saturation_precision_margin"]) == {0.02}
    assert set(outcomes["saturation_evidence_margin"]) == {0.05}
    assert set(outcomes["minimum_positive_stability_fraction"]) == {0.8}
    assert set(outcomes["minimum_saturation_alert_positives"]) == {20}
    assert set(outcomes["minimum_saturation_matched_pairs"]) == {20}
    assert set(outcomes["minimum_saturation_bootstrap_replicates"]) == {200}
    assert outcomes["core_equal_count_score_only_comparison"].astype(bool).all()
    assert {
        "shuffled_control_comparison_valid",
        "shuffled_control_counts_match_primary",
        "shuffled_controls_below_locked_primary",
        "shuffled_controls_no_material_gain",
    }.issubset(outcomes.columns)
