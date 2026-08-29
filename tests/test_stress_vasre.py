import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.stress_testing.matched_risk import (
    frozen_risk_bin_edges,
    match_selected_to_controls,
    paired_selection_bootstrap,
    residual_tp_fp_by_risk_bin,
    summarize_matched_risk,
)
from src.stress_testing.rule_audit import (
    audit_rule_candidates,
    build_evidence_scores,
    compute_rule_weights,
    deduplicate_audited_rules,
)
from src.stress_testing.selective_policy import (
    apply_locked_policy,
    budget_count,
    evaluate_locked_policy,
    fixed_budget_mask,
    lock_selective_policies,
)


def test_rule_audit_and_deduplication_are_validation_only():
    labels_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_audit = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    strong = np.array([0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.8])
    weak = np.array([0.8, 0.2, 0.7, 0.1, 0.3, 0.2, 0.4, 0.3])
    train_truth = pd.DataFrame({"strong": strong, "duplicate": strong, "weak": weak})
    audit_truth = train_truth.copy()
    registry = pd.DataFrame(
        {
            "rule": ["strong", "duplicate", "weak"],
            "tier": ["A", "B", "C"],
            "features": [["amount"], ["amount"], ["hour"]],
        }
    )
    attribution = pd.DataFrame(np.ones((8, 3)), columns=train_truth.columns)
    audit = audit_rule_candidates(
        train_truth,
        audit_truth,
        labels_train,
        labels_audit,
        train_alert_mask=np.ones(8, dtype=bool),
        audit_alert_mask=np.ones(8, dtype=bool),
        registry=registry,
        attribution_hit=attribution,
        attribution_share=attribution * 0.5,
        criteria={
            "min_active_count": 2,
            "min_coverage": 0.1,
            "max_coverage": 0.9,
            "min_lift": 1.1,
            "min_tp_fp_gap": 0.1,
            "max_fp_activation": 0.8,
        },
    )
    completed, overlaps = deduplicate_audited_rules(
        audit,
        audit_truth,
        audit_alert_mask=np.ones(8, dtype=bool),
        max_jaccard=0.8,
        max_rules=3,
    )
    assert completed["selected"].sum() == 1
    assert set(completed.query("selected")["rule"]) <= {"strong", "duplicate"}
    assert overlaps["jaccard"].max() == 1.0
    weights = compute_rule_weights(completed)
    evidence = build_evidence_scores(audit_truth, weights, method="fp_penalized")
    assert evidence.shape == (8,)
    assert np.all((0.0 <= evidence) & (evidence <= 1.0))


def test_non_alert_tn_fn_changes_cannot_affect_audit_selection_or_weights():
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    alerts = np.array([False, False, False, False, True, True, True, True, True, True, True, True])
    truth = pd.DataFrame(
        {
            "strong": [0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.8],
            "mixed": [0.8, 0.1, 0.2, 0.9, 0.8, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.2],
        }
    )
    attribution = pd.DataFrame(np.ones_like(truth, dtype=float), columns=truth.columns)
    criteria = {
        "min_active_count": 2,
        "min_coverage": 0.1,
        "max_coverage": 0.9,
        "min_lift": 1.0,
        "min_tp_fp_gap": 0.0,
        "max_fp_activation": 0.9,
    }

    def run(
        candidate_truth: pd.DataFrame,
        candidate_labels: np.ndarray,
        candidate_attribution: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        audited = audit_rule_candidates(
            candidate_truth,
            candidate_truth.copy(),
            candidate_labels,
            candidate_labels.copy(),
            train_alert_mask=alerts,
            audit_alert_mask=alerts,
            attribution_hit=candidate_attribution,
            attribution_share=candidate_attribution * 0.5,
            criteria=criteria,
        )
        completed, _ = deduplicate_audited_rules(
            audited,
            candidate_truth,
            audit_alert_mask=alerts,
            max_jaccard=0.99,
            max_rules=2,
        )
        return completed.reset_index(drop=True), compute_rule_weights(completed)

    baseline_audit, baseline_weights = run(truth, labels, attribution)
    changed_truth = truth.copy()
    changed_truth.loc[~alerts, :] = 1.0
    changed_labels = labels.copy()
    changed_labels[~alerts] = 1 - changed_labels[~alerts]
    changed_attribution = attribution.copy()
    changed_attribution.loc[~alerts, :] = 0.0
    changed_audit, changed_weights = run(
        changed_truth, changed_labels, changed_attribution
    )

    decision_columns = [
        "rule",
        "audit_pass",
        "audit_failures",
        "audit_alert_score",
        "audit_alert_precision",
        "audit_alert_lift",
        "audit_alert_tp_fp_gap",
        "audit_alert_fp_activation",
        "audit_alert_attribution_hit_rate",
        "audit_alert_attribution_share",
        "selection_status",
        "selected",
    ]
    assert_frame_equal(
        baseline_audit[decision_columns],
        changed_audit[decision_columns],
    )
    assert_frame_equal(baseline_weights, changed_weights)
    assert not np.allclose(
        baseline_audit["audit_population_mean_truth"],
        changed_audit["audit_population_mean_truth"],
    )


def test_alert_masks_are_required_aligned_and_binary():
    truth = pd.DataFrame({"rule": [0.1, 0.2, 0.8, 0.9]})
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="train_alert_mask.*align"):
        audit_rule_candidates(
            truth,
            truth,
            labels,
            labels,
            train_alert_mask=np.ones(3, dtype=bool),
            audit_alert_mask=np.ones(4, dtype=bool),
        )
    with pytest.raises(ValueError, match="audit_alert_mask.*boolean"):
        audit_rule_candidates(
            truth,
            truth,
            labels,
            labels,
            train_alert_mask=np.ones(4, dtype=bool),
            audit_alert_mask=np.array([0, 1, 2, 1]),
        )
    with pytest.raises(ValueError, match="audit_alert_mask.*align"):
        deduplicate_audited_rules(
            pd.DataFrame(
                {
                    "rule": ["rule"],
                    "audit_pass": [False],
                    "audit_alert_score": [0.0],
                }
            ),
            truth,
            audit_alert_mask=np.ones(3, dtype=bool),
        )


@pytest.mark.parametrize(
    ("alerts", "failure"),
    [
        (np.zeros(4, dtype=bool), "alert_empty"),
        (np.array([True, True, False, False]), "alert_one_class"),
    ],
)
def test_empty_or_one_class_alert_subsets_fail_closed_without_metric_errors(
    alerts: np.ndarray,
    failure: str,
):
    truth = pd.DataFrame({"rule": [0.1, 0.8, 0.2, 0.9]})
    labels = np.array([0, 0, 1, 1])
    audited = audit_rule_candidates(
        truth,
        truth,
        labels,
        labels,
        train_alert_mask=alerts,
        audit_alert_mask=alerts,
        criteria={"min_active_count": 1},
    )
    assert not audited.loc[0, "audit_pass"]
    assert f"train_{failure}" in audited.loc[0, "audit_failures"]
    assert f"audit_{failure}" in audited.loc[0, "audit_failures"]
    assert np.isfinite(audited.loc[0, "audit_alert_score"])


def test_fixed_budget_and_locked_policy_do_not_use_test_labels():
    risk = np.linspace(0.51, 0.99, 10)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    evidence = np.array([1.0, 0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    alerts = risk >= 0.5
    selected = fixed_budget_mask(evidence, risk, alerts, 0.4)
    assert selected.sum() == budget_count(10, 0.4) == 4

    policies, candidates = lock_selective_policies(
        labels,
        risk,
        0.5,
        {"audit_weighted": evidence},
        dataset="fixture",
        predictor="tree",
        coverages=[0.4],
    )
    assert not policies[0].abstain
    assert candidates.loc[0, "delta_vs_score_only"] > 0.0
    before = apply_locked_policy(policies[0], risk, 0.5, {"audit_weighted": evidence})
    coverage_result, selected_on_labels, _ = evaluate_locked_policy(
        policies[0], labels, risk, 0.5, {"audit_weighted": evidence}
    )
    np.testing.assert_array_equal(before, selected_on_labels)
    assert coverage_result["selected_positive_count"] == 4
    assert coverage_result["test_positive_count"] == 4
    assert coverage_result["alert_positive_count"] == 4
    assert coverage_result["unexplained_alert_count"] == 6
    assert coverage_result["explanation_coverage"] == 0.4
    assert coverage_result["abstention_rate"] == 0.6
    assert coverage_result["selected_recall_test_positives"] == 1.0
    assert coverage_result["selected_recall_alert_positives"] == 1.0
    # Changing test outcomes cannot change ranking, selected count or row IDs.
    after_result, after, _ = evaluate_locked_policy(
        policies[0], 1 - labels, risk, 0.5, {"audit_weighted": evidence}
    )
    np.testing.assert_array_equal(before, after)
    assert after_result["selected_count"] == 4


def test_fixed_budget_never_fills_with_unsupported_alerts():
    risk = np.linspace(0.60, 0.99, 10)
    evidence = np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    alerts = np.ones(10, dtype=bool)
    selected = fixed_budget_mask(
        evidence,
        risk,
        alerts,
        0.50,
        eligible_mask=evidence > 0.0,
    )
    # The requested budget is five alerts, but only two carry active support.
    # The selector must underfill instead of relabelling unsupported rows as
    # explained.
    assert budget_count(int(alerts.sum()), 0.50) == 5
    assert selected.sum() == 2
    assert np.all(evidence[selected] > 0.0)


def test_score_only_comparator_matches_realized_count_under_test_support_shift():
    risk = np.linspace(0.60, 0.99, 10)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    evidence = np.array([0.90, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    policy = lock_selective_policies(
        np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        risk,
        0.5,
        {"audit_weighted": np.linspace(1.0, 0.1, 10)},
        dataset="fixture",
        predictor="tree",
        coverages=[0.5],
    )[0][0]
    assert not policy.abstain

    result, selected, score_only = evaluate_locked_policy(
        policy,
        labels,
        risk,
        0.5,
        {"audit_weighted": evidence},
    )

    assert result["requested_selected_count"] == 5
    assert selected.sum() == result["realized_selected_count"] == 2
    assert result["support_shift_underfill"]
    assert score_only.sum() == result["score_only_selected_count"] == 2
    assert result["score_only_requested_budget_selected_count"] == 5
    assert result["equal_count_score_only_comparison"]
    assert result["realized_explanation_coverage"] == pytest.approx(0.2)
    assert result["score_only_realized_coverage"] == pytest.approx(0.2)
    expected_delta = labels[selected].mean() - labels[score_only].mean()
    assert result["delta_vs_score_only"] == pytest.approx(expected_delta)


def test_policy_abstains_when_evidence_cannot_beat_score_only():
    labels = np.array([0, 0, 0, 1, 1, 1])
    risk = np.array([0.55, 0.60, 0.65, 0.80, 0.90, 0.95])
    policies, _ = lock_selective_policies(
        labels,
        risk,
        0.5,
        {"audit_weighted": risk.copy()},
        dataset="fixture",
        predictor="tree",
        coverages=[0.5],
    )
    assert policies[0].abstain
    assert policies[0].reason == "no_method_beats_score_only_guardrail"
    result, selected, _ = evaluate_locked_policy(
        policies[0], labels, risk, 0.5, {"audit_weighted": risk.copy()}
    )
    assert not selected.any()
    assert result["explanation_coverage"] == 0.0
    assert result["abstention_rate"] == 1.0
    assert result["selected_positive_count"] == 0
    assert result["selected_recall_test_positives"] == 0.0


def test_matching_is_label_invariant_and_bootstrap_is_deterministic():
    risk = np.array([0.51, 0.52, 0.61, 0.62, 0.71, 0.72, 0.81, 0.82])
    evidence = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    selected = np.array([True, False, True, False, True, False, True, False])
    alerts = np.ones(8, dtype=bool)
    labels = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    edges = frozen_risk_bin_edges(risk, n_bins=4)
    pairs = match_selected_to_controls(risk, evidence, selected, alerts, edges)
    summary_a = summarize_matched_risk(pairs, labels, n_bootstrap=100, seed=7)
    summary_b = summarize_matched_risk(pairs, 1 - labels, n_bootstrap=100, seed=7)
    assert pairs["matched"].all()
    assert summary_a["matched_pairs"] == summary_b["matched_pairs"] == 4
    # Pair construction itself has no label input.
    assert pairs[["selected_index", "control_index"]].to_dict("records") == [
        {"selected_index": 0, "control_index": 1},
        {"selected_index": 2, "control_index": 3},
        {"selected_index": 4, "control_index": 5},
        {"selected_index": 6, "control_index": 7},
    ]
    baseline = np.array([False, True, False, True, False, True, False, True])
    first = paired_selection_bootstrap(
        labels, selected, baseline, n_bootstrap=200, seed=19
    )
    second = paired_selection_bootstrap(
        labels, selected, baseline, n_bootstrap=200, seed=19
    )
    assert first == second


def test_residual_evidence_conditions_on_validation_fitted_risk_bins():
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    risk = np.array([0.51, 0.52, 0.61, 0.62, 0.71, 0.72, 0.81, 0.82])
    evidence = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    edges = frozen_risk_bin_edges(risk, n_bins=4)
    table, summary = residual_tp_fp_by_risk_bin(
        labels, risk, evidence, np.ones(8, dtype=bool), edges
    )
    assert len(table) == 4
    assert summary["comparable_bins"] == 4
    assert summary["risk_conditioned_tp_fp_gap"] > 0.0
