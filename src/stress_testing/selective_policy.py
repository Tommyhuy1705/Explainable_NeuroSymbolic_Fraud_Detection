"""Coverage-constrained VASRE policy selection and abstention.

Policy choice is performed only on the reserved policy-selection partition.
Applying a locked policy to test data uses scores and deterministic ranks but
never test labels.  A policy that does not beat the predictor-score-only
guardrail on validation is represented explicitly as an abstention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class LockedSelectivePolicy:
    """A validation-selected explanation policy safe to apply to test scores."""

    dataset: str
    predictor: str
    coverage: float
    method: str | None
    abstain: bool
    reason: str
    validation_alert_count: int
    validation_selected_count: int
    validation_all_alert_precision: float
    validation_score_only_precision: float
    validation_selected_precision: float
    validation_delta_vs_score_only: float
    validation_evidence_cutoff: float
    minimum_score_only_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "LockedSelectivePolicy":
        return cls(**dict(values))


def _binary_labels(values: np.ndarray | Sequence[int]) -> np.ndarray:
    labels = np.asarray(values, dtype=int)
    if labels.ndim != 1 or not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("Labels must be a one-dimensional binary array")
    return labels


def budget_count(alert_count: int, coverage: float) -> int:
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    if int(alert_count) <= 0:
        return 0
    return min(int(alert_count), max(1, int(np.ceil(float(coverage) * int(alert_count)))))


def fixed_budget_mask(
    ranking_score: np.ndarray | Sequence[float],
    predictor_score: np.ndarray | Sequence[float],
    alert_mask: np.ndarray | Sequence[bool],
    coverage: float,
    *,
    eligible_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Select up to the fixed budget from eligible alerts with stable ties."""

    alerts = np.asarray(alert_mask, dtype=bool)
    return fixed_count_mask(
        ranking_score,
        predictor_score,
        alerts,
        budget_count(int(alerts.sum()), coverage),
        eligible_mask=eligible_mask,
    )


def fixed_count_mask(
    ranking_score: np.ndarray | Sequence[float],
    predictor_score: np.ndarray | Sequence[float],
    alert_mask: np.ndarray | Sequence[bool],
    selected_count: int,
    *,
    eligible_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Select up to an explicit count from eligible alerts with stable ties.

    This is the count-matched primitive used for fair score-only comparisons
    when rule support shifts after policy lock. It never fills a shortfall with
    ineligible rows; callers can therefore detect an underfilled rule budget
    and either compare against the same realized count or fail closed.
    """

    ranking = np.asarray(ranking_score, dtype=float)
    risk = np.asarray(predictor_score, dtype=float)
    alerts = np.asarray(alert_mask, dtype=bool)
    eligible = (
        np.ones(len(alerts), dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    if not (len(ranking) == len(risk) == len(alerts) == len(eligible)):
        raise ValueError("Ranking, predictor and alert arrays must align")
    if not np.isfinite(ranking).all() or not np.isfinite(risk).all():
        raise ValueError("Selection scores must be finite")
    requested_count = int(selected_count)
    if requested_count < 0:
        raise ValueError("selected_count must be non-negative")
    alert_indices = np.flatnonzero(alerts & eligible)
    selected = np.zeros(len(alerts), dtype=bool)
    count = min(requested_count, len(alert_indices))
    if count == 0:
        return selected
    # np.lexsort uses the last key as primary.  Lower negative values represent
    # higher evidence/risk, and original row position is the final stable key.
    order = np.lexsort(
        (alert_indices, -risk[alert_indices], -ranking[alert_indices])
    )
    selected[alert_indices[order[:count]]] = True
    return selected


def _precision(labels: np.ndarray, mask: np.ndarray) -> float:
    return float(labels[mask].mean()) if mask.any() else float("nan")


def _safe_fidelity(alerts: np.ndarray, risk: np.ndarray, evidence: np.ndarray) -> tuple[float, float]:
    if np.unique(alerts.astype(int)).size == 2:
        positive_region = float(roc_auc_score(alerts.astype(int), evidence))
    else:
        positive_region = float("nan")
    if alerts.sum() >= 3 and np.unique(evidence[alerts]).size > 1:
        correlation = float(spearmanr(risk[alerts], evidence[alerts]).statistic)
    else:
        correlation = float("nan")
    return positive_region, correlation


def _candidate_row(
    labels: np.ndarray,
    probabilities: np.ndarray,
    alerts: np.ndarray,
    evidence: np.ndarray,
    coverage: float,
    method: str,
) -> dict[str, Any]:
    active_evidence = evidence > 0.0
    selected = fixed_budget_mask(
        evidence,
        probabilities,
        alerts,
        coverage,
        eligible_mask=active_evidence,
    )
    selected_precision = _precision(labels, selected)
    requested_count = budget_count(int(alerts.sum()), float(coverage))
    score_only = fixed_count_mask(
        probabilities,
        probabilities,
        alerts,
        int(selected.sum()),
    )
    requested_score_only = fixed_count_mask(
        probabilities,
        probabilities,
        alerts,
        requested_count,
    )
    if int(score_only.sum()) != int(selected.sum()):  # pragma: no cover - invariant
        raise AssertionError(
            "Score-only comparator must match the locked policy's realized selected count"
        )
    score_only_precision = _precision(labels, score_only)
    requested_score_only_precision = _precision(labels, requested_score_only)
    all_alert_precision = _precision(labels, alerts)
    support_count = int((active_evidence & alerts).sum())
    tp = alerts & (labels == 1)
    fp = alerts & (labels == 0)
    evidence_tp = float(evidence[tp].mean()) if tp.any() else float("nan")
    evidence_fp = float(evidence[fp].mean()) if fp.any() else float("nan")
    positive_region, alert_correlation = _safe_fidelity(alerts, probabilities, evidence)
    selected_values = evidence[selected]
    return {
        "method": method,
        "coverage": float(coverage),
        "alert_count": int(alerts.sum()),
        "selected_count": int(selected.sum()),
        "requested_selected_count": requested_count,
        "realized_selected_count": int(selected.sum()),
        "realized_alert_coverage": (
            float(selected.sum() / alerts.sum()) if alerts.any() else float("nan")
        ),
        "support_shift_underfill": int(selected.sum()) < requested_count,
        "supported_alert_count": support_count,
        "supported_alert_fraction": float(support_count / alerts.sum()) if alerts.any() else float("nan"),
        "all_alert_precision": all_alert_precision,
        "selected_precision": selected_precision,
        "precision_gain_vs_all_alerts": selected_precision - all_alert_precision,
        "score_only_selected_count": int(score_only.sum()),
        "score_only_precision": score_only_precision,
        "score_only_requested_budget_selected_count": int(requested_score_only.sum()),
        "score_only_requested_budget_precision": requested_score_only_precision,
        "equal_count_score_only_comparison": int(score_only.sum()) == int(selected.sum()),
        "delta_vs_score_only": selected_precision - score_only_precision,
        "validation_headroom_after_score_only": 1.0 - score_only_precision,
        "evidence_tp_mean": evidence_tp,
        "evidence_fp_mean": evidence_fp,
        "evidence_tp_fp_gap": evidence_tp - evidence_fp,
        "positive_region_fidelity": positive_region,
        "alert_conditional_rank_fidelity": alert_correlation,
        "evidence_cutoff": float(selected_values.min()) if len(selected_values) else float("nan"),
    }


def lock_selective_policies(
    y_policy_validation: np.ndarray | Sequence[int],
    validation_probabilities: np.ndarray | Sequence[float],
    decision_threshold: float,
    evidence_by_method: Mapping[str, np.ndarray | Sequence[float]],
    *,
    dataset: str,
    predictor: str,
    coverages: Sequence[float] = (0.05, 0.10, 0.25, 0.50),
    minimum_score_only_delta: float = 0.0,
    minimum_supported_alert_fraction: float = 0.01,
    minimum_evidence_tp_fp_gap: float = 0.0,
    minimum_positive_region_fidelity: float | None = None,
    require_full_budget_support: bool = True,
    method_precedence: Sequence[str] = (
        "guarded_ensemble",
        "contrastive_meta",
        "attribution_gated",
        "fp_penalized",
        "audit_weighted",
        "unweighted",
    ),
) -> tuple[list[LockedSelectivePolicy], pd.DataFrame]:
    """Choose one method per coverage or abstain when no guardrail is met."""

    labels = _binary_labels(y_policy_validation)
    probabilities = np.asarray(validation_probabilities, dtype=float)
    if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Validation labels and probabilities must align and be finite")
    alerts = probabilities >= float(decision_threshold)
    all_alert_precision = _precision(labels, alerts)
    policies: list[LockedSelectivePolicy] = []
    candidate_rows: list[dict[str, Any]] = []
    precedence = {name: index for index, name in enumerate(method_precedence)}

    for coverage in coverages:
        score_only = fixed_budget_mask(probabilities, probabilities, alerts, float(coverage))
        score_only_precision = _precision(labels, score_only)
        rows: list[dict[str, Any]] = []
        for method, raw_values in evidence_by_method.items():
            evidence = np.asarray(raw_values, dtype=float)
            if len(evidence) != len(labels) or not np.isfinite(evidence).all():
                raise ValueError(f"Evidence method {method!r} is misaligned or non-finite")
            row = _candidate_row(
                labels,
                probabilities,
                alerts,
                evidence,
                float(coverage),
                str(method),
            )
            row["eligible"] = bool(
                alerts.any()
                and np.isfinite(float(row["selected_precision"]))
                and float(row["delta_vs_score_only"]) > float(minimum_score_only_delta)
                and float(row["supported_alert_fraction"]) >= float(minimum_supported_alert_fraction)
                and (
                    not require_full_budget_support
                    or int(row["supported_alert_count"]) >= int(row["requested_selected_count"])
                )
                and np.isfinite(float(row["evidence_tp_fp_gap"]))
                and float(row["evidence_tp_fp_gap"]) > float(minimum_evidence_tp_fp_gap)
                and (
                    minimum_positive_region_fidelity is None
                    or (
                        np.isfinite(float(row["positive_region_fidelity"]))
                        and float(row["positive_region_fidelity"])
                        >= float(minimum_positive_region_fidelity)
                    )
                )
                and np.unique(evidence[alerts]).size > 1
            )
            reasons: list[str] = []
            if not alerts.any():
                reasons.append("no_validation_alerts")
            if np.unique(evidence[alerts]).size <= 1 if alerts.any() else True:
                reasons.append("constant_evidence")
            if float(row["supported_alert_fraction"]) < float(minimum_supported_alert_fraction):
                reasons.append("insufficient_rule_support")
            if require_full_budget_support and int(row["supported_alert_count"]) < int(
                row["requested_selected_count"]
            ):
                reasons.append("insufficient_active_rules_for_fixed_budget")
            if not np.isfinite(float(row["evidence_tp_fp_gap"])) or float(
                row["evidence_tp_fp_gap"]
            ) <= float(minimum_evidence_tp_fp_gap):
                reasons.append("tp_fp_evidence_guardrail_not_met")
            if minimum_positive_region_fidelity is not None and (
                not np.isfinite(float(row["positive_region_fidelity"]))
                or float(row["positive_region_fidelity"])
                < float(minimum_positive_region_fidelity)
            ):
                reasons.append("fidelity_guardrail_not_met")
            if not np.isfinite(float(row["delta_vs_score_only"])) or float(
                row["delta_vs_score_only"]
            ) <= float(minimum_score_only_delta):
                reasons.append("score_only_guardrail_not_met")
            row["ineligible_reasons"] = ";".join(reasons)
            rows.append(row)
            candidate_rows.append(row)

        eligible = [row for row in rows if bool(row["eligible"])]
        if eligible:
            best = sorted(
                eligible,
                key=lambda row: (
                    -float(row["delta_vs_score_only"]),
                    -float(row["selected_precision"]),
                    -float(row["evidence_tp_fp_gap"]),
                    precedence.get(str(row["method"]), len(precedence)),
                    str(row["method"]),
                ),
            )[0]
            policy = LockedSelectivePolicy(
                dataset=str(dataset),
                predictor=str(predictor),
                coverage=float(coverage),
                method=str(best["method"]),
                abstain=False,
                reason="validation_guardrails_passed",
                validation_alert_count=int(best["alert_count"]),
                validation_selected_count=int(best["selected_count"]),
                validation_all_alert_precision=float(best["all_alert_precision"]),
                validation_score_only_precision=float(best["score_only_precision"]),
                validation_selected_precision=float(best["selected_precision"]),
                validation_delta_vs_score_only=float(best["delta_vs_score_only"]),
                validation_evidence_cutoff=float(best["evidence_cutoff"]),
                minimum_score_only_delta=float(minimum_score_only_delta),
            )
        else:
            reason = "no_validation_alerts" if not alerts.any() else "no_method_beats_score_only_guardrail"
            policy = LockedSelectivePolicy(
                dataset=str(dataset),
                predictor=str(predictor),
                coverage=float(coverage),
                method=None,
                abstain=True,
                reason=reason,
                validation_alert_count=int(alerts.sum()),
                validation_selected_count=0,
                validation_all_alert_precision=all_alert_precision,
                validation_score_only_precision=score_only_precision,
                validation_selected_precision=float("nan"),
                validation_delta_vs_score_only=float("nan"),
                validation_evidence_cutoff=float("nan"),
                minimum_score_only_delta=float(minimum_score_only_delta),
            )
        policies.append(policy)

    candidates = pd.DataFrame(candidate_rows)
    return policies, candidates


def apply_locked_policy(
    policy: LockedSelectivePolicy,
    probabilities: np.ndarray | Sequence[float],
    decision_threshold: float,
    evidence_by_method: Mapping[str, np.ndarray | Sequence[float]],
) -> np.ndarray:
    """Apply the locked method and fixed coverage without consulting labels."""

    risk = np.asarray(probabilities, dtype=float)
    alerts = risk >= float(decision_threshold)
    if policy.abstain:
        return np.zeros(len(risk), dtype=bool)
    if policy.method not in evidence_by_method:
        raise KeyError(f"Locked evidence method is unavailable: {policy.method}")
    evidence = np.asarray(evidence_by_method[str(policy.method)], dtype=float)
    return fixed_budget_mask(
        evidence,
        risk,
        alerts,
        policy.coverage,
        eligible_mask=evidence > 0.0,
    )


def evaluate_locked_policy(
    policy: LockedSelectivePolicy,
    y_test: np.ndarray | Sequence[int],
    test_probabilities: np.ndarray | Sequence[float],
    decision_threshold: float,
    evidence_by_method: Mapping[str, np.ndarray | Sequence[float]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Report test outcomes after policy lock; this function never changes policy."""

    labels = _binary_labels(y_test)
    probabilities = np.asarray(test_probabilities, dtype=float)
    if len(labels) != len(probabilities):
        raise ValueError("Test labels and probabilities must align")
    alerts = probabilities >= float(decision_threshold)
    selected = apply_locked_policy(policy, probabilities, decision_threshold, evidence_by_method)
    requested_count = budget_count(int(alerts.sum()), policy.coverage)
    # The core score-only comparator is always selected at the VASRE policy's
    # realized count.  If rule support shifts and the locked policy underfills,
    # comparing with score-only at the larger requested count would confound
    # ranking quality with sample size.  The requested-budget score-only result
    # is retained under an explicit diagnostic name and never feeds the delta.
    score_only = fixed_count_mask(
        probabilities,
        probabilities,
        alerts,
        int(selected.sum()),
    )
    requested_score_only = fixed_count_mask(
        probabilities,
        probabilities,
        alerts,
        requested_count,
    )
    if int(score_only.sum()) != int(selected.sum()):  # pragma: no cover - invariant
        raise AssertionError(
            "Score-only comparator must match the locked policy's realized selected count"
        )
    all_precision = _precision(labels, alerts)
    score_precision = _precision(labels, score_only)
    requested_score_precision = _precision(labels, requested_score_only)
    selected_precision = _precision(labels, selected)
    test_positive_count = int((labels == 1).sum())
    alert_positive_count = int((alerts & (labels == 1)).sum())
    selected_positive_count = int((selected & (labels == 1)).sum())
    score_only_positive_count = int((score_only & (labels == 1)).sum())
    requested_score_only_positive_count = int(
        (requested_score_only & (labels == 1)).sum()
    )
    unexplained_alert_count = int((alerts & ~selected).sum())
    evidence = (
        np.asarray(evidence_by_method[str(policy.method)], dtype=float)
        if not policy.abstain and policy.method is not None
        else np.zeros(len(labels), dtype=float)
    )
    positive_region, alert_correlation = _safe_fidelity(alerts, probabilities, evidence)
    tp = alerts & (labels == 1)
    fp = alerts & (labels == 0)
    evidence_tp = float(evidence[tp].mean()) if tp.any() else float("nan")
    evidence_fp = float(evidence[fp].mean()) if fp.any() else float("nan")
    result = {
        "dataset": policy.dataset,
        "predictor": policy.predictor,
        "coverage_budget": policy.coverage,
        "method": policy.method,
        "abstain": policy.abstain,
        "abstention_reason": policy.reason,
        "test_rows": int(len(labels)),
        "alert_count": int(alerts.sum()),
        "requested_selected_count": requested_count,
        "selected_count": int(selected.sum()),
        "realized_selected_count": int(selected.sum()),
        "support_shift_underfill": int(selected.sum()) < requested_count,
        "selected_positive_count": selected_positive_count,
        "score_only_selected_count": int(score_only.sum()),
        "score_only_positive_count": score_only_positive_count,
        "score_only_requested_budget_selected_count": int(requested_score_only.sum()),
        "score_only_requested_budget_positive_count": requested_score_only_positive_count,
        "test_positive_count": test_positive_count,
        "alert_positive_count": alert_positive_count,
        "unexplained_alert_count": unexplained_alert_count,
        "explanation_coverage": float(selected.sum() / alerts.sum()) if alerts.any() else float("nan"),
        "requested_explanation_coverage": float(policy.coverage),
        "realized_explanation_coverage": (
            float(selected.sum() / alerts.sum()) if alerts.any() else float("nan")
        ),
        "score_only_realized_coverage": (
            float(score_only.sum() / alerts.sum()) if alerts.any() else float("nan")
        ),
        "score_only_requested_budget_coverage": (
            float(requested_score_only.sum() / alerts.sum())
            if alerts.any()
            else float("nan")
        ),
        "equal_count_score_only_comparison": (
            int(score_only.sum()) == int(selected.sum())
        ),
        # Row-level abstention is the complement of explanation coverage over
        # frozen-predictor alerts.  ``abstain`` above remains the separate
        # policy-level guardrail decision.
        "abstention_rate": (
            float(unexplained_alert_count / alerts.sum()) if alerts.any() else float("nan")
        ),
        "selected_recall_test_positives": (
            float(selected_positive_count / test_positive_count)
            if test_positive_count
            else float("nan")
        ),
        "selected_recall_alert_positives": (
            float(selected_positive_count / alert_positive_count)
            if alert_positive_count
            else float("nan")
        ),
        "score_only_recall_test_positives": (
            float(score_only_positive_count / test_positive_count)
            if test_positive_count
            else float("nan")
        ),
        "score_only_recall_alert_positives": (
            float(score_only_positive_count / alert_positive_count)
            if alert_positive_count
            else float("nan")
        ),
        "score_only_requested_budget_recall_test_positives": (
            float(requested_score_only_positive_count / test_positive_count)
            if test_positive_count
            else float("nan")
        ),
        "score_only_requested_budget_recall_alert_positives": (
            float(requested_score_only_positive_count / alert_positive_count)
            if alert_positive_count
            else float("nan")
        ),
        "all_alert_precision": all_precision,
        "selected_alert_precision": selected_precision,
        "precision_gain_vs_all_alerts": (
            selected_precision - all_precision if np.isfinite(selected_precision) else float("nan")
        ),
        "alert_triage_lift": (
            selected_precision / all_precision
            if np.isfinite(selected_precision) and np.isfinite(all_precision) and all_precision > 0.0
            else float("nan")
        ),
        "score_only_precision": score_precision,
        "score_only_requested_budget_precision": requested_score_precision,
        "delta_vs_score_only": (
            selected_precision - score_precision
            if np.isfinite(selected_precision) and np.isfinite(score_precision)
            else float("nan")
        ),
        "score_only_headroom": (
            1.0 - score_precision if np.isfinite(score_precision) else float("nan")
        ),
        "score_only_requested_budget_headroom": (
            1.0 - requested_score_precision
            if np.isfinite(requested_score_precision)
            else float("nan")
        ),
        "positive_region_fidelity": positive_region,
        "alert_conditional_rank_fidelity": alert_correlation,
        "residual_tp_fp_unadjusted": evidence_tp - evidence_fp,
        "evidence_tp_mean": evidence_tp,
        "evidence_fp_mean": evidence_fp,
    }
    return result, selected, score_only


def validation_policy_stability(
    y_policy_validation: np.ndarray | Sequence[int],
    validation_probabilities: np.ndarray | Sequence[float],
    decision_threshold: float,
    evidence_by_method: Mapping[str, np.ndarray | Sequence[float]],
    *,
    dataset: str,
    predictor: str,
    coverages: Sequence[float],
    seeds: Sequence[int] = (42, 123, 2026),
    minimum_score_only_delta: float = 0.0,
    minimum_supported_alert_fraction: float = 0.01,
    minimum_evidence_tp_fp_gap: float = 0.0,
    minimum_positive_region_fidelity: float | None = None,
    require_full_budget_support: bool = True,
) -> pd.DataFrame:
    """Repeat selection on validation bootstraps and reapply it to full validation.

    Besides the bootstrap-sample policy metrics, the returned table records
    method/abstention stability, selected-row Jaccard against the full-
    validation policy, realized coverage error, and precision/delta dispersion
    on the unchanged validation rows.  No test input is accepted.
    """

    labels = _binary_labels(y_policy_validation)
    risk = np.asarray(validation_probabilities, dtype=float)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    reference_policies, _ = lock_selective_policies(
        labels,
        risk,
        decision_threshold,
        evidence_by_method,
        dataset=dataset,
        predictor=predictor,
        coverages=coverages,
        minimum_score_only_delta=minimum_score_only_delta,
        minimum_supported_alert_fraction=minimum_supported_alert_fraction,
        minimum_evidence_tp_fp_gap=minimum_evidence_tp_fp_gap,
        minimum_positive_region_fidelity=minimum_positive_region_fidelity,
        require_full_budget_support=require_full_budget_support,
    )
    reference_by_coverage = {float(policy.coverage): policy for policy in reference_policies}
    alerts = risk >= float(decision_threshold)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        rng.shuffle(sampled)
        sampled_evidence = {
            method: np.asarray(values, dtype=float)[sampled]
            for method, values in evidence_by_method.items()
        }
        policies, _ = lock_selective_policies(
            labels[sampled],
            risk[sampled],
            decision_threshold,
            sampled_evidence,
            dataset=dataset,
            predictor=predictor,
            coverages=coverages,
            minimum_score_only_delta=minimum_score_only_delta,
            minimum_supported_alert_fraction=minimum_supported_alert_fraction,
            minimum_evidence_tp_fp_gap=minimum_evidence_tp_fp_gap,
            minimum_positive_region_fidelity=minimum_positive_region_fidelity,
            require_full_budget_support=require_full_budget_support,
        )
        for policy in policies:
            coverage = float(policy.coverage)
            reference = reference_by_coverage[coverage]
            selected = apply_locked_policy(policy, risk, decision_threshold, evidence_by_method)
            reference_selected = apply_locked_policy(
                reference, risk, decision_threshold, evidence_by_method
            )
            union = selected | reference_selected
            intersection = selected & reference_selected
            jaccard = float(intersection.sum() / union.sum()) if union.any() else 1.0
            selected_precision = _precision(labels, selected)
            score_only = fixed_count_mask(risk, risk, alerts, int(selected.sum()))
            requested_score_only = fixed_budget_mask(risk, risk, alerts, coverage)
            score_only_precision = _precision(labels, score_only)
            requested_score_only_precision = _precision(labels, requested_score_only)
            realized_coverage = float(selected.sum() / alerts.sum()) if alerts.any() else float("nan")
            rows.append(
                {
                    "seed": int(seed),
                    **policy.to_dict(),
                    "reference_method": reference.method,
                    "reference_abstain": reference.abstain,
                    "method_matches_reference": policy.method == reference.method,
                    "abstention_matches_reference": policy.abstain == reference.abstain,
                    "full_validation_selected_count": int(selected.sum()),
                    "full_validation_requested_selected_count": budget_count(
                        int(alerts.sum()), coverage
                    ),
                    "full_validation_realized_coverage": realized_coverage,
                    "full_validation_coverage_error": (
                        realized_coverage - coverage if np.isfinite(realized_coverage) else float("nan")
                    ),
                    "full_validation_selection_jaccard": jaccard,
                    "full_validation_selected_precision": selected_precision,
                    "full_validation_score_only_selected_count": int(score_only.sum()),
                    "full_validation_score_only_precision": score_only_precision,
                    "full_validation_score_only_requested_budget_precision": (
                        requested_score_only_precision
                    ),
                    "full_validation_equal_count_score_only_comparison": (
                        int(score_only.sum()) == int(selected.sum())
                    ),
                    "full_validation_delta_vs_score_only": (
                        selected_precision - score_only_precision
                        if np.isfinite(selected_precision) and np.isfinite(score_only_precision)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)
