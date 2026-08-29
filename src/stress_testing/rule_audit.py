"""Validation-only rule audit for the thesis stress-test extension.

This module is intentionally separate from the executed Notebook 01-08
pipeline.  It implements the stricter VASRE contract from the thesis proposal:
rule thresholds are fitted on train, candidate quality is audited on a reserved
validation partition, redundant candidates are removed without consulting the
test labels, and only the surviving rules can contribute to a locked policy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_AUDIT_CRITERIA: dict[str, float | int] = {
    "min_active_count": 10,
    "min_coverage": 0.001,
    "max_coverage": 0.80,
    "min_precision_gain": 0.0,
    "min_lift": 1.0,
    "min_tp_fp_gap": 0.0,
    "max_fp_activation": 0.90,
    "max_coverage_shift": 0.20,
    "max_lift_log_shift": 1.50,
    "min_attribution_hit_rate": 0.0,
    "min_attribution_share": 0.0,
}


def _as_binary(values: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        numeric = labels.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain binary 0/1 values") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain finite binary 0/1 values")
    observed = set(np.unique(numeric).tolist())
    if not observed.issubset({0.0, 1.0}):
        raise ValueError(f"{name} must contain binary 0/1 values; observed={sorted(observed)}")
    return numeric.astype(np.int8)


def _as_alert_mask(
    values: np.ndarray | Sequence[bool],
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    mask = np.asarray(values)
    if mask.ndim != 1 or len(mask) != int(expected_rows):
        raise ValueError(f"{name} must be one-dimensional and align with its truth table")
    if mask.dtype == np.bool_:
        return mask.astype(bool, copy=True)
    try:
        numeric = mask.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only boolean or binary 0/1 values") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only boolean or binary 0/1 values")
    return numeric.astype(bool)


def _validate_truth(truth: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    if len(truth) != len(labels):
        raise ValueError(f"{name} truth values and labels must align")
    if truth.empty or not len(truth.columns):
        raise ValueError(f"{name} truth table must contain at least one rule")
    matrix = truth.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} truth table contains non-finite values")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError(f"{name} truth values must lie in [0, 1]")


def _single_rule_statistics(
    scores: np.ndarray,
    labels: np.ndarray,
    activation_threshold: float,
) -> dict[str, float | int]:
    active = scores >= float(activation_threshold)
    active_count = int(active.sum())
    base_rate = float(labels.mean()) if len(labels) else float("nan")
    precision = float(labels[active].mean()) if active_count else float("nan")
    lift = precision / base_rate if active_count and base_rate > 0.0 else float("nan")
    tp = labels == 1
    fp = labels == 0
    return {
        "rows": int(len(labels)),
        "active_count": active_count,
        "coverage": float(active.mean()) if len(active) else float("nan"),
        "precision": precision,
        "precision_gain": precision - base_rate if active_count else float("nan"),
        "lift": lift,
        "tp_activation": float(active[tp].mean()) if tp.any() else float("nan"),
        "fp_activation": float(active[fp].mean()) if fp.any() else float("nan"),
        "tp_fp_gap": (
            float(active[tp].mean() - active[fp].mean()) if tp.any() and fp.any() else float("nan")
        ),
        "rule_auc": (
            float(roc_auc_score(labels, scores))
            if len(labels) and len(np.unique(labels)) == 2
            else float("nan")
        ),
        "mean_truth": float(scores.mean()) if len(scores) else float("nan"),
    }


def _attribution_statistics(
    rule: str,
    active: np.ndarray,
    attribution_hit: pd.DataFrame | None,
    attribution_share: pd.DataFrame | None,
) -> tuple[float, float]:
    if not active.any():
        return float("nan"), float("nan")
    if attribution_hit is None or rule not in attribution_hit:
        hit_rate = float("nan")
    else:
        values = attribution_hit[rule].to_numpy(dtype=float)
        if len(values) != len(active):
            raise ValueError("Attribution-hit rows must align with the audited truth table")
        hit_rate = float(values[active].mean())
    if attribution_share is None or rule not in attribution_share:
        share = float("nan")
    else:
        values = attribution_share[rule].to_numpy(dtype=float)
        if len(values) != len(active):
            raise ValueError("Attribution-share rows must align with the audited truth table")
        share = float(values[active].mean())
    return hit_rate, share


def audit_rule_candidates(
    train_truth: pd.DataFrame,
    audit_truth: pd.DataFrame,
    y_train: np.ndarray | Sequence[int],
    y_audit: np.ndarray | Sequence[int],
    *,
    train_alert_mask: np.ndarray | Sequence[bool],
    audit_alert_mask: np.ndarray | Sequence[bool],
    registry: pd.DataFrame | None = None,
    attribution_hit: pd.DataFrame | None = None,
    attribution_share: pd.DataFrame | None = None,
    activation_threshold: float = 0.60,
    criteria: Mapping[str, float | int] | None = None,
) -> pd.DataFrame:
    """Audit candidate rules using train and a reserved validation partition.

    Every pass/fail statistic is conditional on the explicitly supplied frozen
    predictor-alert masks. Population statistics are emitted under separately
    named diagnostic columns and never drive selection. ``registry`` may
    provide ``rule``, ``tier``, ``description`` and ``features`` columns. Test
    labels are deliberately absent from this API.
    """

    train_labels = _as_binary(y_train, "y_train")
    audit_labels = _as_binary(y_audit, "y_audit")
    _validate_truth(train_truth, train_labels, "train")
    _validate_truth(audit_truth, audit_labels, "audit")
    train_alerts = _as_alert_mask(
        train_alert_mask,
        expected_rows=len(train_truth),
        name="train_alert_mask",
    )
    audit_alerts = _as_alert_mask(
        audit_alert_mask,
        expected_rows=len(audit_truth),
        name="audit_alert_mask",
    )
    if list(train_truth.columns) != list(audit_truth.columns):
        raise ValueError("Train and audit rule columns must have identical order")
    if not 0.0 < float(activation_threshold) < 1.0:
        raise ValueError("activation_threshold must lie strictly between 0 and 1")

    effective: dict[str, float | int] = dict(DEFAULT_AUDIT_CRITERIA)
    effective.update(dict(criteria or {}))
    registry_rows: dict[str, dict[str, Any]] = {}
    if registry is not None and not registry.empty:
        if "rule" not in registry:
            raise KeyError("Rule registry must contain a 'rule' column")
        registry_rows = {
            str(row["rule"]): row.to_dict() for _, row in registry.drop_duplicates("rule").iterrows()
        }

    rows: list[dict[str, Any]] = []
    for rule in train_truth.columns:
        train_scores = train_truth[rule].to_numpy(dtype=float)
        audit_scores = audit_truth[rule].to_numpy(dtype=float)
        train_population_metrics = _single_rule_statistics(
            train_scores, train_labels, activation_threshold
        )
        audit_population_metrics = _single_rule_statistics(
            audit_scores, audit_labels, activation_threshold
        )
        train_alert_metrics = _single_rule_statistics(
            train_scores[train_alerts], train_labels[train_alerts], activation_threshold
        )
        audit_alert_metrics = _single_rule_statistics(
            audit_scores[audit_alerts], audit_labels[audit_alerts], activation_threshold
        )
        active_alert = audit_alerts & (audit_scores >= float(activation_threshold))
        attribution_hit_rate, mean_attribution_share = _attribution_statistics(
            rule, active_alert, attribution_hit, attribution_share
        )
        alert_coverage_shift = abs(
            float(audit_alert_metrics["coverage"])
            - float(train_alert_metrics["coverage"])
        )
        if np.isfinite(float(train_alert_metrics["lift"])) and np.isfinite(
            float(audit_alert_metrics["lift"])
        ):
            alert_lift_log_shift = abs(
                float(
                    np.log(
                        (float(audit_alert_metrics["lift"]) + 1e-8)
                        / (float(train_alert_metrics["lift"]) + 1e-8)
                    )
                )
            )
        else:
            alert_lift_log_shift = float("inf")

        failures: list[str] = []
        train_alert_classes = np.unique(train_labels[train_alerts])
        audit_alert_classes = np.unique(audit_labels[audit_alerts])
        if not train_alerts.any():
            failures.append("train_alert_empty")
        elif len(train_alert_classes) < 2:
            failures.append("train_alert_one_class")
        if not audit_alerts.any():
            failures.append("audit_alert_empty")
        elif len(audit_alert_classes) < 2:
            failures.append("audit_alert_one_class")
        checks = (
            (
                int(audit_alert_metrics["active_count"]) >= int(effective["min_active_count"]),
                "audit_alert_active_count",
            ),
            (
                float(audit_alert_metrics["coverage"]) >= float(effective["min_coverage"]),
                "audit_alert_min_coverage",
            ),
            (
                float(audit_alert_metrics["coverage"]) <= float(effective["max_coverage"]),
                "audit_alert_max_coverage",
            ),
            (
                np.isfinite(float(audit_alert_metrics["precision_gain"]))
                and float(audit_alert_metrics["precision_gain"])
                >= float(effective["min_precision_gain"]),
                "audit_alert_precision_gain",
            ),
            (
                np.isfinite(float(audit_alert_metrics["lift"]))
                and float(audit_alert_metrics["lift"]) >= float(effective["min_lift"]),
                "audit_alert_lift",
            ),
            (
                np.isfinite(float(audit_alert_metrics["tp_fp_gap"]))
                and float(audit_alert_metrics["tp_fp_gap"])
                >= float(effective["min_tp_fp_gap"]),
                "audit_alert_tp_fp_gap",
            ),
            (
                np.isfinite(float(audit_alert_metrics["fp_activation"]))
                and float(audit_alert_metrics["fp_activation"])
                <= float(effective["max_fp_activation"]),
                "audit_alert_fp_activation",
            ),
            (
                alert_coverage_shift <= float(effective["max_coverage_shift"]),
                "alert_coverage_shift",
            ),
            (
                alert_lift_log_shift <= float(effective["max_lift_log_shift"]),
                "alert_lift_log_shift",
            ),
        )
        failures.extend(reason for passed, reason in checks if not passed)
        if float(effective["min_attribution_hit_rate"]) > 0.0:
            if not np.isfinite(attribution_hit_rate) or attribution_hit_rate < float(
                effective["min_attribution_hit_rate"]
            ):
                failures.append("audit_alert_attribution_hit_rate")
        if float(effective["min_attribution_share"]) > 0.0:
            if not np.isfinite(mean_attribution_share) or mean_attribution_share < float(
                effective["min_attribution_share"]
            ):
                failures.append("audit_alert_attribution_share")

        gain = (
            max(float(audit_alert_metrics["precision_gain"]), 0.0)
            if np.isfinite(float(audit_alert_metrics["precision_gain"]))
            else 0.0
        )
        gap = (
            max(float(audit_alert_metrics["tp_fp_gap"]), 0.0)
            if np.isfinite(float(audit_alert_metrics["tp_fp_gap"]))
            else 0.0
        )
        lift_component = (
            max(
                float(
                    np.log1p(max(float(audit_alert_metrics["lift"]) - 1.0, 0.0))
                ),
                0.0,
            )
            if np.isfinite(float(audit_alert_metrics["lift"]))
            else 0.0
        )
        hit_component = attribution_hit_rate if np.isfinite(attribution_hit_rate) else 0.0
        share_component = mean_attribution_share if np.isfinite(mean_attribution_share) else 0.0
        fp_component = (
            float(audit_alert_metrics["fp_activation"])
            if np.isfinite(float(audit_alert_metrics["fp_activation"]))
            else 1.0
        )
        coverage_penalty = alert_coverage_shift if np.isfinite(alert_coverage_shift) else 1.0
        lift_shift_penalty = (
            min(alert_lift_log_shift, 5.0) if np.isfinite(alert_lift_log_shift) else 5.0
        )
        audit_alert_score = (
            0.30 * gain
            + 0.25 * gap
            + 0.15 * lift_component
            + 0.15 * hit_component
            + 0.15 * share_component
            - 0.15 * coverage_penalty
            - 0.10 * lift_shift_penalty
            - 0.10 * fp_component
        )
        info = registry_rows.get(str(rule), {})
        rows.append(
            {
                "rule": str(rule),
                "tier": str(info.get("tier", "unspecified")),
                "description": str(info.get("description", str(rule).replace("_", " "))),
                "features": json.dumps(info.get("features", []), ensure_ascii=False),
                **{
                    f"train_population_{key}": value
                    for key, value in train_population_metrics.items()
                },
                **{
                    f"audit_population_{key}": value
                    for key, value in audit_population_metrics.items()
                },
                **{
                    f"train_alert_{key}": value
                    for key, value in train_alert_metrics.items()
                },
                **{
                    f"audit_alert_{key}": value
                    for key, value in audit_alert_metrics.items()
                },
                "alert_coverage_shift": alert_coverage_shift,
                "alert_lift_log_shift": alert_lift_log_shift,
                "audit_alert_attribution_hit_rate": attribution_hit_rate,
                "audit_alert_attribution_share": mean_attribution_share,
                "audit_alert_score": float(audit_alert_score),
                "audit_pass": not failures,
                "audit_failures": ";".join(failures),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["audit_pass", "audit_alert_score", "rule"], ascending=[False, False, True]
    ).reset_index(drop=True)


def deduplicate_audited_rules(
    audit_table: pd.DataFrame,
    audit_truth: pd.DataFrame,
    *,
    audit_alert_mask: np.ndarray | Sequence[bool],
    activation_threshold: float = 0.60,
    max_jaccard: float = 0.85,
    max_rules: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily retain high-quality audited rules with bounded overlap."""

    required = {"rule", "audit_pass", "audit_alert_score"}
    missing = required - set(audit_table.columns)
    if missing:
        raise KeyError(f"Audit table is missing columns: {sorted(missing)}")
    if not 0.0 <= float(max_jaccard) <= 1.0:
        raise ValueError("max_jaccard must lie in [0, 1]")
    if int(max_rules) < 1:
        raise ValueError("max_rules must be at least one")
    audit_alerts = _as_alert_mask(
        audit_alert_mask,
        expected_rows=len(audit_truth),
        name="audit_alert_mask",
    )

    ordered = audit_table.loc[audit_table["audit_pass"].astype(bool)].sort_values(
        ["audit_alert_score", "rule"], ascending=[False, True]
    )
    selected: list[str] = []
    decisions: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    masks = {
        rule: (
            audit_truth.loc[audit_alerts, rule].to_numpy(dtype=float)
            >= float(activation_threshold)
        )
        for rule in audit_truth.columns
    }
    for _, row in ordered.iterrows():
        rule = str(row["rule"])
        if rule not in masks:
            raise KeyError(f"Audited rule is absent from truth table: {rule}")
        blocker: str | None = None
        maximum_overlap = 0.0
        for kept in selected:
            union = masks[rule] | masks[kept]
            overlap = float((masks[rule] & masks[kept]).sum() / union.sum()) if union.any() else 1.0
            jaccard_rows.append({"rule_a": rule, "rule_b": kept, "jaccard": overlap})
            if overlap > maximum_overlap:
                maximum_overlap = overlap
            if blocker is None and overlap > float(max_jaccard):
                blocker = kept
        if blocker is None and len(selected) < int(max_rules):
            selected.append(rule)
            status = "selected"
        elif blocker is not None:
            status = "redundant"
        else:
            status = "rule_budget_exceeded"
        decisions.append(
            {
                "rule": rule,
                "selection_status": status,
                "redundant_with": blocker,
                "max_jaccard_with_selected": maximum_overlap,
            }
        )

    completed = audit_table.merge(pd.DataFrame(decisions), on="rule", how="left")
    completed["selection_status"] = completed["selection_status"].fillna("audit_failed")
    completed["selected"] = completed["selection_status"].eq("selected")
    jaccard = pd.DataFrame(jaccard_rows, columns=["rule_a", "rule_b", "jaccard"])
    return completed.sort_values(
        ["selected", "audit_alert_score"], ascending=[False, False]
    ), jaccard


def compute_rule_weights(
    completed_audit: pd.DataFrame,
    *,
    fp_penalty: float = 1.0,
) -> pd.DataFrame:
    """Create validation-frozen standard and FP-penalized rule weights."""

    required = {
        "rule",
        "selected",
        "audit_alert_score",
        "audit_alert_tp_activation",
        "audit_alert_fp_activation",
    }
    missing = required - set(completed_audit.columns)
    if missing:
        raise KeyError(f"Completed audit table is missing alert fields: {sorted(missing)}")
    selected = completed_audit.loc[completed_audit["selected"].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "rule",
                "audit_weight",
                "fp_penalized_weight",
                "audit_alert_score_component",
            ]
        )
    score = selected["audit_alert_score"].to_numpy(dtype=float)
    score = np.maximum(score - min(float(score.min()), 0.0), 0.0) + 1e-8
    audit_weight = score / score.sum()
    gap = np.maximum(
        selected["audit_alert_tp_activation"].to_numpy(dtype=float)
        - float(fp_penalty)
        * selected["audit_alert_fp_activation"].to_numpy(dtype=float),
        0.0,
    )
    penalized = audit_weight * (gap + 1e-8)
    penalized = penalized / penalized.sum()
    return pd.DataFrame(
        {
            "rule": selected["rule"].astype(str).to_numpy(),
            "audit_weight": audit_weight,
            "fp_penalized_weight": penalized,
            "audit_alert_score_component": score,
        }
    )


def build_evidence_scores(
    truth_values: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    method: str,
    attribution_share: pd.DataFrame | None = None,
) -> np.ndarray:
    """Compute a read-only evidence score from validation-frozen rule weights."""

    if weights.empty:
        return np.zeros(len(truth_values), dtype=float)
    rules = weights["rule"].astype(str).tolist()
    missing = [rule for rule in rules if rule not in truth_values]
    if missing:
        raise KeyError(f"Truth table is missing selected rules: {missing}")
    matrix = truth_values[rules].to_numpy(dtype=float)
    if method == "unweighted":
        coefficients = np.full(len(rules), 1.0 / len(rules))
    elif method == "audit_weighted":
        coefficients = weights.set_index("rule").loc[rules, "audit_weight"].to_numpy(dtype=float)
    elif method in {"fp_penalized", "attribution_gated"}:
        coefficients = weights.set_index("rule").loc[rules, "fp_penalized_weight"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported evidence method: {method}")
    if method == "attribution_gated":
        if attribution_share is None:
            raise ValueError("attribution_gated evidence requires per-rule attribution shares")
        missing_attribution = [rule for rule in rules if rule not in attribution_share]
        if missing_attribution:
            raise KeyError(f"Attribution table is missing selected rules: {missing_attribution}")
        gate = np.clip(attribution_share[rules].to_numpy(dtype=float), 0.0, 1.0)
        matrix = matrix * gate
    scores = matrix @ coefficients
    if not np.isfinite(scores).all():
        raise ValueError("Evidence scores contain non-finite values")
    return np.clip(scores, 0.0, 1.0)
