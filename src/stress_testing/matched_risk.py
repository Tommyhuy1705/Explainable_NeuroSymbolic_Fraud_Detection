"""Matched-risk, residual-evidence and paired-bootstrap guardrails.

All matching decisions depend only on frozen predictor scores, evidence scores
and row order. Labels are read only after pairs have been fixed, so the test set
cannot influence which rows are compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def frozen_risk_bin_edges(
    validation_alert_scores: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> np.ndarray:
    """Fit risk-bin boundaries on validation alerts only."""

    scores = np.asarray(validation_alert_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 2:
        raise ValueError("At least two finite validation alert scores are required")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least two")
    interior = np.quantile(scores, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    unique = np.unique(interior)
    return np.concatenate(([-np.inf], unique, [np.inf])).astype(float)


def assign_risk_bins(scores: np.ndarray | Sequence[float], edges: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    boundaries = np.asarray(edges, dtype=float)
    if boundaries.ndim != 1 or len(boundaries) < 2 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("Risk-bin edges must be a strictly increasing one-dimensional array")
    return np.searchsorted(boundaries[1:-1], values, side="right").astype(int)


def match_selected_to_controls(
    predictor_scores: np.ndarray | Sequence[float],
    evidence_scores: np.ndarray | Sequence[float],
    selected_mask: np.ndarray | Sequence[bool],
    alert_mask: np.ndarray | Sequence[bool],
    risk_bin_edges: np.ndarray,
    *,
    caliper: float | None = None,
) -> pd.DataFrame:
    """Create one-to-one nearest-risk pairs without using outcome labels.

    Controls are sorted once per validation-frozen risk bin.  A Fenwick tree
    then returns the nearest still-available predecessor/successor in
    ``O(log n)`` time.  This preserves the original deterministic greedy
    contract while avoiding the quadratic list scan/removal that is not
    viable for multi-million-row stress datasets.
    """

    risk = np.asarray(predictor_scores, dtype=float)
    evidence = np.asarray(evidence_scores, dtype=float)
    selected = np.asarray(selected_mask, dtype=bool)
    alerts = np.asarray(alert_mask, dtype=bool)
    if not (len(risk) == len(evidence) == len(selected) == len(alerts)):
        raise ValueError("Risk, evidence, selected and alert arrays must align")
    if (selected & ~alerts).any():
        raise ValueError("Every selected row must be a predictor alert")
    if not np.isfinite(risk).all() or not np.isfinite(evidence).all():
        raise ValueError("Matching scores must be finite")
    bins = assign_risk_bins(risk, risk_bin_edges)
    effective_caliper = float("inf") if caliper is None else float(caliper)
    if effective_caliper < 0.0:
        raise ValueError("caliper must be non-negative")

    rows: list[dict[str, Any]] = []
    for risk_bin in sorted(np.unique(bins[alerts]).tolist()):
        selected_indices = np.flatnonzero(alerts & selected & (bins == risk_bin))
        control_indices = np.flatnonzero(alerts & ~selected & (bins == risk_bin))
        control_order = np.lexsort((control_indices, risk[control_indices]))
        sorted_controls = control_indices[control_order]
        sorted_control_risk = risk[sorted_controls]

        # Fenwick tree of availability.  kth(order) returns the zero-based
        # position of the requested live control in logarithmic time.
        tree = np.zeros(len(sorted_controls) + 1, dtype=np.int64)

        def update(position: int, delta: int) -> None:
            index = int(position) + 1
            while index < len(tree):
                tree[index] += int(delta)
                index += index & -index

        def prefix_count(stop: int) -> int:
            total = 0
            index = int(stop)
            while index > 0:
                total += int(tree[index])
                index -= index & -index
            return total

        def kth(order: int) -> int:
            if order < 1 or order > prefix_count(len(sorted_controls)):
                raise IndexError("Live-control order is out of range")
            index = 0
            bit = 1 << (len(tree).bit_length() - 1)
            remaining = int(order)
            while bit:
                candidate = index + bit
                if candidate < len(tree) and int(tree[candidate]) < remaining:
                    index = candidate
                    remaining -= int(tree[candidate])
                bit >>= 1
            return index

        for position in range(len(sorted_controls)):
            update(position, 1)
        available_count = len(sorted_controls)
        # Pair the most extreme evidence rows first; ties are resolved by row
        # position.  Neither labels nor future information enter this order.
        selected_order = sorted(selected_indices.tolist(), key=lambda idx: (-evidence[idx], idx))
        for selected_index in selected_order:
            if available_count == 0:
                rows.append(
                    {
                        "risk_bin": int(risk_bin),
                        "selected_index": int(selected_index),
                        "control_index": -1,
                        "selected_risk": float(risk[selected_index]),
                        "control_risk": float("nan"),
                        "absolute_risk_gap": float("nan"),
                        "selected_evidence": float(evidence[selected_index]),
                        "control_evidence": float("nan"),
                        "matched": False,
                    }
                )
                continue
            insertion = int(np.searchsorted(sorted_control_risk, risk[selected_index], side="left"))
            before = prefix_count(insertion)
            total_live = prefix_count(len(sorted_controls))
            candidate_positions: list[int] = []
            if before > 0:
                candidate_positions.append(kth(before))
            if before < total_live:
                candidate_positions.append(kth(before + 1))
            control_position = min(
                candidate_positions,
                key=lambda position: (
                    abs(float(sorted_control_risk[position] - risk[selected_index])),
                    int(sorted_controls[position]),
                ),
            )
            control = int(sorted_controls[control_position])
            gap = abs(float(risk[control] - risk[selected_index]))
            if gap <= effective_caliper:
                update(control_position, -1)
                available_count -= 1
                rows.append(
                    {
                        "risk_bin": int(risk_bin),
                        "selected_index": int(selected_index),
                        "control_index": int(control),
                        "selected_risk": float(risk[selected_index]),
                        "control_risk": float(risk[control]),
                        "absolute_risk_gap": gap,
                        "selected_evidence": float(evidence[selected_index]),
                        "control_evidence": float(evidence[control]),
                        "matched": True,
                    }
                )
            else:
                rows.append(
                    {
                        "risk_bin": int(risk_bin),
                        "selected_index": int(selected_index),
                        "control_index": -1,
                        "selected_risk": float(risk[selected_index]),
                        "control_risk": float("nan"),
                        "absolute_risk_gap": gap,
                        "selected_evidence": float(evidence[selected_index]),
                        "control_evidence": float("nan"),
                        "matched": False,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "risk_bin",
            "selected_index",
            "control_index",
            "selected_risk",
            "control_risk",
            "absolute_risk_gap",
            "selected_evidence",
            "control_evidence",
            "matched",
        ],
    )


def summarize_matched_risk(
    pairs: pd.DataFrame,
    y_test: np.ndarray | Sequence[int],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Attach labels after matching and estimate a paired outcome-difference CI."""

    labels = np.asarray(y_test, dtype=int)
    matched = pairs.loc[pairs["matched"].astype(bool)].copy()
    if matched.empty:
        return {
            "requested_selected_rows": int(len(pairs)),
            "matched_pairs": 0,
            "unmatched_rows": int(len(pairs)),
            "match_rate": 0.0 if len(pairs) else float("nan"),
            "mean_absolute_risk_gap": float("nan"),
            "selected_precision": float("nan"),
            "matched_control_precision": float("nan"),
            "matched_precision_difference": float("nan"),
            "matched_ci_low": float("nan"),
            "matched_ci_high": float("nan"),
            "probability_positive_difference": float("nan"),
        }
    selected_indices = matched["selected_index"].to_numpy(dtype=int)
    control_indices = matched["control_index"].to_numpy(dtype=int)
    if selected_indices.max(initial=-1) >= len(labels) or control_indices.max(initial=-1) >= len(labels):
        raise IndexError("Matched-pair indices exceed the label array")
    pair_differences = labels[selected_indices].astype(float) - labels[control_indices].astype(float)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(n_bootstrap), dtype=float)
    for iteration in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(pair_differences), size=len(pair_differences))
        bootstrap[iteration] = float(pair_differences[sampled].mean())
    return {
        "requested_selected_rows": int(len(pairs)),
        "matched_pairs": int(len(matched)),
        "unmatched_rows": int(len(pairs) - len(matched)),
        "match_rate": float(len(matched) / len(pairs)) if len(pairs) else float("nan"),
        "mean_absolute_risk_gap": float(matched["absolute_risk_gap"].mean()),
        "selected_precision": float(labels[selected_indices].mean()),
        "matched_control_precision": float(labels[control_indices].mean()),
        "matched_precision_difference": float(pair_differences.mean()),
        "matched_ci_low": float(np.quantile(bootstrap, 0.025)),
        "matched_ci_high": float(np.quantile(bootstrap, 0.975)),
        "probability_positive_difference": float((bootstrap > 0.0).mean()),
    }


def residual_tp_fp_by_risk_bin(
    y_test: np.ndarray | Sequence[int],
    predictor_scores: np.ndarray | Sequence[float],
    evidence_scores: np.ndarray | Sequence[float],
    alert_mask: np.ndarray | Sequence[bool],
    risk_bin_edges: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Measure the TP-FP evidence gap after coarse conditioning on risk."""

    labels = np.asarray(y_test, dtype=int)
    risk = np.asarray(predictor_scores, dtype=float)
    evidence = np.asarray(evidence_scores, dtype=float)
    alerts = np.asarray(alert_mask, dtype=bool)
    if not (len(labels) == len(risk) == len(evidence) == len(alerts)):
        raise ValueError("Residual-evidence inputs must align")
    bins = assign_risk_bins(risk, risk_bin_edges)
    rows: list[dict[str, Any]] = []
    weighted_numerator = 0.0
    weighted_denominator = 0
    for risk_bin in sorted(np.unique(bins[alerts]).tolist()):
        in_bin = alerts & (bins == risk_bin)
        tp = in_bin & (labels == 1)
        fp = in_bin & (labels == 0)
        tp_mean = float(evidence[tp].mean()) if tp.any() else float("nan")
        fp_mean = float(evidence[fp].mean()) if fp.any() else float("nan")
        gap = tp_mean - fp_mean if tp.any() and fp.any() else float("nan")
        comparable = int(min(tp.sum(), fp.sum()))
        if np.isfinite(gap) and comparable:
            weighted_numerator += gap * comparable
            weighted_denominator += comparable
        rows.append(
            {
                "risk_bin": int(risk_bin),
                "rows": int(in_bin.sum()),
                "tp_count": int(tp.sum()),
                "fp_count": int(fp.sum()),
                "risk_min": float(risk[in_bin].min()),
                "risk_max": float(risk[in_bin].max()),
                "tp_evidence_mean": tp_mean,
                "fp_evidence_mean": fp_mean,
                "tp_fp_evidence_gap": gap,
                "comparison_weight": comparable,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "alert_rows": int(alerts.sum()),
        "comparable_bins": int(table["tp_fp_evidence_gap"].notna().sum()) if not table.empty else 0,
        "comparison_weight": int(weighted_denominator),
        "risk_conditioned_tp_fp_gap": (
            float(weighted_numerator / weighted_denominator)
            if weighted_denominator
            else float("nan")
        ),
    }
    return table, summary


def paired_selection_bootstrap(
    y_test: np.ndarray | Sequence[int],
    selected_mask: np.ndarray | Sequence[bool],
    score_only_mask: np.ndarray | Sequence[bool],
    *,
    block_ids: np.ndarray | Sequence[Any] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Paired CI for VASRE minus score-only precision on the same test rows."""

    labels = np.asarray(y_test, dtype=int)
    selected = np.asarray(selected_mask, dtype=bool)
    baseline = np.asarray(score_only_mask, dtype=bool)
    if not (len(labels) == len(selected) == len(baseline)):
        raise ValueError("Bootstrap arrays must align")
    if not selected.any() or not baseline.any():
        return {
            "observed_difference": float("nan"),
            "bootstrap_mean_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_vasre_better": float("nan"),
            "valid_iterations": 0,
            "bootstrap_unit": "temporal_block" if block_ids is not None else "row",
        }
    observed = float(labels[selected].mean() - labels[baseline].mean())
    rng = np.random.default_rng(int(seed))
    differences: list[float] = []
    if block_ids is not None:
        blocks = np.asarray(block_ids)
        if len(blocks) != len(labels):
            raise ValueError("block_ids must align with test rows")
        unique_blocks = pd.unique(blocks)
        block_rows = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
        for _ in range(int(n_bootstrap)):
            sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            indices = np.concatenate([block_rows[block] for block in sampled_blocks])
            if selected[indices].any() and baseline[indices].any():
                differences.append(
                    float(labels[indices][selected[indices]].mean() - labels[indices][baseline[indices]].mean())
                )
        unit = "temporal_block"
    else:
        for _ in range(int(n_bootstrap)):
            indices = rng.integers(0, len(labels), size=len(labels))
            if selected[indices].any() and baseline[indices].any():
                differences.append(
                    float(labels[indices][selected[indices]].mean() - labels[indices][baseline[indices]].mean())
                )
        unit = "row"
    if not differences:
        return {
            "observed_difference": observed,
            "bootstrap_mean_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_vasre_better": float("nan"),
            "valid_iterations": 0,
            "bootstrap_unit": unit,
        }
    values = np.asarray(differences, dtype=float)
    return {
        "observed_difference": observed,
        "bootstrap_mean_difference": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_vasre_better": float((values > 0.0).mean()),
        "valid_iterations": int(len(values)),
        "bootstrap_unit": unit,
    }


def classify_stress_outcome(
    coverage_result: dict[str, Any],
    bootstrap_result: dict[str, Any],
    *,
    saturation_headroom: float = 0.05,
) -> str:
    """Create a conservative, proposal-aligned stress-test outcome label."""

    if bool(coverage_result.get("abstain", False)):
        if float(coverage_result.get("score_only_headroom", 1.0)) <= float(saturation_headroom):
            return "saturation_with_abstention"
        return "guardrail_abstention"
    ci_low = float(bootstrap_result.get("ci_low", float("nan")))
    headroom = float(coverage_result.get("score_only_headroom", float("nan")))
    if np.isfinite(headroom) and headroom <= float(saturation_headroom):
        return "score_only_saturation"
    if np.isfinite(ci_low) and ci_low > 0.0:
        return "positive_incremental_evidence"
    return "no_supported_incremental_evidence"
