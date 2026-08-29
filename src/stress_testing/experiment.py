"""State-ordered orchestration for the two thesis stress tests.

The public :func:`run_stress_test` entry point deliberately implements a
stronger protocol than an ordinary benchmark script.  Model fitting uses the
training period only.  Calibrators are fitted on ``calibration_fit``;
calibration method, operating threshold and the reference predictor are locked
on ``calibration_select``; candidate rules are audited on ``rule_audit``; and the
selective explanation policy is locked on ``policy_select``.  The splitter may
inspect test labels only for schema/class-count integrity.  Test outcomes never
enter model, rule or policy selection, and every test selection mask is
materialised before outcomes are released for aggregate evaluation.  This
ordering is recorded in ``protocol_ledger.json``.

Quick runs and generated fixtures are explicit engineering checks.  They
produce the same artifact schema but are marked ineligible for thesis claims.
Full runs never replace a missing XGBoost or LightGBM backend silently.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.preprocessing import FraudPreprocessor
from .artifacts import (
    load_yaml_config,
    write_json,
    write_stress_lineage,
    write_stress_manifest,
)
from .attribution import AttributionResult, compute_attributions
from .candidate_rules import (
    NON_INTERVENABLE_HISTORY_DERIVED_FEATURES,
    combine_candidate_registries,
    fit_cart_surrogate_candidates,
    fit_configured_candidates,
    generate_counterfactual_numeric_candidates,
    score_only_baseline_registry,
)
from .contrastive import ContrastiveMetaEvidence, fit_contrastive_meta_evidence
from .data import (
    StressDataset,
    TemporalStressSplit,
    load_stress_dataset,
    make_synthetic_amlnet_test_fixture,
    make_synthetic_transxion_test_fixture,
    temporal_stress_split,
)
from .matched_risk import (
    classify_stress_outcome,
    frozen_risk_bin_edges,
    match_selected_to_controls,
    paired_selection_bootstrap,
    residual_tp_fp_by_risk_bin,
    summarize_matched_risk,
)
from .metrics import evaluate_stress_predictions, fit_reserved_validation_protocol
from .predictors import (
    FrozenPredictor,
    freeze_predictor,
    predict_probabilities,
    torch,
    train_stress_predictor,
)
from .rule_audit import (
    audit_rule_candidates,
    build_evidence_scores,
    compute_rule_weights,
    deduplicate_audited_rules,
)
from .selective_policy import (
    LockedSelectivePolicy,
    apply_locked_policy,
    budget_count,
    evaluate_locked_policy,
    fixed_count_mask,
    lock_selective_policies,
    validation_policy_stability,
)


PROTOCOL_PHASES: tuple[str, ...] = (
    "data_loaded",
    "temporal_split_locked",
    "preprocessor_fitted_on_train",
    "predictors_trained_on_train",
    "calibration_and_threshold_frozen",
    "reference_predictor_selected",
    "rules_fitted_on_train",
    "rules_audited_on_reserved_validation",
    "policies_locked_on_reserved_validation",
    "test_selections_materialized_without_outcome_use",
    "test_outcomes_released_for_evaluation",
    "test_evaluated_without_reselection",
)


@dataclass
class _ProtocolLedger:
    """Fail closed if orchestration attempts to skip or reorder a phase."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def advance(self, phase: str, **details: Any) -> None:
        position = len(self.events)
        if position >= len(PROTOCOL_PHASES) or PROTOCOL_PHASES[position] != phase:
            expected = PROTOCOL_PHASES[position] if position < len(PROTOCOL_PHASES) else None
            raise RuntimeError(f"Invalid protocol phase {phase!r}; expected {expected!r}")
        self.events.append(
            {
                "sequence": position + 1,
                "phase": phase,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                **details,
            }
        )

    @property
    def complete(self) -> bool:
        return [event["phase"] for event in self.events] == list(PROTOCOL_PHASES)


@dataclass
class _CandidateState:
    family: str
    seed: int
    frozen: FrozenPredictor
    validation_metrics: dict[str, float]
    calibration_comparison: tuple[dict[str, float | str], ...]
    training_history: tuple[dict[str, float], ...]


def _find_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "src").is_dir() and (cwd / "configs").is_dir():
        return cwd
    raise FileNotFoundError("Could not locate the repository root containing src/ and configs/")


def _normalise_dataset_name(config: Mapping[str, Any]) -> str:
    return (
        str(config.get("dataset", {}).get("name", ""))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def _validate_protocol_config(config: Mapping[str, Any]) -> None:
    """Reject declarative settings that the confirmatory runner cannot honor."""

    split = dict(config.get("split", {}))
    if str(split.get("strategy", "timestamp_safe_temporal")) != "timestamp_safe_temporal":
        raise ValueError("Stress split.strategy must be timestamp_safe_temporal")
    expected_roles = ["calibration_fit", "calibration_select", "rule_audit", "policy_select"]
    if list(split.get("validation_roles", expected_roles)) != expected_roles:
        raise ValueError(f"split.validation_roles must equal {expected_roles}")

    models = dict(config.get("models", {}))
    if str(models.get("selection_metric", "pr_auc")).lower() != "pr_auc":
        raise ValueError("The confirmatory stress predictor selection metric must be pr_auc")

    evaluation = dict(config.get("evaluation", {}))
    if str(evaluation.get("bootstrap_block", "calendar_day")) != "calendar_day":
        raise ValueError("evaluation.bootstrap_block must be calendar_day")
    if not np.isclose(float(evaluation.get("confidence_level", 0.95)), 0.95):
        raise ValueError("The implemented percentile intervals require confidence_level=0.95")
    required_outcome_thresholds = {
        "saturation_headroom",
        "saturation_precision_margin",
        "saturation_evidence_margin",
        "minimum_positive_stability_fraction",
        "minimum_saturation_alert_positives",
        "minimum_saturation_matched_pairs",
        "minimum_saturation_bootstrap_replicates",
    }
    missing_thresholds = required_outcome_thresholds - set(evaluation)
    if missing_thresholds:
        raise ValueError(
            "Confirmatory stress configs must pre-register outcome thresholds: "
            f"{sorted(missing_thresholds)}"
        )
    bounded_thresholds = {
        name: float(evaluation[name]) for name in required_outcome_thresholds
    }
    for name in (
        "saturation_headroom",
        "saturation_precision_margin",
        "saturation_evidence_margin",
    ):
        if not 0.0 <= bounded_thresholds[name] <= 1.0:
            raise ValueError(f"evaluation.{name} must lie in [0, 1]")
    if not 0.0 < bounded_thresholds["minimum_positive_stability_fraction"] <= 1.0:
        raise ValueError("evaluation.minimum_positive_stability_fraction must lie in (0, 1]")
    for name in (
        "minimum_saturation_alert_positives",
        "minimum_saturation_matched_pairs",
        "minimum_saturation_bootstrap_replicates",
    ):
        if int(evaluation[name]) < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer")

    logic = dict(config.get("logic", {}))
    evidence_methods = [str(value) for value in logic.get("evidence_methods", [])]
    required_evidence_methods = {
        "unweighted",
        "audit_weighted",
        "fp_penalized",
        "attribution_gated",
        "contrastive_meta",
        "guarded_ensemble",
    }
    missing_evidence_methods = required_evidence_methods - set(evidence_methods)
    if missing_evidence_methods:
        raise ValueError(
            "Confirmatory stress configs must include all pre-registered evidence methods: "
            f"{sorted(missing_evidence_methods)}"
        )
    contrastive = dict(logic.get("contrastive_meta", {}))
    required_contrastive = {
        "enabled",
        "minimum_alert_rows",
        "minimum_rows_per_class",
        "regularization_c",
        "max_iter",
    }
    missing_contrastive = required_contrastive - set(contrastive)
    if missing_contrastive:
        raise ValueError(
            "Confirmatory stress configs must pre-register contrastive meta-scorer settings: "
            f"{sorted(missing_contrastive)}"
        )
    if not bool(contrastive["enabled"]):
        raise ValueError("Confirmatory stress protocol requires logic.contrastive_meta.enabled=true")
    if int(contrastive["minimum_alert_rows"]) < 1:
        raise ValueError("logic.contrastive_meta.minimum_alert_rows must be positive")
    if int(contrastive["minimum_rows_per_class"]) < 2:
        raise ValueError("logic.contrastive_meta.minimum_rows_per_class must be at least 2")
    if float(contrastive["regularization_c"]) <= 0.0:
        raise ValueError("logic.contrastive_meta.regularization_c must be positive")
    if int(contrastive["max_iter"]) < 1:
        raise ValueError("logic.contrastive_meta.max_iter must be positive")
    guarded = dict(logic.get("guarded_ensemble", {}))
    required_guarded = {"enabled", "aggregation", "components"}
    missing_guarded = required_guarded - set(guarded)
    if missing_guarded:
        raise ValueError(
            "Confirmatory stress configs must pre-register guarded ensemble settings: "
            f"{sorted(missing_guarded)}"
        )
    if not bool(guarded["enabled"]):
        raise ValueError("Confirmatory stress protocol requires logic.guarded_ensemble.enabled=true")
    if str(guarded["aggregation"]) != "arithmetic_mean":
        raise ValueError("logic.guarded_ensemble.aggregation must be arithmetic_mean")
    guarded_components = [str(value) for value in guarded["components"]]
    if len(guarded_components) < 2 or len(set(guarded_components)) != len(
        guarded_components
    ):
        raise ValueError("logic.guarded_ensemble.components must contain unique methods")
    if "guarded_ensemble" in guarded_components:
        raise ValueError("guarded_ensemble cannot contain itself")
    missing_guarded_components = set(guarded_components) - set(evidence_methods)
    if missing_guarded_components:
        raise ValueError(
            "logic.guarded_ensemble.components are absent from logic.evidence_methods: "
            f"{sorted(missing_guarded_components)}"
        )
    counterfactual = dict(config.get("counterfactual_candidates", {}))
    if not bool(counterfactual.get("enabled", False)):
        raise ValueError("Confirmatory stress protocol requires counterfactual candidates")
    derived_contract = counterfactual.get("derived_feature_contract")
    if not isinstance(derived_contract, Mapping) or not derived_contract:
        raise ValueError(
            "Confirmatory stress configs must pre-register a derived_feature_contract"
        )
    derived_outputs = {
        str(specification.get("output"))
        for specifications in derived_contract.values()
        for specification in specifications
        if isinstance(specification, Mapping)
    }
    directly_intervened = {
        str(value) for value in counterfactual.get("numeric_features", [])
    }
    declared_non_intervenable = counterfactual.get(
        "non_intervenable_derived_features"
    )
    if (
        isinstance(declared_non_intervenable, (str, bytes))
        or not isinstance(declared_non_intervenable, Sequence)
    ):
        raise ValueError(
            "counterfactual_candidates.non_intervenable_derived_features must be a sequence"
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for value in declared_non_intervenable
    ):
        raise ValueError(
            "counterfactual_candidates.non_intervenable_derived_features must contain "
            "non-empty strings"
        )
    declared_non_intervenable_set = {
        value.strip() for value in declared_non_intervenable
    }
    if len(declared_non_intervenable_set) != len(declared_non_intervenable):
        raise ValueError(
            "counterfactual_candidates.non_intervenable_derived_features must be unique"
        )
    missing_history_declarations = (
        set(NON_INTERVENABLE_HISTORY_DERIVED_FEATURES)
        - declared_non_intervenable_set
    )
    if missing_history_declarations:
        raise ValueError(
            "Confirmatory stress configs must declare all non-intervenable causal "
            f"history features: {sorted(missing_history_declarations)}"
        )
    forbidden_history_interventions = (
        directly_intervened & declared_non_intervenable_set
    )
    if forbidden_history_interventions:
        raise ValueError(
            "Counterfactual candidates cannot directly intervene on causal "
            f"history-derived fields: {sorted(forbidden_history_interventions)}"
        )
    derived_interventions = directly_intervened & derived_outputs
    if derived_interventions:
        raise ValueError(
            "Counterfactual candidates must intervene on raw/causal fields, not derived outputs: "
            f"{sorted(derived_interventions)}"
        )
    guardrails = dict(logic.get("policy_guardrails", {}))
    required_guardrails = {
        "minimum_supported_alert_fraction",
        "minimum_evidence_tp_fp_gap",
        "minimum_positive_region_fidelity",
        "require_full_budget_support",
        "minimum_policy_stability_fraction",
    }
    missing_guardrails = required_guardrails - set(guardrails)
    if missing_guardrails:
        raise ValueError(
            "Confirmatory stress configs must pre-register policy guardrails: "
            f"{sorted(missing_guardrails)}"
        )
    for name in (
        "minimum_supported_alert_fraction",
        "minimum_positive_region_fidelity",
        "minimum_policy_stability_fraction",
    ):
        if not 0.0 <= float(guardrails[name]) <= 1.0:
            raise ValueError(f"logic.policy_guardrails.{name} must lie in [0, 1]")
    if float(guardrails["minimum_evidence_tp_fp_gap"]) < 0.0:
        raise ValueError("logic.policy_guardrails.minimum_evidence_tp_fp_gap must be non-negative")
    if not bool(guardrails["require_full_budget_support"]):
        raise ValueError("Confirmatory stress protocol requires full active-rule budget support")

    reporting = dict(config.get("reporting", {}))
    for field in ("report_negative_results", "report_fallbacks", "report_saturation"):
        if field in reporting and not bool(reporting[field]):
            raise ValueError(f"Confirmatory stress protocol requires reporting.{field}=true")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _partition_table(split: TemporalStressSplit) -> pd.DataFrame:
    requested = split.manifest["requested_ratios"]
    total_rows = len(split.train) + len(split.validation) + len(split.test)
    requested_by_partition = {
        "train": float(requested["train"]),
        "validation": float(requested["validation"]),
        "test": float(requested["test"]),
        "calibration_fit": float(requested["validation"]) / 4.0,
        "calibration_select": float(requested["validation"]) / 4.0,
        "rule_audit": float(requested["validation"]) / 4.0,
        "policy_select": float(requested["validation"]) / 4.0,
    }
    rows: list[dict[str, Any]] = []
    for name, details in split.manifest["partitions"].items():
        rows.append(
            {
                "partition": name,
                "rows": details["rows"],
                "requested_proportion": requested_by_partition[name],
                "observed_proportion": float(details["rows"] / total_rows),
                "positive_count": details["positive_count"],
                "positive_rate": details["positive_rate"],
                "time_minimum": details["time"]["minimum"],
                "time_maximum": details["time"]["maximum"],
                "both_classes": int(details["positive_count"]) > 0
                and int(details["positive_count"]) < int(details["rows"]),
                "chronologically_disjoint": True,
            }
        )
    return pd.DataFrame(rows)


def _preprocessor_from_config(
    config: Mapping[str, Any],
    leakage_denylist: Sequence[str],
) -> FraudPreprocessor:
    settings = dict(config.get("preprocessing", {}))
    allowed = {
        "max_missing_fraction",
        "max_features",
        "categorical_max_cardinality",
        "scale_numeric",
        "drop_columns",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unsupported preprocessing settings: {sorted(unknown)}")
    settings["drop_columns"] = sorted(
        set(settings.get("drop_columns", [])) | set(leakage_denylist) | {"label", "event_time", "__source_order"}
    )
    return FraudPreprocessor(**settings)


def _assert_no_leakage_features(
    selected_features: Sequence[str],
    leakage_denylist: Sequence[str],
) -> None:
    """Reject exact or case-insensitive outcome proxies before model fitting."""

    denied = {str(value).casefold() for value in leakage_denylist}
    denied.update(
        {
            "label",
            "isfraud",
            "ismoneylaundering",
            "is laundering",
            "fraud_probability",
            "laundering_typology",
            "money_laundering_typology",
            "auxiliary_fraud_label",
            "auxiliary_money_laundering_label",
            "metadata",
        }
    )
    collisions = sorted({str(feature) for feature in selected_features if str(feature).casefold() in denied})
    if collisions:
        raise AssertionError(f"Leakage-denied predictor features were selected: {collisions}")


def _feature_frame(frame: pd.DataFrame, dataset: StressDataset) -> pd.DataFrame:
    missing = [column for column in dataset.feature_columns if column not in frame]
    if missing:
        raise KeyError(f"Canonical predictor columns are missing: {missing}")
    return frame.loc[:, list(dataset.feature_columns)]


def _labels(frame: pd.DataFrame) -> np.ndarray:
    values = frame["label"].to_numpy(dtype=np.int8, copy=True)
    if set(np.unique(values).tolist()) != {0, 1}:
        raise ValueError("Each protocol partition must retain both target classes")
    return values


def _quick_model_config(
    family: str,
    models: Mapping[str, Any],
    quick_run: bool,
) -> dict[str, Any]:
    settings = dict(models.get(family, {}))
    if not quick_run:
        return settings
    overrides = dict(models.get("quick_overrides", {}))
    if family in {"xgboost", "lightgbm"}:
        settings["n_estimators"] = int(overrides.get("tree_n_estimators", 25))
    elif family == "tabular_resnet_v2":
        settings.update(
            {
                "epochs": int(overrides.get("neural_epochs", 2)),
                "hidden_dim": int(overrides.get("neural_hidden_dim", 48)),
                "num_blocks": int(overrides.get("neural_num_blocks", 2)),
                "batch_size": min(int(settings.get("batch_size", 128)), 256),
            }
        )
    return settings


def _train_and_select_reference(
    config: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    *,
    quick_run: bool,
    device: str | None,
) -> tuple[list[_CandidateState], int, pd.DataFrame, pd.DataFrame]:
    models = dict(config.get("models", {}))
    families = [str(value).lower().replace("-", "_") for value in models.get("families", [])]
    required = {"xgboost", "lightgbm", "tabular_resnet_v2"}
    if not families:
        families = ["xgboost", "lightgbm", "tabular_resnet_v2"]
    if not quick_run and set(families) != required:
        raise ValueError("A full stress test must configure XGBoost, LightGBM and TabularResNetV2")
    seeds = [int(value) for value in models.get("seeds", [config.get("project", {}).get("seed", 42)])]
    if not seeds:
        raise ValueError("models.seeds must contain at least one seed")
    effective_seeds = seeds[:1] if quick_run else seeds

    evaluation = dict(config.get("evaluation", {}))
    calibration_methods = tuple(evaluation.get("calibration_methods", ["none", "platt", "isotonic"]))
    candidates: list[_CandidateState] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    ranking: list[tuple[tuple[float, float, int, int], int]] = []

    for family_index, family in enumerate(families):
        for seed in effective_seeds:
            bundle = train_stress_predictor(
                family,
                matrices["train"],
                labels["train"],
                feature_names=feature_names,
                config=_quick_model_config(family, models, quick_run),
                seed=seed,
                quick_run=quick_run,
                allow_quick_fallback=quick_run,
                device="cpu" if quick_run else device,
            )
            raw_fit = predict_probabilities(bundle, matrices["calibration_fit"], calibrated=False, device=device)
            raw_select = predict_probabilities(
                bundle, matrices["calibration_select"], calibrated=False, device=device
            )
            protocol = fit_reserved_validation_protocol(
                labels["calibration_fit"],
                raw_fit,
                labels["calibration_select"],
                raw_select,
                calibration_methods=calibration_methods,
                selection_metric=str(evaluation.get("calibration_selection_metric", "brier")),
                beta=float(evaluation.get("beta", 2.0)),
                n_calibration_bins=int(evaluation.get("n_calibration_bins", 15)),
                threshold_grid_size=int(evaluation.get("threshold_grid_size", 501)),
                seed=seed,
            )
            frozen = freeze_predictor(bundle, protocol, protocol_metadata=protocol.provenance)
            selection_probabilities = predict_probabilities(
                frozen, matrices["calibration_select"], calibrated=True, device=device
            )
            metrics = evaluate_stress_predictions(
                labels["calibration_select"],
                selection_probabilities,
                threshold=frozen.threshold,
                beta=float(evaluation.get("beta", 2.0)),
                n_calibration_bins=int(evaluation.get("n_calibration_bins", 15)),
                target_fpr=float(evaluation.get("recall_fpr", 0.01)),
            )
            state = _CandidateState(
                family=family,
                seed=seed,
                frozen=frozen,
                validation_metrics=metrics,
                calibration_comparison=protocol.comparison,
                training_history=bundle.training_history,
            )
            candidate_index = len(candidates)
            candidates.append(state)
            metric_rows.append(
                {
                    "evaluation_partition": "calibration_select",
                    "family": family,
                    "backend": frozen.backend,
                    "seed": seed,
                    "calibration_method": protocol.calibrator.method,
                    "selection_split_pr_auc": float(metrics["pr_auc"]),
                    **metrics,
                }
            )
            for row in protocol.comparison:
                calibration_rows.append(
                    {
                        "family": family,
                        "backend": frozen.backend,
                        "seed": seed,
                        "selected": str(row["method"]) == protocol.calibrator.method,
                        **row,
                    }
                )
            pr_auc = float(average_precision_score(labels["calibration_select"], selection_probabilities))
            brier = float(metrics["brier"])
            ranking.append(((-pr_auc, brier, family_index, seed), candidate_index))

    if not candidates:
        raise RuntimeError("No predictor candidate was trained")
    reference_index = min(ranking, key=lambda item: item[0])[1]
    metric_table = pd.DataFrame(metric_rows)
    metric_table["reference_selected"] = False
    metric_table.loc[reference_index, "reference_selected"] = True
    return candidates, reference_index, metric_table, pd.DataFrame(calibration_rows)


def _empty_attribution_tables(
    rows: int,
    rules: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zeros = np.zeros((rows, len(rules)), dtype=float)
    return pd.DataFrame(zeros, columns=rules), pd.DataFrame(zeros.copy(), columns=rules)


def _per_rule_attribution(
    result: AttributionResult,
    registry: pd.DataFrame,
    rule_names: Sequence[str],
    *,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute = np.abs(np.asarray(result.values, dtype=float))
    denominator = np.maximum(absolute.sum(axis=1), 1e-12)
    name_to_index = {name: index for index, name in enumerate(result.feature_names)}
    k = min(max(int(top_k), 1), absolute.shape[1])
    top = np.argpartition(absolute, kth=absolute.shape[1] - k, axis=1)[:, -k:]
    hit: dict[str, np.ndarray] = {}
    share: dict[str, np.ndarray] = {}
    registry_by_rule = registry.set_index("rule", drop=False).to_dict("index")
    for rule_name in rule_names:
        raw_features = registry_by_rule.get(str(rule_name), {}).get("features", [])
        if isinstance(raw_features, str):
            try:
                raw_features = json.loads(raw_features)
            except json.JSONDecodeError:
                raw_features = [raw_features]
        indices = sorted(
            {
                name_to_index[str(feature)]
                for feature in raw_features
                if str(feature) in name_to_index
            }
        )
        if not indices:
            hit[str(rule_name)] = np.zeros(len(absolute), dtype=float)
            share[str(rule_name)] = np.zeros(len(absolute), dtype=float)
            continue
        hit[str(rule_name)] = np.isin(top, np.asarray(indices, dtype=int)).any(axis=1).astype(float)
        share[str(rule_name)] = absolute[:, indices].sum(axis=1) / denominator
    return pd.DataFrame(hit), pd.DataFrame(share)


def _positive_top_k_attribution_evidence(
    attribution_values: np.ndarray,
    *,
    top_k: int = 5,
) -> np.ndarray:
    """Sum fraud-direction contributions among the largest local effects."""

    signed = np.asarray(attribution_values, dtype=float)
    if signed.ndim != 2 or signed.shape[1] < 1:
        raise ValueError("Attribution values must be a non-empty two-dimensional matrix")
    if not np.isfinite(signed).all() or int(top_k) < 1:
        raise ValueError("Attribution values must be finite and top_k must be positive")
    absolute = np.abs(signed)
    effective_k = min(int(top_k), signed.shape[1])
    top_indices = np.argpartition(
        absolute, kth=absolute.shape[1] - effective_k, axis=1
    )[:, -effective_k:]
    top_signed = np.take_along_axis(signed, top_indices, axis=1)
    return np.maximum(top_signed, 0.0).sum(axis=1)


def _compute_rule_attribution(
    predictor: FrozenPredictor,
    matrix: np.ndarray,
    registry: pd.DataFrame,
    rule_names: Sequence[str],
    *,
    device: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], str | None]:
    rules = [str(value) for value in rule_names]
    try:
        result = compute_attributions(
            predictor,
            matrix,
            feature_names=predictor.feature_names,
            device=device,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        hit, share = _empty_attribution_tables(len(matrix), rules)
        share["__attribution_top_k_positive_evidence__"] = 0.0
        importance = pd.DataFrame(
            {"feature": list(predictor.feature_names), "mean_absolute_attribution": np.nan}
        )
        return hit, share, importance, {"available": False, "error": str(exc)}, str(exc)
    hit, share = _per_rule_attribution(result, registry, rules)
    # Native SHAP/pred_contribs and neural gradient-times-input are signed in
    # the raw fraud-logit direction.  The baseline therefore sums only the
    # positive contributions among the five largest absolute local effects.
    # Unlike the former concentration ratio, this is actual row-level fraud
    # evidence: a row with large negative attributions receives no support.
    share["__attribution_top_k_positive_evidence__"] = (
        _positive_top_k_attribution_evidence(result.values, top_k=5)
    )
    importance = pd.DataFrame(
        {
            "feature": list(result.feature_names),
            "mean_absolute_attribution": np.abs(result.values).mean(axis=0),
        }
    ).sort_values("mean_absolute_attribution", ascending=False)
    return hit, share, importance, {"available": True, **result.provenance}, None


def _complete_audit_without_survivors(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = audit.copy()
    completed["selection_status"] = "audit_failed"
    completed["redundant_with"] = None
    completed["max_jaccard_with_selected"] = np.nan
    completed["selected"] = False
    return completed, pd.DataFrame(columns=["rule_a", "rule_b", "jaccard"])


def _independent_ablation_rule_pool(
    audit_table: pd.DataFrame,
    audit_truth: pd.DataFrame,
    rule_names: Iterable[str],
    *,
    variant: str,
    pool_definition: str,
    audit_alert_mask: np.ndarray | Sequence[bool],
    activation_threshold: float,
    max_jaccard: float,
    max_rules: int,
    fp_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select and weight one ablation pool independently on rule-audit data.

    Tier ablations are complete alternative pipelines, not filters over the
    primary A/B survivors.  Starting from the pre-dedup audit table ensures a
    rule excluded by cross-tier redundancy or the primary rule budget can
    still enter a domain-only or Tier-B-only baseline.  The function has no
    test inputs and records enough provenance to audit that separation.
    """

    required = {
        "rule",
        "audit_pass",
        "audit_alert_score",
        "audit_alert_tp_activation",
        "audit_alert_fp_activation",
    }
    missing = required - set(audit_table.columns)
    if missing:
        raise KeyError(f"Ablation audit table is missing columns: {sorted(missing)}")
    if not str(variant).strip() or not str(pool_definition).strip():
        raise ValueError("Ablation variant and pool definition must be non-empty")

    requested_names = {str(value) for value in rule_names}
    audit_names = set(audit_table["rule"].astype(str))
    truth_names = set(audit_truth.columns.astype(str))
    unknown = requested_names - audit_names
    if unknown:
        raise KeyError(f"Ablation pool contains rules absent from the audit table: {sorted(unknown)}")
    missing_truth = requested_names - truth_names
    if missing_truth:
        raise KeyError(f"Ablation pool contains rules absent from audit truth: {sorted(missing_truth)}")

    subset_audit = audit_table.loc[
        audit_table["rule"].astype(str).isin(requested_names)
    ].copy()
    ordered_names = subset_audit["rule"].astype(str).tolist()
    subset_truth = audit_truth.loc[:, ordered_names].copy()
    if subset_audit["audit_pass"].astype(bool).any():
        completed, redundancy = deduplicate_audited_rules(
            subset_audit,
            subset_truth,
            audit_alert_mask=audit_alert_mask,
            activation_threshold=float(activation_threshold),
            max_jaccard=float(max_jaccard),
            max_rules=int(max_rules),
        )
    else:
        completed, redundancy = _complete_audit_without_survivors(subset_audit)
    weights = compute_rule_weights(completed, fp_penalty=float(fp_penalty))

    selected_rules = weights["rule"].astype(str).tolist() if not weights.empty else []
    provenance: dict[str, Any] = {
        "variant": str(variant),
        "pool_definition": str(pool_definition),
        "selection_scope": "independent_complete_pre_dedup_audit_pool",
        "candidate_source_table": "rule_audit_before_primary_deduplication",
        "fit_partition": "rule_audit",
        "candidate_rule_count": int(len(subset_audit)),
        "audit_passing_rule_count": int(subset_audit["audit_pass"].astype(bool).sum()),
        "selected_rule_count": int(len(selected_rules)),
        "selected_rules": selected_rules,
        "max_jaccard": float(max_jaccard),
        "max_rules": int(max_rules),
        "fp_penalty": float(fp_penalty),
        "deduplication_recomputed_within_pool": True,
        "weights_recomputed_within_pool": True,
        "primary_selected_weights_filtered": False,
        "test_labels_used": False,
    }
    completed = completed.copy()
    completed.insert(0, "variant", str(variant))
    completed.insert(1, "pool_definition", str(pool_definition))
    completed.insert(2, "selection_scope", provenance["selection_scope"])
    completed.insert(3, "fit_partition", provenance["fit_partition"])
    completed.insert(4, "test_labels_used", False)
    if not redundancy.empty:
        redundancy = redundancy.copy()
        redundancy.insert(0, "variant", str(variant))
    else:
        redundancy = pd.DataFrame(columns=["variant", "rule_a", "rule_b", "jaccard"])
    return completed, redundancy, weights, provenance


def _rule_audit_bootstrap_stability(
    train_truth: pd.DataFrame,
    audit_truth: pd.DataFrame,
    y_train: np.ndarray,
    y_audit: np.ndarray,
    registry: pd.DataFrame,
    attribution_hit: pd.DataFrame,
    attribution_share: pd.DataFrame,
    original_completed: pd.DataFrame,
    *,
    train_alert_mask: np.ndarray,
    audit_alert_mask: np.ndarray,
    activation_threshold: float,
    criteria: Mapping[str, Any],
    max_jaccard: float,
    max_rules: int,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Measure audited-rule survival under stratified validation resampling."""

    train_alerts = np.asarray(train_alert_mask, dtype=bool)
    audit_alerts = np.asarray(audit_alert_mask, dtype=bool)
    if len(train_alerts) != len(y_train) or len(audit_alerts) != len(y_audit):
        raise ValueError("Rule-audit stability alert masks must align with their partitions")
    positive = np.flatnonzero(audit_alerts & (y_audit == 1))
    negative = np.flatnonzero(audit_alerts & (y_audit == 0))
    if not len(positive) or not len(negative):
        return pd.DataFrame(
            [
                {
                    "seed": int(seed),
                    "selected_rule_count": 0,
                    "reference_selected_rule_count": int(
                        original_completed["selected"].astype(bool).sum()
                    ),
                    "surviving_reference_rule_count": 0,
                    "reference_rule_survival_rate": 0.0,
                    "selected_rule_jaccard": 0.0,
                    "selected_rules": "[]",
                    "mean_selected_audit_score": float("nan"),
                    "mean_selected_audit_alert_coverage": float("nan"),
                    "mean_selected_audit_alert_lift": float("nan"),
                    "audit_alert_sample_count": 0,
                    "stability_blocker": "insufficient_tp_or_fp_predictor_alerts",
                    "validation_or_test_refit": False,
                    "test_data_used": False,
                }
                for seed in seeds
            ]
        )
    original = set(
        original_completed.loc[original_completed["selected"].astype(bool), "rule"].astype(str)
    )
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
        audit = audit_rule_candidates(
            train_truth,
            audit_truth.iloc[sampled].reset_index(drop=True),
            y_train,
            y_audit[sampled],
            train_alert_mask=train_alerts,
            audit_alert_mask=np.ones(len(sampled), dtype=bool),
            registry=registry,
            attribution_hit=attribution_hit.iloc[sampled].reset_index(drop=True),
            attribution_share=attribution_share.iloc[sampled].reset_index(drop=True),
            activation_threshold=activation_threshold,
            criteria=criteria,
        )
        if audit["audit_pass"].astype(bool).any():
            completed, _ = deduplicate_audited_rules(
                audit,
                audit_truth.iloc[sampled].reset_index(drop=True),
                audit_alert_mask=np.ones(len(sampled), dtype=bool),
                activation_threshold=activation_threshold,
                max_jaccard=max_jaccard,
                max_rules=max_rules,
            )
        else:
            completed, _ = _complete_audit_without_survivors(audit)
        selected = set(completed.loc[completed["selected"].astype(bool), "rule"].astype(str))
        union = original | selected
        intersection = original & selected
        selected_rows = completed.loc[completed["selected"].astype(bool)]
        rows.append(
            {
                "seed": int(seed),
                "selected_rule_count": int(len(selected)),
                "reference_selected_rule_count": int(len(original)),
                "surviving_reference_rule_count": int(len(intersection)),
                "reference_rule_survival_rate": (
                    float(len(intersection) / len(original)) if original else float("nan")
                ),
                "selected_rule_jaccard": float(len(intersection) / len(union)) if union else 1.0,
                "selected_rules": json.dumps(sorted(selected), ensure_ascii=False),
                "mean_selected_audit_score": (
                    float(selected_rows["audit_alert_score"].mean())
                    if not selected_rows.empty
                    else float("nan")
                ),
                "mean_selected_audit_alert_coverage": (
                    float(selected_rows["audit_alert_coverage"].mean())
                    if not selected_rows.empty
                    else float("nan")
                ),
                "mean_selected_audit_alert_lift": (
                    float(selected_rows["audit_alert_lift"].mean())
                    if not selected_rows.empty
                    else float("nan")
                ),
                "audit_alert_sample_count": int(len(sampled)),
                "stability_blocker": None,
                "validation_or_test_refit": False,
                "test_data_used": False,
            }
        )
    return pd.DataFrame(rows)


def _evidence_methods(
    truth: pd.DataFrame,
    weights: pd.DataFrame,
    attribution_share: pd.DataFrame,
    methods: Sequence[str],
    *,
    activation_threshold: float,
) -> dict[str, np.ndarray]:
    selected_rules = [
        str(rule) for rule in weights.get("rule", pd.Series(dtype=str)).astype(str) if str(rule) in truth
    ]
    supported = (
        (truth[selected_rules].to_numpy(dtype=float) >= float(activation_threshold)).any(axis=1)
        if selected_rules
        else np.zeros(len(truth), dtype=bool)
    )
    values: dict[str, np.ndarray] = {}
    for method in methods:
        scores = build_evidence_scores(
            truth,
            weights,
            method=str(method),
            attribution_share=attribution_share if str(method) == "attribution_gated" else None,
        )
        # A selective rule explanation exists only when at least one audited
        # rule reaches the declared activation threshold.  Tiny positive soft
        # truth values must not convert unsupported alerts into explanations.
        values[str(method)] = np.where(supported, scores, 0.0)
    if not values:
        raise ValueError("logic.evidence_methods must contain at least one method")
    return values


def _add_guarded_ensemble(
    evidence_by_method: Mapping[str, np.ndarray],
    components: Sequence[str],
) -> dict[str, np.ndarray]:
    """Add one pre-registered, label-free ensemble to an evidence mapping.

    Component scores are all finite non-negative evidence values on compatible
    scales (rule aggregations and logistic probabilities).  The arithmetic
    mean is fixed in YAML before validation or test is opened; validation may
    select this candidate exactly like any individual method.  Because every
    component is already fail-closed outside active audited-rule support, the
    mean cannot manufacture explanations for unsupported rows.
    """

    values = {str(name): np.asarray(score, dtype=float) for name, score in evidence_by_method.items()}
    names = [str(value) for value in components]
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"Guarded ensemble components are unavailable: {missing}")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Guarded ensemble requires at least two unique components")
    lengths = {len(values[name]) for name in names}
    if len(lengths) != 1 or any(not np.isfinite(values[name]).all() for name in names):
        raise ValueError("Guarded ensemble components must be aligned and finite")
    values["guarded_ensemble"] = np.mean(
        np.column_stack([values[name] for name in names]), axis=1
    )
    return values


def _selection_signature(
    policies: Sequence[LockedSelectivePolicy],
    probabilities: np.ndarray,
    threshold: float,
    evidence: Mapping[str, np.ndarray],
) -> tuple[str, dict[float, np.ndarray]]:
    digest = hashlib.sha256()
    masks: dict[float, np.ndarray] = {}
    for policy in policies:
        mask = apply_locked_policy(policy, probabilities, threshold, evidence)
        masks[float(policy.coverage)] = mask
        digest.update(f"{policy.coverage:.12g}".encode("ascii"))
        digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
    return digest.hexdigest(), masks


def _negative_control_rows(
    policy: LockedSelectivePolicy,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selected: np.ndarray,
    score_only: np.ndarray,
    evidence: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    alerts = probabilities >= float(threshold)
    rng = np.random.default_rng(int(seed) + int(round(policy.coverage * 10_000)))
    permuted = np.asarray(evidence, dtype=float).copy()
    alert_indices = np.flatnonzero(alerts)
    permuted_alert_values = permuted[alert_indices].copy()
    rng.shuffle(permuted_alert_values)
    permuted[alert_indices] = permuted_alert_values
    random_evidence = rng.random(len(labels))
    # Null rankings must obey the same explanation-support constraint as the
    # locked policy.  Otherwise a control could call an alert "explained" even
    # when the permuted/random evidence has no active rule support.
    random_evidence = np.where(np.asarray(evidence, dtype=float) > 0.0, random_evidence, 0.0)
    realized_count = int(selected.sum())
    masks = {
        "locked_vasre": selected,
        "predictor_score_only": score_only,
        "row_permuted_evidence": fixed_count_mask(
            permuted,
            probabilities,
            alerts,
            realized_count,
            eligible_mask=permuted > 0.0,
        ),
        "random_evidence": fixed_count_mask(
            random_evidence,
            probabilities,
            alerts,
            realized_count,
            eligible_mask=random_evidence > 0.0,
        ),
    }
    rows: list[dict[str, Any]] = []
    requested_count = budget_count(int(alerts.sum()), policy.coverage)
    for control, mask in masks.items():
        precision = float(labels[mask].mean()) if mask.any() else float("nan")
        rows.append(
            {
                "coverage_budget": policy.coverage,
                "control": control,
                "alert_count": int(alerts.sum()),
                "requested_selected_count": int(requested_count),
                "matched_realized_selected_count": realized_count,
                "selected_count": int(mask.sum()),
                "equal_count_with_locked_vasre": int(mask.sum()) == realized_count,
                "achieved_alert_coverage": (
                    float(mask.sum() / alerts.sum()) if alerts.any() else float("nan")
                ),
                "selected_precision": precision,
                "uses_test_labels_for_ranking": False,
                "policy_abstained": policy.abstain,
            }
        )
    return rows


_MANDATORY_SHUFFLED_CONTROLS: tuple[str, str] = (
    "validation_label_shuffle_control",
    "validation_weight_shuffle_control",
)


def _shuffled_control_confirmation_diagnostics(
    coverage_result: Mapping[str, Any],
    shuffled_controls: Sequence[Mapping[str, Any]],
    *,
    material_precision_margin: float,
) -> dict[str, Any]:
    """Audit mandatory shuffled controls at the primary realized count.

    A shuffled control is only comparable when it selected exactly as many
    alerts as the locked primary policy and its own score-only comparator used
    that same count. Missing, duplicate, abstaining, non-finite or count-
    mismatched controls fail closed; unsupported alerts are never inserted to
    manufacture equal coverage.
    """

    primary_count = int(coverage_result.get("selected_count", 0))
    primary_precision = float(
        coverage_result.get(
            "selected_alert_precision",
            coverage_result.get("selected_precision", float("nan")),
        )
    )
    rows = [dict(row) for row in shuffled_controls]
    blockers: list[str] = []
    complete = True
    equal_count = primary_count > 0
    finite = np.isfinite(primary_precision)
    control_precisions: list[float] = []
    control_deltas: list[float] = []
    details: dict[str, Any] = {}

    if primary_count <= 0:
        blockers.append("primary_realized_count_is_zero")
    if not np.isfinite(primary_precision):
        blockers.append("primary_precision_nonfinite")

    for variant in _MANDATORY_SHUFFLED_CONTROLS:
        matches = [row for row in rows if str(row.get("variant", "")) == variant]
        prefix = variant.removesuffix("_control")
        details[f"{prefix}_row_count"] = len(matches)
        if len(matches) != 1:
            complete = False
            equal_count = False
            finite = False
            blockers.append(f"{variant}_missing_or_duplicate")
            details[f"{prefix}_selected_count"] = -1
            details[f"{prefix}_score_only_selected_count"] = -1
            details[f"{prefix}_selected_precision"] = float("nan")
            details[f"{prefix}_delta_vs_score_only"] = float("nan")
            continue

        row = matches[0]
        selected_count = int(row.get("selected_count", -1))
        score_count = int(row.get("score_only_selected_count", -1))
        precision = float(
            row.get(
                "selected_alert_precision",
                row.get("selected_precision", float("nan")),
            )
        )
        delta = float(row.get("delta_vs_score_only", float("nan")))
        abstained = bool(row.get("abstain", False))
        details[f"{prefix}_selected_count"] = selected_count
        details[f"{prefix}_score_only_selected_count"] = score_count
        details[f"{prefix}_selected_precision"] = precision
        details[f"{prefix}_delta_vs_score_only"] = delta

        row_equal = selected_count == primary_count and score_count == primary_count
        row_finite = np.isfinite(precision) and np.isfinite(delta) and not abstained
        equal_count = equal_count and row_equal
        finite = finite and row_finite
        if not row_equal:
            blockers.append(f"{variant}_realized_count_mismatch")
        if not row_finite:
            blockers.append(f"{variant}_abstained_or_nonfinite")
        if row_finite:
            control_precisions.append(precision)
            control_deltas.append(delta)

    comparison_valid = bool(complete and equal_count and finite)
    maximum_precision = (
        float(max(control_precisions))
        if len(control_precisions) == len(_MANDATORY_SHUFFLED_CONTROLS)
        else float("nan")
    )
    maximum_delta = (
        float(max(control_deltas))
        if len(control_deltas) == len(_MANDATORY_SHUFFLED_CONTROLS)
        else float("nan")
    )
    below_primary = bool(
        comparison_valid
        and np.isfinite(maximum_precision)
        and primary_precision > maximum_precision
    )
    no_material_gain = bool(
        comparison_valid
        and np.isfinite(maximum_delta)
        and maximum_delta <= float(material_precision_margin)
    )
    if comparison_valid and not below_primary:
        blockers.append("shuffled_control_equivalent_to_or_better_than_primary")
    if comparison_valid and not no_material_gain:
        blockers.append("shuffled_control_has_material_positive_gain")

    return {
        "mandatory_shuffled_control_names": ";".join(_MANDATORY_SHUFFLED_CONTROLS),
        "shuffled_control_primary_realized_count": primary_count,
        "shuffled_control_primary_selected_precision": primary_precision,
        "shuffled_control_rows_complete": bool(complete),
        "shuffled_control_counts_match_primary": bool(equal_count),
        "shuffled_control_metrics_finite": bool(finite),
        "shuffled_control_comparison_valid": comparison_valid,
        "shuffled_control_max_selected_precision": maximum_precision,
        "shuffled_control_max_delta_vs_score_only": maximum_delta,
        "shuffled_controls_below_locked_primary": below_primary,
        "shuffled_controls_no_material_gain": no_material_gain,
        "shuffled_control_blockers": ";".join(dict.fromkeys(blockers)),
        **details,
    }


def _conservative_stress_outcome(
    coverage_result: Mapping[str, Any],
    bootstrap_result: Mapping[str, Any],
    matched_result: Mapping[str, Any],
    residual_result: Mapping[str, Any],
    negative_controls: Sequence[Mapping[str, Any]],
    stability_rows: pd.DataFrame,
    *,
    shuffled_controls: Sequence[Mapping[str, Any]] = (),
    saturation_headroom: float,
    material_precision_margin: float = 0.02,
    material_evidence_margin: float = 0.05,
    minimum_positive_stability_fraction: float = 0.80,
    saturation_scope_eligible: bool = True,
    minimum_saturation_alert_positives: int = 20,
    minimum_saturation_matched_pairs: int = 20,
    minimum_saturation_bootstrap_replicates: int = 200,
) -> tuple[str, dict[str, Any]]:
    """Require convergent diagnostics for positive and saturation claims.

    Low score-only headroom by itself is descriptive, not proof that the rule
    layer has saturated.  Confirmation additionally requires no material gain
    under the paired temporal bootstrap, matched-risk comparison, residual
    evidence check, negative controls, and validation resampling.  Missing
    diagnostics fail closed to ``low_score_only_headroom_unconfirmed``.

    Likewise, a positive paired-bootstrap interval is necessary but not
    sufficient for a rule-added-value label. Matched-risk, residual evidence,
    validation stability, and negative controls must point in the same
    direction; otherwise the result is explicitly inconclusive.
    """

    decision_thresholds = {
        "saturation_headroom_threshold": float(saturation_headroom),
        "saturation_precision_margin": float(material_precision_margin),
        "saturation_evidence_margin": float(material_evidence_margin),
        "minimum_positive_stability_fraction": float(
            minimum_positive_stability_fraction
        ),
        "minimum_saturation_alert_positives": int(minimum_saturation_alert_positives),
        "minimum_saturation_matched_pairs": int(minimum_saturation_matched_pairs),
        "minimum_saturation_bootstrap_replicates": int(
            minimum_saturation_bootstrap_replicates
        ),
    }
    preliminary = classify_stress_outcome(
        dict(coverage_result),
        dict(bootstrap_result),
        saturation_headroom=float(saturation_headroom),
    )
    core_equal_count_comparison = bool(
        coverage_result.get("equal_count_score_only_comparison", False)
        and int(coverage_result.get("selected_count", -1))
        == int(coverage_result.get("score_only_selected_count", -2))
    )
    shuffled_diagnostics = _shuffled_control_confirmation_diagnostics(
        coverage_result,
        shuffled_controls,
        material_precision_margin=float(material_precision_margin),
    )

    negative = pd.DataFrame(list(negative_controls))
    if negative.empty:
        null_control_supports_positive = False
        negative_consistent_with_no_gain = False
        negative_control_count_fair = False
        locked_precision = float("nan")
        maximum_null_precision = float("nan")
    else:
        locked_rows = negative.loc[negative["control"] == "locked_vasre"]
        score_rows = negative.loc[negative["control"] == "predictor_score_only"]
        null_rows = negative.loc[
            negative["control"].isin(["row_permuted_evidence", "random_evidence"])
        ]
        locked_values = pd.to_numeric(
            locked_rows.get("selected_precision", pd.Series(dtype=float)), errors="coerce"
        )
        score_values = pd.to_numeric(
            score_rows.get("selected_precision", pd.Series(dtype=float)), errors="coerce"
        )
        null_values = pd.to_numeric(
            null_rows.get("selected_precision", pd.Series(dtype=float)), errors="coerce"
        )
        locked_precision = (
            float(locked_values.iloc[0])
            if len(locked_values) and np.isfinite(float(locked_values.iloc[0]))
            else float("nan")
        )
        maximum_null_precision = (
            float(null_values.max())
            if len(null_values) and null_values.notna().all()
            else float("nan")
        )
        locked_counts = pd.to_numeric(
            locked_rows.get("selected_count", pd.Series(dtype=float)), errors="coerce"
        )
        comparison_counts = pd.to_numeric(
            negative.loc[
                negative["control"].isin(
                    [
                        "predictor_score_only",
                        "row_permuted_evidence",
                        "random_evidence",
                    ]
                ),
                "selected_count",
            ],
            errors="coerce",
        )
        negative_control_count_fair = bool(
            len(locked_counts) == 1
            and np.isfinite(float(locked_counts.iloc[0]))
            and len(comparison_counts) == 3
            and comparison_counts.notna().all()
            and (comparison_counts == float(locked_counts.iloc[0])).all()
        )
        null_control_supports_positive = bool(
            negative_control_count_fair
            and np.isfinite(locked_precision)
            and np.isfinite(maximum_null_precision)
            and locked_precision > maximum_null_precision
        )
        if len(score_values) and np.isfinite(float(score_values.iloc[0])):
            score_precision = float(score_values.iloc[0])
            negative_consistent_with_no_gain = bool(
                negative_control_count_fair
                and len(null_values)
                and null_values.notna().all()
                and (null_values <= score_precision + material_precision_margin).all()
            )
        else:
            negative_consistent_with_no_gain = False

    if preliminary == "positive_incremental_evidence":
        bootstrap_low = float(bootstrap_result.get("ci_low", float("nan")))
        matched_low = float(matched_result.get("matched_ci_low", float("nan")))
        residual_gap = float(residual_result.get("risk_conditioned_tp_fp_gap", float("nan")))
        bootstrap_supports_positive = np.isfinite(bootstrap_low) and bootstrap_low > 0.0
        matched_supports_positive = np.isfinite(matched_low) and matched_low > 0.0
        residual_supports_positive = np.isfinite(residual_gap) and residual_gap > 0.0

        if stability_rows.empty:
            positive_stability_fraction = float("nan")
            stability_supports_positive = False
        else:
            delta_column = (
                "full_validation_delta_vs_score_only"
                if "full_validation_delta_vs_score_only" in stability_rows
                else "validation_delta_vs_score_only"
            )
            if delta_column not in stability_rows:
                positive_stability_fraction = float("nan")
                stability_supports_positive = False
            else:
                deltas = pd.to_numeric(stability_rows[delta_column], errors="coerce")
                method_matches = (
                    stability_rows["method_matches_reference"].astype(bool)
                    if "method_matches_reference" in stability_rows
                    else pd.Series(False, index=stability_rows.index)
                )
                non_abstaining = (
                    ~stability_rows["abstain"].astype(bool)
                    if "abstain" in stability_rows
                    else pd.Series(False, index=stability_rows.index)
                )
                support = deltas.notna() & deltas.gt(0.0) & method_matches & non_abstaining
                positive_stability_fraction = float(support.mean())
                stability_supports_positive = bool(
                    positive_stability_fraction >= minimum_positive_stability_fraction
                )

        confirmations = {
            "core_equal_count_score_only_comparison": core_equal_count_comparison,
            "paired_bootstrap_supports_positive": bool(bootstrap_supports_positive),
            "matched_risk_supports_positive": bool(matched_supports_positive),
            "residual_evidence_supports_positive": bool(residual_supports_positive),
            "negative_controls_support_positive": bool(null_control_supports_positive),
            "mandatory_shuffled_controls_support_positive": bool(
                shuffled_diagnostics["shuffled_control_comparison_valid"]
                and shuffled_diagnostics["shuffled_controls_below_locked_primary"]
            ),
            "validation_stability_supports_positive": bool(stability_supports_positive),
        }
        confirmed = all(confirmations.values())
        return (
            "positive_incremental_evidence"
            if confirmed
            else "inconclusive_incremental_evidence"
        ), {
            "preliminary_outcome": preliminary,
            "positive_confirmation_required": True,
            "positive_incremental_evidence_confirmed": bool(confirmed),
            "positive_stability_fraction": positive_stability_fraction,
            "negative_control_locked_precision": locked_precision,
            "negative_control_maximum_null_precision": maximum_null_precision,
            "saturation_confirmation_required": False,
            "saturation_confirmed": False,
            **decision_thresholds,
            "core_equal_count_score_only_comparison": core_equal_count_comparison,
            "negative_control_count_fair": bool(negative_control_count_fair),
            **shuffled_diagnostics,
            **confirmations,
        }

    if "saturation" not in preliminary:
        return preliminary, {
            "preliminary_outcome": preliminary,
            "positive_confirmation_required": False,
            "positive_incremental_evidence_confirmed": False,
            "saturation_confirmation_required": False,
            "saturation_confirmed": False,
            **decision_thresholds,
            "core_equal_count_score_only_comparison": core_equal_count_comparison,
            "negative_control_count_fair": bool(negative_control_count_fair),
            **shuffled_diagnostics,
        }

    alert_positives = int(coverage_result.get("alert_positive_count", 0))
    matched_pairs = int(matched_result.get("matched_pairs", 0))
    valid_bootstrap = int(bootstrap_result.get("valid_iterations", 0))
    saturation_sample_adequate = bool(
        alert_positives >= int(minimum_saturation_alert_positives)
        and matched_pairs >= int(minimum_saturation_matched_pairs)
        and valid_bootstrap >= int(minimum_saturation_bootstrap_replicates)
    )
    if not bool(saturation_scope_eligible) or not saturation_sample_adequate:
        return "low_score_only_headroom_unconfirmed", {
            "preliminary_outcome": preliminary,
            "positive_confirmation_required": False,
            "positive_incremental_evidence_confirmed": False,
            "saturation_confirmation_required": True,
            "saturation_scope_eligible": bool(saturation_scope_eligible),
            "saturation_sample_adequate": saturation_sample_adequate,
            "saturation_alert_positive_count": alert_positives,
            "saturation_matched_pair_count": matched_pairs,
            "saturation_valid_bootstrap_replicates": valid_bootstrap,
            "saturation_confirmed": False,
            **decision_thresholds,
            "core_equal_count_score_only_comparison": core_equal_count_comparison,
            "negative_control_count_fair": bool(negative_control_count_fair),
            **shuffled_diagnostics,
        }

    bootstrap_high = float(bootstrap_result.get("ci_high", float("nan")))
    matched_high = float(matched_result.get("matched_ci_high", float("nan")))
    residual_gap = float(residual_result.get("risk_conditioned_tp_fp_gap", float("nan")))
    bootstrap_no_gain = np.isfinite(bootstrap_high) and bootstrap_high <= material_precision_margin
    matched_no_gain = np.isfinite(matched_high) and matched_high <= material_precision_margin
    residual_no_gain = np.isfinite(residual_gap) and abs(residual_gap) <= material_evidence_margin

    if stability_rows.empty or "validation_delta_vs_score_only" not in stability_rows:
        stability_no_gain = False
    else:
        deltas = pd.to_numeric(
            stability_rows["validation_delta_vs_score_only"], errors="coerce"
        )
        stability_no_gain = bool(
            deltas.notna().any() and float(deltas.dropna().max()) <= material_precision_margin
        )
        if stability_rows["abstain"].astype(bool).all():
            stability_no_gain = True

    confirmations = {
        "core_equal_count_score_only_comparison": core_equal_count_comparison,
        "paired_bootstrap_no_material_gain": bool(bootstrap_no_gain),
        "matched_risk_no_material_gain": bool(matched_no_gain),
        "residual_evidence_no_material_gain": bool(residual_no_gain),
        "negative_controls_consistent": bool(negative_consistent_with_no_gain),
        "mandatory_shuffled_controls_consistent_with_no_gain": bool(
            shuffled_diagnostics["shuffled_control_comparison_valid"]
            and shuffled_diagnostics["shuffled_controls_no_material_gain"]
        ),
        "validation_stability_no_material_gain": bool(stability_no_gain),
    }
    confirmed = all(confirmations.values())
    if confirmed:
        outcome = (
            "confirmed_saturation_with_abstention"
            if bool(coverage_result.get("abstain", False))
            else "confirmed_score_only_saturation"
        )
    else:
        outcome = "low_score_only_headroom_unconfirmed"
    return outcome, {
        "preliminary_outcome": preliminary,
        "positive_confirmation_required": False,
        "positive_incremental_evidence_confirmed": False,
        "saturation_confirmation_required": True,
        "saturation_scope_eligible": bool(saturation_scope_eligible),
        "saturation_sample_adequate": saturation_sample_adequate,
        "saturation_alert_positive_count": alert_positives,
        "saturation_matched_pair_count": matched_pairs,
        "saturation_valid_bootstrap_replicates": valid_bootstrap,
        "saturation_confirmed": confirmed,
        **decision_thresholds,
        "negative_control_count_fair": bool(negative_control_count_fair),
        **shuffled_diagnostics,
        **confirmations,
    }


def _claim_assessment(
    dataset: StressDataset,
    candidates: Sequence[_CandidateState],
    *,
    quick_run: bool,
    use_test_fixture: bool,
    attribution_errors: Sequence[str],
    policy_locked_before_test: bool,
) -> tuple[bool, list[str], dict[str, bool]]:
    blockers: list[str] = []
    if quick_run:
        blockers.append("quick_run_not_for_thesis_claims")
    if use_test_fixture:
        blockers.append("generated_test_fixture_not_for_thesis_claims")
    if bool(dataset.manifest.get("quick_sample", {}).get("enabled", False)):
        blockers.append("partial_dataset_sample_not_for_thesis_claims")
    source_files = dataset.manifest.get("source_files", [])
    checksum_verified = bool(source_files) and all(bool(item.get("verified")) for item in source_files)
    if not checksum_verified:
        blockers.append("canonical_dataset_checksum_not_verified")
    fallback_used = any(
        bool(state.frozen.provenance.get("quick_fallback_used", False)) for state in candidates
    )
    if fallback_used:
        blockers.append("quick_fixture_model_backend_substitution_used")
    if attribution_errors:
        blockers.append("required_predictor_attribution_backend_unavailable")
    if not policy_locked_before_test:
        blockers.append("policy_was_not_locked_before_test")
    flags = {
        "checksum_verified": checksum_verified,
        "full_dataset": not bool(dataset.manifest.get("quick_sample", {}).get("enabled", False))
        and not use_test_fixture,
        "policy_locked_before_test": bool(policy_locked_before_test),
        "exact_predictor_backends": not fallback_used,
        "reference_attributions_available": not attribution_errors,
        "all_predictor_attributions_available": not attribution_errors,
    }
    return not blockers, list(dict.fromkeys(blockers)), flags


def run_stress_test(
    config_path: str | Path,
    *,
    data_roots: str | Path | Iterable[str | Path] | None = None,
    output_dir: str | Path | None = None,
    quick_run: bool = False,
    use_test_fixture: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Run one complete, proposal-aligned TransXion or AMLNet stress test.

    Parameters
    ----------
    config_path:
        Repository YAML for exactly one stress dataset.
    data_roots:
        One or more roots searched recursively for the exact canonical file.
    output_dir:
        Optional artifact directory override.
    quick_run / use_test_fixture:
        Explicit smoke-test switches.  Both are permanently recorded as claim
        blockers; no synthetic fallback is available in full mode.
    device:
        Requested Torch device.  Quick runs always use CPU.
    """

    config_file = Path(config_path).resolve()
    config = load_yaml_config(config_file)
    _validate_protocol_config(config)
    project_root = _find_project_root(config_file)
    dataset_name = _normalise_dataset_name(config)
    supported = {"transxion", "transxion_v2", "amlnet", "amlnet_v1", "amlnet_v1_0"}
    if dataset_name not in supported:
        raise ValueError(f"Unsupported stress dataset: {dataset_name!r}")
    if use_test_fixture and not quick_run:
        raise ValueError("use_test_fixture=True requires quick_run=True")

    configured_output = config.get("project", {}).get("output_dir", f"results/stress/{dataset_name}")
    destination = Path(output_dir) if output_dir is not None else Path(configured_output)
    if not destination.is_absolute():
        destination = project_root / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    effective_roots = data_roots
    verify_checksum: bool | None = None
    if use_test_fixture:
        fixture_root = destination / ".stress_test_fixture"
        fixture_root.mkdir(parents=True, exist_ok=True)
        fixture_rows = int(config.get("testing", {}).get("fixture_rows", 800))
        fixture_seed = int(config.get("project", {}).get("seed", 42))
        if dataset_name.startswith("transxion"):
            make_synthetic_transxion_test_fixture(fixture_root, n_rows=fixture_rows, seed=fixture_seed)
        else:
            make_synthetic_amlnet_test_fixture(fixture_root, n_rows=fixture_rows, seed=fixture_seed)
        effective_roots = [fixture_root]
        verify_checksum = False

    ledger = _ProtocolLedger()
    quick_rows = config.get("testing", {}).get("quick_rows") if quick_run and not use_test_fixture else None
    dataset = load_stress_dataset(
        config,
        effective_roots,
        verify_checksum=verify_checksum,
        quick_rows=int(quick_rows) if quick_rows is not None else None,
    )
    ledger.advance(
        "data_loaded",
        dataset=dataset.manifest["dataset"],
        rows=len(dataset.frame),
        checksum_verification_requested=not use_test_fixture,
    )
    split = temporal_stress_split(dataset.frame, config)
    ledger.advance(
        "temporal_split_locked",
        strategy=split.manifest["strategy"],
        validation_roles=["calibration_fit", "calibration_select", "rule_audit", "policy_select"],
        test_labels_used_for_split_integrity_only=True,
        test_labels_used_for_model_rule_or_policy_selection=False,
    )

    preprocessor = _preprocessor_from_config(config, dataset.leakage_denylist)
    X_train = preprocessor.fit_transform(_feature_frame(split.train, dataset))
    feature_names = tuple(preprocessor.get_feature_names_out())
    _assert_no_leakage_features(feature_names, dataset.leakage_denylist)
    matrices: dict[str, np.ndarray] = {
        "train": X_train,
        "calibration_fit": preprocessor.transform(_feature_frame(split.calibration_fit, dataset)),
        "calibration_select": preprocessor.transform(_feature_frame(split.calibration_select, dataset)),
        "rule_audit": preprocessor.transform(_feature_frame(split.rule_audit, dataset)),
        "policy_select": preprocessor.transform(_feature_frame(split.policy_select, dataset)),
    }
    pretest_labels = {
        "train": _labels(split.train),
        "calibration_fit": _labels(split.calibration_fit),
        "calibration_select": _labels(split.calibration_select),
        "rule_audit": _labels(split.rule_audit),
        "policy_select": _labels(split.policy_select),
    }
    ledger.advance(
        "preprocessor_fitted_on_train",
        selected_feature_count=len(feature_names),
        selected_features=list(feature_names),
        leakage_denylist_asserted=True,
    )

    candidates, reference_index, validation_metrics, calibration_table = _train_and_select_reference(
        config,
        matrices,
        pretest_labels,
        feature_names,
        quick_run=quick_run,
        device=device,
    )
    ledger.advance(
        "predictors_trained_on_train",
        candidate_count=len(candidates),
        families=sorted({state.family for state in candidates}),
        seeds=sorted({state.seed for state in candidates}),
    )
    ledger.advance(
        "calibration_and_threshold_frozen",
        calibration_source=(
            "calibrator_fit=calibration_fit; method_and_threshold_selection=calibration_select"
        ),
        candidates=len(candidates),
    )
    reference = candidates[reference_index]
    ledger.advance(
        "reference_predictor_selected",
        selection_partition="calibration_select",
        selection_metric="pr_auc",
        family=reference.family,
        backend=reference.frozen.backend,
        seed=reference.seed,
    )

    logic = dict(config.get("logic", {}))
    rule_definitions = list(logic.get("rules", []))
    if not rule_definitions:
        raise ValueError("logic.rules must contain pre-registered or train-fitted candidates")

    counterfactual_settings = dict(config.get("counterfactual_candidates", {}))
    if bool(counterfactual_settings.get("enabled", True)):
        counterfactual_train = split.train.copy()
        # Pandas 3 rejects assigning a non-integral training median into an
        # integer column. Counterfactual perturbations are numerical by
        # definition, so use an explicit float workspace without changing the
        # canonical frame or the train-fitted preprocessor.
        for feature in counterfactual_settings.get("numeric_features", []):
            if feature in counterfactual_train:
                counterfactual_train[feature] = pd.to_numeric(
                    counterfactual_train[feature], errors="coerce"
                ).astype(float)
        counterfactual_result = generate_counterfactual_numeric_candidates(
            counterfactual_train,
            preprocessor,
            reference.frozen,
            numeric_features=counterfactual_settings.get("numeric_features"),
            derived_feature_contract=counterfactual_settings.get(
                "derived_feature_contract"
            ),
            non_intervenable_derived_features=counterfactual_settings.get(
                "non_intervenable_derived_features"
            ),
            quantile=float(counterfactual_settings.get("rule_quantile", 0.95)),
            softness=float(counterfactual_settings.get("softness", 0.15)),
            min_mean_score_drop=float(counterfactual_settings.get("minimum_mean_score_drop", 0.002)),
            min_affected_fraction=float(counterfactual_settings.get("minimum_affected_fraction", 0.05)),
            min_tail_rows=int(counterfactual_settings.get("minimum_tail_rows", 25)),
            max_candidates=int(counterfactual_settings.get("max_rules", 5)),
            max_rows=(
                min(int(counterfactual_settings.get("max_rows", 100_000)), 2_000)
                if quick_run
                else int(counterfactual_settings.get("max_rows", 100_000))
            ),
            score_mode=str(counterfactual_settings.get("score_mode", "raw")),
            device="cpu" if quick_run else device,
        )
    else:
        counterfactual_result = None
    counterfactual_definitions = (
        list(counterfactual_result.definitions) if counterfactual_result is not None else []
    )
    configured_result = fit_configured_candidates(
        split.train,
        "label",
        [*rule_definitions, *counterfactual_definitions],
    )
    counterfactual_names = {str(item["name"]) for item in counterfactual_definitions}
    configured_registry = configured_result.registry.loc[
        ~configured_result.registry["rule"].astype(str).isin(counterfactual_names)
    ].copy()
    primary_registries = [configured_registry]
    if counterfactual_result is not None:
        primary_registries.append(counterfactual_result.registry)

    surrogate_settings = dict(config.get("surrogate", {}))
    surrogate_result = None
    if bool(surrogate_settings.get("enabled", True)):
        surrogate_result = fit_cart_surrogate_candidates(
            matrices["train"],
            reference.frozen,
            feature_names=feature_names,
            surrogate_target=str(surrogate_settings.get("target", "score")),
            score_mode=str(surrogate_settings.get("score_mode", "calibrated")),
            max_depth=int(surrogate_settings.get("max_depth", 3)),
            min_samples_leaf=surrogate_settings.get("min_samples_leaf", 100),
            max_rules=int(surrogate_settings.get("max_rules", 8)),
            random_state=int(config.get("project", {}).get("seed", 42)),
            device="cpu" if quick_run else device,
        )
    # Tier C is a score-surrogate diagnostic/baseline.  It is deliberately not
    # admitted to the primary A/B evidence pool because doing so could let a
    # shallow reconstruction of predictor score drive the claimed rule layer.
    evidence_registry = combine_candidate_registries(*primary_registries)
    reporting_registries = [evidence_registry]
    if surrogate_result is not None:
        reporting_registries.append(surrogate_result.registry)
    registry = combine_candidate_registries(
        *reporting_registries,
        score_only_baseline_registry(reference.frozen),
    )

    train_truth = configured_result.evaluate(split.train).reset_index(drop=True)
    audit_truth = configured_result.evaluate(split.rule_audit).reset_index(drop=True)
    policy_truth = configured_result.evaluate(split.policy_select).reset_index(drop=True)
    evidence_registry = evidence_registry.loc[
        evidence_registry["rule"].astype(str).isin(train_truth.columns.astype(str))
    ].reset_index(drop=True)
    candidate_provenance = {
        "configured_and_counterfactual": configured_result.provenance,
        "counterfactual": (
            counterfactual_result.provenance if counterfactual_result is not None else {"enabled": False}
        ),
        "surrogate": surrogate_result.provenance if surrogate_result is not None else {"enabled": False},
        "surrogate_tier_c_role": "diagnostic_baseline_only",
        "surrogate_admitted_to_primary_evidence": False,
        "score_only_baseline_is_rule_evidence": False,
    }
    ledger.advance(
        "rules_fitted_on_train",
        fitted_rule_count=len(train_truth.columns),
        configured_and_counterfactual_count=len(configured_result.engine.rules),
        cart_surrogate_count=(len(surrogate_result.rules) if surrogate_result is not None else 0),
        cart_surrogate_primary_evidence=False,
        skipped_rules=configured_result.engine.skipped_rules,
        score_only_baseline_registered_but_not_a_rule=True,
    )

    audit_hit, audit_share, audit_importance, audit_attribution, audit_error = _compute_rule_attribution(
        reference.frozen,
        matrices["rule_audit"],
        evidence_registry,
        list(audit_truth.columns),
        device="cpu" if quick_run else device,
    )
    activation_threshold = float(logic.get("activation_threshold", 0.60))
    reference_train_probabilities = predict_probabilities(
        reference.frozen,
        matrices["train"],
        calibrated=True,
        device=device,
    )
    reference_audit_probabilities = predict_probabilities(
        reference.frozen,
        matrices["rule_audit"],
        calibrated=True,
        device=device,
    )
    train_alert_mask = reference_train_probabilities >= reference.frozen.threshold
    audit_alert_mask = reference_audit_probabilities >= reference.frozen.threshold
    audit = audit_rule_candidates(
        train_truth,
        audit_truth,
        pretest_labels["train"],
        pretest_labels["rule_audit"],
        train_alert_mask=train_alert_mask,
        audit_alert_mask=audit_alert_mask,
        registry=evidence_registry,
        attribution_hit=audit_hit,
        attribution_share=audit_share,
        activation_threshold=activation_threshold,
        criteria=logic.get("audit_criteria", {}),
    )
    max_jaccard = float(logic.get("max_jaccard", 0.85))
    max_rules = int(logic.get("max_rules", 12))
    fp_penalty = float(logic.get("fp_penalty", 1.0))
    if audit["audit_pass"].astype(bool).any():
        completed_audit, redundancy = deduplicate_audited_rules(
            audit,
            audit_truth,
            audit_alert_mask=audit_alert_mask,
            activation_threshold=activation_threshold,
            max_jaccard=max_jaccard,
            max_rules=max_rules,
        )
    else:
        completed_audit, redundancy = _complete_audit_without_survivors(audit)
    weights = compute_rule_weights(completed_audit, fp_penalty=fp_penalty)
    contrastive_settings = dict(logic["contrastive_meta"])

    def fit_contrastive_for(
        variant_weights: pd.DataFrame,
        audit_labels: np.ndarray,
        audit_probabilities: np.ndarray,
        threshold: float,
        *,
        seed_offset: int = 0,
    ) -> ContrastiveMetaEvidence:
        rule_names = (
            variant_weights["rule"].astype(str).tolist()
            if not variant_weights.empty
            else []
        )
        return fit_contrastive_meta_evidence(
            audit_truth.loc[:, rule_names],
            audit_labels,
            audit_probabilities,
            threshold,
            activation_threshold=activation_threshold,
            seed=int(config.get("project", {}).get("seed", 42)) + int(seed_offset),
            min_alert_rows=int(contrastive_settings["minimum_alert_rows"]),
            min_class_rows=int(contrastive_settings["minimum_rows_per_class"]),
            regularization_c=float(contrastive_settings["regularization_c"]),
            max_iter=int(contrastive_settings["max_iter"]),
        )

    contrastive_meta = fit_contrastive_for(
        weights,
        pretest_labels["rule_audit"],
        reference_audit_probabilities,
        reference.frozen.threshold,
    )
    stability_seeds = [
        int(value)
        for value in config.get("evaluation", {}).get(
            "policy_resampling_seeds", [42, 123, 2026]
        )
    ]
    rule_audit_stability = _rule_audit_bootstrap_stability(
        train_truth,
        audit_truth,
        pretest_labels["train"],
        pretest_labels["rule_audit"],
        evidence_registry,
        audit_hit,
        audit_share,
        completed_audit,
        train_alert_mask=train_alert_mask,
        audit_alert_mask=audit_alert_mask,
        activation_threshold=activation_threshold,
        criteria=logic.get("audit_criteria", {}),
        max_jaccard=max_jaccard,
        max_rules=max_rules,
        seeds=stability_seeds,
    )
    ledger.advance(
        "rules_audited_on_reserved_validation",
        audit_partition="rule_audit",
        audit_scope="frozen_predictor_alerts_only",
        train_alert_count=int(train_alert_mask.sum()),
        rule_audit_alert_count=int(audit_alert_mask.sum()),
        rule_audit_tp_alert_count=int(
            (audit_alert_mask & (pretest_labels["rule_audit"] == 1)).sum()
        ),
        rule_audit_fp_alert_count=int(
            (audit_alert_mask & (pretest_labels["rule_audit"] == 0)).sum()
        ),
        audited_rule_count=len(audit),
        selected_rule_count=int(completed_audit["selected"].sum()),
    )

    policy_probabilities = predict_probabilities(
        reference.frozen, matrices["policy_select"], calibrated=True, device=device
    )
    policy_hit, policy_share, policy_importance, policy_attribution, policy_error = _compute_rule_attribution(
        reference.frozen,
        matrices["policy_select"],
        evidence_registry,
        list(policy_truth.columns),
        device="cpu" if quick_run else device,
    )
    del policy_hit  # only per-rule attribution shares are required by the locked methods
    evidence_method_names = [
        str(value) for value in logic.get("evidence_methods", ["unweighted"])
    ]
    guarded_components = [
        str(value) for value in logic["guarded_ensemble"]["components"]
    ]
    base_evidence_method_names = [
        name
        for name in evidence_method_names
        if name not in {"contrastive_meta", "guarded_ensemble"}
    ]
    policy_evidence = _evidence_methods(
        policy_truth,
        weights,
        policy_share,
        base_evidence_method_names,
        activation_threshold=activation_threshold,
    )
    policy_evidence["contrastive_meta"] = contrastive_meta.transform(
        policy_truth, policy_probabilities
    )
    policy_evidence = _add_guarded_ensemble(policy_evidence, guarded_components)
    evaluation = dict(config.get("evaluation", {}))
    policy_guardrails = dict(logic.get("policy_guardrails", {}))
    coverages = [float(value) for value in evaluation.get("coverages", [0.05, 0.10, 0.25, 0.50])]
    minimum_fidelity = policy_guardrails.get("minimum_positive_region_fidelity")
    minimum_fidelity = float(minimum_fidelity) if minimum_fidelity is not None else None
    minimum_lock_stability = float(
        policy_guardrails.get("minimum_policy_stability_fraction", 0.60)
    )

    def lock_with_stability_guard(
        evidence: Mapping[str, np.ndarray],
        *,
        predictor_label: str,
    ) -> tuple[list[LockedSelectivePolicy], pd.DataFrame, pd.DataFrame]:
        locked, candidates_table = lock_selective_policies(
            pretest_labels["policy_select"],
            policy_probabilities,
            reference.frozen.threshold,
            evidence,
            dataset=dataset.manifest["dataset"],
            predictor=predictor_label,
            coverages=coverages,
            minimum_score_only_delta=float(evaluation.get("minimum_score_only_delta", 0.0)),
            minimum_supported_alert_fraction=float(
                policy_guardrails.get("minimum_supported_alert_fraction", 0.01)
            ),
            minimum_evidence_tp_fp_gap=float(
                policy_guardrails.get("minimum_evidence_tp_fp_gap", 0.0)
            ),
            minimum_positive_region_fidelity=minimum_fidelity,
            require_full_budget_support=bool(
                policy_guardrails.get("require_full_budget_support", True)
            ),
        )
        stability_table = validation_policy_stability(
            pretest_labels["policy_select"],
            policy_probabilities,
            reference.frozen.threshold,
            evidence,
            dataset=dataset.manifest["dataset"],
            predictor=predictor_label,
            coverages=coverages,
            seeds=stability_seeds,
            minimum_score_only_delta=float(evaluation.get("minimum_score_only_delta", 0.0)),
            minimum_supported_alert_fraction=float(
                policy_guardrails.get("minimum_supported_alert_fraction", 0.01)
            ),
            minimum_evidence_tp_fp_gap=float(
                policy_guardrails.get("minimum_evidence_tp_fp_gap", 0.0)
            ),
            minimum_positive_region_fidelity=minimum_fidelity,
            require_full_budget_support=bool(
                policy_guardrails.get("require_full_budget_support", True)
            ),
        )
        guarded: list[LockedSelectivePolicy] = []
        candidates_table = candidates_table.copy()
        candidates_table["stability_guardrail_applied"] = True
        for policy in locked:
            rows = stability_table.loc[
                np.isclose(stability_table["coverage"].astype(float), policy.coverage)
            ]
            stable_fraction = (
                float(
                    (
                        rows["method_matches_reference"].astype(bool)
                        & rows["abstention_matches_reference"].astype(bool)
                        & (
                            pd.to_numeric(
                                rows["full_validation_delta_vs_score_only"], errors="coerce"
                            )
                            > float(evaluation.get("minimum_score_only_delta", 0.0))
                        )
                    ).mean()
                )
                if not rows.empty and not policy.abstain
                else (1.0 if policy.abstain else 0.0)
            )
            candidates_table.loc[
                np.isclose(candidates_table["coverage"].astype(float), policy.coverage),
                "policy_stability_fraction",
            ] = stable_fraction
            if not policy.abstain and stable_fraction < minimum_lock_stability:
                policy = replace(
                    policy,
                    method=None,
                    abstain=True,
                    reason="validation_policy_stability_guardrail_not_met",
                    validation_selected_count=0,
                    validation_selected_precision=float("nan"),
                    validation_delta_vs_score_only=float("nan"),
                    validation_evidence_cutoff=float("nan"),
                )
            guarded.append(policy)
        return guarded, candidates_table, stability_table

    policies, policy_candidates, stability = lock_with_stability_guard(
        policy_evidence,
        predictor_label=f"{reference.family}:seed={reference.seed}",
    )

    registry_index = evidence_registry.set_index("rule", drop=False)
    domain_names = {
        str(rule) for rule in registry_index.index if str(registry_index.loc[rule, "tier"]).upper() == "A"
    }
    tier_b_names = {
        str(rule)
        for rule in registry_index.index
        if str(registry_index.loc[rule, "tier"]).upper() == "B"
    }
    counterfactual_only_names = {
        str(rule)
        for rule in registry_index.index
        if str(registry_index.loc[rule, "source"]) == "train_model_counterfactual"
    }
    independent_pool_definitions = {
        "domain_tier_a_only": (
            domain_names,
            "all_tier_a_rules_from_complete_pre_dedup_audit_table",
        ),
        "counterfactual_only": (
            counterfactual_only_names,
            "all_source_train_model_counterfactual_rules_from_complete_pre_dedup_audit_table",
        ),
        "train_derived_tier_b_only": (
            tier_b_names,
            "all_tier_b_rules_from_complete_pre_dedup_audit_table",
        ),
    }
    independent_rule_pools: dict[str, dict[str, Any]] = {}
    for variant, (pool_names, pool_definition) in independent_pool_definitions.items():
        pool_audit, pool_redundancy, pool_weights, pool_provenance = (
            _independent_ablation_rule_pool(
                audit,
                audit_truth,
                pool_names,
                variant=variant,
                pool_definition=pool_definition,
                audit_alert_mask=audit_alert_mask,
                activation_threshold=activation_threshold,
                max_jaccard=max_jaccard,
                max_rules=max_rules,
                fp_penalty=fp_penalty,
            )
        )
        independent_rule_pools[variant] = {
            "audit": pool_audit,
            "redundancy": pool_redundancy,
            "weights": pool_weights,
            "provenance": pool_provenance,
        }
    domain_weights = independent_rule_pools["domain_tier_a_only"]["weights"]
    counterfactual_only_weights = independent_rule_pools["counterfactual_only"]["weights"]
    tier_b_weights = independent_rule_pools["train_derived_tier_b_only"]["weights"]
    domain_contrastive = fit_contrastive_for(
        domain_weights,
        pretest_labels["rule_audit"],
        reference_audit_probabilities,
        reference.frozen.threshold,
        seed_offset=101,
    )
    tier_b_contrastive = fit_contrastive_for(
        tier_b_weights,
        pretest_labels["rule_audit"],
        reference_audit_probabilities,
        reference.frozen.threshold,
        seed_offset=202,
    )
    counterfactual_only_contrastive = fit_contrastive_for(
        counterfactual_only_weights,
        pretest_labels["rule_audit"],
        reference_audit_probabilities,
        reference.frozen.threshold,
        seed_offset=252,
    )

    control_rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)) + 54_321)
    shuffled_labels = pretest_labels["rule_audit"].copy()
    shuffled_labels[audit_alert_mask] = control_rng.permutation(
        shuffled_labels[audit_alert_mask]
    )
    shuffled_audit = audit_rule_candidates(
        train_truth,
        audit_truth,
        pretest_labels["train"],
        shuffled_labels,
        train_alert_mask=train_alert_mask,
        audit_alert_mask=audit_alert_mask,
        registry=evidence_registry,
        attribution_hit=audit_hit,
        attribution_share=audit_share,
        activation_threshold=activation_threshold,
        criteria=logic.get("audit_criteria", {}),
    )
    if shuffled_audit["audit_pass"].astype(bool).any():
        shuffled_completed, _ = deduplicate_audited_rules(
            shuffled_audit,
            audit_truth,
            audit_alert_mask=audit_alert_mask,
            activation_threshold=activation_threshold,
            max_jaccard=float(logic.get("max_jaccard", 0.85)),
            max_rules=int(logic.get("max_rules", 12)),
        )
    else:
        shuffled_completed, _ = _complete_audit_without_survivors(shuffled_audit)
    label_shuffle_weights = compute_rule_weights(
        shuffled_completed, fp_penalty=float(logic.get("fp_penalty", 1.0))
    )
    label_shuffle_contrastive = fit_contrastive_for(
        label_shuffle_weights,
        shuffled_labels,
        reference_audit_probabilities,
        reference.frozen.threshold,
        seed_offset=303,
    )
    weight_shuffle_weights = weights.copy().reset_index(drop=True)
    if len(weight_shuffle_weights) > 1:
        permutation = control_rng.permutation(len(weight_shuffle_weights))
        for column in ("audit_weight", "fp_penalized_weight"):
            weight_shuffle_weights[column] = weight_shuffle_weights[column].to_numpy()[permutation]

    surrogate_policy_truth = (
        surrogate_result.evaluate(matrices["policy_select"]).reset_index(drop=True)
        if surrogate_result is not None and surrogate_result.rules
        else pd.DataFrame(index=np.arange(len(split.policy_select)))
    )
    if not surrogate_policy_truth.empty:
        count = len(surrogate_policy_truth.columns)
        surrogate_weights = pd.DataFrame(
            {
                "rule": surrogate_policy_truth.columns.astype(str),
                "audit_weight": np.full(count, 1.0 / count),
                "fp_penalized_weight": np.full(count, 1.0 / count),
                "audit_alert_score_component": np.ones(count),
            }
        )
    else:
        surrogate_weights = pd.DataFrame(
            columns=["rule", "audit_weight", "fp_penalized_weight", "audit_alert_score_component"]
        )

    ablation_specs: dict[str, dict[str, Any]] = {
        "full_primary_ab": {
            "weights": weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "evidence_keys": evidence_method_names,
            "policies": policies,
            "policy_evidence": policy_evidence,
            "stability": stability,
        },
        "domain_tier_a_only": {
            "weights": domain_weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": base_evidence_method_names,
            "contrastive_scorer": domain_contrastive,
            "include_guarded": True,
            "rule_pool_provenance": independent_rule_pools["domain_tier_a_only"][
                "provenance"
            ],
        },
        "counterfactual_only": {
            "weights": counterfactual_only_weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": base_evidence_method_names,
            "contrastive_scorer": counterfactual_only_contrastive,
            "include_guarded": True,
            "rule_pool_provenance": independent_rule_pools["counterfactual_only"][
                "provenance"
            ],
        },
        "train_derived_tier_b_only": {
            "weights": tier_b_weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": base_evidence_method_names,
            "contrastive_scorer": tier_b_contrastive,
            "include_guarded": True,
            "rule_pool_provenance": independent_rule_pools[
                "train_derived_tier_b_only"
            ]["provenance"],
        },
        "method_unweighted": {
            "evidence_keys": ["unweighted"],
            "policy_evidence": {"unweighted": policy_evidence["unweighted"]},
        },
        "method_audit_weighted_no_attribution": {
            "evidence_keys": ["audit_weighted"],
            "policy_evidence": {"audit_weighted": policy_evidence["audit_weighted"]},
        },
        "method_fp_penalized_no_attribution": {
            "evidence_keys": ["fp_penalized"],
            "policy_evidence": {"fp_penalized": policy_evidence["fp_penalized"]},
        },
        "method_attribution_gated": {
            "evidence_keys": ["attribution_gated"],
            "policy_evidence": {"attribution_gated": policy_evidence["attribution_gated"]},
        },
        "method_contrastive_meta": {
            "evidence_keys": ["contrastive_meta"],
            "policy_evidence": {"contrastive_meta": policy_evidence["contrastive_meta"]},
        },
        "method_guarded_ensemble": {
            "evidence_keys": ["guarded_ensemble"],
            "policy_evidence": {"guarded_ensemble": policy_evidence["guarded_ensemble"]},
        },
        "cart_tier_c_baseline": {
            "weights": surrogate_weights,
            "policy_truth": surrogate_policy_truth,
            "policy_share": pd.DataFrame(index=np.arange(len(split.policy_select))),
            "methods": ["unweighted"],
        },
        "attribution_top_k_baseline": {
            "weights": pd.DataFrame(),
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": ["attribution_top_k"],
            "policy_evidence": {
                "attribution_top_k": policy_share[
                    "__attribution_top_k_positive_evidence__"
                ].to_numpy(dtype=float)
            },
            "attribution_formula": "sum_positive_among_top5_absolute_native_attributions",
        },
        "validation_label_shuffle_control": {
            "weights": label_shuffle_weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": base_evidence_method_names,
            "contrastive_scorer": label_shuffle_contrastive,
            "include_guarded": True,
        },
        "validation_weight_shuffle_control": {
            "weights": weight_shuffle_weights,
            "policy_truth": policy_truth,
            "policy_share": policy_share,
            "methods": ["audit_weighted", "fp_penalized", "attribution_gated"],
        },
    }
    ablation_rule_selection = pd.concat(
        [
            independent_rule_pools[variant]["audit"]
            for variant in independent_pool_definitions
        ],
        ignore_index=True,
    )
    ablation_rule_pool_provenance = [
        independent_rule_pools[variant]["provenance"]
        for variant in independent_pool_definitions
    ]

    def rule_pool_result_columns(specification: Mapping[str, Any]) -> dict[str, Any]:
        provenance = specification.get("rule_pool_provenance")
        if not isinstance(provenance, Mapping):
            return {}
        return {
            "rule_pool_selection_scope": str(provenance["selection_scope"]),
            "rule_pool_fit_partition": str(provenance["fit_partition"]),
            "rule_pool_candidate_count": int(provenance["candidate_rule_count"]),
            "rule_pool_audit_passing_count": int(
                provenance["audit_passing_rule_count"]
            ),
            "rule_pool_selected_count": int(provenance["selected_rule_count"]),
            "rule_pool_selected_rules": json.dumps(
                list(provenance["selected_rules"]), sort_keys=True
            ),
            "rule_pool_dedup_recomputed": bool(
                provenance["deduplication_recomputed_within_pool"]
            ),
            "rule_pool_weights_recomputed": bool(
                provenance["weights_recomputed_within_pool"]
            ),
            "rule_pool_primary_weights_filtered": bool(
                provenance["primary_selected_weights_filtered"]
            ),
            "rule_pool_test_labels_used": bool(provenance["test_labels_used"]),
        }

    ablation_validation_tables: list[pd.DataFrame] = []
    ablation_stability_tables: list[pd.DataFrame] = []
    for variant, specification in ablation_specs.items():
        if "policy_evidence" not in specification:
            specification["policy_evidence"] = _evidence_methods(
                specification["policy_truth"],
                specification["weights"],
                specification["policy_share"],
                specification["methods"],
                activation_threshold=activation_threshold,
            )
            contrastive_scorer = specification.get("contrastive_scorer")
            if contrastive_scorer is not None:
                specification["policy_evidence"]["contrastive_meta"] = (
                    contrastive_scorer.transform(
                        specification["policy_truth"], policy_probabilities
                    )
                )
            if bool(specification.get("include_guarded", False)):
                specification["policy_evidence"] = _add_guarded_ensemble(
                    specification["policy_evidence"], guarded_components
                )
        if "policies" not in specification:
            variant_policies, variant_candidates, variant_stability = (
                lock_with_stability_guard(
                    specification["policy_evidence"],
                    predictor_label=(
                        f"{reference.family}:seed={reference.seed}:variant={variant}"
                    ),
                )
            )
            specification["policies"] = variant_policies
            specification["stability"] = variant_stability
        else:
            variant_candidates = policy_candidates.copy()
            variant_stability = specification["stability"].copy()
        provenance_columns = rule_pool_result_columns(specification)
        variant_candidates.insert(0, "variant", variant)
        variant_candidates["primary_confirmatory_variant"] = variant == "full_primary_ab"
        for column, value in provenance_columns.items():
            variant_candidates[column] = value
        ablation_validation_tables.append(variant_candidates)
        variant_stability.insert(0, "variant", variant)
        variant_stability["primary_confirmatory_variant"] = variant == "full_primary_ab"
        for column, value in provenance_columns.items():
            variant_stability[column] = value
        ablation_stability_tables.append(variant_stability)

    policy_alert_scores = policy_probabilities[policy_probabilities >= reference.frozen.threshold]
    risk_bin_fallback = False
    if len(policy_alert_scores) >= 2:
        risk_edges = frozen_risk_bin_edges(
            policy_alert_scores,
            n_bins=int(evaluation.get("matched_risk_bins", 20)),
        )
    else:
        risk_edges = np.asarray([-np.inf, np.inf], dtype=float)
        risk_bin_fallback = True
    ledger.advance(
        "policies_locked_on_reserved_validation",
        policy_partition="policy_select",
        coverages=coverages,
        abstention_count=sum(policy.abstain for policy in policies),
        risk_bin_edges_fitted_on_validation_alerts=True,
    )

    # Test features can now be scored and explained.  The splitter previously
    # inspected the test target only to assert binary/class-count integrity;
    # no test outcome enters this scoring, explanation or ranking block.
    X_test = preprocessor.transform(_feature_frame(split.test, dataset))
    test_probabilities = predict_probabilities(reference.frozen, X_test, calibrated=True, device=device)
    test_truth = configured_result.evaluate(split.test).reset_index(drop=True)
    test_hit, test_share, test_importance, test_attribution, test_error = _compute_rule_attribution(
        reference.frozen,
        X_test,
        evidence_registry,
        list(test_truth.columns),
        device="cpu" if quick_run else device,
    )
    del test_hit
    test_evidence = _evidence_methods(
        test_truth,
        weights,
        test_share,
        base_evidence_method_names,
        activation_threshold=activation_threshold,
    )
    test_evidence["contrastive_meta"] = contrastive_meta.transform(
        test_truth, test_probabilities
    )
    test_evidence = _add_guarded_ensemble(test_evidence, guarded_components)
    surrogate_test_truth = (
        surrogate_result.evaluate(X_test).reset_index(drop=True)
        if surrogate_result is not None and surrogate_result.rules
        else pd.DataFrame(index=np.arange(len(split.test)))
    )
    ablation_prelabel: list[dict[str, Any]] = []
    ablation_digest = hashlib.sha256()
    for variant, specification in ablation_specs.items():
        if variant == "full_primary_ab":
            variant_test_evidence = test_evidence
        elif variant == "cart_tier_c_baseline":
            variant_test_evidence = _evidence_methods(
                surrogate_test_truth,
                specification["weights"],
                pd.DataFrame(index=np.arange(len(split.test))),
                specification["methods"],
                activation_threshold=activation_threshold,
            )
        elif variant == "attribution_top_k_baseline":
            variant_test_evidence = {
                "attribution_top_k": test_share[
                    "__attribution_top_k_positive_evidence__"
                ].to_numpy(dtype=float)
            }
        elif "evidence_keys" in specification:
            variant_test_evidence = {
                str(key): test_evidence[str(key)]
                for key in specification["evidence_keys"]
            }
        else:
            variant_test_evidence = _evidence_methods(
                test_truth,
                specification["weights"],
                test_share,
                specification["methods"],
                activation_threshold=activation_threshold,
            )
            contrastive_scorer = specification.get("contrastive_scorer")
            if contrastive_scorer is not None:
                variant_test_evidence["contrastive_meta"] = contrastive_scorer.transform(
                    test_truth, test_probabilities
                )
            if bool(specification.get("include_guarded", False)):
                variant_test_evidence = _add_guarded_ensemble(
                    variant_test_evidence, guarded_components
                )
        for policy in specification["policies"]:
            selected_mask = apply_locked_policy(
                policy,
                test_probabilities,
                reference.frozen.threshold,
                variant_test_evidence,
            )
            score_only_mask = fixed_count_mask(
                test_probabilities,
                test_probabilities,
                test_probabilities >= reference.frozen.threshold,
                int(selected_mask.sum()),
            )
            ablation_digest.update(variant.encode("utf-8"))
            ablation_digest.update(f"{policy.coverage:.12g}".encode("ascii"))
            ablation_digest.update(np.ascontiguousarray(selected_mask.astype(np.uint8)).tobytes())
            ablation_prelabel.append(
                {
                    "variant": variant,
                    "policy": policy,
                    "test_evidence": variant_test_evidence,
                    "selected_mask": selected_mask,
                    "score_only_mask": score_only_mask,
                    "rule_pool_result_columns": rule_pool_result_columns(specification),
                }
            )

    # Predictor-sensitivity analysis replays the reference-fitted A/B rule set
    # across every trained predictor family.  Each family gets its own native
    # attribution, calibrated alert region and validation-locked policy; test
    # labels are still unavailable while all masks below are materialised.
    predictor_sensitivity_prelabel: list[dict[str, Any]] = []
    predictor_attribution_rows: list[dict[str, Any]] = []
    predictor_attribution_provenance: dict[str, Any] = {}
    predictor_attribution_errors: list[str] = []
    predictor_contrastive_rows: list[pd.DataFrame] = []
    predictor_contrastive_provenance: dict[str, Any] = {}
    candidate_test_probabilities_prelabel: dict[str, np.ndarray] = {}
    predictor_sensitivity_digest = hashlib.sha256()
    selected_rule_list = weights["rule"].astype(str).tolist() if not weights.empty else []
    for candidate_index, state in enumerate(candidates):
        candidate_key = f"{state.family}__seed_{state.seed}"
        if candidate_index == reference_index:
            candidate_audit_probabilities = reference_audit_probabilities
            candidate_policy_probabilities = policy_probabilities
            candidate_test_probabilities = test_probabilities
            candidate_audit_hit = audit_hit
            candidate_audit_share = audit_share
            candidate_policy_share = policy_share
            candidate_test_share = test_share
            candidate_provenance = {
                "rule_audit": audit_attribution,
                "policy_select": policy_attribution,
                "test": test_attribution,
            }
            candidate_errors = [value for value in (audit_error, policy_error, test_error) if value]
        else:
            candidate_audit_probabilities = predict_probabilities(
                state.frozen, matrices["rule_audit"], calibrated=True, device=device
            )
            candidate_policy_probabilities = predict_probabilities(
                state.frozen, matrices["policy_select"], calibrated=True, device=device
            )
            candidate_test_probabilities = predict_probabilities(
                state.frozen, X_test, calibrated=True, device=device
            )
            (
                candidate_audit_hit,
                candidate_audit_share,
                _,
                candidate_audit_provenance,
                candidate_audit_error,
            ) = _compute_rule_attribution(
                state.frozen,
                matrices["rule_audit"],
                evidence_registry,
                list(audit_truth.columns),
                device="cpu" if quick_run else device,
            )
            (
                _,
                candidate_policy_share,
                _,
                candidate_policy_provenance,
                candidate_policy_error,
            ) = _compute_rule_attribution(
                state.frozen,
                matrices["policy_select"],
                evidence_registry,
                list(policy_truth.columns),
                device="cpu" if quick_run else device,
            )
            (
                _,
                candidate_test_share,
                _,
                candidate_test_provenance,
                candidate_test_error,
            ) = _compute_rule_attribution(
                state.frozen,
                X_test,
                evidence_registry,
                list(test_truth.columns),
                device="cpu" if quick_run else device,
            )
            candidate_provenance = {
                "rule_audit": candidate_audit_provenance,
                "policy_select": candidate_policy_provenance,
                "test": candidate_test_provenance,
            }
            candidate_errors = [
                value
                for value in (
                    candidate_audit_error,
                    candidate_policy_error,
                    candidate_test_error,
                )
                if value
            ]
        candidate_test_probabilities_prelabel[candidate_key] = candidate_test_probabilities
        predictor_attribution_provenance[candidate_key] = candidate_provenance
        predictor_attribution_errors.extend(
            f"{candidate_key}:{error}" for error in candidate_errors
        )
        candidate_contrastive = fit_contrastive_for(
            weights,
            pretest_labels["rule_audit"],
            candidate_audit_probabilities,
            state.frozen.threshold,
            seed_offset=1_000 + candidate_index,
        )
        predictor_contrastive_provenance[candidate_key] = candidate_contrastive.provenance
        predictor_contrastive_rows.append(
            candidate_contrastive.coefficient_table().assign(
                candidate=candidate_key,
                family=state.family,
                seed=state.seed,
                reference_selected=candidate_index == reference_index,
                rule_set_origin="reference_predictor_train_and_rule_audit",
            )
        )
        for rule in selected_rule_list:
            predictor_attribution_rows.append(
                {
                    "candidate": candidate_key,
                    "family": state.family,
                    "seed": state.seed,
                    "reference_selected": candidate_index == reference_index,
                    "rule": rule,
                    "rule_set_origin": "reference_predictor_train_and_rule_audit",
                    "audit_attribution_hit_rate": float(candidate_audit_hit[rule].mean()),
                    "audit_attribution_share": float(candidate_audit_share[rule].mean()),
                    "policy_attribution_share": float(candidate_policy_share[rule].mean()),
                    "test_attribution_share": float(candidate_test_share[rule].mean()),
                }
            )
        candidate_policy_evidence = _evidence_methods(
            policy_truth,
            weights,
            candidate_policy_share,
            base_evidence_method_names,
            activation_threshold=activation_threshold,
        )
        candidate_policy_evidence["contrastive_meta"] = candidate_contrastive.transform(
            policy_truth, candidate_policy_probabilities
        )
        candidate_policy_evidence = _add_guarded_ensemble(
            candidate_policy_evidence, guarded_components
        )
        candidate_test_evidence = _evidence_methods(
            test_truth,
            weights,
            candidate_test_share,
            base_evidence_method_names,
            activation_threshold=activation_threshold,
        )
        candidate_test_evidence["contrastive_meta"] = candidate_contrastive.transform(
            test_truth, candidate_test_probabilities
        )
        candidate_test_evidence = _add_guarded_ensemble(
            candidate_test_evidence, guarded_components
        )
        candidate_policies, _ = lock_selective_policies(
            pretest_labels["policy_select"],
            candidate_policy_probabilities,
            state.frozen.threshold,
            candidate_policy_evidence,
            dataset=dataset.manifest["dataset"],
            predictor=f"{state.family}:seed={state.seed}",
            coverages=coverages,
            minimum_score_only_delta=float(evaluation.get("minimum_score_only_delta", 0.0)),
            minimum_supported_alert_fraction=float(
                policy_guardrails.get("minimum_supported_alert_fraction", 0.01)
            ),
            minimum_evidence_tp_fp_gap=float(
                policy_guardrails.get("minimum_evidence_tp_fp_gap", 0.0)
            ),
            minimum_positive_region_fidelity=minimum_fidelity,
            require_full_budget_support=bool(
                policy_guardrails.get("require_full_budget_support", True)
            ),
        )
        for candidate_policy in candidate_policies:
            candidate_selected = apply_locked_policy(
                candidate_policy,
                candidate_test_probabilities,
                state.frozen.threshold,
                candidate_test_evidence,
            )
            candidate_score_only = fixed_count_mask(
                candidate_test_probabilities,
                candidate_test_probabilities,
                candidate_test_probabilities >= state.frozen.threshold,
                int(candidate_selected.sum()),
            )
            predictor_sensitivity_prelabel.append(
                {
                    "candidate": candidate_key,
                    "family": state.family,
                    "seed": state.seed,
                    "reference_selected": candidate_index == reference_index,
                    "rule_set_origin": "reference_predictor_train_and_rule_audit",
                    "diagnostic_only": True,
                    "policy_stability_guardrail_applied": False,
                    "contrastive_meta_available": candidate_contrastive.available,
                    "contrastive_meta_fallback_reason": candidate_contrastive.fallback_reason,
                    "policy": candidate_policy,
                    "threshold": state.frozen.threshold,
                    "test_probabilities": candidate_test_probabilities,
                    "test_evidence": candidate_test_evidence,
                    "selected_mask": candidate_selected,
                    "score_only_mask": candidate_score_only,
                }
            )
            predictor_sensitivity_digest.update(candidate_key.encode("utf-8"))
            predictor_sensitivity_digest.update(
                f"{candidate_policy.coverage:.12g}".encode("ascii")
            )
            predictor_sensitivity_digest.update(
                np.ascontiguousarray(candidate_selected.astype(np.uint8)).tobytes()
            )
    selection_signature, prelabel_masks = _selection_signature(
        policies,
        test_probabilities,
        reference.frozen.threshold,
        test_evidence,
    )
    ledger.advance(
        "test_selections_materialized_without_outcome_use",
        selection_signature_sha256=selection_signature,
        ablation_selection_signature_sha256=ablation_digest.hexdigest(),
        predictor_sensitivity_signature_sha256=predictor_sensitivity_digest.hexdigest(),
        test_outcomes_used_for_selection=False,
        earlier_test_target_use="split-integrity checks only",
    )

    # Release the target for aggregate outcome evaluation only.  The locked
    # masks are recomputed after an explicit label permutation and must remain
    # byte-identical because labels are not an input to policy application.
    y_test = _labels(split.test)
    permutation_rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)) + 90_001)
    permuted_test_labels = permutation_rng.permutation(y_test)
    repeated_signature, repeated_masks = _selection_signature(
        policies,
        test_probabilities,
        reference.frozen.threshold,
        test_evidence,
    )
    if repeated_signature != selection_signature or any(
        not np.array_equal(prelabel_masks[key], repeated_masks[key]) for key in prelabel_masks
    ):
        raise AssertionError("Locked test selections changed after test labels were opened")
    if len(permuted_test_labels) != len(y_test):  # pragma: no cover - defensive invariant
        raise AssertionError("Test-label permutation changed row alignment")
    ledger.advance(
        "test_outcomes_released_for_evaluation",
        selection_signature_unchanged=True,
        test_label_permutation_invariance_asserted=True,
        test_labels_used_for_reselection=False,
    )

    ablation_rows: list[dict[str, Any]] = []
    for prepared_ablation in ablation_prelabel:
        ablation_result, selected_mask, score_only_mask = evaluate_locked_policy(
            prepared_ablation["policy"],
            y_test,
            test_probabilities,
            reference.frozen.threshold,
            prepared_ablation["test_evidence"],
        )
        if not np.array_equal(selected_mask, prepared_ablation["selected_mask"]):
            raise AssertionError("Ablation selection changed after test outcomes were released")
        if not np.array_equal(score_only_mask, prepared_ablation["score_only_mask"]):
            raise AssertionError("Ablation score-only mask changed after test outcomes were released")
        ablation_rows.append(
            {
                "variant": prepared_ablation["variant"],
                "primary_confirmatory_variant": prepared_ablation["variant"]
                == "full_primary_ab",
                "test_labels_used_for_selection": False,
                **prepared_ablation["rule_pool_result_columns"],
                **ablation_result,
            }
        )
    ablation_results = pd.DataFrame(ablation_rows)
    ablation_validation = (
        pd.concat(ablation_validation_tables, ignore_index=True)
        if ablation_validation_tables
        else pd.DataFrame(columns=["variant", "coverage"])
    )
    ablation_stability = (
        pd.concat(ablation_stability_tables, ignore_index=True)
        if ablation_stability_tables
        else pd.DataFrame(columns=["variant", "coverage", "seed"])
    )
    predictor_sensitivity_rows: list[dict[str, Any]] = []
    for prepared_sensitivity in predictor_sensitivity_prelabel:
        sensitivity_result, selected_mask, score_only_mask = evaluate_locked_policy(
            prepared_sensitivity["policy"],
            y_test,
            prepared_sensitivity["test_probabilities"],
            prepared_sensitivity["threshold"],
            prepared_sensitivity["test_evidence"],
        )
        if not np.array_equal(selected_mask, prepared_sensitivity["selected_mask"]):
            raise AssertionError("Predictor-sensitivity selection changed after outcomes were released")
        if not np.array_equal(score_only_mask, prepared_sensitivity["score_only_mask"]):
            raise AssertionError("Predictor-sensitivity score-only mask changed after outcomes were released")
        predictor_sensitivity_rows.append(
            {
                "candidate": prepared_sensitivity["candidate"],
                "family": prepared_sensitivity["family"],
                "seed": prepared_sensitivity["seed"],
                "reference_selected": prepared_sensitivity["reference_selected"],
                "rule_set_origin": prepared_sensitivity["rule_set_origin"],
                "diagnostic_only": prepared_sensitivity["diagnostic_only"],
                "policy_stability_guardrail_applied": prepared_sensitivity[
                    "policy_stability_guardrail_applied"
                ],
                "test_labels_used_for_selection": False,
                **sensitivity_result,
            }
        )
    predictor_sensitivity = pd.DataFrame(predictor_sensitivity_rows)
    predictor_attribution_sensitivity = pd.DataFrame(
        predictor_attribution_rows,
        columns=[
            "candidate",
            "family",
            "seed",
            "reference_selected",
            "rule",
            "rule_set_origin",
            "audit_attribution_hit_rate",
            "audit_attribution_share",
            "policy_attribution_share",
            "test_attribution_share",
        ],
    )
    predictor_contrastive_sensitivity = (
        pd.concat(predictor_contrastive_rows, ignore_index=True)
        if predictor_contrastive_rows and any(not frame.empty for frame in predictor_contrastive_rows)
        else pd.DataFrame(
            columns=[
                "feature",
                "feature_type",
                "rule",
                "coefficient",
                "absolute_coefficient",
                "odds_ratio",
                "absolute_coefficient_rank",
                "intercept",
                "available",
                "fallback_reason",
                "fit_partition",
                "candidate",
                "family",
                "seed",
                "reference_selected",
                "rule_set_origin",
            ]
        )
    )

    predictive_rows: list[dict[str, Any]] = []
    all_candidate_test_probabilities: dict[str, np.ndarray] = {}
    selection_pr_auc = {
        (str(row["family"]), int(row["seed"])): float(row["pr_auc"])
        for _, row in validation_metrics.iterrows()
    }
    for index, state in enumerate(candidates):
        candidate_key = f"{state.family}__seed_{state.seed}"
        probabilities = candidate_test_probabilities_prelabel[candidate_key]
        all_candidate_test_probabilities[candidate_key] = probabilities
        predictive_rows.append(
            {
                "evaluation_partition": "test",
                "family": state.family,
                "backend": state.frozen.backend,
                "seed": state.seed,
                "reference_selected": index == reference_index,
                "selection_split_pr_auc": selection_pr_auc[(state.family, state.seed)],
                **evaluate_stress_predictions(
                    y_test,
                    probabilities,
                    threshold=state.frozen.threshold,
                    beta=float(evaluation.get("beta", 2.0)),
                    n_calibration_bins=int(evaluation.get("n_calibration_bins", 15)),
                    target_fpr=float(evaluation.get("recall_fpr", 0.01)),
                ),
            }
        )
    predictive_metrics = pd.DataFrame(predictive_rows)

    coverage_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    matched_pair_tables: list[pd.DataFrame] = []
    residual_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    block_ids = pd.to_datetime(split.test["event_time"], utc=True).dt.floor("D").astype(str).to_numpy()
    selected_rule_names = weights["rule"].astype(str).tolist() if not weights.empty else []
    if selected_rule_names:
        selected_rule_active = (
            test_truth[selected_rule_names].to_numpy(dtype=float) >= activation_threshold
        )
        active_selected_rule_count = selected_rule_active.sum(axis=1)
        any_selected_rule_evidence = selected_rule_active.any(axis=1)
    else:
        active_selected_rule_count = np.zeros(len(test_truth), dtype=int)
        any_selected_rule_evidence = np.zeros(len(test_truth), dtype=bool)
    predictor_alerts = test_probabilities >= reference.frozen.threshold
    display_top_k = int(logic.get("display_top_k_rules", logic.get("top_k_rules", 3)))
    supported_alert_count = int((predictor_alerts & any_selected_rule_evidence).sum())
    unsupported_alert_count = int((predictor_alerts & ~any_selected_rule_evidence).sum())
    evidence_without_alert_count = int((~predictor_alerts & any_selected_rule_evidence).sum())
    contradiction_xor = predictor_alerts != any_selected_rule_evidence
    bootstrap_iterations = int(
        min(evaluation.get("bootstrap_iterations", 1000), 100) if quick_run else evaluation.get("bootstrap_iterations", 1000)
    )
    source_checksum_verified = bool(dataset.manifest.get("source_files")) and all(
        bool(item.get("verified")) for item in dataset.manifest.get("source_files", [])
    )
    exact_backends = not any(
        bool(state.frozen.provenance.get("quick_fallback_used", False)) for state in candidates
    )
    saturation_scope_eligible = bool(
        dataset.manifest.get("dataset") == "amlnet_v1_0"
        and not quick_run
        and not use_test_fixture
        and not bool(dataset.manifest.get("quick_sample", {}).get("enabled", False))
        and source_checksum_verified
        and exact_backends
        and not predictor_attribution_errors
        and not any(value for value in (audit_error, policy_error, test_error) if value)
        and not risk_bin_fallback
    )
    for policy in policies:
        result, selected, score_only = evaluate_locked_policy(
            policy,
            y_test,
            test_probabilities,
            reference.frozen.threshold,
            test_evidence,
        )
        result.update(
            {
                "selected_rule_count_total": int(len(selected_rule_names)),
                "supported_alert_count": supported_alert_count,
                "supported_alert_rate": (
                    float(supported_alert_count / predictor_alerts.sum())
                    if predictor_alerts.any()
                    else float("nan")
                ),
                "unsupported_alert_count": unsupported_alert_count,
                "unsupported_alert_rate": (
                    float(unsupported_alert_count / predictor_alerts.sum())
                    if predictor_alerts.any()
                    else float("nan")
                ),
                "mean_active_selected_rule_count_alerts": (
                    float(active_selected_rule_count[predictor_alerts].mean())
                    if predictor_alerts.any()
                    else float("nan")
                ),
                "mean_active_selected_rule_count_explained": (
                    float(active_selected_rule_count[selected].mean())
                    if selected.any()
                    else float("nan")
                ),
                "mean_displayed_rule_count_explained": (
                    float(np.minimum(active_selected_rule_count[selected], display_top_k).mean())
                    if selected.any()
                    else float("nan")
                ),
                "rule_activation_density_alerts": (
                    float(active_selected_rule_count[predictor_alerts].mean() / len(selected_rule_names))
                    if predictor_alerts.any() and selected_rule_names
                    else 0.0
                ),
                "rule_sparsity_alerts": (
                    float(1.0 - active_selected_rule_count[predictor_alerts].mean() / len(selected_rule_names))
                    if predictor_alerts.any() and selected_rule_names
                    else 1.0
                ),
                "alert_rule_evidence_xor_rate_all_rows": float(contradiction_xor.mean()),
                "contradiction_rate": float(contradiction_xor.mean()),
                "evidence_without_alert_count": evidence_without_alert_count,
                "evidence_without_alert_rate": (
                    float(evidence_without_alert_count / any_selected_rule_evidence.sum())
                    if any_selected_rule_evidence.any()
                    else float("nan")
                ),
            }
        )
        coverage_rows.append(result)
        evidence = (
            test_evidence[str(policy.method)]
            if not policy.abstain and policy.method is not None
            else np.zeros(len(y_test), dtype=float)
        )
        pairs = match_selected_to_controls(
            test_probabilities,
            evidence,
            selected,
            test_probabilities >= reference.frozen.threshold,
            risk_edges,
            caliper=float(evaluation.get("matched_risk_caliper", 0.02)),
        )
        pairs.insert(0, "coverage_budget", policy.coverage)
        matched_pair_tables.append(pairs)
        matched = summarize_matched_risk(
            pairs.drop(columns="coverage_budget"),
            y_test,
            n_bootstrap=bootstrap_iterations,
            seed=int(config.get("project", {}).get("seed", 42)),
        )
        matched_rows.append({"coverage_budget": policy.coverage, **matched})
        residual_table, residual_summary = residual_tp_fp_by_risk_bin(
            y_test,
            test_probabilities,
            evidence,
            test_probabilities >= reference.frozen.threshold,
            risk_edges,
        )
        residual_table.insert(0, "coverage_budget", policy.coverage)
        residual_rows.append(residual_table)
        bootstrap = paired_selection_bootstrap(
            y_test,
            selected,
            score_only,
            block_ids=block_ids,
            n_bootstrap=bootstrap_iterations,
            seed=int(config.get("project", {}).get("seed", 42)),
        )
        control_rows = _negative_control_rows(
            policy,
            y_test,
            test_probabilities,
            reference.frozen.threshold,
            selected,
            score_only,
            evidence,
            seed=int(config.get("project", {}).get("seed", 42)),
        )
        negative_rows.extend(control_rows)
        stability_for_coverage = stability.loc[
            np.isclose(stability["coverage"].astype(float), policy.coverage)
        ]
        shuffled_controls_for_coverage = ablation_results.loc[
            ablation_results["variant"].isin(_MANDATORY_SHUFFLED_CONTROLS)
            & np.isclose(
                ablation_results["coverage_budget"].astype(float), policy.coverage
            )
        ].to_dict("records")
        outcome, outcome_diagnostics = _conservative_stress_outcome(
            result,
            bootstrap,
            matched,
            residual_summary,
            control_rows,
            stability_for_coverage,
            shuffled_controls=shuffled_controls_for_coverage,
            saturation_headroom=float(evaluation["saturation_headroom"]),
            material_precision_margin=float(evaluation["saturation_precision_margin"]),
            material_evidence_margin=float(evaluation["saturation_evidence_margin"]),
            minimum_positive_stability_fraction=float(
                evaluation["minimum_positive_stability_fraction"]
            ),
            saturation_scope_eligible=saturation_scope_eligible,
            minimum_saturation_alert_positives=int(
                evaluation["minimum_saturation_alert_positives"]
            ),
            minimum_saturation_matched_pairs=int(
                evaluation["minimum_saturation_matched_pairs"]
            ),
            minimum_saturation_bootstrap_replicates=int(
                evaluation["minimum_saturation_bootstrap_replicates"]
            ),
        )
        bootstrap_rows.append(
            {
                "coverage": policy.coverage,
                "coverage_budget": policy.coverage,
                "outcome": outcome,
                **outcome_diagnostics,
                **bootstrap,
            }
        )
        outcome_rows.append(
            {
                "coverage_budget": policy.coverage,
                "outcome": outcome,
                **outcome_diagnostics,
                **residual_summary,
            }
        )
    ledger.advance(
        "test_evaluated_without_reselection",
        coverage_count=len(coverage_rows),
        candidate_predictor_count=len(predictive_rows),
        policy_reselection_on_test=False,
    )
    if not ledger.complete:
        raise AssertionError("Stress protocol ledger is incomplete")

    attribution_errors = list(
        dict.fromkeys(
            [
                *[value for value in (audit_error, policy_error, test_error) if value],
                *predictor_attribution_errors,
            ]
        )
    )
    claim_eligible, claim_blockers, claim_flags = _claim_assessment(
        dataset,
        candidates,
        quick_run=quick_run,
        use_test_fixture=use_test_fixture,
        attribution_errors=attribution_errors,
        policy_locked_before_test=True,
    )
    claim_flags["contrastive_meta_available"] = bool(contrastive_meta.available)
    audit_alert_class_counts = {
        "tp": int((audit_alert_mask & (pretest_labels["rule_audit"] == 1)).sum()),
        "fp": int((audit_alert_mask & (pretest_labels["rule_audit"] == 0)).sum()),
    }
    claim_flags["rule_audit_has_tp_and_fp_alerts"] = bool(
        min(audit_alert_class_counts.values()) > 0
    )
    if not claim_flags["rule_audit_has_tp_and_fp_alerts"]:
        claim_blockers.append("rule_audit_lacks_tp_or_fp_predictor_alerts")
        claim_eligible = False
    reporting = dict(config.get("reporting", {}))
    for required_flag in reporting.get("required_claim_flags", []):
        flag = str(required_flag)
        if flag not in claim_flags:
            claim_blockers.append(f"unknown_required_claim_flag:{flag}")
            claim_eligible = False
        elif not bool(claim_flags[flag]):
            claim_blockers.append(f"required_claim_flag_not_met:{flag}")
            claim_eligible = False
    if risk_bin_fallback:
        claim_blockers.append("fewer_than_two_validation_alerts_for_risk_bins")
        claim_eligible = False

    coverage_table = pd.DataFrame(coverage_rows)
    matched_table = pd.DataFrame(matched_rows)
    matched_pairs = (
        pd.concat(matched_pair_tables, ignore_index=True)
        if matched_pair_tables
        else pd.DataFrame(columns=["coverage_budget"])
    )
    residual_table = (
        pd.concat(residual_rows, ignore_index=True)
        if residual_rows
        else pd.DataFrame(columns=["coverage_budget"])
    )
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    negative_table = pd.DataFrame(negative_rows)
    outcome_table = pd.DataFrame(outcome_rows)
    attribution_importance = pd.concat(
        [
            audit_importance.assign(partition="rule_audit"),
            policy_importance.assign(partition="policy_select"),
            test_importance.assign(partition="test"),
        ],
        ignore_index=True,
    )
    attribution_provenance = {
        "rule_audit": audit_attribution,
        "policy_select": policy_attribution,
        "test": test_attribution,
    }

    paths: dict[str, Path] = {}
    paths["data_manifest"] = write_json(destination / "data_manifest.json", _jsonable(dataset.manifest))
    paths["split_manifest"] = write_json(destination / "split_manifest.json", _jsonable(split.manifest))
    paths["split_integrity"] = _write_csv(destination / "split_integrity.csv", _partition_table(split))
    paths["protocol_ledger"] = write_json(destination / "protocol_ledger.json", _jsonable(ledger.events))
    paths["predictor_selection"] = _write_csv(
        destination / "predictor_selection_validation.csv", validation_metrics
    )
    paths["calibration_comparison"] = _write_csv(
        destination / "calibration_comparison.csv", calibration_table
    )
    history_rows = [
        {"family": state.family, "seed": state.seed, **row}
        for state in candidates
        for row in state.training_history
    ]
    paths["training_history"] = _write_csv(
        destination / "training_history.csv",
        pd.DataFrame(history_rows, columns=["family", "seed", "epoch", "train_loss"]),
    )
    paths["predictive_metrics"] = _write_csv(
        destination / "predictive_metrics.csv", predictive_metrics
    )
    paths["predictor_explanation_sensitivity"] = _write_csv(
        destination / "predictor_explanation_sensitivity.csv", predictor_sensitivity
    )
    paths["predictor_attribution_sensitivity"] = _write_csv(
        destination / "predictor_attribution_sensitivity.csv",
        predictor_attribution_sensitivity,
    )
    paths["predictor_attribution_sensitivity_provenance"] = write_json(
        destination / "predictor_attribution_sensitivity_provenance.json",
        _jsonable(predictor_attribution_provenance),
    )
    paths["predictor_contrastive_sensitivity"] = _write_csv(
        destination / "predictor_contrastive_sensitivity.csv",
        predictor_contrastive_sensitivity,
    )
    paths["predictor_contrastive_sensitivity_provenance"] = write_json(
        destination / "predictor_contrastive_sensitivity_provenance.json",
        _jsonable(predictor_contrastive_provenance),
    )
    registry_for_csv = registry.copy()
    for column in ("features", "conditions", "fitted_conditions"):
        if column in registry_for_csv:
            registry_for_csv[column] = registry_for_csv[column].map(
                lambda value: json.dumps(_jsonable(value), ensure_ascii=False)
            )
    paths["rule_registry"] = _write_csv(destination / "rule_registry.csv", registry_for_csv)
    paths["fitted_rule_thresholds"] = _write_csv(
        destination / "fitted_rule_thresholds.csv", configured_result.engine.fitted_thresholds()
    )
    counterfactual_diagnostics = (
        counterfactual_result.diagnostics
        if counterfactual_result is not None
        else pd.DataFrame(columns=["feature", "eligible", "selected", "reason"])
    )
    paths["counterfactual_diagnostics"] = _write_csv(
        destination / "counterfactual_diagnostics.csv", counterfactual_diagnostics
    )
    paths["candidate_generation_provenance"] = write_json(
        destination / "candidate_generation_provenance.json", _jsonable(candidate_provenance)
    )
    surrogate_registry = (
        surrogate_result.registry
        if surrogate_result is not None
        else pd.DataFrame(columns=["rule", "tier", "source"])
    )
    surrogate_registry_csv = surrogate_registry.copy()
    for column in ("features", "conditions", "fitted_conditions"):
        if column in surrogate_registry_csv:
            surrogate_registry_csv[column] = surrogate_registry_csv[column].map(
                lambda value: json.dumps(_jsonable(value), ensure_ascii=False)
            )
    paths["surrogate_rules"] = _write_csv(
        destination / "surrogate_rules.csv", surrogate_registry_csv
    )
    paths["surrogate_provenance"] = write_json(
        destination / "surrogate_provenance.json",
        _jsonable(surrogate_result.provenance if surrogate_result is not None else {"enabled": False}),
    )
    if surrogate_result is not None:
        surrogate_artifact = destination / "cart_surrogate_artifact.joblib"
        joblib.dump(surrogate_result, surrogate_artifact, compress=3)
        paths["surrogate_artifact"] = surrogate_artifact
    paths["rule_audit"] = _write_csv(destination / "rule_audit.csv", completed_audit)
    paths["rule_audit_stability"] = _write_csv(
        destination / "rule_audit_stability.csv", rule_audit_stability
    )
    paths["rule_redundancy"] = _write_csv(destination / "rule_redundancy.csv", redundancy)
    paths["rule_weights"] = _write_csv(destination / "rule_weights.csv", weights)
    paths["contrastive_meta_coefficients"] = _write_csv(
        destination / "contrastive_meta_coefficients.csv",
        contrastive_meta.coefficient_table(),
    )
    paths["contrastive_meta_provenance"] = write_json(
        destination / "contrastive_meta_provenance.json",
        _jsonable(contrastive_meta.provenance),
    )
    contrastive_artifact_path = destination / "contrastive_meta_artifact.joblib"
    joblib.dump(contrastive_meta, contrastive_artifact_path, compress=3)
    paths["contrastive_meta_artifact"] = contrastive_artifact_path
    paths["attribution_importance"] = _write_csv(
        destination / "attribution_feature_importance.csv", attribution_importance
    )
    paths["attribution_provenance"] = write_json(
        destination / "attribution_provenance.json", _jsonable(attribution_provenance)
    )
    paths["policy_candidates"] = _write_csv(
        destination / "policy_candidates_validation.csv", policy_candidates
    )
    paths["ablation_validation"] = _write_csv(
        destination / "ablation_validation_results.csv", ablation_validation
    )
    paths["ablation_validation_stability"] = _write_csv(
        destination / "ablation_validation_stability.csv", ablation_stability
    )
    paths["ablation_rule_selection"] = _write_csv(
        destination / "ablation_rule_selection.csv", ablation_rule_selection
    )
    paths["ablation_rule_pool_provenance"] = write_json(
        destination / "ablation_rule_pool_provenance.json",
        _jsonable(ablation_rule_pool_provenance),
    )
    paths["ablation_results"] = _write_csv(
        destination / "ablation_results.csv", ablation_results
    )
    paths["locked_policies"] = write_json(
        destination / "locked_policies.json",
        [policy.to_dict() for policy in policies],
    )
    paths["risk_bin_edges"] = write_json(
        destination / "risk_bin_edges.json",
        {
            "edges": risk_edges.tolist(),
            "source": "policy_select_alert_scores",
            "test_labels_used": False,
        },
    )
    paths["validation_stability"] = _write_csv(
        destination / "validation_stability.csv", stability
    )
    paths["validation_policy_stability_alias"] = _write_csv(
        destination / "validation_policy_stability.csv", stability
    )
    paths["coverage_results"] = _write_csv(destination / "coverage_results.csv", coverage_table)
    paths["matched_risk_results"] = _write_csv(
        destination / "matched_risk_results.csv", matched_table
    )
    paths["matched_risk_pairs"] = _write_csv(
        destination / "matched_risk_pairs.csv", matched_pairs
    )
    paths["residual_evidence"] = _write_csv(
        destination / "residual_evidence.csv", residual_table
    )
    paths["residual_tp_fp_alias"] = _write_csv(
        destination / "residual_tp_fp_by_risk_bin.csv", residual_table
    )
    paths["paired_bootstrap"] = _write_csv(
        destination / "paired_bootstrap.csv", bootstrap_table
    )
    paths["paired_calendar_day_bootstrap_alias"] = _write_csv(
        destination / "paired_calendar_day_bootstrap.csv", bootstrap_table
    )
    paths["negative_controls"] = _write_csv(
        destination / "negative_controls.csv", negative_table
    )
    paths["stress_outcomes"] = _write_csv(destination / "stress_outcomes.csv", outcome_table)

    portable_reference = reference.frozen
    if reference.frozen.backend == "torch":
        if torch is None:  # pragma: no cover - training could not have succeeded without torch
            raise RuntimeError("Torch reference model cannot be serialized because torch is unavailable")
        portable_model = deepcopy(reference.frozen.model).to("cpu").eval()
        portable_reference = replace(reference.frozen, model=portable_model)
    frozen_artifact_path = destination / "frozen_reference_artifact.joblib"
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "predictor": portable_reference,
            "contrastive_meta_evidence": contrastive_meta,
            "dataset": dataset.manifest["dataset"],
            "feature_names": feature_names,
            "selection_signature_sha256": selection_signature,
            "ablation_selection_signature_sha256": ablation_digest.hexdigest(),
            "predictor_sensitivity_signature_sha256": predictor_sensitivity_digest.hexdigest(),
        },
        frozen_artifact_path,
        compress=3,
    )
    paths["frozen_reference_artifact"] = frozen_artifact_path
    preprocessor_path = destination / "frozen_preprocessor.joblib"
    calibrator_path = destination / "frozen_calibrator.joblib"
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    joblib.dump(reference.frozen.calibrator, calibrator_path, compress=3)
    paths["frozen_preprocessor"] = preprocessor_path
    paths["frozen_calibrator"] = calibrator_path
    native_model_files: list[str] = []
    if reference.frozen.backend == "torch":
        if torch is None:  # pragma: no cover - training could not have succeeded without torch
            raise RuntimeError("Torch reference model cannot be serialized because torch is unavailable")
        native_model_path = destination / "frozen_reference_model_state.pt"
        portable_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in portable_reference.model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": portable_state,
                "family": reference.family,
                "architecture": "TabularResNetV2",
                "architecture_version": getattr(portable_reference.model, "architecture_version", "2.0"),
                "input_dim": int(portable_reference.model.input_dim),
                "feature_names": list(feature_names),
                "training_parameters": reference.frozen.provenance.get("training_parameters", {}),
            },
            native_model_path,
        )
        paths["native_reference_model"] = native_model_path
        native_model_files.append(native_model_path.name)
    elif reference.frozen.backend == "xgboost":
        native_model_path = destination / "frozen_reference_model.json"
        reference.frozen.model.save_model(native_model_path)
        paths["native_reference_model"] = native_model_path
        native_model_files.append(native_model_path.name)
    elif reference.frozen.backend == "lightgbm":
        native_model_path = destination / "frozen_reference_model.txt"
        # LightGBM's Windows C API cannot open some Unicode paths.  The model
        # string is the same native text format and Python writes it safely.
        native_model_path.write_text(
            reference.frozen.model.booster_.model_to_string(), encoding="utf-8"
        )
        paths["native_reference_model"] = native_model_path
        native_model_files.append(native_model_path.name)
    calibration_method = str(
        reference.frozen.provenance.get("validation_protocol", {}).get(
            "selected_calibration_method",
            getattr(getattr(reference.frozen.calibrator, "calibrator", None), "method", "unknown"),
        )
    )
    predictor_manifest_payload = {
        "reference_family": reference.family,
        "reference_backend": reference.frozen.backend,
        "reference_seed": reference.seed,
        "threshold": reference.frozen.threshold,
        "calibration_method": calibration_method,
        "frozen_before_explanation": True,
        "selection_partition": "calibration_select",
        "selection_metric": "pr_auc",
        "selection_split_pr_auc": float(reference.validation_metrics["pr_auc"]),
        "feature_names": list(feature_names),
        "candidate_count": len(candidates),
        "serializations": {
            "joblib_bundle": frozen_artifact_path.name,
            "preprocessor_joblib": preprocessor_path.name,
            "calibrator_joblib": calibrator_path.name,
            "native_model_files": native_model_files,
            "torch_tensor_device": "cpu" if reference.frozen.backend == "torch" else None,
        },
        "training_provenance": _jsonable(reference.frozen.provenance),
    }
    paths["predictor_manifest"] = write_json(
        destination / "predictor_manifest.json", predictor_manifest_payload
    )
    paths["frozen_reference_manifest"] = write_json(
        destination / "frozen_reference_manifest.json",
        {
            "family": reference.family,
            "backend": reference.frozen.backend,
            "seed": reference.seed,
            "threshold": reference.frozen.threshold,
            "calibrator": calibration_method,
            "feature_names": list(feature_names),
            "training_provenance": _jsonable(reference.frozen.provenance),
            "selection_partition": "calibration_select",
            "selection_metric": "pr_auc",
            "predictor_frozen_before_explanation": True,
            "serialization": predictor_manifest_payload["serializations"],
        },
    )
    score_payload: dict[str, np.ndarray] = {
        "reference_test_probability": test_probabilities,
        "reference_test_alert": (test_probabilities >= reference.frozen.threshold).astype(np.uint8),
    }
    for coverage, mask in prelabel_masks.items():
        score_payload[f"selected_coverage_{coverage:g}"] = mask.astype(np.uint8)
    for key, values in all_candidate_test_probabilities.items():
        score_payload[f"test_probability__{key}"] = values
    reference_scores_path = destination / "frozen_reference_scores.npz"
    np.savez_compressed(reference_scores_path, **score_payload)
    paths["frozen_reference_scores"] = reference_scores_path
    paths["claim_assessment"] = write_json(
        destination / "claim_assessment.json",
        {
            "claim_eligible": claim_eligible,
            "claim_blockers": claim_blockers,
            "claim_flags": claim_flags,
            "quick_run": bool(quick_run),
            "test_fixture": bool(use_test_fixture),
            "negative_results_are_reportable": True,
        },
    )

    primary_coverages = {float(value) for value in evaluation.get("primary_coverages", [0.10, 0.25])}
    primary_outcomes = outcome_table.loc[outcome_table["coverage_budget"].isin(primary_coverages)]
    reference_test_row = predictive_metrics.loc[predictive_metrics["reference_selected"]].iloc[0]
    summary = {
        "dataset": dataset.manifest["dataset"],
        "dataset_label": reporting.get("dataset_label", dataset.manifest["dataset"]),
        "reference_predictor": reference.family,
        "reference_backend": reference.frozen.backend,
        "reference_seed": reference.seed,
        "reference_test_pr_auc": float(reference_test_row["pr_auc"]),
        "reference_test_roc_auc": float(reference_test_row["roc_auc"]),
        "audited_rules": int(len(completed_audit)),
        "selected_rules": int(completed_audit["selected"].sum()),
        "policy_abstentions": int(sum(policy.abstain for policy in policies)),
        "primary_outcomes": primary_outcomes[["coverage_budget", "outcome"]].to_dict("records"),
        "outcome_decision_thresholds": {
            "saturation_headroom": float(evaluation["saturation_headroom"]),
            "saturation_precision_margin": float(
                evaluation["saturation_precision_margin"]
            ),
            "saturation_evidence_margin": float(
                evaluation["saturation_evidence_margin"]
            ),
            "minimum_positive_stability_fraction": float(
                evaluation["minimum_positive_stability_fraction"]
            ),
            "minimum_saturation_alert_positives": int(
                evaluation["minimum_saturation_alert_positives"]
            ),
            "minimum_saturation_matched_pairs": int(
                evaluation["minimum_saturation_matched_pairs"]
            ),
            "minimum_saturation_bootstrap_replicates": int(
                evaluation["minimum_saturation_bootstrap_replicates"]
            ),
        },
        "selection_signature_sha256": selection_signature,
        "ablation_selection_signature_sha256": ablation_digest.hexdigest(),
        "predictor_sensitivity_signature_sha256": predictor_sensitivity_digest.hexdigest(),
        "ablation_variants": sorted(ablation_results["variant"].unique().tolist()),
        "independent_ablation_rule_pools": _jsonable(ablation_rule_pool_provenance),
        "predictor_sensitivity_candidates": sorted(
            predictor_sensitivity["candidate"].unique().tolist()
        ),
        "predictor_sensitivity_rule_set_origin": "reference_predictor_train_and_rule_audit",
        "evidence_methods": evidence_method_names,
        "contrastive_meta": {
            "available": bool(contrastive_meta.available),
            "fallback_reason": contrastive_meta.fallback_reason,
            "fit_partition": contrastive_meta.provenance.get("fit_partition"),
            "test_data_used_for_fit": False,
        },
        "rule_audit_scope": {
            "scope": "frozen_predictor_alerts_only",
            "train_alert_count": int(train_alert_mask.sum()),
            "rule_audit_alert_count": int(audit_alert_mask.sum()),
            "rule_audit_tp_alert_count": audit_alert_class_counts["tp"],
            "rule_audit_fp_alert_count": audit_alert_class_counts["fp"],
            "non_alert_rows_used_for_rule_selection_or_weighting": False,
        },
        "guarded_ensemble": {
            "aggregation": "arithmetic_mean",
            "components": guarded_components,
            "pre_registered_before_test": True,
            "selected_only_on_policy_select": True,
        },
        "fixed_budget_fairness": {
            "requested_budget_reported_separately": True,
            "core_score_only_comparator": "matched_realized_selected_count",
            "unsupported_alerts_used_to_fill_budget": False,
            "mandatory_shuffled_controls": list(_MANDATORY_SHUFFLED_CONTROLS),
            "claim_gate_requires_equal_realized_counts": True,
        },
        "attribution_top_k_baseline": {
            "k": 5,
            "formula": "sum_positive_among_top5_absolute_native_attributions",
            "signed_direction": "positive_raw_fraud_logit_contribution_only",
            "methods": "tree_native_pred_contribs_or_neural_gradient_times_input",
            "test_labels_used": False,
        },
        "tier_c_primary_evidence": False,
        "test_label_permutation_invariance_asserted": True,
        "metric_definitions": {
            "supported_alert_rate": (
                "fraction of frozen-predictor alerts activating at least one validation-selected rule"
            ),
            "unsupported_alert_rate": (
                "fraction of frozen-predictor alerts activating no validation-selected rule"
            ),
            "rule_sparsity_alerts": (
                "one minus the mean fraction of validation-selected rules active per predictor alert"
            ),
            "contradiction_rate": (
                "descriptive XOR rate between the frozen predictor alert flag and any selected-rule "
                "activation over all test rows; it is not a causal contradiction measure"
            ),
            "evidence_without_alert_rate": (
                "fraction of rows with selected-rule activation that are outside the predictor alert region"
            ),
            "delta_vs_score_only": (
                "selected-alert precision minus predictor-score-only precision at the identical "
                "realized selected count; requested-budget score-only is diagnostic only"
            ),
            "abstention_rate": (
                "fraction of frozen-predictor alerts not selected for an explanation; this is distinct "
                "from the policy-level guardrail abstain flag"
            ),
            "selected_recall_test_positives": (
                "selected positive rows divided by all positive rows in the locked test partition"
            ),
            "selected_recall_alert_positives": (
                "selected positive rows divided by positive frozen-predictor alerts"
            ),
        },
        "claim_boundary": (
            "Stress-test outcomes concern robustness, saturation and incremental rule evidence under the "
            "locked protocol; they do not establish causality or production generalization."
        ),
    }
    paths["stress_summary"] = write_json(
        destination / "stress_summary.json", _jsonable(summary)
    )
    manifest_inputs = list(paths.values())
    paths["stress_test_manifest"] = write_stress_manifest(
        destination,
        dataset=dataset.manifest["dataset"],
        config_path=config_file,
        project_root=project_root,
        claim_eligible=claim_eligible,
        claim_blockers=claim_blockers,
        data_manifest=dataset.manifest,
        summary=_jsonable(summary),
        output_files=manifest_inputs,
    )
    paths["stress_lineage"] = write_stress_lineage(
        destination,
        notebook_id=str(config.get("project", {}).get("experiment", dataset.manifest["dataset"])),
        config_path=config_file,
        project_root=project_root,
        output_files=[*manifest_inputs, paths["stress_test_manifest"]],
    )
    return {
        "output_dir": destination,
        "summary": summary,
        "claim_eligible": claim_eligible,
        "claim_blockers": claim_blockers,
        "paths": paths,
    }


__all__ = ["PROTOCOL_PHASES", "run_stress_test"]
