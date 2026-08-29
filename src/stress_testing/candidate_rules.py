"""Leakage-safe candidate-rule generation for the thesis stress tests.

The functions in this module deliberately expose *training data only*.  They
cover the three candidate sources required by the thesis protocol:

* Tier A/B rules declared in configuration and fitted by
  :class:`~src.logic.fraud_rules.FraudRuleEngine` on the training split;
* model-counterfactual Tier B candidates, discovered by replacing one raw
  numeric feature at a time with its training median and retaining only tails
  whose frozen-predictor score materially decreases; and
* Tier C shallow-CART surrogate paths, fitted to frozen-predictor outputs on
  the transformed training matrix (never to ground-truth test outcomes).

The predictor-score-only comparator is represented explicitly in the registry
so downstream selection code cannot accidentally describe it as a rule-based
explanation.  Candidate generation does not audit, weight, select, or test a
rule; those later stages belong to the reserved validation and test protocols.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.logic.fraud_rules import FraudRuleEngine


_CONFIGURED_TIERS = {"A", "B"}
_SCORE_MODES = {"raw", "calibrated"}
_DERIVED_FORMULA_ARITY = {
    "log1p_nonnegative": 1,
    "absolute_relative_difference": 2,
    "amount_to_absolute_balance_fraction": 2,
}
NON_INTERVENABLE_HISTORY_DERIVED_FEATURES = frozenset(
    {
        "sender_seconds_since_previous",
        "receiver_seconds_since_previous",
        "pair_prior_transaction_count",
    }
)
_KNOWN_SEMANTIC_DERIVED_FEATURES = frozenset(
    {
        "amount_paid_log",
        "amount_received_log",
        "amount_relative_difference",
        "amount_log",
        "amount_to_sender_balance_fraction",
    }
)
_KNOWN_DERIVED_FEATURES = (
    NON_INTERVENABLE_HISTORY_DERIVED_FEATURES | _KNOWN_SEMANTIC_DERIVED_FEATURES
)


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/pandas scalars into artifact-safe Python values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _fingerprint_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fingerprint_definitions(definitions: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(_jsonable(list(definitions)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_frozen_predictor(predictor: Any) -> None:
    if not bool(getattr(predictor, "frozen", False)):
        raise TypeError("candidate generation requires a FrozenPredictor")
    required = ("feature_names", "threshold", "predict_raw_proba", "predict_proba")
    missing = [name for name in required if not hasattr(predictor, name)]
    if missing:
        raise TypeError(f"frozen predictor is missing required attributes: {missing}")


def _predict_scores(
    predictor: Any,
    matrix: np.ndarray,
    *,
    score_mode: str,
    device: str | None,
) -> np.ndarray:
    _require_frozen_predictor(predictor)
    normalized = str(score_mode).lower()
    if normalized not in _SCORE_MODES:
        raise ValueError(f"score_mode must be one of {sorted(_SCORE_MODES)}")
    method = predictor.predict_raw_proba if normalized == "raw" else predictor.predict_proba
    scores = method(matrix) if device is None else method(matrix, device=device)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
        raise RuntimeError("frozen predictor returned invalid or misaligned scores")
    return np.clip(scores, 0.0, 1.0)


def _safe_rule_name(prefix: str, feature: str, used: set[str]) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(feature).lower()).strip("_") or "feature"
    candidate = f"{prefix}_{token}"
    if candidate in used:
        suffix = hashlib.sha256(str(feature).encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate}_{suffix}"
    if candidate in used:
        raise ValueError(f"could not create a unique candidate name for feature {feature!r}")
    used.add(candidate)
    return candidate


def _validate_rule_definitions(
    rule_definitions: Sequence[Mapping[str, Any]],
    *,
    default_tier: str,
) -> tuple[dict[str, Any], ...]:
    if not rule_definitions:
        raise ValueError("at least one configured rule definition is required")
    default = str(default_tier).upper()
    if default not in _CONFIGURED_TIERS:
        raise ValueError("default_tier must be 'A' or 'B'")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rule_definitions:
        definition = copy.deepcopy(dict(raw))
        name = str(definition.get("name", "")).strip()
        if not name:
            raise ValueError("every configured rule requires a non-empty name")
        if name in seen:
            raise ValueError(f"configured rule names must be unique: {name}")
        seen.add(name)
        tier = str(definition.get("tier", default)).upper()
        if tier not in _CONFIGURED_TIERS:
            raise ValueError(f"configured rule {name!r} has unsupported tier {tier!r}")
        conditions = definition.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"configured rule {name!r} requires non-empty conditions")
        for condition in conditions:
            if not isinstance(condition, Mapping):
                raise TypeError(f"configured rule {name!r} has a non-mapping condition")
            missing = {"feature", "operator", "value"} - set(condition)
            if missing:
                raise KeyError(f"configured rule {name!r} condition is missing {sorted(missing)}")
        definition["name"] = name
        definition["tier"] = tier
        definition.setdefault("description", name.replace("_", " "))
        normalized.append(definition)
    return tuple(normalized)


def configured_rule_registry(
    rule_definitions: Sequence[Mapping[str, Any]],
    *,
    fitted_engine: FraudRuleEngine | None = None,
    default_tier: str = "A",
) -> pd.DataFrame:
    """Convert configured Tier A/B definitions into an auditable registry.

    When ``fitted_engine`` is supplied, train-fitted thresholds, numeric
    imputation values, and skipped-rule reasons are copied into the registry.
    The function never fits against validation or test data.
    """

    definitions = _validate_rule_definitions(rule_definitions, default_tier=default_tier)
    fitted_by_name = {}
    skipped: Mapping[str, str] = {}
    if fitted_engine is not None:
        fitted_by_name = {rule.name: rule for rule in fitted_engine.rules}
        skipped = fitted_engine.skipped_rules

    rows: list[dict[str, Any]] = []
    for definition in definitions:
        name = str(definition["name"])
        condition_features = [
            str(item["feature"]) for item in definition["conditions"]
        ]
        derived_features = sorted(set(condition_features) & _KNOWN_DERIVED_FEATURES)
        raw_features = sorted(set(condition_features) - set(derived_features))
        if derived_features and raw_features:
            feature_space = "mixed_raw_and_causally_derived_features"
        elif derived_features:
            feature_space = "causally_derived_features"
        else:
            feature_space = "raw_causal_features"
        fitted = fitted_by_name.get(name)
        fitted_conditions: list[dict[str, Any]] = []
        if fitted is not None:
            for condition in fitted.conditions:
                fitted_conditions.append(
                    {
                        "feature": condition.feature,
                        "operator": condition.operator,
                        "configured_value": _jsonable(condition.configured_value),
                        "fitted_threshold": _jsonable(condition.threshold),
                        "softness": float(condition.softness),
                        "softness_mode": condition.softness_mode,
                        "fitted_missing_value": _jsonable(condition.numeric_fill_value),
                    }
                )
        if fitted is not None:
            status = "fitted"
        elif name in skipped:
            status = "skipped"
        else:
            status = "declared"
        rows.append(
            {
                "rule": name,
                "tier": str(definition["tier"]),
                "candidate_kind": "configured_domain_rule",
                "source": str(definition.get("source", "yaml_configuration")),
                "description": str(definition["description"]),
                "features": condition_features,
                "raw_features": raw_features,
                "derived_features": derived_features,
                "feature_space": feature_space,
                "conditions": _jsonable(definition["conditions"]),
                "fitted_conditions": fitted_conditions,
                "fit_partition": "train",
                "status": status,
                "skip_reason": skipped.get(name),
                "requires_rule_evidence": True,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ConfiguredCandidateResult:
    """Train-fitted configured candidates and their audit provenance."""

    engine: FraudRuleEngine
    definitions: tuple[dict[str, Any], ...]
    registry: pd.DataFrame
    provenance: dict[str, Any]

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.engine.evaluate(frame)


def fit_configured_candidates(
    train_frame: pd.DataFrame,
    target_column: str,
    rule_definitions: Sequence[Mapping[str, Any]],
    *,
    conjunction: str = "minimum",
    default_tier: str = "A",
) -> ConfiguredCandidateResult:
    """Fit configured Tier A/B thresholds using the named training frame."""

    if target_column not in train_frame:
        raise KeyError(f"training target column is missing: {target_column}")
    definitions = _validate_rule_definitions(rule_definitions, default_tier=default_tier)
    engine = FraudRuleEngine([copy.deepcopy(item) for item in definitions], conjunction=conjunction)
    engine.fit(train_frame, target_column)
    registry = configured_rule_registry(
        definitions, fitted_engine=engine, default_tier=default_tier
    )
    target = train_frame[target_column].to_numpy(dtype=np.int8)
    provenance = {
        "protocol": "configured-train-only-v1",
        "fit_partition": "train",
        "training_rows": int(len(train_frame)),
        "target_column": str(target_column),
        "training_target_sha256": _fingerprint_array(target),
        "definitions_sha256": _fingerprint_definitions(definitions),
        "fitted_rule_count": int(len(engine.rules)),
        "skipped_rule_count": int(len(engine.skipped_rules)),
        "configured_derived_features": sorted(
            {
                str(condition["feature"])
                for definition in definitions
                for condition in definition["conditions"]
                if str(condition["feature"]) in _KNOWN_DERIVED_FEATURES
            }
        ),
        "configured_history_features_are_direct_interventions": False,
        "validation_or_test_data_used": False,
    }
    return ConfiguredCandidateResult(engine, definitions, registry, provenance)


@dataclass
class CounterfactualCandidateResult:
    """Train-only model-counterfactual definitions and diagnostics."""

    definitions: tuple[dict[str, Any], ...]
    registry: pd.DataFrame
    diagnostics: pd.DataFrame
    provenance: dict[str, Any]


def _deterministic_positions(row_count: int, max_rows: int | None) -> np.ndarray:
    if row_count < 1:
        raise ValueError("training frame must contain at least one row")
    if max_rows is None or int(max_rows) >= row_count:
        return np.arange(row_count, dtype=int)
    if int(max_rows) < 2:
        raise ValueError("max_rows must be at least two when sampling is enabled")
    # Systematic positions are independent of labels and reproducible across
    # platforms.  ``unique`` protects against floating-point rounding repeats.
    positions = np.unique(np.linspace(0, row_count - 1, num=int(max_rows), dtype=int))
    return positions


def _normalize_derived_feature_contract(
    contract: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    frame_columns: Sequence[str],
    intervention_features: Sequence[str],
) -> tuple[dict[str, tuple[dict[str, Any], ...]], tuple[dict[str, Any], ...]]:
    """Validate and normalize deterministic derived-feature recomputations.

    The contract is deliberately declarative: intervention features map to
    derived outputs, named formulas, and their ordered input columns.  It does
    not accept callables or expressions, so a serialized contract is both the
    executable specification and an auditable record of the perturbation.
    """

    if contract is None:
        return {}, ()
    if not isinstance(contract, Mapping):
        raise TypeError("derived_feature_contract must be a mapping or None")

    columns = {str(column) for column in frame_columns}
    candidate_set = {str(feature) for feature in intervention_features}
    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    canonical_by_output: dict[str, dict[str, Any]] = {}

    for raw_intervention, raw_specs in sorted(contract.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_intervention, str) or not raw_intervention.strip():
            raise ValueError("derived-feature intervention names must be non-empty strings")
        intervention = raw_intervention.strip()
        if intervention not in columns:
            raise KeyError(
                f"derived-feature intervention is missing from training frame: {intervention}"
            )
        if isinstance(raw_specs, (str, bytes)) or not isinstance(raw_specs, Sequence):
            raise TypeError(
                f"derived-feature contract for {intervention!r} must be a sequence of mappings"
            )
        if not raw_specs:
            raise ValueError(
                f"derived-feature contract for {intervention!r} must not be empty"
            )

        intervention_specs: list[dict[str, Any]] = []
        outputs_for_intervention: set[str] = set()
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, Mapping):
                raise TypeError(
                    f"derived-feature contract for {intervention!r} contains a non-mapping spec"
                )
            unexpected = set(raw_spec) - {"output", "formula", "inputs"}
            missing = {"output", "formula", "inputs"} - set(raw_spec)
            if unexpected or missing:
                raise ValueError(
                    f"derived-feature spec for {intervention!r} must contain exactly "
                    f"output/formula/inputs; missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            if not isinstance(raw_spec["output"], str) or not raw_spec["output"].strip():
                raise ValueError("derived-feature output names must be non-empty strings")
            if not isinstance(raw_spec["formula"], str) or not raw_spec["formula"].strip():
                raise ValueError("derived-feature formula names must be non-empty strings")
            output = raw_spec["output"].strip()
            formula = raw_spec["formula"].strip()
            raw_inputs = raw_spec["inputs"]
            if output in outputs_for_intervention:
                raise ValueError(
                    f"duplicate derived output {output!r} for intervention {intervention!r}"
                )
            if formula not in _DERIVED_FORMULA_ARITY:
                raise ValueError(
                    f"unsupported derived-feature formula {formula!r}; expected one of "
                    f"{sorted(_DERIVED_FORMULA_ARITY)}"
                )
            if isinstance(raw_inputs, (str, bytes)) or not isinstance(raw_inputs, Sequence):
                raise TypeError("derived-feature inputs must be a sequence of column names")
            if any(not isinstance(item, str) for item in raw_inputs):
                raise ValueError("derived-feature inputs must be non-empty strings")
            inputs = tuple(item.strip() for item in raw_inputs)
            if len(inputs) != _DERIVED_FORMULA_ARITY[formula]:
                raise ValueError(
                    f"formula {formula!r} requires {_DERIVED_FORMULA_ARITY[formula]} inputs"
                )
            if any(not item for item in inputs) or len(set(inputs)) != len(inputs):
                raise ValueError("derived-feature inputs must be non-empty and unique")
            if intervention not in inputs:
                raise ValueError(
                    f"derived output {output!r} does not depend on mapped intervention "
                    f"{intervention!r}"
                )
            if output in inputs:
                raise ValueError("a derived-feature output cannot also be one of its inputs")
            missing_columns = ({output, *inputs} - columns)
            if missing_columns:
                raise KeyError(
                    "derived-feature contract references columns missing from training frame: "
                    f"{sorted(missing_columns)}"
                )

            spec = {"output": output, "formula": formula, "inputs": list(inputs)}
            prior = canonical_by_output.get(output)
            if prior is not None and prior != spec:
                raise ValueError(
                    f"derived output {output!r} has conflicting recomputation specifications"
                )
            canonical_by_output[output] = spec
            outputs_for_intervention.add(output)
            intervention_specs.append(spec)
        normalized[intervention] = tuple(
            sorted(intervention_specs, key=lambda item: (item["output"], item["formula"]))
        )

    derived_outputs = set(canonical_by_output)
    direct_derived_interventions = candidate_set & derived_outputs
    if direct_derived_interventions:
        raise ValueError(
            "counterfactual candidates cannot directly intervene on declared derived "
            f"outputs: {sorted(direct_derived_interventions)}"
        )
    nested_dependencies = {
        input_name
        for spec in canonical_by_output.values()
        for input_name in spec["inputs"]
        if input_name in derived_outputs
    }
    if nested_dependencies:
        raise ValueError(
            "derived-feature contracts are single-pass; derived outputs cannot be formula "
            f"inputs: {sorted(nested_dependencies)}"
        )

    # Every candidate raw input must list the same dependent formula.  This
    # prevents an intervention on the second operand from silently leaving a
    # multi-input semantic feature stale.
    for output, spec in canonical_by_output.items():
        for input_name in spec["inputs"]:
            if input_name not in candidate_set:
                continue
            if spec not in normalized.get(input_name, ()):
                raise ValueError(
                    f"derived output {output!r} must be registered for candidate "
                    f"intervention {input_name!r}"
                )

    unique_specs = tuple(
        canonical_by_output[output] for output in sorted(canonical_by_output)
    )
    return normalized, unique_specs


def _recompute_derived_features(
    frame: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return a copy with approved derived formulas recomputed deterministically."""

    result = frame.copy()
    epsilon = np.finfo("float64").eps
    for spec in specs:
        output = str(spec["output"])
        formula = str(spec["formula"])
        inputs = [str(item) for item in spec["inputs"]]
        if formula == "log1p_nonnegative":
            values = pd.to_numeric(result[inputs[0]], errors="coerce").astype("float64")
            result[output] = np.log1p(values.clip(lower=0.0))
        elif formula == "absolute_relative_difference":
            left = pd.to_numeric(result[inputs[0]], errors="coerce").astype("float64")
            right = pd.to_numeric(result[inputs[1]], errors="coerce").astype("float64")
            denominator = left.abs() + right.abs() + epsilon
            result[output] = ((left - right).abs() / denominator).astype("float64")
        elif formula == "amount_to_absolute_balance_fraction":
            amount = pd.to_numeric(result[inputs[0]], errors="coerce").astype("float64")
            balance = pd.to_numeric(result[inputs[1]], errors="coerce").astype("float64")
            denominator = balance.abs().clip(lower=epsilon)
            ratio = (amount / denominator).replace([np.inf, -np.inf], np.nan)
            result[output] = ratio.fillna(0.0).clip(0.0, 10.0)
        else:  # pragma: no cover - normalization rejects unknown formulas.
            raise RuntimeError(f"unvalidated derived-feature formula: {formula}")
    return result


def generate_counterfactual_numeric_candidates(
    train_frame: pd.DataFrame,
    preprocessor: Any,
    frozen_predictor: Any,
    *,
    numeric_features: Sequence[str] | None = None,
    quantile: float = 0.90,
    softness: float = 0.15,
    min_mean_score_drop: float = 0.005,
    min_affected_fraction: float = 0.05,
    per_row_material_drop: float | None = None,
    min_tail_rows: int = 25,
    max_candidates: int = 8,
    max_rows: int | None = 100_000,
    score_mode: str = "raw",
    device: str | None = None,
    derived_feature_contract: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    non_intervenable_derived_features: Sequence[str] | None = None,
) -> CounterfactualCandidateResult:
    """Generate Tier B candidates by train-median counterfactual perturbation.

    For every eligible raw numeric feature, the function replaces that feature
    with its median fitted on ``train_frame``, recomputes any pre-registered
    dependent semantic features, transforms the coherent perturbed rows with
    the already-fitted preprocessor, and measures the decrease in the
    already-frozen predictor score.  The upper and lower training tails are
    compared, and only a tail with a positive signed mean decrease and enough
    materially affected rows becomes a quantile rule.  No outcome labels are
    accepted by this API.  ``derived_feature_contract`` accepts only the
    deterministic formula names validated by this module; expressions and
    callables are intentionally unsupported. Causal history summaries are not
    valid single-row interventions: they must be recomputed from a perturbed
    prior transaction sequence, which this API does not fabricate. Explicit
    attempts to intervene on such fields therefore fail closed; automatic
    feature discovery excludes them.
    """

    _require_frozen_predictor(frozen_predictor)
    if not hasattr(preprocessor, "transform"):
        raise TypeError("preprocessor must be fitted and expose transform(frame)")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between 0.5 and 1.0")
    if float(softness) <= 0.0:
        raise ValueError("softness must be positive")
    if float(min_mean_score_drop) < 0.0 or float(min_affected_fraction) < 0.0:
        raise ValueError("counterfactual materiality thresholds must be non-negative")
    if not 0.0 <= float(min_affected_fraction) <= 1.0:
        raise ValueError("min_affected_fraction must lie in [0, 1]")
    if int(min_tail_rows) < 1 or int(max_candidates) < 1:
        raise ValueError("min_tail_rows and max_candidates must be positive")
    normalized_mode = str(score_mode).lower()
    if normalized_mode not in _SCORE_MODES:
        raise ValueError(f"score_mode must be one of {sorted(_SCORE_MODES)}")

    selected_features = set(getattr(preprocessor, "selected_features", train_frame.columns))
    if isinstance(non_intervenable_derived_features, (str, bytes)):
        raise TypeError("non_intervenable_derived_features must be a sequence of names")
    declared_non_intervenable: list[str] = []
    for value in non_intervenable_derived_features or ():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "non_intervenable_derived_features must contain non-empty strings"
            )
        declared_non_intervenable.append(value.strip())
    if len(set(declared_non_intervenable)) != len(declared_non_intervenable):
        raise ValueError("non_intervenable_derived_features must be unique")
    forbidden_direct_interventions = (
        set(declared_non_intervenable) | set(_KNOWN_DERIVED_FEATURES)
    )
    automatically_excluded: list[str] = []
    if numeric_features is None:
        declared_numeric = getattr(preprocessor, "numeric_features", None)
        if declared_numeric is None:
            declared_numeric = [
                column
                for column in train_frame.columns
                if pd.api.types.is_numeric_dtype(train_frame[column])
            ]
        discovered = [str(column) for column in declared_numeric]
        automatically_excluded = sorted(
            set(discovered) & forbidden_direct_interventions
        )
        candidates = [
            feature for feature in discovered if feature not in forbidden_direct_interventions
        ]
    else:
        candidates = [str(column) for column in numeric_features]
        forbidden_requested = sorted(set(candidates) & forbidden_direct_interventions)
        if forbidden_requested:
            history_requested = sorted(
                set(forbidden_requested)
                & set(NON_INTERVENABLE_HISTORY_DERIVED_FEATURES)
            )
            if history_requested:
                raise ValueError(
                    "direct counterfactual intervention on causal history-derived features "
                    f"is forbidden: {history_requested}"
                )
            raise ValueError(
                "direct counterfactual intervention on known or declared derived features "
                f"is forbidden: {forbidden_requested}"
            )
    if not candidates:
        raise ValueError(
            "at least one raw numeric feature is required for counterfactual generation"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("numeric_features must be unique")

    normalized_contract, baseline_derived_specs = _normalize_derived_feature_contract(
        derived_feature_contract,
        frame_columns=train_frame.columns,
        intervention_features=candidates,
    )

    def recomputation_diagnostics(feature: str) -> dict[str, Any]:
        specs = normalized_contract.get(feature, ())
        return {
            "intervention_feature_space": "raw_transaction_feature",
            "direct_intervention_is_derived": False,
            "baseline_derived_features_recomputed": bool(baseline_derived_specs),
            "derived_recomputation_count": int(len(specs)),
            "derived_recomputation_specs": [_jsonable(spec) for spec in specs],
            "derived_recomputed_outputs": [str(spec["output"]) for spec in specs],
            "derived_recomputation_formulas": [str(spec["formula"]) for spec in specs],
        }

    positions = _deterministic_positions(len(train_frame), max_rows)
    sampled = train_frame.iloc[positions].copy()
    sampled = _recompute_derived_features(sampled, baseline_derived_specs)
    baseline_matrix = np.asarray(preprocessor.transform(sampled), dtype=np.float32)
    expected_names = tuple(str(name) for name in frozen_predictor.feature_names)
    if hasattr(preprocessor, "get_feature_names_out"):
        transformed_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
        if transformed_names != expected_names:
            raise ValueError(
                "preprocessor feature names/order do not match the frozen predictor"
            )
    if baseline_matrix.ndim != 2 or baseline_matrix.shape[1] != len(expected_names):
        raise ValueError("preprocessor output does not align with frozen predictor features")
    baseline_scores = _predict_scores(
        frozen_predictor, baseline_matrix, score_mode=normalized_mode, device=device
    )
    material_drop = (
        float(min_mean_score_drop)
        if per_row_material_drop is None
        else float(per_row_material_drop)
    )
    if material_drop < 0.0:
        raise ValueError("per_row_material_drop must be non-negative")

    diagnostics: list[dict[str, Any]] = []
    viable: list[dict[str, Any]] = []
    for feature in candidates:
        reason: str | None = None
        if feature not in train_frame:
            reason = "missing_from_training_frame"
        elif feature not in selected_features:
            reason = "not_selected_by_preprocessor"
        full_numeric = (
            pd.to_numeric(train_frame[feature], errors="coerce")
            if reason is None
            else pd.Series(dtype=float)
        )
        finite_full = full_numeric[np.isfinite(full_numeric.to_numpy(dtype=float))]
        if reason is None and (finite_full.empty or finite_full.nunique() < 2):
            reason = "non_varying_or_non_numeric"
        if reason is not None:
            diagnostics.append(
                {
                    "feature": feature,
                    "eligible": False,
                    "selected": False,
                    "reason": reason,
                    **recomputation_diagnostics(feature),
                }
            )
            continue

        median = float(finite_full.median())
        high_threshold = float(finite_full.quantile(float(quantile)))
        low_quantile = 1.0 - float(quantile)
        low_threshold = float(finite_full.quantile(low_quantile))
        perturbed = sampled.copy()
        sampled_numeric = pd.to_numeric(sampled[feature], errors="coerce")
        observed = np.isfinite(sampled_numeric.to_numpy(dtype=float))
        perturbed.loc[observed, feature] = median
        perturbed = _recompute_derived_features(
            perturbed, normalized_contract.get(feature, ())
        )
        perturbed_matrix = np.asarray(preprocessor.transform(perturbed), dtype=np.float32)
        perturbed_scores = _predict_scores(
            frozen_predictor, perturbed_matrix, score_mode=normalized_mode, device=device
        )
        delta = baseline_scores - perturbed_scores
        sampled_values = sampled_numeric.to_numpy(dtype=float)
        tail_specs = (
            (
                "high",
                "greater_quantile",
                float(quantile),
                high_threshold,
                observed & (sampled_values >= high_threshold),
            ),
            (
                "low",
                "less_quantile",
                low_quantile,
                low_threshold,
                observed & (sampled_values <= low_threshold),
            ),
        )
        tail_rows: list[dict[str, Any]] = []
        for direction, operator, configured_value, threshold, mask in tail_specs:
            count = int(mask.sum())
            signed_mean = float(delta[mask].mean()) if count else float("nan")
            positive_mean = float(np.maximum(delta[mask], 0.0).mean()) if count else float("nan")
            affected_fraction = (
                float((delta[mask] >= material_drop).mean()) if count else float("nan")
            )
            tail_rows.append(
                {
                    "direction": direction,
                    "operator": operator,
                    "configured_value": configured_value,
                    "fitted_threshold": threshold,
                    "tail_rows": count,
                    "tail_mean_score_drop": signed_mean,
                    "tail_mean_positive_score_drop": positive_mean,
                    "tail_affected_fraction": affected_fraction,
                }
            )
        chosen = sorted(
            tail_rows,
            key=lambda row: (
                -np.nan_to_num(float(row["tail_mean_score_drop"]), nan=-np.inf),
                str(row["direction"]),
            ),
        )[0]
        eligible = bool(
            int(chosen["tail_rows"]) >= int(min_tail_rows)
            and np.isfinite(float(chosen["tail_mean_score_drop"]))
            and float(chosen["tail_mean_score_drop"]) >= float(min_mean_score_drop)
            and np.isfinite(float(chosen["tail_affected_fraction"]))
            and float(chosen["tail_affected_fraction"]) >= float(min_affected_fraction)
        )
        record = {
            "feature": feature,
            "eligible": eligible,
            "selected": False,
            "reason": "material_score_drop" if eligible else "below_materiality_gate",
            "train_median": median,
            "score_mode": normalized_mode,
            "perturbation": "replace_observed_value_with_train_median",
            "sample_rows": int(len(sampled)),
            **recomputation_diagnostics(feature),
            "overall_mean_score_drop": (
                float(delta[observed].mean()) if observed.any() else float("nan")
            ),
            **chosen,
            "upper_tail_mean_score_drop": float(tail_rows[0]["tail_mean_score_drop"]),
            "lower_tail_mean_score_drop": float(tail_rows[1]["tail_mean_score_drop"]),
        }
        diagnostics.append(record)
        if eligible:
            viable.append(record)

    viable.sort(
        key=lambda row: (
            -float(row["tail_mean_score_drop"]),
            -float(row["tail_affected_fraction"]),
            str(row["feature"]),
        )
    )
    selected = viable[: int(max_candidates)]
    selected_features_set = {str(item["feature"]) for item in selected}
    for row in diagnostics:
        if str(row["feature"]) in selected_features_set:
            row["selected"] = True
            row["reason"] = "selected"
        elif bool(row.get("eligible", False)):
            row["reason"] = "candidate_budget_exceeded"

    used_names: set[str] = set()
    definitions: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for row in selected:
        direction = str(row["direction"])
        feature = str(row["feature"])
        name = _safe_rule_name(f"counterfactual_{direction}", feature, used_names)
        condition = {
            "feature": feature,
            "operator": str(row["operator"]),
            "value": float(row["configured_value"]),
            "softness": float(softness),
        }
        definition = {
            "name": name,
            "tier": "B",
            "source": "train_model_counterfactual",
            "description": (
                f"{feature} lies in the training {direction} tail and replacing it with "
                "the training median materially lowers the frozen predictor score."
            ),
            "conditions": [condition],
        }
        definitions.append(definition)
        registry_rows.append(
            {
                "rule": name,
                "tier": "B",
                "candidate_kind": "model_counterfactual_numeric",
                "source": "train_model_counterfactual",
                "description": definition["description"],
                "features": [feature],
                "feature_space": "raw_causal_features",
                "raw_features": [feature],
                "derived_features": [],
                "direct_intervention_is_derived": False,
                "conditions": [condition],
                "fitted_conditions": [
                    {
                        **condition,
                        "configured_value": float(row["configured_value"]),
                        "fitted_threshold": float(row["fitted_threshold"]),
                        "train_median": float(row["train_median"]),
                    }
                ],
                "fit_partition": "train",
                "status": "generated",
                "requires_rule_evidence": True,
                "score_mode": normalized_mode,
                "counterfactual_mean_score_drop": float(row["tail_mean_score_drop"]),
                "counterfactual_affected_fraction": float(row["tail_affected_fraction"]),
                "counterfactual_tail_rows": int(row["tail_rows"]),
            }
        )
    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(
        ["selected", "eligible", "feature"], ascending=[False, False, True]
    ).reset_index(drop=True)
    registry = pd.DataFrame(
        registry_rows,
        columns=[
            "rule",
            "tier",
            "candidate_kind",
            "source",
            "description",
            "features",
            "feature_space",
            "raw_features",
            "derived_features",
            "direct_intervention_is_derived",
            "conditions",
            "fitted_conditions",
            "fit_partition",
            "status",
            "requires_rule_evidence",
            "score_mode",
            "counterfactual_mean_score_drop",
            "counterfactual_affected_fraction",
            "counterfactual_tail_rows",
        ],
    )
    provenance = {
        "protocol": "single-feature-train-median-counterfactual-v2",
        "derived_feature_recomputation_protocol": (
            "pre_registered_deterministic_formulas-v1"
        ),
        "derived_feature_recomputation_contract": {
            intervention: [_jsonable(spec) for spec in specs]
            for intervention, specs in normalized_contract.items()
        },
        "derived_feature_recomputed_outputs": [
            str(spec["output"]) for spec in baseline_derived_specs
        ],
        "baseline_derived_features_recomputed": bool(baseline_derived_specs),
        "direct_intervention_feature_space": "raw_transaction_features_only",
        "direct_derived_feature_intervention_used": False,
        "non_intervenable_derived_features": sorted(forbidden_direct_interventions),
        "automatically_excluded_derived_features": automatically_excluded,
        "arbitrary_code_execution_used": False,
        "fit_partition": "train",
        "training_rows": int(len(train_frame)),
        "counterfactual_sample_rows": int(len(sampled)),
        "sampling": (
            "all_train_rows"
            if len(sampled) == len(train_frame)
            else "deterministic_systematic_train_sample"
        ),
        "sample_positions_sha256": _fingerprint_array(positions),
        "baseline_scores_sha256": _fingerprint_array(baseline_scores),
        "score_mode": normalized_mode,
        "quantile": float(quantile),
        "lower_quantile": 1.0 - float(quantile),
        "min_mean_score_drop": float(min_mean_score_drop),
        "per_row_material_drop": material_drop,
        "min_affected_fraction": float(min_affected_fraction),
        "min_tail_rows": int(min_tail_rows),
        "candidate_budget": int(max_candidates),
        "eligible_feature_count": int(len(viable)),
        "selected_candidate_count": int(len(definitions)),
        "predictor_family": str(getattr(frozen_predictor, "family", "unknown")),
        "validation_or_test_labels_used": False,
    }
    return CounterfactualCandidateResult(
        tuple(definitions), registry, diagnostics_frame, provenance
    )


@dataclass(frozen=True)
class PathCondition:
    """One finite condition in transformed CART feature space."""

    feature: str
    feature_index: int
    operator: str
    threshold: float

    def evaluate(self, matrix: np.ndarray) -> np.ndarray:
        values = matrix[:, self.feature_index]
        if self.operator == "less_or_equal":
            return values <= self.threshold
        if self.operator == "greater":
            return values > self.threshold
        raise ValueError(f"unsupported path operator: {self.operator}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_index": int(self.feature_index),
            "operator": self.operator,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class SurrogatePathRule:
    """Reproducible root-to-leaf path extracted from a fitted shallow CART."""

    name: str
    leaf_id: int
    conditions: tuple[PathCondition, ...]
    train_rows: int
    train_mean_predictor_score: float
    tree_prediction: float

    def evaluate(self, matrix: np.ndarray) -> np.ndarray:
        X = np.asarray(matrix, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("path evaluation matrix must be two-dimensional")
        if not self.conditions:
            return np.ones(len(X), dtype=float)
        mask = np.ones(len(X), dtype=bool)
        for condition in self.conditions:
            if condition.feature_index >= X.shape[1]:
                raise ValueError("path condition feature index is outside the matrix")
            mask &= condition.evaluate(X)
        return mask.astype(float)


@dataclass
class CartSurrogateResult:
    """Fitted shallow surrogate, selected paths, registry, and provenance."""

    model: DecisionTreeRegressor | DecisionTreeClassifier
    rules: tuple[SurrogatePathRule, ...]
    registry: pd.DataFrame
    provenance: dict[str, Any]
    feature_names: tuple[str, ...]

    def evaluate(
        self,
        matrix: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        X = np.asarray(matrix, dtype=np.float32)
        names = self.feature_names if feature_names is None else tuple(map(str, feature_names))
        if names != self.feature_names:
            raise ValueError("feature names/order do not match the fitted CART surrogate")
        if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
            raise ValueError("CART evaluation matrix is invalid or feature-misaligned")
        return pd.DataFrame({rule.name: rule.evaluate(X) for rule in self.rules})


def _extract_tree_paths(
    tree: DecisionTreeRegressor | DecisionTreeClassifier,
    feature_names: tuple[str, ...],
    train_matrix: np.ndarray,
    train_scores: np.ndarray,
) -> list[SurrogatePathRule]:
    structure = tree.tree_
    applied = tree.apply(train_matrix)
    paths: list[SurrogatePathRule] = []

    def walk(node: int, conditions: tuple[PathCondition, ...]) -> None:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        if left == right:
            mask = applied == node
            if not mask.any():
                return
            prediction = np.asarray(structure.value[node], dtype=float).reshape(-1)
            if isinstance(tree, DecisionTreeClassifier):
                total = float(prediction.sum())
                tree_prediction = float(prediction[-1] / total) if total > 0.0 else 0.0
            else:
                tree_prediction = float(prediction[0])
            paths.append(
                SurrogatePathRule(
                    name=f"cart_path_leaf_{node}",
                    leaf_id=node,
                    conditions=conditions,
                    train_rows=int(mask.sum()),
                    train_mean_predictor_score=float(train_scores[mask].mean()),
                    tree_prediction=tree_prediction,
                )
            )
            return
        feature_index = int(structure.feature[node])
        threshold = float(structure.threshold[node])
        if feature_index < 0 or feature_index >= len(feature_names) or not np.isfinite(threshold):
            raise RuntimeError("CART produced an invalid non-leaf split")
        base = {
            "feature": feature_names[feature_index],
            "feature_index": feature_index,
            "threshold": threshold,
        }
        walk(left, conditions + (PathCondition(operator="less_or_equal", **base),))
        walk(right, conditions + (PathCondition(operator="greater", **base),))

    walk(0, ())
    return paths


def fit_cart_surrogate_candidates(
    X_train: np.ndarray,
    frozen_predictor: Any,
    *,
    feature_names: Sequence[str] | None = None,
    surrogate_target: str = "score",
    score_mode: str = "calibrated",
    max_depth: int = 3,
    min_samples_leaf: int | float = 25,
    max_rules: int = 8,
    random_state: int = 42,
    device: str | None = None,
) -> CartSurrogateResult:
    """Fit a deterministic Tier C shallow CART to TRAIN predictor behaviour.

    ``surrogate_target='score'`` uses regression on predictor probabilities;
    ``'decision'`` uses classification on the frozen predictor threshold.  The
    selected high-risk leaves are returned as finite transformed-space paths.
    Ground-truth labels are intentionally absent from the signature.
    """

    _require_frozen_predictor(frozen_predictor)
    X = np.asarray(X_train, dtype=np.float32)
    names = (
        tuple(str(name) for name in frozen_predictor.feature_names)
        if feature_names is None
        else tuple(str(name) for name in feature_names)
    )
    if X.ndim != 2 or len(X) < 2 or X.shape[1] != len(names):
        raise ValueError("X_train must be a non-empty, feature-aligned two-dimensional matrix")
    if not np.isfinite(X).all() or len(set(names)) != len(names):
        raise ValueError("X_train must be finite and feature names must be unique")
    if names != tuple(str(name) for name in frozen_predictor.feature_names):
        raise ValueError("feature names/order do not match the frozen predictor")
    if int(max_depth) < 1 or int(max_rules) < 1:
        raise ValueError("max_depth and max_rules must be positive")
    normalized_target = str(surrogate_target).lower()
    if normalized_target not in {"score", "decision"}:
        raise ValueError("surrogate_target must be 'score' or 'decision'")
    if normalized_target == "decision" and str(score_mode).lower() != "calibrated":
        raise ValueError(
            "decision surrogates require calibrated scores because the frozen threshold "
            "was selected in calibrated probability space"
        )
    scores = _predict_scores(frozen_predictor, X, score_mode=score_mode, device=device)

    common = {
        "max_depth": int(max_depth),
        "min_samples_leaf": min_samples_leaf,
        "random_state": int(random_state),
    }
    if normalized_target == "score":
        model: DecisionTreeRegressor | DecisionTreeClassifier = DecisionTreeRegressor(**common)
        model.fit(X, scores)
        predictions = np.asarray(model.predict(X), dtype=float)
        fidelity_metric = "r2"
        fidelity = float(r2_score(scores, predictions))
    else:
        threshold = float(getattr(frozen_predictor, "threshold"))
        decisions = (scores >= threshold).astype(np.int8)
        if np.unique(decisions).size < 2:
            raise ValueError(
                "frozen predictor decisions contain one class; CART classifier is undefined"
            )
        model = DecisionTreeClassifier(**common)
        model.fit(X, decisions)
        predictions = np.asarray(model.predict(X), dtype=np.int8)
        fidelity_metric = "accuracy"
        fidelity = float(accuracy_score(decisions, predictions))

    all_paths = _extract_tree_paths(model, names, X, scores)
    all_paths.sort(
        key=lambda path: (-path.train_mean_predictor_score, -path.train_rows, path.leaf_id)
    )
    selected = tuple(all_paths[: int(max_rules)])
    registry_rows = []
    for path in selected:
        registry_rows.append(
            {
                "rule": path.name,
                "tier": "C",
                "candidate_kind": "shallow_cart_surrogate_path",
                "source": "train_predictor_surrogate",
                "description": (
                    f"Shallow CART leaf {path.leaf_id} approximating high frozen-predictor "
                    "scores on training rows."
                ),
                "features": list(dict.fromkeys(item.feature for item in path.conditions)),
                "feature_space": "preprocessed_predictor_features",
                "conditions": [item.as_dict() for item in path.conditions],
                "fitted_conditions": [item.as_dict() for item in path.conditions],
                "fit_partition": "train",
                "status": "generated",
                "requires_rule_evidence": True,
                "leaf_id": int(path.leaf_id),
                "train_active_count": int(path.train_rows),
                "train_mean_predictor_score": float(path.train_mean_predictor_score),
                "tree_prediction": float(path.tree_prediction),
            }
        )
    provenance = {
        "protocol": "shallow-cart-train-predictor-surrogate-v1",
        "fit_partition": "train",
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "training_matrix_sha256": _fingerprint_array(X),
        "training_predictor_scores_sha256": _fingerprint_array(scores),
        "surrogate_target": normalized_target,
        "score_mode": str(score_mode).lower(),
        "max_depth": int(max_depth),
        "min_samples_leaf": min_samples_leaf,
        "random_state": int(random_state),
        "all_leaf_count": int(len(all_paths)),
        "selected_path_count": int(len(selected)),
        "fidelity_metric": fidelity_metric,
        "train_fidelity": fidelity,
        "ground_truth_labels_used": False,
        "validation_or_test_data_used": False,
    }
    return CartSurrogateResult(model, selected, pd.DataFrame(registry_rows), provenance, names)


def score_only_baseline_registry(frozen_predictor: Any) -> pd.DataFrame:
    """Return the explicit frozen predictor-score-only comparator entry."""

    _require_frozen_predictor(frozen_predictor)
    return pd.DataFrame(
        [
            {
                "rule": "predictor_score_only",
                "tier": "baseline",
                "candidate_kind": "predictor_score_only_baseline",
                "source": "frozen_predictor",
                "description": (
                    "Frozen calibrated predictor score without logical-rule evidence; "
                    "mandatory guardrail comparator for selective explanations."
                ),
                "features": list(map(str, frozen_predictor.feature_names)),
                "feature_space": "preprocessed_predictor_features",
                "conditions": [],
                "fitted_conditions": [],
                "fit_partition": "pretrained_then_validation_calibrated",
                "status": "frozen",
                "requires_rule_evidence": False,
                "predictor_family": str(getattr(frozen_predictor, "family", "unknown")),
                "predictor_backend": str(getattr(frozen_predictor, "backend", "unknown")),
                "decision_threshold": float(getattr(frozen_predictor, "threshold")),
            }
        ]
    )


def combine_candidate_registries(*registries: pd.DataFrame) -> pd.DataFrame:
    """Combine candidate sources with deterministic ordering and unique names."""

    nonempty = [
        registry.copy()
        for registry in registries
        if registry is not None and not registry.empty
    ]
    if not nonempty:
        return pd.DataFrame()
    for registry in nonempty:
        if "rule" not in registry or "tier" not in registry:
            raise KeyError("every candidate registry requires 'rule' and 'tier' columns")
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    duplicates = combined.loc[combined["rule"].astype(str).duplicated(keep=False), "rule"].tolist()
    if duplicates:
        raise ValueError(
            "candidate rule names must be unique across registries: "
            f"{sorted(set(duplicates))}"
        )
    rank = combined["tier"].astype(str).map({"A": 0, "B": 1, "C": 2, "baseline": 3}).fillna(9)
    combined = combined.assign(_tier_rank=rank)
    return (
        combined.sort_values(["_tier_rank", "rule"])
        .drop(columns="_tier_rank")
        .reset_index(drop=True)
    )
