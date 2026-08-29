"""Validation-frozen contrastive meta-evidence for stress-test alerts.

This module implements the proposal's contrastive meta-scorer as a standalone
component.  Fitting is intentionally restricted to the reserved ``rule_audit``
role and to rows that the already-frozen predictor marks as alerts. Within that
fixed alert region, a balanced logistic regression distinguishes true positive
from false positive alerts using audited rule-truth values and a standardized
copy of the already-frozen calibrated predictor risk.

The fitted object never accepts labels at transform time. It requires frozen
probabilities because risk is an explicit proposal covariate, but it cannot
refit its audit-alert mean, scale, coefficients, or schema. Rows without an
active rule receive exactly zero evidence even though the logistic model has an
intercept and risk term. When the audit partition cannot support a valid fit,
the API returns an explicit zero scorer with auditable provenance instead of
silently fitting on another partition or relaxing the protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


_PROTOCOL = "rule-audit-alert-contrastive-logistic-v1"
_RISK_FEATURE = "frozen_calibrated_risk_z"
_RISK_STD_EPSILON = float(np.sqrt(np.finfo(np.float64).eps))


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _rule_matrix(
    rule_truth: pd.DataFrame,
    *,
    expected_rules: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not isinstance(rule_truth, pd.DataFrame):
        raise TypeError("rule_truth must be a pandas DataFrame")
    if rule_truth.columns.duplicated().any():
        duplicates = sorted(
            set(rule_truth.columns[rule_truth.columns.duplicated(keep=False)].astype(str))
        )
        raise ValueError(f"rule_truth columns must be unique: {duplicates}")

    observed = tuple(str(column) for column in rule_truth.columns)
    if expected_rules is None:
        rules = observed
    else:
        rules = tuple(str(rule) for rule in expected_rules)
        missing = [rule for rule in rules if rule not in rule_truth.columns]
        if missing:
            raise KeyError(f"rule_truth is missing fitted rules: {missing}")

    if not rules:
        return np.empty((len(rule_truth), 0), dtype=np.float64), rules
    matrix = rule_truth.loc[:, list(rules)].to_numpy(dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape != (len(rule_truth), len(rules)):
        raise ValueError("rule_truth could not be converted to a row-aligned matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("rule_truth contains NaN or infinite values")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError("audited rule-truth values must lie in [0, 1]")
    return matrix, rules


def _binary_labels(values: np.ndarray | Sequence[int], *, expected_rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != int(expected_rows):
        raise ValueError("labels must be one-dimensional and aligned with rule_truth")
    if labels.size and not np.isfinite(labels.astype(float)).all():
        raise ValueError("labels contain non-finite values")
    observed = set(np.unique(labels).tolist())
    if not observed.issubset({0, 1, False, True}):
        raise ValueError("labels must be binary 0/1 values")
    return labels.astype(np.int8, copy=False)


def _probabilities(values: np.ndarray | Sequence[float], *, expected_rows: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) != int(expected_rows):
        raise ValueError("probabilities must be one-dimensional and aligned with rule_truth")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return probabilities


def _coefficient_table(
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    coefficients: np.ndarray | None,
    *,
    intercept: float,
    available: bool,
    fallback_reason: str | None,
) -> pd.DataFrame:
    values = (
        np.zeros(len(feature_names), dtype=np.float64)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64).reshape(-1)
    )
    if len(values) != len(feature_names) or len(feature_types) != len(feature_names):
        raise RuntimeError(
            "contrastive coefficients do not align with the frozen feature schema"
        )
    absolute = np.abs(values)
    order = np.argsort(-absolute, kind="stable") if len(values) else np.asarray([], dtype=int)
    ranks = np.empty(len(values), dtype=int)
    if len(values):
        ranks[order] = np.arange(1, len(values) + 1, dtype=int)
    return pd.DataFrame(
        {
            "feature": list(map(str, feature_names)),
            "feature_type": list(map(str, feature_types)),
            "rule": [
                str(name) if kind == "audited_rule" else None
                for name, kind in zip(feature_names, feature_types, strict=True)
            ],
            "coefficient": values,
            "absolute_coefficient": absolute,
            "odds_ratio": np.exp(np.clip(values, -700.0, 700.0)),
            "absolute_coefficient_rank": ranks,
            "intercept": float(intercept),
            "available": bool(available),
            "fallback_reason": fallback_reason,
            "fit_partition": "rule_audit",
        }
    )


@dataclass
class ContrastiveMetaEvidence:
    """Frozen TP-vs-FP scorer fitted on audited rule truth and frozen risk."""

    rule_names: tuple[str, ...]
    feature_schema: tuple[str, ...]
    decision_threshold: float
    activation_threshold: float
    frozen_risk_mean: float
    frozen_risk_std: float
    frozen_risk_coefficient: float
    seed: int
    model: LogisticRegression | None
    provenance: dict[str, Any]
    _coefficients: pd.DataFrame

    @property
    def available(self) -> bool:
        """Whether the validation role contained enough evidence to fit."""

        return self.model is not None and bool(self.provenance.get("available", False))

    @property
    def fallback_reason(self) -> str | None:
        value = self.provenance.get("fallback_reason")
        return None if value is None else str(value)

    def coefficient_table(self) -> pd.DataFrame:
        """Return a CSV-serializable copy of the frozen coefficient audit."""

        return self._coefficients.copy(deep=True)

    def transform(
        self,
        rule_truth: pd.DataFrame,
        frozen_probabilities: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Score later rule truth and frozen risk without labels or refitting.

        A fail-closed object always returns zeros.  A fitted object also returns
        zero for every row on which no rule reaches the validation-frozen
        ``activation_threshold``.
        """

        matrix, _ = _rule_matrix(rule_truth, expected_rules=self.rule_names)
        probabilities = _probabilities(frozen_probabilities, expected_rows=len(matrix))
        if len(matrix) == 0:
            return np.empty(0, dtype=np.float64)
        if not self.available:
            return np.zeros(len(matrix), dtype=np.float64)

        if not np.isfinite(self.frozen_risk_std) or self.frozen_risk_std <= _RISK_STD_EPSILON:
            raise RuntimeError("available contrastive model has an invalid frozen risk scale")
        risk_z = (probabilities - float(self.frozen_risk_mean)) / float(self.frozen_risk_std)
        if tuple(self.feature_schema) != (*self.rule_names, _RISK_FEATURE):
            raise RuntimeError("contrastive frozen feature schema is inconsistent")
        design = np.column_stack((matrix, risk_z))
        evidence = np.asarray(self.model.predict_proba(design)[:, 1], dtype=np.float64)
        if evidence.shape != (len(matrix),) or not np.isfinite(evidence).all():
            raise RuntimeError("contrastive logistic model returned invalid evidence")
        supported = (matrix >= float(self.activation_threshold)).any(axis=1)
        evidence = np.clip(evidence, 0.0, 1.0)
        evidence[~supported] = 0.0
        return evidence


def _build_result(
    *,
    rule_names: tuple[str, ...],
    decision_threshold: float,
    activation_threshold: float,
    frozen_risk_mean: float,
    frozen_risk_std: float,
    seed: int,
    model: LogisticRegression | None,
    provenance: Mapping[str, Any],
    fallback_reason: str | None,
) -> ContrastiveMetaEvidence:
    available = model is not None and fallback_reason is None
    feature_schema = (*rule_names, _RISK_FEATURE)
    feature_types = (*("audited_rule" for _ in rule_names), "frozen_risk")
    coefficients = None if model is None else np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = 0.0 if model is None else float(np.asarray(model.intercept_).reshape(-1)[0])
    frozen_risk_coefficient = 0.0 if coefficients is None else float(coefficients[-1])
    completed = {
        **dict(provenance),
        "available": bool(available),
        "fail_closed": not bool(available),
        "fallback_reason": fallback_reason,
        "feature_schema": list(feature_schema),
        "frozen_risk_feature": _RISK_FEATURE,
        "frozen_risk_mean": float(frozen_risk_mean),
        "frozen_risk_std": float(frozen_risk_std),
        "frozen_risk_coefficient": float(frozen_risk_coefficient),
    }
    table = _coefficient_table(
        feature_schema,
        feature_types,
        coefficients,
        intercept=intercept,
        available=available,
        fallback_reason=fallback_reason,
    )
    return ContrastiveMetaEvidence(
        rule_names=rule_names,
        feature_schema=feature_schema,
        decision_threshold=float(decision_threshold),
        activation_threshold=float(activation_threshold),
        frozen_risk_mean=float(frozen_risk_mean),
        frozen_risk_std=float(frozen_risk_std),
        frozen_risk_coefficient=float(frozen_risk_coefficient),
        seed=int(seed),
        model=model,
        provenance=completed,
        _coefficients=table,
    )


def fit_contrastive_meta_evidence(
    rule_audit_truth: pd.DataFrame,
    rule_audit_labels: np.ndarray | Sequence[int],
    rule_audit_probabilities: np.ndarray | Sequence[float],
    decision_threshold: float,
    *,
    activation_threshold: float = 0.60,
    seed: int = 42,
    min_alert_rows: int = 20,
    min_class_rows: int = 2,
    regularization_c: float = 1.0,
    max_iter: int = 1000,
) -> ContrastiveMetaEvidence:
    """Fit balanced logistic TP-vs-FP evidence on ``rule_audit`` alerts.

    Insufficient alerts, alert classes, active/variable rule evidence, or
    variable frozen risk produce a frozen zero scorer. Malformed or misaligned
    inputs raise because silently accepting a schema error would not be a valid
    fail-closed scientific result.
    """

    matrix, rule_names = _rule_matrix(rule_audit_truth)
    labels = _binary_labels(rule_audit_labels, expected_rows=len(matrix))
    probabilities = _probabilities(rule_audit_probabilities, expected_rows=len(matrix))
    if not np.isfinite(float(decision_threshold)) or not 0.0 <= float(decision_threshold) <= 1.0:
        raise ValueError("decision_threshold must be finite and lie in [0, 1]")
    if not np.isfinite(float(activation_threshold)) or not 0.0 < float(activation_threshold) <= 1.0:
        raise ValueError("activation_threshold must be finite and lie in (0, 1]")
    if int(min_alert_rows) < 1 or int(min_class_rows) < 1:
        raise ValueError("min_alert_rows and min_class_rows must be positive")
    if not np.isfinite(float(regularization_c)) or float(regularization_c) <= 0.0:
        raise ValueError("regularization_c must be finite and positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive")

    alerts = probabilities >= float(decision_threshold)
    alert_matrix = matrix[alerts]
    alert_labels = labels[alerts]
    alert_risk = probabilities[alerts]
    frozen_risk_mean = float(np.mean(alert_risk)) if len(alert_risk) else 0.0
    frozen_risk_std = float(np.std(alert_risk, ddof=0)) if len(alert_risk) else 0.0
    class_counts = np.bincount(alert_labels, minlength=2) if len(alert_labels) else np.zeros(2, dtype=int)
    supported_alerts = (
        (alert_matrix >= float(activation_threshold)).any(axis=1)
        if len(rule_names)
        else np.zeros(len(alert_matrix), dtype=bool)
    )
    base_provenance: dict[str, Any] = {
        "protocol": _PROTOCOL,
        "fit_partition": "rule_audit",
        "fit_scope": "frozen_predictor_alerts_only",
        "target": "true_positive_vs_false_positive_predictor_alert",
        "input_rows": int(len(matrix)),
        "alert_rows": int(alerts.sum()),
        "tp_alert_rows": int(class_counts[1]),
        "fp_alert_rows": int(class_counts[0]),
        "supported_alert_rows": int(supported_alerts.sum()),
        "rule_count": int(len(rule_names)),
        "rule_names": list(rule_names),
        "feature_schema": [*rule_names, _RISK_FEATURE],
        "decision_threshold": float(decision_threshold),
        "activation_threshold": float(activation_threshold),
        "seed": int(seed),
        "class_weight": "balanced",
        "solver": "liblinear",
        "penalty": "l2",
        "regularization_c": float(regularization_c),
        "max_iter": int(max_iter),
        "risk_standardization_partition": "rule_audit_predictor_alerts_only",
        "frozen_risk_mean": float(frozen_risk_mean),
        "frozen_risk_std": float(frozen_risk_std),
        "minimum_alert_rows": int(min_alert_rows),
        "minimum_rows_per_class": int(min_class_rows),
        "ground_truth_labels_used": True,
        "ground_truth_label_role": "rule_audit_only",
        "policy_select_data_used_for_fit": False,
        "test_data_used_for_fit": False,
        # Only the alert subset is fingerprinted: labels and rule truth outside
        # the frozen alert region are deliberately irrelevant to this fit.
        "alert_fit_fingerprint_sha256": _fingerprint(
            alert_matrix,
            alert_labels,
            probabilities[alerts],
        ),
    }

    reason: str | None = None
    if not rule_names:
        reason = "no_audited_rules"
    elif int(alerts.sum()) < int(min_alert_rows):
        reason = "insufficient_predictor_alerts"
    elif int(class_counts.min()) < int(min_class_rows):
        reason = "insufficient_tp_or_fp_alerts"
    elif not supported_alerts.any():
        reason = "no_active_rule_support_among_alerts"
    elif not np.any(np.ptp(alert_matrix, axis=0) > 0.0):
        reason = "non_varying_rule_evidence_among_alerts"
    elif frozen_risk_std <= _RISK_STD_EPSILON:
        reason = "non_varying_frozen_risk_among_alerts"
    if reason is not None:
        return _build_result(
            rule_names=rule_names,
            decision_threshold=float(decision_threshold),
            activation_threshold=float(activation_threshold),
            frozen_risk_mean=frozen_risk_mean,
            frozen_risk_std=frozen_risk_std,
            seed=int(seed),
            model=None,
            provenance=base_provenance,
            fallback_reason=reason,
        )

    risk_z = (alert_risk - frozen_risk_mean) / frozen_risk_std
    design = np.column_stack((alert_matrix, risk_z))
    model = LogisticRegression(
        class_weight="balanced",
        C=float(regularization_c),
        max_iter=int(max_iter),
        random_state=int(seed),
        solver="liblinear",
    )
    try:
        model.fit(design, alert_labels)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _build_result(
            rule_names=rule_names,
            decision_threshold=float(decision_threshold),
            activation_threshold=float(activation_threshold),
            frozen_risk_mean=frozen_risk_mean,
            frozen_risk_std=frozen_risk_std,
            seed=int(seed),
            model=None,
            provenance={
                **base_provenance,
                "fit_error_type": type(error).__name__,
                "fit_error_message": str(error),
            },
            fallback_reason="logistic_fit_failed",
        )
    return _build_result(
        rule_names=rule_names,
        decision_threshold=float(decision_threshold),
        activation_threshold=float(activation_threshold),
        frozen_risk_mean=frozen_risk_mean,
        frozen_risk_std=frozen_risk_std,
        seed=int(seed),
        model=model,
        provenance=base_provenance,
        fallback_reason=None,
    )


__all__ = ["ContrastiveMetaEvidence", "fit_contrastive_meta_evidence"]
