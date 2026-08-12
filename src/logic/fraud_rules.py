"""Train-fitted, human-readable fuzzy rules for fraud evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .predicates import fuzzy_and, greater_than, less_than, outside_range


@dataclass
class FittedCondition:
    feature: str
    operator: str
    configured_value: Any
    threshold: Any
    softness: float
    category_risk: dict[str, float] = field(default_factory=dict)

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature not in frame:
            raise KeyError(f"Rule feature is missing: {self.feature}")
        series = frame[self.feature]
        if self.operator == "category_risk":
            risks = series.astype("string").fillna("<MISSING>").map(self.category_risk).fillna(0.0)
            return greater_than(risks.to_numpy(float), float(self.threshold), self.softness)
        if self.operator == "equals":
            return (series.fillna("<MISSING>").astype(str) == str(self.threshold)).to_numpy(float)

        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        fill_value = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        values = np.nan_to_num(values, nan=fill_value)
        if self.operator in {"greater", "greater_quantile"}:
            return greater_than(values, float(self.threshold), self.softness)
        if self.operator in {"less", "less_quantile"}:
            return less_than(values, float(self.threshold), self.softness)
        if self.operator == "outside_range":
            lower, upper = self.threshold
            return outside_range(values, float(lower), float(upper), self.softness)
        raise ValueError(f"Unsupported rule operator: {self.operator}")


@dataclass
class FittedRule:
    name: str
    description: str
    conditions: list[FittedCondition]
    conjunction: str = "minimum"

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        return fuzzy_and([condition.evaluate(frame) for condition in self.conditions], self.conjunction)


def _smoothed_category_risk(series: pd.Series, target: pd.Series) -> dict[str, float]:
    categories = series.astype("string").fillna("<MISSING>")
    summary = pd.DataFrame({"category": categories, "target": target.to_numpy()}).groupby("category")[
        "target"
    ].agg(["sum", "count"])
    global_rate = float(target.mean())
    smoothing = 20.0
    risk = (summary["sum"] + smoothing * global_rate) / (summary["count"] + smoothing)
    if risk.max() > risk.min():
        risk = (risk - risk.min()) / (risk.max() - risk.min())
    else:
        risk = pd.Series(0.0, index=risk.index)
    return {str(key): float(value) for key, value in risk.items()}


class FraudRuleEngine:
    """Fit rule thresholds on training data and evaluate them on later splits."""

    def __init__(self, rule_config: list[dict[str, Any]], conjunction: str = "minimum") -> None:
        self.rule_config = rule_config
        self.conjunction = conjunction
        self.rules: list[FittedRule] = []
        self.skipped_rules: dict[str, str] = {}

    def _fit_condition(
        self,
        condition: dict[str, Any],
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> FittedCondition:
        feature = condition["feature"]
        operator = condition["operator"]
        configured_value = condition["value"]
        softness = float(condition.get("softness", 0.1))
        category_risk: dict[str, float] = {}

        if operator in {"greater_quantile", "less_quantile"}:
            numeric = pd.to_numeric(frame[feature], errors="coerce")
            threshold: Any = float(numeric.quantile(float(configured_value)))
        elif operator == "category_risk":
            category_risk = _smoothed_category_risk(frame[feature], target)
            threshold = float(configured_value)
        else:
            threshold = configured_value
        return FittedCondition(
            feature=feature,
            operator=operator,
            configured_value=configured_value,
            threshold=threshold,
            softness=softness,
            category_risk=category_risk,
        )

    def fit(self, frame: pd.DataFrame, target_column: str) -> "FraudRuleEngine":
        if target_column not in frame:
            raise KeyError(f"Target column is missing: {target_column}")
        target = frame[target_column].astype(int)
        self.rules = []
        self.skipped_rules = {}
        for definition in self.rule_config:
            name = definition["name"]
            missing = [c["feature"] for c in definition["conditions"] if c["feature"] not in frame]
            if missing:
                self.skipped_rules[name] = f"missing features: {', '.join(missing)}"
                continue
            fitted_conditions = [
                self._fit_condition(condition, frame, target) for condition in definition["conditions"]
            ]
            self.rules.append(
                FittedRule(
                    name=name,
                    description=definition.get("description", name.replace("_", " ")),
                    conditions=fitted_conditions,
                    conjunction=definition.get("conjunction", self.conjunction),
                )
            )
        if not self.rules:
            raise ValueError("No logical rules are available for the provided training columns")
        return self

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.rules:
            raise RuntimeError("FraudRuleEngine must be fitted before evaluation")
        return pd.DataFrame(
            {rule.name: rule.evaluate(frame) for rule in self.rules},
            index=frame.index,
        )

    def descriptions(self) -> dict[str, str]:
        return {rule.name: rule.description for rule in self.rules}

    def fitted_thresholds(self) -> pd.DataFrame:
        rows = []
        for rule in self.rules:
            for condition in rule.conditions:
                rows.append(
                    {
                        "rule": rule.name,
                        "feature": condition.feature,
                        "operator": condition.operator,
                        "configured_value": condition.configured_value,
                        "fitted_threshold": condition.threshold,
                    }
                )
        return pd.DataFrame(rows)

