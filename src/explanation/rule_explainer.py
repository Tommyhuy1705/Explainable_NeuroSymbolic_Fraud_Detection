"""Convert fuzzy rule truth values into concise, auditable explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logic import FraudRuleEngine


class RuleExplainer:
    def __init__(
        self,
        rule_engine: FraudRuleEngine,
        activation_threshold: float = 0.6,
        top_k_rules: int = 3,
    ) -> None:
        self.rule_engine = rule_engine
        self.activation_threshold = float(activation_threshold)
        self.top_k_rules = int(top_k_rules)

    def explain(
        self,
        frame: pd.DataFrame,
        predicted_probabilities: np.ndarray | None = None,
        decision_threshold: float = 0.5,
    ) -> pd.DataFrame:
        truth = self.rule_engine.evaluate(frame)
        descriptions = self.rule_engine.descriptions()
        probabilities = (
            np.asarray(predicted_probabilities, dtype=float)
            if predicted_probabilities is not None
            else np.full(len(frame), np.nan)
        )
        if len(probabilities) != len(frame):
            raise ValueError("predicted_probabilities must align with frame rows")

        records: list[dict[str, Any]] = []
        for position, (index, row) in enumerate(truth.iterrows()):
            active = row[row >= self.activation_threshold].sort_values(ascending=False).head(self.top_k_rules)
            rule_names = active.index.tolist()
            strengths = [float(value) for value in active.to_numpy()]
            evidence = [descriptions[name] for name in rule_names]
            records.append(
                {
                    "row_index": index,
                    "predicted_probability": float(probabilities[position]),
                    "predicted_alert": bool(probabilities[position] >= decision_threshold)
                    if np.isfinite(probabilities[position])
                    else None,
                    "explained": bool(rule_names),
                    "rule_count": len(rule_names),
                    "rule_names": rule_names,
                    "rule_strengths": strengths,
                    "max_rule_strength": max(strengths, default=0.0),
                    "explanation": " ".join(evidence) if evidence else "No configured rule reached the activation threshold.",
                }
            )
        return pd.DataFrame(records).set_index("row_index")
