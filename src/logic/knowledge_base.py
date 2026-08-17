"""LTN-style differentiable knowledge aggregation for fraud evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fraud_rules import FraudRuleEngine
from .predicates import fuzzy_or


class FraudKnowledgeBase:
    """Aggregate fitted fuzzy rules into a transparent suspiciousness score."""

    def __init__(self, rule_engine: FraudRuleEngine, disjunction: str = "maximum") -> None:
        self.rule_engine = rule_engine
        self.disjunction = disjunction

    def rule_truth_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.rule_engine.evaluate(frame)

    def suspiciousness(self, frame: pd.DataFrame) -> np.ndarray:
        truth = self.rule_truth_values(frame)
        return fuzzy_or([truth[column].to_numpy(float) for column in truth], self.disjunction)

    def satisfaction(self, frame: pd.DataFrame, target_column: str) -> float:
        """Mean agreement between aggregated rule evidence and observed labels."""
        score = self.suspiciousness(frame)
        target = frame[target_column].to_numpy(dtype=float)
        agreement = target * score + (1.0 - target) * (1.0 - score)
        return float(np.mean(agreement))

    def satisfaction_breakdown(self, frame: pd.DataFrame, target_column: str) -> dict[str, float]:
        """Report class-balanced satisfaction so rare fraud is not hidden by negatives."""
        score = self.suspiciousness(frame)
        target = frame[target_column].to_numpy(dtype=int)
        positive = target == 1
        negative = target == 0
        if not positive.any() or not negative.any():
            raise ValueError("Knowledge-base satisfaction requires both target classes")
        positive_satisfaction = float(score[positive].mean())
        negative_satisfaction = float((1.0 - score[negative]).mean())
        return {
            "overall_satisfaction": self.satisfaction(frame, target_column),
            "positive_satisfaction": positive_satisfaction,
            "negative_satisfaction": negative_satisfaction,
            "balanced_satisfaction": 0.5 * (positive_satisfaction + negative_satisfaction),
        }
