"""Small PyTorch implementation of the tensor/fuzzy semantics used by LTN-style rules.

The project keeps this layer dependency-light instead of requiring a specific
LTN library version. All operators are differentiable and can be embedded in a
training objective or used to inspect tensor-valued predicate truth values.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class SoftThresholdPredicate(nn.Module):
    """Differentiable greater-than or less-than predicate."""

    def __init__(
        self,
        initial_threshold: float,
        temperature: float = 1.0,
        direction: str = "greater",
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if direction not in {"greater", "less"}:
            raise ValueError("direction must be 'greater' or 'less'")
        threshold = torch.tensor(float(initial_threshold), dtype=torch.float32)
        if learnable:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)
        self.register_buffer("temperature", torch.tensor(max(float(temperature), 1e-6)))
        self.direction = direction

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        signed_distance = values - self.threshold
        if self.direction == "less":
            signed_distance = -signed_distance
        return torch.sigmoid(signed_distance / self.temperature)


class TensorLogic:
    """Differentiable fuzzy connectives and p-mean quantification."""

    @staticmethod
    def negation(truth: torch.Tensor) -> torch.Tensor:
        return 1.0 - truth

    @staticmethod
    def conjunction(*truth_values: torch.Tensor) -> torch.Tensor:
        if not truth_values:
            raise ValueError("conjunction requires at least one tensor")
        return torch.stack(truth_values, dim=0).prod(dim=0)

    @staticmethod
    def disjunction(*truth_values: torch.Tensor) -> torch.Tensor:
        if not truth_values:
            raise ValueError("disjunction requires at least one tensor")
        stacked = torch.stack(truth_values, dim=0)
        return 1.0 - (1.0 - stacked).prod(dim=0)

    @staticmethod
    def implication(antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Reichenbach implication: 1 - a + a*b."""
        return 1.0 - antecedent + antecedent * consequent

    @staticmethod
    def equivalence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return TensorLogic.conjunction(
            TensorLogic.implication(left, right),
            TensorLogic.implication(right, left),
        )

    @staticmethod
    def forall(truth: torch.Tensor, p: float = 2.0, dim: int | None = None) -> torch.Tensor:
        """Generalized p-mean error aggregation used for universal quantification."""
        error = (1.0 - truth).clamp(0.0, 1.0).pow(p)
        return 1.0 - error.mean(dim=dim).pow(1.0 / p)


class TensorFraudKnowledgeBase(nn.Module):
    """Aggregate named tensor predicates into differentiable fraud rules."""

    def __init__(
        self,
        predicates: dict[str, nn.Module],
        rules: dict[str, Iterable[str]],
    ) -> None:
        super().__init__()
        self.predicates = nn.ModuleDict(predicates)
        self.rules = {name: tuple(predicate_names) for name, predicate_names in rules.items()}
        unknown = {
            predicate_name
            for names in self.rules.values()
            for predicate_name in names
            if predicate_name not in self.predicates
        }
        if unknown:
            raise KeyError(f"Rules reference unknown predicates: {sorted(unknown)}")

    def forward(self, feature_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predicate_truth = {
            name: predicate(feature_tensors[name]) for name, predicate in self.predicates.items()
        }
        return {
            rule_name: TensorLogic.conjunction(
                *(predicate_truth[predicate_name] for predicate_name in predicate_names)
            )
            for rule_name, predicate_names in self.rules.items()
        }

    @staticmethod
    def suspiciousness(rule_truth: dict[str, torch.Tensor]) -> torch.Tensor:
        return TensorLogic.disjunction(*rule_truth.values())

    @staticmethod
    def supervised_satisfaction(
        rule_truth: dict[str, torch.Tensor],
        labels: torch.Tensor,
        p: float = 2.0,
    ) -> torch.Tensor:
        evidence = TensorFraudKnowledgeBase.suspiciousness(rule_truth)
        return TensorLogic.forall(TensorLogic.equivalence(evidence, labels.float()), p=p)
