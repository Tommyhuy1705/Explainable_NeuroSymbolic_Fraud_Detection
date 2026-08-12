"""Multilayer perceptron baseline for tabular fraud data."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FraudMLP(nn.Module):
    """A compact MLP returning one fraud logit per row."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
