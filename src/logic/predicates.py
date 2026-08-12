"""Numerically stable fuzzy predicates used by the logical rule engine."""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def greater_than(values: np.ndarray, threshold: float, softness: float = 1.0) -> np.ndarray:
    scale = max(abs(threshold) * softness, softness, 1e-6)
    return sigmoid((np.asarray(values, dtype=float) - threshold) / scale)


def less_than(values: np.ndarray, threshold: float, softness: float = 1.0) -> np.ndarray:
    scale = max(abs(threshold) * softness, softness, 1e-6)
    return sigmoid((threshold - np.asarray(values, dtype=float)) / scale)


def outside_range(
    values: np.ndarray,
    lower: float,
    upper: float,
    softness: float = 1.0,
) -> np.ndarray:
    return np.maximum(
        less_than(values, lower, softness),
        greater_than(values, upper, softness),
    )


def fuzzy_and(truth_values: list[np.ndarray], mode: str = "minimum") -> np.ndarray:
    if not truth_values:
        raise ValueError("fuzzy_and requires at least one truth-value array")
    stacked = np.vstack(truth_values)
    if mode == "product":
        return np.prod(stacked, axis=0)
    if mode == "minimum":
        return np.min(stacked, axis=0)
    raise ValueError(f"Unsupported fuzzy conjunction: {mode}")


def fuzzy_or(truth_values: list[np.ndarray], mode: str = "maximum") -> np.ndarray:
    if not truth_values:
        raise ValueError("fuzzy_or requires at least one truth-value array")
    stacked = np.vstack(truth_values)
    if mode == "probabilistic_sum":
        return 1.0 - np.prod(1.0 - stacked, axis=0)
    if mode == "maximum":
        return np.max(stacked, axis=0)
    raise ValueError(f"Unsupported fuzzy disjunction: {mode}")
