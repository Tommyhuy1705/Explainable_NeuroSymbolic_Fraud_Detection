"""Numerically stable fuzzy predicates used by the logical rule engine.

``softness`` can represent either a fraction of a fitted threshold or an
absolute distance in the feature's native units.  Keeping those meanings
explicit prevents domain thresholds such as ``age < 25`` from accidentally
receiving a 75-year transition width when their intended softness is 3 years.
"""

from __future__ import annotations

import numpy as np


SOFTNESS_MODES = {"relative", "absolute"}


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _resolve_scale(threshold: float, softness: float, softness_mode: str) -> float:
    """Convert a configured softness into a positive sigmoid scale.

    ``relative`` preserves the project's original behaviour: softness is a
    fraction of the (usually train-fitted) threshold, with a unit floor for
    thresholds close to zero. ``absolute`` interprets softness directly in the
    feature's native units.  The public predicate functions default to
    ``relative`` for backward compatibility; the rule engine chooses the mode
    from each operator's semantics.
    """

    if softness_mode not in SOFTNESS_MODES:
        raise ValueError(
            f"Unsupported softness mode: {softness_mode!r}; "
            f"expected one of {sorted(SOFTNESS_MODES)}"
        )
    softness = float(softness)
    if not np.isfinite(softness) or softness <= 0.0:
        raise ValueError("softness must be a finite, positive number")
    if softness_mode == "absolute":
        return max(softness, 1e-6)
    return max(abs(float(threshold)) * softness, softness, 1e-6)


def greater_than(
    values: np.ndarray,
    threshold: float,
    softness: float = 1.0,
    softness_mode: str = "relative",
) -> np.ndarray:
    """Fuzzy ``values > threshold`` truth values.

    Use ``softness_mode="absolute"`` when ``softness`` is expressed in the
    feature's units, or ``"relative"`` when it is a threshold fraction.
    """

    scale = _resolve_scale(threshold, softness, softness_mode)
    return sigmoid((np.asarray(values, dtype=float) - threshold) / scale)


def less_than(
    values: np.ndarray,
    threshold: float,
    softness: float = 1.0,
    softness_mode: str = "relative",
) -> np.ndarray:
    """Fuzzy ``values < threshold`` truth values with explicit scale units."""

    scale = _resolve_scale(threshold, softness, softness_mode)
    return sigmoid((threshold - np.asarray(values, dtype=float)) / scale)


def outside_range(
    values: np.ndarray,
    lower: float,
    upper: float,
    softness: float = 1.0,
    softness_mode: str = "relative",
) -> np.ndarray:
    """Fuzzy evidence that values lie below ``lower`` or above ``upper``."""

    return np.maximum(
        less_than(values, lower, softness, softness_mode),
        greater_than(values, upper, softness, softness_mode),
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
