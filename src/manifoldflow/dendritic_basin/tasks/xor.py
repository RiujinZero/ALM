"""The static XOR task (design doc section 24): the canonical 4-row table."""

from __future__ import annotations

import numpy as np

XOR_TABLE: list[tuple[np.ndarray, float]] = [
    (np.array([0.0, 0.0]), 0.0),
    (np.array([0.0, 1.0]), 1.0),
    (np.array([1.0, 0.0]), 1.0),
    (np.array([1.0, 1.0]), 0.0),
]

N_INPUTS = 2


def sample(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Draw one (x, y) pair uniformly at random from the XOR table."""
    x, y = XOR_TABLE[int(rng.integers(0, len(XOR_TABLE)))]
    return x, y


def evaluate_all_rows(predict_fn) -> tuple[float, list[bool]]:
    """Exact accuracy over the full (and only) 4-row input space.

    `predict_fn(x) -> float` should not update any model state (see
    `DendriticBasinSystem.predict`). Returns (accuracy, per_row_correct).
    """
    correct = []
    for x, y in XOR_TABLE:
        prediction = predict_fn(x)
        correct.append((prediction >= 0.5) == (y >= 0.5))
    return sum(correct) / len(correct), correct
