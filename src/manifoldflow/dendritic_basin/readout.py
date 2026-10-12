"""Local linear readout with a delta-rule update (design doc section 13).

The output unit sees its own inputs, its own activation, and the provided
target directly -- so its update can be a plain local delta rule; no
derivative is propagated to whatever produced `h`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ReadoutConfig:
    n_inputs: int
    learning_rate: float = 0.5  # eta_y
    seed: int = 0


class LocalReadout:
    def __init__(self, config: ReadoutConfig):
        rng = np.random.default_rng(config.seed)
        self.config = config
        self.weights = rng.normal(0.0, 0.1, size=config.n_inputs)
        self.bias = 0.0

    def predict(self, h: np.ndarray) -> float:
        return float(sigmoid(self.weights @ h + self.bias))

    def learn(self, h: np.ndarray, modulator: float) -> None:
        """Delta W_y = eta_y * M * h (section 13)."""
        cfg = self.config
        self.weights = self.weights + cfg.learning_rate * modulator * h
        self.bias = self.bias + cfg.learning_rate * modulator
