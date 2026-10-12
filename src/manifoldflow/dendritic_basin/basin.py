"""Competitive recurrent basin (design doc sections 10-12).

Settles the dendritic drive into a settled state through a small number
of local iterations combining recurrent excitation with global (lateral)
inhibition, then updates its own recurrent map with local Hebbian
plasticity. Nothing here differentiates through the settling iterations
-- `settle` just iterates a fixed-point map; `learn` is a plain Hebbian
outer-product update on the already-settled state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BasinConfig:
    n_units: int
    settle_steps: int = 3
    recurrent_gain: float = 0.3  # alpha
    inhibition_gain: float = 0.1  # gamma
    hebbian_lr: float = 0.05  # eta_r
    hebbian_decay: float = 0.02  # lambda_r
    normalize_rows: bool = True  # section 12's "optional competitive
    # normalization" -- on by default: plain Hebbian growth with decay
    # alone is not self-limiting and reliably explodes numerically after
    # a few thousand updates (failure mode 38.3, "basin collapse" /
    # runaway attraction). Row-normalizing keeps each unit's outgoing
    # recurrent influence bounded regardless of how long training runs.
    seed: int = 0


class RecurrentBasin:
    """K-unit competitive recurrent attractor over the dendritic drive."""

    def __init__(self, config: BasinConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        n = config.n_units
        self.w_rec = rng.normal(0.0, 0.01, size=(n, n))
        np.fill_diagonal(self.w_rec, 0.0)

    def settle(self, drive: np.ndarray) -> np.ndarray:
        """h_{t+1} = relu(drive + alpha*W_rec@h_t - gamma*(sum(h_t) - h_t))
        (section 10), run for `settle_steps` iterations from h_0 = drive.
        """
        cfg = self.config
        h = drive.copy()
        for _ in range(cfg.settle_steps):
            recurrent = self.w_rec @ h
            inhibition = h.sum() - h
            h = (
                drive
                + cfg.recurrent_gain * recurrent
                - cfg.inhibition_gain * inhibition
            )
            np.clip(h, 0.0, None, out=h)
        return h

    def learn(self, h: np.ndarray) -> None:
        """W_rec <- (1-lambda_r)*W_rec + eta_r*h h^T, W_ii = 0 (section 12),
        then row-normalized (`W_ij <- W_ij / (eps + sum_j |W_ij|)`, section
        12's suggested formula, using the absolute row sum so a row with
        net-negative entries doesn't flip sign) if `normalize_rows`.
        """
        cfg = self.config
        decayed = (1 - cfg.hebbian_decay) * self.w_rec
        self.w_rec = decayed + cfg.hebbian_lr * np.outer(h, h)
        np.fill_diagonal(self.w_rec, 0.0)
        if cfg.normalize_rows:
            row_scale = np.abs(self.w_rec).sum(axis=1, keepdims=True)
            self.w_rec = self.w_rec / (1e-6 + row_scale)
