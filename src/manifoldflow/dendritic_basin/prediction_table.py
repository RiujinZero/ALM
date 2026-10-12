"""Local next-token prediction readout (design doc section 27 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`).

Each active assembly/relation maintains local predictive statistics --
how often each output followed it -- entirely from local counts, no
gradient:

  C_k[j] <- lambda*C_k[j] + eta*a_k*1[x_{t+1}=j]

Active assemblies vote for the next output; votes are weighted by
activation. This is the mechanism that turns a discovered relation into
an actual prediction, used to score bits/token in the language smokes
(sections 28-32).
"""

from __future__ import annotations

import numpy as np


class PredictionTable:
    """A single local next-symbol count table, decayed and gated by an
    activation strength each update (design doc section 27).
    """

    def __init__(self, vocab_size: int, decay: float = 0.98, lr: float = 1.0):
        self.vocab_size = vocab_size
        self.decay = decay
        self.lr = lr
        self.counts = np.zeros(vocab_size)

    def update(self, activation: float, next_token_id: int) -> None:
        self.counts *= self.decay
        self.counts[next_token_id] += self.lr * activation

    def distribution(self, smoothing: float = 1e-3) -> np.ndarray:
        p = self.counts + smoothing
        return p / p.sum()


def mix_distributions(
    base: np.ndarray, weighted_predictors: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    """Weighted mixture of a base distribution with zero or more
    (weight, distribution) pairs from active relations, renormalized.
    Weights need not sum to 1; `base` always contributes weight 1.
    """
    total_weight = 1.0
    mixture = base.copy()
    for weight, distribution in weighted_predictors:
        if weight <= 0:
            continue
        mixture = mixture + weight * distribution
        total_weight += weight
    return mixture / total_weight


def bits_per_token(probs: list[float]) -> float:
    """Mean -log2(p) over a sequence of assigned probabilities for the
    actually-observed next token -- the evaluation metric used throughout
    sections 28-32 and 41.
    """
    arr = np.asarray(probs, dtype=np.float64)
    return float(-np.mean(np.log2(np.clip(arr, 1e-12, 1.0))))
