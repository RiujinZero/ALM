"""Baseline and representation-health metrics for MF-ALM-TS evaluation.

Design doc section 18.2: "Sanity loss: >=10% cross-entropy reduction from
unigram" and "Representation rank: effective rank >=0.40d; no unit active
>35%". The unigram baseline and rank/density readouts used to check those
gates live here; the rank/density readouts themselves come off
`PlasticColumn` (see model/plastic_column.py).
"""

from __future__ import annotations

import math

import numpy as np


def unigram_cross_entropy(token_stream: np.ndarray, vocab_size: int) -> float:
    """Cross-entropy (nats/token) of the empirical unigram distribution of
    `token_stream` against itself -- the "beat unigram loss" reference from
    section 18.2/23. This is the entropy of the empirical distribution,
    which is what a perfectly-fit unigram model would achieve on this
    exact stream.
    """
    counts = np.bincount(token_stream.astype(np.int64), minlength=vocab_size)
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log(probs)).sum())


def cross_entropy_reduction(model_loss: float, unigram_loss: float) -> float:
    """Fraction of unigram cross-entropy the model has recovered.

    Matches the D0 sanity gate: ">=10% cross-entropy reduction from unigram".
    """
    if unigram_loss <= 0:
        return 0.0
    return (unigram_loss - model_loss) / unigram_loss


def bits_per_token(nats: float) -> float:
    return nats / math.log(2)
