"""N-bit parity (design doc sections 4, 10 of
`Dendritic_Basin_Learning_Design_v0.2_Interaction_Dendrites.md`; consolidated
in `Manifold_Dendritic_Stream_Learner_Design_v0.2.md` sections 4, 10):
the task recursive relation composition is meant to solve with `N-1`
relation units instead of `2^N` memorized patterns.
"""

from __future__ import annotations

import numpy as np


def sample_bits(n_bits: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """One random N-bit vector and its parity (XOR of all bits)."""
    bits = rng.integers(0, 2, size=n_bits).astype(np.float64)
    target = int(bits.sum()) % 2
    return bits, target
