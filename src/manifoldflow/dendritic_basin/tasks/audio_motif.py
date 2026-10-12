"""Phase 10 (design doc section 54: "Audio -- Continuous frequency
vectors."): a synthetic audio-flavored analogue of Phase 2's ambiguous-
object disambiguation task, on multi-bin frequency vectors instead of
one-hot tokens or the 2-3 channel toy vectors of Phases 5/7/9 -- section
57's "only the sensory adapter may differ" applied to a genuinely
higher-dimensional continuous stream.

Each step emits an 8-bin "frequency vector" (bins 0-6 are tone
channels, bin 7 is a phrase-boundary marker) with Gaussian noise on
every bin. Two mutually exclusive phrases, chosen at random per episode:

  phrase 1: A -> B -> ambiguous -> E
  phrase 2: C -> D -> ambiguous -> F

The "ambiguous" tone occupies the *same* bin (4) in both phrases -- its
instantaneous frequency vector carries zero information about which
phrase it belongs to, by construction (same idea as Phase 2's
`P(object|sees) = 1/4`, and Phase 5/7's direction ambiguity). Only
earlier context (whether `A` or `C` fired) resolves whether `E` or `F`
comes next.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 8
A_BIN, B_BIN, C_BIN, D_BIN, AMBIGUOUS_BIN, E_BIN, F_BIN, BOUNDARY_BIN = range(8)


def generate_motif_stream(
    n_episodes: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns `(T, N_CHANNELS)`: one boundary row, then 4 tone rows, per
    episode. Every bin carries Gaussian noise; a tone's own bin is 1.0
    plus noise, every other bin is 0.0 plus noise.
    """

    def emit_tone(active_bin: int) -> list[float]:
        row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        row[active_bin] += 1.0
        return row.tolist()

    vectors: list[list[float]] = []
    for _ in range(n_episodes):
        boundary_row = rng.normal(0.0, noise_std, size=N_CHANNELS)
        boundary_row[BOUNDARY_BIN] += 1.0
        vectors.append(boundary_row.tolist())

        if rng.integers(0, 2) == 0:
            tones = [A_BIN, B_BIN, AMBIGUOUS_BIN, E_BIN]
        else:
            tones = [C_BIN, D_BIN, AMBIGUOUS_BIN, F_BIN]
        for bin_index in tones:
            vectors.append(emit_tone(bin_index))

    return np.array(vectors, dtype=np.float64)
