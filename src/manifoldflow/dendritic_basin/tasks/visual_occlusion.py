"""Phase 9 (design doc section 54: "Visual world -- Moving objects,
persistence, occlusion, identity."): a synthetic visual-world analogue
of Phase 3's distractor-binding task, built on dense continuous vectors
(Phase 7's representation) instead of discrete tokens -- section 57's
"only the sensory adapter may differ."

Each episode: a single object with a constant scalar "identity" value
(`hue`, standing in for e.g. color/shape identity) moves along a 1D line
at a constant per-episode velocity. After several visible steps, the
object is occluded for a random span (as if passing behind an occluder)
-- during occlusion, position and hue are both masked to zero, exactly
what a real vision system would receive (no signal, not "the same value
held constant"). The object then reappears. The task: recover its hue
value -- "which object is this?" -- the instant occlusion ends, exactly
Phase 3's original binding question (design's own finding there: a
plain decaying trace cannot survive a long enough gap; only persistent,
non-decaying state can), now on continuous data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_CHANNELS = 3
POSITION_CHANNEL, HUE_CHANNEL, VISIBLE_CHANNEL = 0, 1, 2


def generate_occlusion_stream(
    n_episodes: int,
    visible_steps_before: int,
    occlusion_min: int,
    occlusion_max: int,
    visible_steps_after: int,
    position_step: float,
    hue_values: Sequence[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(vector_stream, episode_start_indices, reappearance_indices,
    true_hues)`: `vector_stream` is `(T, 3)` -- `(position, hue, visible)`
    per step; `episode_start_indices[e]` is episode `e`'s first row (a new,
    unrelated object -- callers that want to isolate one episode's
    identity signal should reset any persistent state there);
    `reappearance_indices[e]` is the row index of episode `e`'s first
    post-occlusion (reappeared) step; `true_hues[e]` is that episode's
    hue value.
    """
    vectors: list[list[float]] = []
    episode_start_indices: list[int] = []
    reappearance_indices: list[int] = []
    true_hues: list[float] = []
    for _ in range(n_episodes):
        episode_start_indices.append(len(vectors))
        hue = float(rng.choice(hue_values))
        direction = 1.0 if rng.integers(0, 2) == 0 else -1.0
        position = float(rng.uniform(-1.0, 1.0))

        for _ in range(visible_steps_before):
            position += direction * position_step
            vectors.append([position, hue, 1.0])

        occlusion_len = int(rng.integers(occlusion_min, occlusion_max + 1))
        for _ in range(occlusion_len):
            position += direction * position_step  # the real object keeps moving
            vectors.append([0.0, 0.0, 0.0])  # but nothing is observed

        reappearance_indices.append(len(vectors))
        true_hues.append(hue)

        for _ in range(visible_steps_after):
            position += direction * position_step
            vectors.append([position, hue, 1.0])

    return (
        np.array(vectors, dtype=np.float64),
        np.array(episode_start_indices, dtype=np.int64),
        np.array(reappearance_indices, dtype=np.int64),
        np.array(true_hues, dtype=np.float64),
    )
