"""Phase 7 (design doc section 54: "Dense continuous vectors -- Test
arbitrary continuous streams. No sparse categorical assumptions."): a
continuous analogue of Phase 5's `moving_point` ring task. Same
structural ambiguity -- direction is unresolvable from the instantaneous
position alone, only recent history resolves it -- but the stream itself
is now dense, real-valued vectors: no one-hot channels, no discrete
vocabulary anywhere, not even for the episode-boundary marker.

Each step emits a 3-vector `(sin(theta), cos(theta), reset_marker)`,
`theta` advancing by a fixed angular step per event in a random direction
(+1 or -1) chosen once per episode; Gaussian observation noise is added
to `sin`/`cos`. `reset_marker` is a continuous spike (1.0) at the first
event of each episode and 0.0 elsewhere -- still a plain continuous
channel, not a categorical one-hot index.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 3
SIN_CHANNEL, COS_CHANNEL, RESET_CHANNEL = 0, 1, 2


def generate_oscillator_stream(
    n_episodes: int,
    steps_per_episode: int,
    angular_step: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns a `(n_episodes * (1 + steps_per_episode), N_CHANNELS)`
    float array: one boundary-marker row per episode, followed by
    `steps_per_episode` position rows.
    """
    vectors: list[list[float]] = []
    for _ in range(n_episodes):
        vectors.append([0.0, 0.0, 1.0])
        direction = 1.0 if rng.integers(0, 2) == 0 else -1.0
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        for _ in range(steps_per_episode):
            theta += direction * angular_step
            noisy_sin = float(np.sin(theta) + rng.normal(0.0, noise_std))
            noisy_cos = float(np.cos(theta) + rng.normal(0.0, noise_std))
            vectors.append([noisy_sin, noisy_cos, 0.0])
    return np.array(vectors, dtype=np.float64)
