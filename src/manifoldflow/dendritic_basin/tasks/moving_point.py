"""A synthetic non-language event stream: a point moving around a fixed-size
ring (design doc section 31 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`:
the "data-agnostic vector smoke" -- the same relation-discovery mechanism
validated on language in Phases 2-4, tested completely unchanged on a
structurally unrelated stream).

At each episode, a direction (+1 or -1) is chosen at random and held for
the whole episode; the point moves one step per event. The *current*
position alone never predicts the next position (a symmetric ambiguity,
same as `P(object|sees)=1/4` in the language tasks) -- only the recent
history of positions (which way it's been moving) resolves it. This
mirrors the language tasks' structure exactly, just with positions on a
ring standing in for names and objects.
"""

from __future__ import annotations

import numpy as np

RESET_CHANNEL = 0  # a dedicated "episode boundary" event, distinct from any position


def generate_position_stream(
    n_episodes: int, ring_size: int, steps_per_episode: int, rng: np.random.Generator
) -> list[int]:
    """Channel ids: 0 = episode-boundary marker, 1..ring_size = ring
    positions 0..ring_size-1 (offset by 1 so position 0 and the boundary
    marker are distinct channels).
    """
    tokens: list[int] = []
    for _ in range(n_episodes):
        tokens.append(RESET_CHANNEL)
        direction = 1 if rng.integers(0, 2) == 0 else -1
        position = int(rng.integers(0, ring_size))
        for _ in range(steps_per_episode):
            tokens.append(1 + position)
            position = (position + direction) % ring_size
    return tokens


def vocab_size(ring_size: int) -> int:
    return ring_size + 1
