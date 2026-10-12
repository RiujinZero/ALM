"""Tests for the moving-point task generator (design doc section 31 of
`Manifold_Dendritic_Stream_Learner_Design_v0.2.md`: the data-agnostic
vector smoke)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.moving_point import (
    RESET_CHANNEL,
    generate_position_stream,
    vocab_size,
)


def test_vocab_size_is_ring_size_plus_reset_channel() -> None:
    assert vocab_size(8) == 9


def test_stream_starts_each_episode_with_the_reset_channel() -> None:
    rng = np.random.default_rng(0)
    stream = generate_position_stream(5, ring_size=8, steps_per_episode=4, rng=rng)
    episode_length = 1 + 4  # reset marker + steps_per_episode positions
    for e in range(5):
        assert stream[e * episode_length] == RESET_CHANNEL


def test_position_channels_are_within_the_valid_range() -> None:
    rng = np.random.default_rng(1)
    ring_size = 8
    stream = generate_position_stream(10, ring_size, steps_per_episode=6, rng=rng)
    for token in stream:
        assert 0 <= token <= ring_size


def test_consecutive_positions_within_an_episode_move_by_exactly_one_step() -> None:
    """Within one episode, consecutive positions differ by +1 or -1 (mod
    ring_size) -- the direction is constant within an episode but the
    generator itself doesn't leak that direction as a separate channel.
    """
    rng = np.random.default_rng(2)
    ring_size = 8
    stream = generate_position_stream(20, ring_size, steps_per_episode=6, rng=rng)
    episode_length = 1 + 6
    for e in range(20):
        start = e * episode_length + 1  # skip the reset marker
        positions = [stream[start + i] - 1 for i in range(6)]
        diffs = {(positions[i + 1] - positions[i]) % ring_size for i in range(5)}
        assert diffs <= {1, ring_size - 1}  # either always +1 or always -1
        assert len(diffs) == 1  # direction is constant within the episode
