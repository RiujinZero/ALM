"""Tests for the continuous-oscillator task generator (design doc
section 54, "Dense continuous vectors")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.continuous_oscillator import (
    COS_CHANNEL,
    N_CHANNELS,
    RESET_CHANNEL,
    SIN_CHANNEL,
    generate_oscillator_stream,
)


def test_stream_shape_matches_episode_count_and_channel_count() -> None:
    rng = np.random.default_rng(0)
    stream = generate_oscillator_stream(
        n_episodes=5, steps_per_episode=4, angular_step=0.3, noise_std=0.0, rng=rng
    )
    assert stream.shape == (5 * (1 + 4), N_CHANNELS)


def test_stream_starts_each_episode_with_a_reset_spike_and_zero_position() -> None:
    rng = np.random.default_rng(0)
    stream = generate_oscillator_stream(
        n_episodes=5, steps_per_episode=4, angular_step=0.3, noise_std=0.0, rng=rng
    )
    episode_length = 1 + 4
    for e in range(5):
        row = stream[e * episode_length]
        assert row[RESET_CHANNEL] == 1.0
        assert row[SIN_CHANNEL] == 0.0
        assert row[COS_CHANNEL] == 0.0


def test_position_rows_have_no_reset_spike_and_lie_on_the_unit_circle() -> None:
    rng = np.random.default_rng(1)
    stream = generate_oscillator_stream(
        n_episodes=10, steps_per_episode=6, angular_step=0.4, noise_std=0.0, rng=rng
    )
    episode_length = 1 + 6
    for e in range(10):
        for i in range(6):
            row = stream[e * episode_length + 1 + i]
            assert row[RESET_CHANNEL] == 0.0
            radius = float(np.hypot(row[SIN_CHANNEL], row[COS_CHANNEL]))
            assert abs(radius - 1.0) < 1e-9  # no noise here, so exactly on the circle


def test_noise_perturbs_position_away_from_the_unit_circle() -> None:
    rng = np.random.default_rng(2)
    stream = generate_oscillator_stream(
        n_episodes=20, steps_per_episode=6, angular_step=0.4, noise_std=0.2, rng=rng
    )
    radii = [
        float(np.hypot(row[SIN_CHANNEL], row[COS_CHANNEL]))
        for row in stream
        if row[RESET_CHANNEL] == 0.0
    ]
    assert any(abs(r - 1.0) > 1e-3 for r in radii)
