"""Tests for the multimodal cue task generator (design doc section 54,
"Multimodal stream -- Vision + audio + language using the same
relational core")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.multimodal_cue import (
    AMBIGUOUS_TONE,
    BOUNDARY,
    DECOY,
    E_BIN,
    F_BIN,
    N_CHANNELS,
    VISUAL_CUE,
    generate_multimodal_stream,
)


def test_stream_shape_matches_episode_count() -> None:
    rng = np.random.default_rng(0)
    stream = generate_multimodal_stream(n_episodes=10, noise_std=0.0, rng=rng)
    assert stream.shape == (10 * 5, N_CHANNELS)


def test_each_episode_starts_with_a_boundary_row() -> None:
    rng = np.random.default_rng(0)
    stream = generate_multimodal_stream(n_episodes=10, noise_std=0.0, rng=rng)
    for e in range(10):
        assert stream[e * 5, BOUNDARY] == 1.0


def test_visual_cue_is_plus_or_minus_one() -> None:
    rng = np.random.default_rng(1)
    stream = generate_multimodal_stream(n_episodes=100, noise_std=0.0, rng=rng)
    cues = stream[1::5, VISUAL_CUE]
    assert set(cues.tolist()) == {1.0, -1.0}


def test_decoy_channel_is_always_zero_without_noise() -> None:
    rng = np.random.default_rng(2)
    stream = generate_multimodal_stream(n_episodes=50, noise_std=0.0, rng=rng)
    assert np.all(stream[:, DECOY] == 0.0)


def test_resolved_bin_matches_the_cue_from_this_episode() -> None:
    rng = np.random.default_rng(3)
    stream = generate_multimodal_stream(n_episodes=200, noise_std=0.0, rng=rng)
    for e in range(200):
        cue = stream[e * 5 + 1, VISUAL_CUE]
        ambiguous_row = stream[e * 5 + 3]
        resolved_row = stream[e * 5 + 4]
        assert ambiguous_row[AMBIGUOUS_TONE] == 1.0
        if cue > 0:
            assert resolved_row[E_BIN] == 1.0
            assert resolved_row[F_BIN] == 0.0
        else:
            assert resolved_row[F_BIN] == 1.0
            assert resolved_row[E_BIN] == 0.0


def test_ambiguous_row_is_identical_regardless_of_cue() -> None:
    rng = np.random.default_rng(4)
    stream = generate_multimodal_stream(n_episodes=100, noise_std=0.0, rng=rng)
    ambiguous_rows = stream[3::5]
    first = ambiguous_rows[0]
    for row in ambiguous_rows:
        assert np.array_equal(row, first)
