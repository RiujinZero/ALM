"""Tests for the audio-motif task generator (design doc section 54,
"Audio -- Continuous frequency vectors")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.audio_motif import (
    A_BIN,
    AMBIGUOUS_BIN,
    B_BIN,
    BOUNDARY_BIN,
    C_BIN,
    D_BIN,
    E_BIN,
    F_BIN,
    N_CHANNELS,
    generate_motif_stream,
)


def test_stream_shape_matches_episode_count() -> None:
    rng = np.random.default_rng(0)
    stream = generate_motif_stream(n_episodes=10, noise_std=0.0, rng=rng)
    assert stream.shape == (10 * 5, N_CHANNELS)  # 1 boundary + 4 tone rows


def test_each_episode_starts_with_a_boundary_row() -> None:
    rng = np.random.default_rng(0)
    stream = generate_motif_stream(n_episodes=10, noise_std=0.0, rng=rng)
    for e in range(10):
        row = stream[e * 5]
        assert row[BOUNDARY_BIN] == 1.0
        for bin_index in range(N_CHANNELS):
            if bin_index != BOUNDARY_BIN:
                assert row[bin_index] == 0.0


def test_only_two_phrases_ever_occur() -> None:
    rng = np.random.default_rng(1)
    stream = generate_motif_stream(n_episodes=200, noise_std=0.0, rng=rng)
    phrase_a_bins = {A_BIN, B_BIN, AMBIGUOUS_BIN, E_BIN}
    phrase_c_bins = {C_BIN, D_BIN, AMBIGUOUS_BIN, F_BIN}
    for e in range(200):
        tone_rows = stream[e * 5 + 1 : e * 5 + 5]
        active_bins = [int(np.argmax(row)) for row in tone_rows]
        assert set(active_bins) in (phrase_a_bins, phrase_c_bins)


def test_ambiguous_bin_vector_is_identical_regardless_of_phrase() -> None:
    """The whole point of the task: the ambiguous tone's own frequency
    vector must carry no phrase information (noiseless case: it's
    exactly the same one-hot vector in both phrases).
    """
    rng = np.random.default_rng(2)
    stream = generate_motif_stream(n_episodes=100, noise_std=0.0, rng=rng)
    # episode layout: [boundary, tone1, tone2, ambiguous, tone4] -> offset 3
    ambiguous_rows = stream[3::5]
    first = ambiguous_rows[0]
    for row in ambiguous_rows:
        assert np.array_equal(row, first)


def test_noise_perturbs_inactive_bins_away_from_exact_zero() -> None:
    rng = np.random.default_rng(3)
    stream = generate_motif_stream(n_episodes=50, noise_std=0.1, rng=rng)
    # bin B is inactive on the first tone row of an "A phrase" episode, but
    # should not be exactly 0.0 once Gaussian noise is added to every bin
    inactive_bin_values = stream[1::5, B_BIN]
    assert not np.all(inactive_bin_values == 0.0)
