"""Tests for the visual-occlusion task generator (design doc section 54,
"Visual world -- ... occlusion, identity")."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.visual_occlusion import (
    HUE_CHANNEL,
    N_CHANNELS,
    POSITION_CHANNEL,
    VISIBLE_CHANNEL,
    generate_occlusion_stream,
)


def _generate(n_episodes=5, occlusion_min=3, occlusion_max=3, seed=0):
    rng = np.random.default_rng(seed)
    return generate_occlusion_stream(
        n_episodes=n_episodes,
        visible_steps_before=4,
        occlusion_min=occlusion_min,
        occlusion_max=occlusion_max,
        visible_steps_after=4,
        position_step=0.1,
        hue_values=(-1.0, 0.0, 1.0),
        rng=rng,
    )


def test_stream_shape_and_reappearance_index_count() -> None:
    stream, _starts, reappearance_indices, true_hues = _generate(n_episodes=6)
    assert stream.shape[1] == N_CHANNELS
    assert len(reappearance_indices) == 6
    assert len(true_hues) == 6


def test_occluded_rows_are_fully_zeroed() -> None:
    stream, _starts, reappearance_indices, _true_hues = _generate(
        n_episodes=5, occlusion_min=3, occlusion_max=3
    )
    episode_length = 4 + 3 + 4
    for e in range(5):
        occlusion_start = e * episode_length + 4
        for i in range(3):
            row = stream[occlusion_start + i]
            assert row[POSITION_CHANNEL] == 0.0
            assert row[HUE_CHANNEL] == 0.0
            assert row[VISIBLE_CHANNEL] == 0.0


def test_reappearance_row_is_visible_with_the_true_hue() -> None:
    stream, _starts, reappearance_indices, true_hues = _generate(n_episodes=8)
    for i, hue in zip(reappearance_indices, true_hues, strict=True):
        row = stream[i]
        assert row[VISIBLE_CHANNEL] == 1.0
        assert row[HUE_CHANNEL] == hue


def test_hue_is_constant_across_all_visible_rows_of_an_episode() -> None:
    stream, _starts, _reappearance_indices, true_hues = _generate(
        n_episodes=4, occlusion_min=2, occlusion_max=2
    )
    episode_length = 4 + 2 + 4
    for e in range(4):
        for i in range(episode_length):
            row = stream[e * episode_length + i]
            if row[VISIBLE_CHANNEL] == 1.0:
                assert row[HUE_CHANNEL] == true_hues[e]


def test_variable_occlusion_length_is_respected() -> None:
    """Directly count each episode's run of fully-occluded rows (rather
    than back-computing it from cumulative offsets) and confirm the
    randomized `occlusion_min`/`occlusion_max` range is actually used,
    not collapsed to one fixed length.
    """
    stream, _starts, reappearance_indices, _true_hues = _generate(
        n_episodes=50, occlusion_min=1, occlusion_max=6, seed=1
    )
    occlusion_lengths = []
    for reappear_at in reappearance_indices:
        length = 0
        i = int(reappear_at) - 1
        while i >= 0 and stream[i, VISIBLE_CHANNEL] == 0.0:
            length += 1
            i -= 1
        occlusion_lengths.append(length)

    assert all(1 <= length <= 6 for length in occlusion_lengths)
    assert len(set(occlusion_lengths)) > 1
