from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.echolocation import (
    CROSSING_CHIME,
    DOT_POSITION,
    N_CHANNELS,
    RANGE_BEEP,
    TARGET_POSITION,
    VISIBLE,
    dead_reckoning_prediction,
    episode_vectors,
    generate_episodes,
    retarget_episode,
    shuffled_ranging_episodes,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import GrownRelations
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
    _continuous_episode,
    _readout_bytes,
    _relation_bytes,
    persist_target_state,
)


def test_native_input_is_a_raw_five_channel_vector_stream() -> None:
    episode = generate_episodes(1, seed=7, steps=40, hidden_steps=12)[0]
    vectors = episode_vectors(episode, visible=True)
    assert vectors.shape == (40, N_CHANNELS)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[:, DOT_POSITION], episode.positions)
    np.testing.assert_allclose(vectors[:, TARGET_POSITION], episode.target_position)
    np.testing.assert_array_equal(vectors[:, RANGE_BEEP], episode.range_beeps)
    np.testing.assert_array_equal(vectors[:, CROSSING_CHIME], episode.crossing_chimes)


def test_darkness_masks_vision_without_changing_audio() -> None:
    episode = generate_episodes(1, seed=11, steps=32, hidden_steps=9)[0]
    visible = episode_vectors(episode, visible=True)
    dark = episode_vectors(episode, visible=False)
    np.testing.assert_array_equal(visible[:, RANGE_BEEP:], dark[:, RANGE_BEEP:])
    assert np.all(dark[-9:, DOT_POSITION] == 0.0)
    assert np.all(dark[-9:, TARGET_POSITION] == 0.0)
    assert np.all(dark[-9:, VISIBLE] == 0.0)
    np.testing.assert_array_equal(visible[:-9, :3], dark[:-9, :3])


def test_shuffled_ranging_preserves_world_and_crossing_state() -> None:
    episodes = generate_episodes(8, seed=19, steps=40, hidden_steps=12)
    shuffled = shuffled_ranging_episodes(episodes, seed=23)
    changed = 0
    for original, control in zip(episodes, shuffled, strict=True):
        np.testing.assert_array_equal(original.positions, control.positions)
        np.testing.assert_array_equal(original.crossing_chimes, control.crossing_chimes)
        assert original.target_position == control.target_position
        changed += int(not np.array_equal(original.range_beeps, control.range_beeps))
    assert changed > 0


def test_silence_removes_only_audio_channels() -> None:
    episode = generate_episodes(1, seed=29, steps=40, hidden_steps=12)[0]
    real = episode_vectors(episode, visible=False)
    silent = episode_vectors(episode, visible=False, audio_mode="silent")
    np.testing.assert_array_equal(real[:, :RANGE_BEEP], silent[:, :RANGE_BEEP])
    assert np.all(silent[:, RANGE_BEEP:] == 0.0)


def test_retargeting_rebuilds_audio_without_changing_motion() -> None:
    episode = generate_episodes(1, seed=17, steps=40, hidden_steps=12)[0]
    moved = retarget_episode(episode, 0.44)
    np.testing.assert_array_equal(moved.positions, episode.positions)
    assert moved.blackout_start == episode.blackout_start
    assert moved.target_position == 0.44
    assert not np.array_equal(moved.range_beeps, episode.range_beeps)


def test_continuous_playground_moves_both_directions_and_keeps_sounding() -> None:
    episode = _continuous_episode(0.1, steps=180)
    velocity = np.diff(episode.positions)
    assert np.any(velocity > 0.0)
    assert np.any(velocity < 0.0)
    assert episode.range_beeps.sum() > 5
    assert episode.crossing_chimes.sum() >= 2


def test_dead_reckoning_uses_only_last_two_visible_positions() -> None:
    episode = generate_episodes(1, seed=31, steps=40, hidden_steps=12)[0]
    stop = episode.blackout_start
    expected = np.clip(
        episode.positions[stop - 1]
        + (len(episode.positions) - stop)
        * (episode.positions[stop - 1] - episode.positions[stop - 2]),
        -1.0,
        1.0,
    )
    assert dead_reckoning_prediction(episode) == expected


def test_visual_organ_persists_last_supported_target_state() -> None:
    estimate = np.asarray([[0.1, 0.2, 0.3, -0.8, 0.9]], dtype=np.float32)
    visibility = np.asarray([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    state = persist_target_state(estimate, visibility, np)
    np.testing.assert_allclose(state, [[0.1, 0.2, 0.3, 0.3, 0.3]])


def test_alm_64mb_body_class_uses_decimal_megabytes() -> None:
    assert DEFAULT_BODY_BYTES == 64_000_000


def test_three_organ_body_accounting_counts_resident_arrays() -> None:
    relations = GrownRelations(
        parent_a=np.zeros(3, dtype=np.int64),
        parent_b=np.ones(3, dtype=np.int64),
        w_a=np.zeros(3, dtype=np.float32),
        w_b=np.zeros(3, dtype=np.float32),
        mu=np.zeros(3, dtype=np.float32),
        log_sigma=np.zeros(3, dtype=np.float32),
        score=np.zeros(3, dtype=np.float32),
    )
    expected = sum(value.nbytes for value in relations.__dict__.values())
    assert _relation_bytes(relations) == expected
    assert _readout_bytes(
        np.zeros(4, dtype=np.float32), np.zeros(7, dtype=np.float32)
    ) == 44
