from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    MatchedSignedGoalByteHead,
    SignedEpisodeByteTrace,
    shuffled_signed_goal_control,
)


def _trained_head() -> MatchedSignedGoalByteHead:
    head = MatchedSignedGoalByteHead(
        2,
        context_order=2,
        context_bucket_count=127,
        association_bucket_count=17,
        association_prior_mass=1.0,
        goal_gain=2.0,
    )
    episodes = []
    for _ in range(32):
        episodes.extend(
            (
                SignedEpisodeByteTrace(b"XA", (1.0, 0.0), prefix=b"shared>"),
                SignedEpisodeByteTrace(b"XB", (-1.0, 0.0), prefix=b"shared>"),
            )
        )
    head.fit_episodes(episodes)
    return head


def test_signed_direction_reverses_preference_while_q0_stays_matched() -> None:
    head = _trained_head()
    local_positive, goal_positive = head.next_byte_distributions(b"shared>X", (1, 0))
    local_negative, goal_negative = head.next_byte_distributions(b"shared>X", (-1, 0))

    assert np.array_equal(local_positive, local_negative)
    assert goal_positive[ord("A")] > goal_positive[ord("B")]
    assert goal_negative[ord("B")] > goal_negative[ord("A")]

    true = head.score_trace(b"shared>", b"XA", (1, 0)).goal_log_probability
    true += head.score_trace(b"shared>", b"XB", (-1, 0)).goal_log_probability
    swapped = head.score_trace(b"shared>", b"XA", (-1, 0)).goal_log_probability
    swapped += head.score_trace(b"shared>", b"XB", (1, 0)).goal_log_probability
    assert true > swapped


def test_zero_goal_is_exact_q0_and_exact_base_correction_at_nonzero_beta() -> None:
    head = _trained_head()
    local, goal_off = head.next_byte_distributions(b"shared>X", (0.0, 0.0))
    rng = np.random.default_rng(23)
    base = rng.random(256, dtype=np.float32)
    base /= base.sum(dtype=np.float32)

    corrected = head.correct_base_distribution(base, b"shared>X", (0.0, 0.0), beta=7.0)

    assert np.array_equal(local, goal_off)
    assert corrected.dtype == base.dtype
    assert np.array_equal(corrected, base)


def test_extreme_signed_goal_is_norm_and_probability_clamped_safely() -> None:
    head = MatchedSignedGoalByteHead(
        2,
        context_bucket_count=31,
        association_bucket_count=1,
        goal_gain=1e6,
        max_goal_norm=2.0,
        logit_clip=3.0,
        minimum_probability=1e-8,
    )
    for _ in range(8):
        head.fit_episode(b"A", (1e300, -1e300))
        head.fit_episode(b"B", (-1e300, 1e300))

    local, conditioned = head.next_byte_distributions(b"", (1e300, -1e300))

    assert np.all(np.isfinite(local))
    assert np.all(np.isfinite(conditioned))
    assert np.isclose(conditioned.sum(), 1.0)
    assert np.min(conditioned) >= 1e-8
    assert np.max(np.abs(np.log(conditioned / local))) < 20.0


def test_signed_scoring_is_pure_and_snapshot_round_trip_is_exact() -> None:
    head = _trained_head()
    before = head.snapshot()

    expected = head.trace_probabilities(b"shared>", b"XA", (0.75, -0.25))
    head.correct_base_distribution(
        np.full(256, 1.0 / 256.0),
        b"shared>X",
        (0.75, -0.25),
        beta=1.0,
    )
    after = head.snapshot()

    assert before.episodes_seen == after.episodes_seen
    assert before.bytes_seen == after.bytes_seen
    assert np.array_equal(before.local_counts, after.local_counts)
    assert np.array_equal(before.association_counts, after.association_counts)
    assert np.array_equal(before.goal_sums, after.goal_sums)
    restored = MatchedSignedGoalByteHead.from_snapshot(before)
    actual = restored.trace_probabilities(b"shared>", b"XA", (0.75, -0.25))
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


def test_signed_shuffle_is_reproducible_strict_and_preserves_byte_events() -> None:
    episodes = [
        SignedEpisodeByteTrace(bytes([index]), (float(index), -float(index)))
        for index in range(1, 8)
    ]

    first, permutation = shuffled_signed_goal_control(episodes, seed=41)
    second, second_permutation = shuffled_signed_goal_control(episodes, seed=41)

    assert np.array_equal(permutation, second_permutation)
    assert np.all(permutation != np.arange(len(episodes)))
    for index, episode in enumerate(first):
        assert episode.response == episodes[index].response
        np.testing.assert_array_equal(
            episode.goal_vector,
            episodes[int(permutation[index])].goal_vector,
        )
        np.testing.assert_array_equal(episode.goal_vector, second[index].goal_vector)


def test_derived_association_cache_is_excluded_and_invalidated_by_fit() -> None:
    head = _trained_head()
    persistent_bytes = head.allocated_bytes

    _, before = head.next_byte_distributions(b"shared>X", (1.0, 0.0))
    cached_bytes = head.transient_cache_bytes
    head.fit_episode(b"B", (1.0, 0.0), prefix=b"shared>X")
    _, after = head.next_byte_distributions(b"shared>X", (1.0, 0.0))

    assert cached_bytes > 0
    assert head.allocated_bytes == persistent_bytes
    assert head.transient_cache_bytes == cached_bytes
    assert not np.array_equal(before, after)
