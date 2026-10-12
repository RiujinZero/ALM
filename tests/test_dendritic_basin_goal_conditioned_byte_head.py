from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    EpisodeByteTrace,
    MatchedGoalByteHead,
    likelihood_ratio_correction,
    shuffled_goal_control,
)


def _trained_counterbalanced_head() -> MatchedGoalByteHead:
    head = MatchedGoalByteHead(
        2,
        context_order=2,
        context_bucket_count=127,
        local_prior_mass=1.0,
        goal_backoff_mass=1.0,
    )
    episodes = []
    for _ in range(32):
        episodes.extend(
            (
                EpisodeByteTrace(b"XA", (1.0, 0.0), prefix=b"shared>"),
                EpisodeByteTrace(b"XB", (0.0, 1.0), prefix=b"shared>"),
            )
        )
    head.fit_episodes(episodes)
    return head


def test_identical_local_suffix_produces_bit_identical_q0_trace() -> None:
    head = _trained_counterbalanced_head()

    local_a, _ = head.trace_probabilities(
        b"an unrelated beginning::ed", b"XA", (1.0, 0.0)
    )
    local_b, _ = head.trace_probabilities(b"a different episode::ed", b"XA", (0.0, 1.0))

    assert np.array_equal(local_a, local_b)


def test_true_and_swapped_goals_reverse_counterbalanced_preference() -> None:
    head = _trained_counterbalanced_head()
    prefix = b"shared>"
    goal_a = (1.0, 0.0)
    goal_b = (0.0, 1.0)

    true_score = head.score_trace(prefix, b"XA", goal_a).goal_log_probability
    true_score += head.score_trace(prefix, b"XB", goal_b).goal_log_probability
    crossed_score = head.score_trace(prefix, b"XB", goal_a).goal_log_probability
    crossed_score += head.score_trace(prefix, b"XA", goal_b).goal_log_probability

    swapped_true = head.score_trace(prefix, b"XA", goal_b).goal_log_probability
    swapped_true += head.score_trace(prefix, b"XB", goal_a).goal_log_probability
    swapped_crossed = head.score_trace(prefix, b"XB", goal_b).goal_log_probability
    swapped_crossed += head.score_trace(prefix, b"XA", goal_a).goal_log_probability

    assert true_score > crossed_score
    assert swapped_true < swapped_crossed
    assert np.isclose(true_score - crossed_score, swapped_crossed - swapped_true)


def test_scoring_is_pure_and_snapshot_round_trip_is_exact() -> None:
    head = _trained_counterbalanced_head()
    before = head.snapshot()

    head.score_trace(b"shared>", b"XA", (0.75, 0.25))
    head.trace_probabilities(b"shared>", b"XB", (0.25, 0.75))
    head.correct_base_distribution(
        np.full(256, 1.0 / 256.0),
        b"shared>X",
        (0.75, 0.25),
        beta=0.5,
    )
    after = head.snapshot()

    assert before.episodes_seen == after.episodes_seen
    assert before.bytes_seen == after.bytes_seen
    assert np.array_equal(before.local_counts, after.local_counts)
    assert np.array_equal(before.goal_counts, after.goal_counts)
    restored = MatchedGoalByteHead.from_snapshot(before)
    expected = head.trace_probabilities(b"shared>", b"XA", (0.75, 0.25))
    actual = restored.trace_probabilities(b"shared>", b"XA", (0.75, 0.25))
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


def test_beta_zero_is_an_exact_base_bypass() -> None:
    rng = np.random.default_rng(9)
    base = rng.random(256, dtype=np.float32)
    base /= base.sum(dtype=np.float32)
    local = rng.dirichlet(np.ones(256))
    goal = rng.dirichlet(np.ones(256))

    corrected = likelihood_ratio_correction(base, local, goal, beta=0.0)

    assert corrected.dtype == base.dtype
    assert np.array_equal(corrected, base)


def test_episode_training_never_joins_adjacent_responses() -> None:
    head = MatchedGoalByteHead(2, context_order=1, context_bucket_count=4093)
    head.fit_episode(b"B", (1.0, 0.0))
    head.fit_episode(b"A", (0.0, 1.0))

    local_after_b, _ = head.next_byte_distributions(b"B", (1.0, 0.0))

    assert np.array_equal(local_after_b, np.full(256, 1.0 / 256.0))
    assert head.episodes_seen == 2
    assert head.bytes_seen == 2


def test_shuffled_goal_control_is_reproducible_and_has_no_fixed_rows() -> None:
    episodes = [
        EpisodeByteTrace(bytes([index]), (float(index), 1.0), prefix=b"p")
        for index in range(1, 7)
    ]

    first, permutation = shuffled_goal_control(episodes, seed=12)
    second, second_permutation = shuffled_goal_control(episodes, seed=12)

    assert np.array_equal(permutation, second_permutation)
    assert np.all(permutation != np.arange(len(episodes)))
    for index, shuffled in enumerate(first):
        assert shuffled.response == episodes[index].response
        assert shuffled.prefix == episodes[index].prefix
        np.testing.assert_array_equal(
            shuffled.goal_distribution,
            episodes[int(permutation[index])].goal_distribution,
        )
        np.testing.assert_array_equal(
            shuffled.goal_distribution, second[index].goal_distribution
        )
