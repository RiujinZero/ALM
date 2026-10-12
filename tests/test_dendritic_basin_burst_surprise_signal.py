"""burst_surprise_signal: HTM-inspired population-level novelty signal.

See docs/HTM_Inspired_Followup_Experiments.md, Idea 1. This is opt-in and
independent of (not a replacement for) adaptive_competition's existing
per-context TD(0) trust signal.
"""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)


def _stream(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.frombuffer(
        b"the quick brown fox jumps over the lazy dog. " * (size // 46 + 1), dtype=np.uint8
    )[:size]


def test_default_off_is_bit_identical_to_before_this_feature_existed() -> None:
    stream = _stream(0, 20_000)
    baseline = SequenceCortexOpponentLearner()
    with_new_default = SequenceCortexOpponentLearner(burst_surprise_signal=False)
    p1 = baseline.observe_chunk_probabilities(stream.copy(), learn=True)
    p2 = with_new_default.observe_chunk_probabilities(stream.copy(), learn=True)
    assert np.array_equal(p1, p2, equal_nan=True)
    assert baseline.embodied_bytes == with_new_default.embodied_bytes


def test_burst_fraction_rises_on_novel_data_and_falls_on_familiar_data() -> None:
    # Random bytes keep generating new contexts throughout (unlike a short
    # repeating phrase, which exhausts its novelty in the first pass), so
    # burst_fraction stays elevated for as long as genuinely novel data
    # keeps arriving, then decays once the same data repeats verbatim.
    learner = SequenceCortexOpponentLearner(burst_surprise_signal=True, burst_ema_rate=0.01)
    rng = np.random.default_rng(3)
    novel = rng.integers(0, 256, size=50_000, dtype=np.uint8)
    learner.observe_chunk_probabilities(novel, learn=True)
    fraction_after_novel = learner.burst_fraction

    learner.observe_chunk_probabilities(novel, learn=True)  # identical replay
    fraction_after_familiar = learner.burst_fraction

    assert fraction_after_novel > 0.0
    assert fraction_after_familiar < fraction_after_novel


def test_burst_damped_gain_is_identity_when_signal_disabled() -> None:
    learner = SequenceCortexOpponentLearner(burst_surprise_signal=False)
    learner.burst_fraction = 0.9  # pretend it were high; must not matter when off
    assert learner._burst_damped_gain(1.0) == 1.0


def test_burst_damped_gain_shrinks_gain_when_enabled_and_bursting() -> None:
    learner = SequenceCortexOpponentLearner(burst_surprise_signal=True)
    learner.burst_fraction = 0.9
    damped = learner._burst_damped_gain(1.0)
    assert 0.05 <= damped < 1.0


def test_opt_in_does_not_change_default_behavior_end_to_end() -> None:
    """Same discipline as structural_support/connectivity_pruning's own
    regression test: constructing with the new flag left at its default
    must not perturb anything, even with adaptive_competition also on."""
    stream = _stream(2, 30_000)
    a = SequenceCortexOpponentLearner(adaptive_competition=True)
    b = SequenceCortexOpponentLearner(adaptive_competition=True, burst_surprise_signal=False)
    pa = a.observe_chunk_probabilities(stream.copy(), learn=True)
    pb = b.observe_chunk_probabilities(stream.copy(), learn=True)
    assert np.array_equal(pa, pb, equal_nan=True)
