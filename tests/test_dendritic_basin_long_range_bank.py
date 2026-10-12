"""long_range_bank (Idea 2 Stage A) and burst-gated growth (Idea 1 signal
repurposed as an event-capture trigger, per the user's suggestion).

See docs/HTM_Inspired_Followup_Experiments.md, Idea 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    PhraseEventOpponentLearner,
    SequenceCortexOpponentLearner,
)


def _stream(text: bytes, times: int) -> np.ndarray:
    return np.frombuffer(text * times, dtype=np.uint8)


def test_default_off_is_unchanged() -> None:
    stream = _stream(b"the quick brown fox jumps over the lazy dog. ", 500)
    a = SequenceCortexOpponentLearner()
    b = SequenceCortexOpponentLearner(long_range_bank=False)
    pa = a.observe_chunk_probabilities(stream.copy(), learn=True)
    pb = b.observe_chunk_probabilities(stream.copy(), learn=True)
    assert np.array_equal(pa, pb, equal_nan=True)
    assert a.embodied_bytes == b.embodied_bytes


def test_long_range_bank_grows_and_shifts_predictions_when_enabled() -> None:
    stream = _stream(b"the quick brown fox jumps over the lazy dog. ", 500)
    baseline = SequenceCortexOpponentLearner()
    variant = SequenceCortexOpponentLearner(
        long_range_bank=True, long_range_context_bytes=8, long_range_strength=4.0
    )
    pa = baseline.observe_chunk_probabilities(stream.copy(), learn=True)
    pb = variant.observe_chunk_probabilities(stream.copy(), learn=True)
    assert len(variant.long_range_counts) > 0
    assert not np.array_equal(pa, pb, equal_nan=True)


def test_long_range_strength_name_does_not_collide_with_phrase_event_learner() -> None:
    """PhraseEventOpponentLearner already had its own, unrelated
    apical_strength (word-level event modulation) before this feature
    existed. Confirms the rename left that alone and long_range_bank
    correctly defaults to off for that class."""
    learner = PhraseEventOpponentLearner(apical_strength=3.0)
    assert learner.apical_strength == 3.0
    assert learner.long_range_bank is False


def test_gate_on_burst_requires_burst_surprise_signal() -> None:
    with pytest.raises(ValueError, match="requires burst_surprise_signal"):
        SequenceCortexOpponentLearner(
            long_range_bank=True, long_range_gate_on_burst=True, burst_surprise_signal=False
        )


def test_burst_gated_growth_is_sparser_than_unconditional_growth() -> None:
    """The actual claim under test: gating long-range growth on
    burst_fraction (repurposing Idea 1's signal as an event-capture trigger,
    rather than a prediction-time damper) should produce a more selective
    -- i.e. smaller -- long-range bank than growing on every byte
    unconditionally, on the same real-ish data."""
    rng = np.random.default_rng(7)
    # Mix of repetitive (low-novelty) and random (high-novelty) spans so
    # there is a real difference between "quiet" and "bursting" periods for
    # gating to select between.
    repetitive = b"the quick brown fox jumps over the lazy dog. " * 40
    novel = bytes(rng.integers(32, 127, size=2000, dtype=np.uint8))
    stream = np.frombuffer(repetitive + novel + repetitive + novel, dtype=np.uint8)

    unconditional = SequenceCortexOpponentLearner(
        long_range_bank=True,
        long_range_context_bytes=8,
        burst_surprise_signal=True,
        burst_ema_rate=0.02,
    )
    gated = SequenceCortexOpponentLearner(
        long_range_bank=True,
        long_range_context_bytes=8,
        burst_surprise_signal=True,
        burst_ema_rate=0.02,
        long_range_gate_on_burst=True,
        long_range_gate_threshold=0.1,
    )
    unconditional.observe_chunk_probabilities(stream.copy(), learn=True)
    gated.observe_chunk_probabilities(stream.copy(), learn=True)

    assert len(gated.long_range_counts) < len(unconditional.long_range_counts)
