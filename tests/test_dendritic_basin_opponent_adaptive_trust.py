from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
)

_STREAM = np.frombuffer(
    b"the quick brown fox jumps over the lazy dog. " * 60,
    dtype=np.uint8,
)


def test_adaptive_competition_off_by_default_is_bit_identical() -> None:
    """adaptive_competition defaults to False and must not move a single bit."""
    plain = ContextualOpponentAttractorLearner()
    explicit_off = ContextualOpponentAttractorLearner(adaptive_competition=False)
    plain_probabilities = plain.observe_chunk_probabilities(_STREAM, learn=True)
    off_probabilities = explicit_off.observe_chunk_probabilities(_STREAM, learn=True)
    np.testing.assert_array_equal(plain_probabilities, off_probabilities)


def test_adaptive_competition_learns_nontrivial_trust() -> None:
    """Enabling it must actually grow a live, non-empty trust table."""
    learner = ContextualOpponentAttractorLearner(adaptive_competition=True)
    learner.observe_chunk_probabilities(_STREAM, learn=True)

    assert learner.context_attribution.observations > 0
    assert len(learner.context_trust) > 0
    # Trust values must have moved away from the untouched default of 1.0
    # for at least one context -- otherwise the gate never actually fires.
    assert any(value != 1.0 for value in learner.context_trust.values())


def test_adaptive_competition_changes_predictions_vs_plain_competition() -> None:
    """The learned trust must actually reach the emitted probabilities."""
    plain = ContextualOpponentAttractorLearner()
    adaptive = ContextualOpponentAttractorLearner(adaptive_competition=True)

    plain_probabilities = plain.observe_chunk_probabilities(_STREAM, learn=True)
    adaptive_probabilities = adaptive.observe_chunk_probabilities(
        _STREAM, learn=True
    )

    assert not np.array_equal(plain_probabilities, adaptive_probabilities)


def test_credit_context_trust_is_gated_by_causal_attribution() -> None:
    """A posterior at or below the gate threshold must not update trust."""
    learner = ContextualOpponentAttractorLearner(
        adaptive_competition=True, attribution_gate_threshold=1.0
    )
    # threshold=1.0 makes posterior_self() > 1.0 impossible, so the gate
    # must never fire regardless of how much evidence accumulates.
    learner.observe_chunk_probabilities(_STREAM, learn=True)

    assert all(value == 1.0 for value in learner.context_trust.values())


def test_consolidate_refuses_mismatched_adaptive_competition_configuration() -> None:
    left = ContextualOpponentAttractorLearner(adaptive_competition=True)
    right = ContextualOpponentAttractorLearner(adaptive_competition=False)
    try:
        left.consolidate(right)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched configuration")
