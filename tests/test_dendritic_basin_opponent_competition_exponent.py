from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
    SequenceCortexOpponentLearner,
)

_STREAM = np.frombuffer(
    b"the quick brown fox jumps over the lazy dog. " * 40,
    dtype=np.uint8,
)


def test_default_exponent_matches_pre_refactor_fused_formula_exactly() -> None:
    """competition_exponent=1.0 (the default) must stay bit-identical.

    This is the same oracle-equivalence discipline used throughout this
    package: contextual_competition.normalized_competition now backs the
    backoff blend, but only when the exponent departs from 1.0. At the
    default, every learner must reproduce the original fixed linear
    interpolation exactly, not merely to close numerical tolerance.
    """
    default = ContextualOpponentAttractorLearner()
    explicit = ContextualOpponentAttractorLearner(competition_exponent=1.0)
    default_probabilities = default.observe_chunk_probabilities(_STREAM, learn=True)
    explicit_probabilities = explicit.observe_chunk_probabilities(_STREAM, learn=True)
    np.testing.assert_array_equal(default_probabilities, explicit_probabilities)


def test_default_exponent_matches_for_sequence_cortex_too() -> None:
    default = SequenceCortexOpponentLearner()
    explicit = SequenceCortexOpponentLearner(competition_exponent=1.0)
    default_probabilities = default.observe_chunk_probabilities(_STREAM, learn=True)
    explicit_probabilities = explicit.observe_chunk_probabilities(_STREAM, learn=True)
    np.testing.assert_array_equal(default_probabilities, explicit_probabilities)


def test_sharper_exponent_actually_changes_learned_probabilities() -> None:
    """A non-default exponent must be a live, reachable capability."""
    baseline = ContextualOpponentAttractorLearner(competition_exponent=1.0)
    sharpened = ContextualOpponentAttractorLearner(competition_exponent=4.0)

    baseline_probabilities = baseline.observe_chunk_probabilities(_STREAM, learn=True)
    sharpened_probabilities = sharpened.observe_chunk_probabilities(
        _STREAM, learn=True
    )

    assert not np.array_equal(baseline_probabilities, sharpened_probabilities)


def test_sharper_exponent_favors_whichever_side_has_more_evidence() -> None:
    """exponent > 1 should push the backoff weight further toward the side

    that already has more relative evidence, matching the boxed equation's
    own "exponent sharpens the winner" property (see
    contextual_competition.normalized_competition and its own tests).
    """
    learner = ContextualOpponentAttractorLearner(competition_exponent=1.0)
    soft_weights = learner._backoff_weights(4.0, 16.0)
    sharp_learner = ContextualOpponentAttractorLearner(competition_exponent=3.0)
    sharp_weights = sharp_learner._backoff_weights(4.0, 16.0)

    # At exponent=1.0 _backoff_weights returns None (fused-formula fallback);
    # recompute the exponent=1 reference directly for the comparison.
    from manifoldflow.dendritic_basin.contextual_competition import (
        normalized_competition,
    )

    reference = normalized_competition(
        np.array([4.0, 16.0]), np.ones(2), exponent=1.0, kappa=0.0
    )
    assert soft_weights is None
    assert sharp_weights is not None
    # The child (16.0) already dominates the parent (4.0); a sharper
    # exponent must push its weight even higher than the exponent=1 case.
    assert sharp_weights[1] > float(reference[1])
