from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_language_resonance_shadow import (
    GENERIC_FEATURES,
    RECEPTORS,
    fit_resonance,
    predicted_utility,
    receptor_views,
    select_routes,
)


def _developed_cortex() -> SequenceCortexOpponentLearner:
    cortex = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=1,
        sequence_confirmations=1,
        max_context_bytes=6,
    )
    cortex.observe_chunk_probabilities(
        np.frombuffer(b"abcabcabcabdabcabcabcabd", dtype=np.uint8), learn=True
    )
    cortex.reset_episode()
    cortex.observe_chunk_probabilities(
        np.frombuffer(b"abc", dtype=np.uint8), learn=False
    )
    return cortex


def test_receptor_views_are_normalized_and_do_not_advance_cortex() -> None:
    cortex = _developed_cortex()
    state = (cortex.previous_assembly, tuple(cortex.sequence_history), cortex.steps)
    views = receptor_views(cortex)
    assert views.distributions.shape == (len(RECEPTORS), 256)
    assert views.features.shape == (len(RECEPTORS), GENERIC_FEATURES)
    np.testing.assert_allclose(views.distributions.sum(axis=1), 1.0)
    assert state == (
        cortex.previous_assembly,
        tuple(cortex.sequence_history),
        cortex.steps,
    )


def test_deep_shadow_view_matches_the_existing_cortex_prediction() -> None:
    cortex = _developed_cortex()
    views = receptor_views(cortex)
    np.testing.assert_allclose(
        views.distributions[-1], cortex.predict_next_distribution()
    )


def test_resonance_fit_prefers_the_locally_predictable_receptor() -> None:
    features = np.zeros((64, len(RECEPTORS), GENERIC_FEATURES), dtype=np.float64)
    signal = np.linspace(0.0, 1.0, 64)
    features[:, 0, 0] = signal
    bits = np.full((64, len(RECEPTORS)), 2.0, dtype=np.float64)
    bits[:, 0] = 2.0 - signal
    model = fit_resonance(features, bits)
    assert predicted_utility(features[-1], model)[0] > 0.0
    assert int(select_routes(features[-1], model)) == 0
