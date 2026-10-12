from types import SimpleNamespace

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase145_residual_route_codevelopment as phase145,
)
from manifoldflow.experiments import (
    dendritic_basin_phase146_signed_residual_permission as phase146,
)


def test_residual_route_expresses_only_signed_missing_word_current() -> None:
    baseline = np.full(256, 1e-9, dtype=np.float64)
    baseline[ord("a")] = 0.4
    baseline[ord("b")] = 0.6
    baseline /= baseline.sum()
    learner = SimpleNamespace(
        current_word=bytearray(),
        word_forms=[b"a", b"b"],
        predict_next_distribution=lambda: baseline,
    )

    distribution = phase145._residual_overlay_distribution(
        learner,
        {(4, 9): {0: 0.5, 1: -0.5}},
        ((9, 4),),
        support={(9, 4): 1.0},
        gain=1.0,
    )

    assert distribution[ord("a")] > baseline[ord("a")]
    assert distribution[ord("b")] < baseline[ord("b")]
    assert np.isclose(distribution.sum(), 1.0)


def test_residual_identity_scramble_preserves_receptor() -> None:
    routes = {
        (4, 9): {0: 1.0},
        (4, 10): {1: 1.0},
        (8, 11): {1: 1.0},
    }

    assert phase145._scramble_residual_items(routes, ((9, 4),)) == ((10, 4),)


def test_residual_route_can_reuse_identity_across_receptors() -> None:
    baseline = np.full(256, 1e-9, dtype=np.float64)
    baseline[ord("a")] = 0.4
    baseline[ord("b")] = 0.6
    baseline /= baseline.sum()
    learner = SimpleNamespace(
        current_word=bytearray(),
        word_forms=[b"a", b"b"],
        predict_next_distribution=lambda: baseline,
    )

    distribution = phase145._residual_overlay_distribution(
        learner,
        {(8, 9): {0: 0.5, 1: -0.5}},
        ((9, 4),),
        support={(9, 4): 1.0},
        gain=1.0,
        cross_receptor_fallback=True,
    )

    assert distribution[ord("a")] > baseline[ord("a")]
    assert distribution[ord("b")] < baseline[ord("b")]


def test_shadow_prediction_is_relation_local_and_does_not_mutate_baseline() -> None:
    baseline = np.full(256, 1e-9, dtype=np.float64)
    baseline[ord("a")] = 0.4
    baseline[ord("b")] = 0.6
    baseline /= baseline.sum()
    learner = SimpleNamespace(
        current_word=bytearray(),
        word_forms=[b"a", b"b"],
        predict_next_distribution=lambda: baseline,
    )
    relation = (3, 9, 4)
    arm = SimpleNamespace(
        learner=learner,
        pending_receptor_relational_population={relation: 0.5},
        relational_residual_routes={(4, 9): {0: 0.5, 1: -0.5}},
    )
    before = baseline.copy()

    predictions = phase146._shadow_predictions(arm, baseline, gain=1.0)

    assert tuple(predictions) == (relation,)
    off, on = predictions[relation]
    np.testing.assert_array_equal(off, before)
    np.testing.assert_array_equal(baseline, before)
    assert on[ord("a")] > off[ord("a")]
