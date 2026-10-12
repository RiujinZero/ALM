"""Tests for the persistent, context-keyed core estimator."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.future_vector_persistent_core import (
    PersistentCoreEstimator,
    PersistentCoreEstimatorConfig,
)


def _estimator(**overrides) -> PersistentCoreEstimator:
    fields = {"context_dim": 2, "core_dim": 2, "max_slots": 4, **overrides}
    return PersistentCoreEstimator(PersistentCoreEstimatorConfig(**fields))


def test_predict_before_any_update_returns_the_zero_global_core() -> None:
    estimator = _estimator()
    prediction = estimator.predict(np.asarray([1.0, 1.0]))
    assert prediction == pytest.approx(np.zeros(2))


def test_update_creates_a_slot_and_predict_recovers_it_for_a_near_context() -> None:
    estimator = _estimator(min_context_affinity=0.4)
    context = np.asarray([0.0, 0.0])
    core = np.asarray([3.0, -1.0])
    for _ in range(20):
        estimator.update(context, core)
    prediction = estimator.predict(context)
    assert prediction == pytest.approx(core, abs=0.05)
    assert estimator.slot_count == 1
    assert estimator.births == 1


def test_far_context_falls_back_to_the_global_blend_not_one_specific_slot() -> None:
    # The global core is updated on every call regardless of slot, so with
    # two well-separated context regions its fallback value is a blend of
    # both -- distinct from either individual slot's own persistent core.
    estimator = _estimator(min_context_affinity=0.9, context_bandwidth=0.2)
    for _ in range(30):
        estimator.update(np.asarray([0.0, 0.0]), np.asarray([5.0, 5.0]))
    for _ in range(30):
        estimator.update(np.asarray([50.0, 50.0]), np.asarray([-5.0, -5.0]))
    assert estimator.slot_count == 2
    far_context = np.asarray([1000.0, 1000.0])
    prediction = estimator.predict(far_context)
    assert not np.allclose(prediction, np.asarray([5.0, 5.0]), atol=0.5)
    assert not np.allclose(prediction, np.asarray([-5.0, -5.0]), atol=0.5)


def test_two_well_separated_contexts_get_distinct_slots() -> None:
    estimator = _estimator(min_context_affinity=0.5, context_bandwidth=0.5)
    for _ in range(15):
        estimator.update(np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]))
    for _ in range(15):
        estimator.update(np.asarray([10.0, 10.0]), np.asarray([0.0, 1.0]))
    assert estimator.slot_count == 2
    near_first = estimator.predict(np.asarray([0.1, 0.1]))
    near_second = estimator.predict(np.asarray([10.1, 10.1]))
    assert near_first == pytest.approx(np.asarray([1.0, 0.0]), abs=0.1)
    assert near_second == pytest.approx(np.asarray([0.0, 1.0]), abs=0.1)


def test_slot_capacity_recycles_the_least_observed_slot() -> None:
    estimator = _estimator(max_slots=2, min_context_affinity=0.99, context_bandwidth=0.1)
    estimator.update(np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]))
    estimator.update(np.asarray([5.0, 5.0]), np.asarray([0.0, 1.0]))
    assert estimator.slot_count == 2
    estimator.update(np.asarray([50.0, 50.0]), np.asarray([9.0, 9.0]))
    assert estimator.slot_count == 2
    assert estimator.recycles == 1


def test_update_rejects_wrong_shaped_inputs() -> None:
    estimator = _estimator()
    with pytest.raises(ValueError):
        estimator.update(np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 1.0]))
    with pytest.raises(ValueError):
        estimator.update(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0, 1.0]))


def test_config_rejects_nonpositive_and_out_of_range_fields() -> None:
    with pytest.raises(ValueError):
        PersistentCoreEstimatorConfig(context_dim=0, core_dim=2)
    with pytest.raises(ValueError):
        PersistentCoreEstimatorConfig(context_dim=2, core_dim=2, min_context_affinity=1.5)
