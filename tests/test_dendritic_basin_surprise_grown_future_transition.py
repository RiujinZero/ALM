"""Focused tests for the local future-vector assembly lifecycle."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
    SurpriseGrownFutureTransitionConfig,
)


def _config(**changes: object) -> SurpriseGrownFutureTransitionConfig:
    base = SurpriseGrownFutureTransitionConfig(
        context_dim=2,
        target_dim=2,
        max_context_assemblies=4,
        max_residual_modes=8,
        max_modes_per_context=3,
        max_nursery_candidates=8,
        min_context_affinity=0.70,
        residual_floor=0.10,
        surprise_birth_z=0.50,
        poor_representation_z=1.00,
        nursery_context_z=0.50,
        nursery_residual_z=0.50,
        nursery_confirmations=2,
        nursery_target_consistency=0.90,
        consolidation_min_age=8,
        consolidation_min_fires=4,
        dormancy_staleness=16,
        dormant_min_residence=2,
        juvenile_min_recycle_age=8,
    )
    return replace(base, **changes)


def _train_residual_mode(
    learner: SurpriseGrownFutureTransition,
    context: np.ndarray,
    residual: np.ndarray,
) -> int:
    basal = np.zeros(learner.config.target_dim)
    born = None
    for _ in range(learner.config.nursery_confirmations):
        learner.forecast(context, basal)
        update = learner.resolve(residual)
        born = update.born_mode if update.born_mode is not None else born
    assert born is not None
    return born


def _assert_mode_states_equal(
    left: tuple[object, ...], right: tuple[object, ...]
) -> None:
    assert len(left) == len(right)
    for actual, expected in zip(left, right, strict=True):
        assert actual.mode_id == expected.mode_id
        assert actual.context_id == expected.context_id
        np.testing.assert_array_equal(actual.residual_mean, expected.residual_mean)
        np.testing.assert_array_equal(
            actual.residual_variance, expected.residual_variance
        )
        assert actual.observations == expected.observations
        assert actual.fires == expected.fires
        assert actual.causal_scores == expected.causal_scores
        assert actual.predictive_gain_bits == expected.predictive_gain_bits
        assert actual.lifecycle == expected.lifecycle
        assert actual.born_step == expected.born_step
        assert actual.last_fire_step == expected.last_fire_step


def test_later_target_is_scored_before_recurrent_candidate_is_born() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    context = np.array([0.25, -0.50])
    basal = np.zeros(2)
    target = np.array([1.0, -1.0])

    first_forecast = learner.forecast(context, basal)
    first = learner.resolve(target)

    np.testing.assert_array_equal(first_forecast.prediction, basal)
    assert first.scored_mode_ids == ()
    assert first.standardized_surprise > learner.config.surprise_birth_z
    assert first.poor_representation
    assert first.born_mode is None
    assert learner.nursery_count == 1

    # Recurrence promotes only after this second pending forecast was scored.
    learner.forecast(context, basal)
    second = learner.resolve(target)
    assert second.scored_mode_ids == ()
    assert second.born_mode == 0
    assert learner.nursery_count == 0
    newborn = learner.mode_states[0]
    assert newborn.observations == 0.0
    assert newborn.fires == 0
    assert newborn.causal_scores == 0
    assert newborn.predictive_gain_bits == 0.0
    assert newborn.lifecycle == "juvenile"

    learned = learner.forecast(context, basal)
    assert 0.0 < learned.basal_weight < 1.0
    assert np.linalg.norm(learned.prediction) < np.linalg.norm(target)
    assert len(learned.candidates) == 1
    np.testing.assert_allclose(learned.candidates[0].prediction, target)
    third = learner.resolve(target)
    assert third.scored_mode_ids == (0,)
    assert third.responsibilities[0] > 0.99
    assert third.mixture_gain_bits_per_coordinate > 0.0
    assert third.mode_marginal_gain_bits[0] > 0.0
    assert learner.mode_states[0].predictive_gain_bits > 0.0
    assert learner.context_states[0].predictive_gain_bits > 0.0
    assert learner.mixture_predictive_gain_bits_per_coordinate > 0.0

    # The externally supplied basal control is input data, not fitted state.
    np.testing.assert_array_equal(basal, np.zeros(2))


def test_birth_needs_both_standardized_surprise_and_poor_representation() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    context = np.array([0.0, 0.0])
    basal = np.array([0.5, -0.5])

    learner.forecast(context, basal)
    unsurprising = learner.resolve(basal)
    assert unsurprising.standardized_surprise == 0.0
    assert unsurprising.poor_representation
    assert learner.nursery_count == 0

    target = np.array([1.5, -1.5])
    mode_id = _train_residual_mode(learner, context, target - basal)
    assert mode_id == 0

    learner.forecast(context, basal)
    represented = learner.resolve(target)
    assert represented.standardized_surprise == pytest.approx(0.0)
    assert represented.best_representation_z == pytest.approx(0.0)
    assert not represented.poor_representation
    assert represented.born_mode is None
    assert learner.nursery_count == 0


def test_one_context_keeps_distinct_residual_modes_without_averaging_them() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    context = np.array([0.2, 0.1])
    positive = np.array([1.0, 0.0])
    negative = np.array([-1.0, 0.0])

    first_id = _train_residual_mode(learner, context, positive)
    first_mean_before = learner.mode_states[0].residual_mean.copy()
    second_id = _train_residual_mode(learner, context, negative)

    assert (first_id, second_id) == (0, 1)
    states = {state.mode_id: state for state in learner.mode_states}
    assert states[first_id].context_id == states[second_id].context_id
    np.testing.assert_allclose(states[first_id].residual_mean, first_mean_before)
    np.testing.assert_allclose(states[first_id].residual_mean, positive)
    np.testing.assert_allclose(states[second_id].residual_mean, negative)
    assert learner.modes_for_context(states[first_id].context_id) == (0, 1)

    forecast = learner.forecast(context, np.zeros(2))
    candidate_predictions = [item.prediction for item in forecast.candidates]
    assert len(candidate_predictions) == 2
    assert any(np.allclose(item, positive) for item in candidate_predictions)
    assert any(np.allclose(item, negative) for item in candidate_predictions)

    observations_before = {
        state.mode_id: state.observations for state in learner.mode_states
    }
    update = learner.resolve(positive)
    responsibilities = dict(
        zip(update.scored_mode_ids, update.responsibilities, strict=True)
    )
    assert responsibilities[first_id] > 0.99
    assert responsibilities[second_id] < 1e-6
    observations_after = {
        state.mode_id: state.observations for state in learner.mode_states
    }
    assert observations_after[first_id] > observations_before[first_id]
    assert observations_after[second_id] == pytest.approx(
        observations_before[second_id]
    )


def test_promoting_one_of_several_nursery_candidates_preserves_the_others() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    first_context = np.array([0.0, 0.0])
    second_context = np.array([4.0, 4.0])

    learner.forecast(first_context, np.zeros(2))
    learner.resolve(np.array([1.0, 0.0]))
    learner.forecast(second_context, np.zeros(2))
    learner.resolve(np.array([0.0, 1.0]))
    assert learner.nursery_count == 2

    learner.forecast(second_context, np.zeros(2))
    update = learner.resolve(np.array([0.0, 1.0]))

    assert update.born_mode == 0
    assert learner.nursery_count == 1


def test_forecast_is_read_only_and_can_be_used_without_pending_training() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    context = np.array([0.0, 0.0])
    _train_residual_mode(learner, context, np.array([1.0, -1.0]))
    states_before = learner.mode_states
    counters_before = (
        learner.step,
        learner.births,
        learner.consolidations,
        learner.dormancies,
        learner.recycles,
        learner.nursery_count,
    )

    for _ in range(5):
        forecast = learner.forecast(context, np.zeros(2), remember=False)
        assert np.all(np.isfinite(forecast.prediction))
        assert 0.0 < forecast.basal_weight < 1.0
        np.testing.assert_allclose(forecast.candidates[0].prediction, [1.0, -1.0])

    assert not learner.pending
    _assert_mode_states_equal(learner.mode_states, states_before)
    assert (
        learner.step,
        learner.births,
        learner.consolidations,
        learner.dormancies,
        learner.recycles,
        learner.nursery_count,
    ) == counters_before

    learner.forecast(context, np.zeros(2))
    with pytest.raises(RuntimeError, match="resolve the pending"):
        learner.forecast(context, np.zeros(2))
    states_before_evaluation = learner.mode_states
    learner.resolve(np.array([1.0, -1.0]), learn=False)
    _assert_mode_states_equal(learner.mode_states, states_before_evaluation)
    assert learner.step == counters_before[0]


def test_signed_gain_consolidates_then_staleness_dormancy_allows_recycling() -> None:
    learner = SurpriseGrownFutureTransition(
        _config(
            max_context_assemblies=1,
            max_residual_modes=1,
            max_modes_per_context=1,
            nursery_confirmations=1,
            utility_decay=0.0,
            consolidation_min_age=1,
            consolidation_min_fires=1,
            consolidation_gain_bits=0.0,
            dormancy_staleness=1,
            dormant_min_residence=2,
            juvenile_min_recycle_age=1,
        )
    )
    first_context = np.array([0.0, 0.0])
    target = np.array([1.0, 0.0])

    learner.forecast(first_context, np.zeros(2))
    birth = learner.resolve(target)
    assert birth.born_mode == 0
    newborn = learner.mode_states[0]
    assert newborn.lifecycle == "juvenile"
    assert newborn.causal_scores == 0
    assert learner.consolidations == 0

    learner.forecast(first_context, np.zeros(2))
    learner.resolve(target)
    state = learner.mode_states[0]
    assert state.predictive_gain_bits > 0.0
    assert state.lifecycle == "consolidated"
    assert state.last_fire_step == learner.step
    assert learner.consolidations == 1

    # A far context does not make this mode responsible.  The signed gain is
    # retained, but staleness still moves the unused assembly to dormancy.
    second_context = np.array([10.0, 10.0])
    learner.forecast(second_context, np.zeros(2))
    learner.resolve(np.zeros(2))
    assert learner.mode_states[0].lifecycle == "dormant"
    assert learner.dormancies == 1

    # The first attempt cannot recycle during the minimum dormant residence.
    learner.forecast(second_context, np.zeros(2))
    first_attempt = learner.resolve(np.array([2.0, 0.0]))
    assert first_attempt.born_mode is None
    learner.forecast(second_context, np.zeros(2))
    recycled = learner.resolve(np.array([2.0, 0.0]))

    assert recycled.born_mode == 0
    assert recycled.recycled_modes == (0,)
    assert learner.recycles == 1
    assert learner.context_count == 1
    state = learner.mode_states[0]
    np.testing.assert_allclose(state.residual_mean, [2.0, 0.0])
    assert state.lifecycle == "juvenile"


def test_negative_signed_gain_over_a_better_basal_control_causes_dormancy() -> None:
    learner = SurpriseGrownFutureTransition(
        _config(
            nursery_confirmations=1,
            utility_decay=0.0,
            consolidation_min_age=1,
            consolidation_min_fires=1,
            consolidation_gain_bits=0.0,
            prune_gain_bits=-0.02,
            dormancy_staleness=100,
        )
    )
    context = np.zeros(2)
    learned_residual = np.array([0.10, 0.0])

    learner.forecast(context, np.zeros(2))
    learner.resolve(learned_residual)
    learner.forecast(context, np.zeros(2))
    learner.resolve(learned_residual)
    assert learner.mode_states[0].lifecycle == "consolidated"
    assert learner.mode_states[0].predictive_gain_bits > 0.0

    # This nearby target still assigns responsibility to the mode, while the
    # unchanged basal prediction is now closer than the mode's forecast.
    learner.forecast(context, np.zeros(2))
    update = learner.resolve(np.array([0.02, 0.0]))

    assert update.responsibilities[0] > 0.1
    assert update.mixture_gain_bits_per_coordinate < 0.0
    assert update.mode_marginal_gain_bits[0] < 0.0
    assert learner.mixture_predictive_gain_bits_per_coordinate < 0.0
    assert learner.mode_states[0].predictive_gain_bits < 0.0
    assert learner.context_states[0].predictive_gain_bits < 0.0
    assert learner.mode_states[0].lifecycle == "dormant"


def test_basal_only_pending_predictor_has_exactly_zero_proper_score_gain() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    learner.forecast(np.zeros(2), np.array([0.25, -0.25]))
    update = learner.resolve(np.array([1.0, -1.0]))

    assert update.mixture_nll_bits_per_coordinate == pytest.approx(
        update.basal_nll_bits_per_coordinate
    )
    assert update.mixture_gain_bits_per_coordinate == pytest.approx(0.0)
    assert learner.mixture_predictive_gain_bits_per_coordinate == pytest.approx(0.0)


def test_warmed_global_variance_cannot_hide_recurring_conditional_surprise() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    basal = np.zeros(2)

    # Large, deliberately non-recurring residuals warm the scoring variance
    # without providing the recurrence needed for structural promotion.
    for index in range(20):
        context = np.array([10.0 * index, -10.0 * index])
        target = np.array([100.0 + index, -100.0 - 2.0 * index])
        learner.forecast(context, basal)
        assert learner.resolve(target).born_mode is None
    assert float(learner._residual_second_moment.mean()) > 1000.0
    assert len(learner.mode_states) == 0

    recurring_context = np.array([1000.0, -1000.0])
    recurring_target = np.array([1.0, 0.0])
    learner.forecast(recurring_context, basal)
    one_off = learner.resolve(recurring_target)
    assert one_off.standardized_surprise > learner.config.surprise_birth_z
    assert one_off.born_mode is None

    learner.forecast(recurring_context, basal)
    recurrent = learner.resolve(recurring_target)
    assert recurrent.standardized_surprise > learner.config.surprise_birth_z
    assert recurrent.born_mode == 0


def test_zero_dormant_weight_falls_back_to_finite_basal_prediction() -> None:
    learner = SurpriseGrownFutureTransition(_config(dormant_forecast_weight=0.0))
    context = np.zeros(2)
    _train_residual_mode(learner, context, np.array([1.0, -1.0]))
    mode = learner._modes[0]
    assert mode is not None
    mode.lifecycle = "dormant"

    basal = np.array([0.25, -0.50])
    forecast = learner.forecast(context, basal, remember=False)

    assert forecast.basal_weight == 1.0
    assert forecast.candidates == ()
    np.testing.assert_array_equal(forecast.prediction, basal)
    assert np.all(np.isfinite(forecast.prediction))


def test_gaussian_responsibilities_use_preupdate_component_variance() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    context = np.zeros(2)
    first_id = _train_residual_mode(learner, context, np.array([1.0, 0.0]))
    second_id = _train_residual_mode(learner, context, np.array([-1.0, 0.0]))
    target = np.array([1.0, 0.0])
    narrow = learner._modes[first_id]
    broad = learner._modes[second_id]
    assert narrow is not None and broad is not None
    narrow.residual_mean[:] = target
    broad.residual_mean[:] = target
    narrow.residual_variance[:] = 0.01
    broad.residual_variance[:] = 1.00
    narrow.observations = broad.observations = 0.0

    learner.forecast(context, np.zeros(2))
    update = learner.resolve(target, learn=False)
    responsibility = dict(
        zip(update.scored_mode_ids, update.responsibilities, strict=True)
    )

    assert responsibility[first_id] > 0.95
    assert responsibility[broad.mode_id] < 0.05
    assert sum(update.responsibilities) + update.unclaimed_responsibility == (
        pytest.approx(1.0)
    )
    assert np.isfinite(update.mixture_nll_bits_per_coordinate)


def test_reported_proper_score_matches_the_pending_gaussian_mixture() -> None:
    learner = SurpriseGrownFutureTransition(_config(nursery_confirmations=1))
    context = np.zeros(2)
    learner.forecast(context, np.zeros(2))
    learner.resolve(np.array([0.50, 0.0]))

    target = np.array([0.45, 0.0])
    learner.forecast(context, np.zeros(2))
    pending = learner._pending
    assert pending is not None and len(pending.mode_ids) == 1
    weights = np.array([pending.basal_weight, pending.mode_weights[0]])
    means = np.stack([pending.basal_prediction, pending.mode_means[0]])
    variances = np.stack([pending.basal_variance, pending.mode_variances[0]])
    densities = np.exp(
        -0.5
        * np.sum(
            np.log(2.0 * math.pi * variances)
            + np.square(target[None, :] - means) / variances,
            axis=1,
        )
    )
    mixture_density = float(np.dot(weights, densities))
    expected_mixture_nll = -math.log2(mixture_density) / 2.0
    expected_basal_nll = -math.log2(float(densities[0])) / 2.0
    expected_mode_responsibility = float(weights[1] * densities[1]) / mixture_density

    update = learner.resolve(target)

    assert update.mixture_nll_bits_per_coordinate == pytest.approx(expected_mixture_nll)
    assert update.basal_nll_bits_per_coordinate == pytest.approx(expected_basal_nll)
    assert update.responsibilities[0] == pytest.approx(expected_mode_responsibility)
    assert update.mode_marginal_gain_bits[0] == pytest.approx(
        expected_basal_nll - expected_mixture_nll
    )
    assert update.context_marginal_gain_bits_per_coordinate == (
        (0, pytest.approx(expected_basal_nll - expected_mixture_nll)),
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_every_nonfinite_config_value_is_rejected(bad_value: float) -> None:
    config = _config()
    for name in config.__dataclass_fields__:
        with pytest.raises(ValueError, match="finite"):
            replace(config, **{name: bad_value})


def test_twin_runs_are_deterministic_and_consolidated_capacity_is_protected() -> None:
    config = _config(
        max_context_assemblies=1,
        max_residual_modes=1,
        max_modes_per_context=1,
        nursery_confirmations=1,
        utility_decay=0.0,
        consolidation_min_age=1,
        consolidation_min_fires=1,
        consolidation_gain_bits=0.0,
        dormancy_staleness=100,
    )
    left = SurpriseGrownFutureTransition(config)
    right = SurpriseGrownFutureTransition(config)
    sequence = [
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([10.0, 10.0]), np.array([2.0, 0.0])),
        (np.array([10.0, 10.0]), np.array([2.0, 0.0])),
    ]
    for context, target in sequence:
        left_forecast = left.forecast(context, np.zeros(2))
        right_forecast = right.forecast(context, np.zeros(2))
        np.testing.assert_array_equal(
            left_forecast.prediction, right_forecast.prediction
        )
        left_update = left.resolve(target)
        right_update = right.resolve(target)
        assert left_update.born_mode == right_update.born_mode
        assert left_update.mixture_gain_bits_per_coordinate == pytest.approx(
            right_update.mixture_gain_bits_per_coordinate, abs=0.0
        )

    _assert_mode_states_equal(left.mode_states, right.mode_states)
    assert left.births == right.births == 1
    assert left.recycles == right.recycles == 0
    assert len(left.mode_states) == 1
    assert left.mode_states[0].lifecycle == "consolidated"
    np.testing.assert_allclose(left.mode_states[0].residual_mean, [1.0, 0.0])


def test_invalid_shapes_and_noncausal_resolution_are_rejected() -> None:
    learner = SurpriseGrownFutureTransition(_config())
    with pytest.raises(RuntimeError, match="must precede"):
        learner.resolve(np.zeros(2))
    with pytest.raises(ValueError, match="context must have shape"):
        learner.forecast(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        learner.forecast(np.array([0.0, np.nan]), np.zeros(2))
