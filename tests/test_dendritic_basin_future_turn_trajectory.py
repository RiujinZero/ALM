from __future__ import annotations

import inspect

import numpy as np
import pytest

from manifoldflow.dendritic_basin.future_turn_trajectory import (
    EncodedTurn,
    FutureTurnTrajectory,
)


def _turn(
    values: tuple[float, ...] | list[float],
    *,
    source: str = "text",
    role: str = "agent",
) -> EncodedTurn:
    return EncodedTurn(np.asarray(values, dtype=np.float64), source, role)


def _cycle_model(*, widths: tuple[int, ...] = (4, 2, 1)) -> tuple[
    FutureTurnTrajectory, tuple[EncodedTurn, ...]
]:
    sequence = (
        _turn([1.0, 0.0, 0.0, 0.0]),
        _turn([0.0, 1.0, 0.0, 0.0]),
        _turn([0.0, 0.0, 1.0, 1.0]),
        _turn([0.0, 0.0, 0.0, 1.0]),
    )
    model = FutureTurnTrajectory(
        4,
        source_ids=("text",),
        role_ids=("agent",),
        horizon_widths=widths,
        memory_decay=0.8,
        ridge=0.0,
    ).fit([sequence] * 8)
    return model, sequence


def test_rollout_reuses_exactly_one_transition_at_every_horizon() -> None:
    model, sequence = _cycle_model()
    assert model.training_objective == "observed_one_step_ridge"
    assert model.trains_through_rollout is False
    assert model.bottlenecks_transition_dynamics is False
    model.observe(sequence[0])
    start = model.snapshot().state
    transition = model.transition_matrix

    slots = model.diagnostic_raw_rollout()

    expected = start.copy()
    for slot in slots:
        expected = expected @ transition
        np.testing.assert_allclose(slot.latent_state, expected, atol=1e-12)
    # There are no horizon-indexed learned transition parameters hidden in the API.
    assert model.transition_matrix.shape == (model.state_dim, model.state_dim)


def test_far_slots_use_nested_lower_rank_bottlenecks_not_gain_scaling() -> None:
    model, sequence = _cycle_model(widths=(4, 2, 1))

    projectors = [model.projection_matrix(horizon) for horizon in (1, 2, 3)]
    assert [np.linalg.matrix_rank(projector) for projector in projectors] == [4, 2, 1]
    np.testing.assert_allclose(projectors[0] @ projectors[1], projectors[1])
    np.testing.assert_allclose(projectors[1] @ projectors[2], projectors[2])
    assert [model.projection_mask(h).sum() for h in (1, 2, 3)] == [4, 2, 1]

    # Reusing one actual vector at every oracle horizon isolates the rate schedule
    # from forecast quality.  The full-rate slot is exact; the lower-rate slot is
    # a projection and cannot be recovered by merely scaling its amplitude back.
    oracle = model.oracle_tape((sequence[2], sequence[2], sequence[2]))
    np.testing.assert_allclose(oracle.slots[0].vector, sequence[2].vector, atol=1e-12)
    assert not np.allclose(oracle.slots[2].vector, sequence[2].vector)
    assert oracle.slots[2].discarded_training_variance >= 0.0
    assert (
        oracle.slots[2].discarded_training_variance
        >= oracle.slots[1].discarded_training_variance
        >= oracle.slots[0].discarded_training_variance
    )


def test_arrival_correction_rerolls_and_refines_an_approaching_slot() -> None:
    model, sequence = _cycle_model(widths=(4, 1, 1))
    model.observe(sequence[0])
    before = model.predicted_tape()
    before_raw = model.diagnostic_raw_rollout()
    formerly_distant = before.slots[1]
    distant_error = np.linalg.norm(formerly_distant.vector - sequence[2].vector)

    after = model.observe_and_rollout(sequence[1])
    after_raw = model.diagnostic_raw_rollout()
    now_immediate = after.slots[0]
    immediate_error = np.linalg.norm(now_immediate.vector - sequence[2].vector)

    assert formerly_distant.information_width == 1
    assert now_immediate.information_width == 4
    np.testing.assert_allclose(
        before_raw[1].raw_vector, sequence[2].vector, atol=1e-12
    )
    np.testing.assert_allclose(
        after_raw[0].raw_vector, sequence[2].vector, atol=1e-12
    )
    assert immediate_error < 1e-12
    assert distant_error > 0.1
    assert after.observed_count == 2


def test_explicit_role_binding_changes_the_forecast_for_the_same_event() -> None:
    shared = [1.0, 0.0]
    alpha_sequence = (
        _turn(shared, role="alpha"),
        _turn([0.0, 1.0], role="output"),
    )
    beta_sequence = (
        _turn(shared, role="beta"),
        _turn([0.0, -1.0], role="output"),
    )
    model = FutureTurnTrajectory(
        2,
        source_ids=("text",),
        role_ids=("alpha", "beta", "output"),
        horizon_widths=(2,),
        ridge=0.0,
    ).fit([alpha_sequence, beta_sequence] * 8)

    model.observe(alpha_sequence[0])
    alpha = model.predicted_tape().slots[0]
    model.reset()
    model.observe(beta_sequence[0])
    beta = model.predicted_tape().slots[0]

    np.testing.assert_allclose(alpha.vector, alpha_sequence[1].vector, atol=1e-12)
    np.testing.assert_allclose(beta.vector, beta_sequence[1].vector, atol=1e-12)
    assert alpha.role_id == beta.role_id == "output"
    assert not np.allclose(alpha.vector, beta.vector)


def test_persistent_history_prevents_current_turn_aliasing() -> None:
    common = _turn([0.0, 0.0])
    left = (_turn([1.0, 0.0]), common, _turn([2.0, 0.0]))
    right = (_turn([0.0, 1.0]), common, _turn([0.0, 2.0]))
    model = FutureTurnTrajectory(
        2,
        source_ids=("text",),
        role_ids=("agent",),
        horizon_widths=(2,),
        memory_decay=0.8,
        ridge=0.0,
    ).fit([left, right] * 8)

    states: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    for sequence in (left, right):
        model.reset()
        model.observe(sequence[0])
        model.observe(sequence[1])
        states.append(model.snapshot().state)
        forecasts.append(model.predicted_tape().slots[0].vector)

    # The current-token halves are identical; only persistent history separates
    # the states and allows the shared operator to choose the correct successor.
    np.testing.assert_array_equal(
        states[0][: model.token_dim], states[1][: model.token_dim]
    )
    assert not np.array_equal(states[0], states[1])
    np.testing.assert_allclose(forecasts[0], left[2].vector, atol=1e-12)
    np.testing.assert_allclose(forecasts[1], right[2].vector, atol=1e-12)


def test_predicted_surface_is_strictly_causal_and_oracle_is_explicit() -> None:
    model, sequence = _cycle_model()
    assert model.observed_count == 0  # fit does not leave a training suffix behind.
    model.observe(sequence[0])
    before_fingerprint = model.fingerprint()
    predicted_before = model.predicted_tape()

    alternative = _turn([-9.0, 4.0, 7.0, 2.0])
    oracle = model.oracle_tape((alternative, sequence[2], sequence[3]))
    predicted_after = model.predicted_tape()

    assert "future" not in inspect.signature(model.predicted_tape).parameters
    assert predicted_before.causal and predicted_before.mode == "predicted"
    assert not oracle.causal and oracle.mode == "oracle"
    assert model.fingerprint() == before_fingerprint
    for first, second in zip(
        predicted_before.slots, predicted_after.slots, strict=True
    ):
        np.testing.assert_array_equal(first.vector, second.vector)
    assert not np.allclose(
        oracle.slots[0].vector, predicted_before.slots[0].vector
    )


def test_snapshots_fingerprints_and_uncertainty_diagnostics_are_deterministic() -> None:
    first, sequence = _cycle_model()
    second, _ = _cycle_model()
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint(include_state=False) == second.fingerprint(
        include_state=False
    )

    for turn in sequence[:2]:
        first.observe(turn)
        second.observe(turn)
    left = first.snapshot()
    right = second.snapshot()

    assert left.fingerprint == right.fingerprint
    np.testing.assert_array_equal(left.state, right.state)
    assert not left.state.flags.writeable
    variances = first.residual_variance_by_source
    assert set(variances) == {"text"}
    assert variances["text"].shape == (4,)
    assert np.all(variances["text"] >= 0.0)
    accumulated = [
        slot.accumulated_residual_variance_heuristic
        for slot in first.predicted_tape().slots
    ]
    assert all(value is not None for value in accumulated)
    assert np.all(accumulated[1] >= accumulated[0])


def test_ablation_controls_are_explicit_deterministic_and_non_mutating() -> None:
    model, sequence = _cycle_model()
    learned_fingerprint = model.fingerprint(include_state=False)
    model.observe(sequence[0])
    fingerprint = model.fingerprint()
    intact = model.predicted_tape()

    shuffled = model.shuffled_transition(seed=17)
    np.testing.assert_array_equal(shuffled, model.shuffled_transition(seed=17))
    assert not np.array_equal(shuffled, model.transition_matrix)
    shuffled_tape = model.predicted_tape(transition_matrix=shuffled)
    reset_tape = model.predicted_tape(reset_state=True)
    permuted = model.predicted_tape(horizon_permutation=(3, 2, 1))

    assert model.fingerprint() == fingerprint
    assert model.fingerprint(include_state=False) == learned_fingerprint
    assert not np.allclose(
        shuffled_tape.slots[0].vector, intact.slots[0].vector
    )
    assert reset_tape.observed_count == 0
    # This first slot is full-rank and therefore reconstructs raw zero.  Lower
    # rank slots use an affine PCA projection and reset_state must not be called
    # a disconnected/no-code control merely because its recurrent input is zero.
    np.testing.assert_allclose(reset_tape.slots[0].vector, 0.0, atol=1e-12)
    assert permuted.slots[0].source_horizon == 3
    np.testing.assert_allclose(
        model.diagnostic_raw_rollout(horizon_permutation=(3, 2, 1))[0].raw_vector,
        model.diagnostic_raw_rollout()[2].raw_vector,
        atol=1e-12,
    )


def test_rollout_enforces_exact_history_update_off_the_training_manifold() -> None:
    model, sequence = _cycle_model()
    model.observe(sequence[0])
    previous = model.snapshot().state

    for slot in model.diagnostic_raw_rollout(
        transition_matrix=model.shuffled_transition(seed=29)
    ):
        state = slot.latent_state
        np.testing.assert_allclose(
            state[model.token_dim :],
            model.memory_decay * previous[model.token_dim :]
            + state[: model.token_dim],
            atol=1e-12,
        )
        previous = state


def test_production_slots_do_not_expose_unbottlenecked_forecasts() -> None:
    model, sequence = _cycle_model(widths=(4, 2, 1))
    model.observe(sequence[0])
    slot = model.predicted_tape().slots[-1]

    assert not hasattr(slot, "raw_vector")
    assert not hasattr(slot, "latent_state")
    diagnostic = model.diagnostic_raw_rollout()[-1]
    assert diagnostic.raw_vector.shape == slot.vector.shape
    assert diagnostic.latent_state.shape == (model.state_dim,)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"event_dim": 0}, ValueError),
        ({"source_ids": ()}, ValueError),
        ({"source_ids": ("text", "text")}, ValueError),
        ({"role_ids": ()}, ValueError),
        ({"horizon_widths": ()}, ValueError),
        ({"horizon_widths": (2, 3)}, ValueError),
        ({"horizon_widths": (0,)}, ValueError),
        ({"memory_decay": 0.0}, ValueError),
        ({"ridge": -1.0}, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    defaults: dict[str, object] = {
        "event_dim": 3,
        "source_ids": ("text",),
        "role_ids": ("agent",),
        "horizon_widths": (3, 1),
    }
    defaults.update(kwargs)
    with pytest.raises(error):
        FutureTurnTrajectory(**defaults)


def test_rejects_invalid_events_training_and_controls() -> None:
    with pytest.raises(ValueError, match="finite"):
        _turn([1.0, np.nan])

    model = FutureTurnTrajectory(
        2,
        source_ids=("text",),
        role_ids=("agent",),
        horizon_widths=(2, 1),
    )
    with pytest.raises(RuntimeError, match="fit"):
        model.predicted_tape()
    with pytest.raises(ValueError, match="adjacent"):
        model.fit([(_turn([1.0, 0.0]),)])

    training = ((_turn([1.0, 0.0]), _turn([0.0, 1.0])),)
    model.fit(training)
    with pytest.raises(ValueError, match="event_dim"):
        model.observe(_turn([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="unknown source"):
        model.observe(_turn([1.0, 0.0], source="audio"))
    with pytest.raises(ValueError, match="transition_matrix"):
        model.predicted_tape(transition_matrix=np.eye(2))
    with pytest.raises(ValueError, match="analytical history"):
        model.predicted_tape(transition_matrix=np.eye(model.state_dim))
    with pytest.raises(ValueError, match="permute"):
        model.predicted_tape(horizon_permutation=(1, 1))
    with pytest.raises(ValueError, match="exactly"):
        model.oracle_tape((_turn([1.0, 0.0]),))
