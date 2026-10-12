"""Focused tests for the local categorical future-mode selector."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from manifoldflow.dendritic_basin import categorical_future_mode_selector as module


def _selector(seed: int = 7) -> module.CategoricalFutureModeSelector:
    return module.CategoricalFutureModeSelector(
        module.CategoricalFutureModeSelectorConfig(
            context_dim=2,
            mode_capacity=4,
            random_features=8,
            learning_rate=0.2,
            seed=seed,
        )
    )


def test_prediction_is_causal_normalized_and_read_only() -> None:
    selector = _selector()
    before = selector.fingerprint()
    first = selector.predict(np.asarray([1.0, -1.0]), (3, 1))
    second = selector.predict(np.asarray([1.0, -1.0]), (3, 1))
    assert first.mode_ids == (3, 1)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert first.probabilities.sum() == pytest.approx(1.0)
    assert np.all(first.probabilities > 0.0)
    assert selector.fingerprint() == before


def test_soft_responsibility_updates_raise_target_probability() -> None:
    selector = _selector()
    context = np.asarray([0.5, 1.5])
    before = selector.predict(context, (0, 2)).probabilities
    for _ in range(20):
        score = selector.update(context, (0, 2), np.asarray([0.0, 0.0, 1.0]))
        assert score["log_loss_bits"] >= 0.0
        assert score["brier"] >= 0.0
    after = selector.predict(context, (0, 2)).probabilities
    assert after[2] > before[2]
    assert selector.updates == 20


def test_zero_residual_expert_exactly_preserves_frozen_prior_then_learns() -> None:
    selector = module.CategoricalFutureModeSelector(
        module.CategoricalFutureModeSelectorConfig(
            context_dim=2,
            mode_capacity=4,
            random_features=8,
            include_direct_context=False,
            initial_readout_scale=0.0,
            learning_rate=0.2,
        )
    )
    context = np.asarray([0.5, -0.5])
    prior = np.asarray([0.6, 0.3, 0.1])
    initial = selector.predict_with_prior(context, (0, 2), prior)
    np.testing.assert_allclose(initial.probabilities, prior, atol=1e-15)
    before = selector.fingerprint()
    selector.update_with_prior(
        context,
        (0, 2),
        prior,
        np.asarray([0.0, 0.0, 1.0]),
    )
    after = selector.predict_with_prior(context, (0, 2), prior)
    assert after.probabilities[2] > initial.probabilities[2]
    assert selector.fingerprint() != before


def test_only_available_classes_are_normalized() -> None:
    selector = _selector()
    context = np.asarray([-0.5, 0.25])
    for _ in range(5):
        selector.update(context, (0,), np.asarray([0.1, 0.9]))
    one = selector.predict(context, (0,))
    other = selector.predict(context, (1,))
    assert one.probabilities.shape == (2,)
    assert other.probabilities.shape == (2,)
    assert one.probabilities.sum() == pytest.approx(1.0)
    assert other.probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"context_dim": 0},
        {"mode_capacity": 0},
        {"random_features": 0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1.0},
        {"initial_readout_scale": -1.0},
        {"probability_floor": 1.0},
    ],
)
def test_invalid_configuration_is_rejected(changes: dict[str, float]) -> None:
    values = {"context_dim": 2, "mode_capacity": 4}
    values.update(changes)
    with pytest.raises(ValueError):
        module.CategoricalFutureModeSelectorConfig(**values)


def test_invalid_inputs_are_rejected() -> None:
    selector = _selector()
    with pytest.raises(ValueError):
        selector.predict(np.asarray([1.0]), (0,))
    with pytest.raises(ValueError):
        selector.predict(np.asarray([1.0, 2.0]), (4,))
    with pytest.raises(ValueError):
        selector.predict(np.asarray([1.0, 2.0]), (1, 1))
    with pytest.raises(ValueError):
        selector.update(np.asarray([1.0, 2.0]), (0,), np.asarray([0.0, 0.0]))


def test_deterministic_seed_and_fingerprint() -> None:
    left = _selector(11)
    right = _selector(11)
    context = np.asarray([0.25, -0.75])
    target = np.asarray([0.1, 0.7, 0.2])
    for _ in range(3):
        left.update(context, (0, 3), target)
        right.update(context, (0, 3), target)
    assert left.fingerprint() == right.fingerprint()
    assert left.allocated_bytes == right.allocated_bytes


def test_random_only_features_keep_trainable_allocation_across_input_widths() -> None:
    narrow = module.CategoricalFutureModeSelector(
        module.CategoricalFutureModeSelectorConfig(
            context_dim=2,
            mode_capacity=4,
            random_features=8,
            include_direct_context=False,
        )
    )
    wide = module.CategoricalFutureModeSelector(
        module.CategoricalFutureModeSelectorConfig(
            context_dim=7,
            mode_capacity=4,
            random_features=8,
            include_direct_context=False,
        )
    )
    assert narrow.feature_dimensions == wide.feature_dimensions == 9
    assert narrow.trainable_bytes == wide.trainable_bytes
    assert narrow.allocated_bytes < wide.allocated_bytes


def test_module_has_no_autograd_or_backprop_framework_imports() -> None:
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"torch", "tensorflow", "jax", "autograd"})
    assert ".backward(" not in source
