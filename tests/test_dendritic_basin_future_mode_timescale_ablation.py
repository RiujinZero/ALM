"""Focused tests for the two-timescale future-mode selector assay."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from manifoldflow.dendritic_basin.categorical_future_mode_selector import (
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelectorConfig,
)
from manifoldflow.dendritic_basin.future_turn_trajectory import EncodedTurn
from manifoldflow.experiments import (
    dendritic_basin_future_mode_timescale_ablation as assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)


def _example(conversation: int, transition: int, token: tuple[float, float]):
    current = np.asarray(token, dtype=np.float64)
    return surprise_assay.TransitionExample(
        conversation_index=conversation,
        transition_index=transition,
        episode_id=str(conversation),
        context_state=np.concatenate((current, np.zeros(2))),
        basal_prediction=np.zeros(1),
        target=np.zeros(1),
        target_turn=EncodedTurn(np.zeros(1), source_id="text", role_id="ASSISTANT"),
    )


def test_slow_states_are_causal_and_reset_between_conversations() -> None:
    examples = (
        _example(0, 0, (1.0, 0.0)),
        _example(0, 1, (0.0, 2.0)),
        _example(1, 0, (3.0, 0.0)),
    )
    states = assay.training_slow_states(
        examples, conversations=2, token_dimensions=2, decay=0.5
    )
    np.testing.assert_allclose(states[0], [1.0, 0.0])
    np.testing.assert_allclose(states[1], [0.5, 2.0])
    np.testing.assert_allclose(states[2], [3.0, 0.0])


def test_slow_state_rejects_invalid_decay_and_missing_conversation() -> None:
    with pytest.raises(ValueError):
        assay.training_slow_states(
            (_example(0, 0, (1.0, 0.0)),),
            conversations=1,
            token_dimensions=2,
            decay=0.0,
        )
    with pytest.raises(ValueError):
        assay.training_slow_states(
            (_example(0, 0, (1.0, 0.0)),),
            conversations=2,
            token_dimensions=2,
            decay=0.9,
        )


def test_readout_allocation_is_matched_across_timescales() -> None:
    selectors = [
        CategoricalFutureModeSelector(
            CategoricalFutureModeSelectorConfig(
                context_dim=width,
                mode_capacity=8,
                random_features=12,
                include_direct_context=False,
            )
        )
        for width in (24, 48)
    ]
    assert selectors[0].trainable_bytes == selectors[1].trainable_bytes
    assert selectors[0].feature_dimensions == selectors[1].feature_dimensions


def test_module_has_no_backprop_framework_imports() -> None:
    source = inspect.getsource(assay)
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"torch", "tensorflow", "jax", "autograd"})
    assert ".backward(" not in source
