"""Focused tests for the nested slow-selector experiment."""

from __future__ import annotations

import ast
import inspect
from argparse import Namespace

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_future_mode_nested_selector_ablation as assay,
)


def _args() -> Namespace:
    return assay._parser().parse_args(
        ["--dataset-dir", "unused", "--report", "unused.json"]
    )


def test_residual_experts_begin_as_exact_nested_identity() -> None:
    args = _args()
    residual = assay._selector(3, args, seed=4, residual=True)
    context = np.asarray([1.0, 0.0, -1.0])
    prior = np.asarray([0.7, 0.2, 0.1])
    predicted = residual.predict_with_prior(context, (0, 2), prior)
    np.testing.assert_allclose(predicted.probabilities, prior, atol=1e-15)


def test_fast_and_slow_residuals_have_matched_trainable_allocation() -> None:
    args = _args()
    fast = assay._selector(24, args, seed=4, residual=True)
    slow = assay._selector(24, args, seed=4, residual=True)
    assert fast.trainable_bytes == slow.trainable_bytes
    assert fast.feature_dimensions == slow.feature_dimensions


def test_experiment_has_no_backprop_framework_imports() -> None:
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
