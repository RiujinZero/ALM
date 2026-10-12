"""Focused tests for the categorical future-mode selector assay."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
    SurpriseGrownFutureTransitionConfig,
)
from manifoldflow.experiments import (
    dendritic_basin_future_mode_selector_ablation as assay,
)


def _learner() -> SurpriseGrownFutureTransition:
    learner = SurpriseGrownFutureTransition(
        SurpriseGrownFutureTransitionConfig(
            context_dim=1,
            target_dim=1,
            max_context_assemblies=2,
            max_residual_modes=2,
            max_modes_per_context=2,
            max_nursery_candidates=2,
            residual_floor=0.1,
            surprise_birth_z=0.1,
            poor_representation_z=0.1,
            nursery_confirmations=1,
            nursery_target_consistency=0.0,
            consolidation_min_age=100,
            consolidation_min_fires=100,
        )
    )
    learner.forecast(np.asarray([0.0]), np.asarray([0.0]))
    learner.resolve(np.asarray([1.0]))
    return learner


def test_responsibility_surface_scores_after_forecast_without_learning() -> None:
    learner = _learner()
    states_before = learner.mode_states
    surface = assay.responsibility_surface(
        learner,
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([1.0]),
    )
    assert surface.target_probabilities.sum() == pytest.approx(1.0)
    assert surface.affinity_probabilities.sum() == pytest.approx(1.0)
    assert len(surface.means) == len(surface.mode_ids) + 1
    assert learner.mode_states == states_before
    assert not learner.pending


def test_shortlist_is_stable_smallest_mass_set() -> None:
    probabilities = np.asarray([0.1, 0.5, 0.25, 0.15])
    assert assay.shortlist_indices(probabilities, mass=0.70, maximum=4) == (1, 2)
    assert assay.shortlist_indices(probabilities, mass=0.99, maximum=2) == (1, 2)
    with pytest.raises(ValueError):
        assay.shortlist_indices(probabilities, mass=0.0, maximum=2)


def test_selection_metrics_use_only_shortlisted_means() -> None:
    surface = assay.ResponsibilitySurface(
        mode_ids=(0, 1),
        target_probabilities=np.asarray([0.0, 1.0, 0.0]),
        affinity_probabilities=np.asarray([1 / 3, 1 / 3, 1 / 3]),
        means=(np.asarray([0.0]), np.asarray([1.0]), np.asarray([9.0])),
    )
    selection = assay._selection_from_probabilities(
        (0, 1), np.asarray([0.01, 0.98, 0.01])
    )
    metrics = assay.selection_metrics(
        np.asarray([1.0]),
        surface,
        selection,
        shortlist_mass=0.8,
        shortlist_maximum=1,
        proper_variance=np.asarray([1.0]),
    )
    assert metrics["shortlist_size"] == 1.0
    assert metrics["posterior_mass_captured"] == pytest.approx(1.0)
    assert metrics["posterior_argmax_recalled"] == 1.0
    assert metrics["mse"] == pytest.approx(0.0)


def test_global_selection_masks_and_renormalizes() -> None:
    selection = assay._global_selection(np.asarray([10.0, 1.0, 2.0, 30.0]), (0, 2))
    np.testing.assert_allclose(selection.probabilities, np.asarray([10, 1, 30]) / 41)


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
