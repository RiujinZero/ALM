"""Focused tests for family-first future shortlisting."""

from __future__ import annotations

import numpy as np

from manifoldflow.experiments import dendritic_basin_future_family_ablation as assay
from manifoldflow.experiments import (
    dendritic_basin_future_mode_selector_ablation as selector_assay,
)


def test_competition_pair_accounting_rewards_hierarchy() -> None:
    chosen = tuple(range(8))
    assert assay._competition_pairs(chosen, None) == 28
    assert assay._competition_pairs(chosen, ((0, 1, 2, 3), (4, 5, 6, 7))) == 13


def test_shortlist_metrics_use_original_edge_vectors() -> None:
    means = (np.asarray([0.0]), np.asarray([1.0]), np.asarray([2.0]))
    surface = selector_assay.ResponsibilitySurface(
        mode_ids=(0, 1),
        target_probabilities=np.asarray([0.1, 0.7, 0.2]),
        affinity_probabilities=np.asarray([0.2, 0.4, 0.4]),
        means=means,
    )
    metrics = assay.shortlist_metrics(
        np.asarray([1.0]),
        surface,
        np.asarray([0.2, 0.6, 0.2]),
        (1, 2),
        proper_variance=np.ones(1),
        families=((0,), (1, 2)),
        selected_family_count=1,
        eligible_candidates=2,
        shared_core_fraction=0.8,
    )
    assert np.isclose(metrics["posterior_mass_captured"], 0.9)
    assert metrics["posterior_argmax_recalled"] == 1.0
    assert metrics["selected_family_count"] == 1.0


def test_manifest_includes_family_and_ledger_mechanisms() -> None:
    paths = assay.implementation_hashes()
    assert "src/manifoldflow/dendritic_basin/future_vector_families.py" in paths
    assert "src/manifoldflow/dendritic_basin/selective_predictive_ledger.py" in paths
