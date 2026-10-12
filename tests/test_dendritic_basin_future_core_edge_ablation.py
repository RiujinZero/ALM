"""Focused tests for the flat-top-k core/edge context-specificity assay."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.experiments import dendritic_basin_future_core_edge_ablation as assay


def test_sattolo_derangement_has_no_fixed_points_and_is_a_permutation() -> None:
    for seed in range(20):
        permutation = assay.sattolo_derangement(9, seed)
        assert sorted(permutation.tolist()) == list(range(9))
        assert not np.any(permutation == np.arange(9))


def test_sattolo_derangement_is_deterministic_for_a_fixed_seed() -> None:
    first = assay.sattolo_derangement(12, seed=42)
    second = assay.sattolo_derangement(12, seed=42)
    assert np.array_equal(first, second)


def test_sattolo_derangement_rejects_zero_rows() -> None:
    with pytest.raises(ValueError):
        assay.sattolo_derangement(0, seed=1)


def test_core_edge_point_metrics_matches_a_hand_computed_point() -> None:
    target = np.asarray([1.0, 0.0])
    core = np.asarray([0.5, 0.0])
    weighted_edge_sum = np.asarray([0.5, 0.0])  # point == target exactly
    metrics = assay.core_edge_point_metrics(
        target, core, weighted_edge_sum, proper_variance=np.ones(2)
    )
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["cosine"] == pytest.approx(1.0)


def test_weighted_edge_sum_reweights_only_over_the_chosen_shortlist() -> None:
    means = tuple(
        np.asarray(item, dtype=np.float64) for item in ((0.0,), (10.0,), (1_000.0,))
    )
    probabilities = np.asarray([0.6, 0.4, 100.0])
    chosen = (0, 1)
    split = mass_weighted_core_edge_split(means, probabilities, chosen)
    weighted = assay._weighted_edge_sum(probabilities, chosen, split)
    # core + weighted_edge_sum must reconstruct the same point as the split's
    # own mass-weighted mean, i.e. drift back to core itself (edges sum to
    # zero under their own generating weights).
    assert (split.core + weighted) == pytest.approx(split.core)


def test_own_core_metrics_exactly_match_foreign_core_metrics_when_cores_tie() -> None:
    target = np.asarray([2.0, -1.0])
    weighted_edge_sum = np.asarray([0.1, 0.2])
    core = np.asarray([0.3, -0.1])
    own = assay.core_edge_point_metrics(
        target, core, weighted_edge_sum, proper_variance=np.ones(2)
    )
    foreign = assay.core_edge_point_metrics(
        target, core.copy(), weighted_edge_sum, proper_variance=np.ones(2)
    )
    assert own == pytest.approx(foreign)


def test_foreign_core_seed_is_deterministic_and_mode_partition_sensitive() -> None:
    first = assay._foreign_core_seed(11, "long_interruption_crossover", "evaluation")
    second = assay._foreign_core_seed(11, "long_interruption_crossover", "evaluation")
    third = assay._foreign_core_seed(11, "long_interruption_crossover", "selection")
    assert first == second
    assert first != third


def test_manifest_includes_core_edge_and_ledger_mechanisms() -> None:
    paths = assay.implementation_hashes()
    assert "src/manifoldflow/dendritic_basin/future_vector_core_edge.py" in paths
    assert "src/manifoldflow/dendritic_basin/selective_predictive_ledger.py" in paths
