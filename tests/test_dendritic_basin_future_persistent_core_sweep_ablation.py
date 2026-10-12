"""Focused tests for the persistent-core knob sweep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.future_vector_persistent_core import (
    PersistentCoreEstimator,
    PersistentCoreEstimatorConfig,
)
from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep,
)


@dataclass(frozen=True)
class _FakePair:
    pair_id: str
    partition: str


def _config_entry(index: int, bits: float, mse: float) -> dict:
    def mode_summary() -> dict:
        return {
            "evaluation_summary": {
                "paired_contrasts": {
                    "persistent_minus_own_proper_bits_per_coordinate": {"mean": bits},
                    "persistent_minus_own_mse": {"mean": mse},
                }
            }
        }

    return {
        "config_index": index,
        "long_interruption_crossover": mode_summary(),
        "original_unanswered_prefix": mode_summary(),
    }


def test_pair_parity_split_is_deterministic_and_disjoint() -> None:
    selection = [
        _FakePair("c", "selection"),
        _FakePair("a", "selection"),
        _FakePair("b", "selection"),
        _FakePair("d", "selection"),
    ]
    fit, evaluation = sweep.pair_parity_split(selection)
    assert [pair.pair_id for pair in fit] == ["a", "c"]
    assert [pair.pair_id for pair in evaluation] == ["b", "d"]
    assert not {pair.pair_id for pair in fit} & {pair.pair_id for pair in evaluation}
    fit_again, evaluation_again = sweep.pair_parity_split(selection)
    assert [pair.pair_id for pair in fit_again] == [pair.pair_id for pair in fit]
    assert [pair.pair_id for pair in evaluation_again] == [
        pair.pair_id for pair in evaluation
    ]


def test_choose_config_prefers_lower_proper_bits_contrast() -> None:
    entries = [
        _config_entry(0, bits=0.20, mse=0.01),
        _config_entry(1, bits=0.05, mse=0.30),
        _config_entry(2, bits=0.10, mse=0.02),
    ]
    chosen = sweep.choose_config(entries, tie_tolerance=1e-12)
    assert chosen["config_index"] == 1


def test_choose_config_tie_breaks_on_mse_then_grid_order() -> None:
    entries = [
        _config_entry(0, bits=0.10, mse=0.20),
        _config_entry(1, bits=0.10, mse=0.05),
        _config_entry(2, bits=0.10 + 1e-13, mse=0.01),
    ]
    chosen = sweep.choose_config(entries, tie_tolerance=1e-12)
    assert chosen["config_index"] == 2


def test_choose_config_full_tie_falls_back_to_grid_order() -> None:
    entries = [
        _config_entry(2, bits=0.10, mse=0.05),
        _config_entry(0, bits=0.10, mse=0.05),
        _config_entry(1, bits=0.10, mse=0.05),
    ]
    chosen = sweep.choose_config(entries, tie_tolerance=1e-12)
    assert chosen["config_index"] == 0


def test_choose_config_empty_grid_raises() -> None:
    try:
        sweep.choose_config([], tie_tolerance=1e-12)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty config list")


def _estimator(bandwidth: float, min_affinity: float) -> PersistentCoreEstimator:
    estimator = PersistentCoreEstimator(
        PersistentCoreEstimatorConfig(
            context_dim=2,
            core_dim=2,
            max_slots=4,
            context_bandwidth=bandwidth,
            min_context_affinity=min_affinity,
        )
    )
    estimator.update(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    return estimator


def test_context_is_eligible_mirrors_estimator_affinity_gate() -> None:
    estimator = _estimator(bandwidth=1.0, min_affinity=0.50)
    assert sweep.context_is_eligible(np.array([0.1, 0.1]), estimator)
    assert not sweep.context_is_eligible(np.array([5.0, 5.0]), estimator)
    # An ineligible context must fall back to the global running core.
    fallback = estimator.predict(np.array([5.0, 5.0]))
    assert np.allclose(fallback, estimator.global_core)


def test_manifest_includes_sweep_and_locked_chain_files() -> None:
    paths = sweep.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_sweep_ablation.py"
        in paths
    )
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_ablation.py"
        in paths
    )
    assert "src/manifoldflow/dendritic_basin/future_vector_persistent_core.py" in paths
