"""Focused tests for the persistent-core generalization assay."""

from __future__ import annotations

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_ablation as assay,
)


def _row(pair_id: str, side: str, own_mse: float, persistent_mse: float) -> dict:
    return {
        "pair_id": pair_id,
        "side": side,
        "arms": {
            "own_core": {
                "mse": own_mse,
                "cosine": 0.5,
                "proper_bits_per_coordinate": own_mse,
            },
            "persistent_core": {
                "mse": persistent_mse,
                "cosine": 0.5,
                "proper_bits_per_coordinate": persistent_mse,
            },
        },
    }


def test_summarize_single_partition_averages_sides_within_a_pair() -> None:
    rows = [
        _row("p1", "left", own_mse=0.10, persistent_mse=0.12),
        _row("p1", "right", own_mse=0.20, persistent_mse=0.18),
        _row("p2", "left", own_mse=0.30, persistent_mse=0.30),
    ]
    summary = assay._summarize_single_partition(rows, seed=1, resamples=200)
    assert summary["pairs"] == 2
    assert summary["sides"] == 3
    # pair-level mean for p1 is (0.10+0.20)/2=0.15, for p2 is 0.30 -> overall mean 0.225
    assert summary["arms"]["own_core"]["mse"]["mean"] == 0.225


def test_gate_passes_when_persistent_is_noninferior_to_own() -> None:
    evaluation = {
        "context_modes": {
            "long_interruption_crossover": {
                "evaluation_summary": {
                    "paired_contrasts": {
                        "persistent_minus_own_mse": {
                            "mean": 0.01,
                            "ci95_low": -0.02,
                            "ci95_high": 0.03,
                        },
                        "persistent_minus_own_proper_bits_per_coordinate": {
                            "mean": -0.01,
                            "ci95_low": -0.04,
                            "ci95_high": 0.02,
                        },
                    }
                }
            }
        }
    }

    class _Args:
        mse_noninferiority_margin = 0.10
        proper_bits_noninferiority_margin = 0.05

    gate = assay.persistent_core_gate(evaluation, _Args())
    assert gate["components"]["persistent_noninferior_to_own_on_mse"] is True
    assert gate["components"]["persistent_noninferior_to_own_on_proper_score"] is True
    assert gate["persistent_core_generalizes_without_regressing"] is True


def test_gate_fails_when_persistent_regresses_past_the_margin() -> None:
    evaluation = {
        "context_modes": {
            "long_interruption_crossover": {
                "evaluation_summary": {
                    "paired_contrasts": {
                        "persistent_minus_own_mse": {
                            "mean": 0.50,
                            "ci95_low": 0.30,
                            "ci95_high": 0.70,
                        },
                        "persistent_minus_own_proper_bits_per_coordinate": {
                            "mean": 0.20,
                            "ci95_low": 0.10,
                            "ci95_high": 0.30,
                        },
                    }
                }
            }
        }
    }

    class _Args:
        mse_noninferiority_margin = 0.10
        proper_bits_noninferiority_margin = 0.05

    gate = assay.persistent_core_gate(evaluation, _Args())
    assert gate["components"]["persistent_noninferior_to_own_on_mse"] is False
    assert gate["persistent_core_generalizes_without_regressing"] is False


def test_manifest_includes_persistent_core_and_core_edge_mechanisms() -> None:
    paths = assay.implementation_hashes()
    assert "src/manifoldflow/dendritic_basin/future_vector_persistent_core.py" in paths
    assert "src/manifoldflow/dendritic_basin/future_vector_core_edge.py" in paths
