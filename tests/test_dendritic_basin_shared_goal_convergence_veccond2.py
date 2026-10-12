"""Focused tests for the dendritic-predictor convergence run."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_veccond2 as v2,
)


def test_mixture_point_weights_by_candidate_mass() -> None:
    class Candidate:
        def __init__(self, prediction: np.ndarray, weight: float) -> None:
            self.prediction = prediction
            self.weight = weight

    prediction = np.zeros(3)
    candidates = (
        Candidate(np.array([2.0, 0.0, 0.0]), 3.0),
        Candidate(np.array([0.0, 1.0, 0.0]), 1.0),
    )
    point = v2.mixture_point(prediction, candidates, use_basal=True)
    # basal mass 1, candidate masses 3 and 1: (3*(2,0,0) + 1*(0,1,0)) / 5
    assert np.allclose(point, [1.2, 0.2, 0.0])
    no_basal = v2.mixture_point(prediction, candidates, use_basal=False)
    # (3*(2,0,0) + 1*(0,1,0)) / 4
    assert np.allclose(no_basal, [1.5, 0.25, 0.0])


def test_mixture_point_single_candidate_ignores_basal_when_excluded() -> None:
    class Candidate:
        prediction = np.array([4.0, 4.0, 4.0])
        weight = 1.0

    point = v2.mixture_point(np.zeros(3), (Candidate(),), use_basal=False)
    assert np.allclose(point, [4.0, 4.0, 4.0])


def test_cross_arm_series_computes_the_2x2_contrasts() -> None:
    def rows(margins: list[float]) -> list[dict]:
        return [
            {
                "pair_id": f"p{index}",
                "partition": "evaluation",
                "sampled_margin": margin,
                "greedy_margin": margin,
                "sampled_printable": 0.99,
                "sampled_utf8": 1.0,
                "sampled_maximum_run": 3,
            }
            for index, margin in enumerate(margins)
        ]

    arm_rows = {
        "A_k1_nooledger": rows([0.0, 0.0]),
        "B_k1_ledger": rows([0.1, 0.1]),
        "C_k4_nooledger": rows([0.2, 0.2]),
        "D_k4_ledger": rows([0.6, 0.6]),
    }
    result = v2.cross_arm_series(
        arm_rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["contrasts"]["D_minus_C"]["mean"] == pytest.approx(0.4)
    assert result["contrasts"]["B_minus_A"]["mean"] == pytest.approx(0.1)


def test_cross_arm_series_rejects_misaligned_rows() -> None:
    def rows(margin: float) -> list[dict]:
        return [
            {
                "pair_id": "p0",
                "partition": "evaluation",
                "sampled_margin": margin,
                "greedy_margin": margin,
                "sampled_printable": 0.99,
                "sampled_utf8": 1.0,
                "sampled_maximum_run": 3,
            }
        ]

    arm_rows = {
        "A_k1_nooledger": rows(0.0),
        "B_k1_ledger": rows(0.1),
        "C_k4_nooledger": rows(0.2),
        "D_k4_ledger": rows(0.6),
    }
    arm_rows["D_k4_ledger"][0]["pair_id"] = "different"
    with pytest.raises(AssertionError):
        v2.cross_arm_series(arm_rows, metric="sampled_margin", seed=1, resamples=200)


def test_manifest_includes_v2_chain() -> None:
    paths = v2.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/"
        "dendritic_basin_shared_goal_convergence_veccond2.py" in paths
    )
    assert "src/manifoldflow/dendritic_basin/vector_conditioned_base.py" in paths
