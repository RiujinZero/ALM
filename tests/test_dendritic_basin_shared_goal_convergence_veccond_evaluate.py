"""Focused tests for the vector-conditioned convergence evaluator."""

from __future__ import annotations

import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_veccond_evaluate as veceval,
)


def _row(pair_id: str, partition: str, margin: float) -> dict:
    return {
        "pair_id": pair_id,
        "partition": partition,
        "sampled_margin": margin,
        "greedy_margin": margin + 0.1,
        "sampled_printable": 0.99,
        "sampled_utf8": 1.0,
        "sampled_maximum_run": 3,
    }


def test_cross_arm_series_computes_the_2x2_contrasts() -> None:
    arm_rows = {
        arm: [_row("p0", "evaluation", margin) for margin in margins]
        for arm, margins in {
            "A_k1_nooledger": [0.0, 0.0],
            "B_k1_ledger": [0.1, 0.1],
            "C_k4_nooledger": [0.2, 0.2],
            "D_k4_ledger": [0.6, 0.6],
        }.items()
    }
    result = veceval.cross_arm_series(
        arm_rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["arms"]["D_k4_ledger"]["mean"] == pytest.approx(0.6)
    assert result["contrasts"]["D_minus_C"]["mean"] == pytest.approx(0.4)
    assert result["contrasts"]["B_minus_A"]["mean"] == pytest.approx(0.1)


def test_cross_arm_series_rejects_misaligned_rows() -> None:
    arm_rows = {
        arm: [_row("p0", "evaluation", 0.0)] for arm in veceval.ARM_NAMES
    }
    arm_rows["D_k4_ledger"][0]["pair_id"] = "different"
    with pytest.raises(AssertionError):
        veceval.cross_arm_series(
            arm_rows, metric="sampled_margin", seed=1, resamples=200
        )


def test_manifest_includes_evaluator_file() -> None:
    paths = veceval.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/"
        "dendritic_basin_shared_goal_convergence_veccond_evaluate.py" in paths
    )
