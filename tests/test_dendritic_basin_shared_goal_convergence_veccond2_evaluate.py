"""Focused tests for the dendritic-predictor evaluator."""

from __future__ import annotations

import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_veccond2_evaluate as v2eval,
)


def _row(pair_id: str, partition: str, margin: float) -> dict:
    return {
        "pair_id": pair_id,
        "partition": partition,
        "sampled_margin": margin,
        "greedy_margin": margin,
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
    result = v2eval.cross_arm_series(
        arm_rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["arms"]["D_k4_ledger"]["mean"] == pytest.approx(0.6)
    assert result["contrasts"]["D_minus_C"]["mean"] == pytest.approx(0.4)
    assert result["contrasts"]["B_minus_A"]["mean"] == pytest.approx(0.1)


def test_cross_arm_series_rejects_misaligned_rows() -> None:
    arm_rows = {
        arm: [_row("p0", "evaluation", 0.0)] for arm in v2eval.ARM_NAMES
    }
    arm_rows["D_k4_ledger"][0]["pair_id"] = "different"
    with pytest.raises(AssertionError):
        v2eval.cross_arm_series(
            arm_rows, metric="sampled_margin", seed=1, resamples=200
        )


def test_arm_contrast_series_skips_absent_reset_condition() -> None:
    rows = {
        condition: [_row("p0", "evaluation", margin)]
        for condition, margin in (
            ("native", 0.0),
            ("oracle", 0.5),
            ("arm", 0.3),
            ("arm_shuffled", -0.1),
        )
    }
    result = v2eval.arm_contrast_series(
        rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["contrasts"]["arm_minus_native"]["mean"] == pytest.approx(0.3)
    assert result["contrasts"]["arm_minus_shuffled"]["mean"] == pytest.approx(0.4)
    assert "arm_reset_minus_arm" not in result["contrasts"]


def test_arm_conditions_add_stateful_controls_without_changing_default() -> None:
    assert v2eval.conditions_for_arm("A_k1_nooledger", stateful=False) == (
        "native",
        "oracle",
        "arm",
        "arm_shuffled",
    )
    stateful = v2eval.conditions_for_arm("A_k1_nooledger", stateful=True)
    assert "arm_stateful" in stateful
    assert "arm_stateful_clear" in stateful
    assert "arm_stateful_no_context_adapt" in stateful
    assert "arm_stateful_no_generated_adapt" in stateful
    assert "arm_stateful_no_updates" in stateful
    assert "arm_context" in stateful
    assert "oracle_context" in stateful
    assert "native_stateful" not in stateful


def test_arm_contrast_series_reports_stateful_controls_when_present() -> None:
    rows = {
        condition: [_row("p0", "evaluation", margin)]
        for condition, margin in (
            ("native", 0.0),
            ("oracle", 0.1),
            ("arm", 0.2),
            ("arm_shuffled", -0.1),
            ("oracle_context", 0.12),
            ("arm_context", 0.25),
            ("oracle_stateful", 0.15),
            ("arm_stateful", 0.3),
            ("arm_stateful_clear", 0.1),
            ("arm_stateful_no_context_adapt", 0.2),
            ("arm_stateful_no_generated_adapt", 0.28),
            ("arm_stateful_no_updates", 0.18),
            ("arm_shuffled_stateful", 0.0),
        )
    }
    result = v2eval.arm_contrast_series(
        rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert (
        result["contrasts"]["arm_stateful_minus_arm_context"]["mean"]
        == pytest.approx(0.05)
    )
    assert (
        result["contrasts"]["arm_stateful_clear_minus_arm_stateful"]["mean"]
        == pytest.approx(-0.2)
    )
    assert (
        result["contrasts"]["arm_stateful_no_context_adapt_minus_arm_stateful"][
            "mean"
        ]
        == pytest.approx(-0.1)
    )
    assert (
        result["contrasts"]["arm_stateful_no_generated_adapt_minus_arm_stateful"][
            "mean"
        ]
        == pytest.approx(-0.02)
    )
    assert (
        result["contrasts"]["arm_stateful_no_updates_minus_arm_stateful"]["mean"]
        == pytest.approx(-0.12)
    )
    assert "arm_reset_minus_arm" not in result["contrasts"]


def test_manifest_includes_evaluator_file() -> None:
    paths = v2eval.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/"
        "dendritic_basin_shared_goal_convergence_veccond2_evaluate.py" in paths
    )
