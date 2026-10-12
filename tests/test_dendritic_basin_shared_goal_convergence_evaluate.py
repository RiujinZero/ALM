"""Focused tests for the convergence evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_evaluate as evaluator,
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


def _rows_by_condition(margins: dict[str, list[float]]) -> dict:
    pair_count = len(next(iter(margins.values())))
    pair_ids = [f"p{index}" for index in range(pair_count)]
    return {
        condition: [
            _row(pair_id, "evaluation", margin)
            for pair_id, margin in zip(pair_ids, condition_margins, strict=True)
        ]
        for condition, condition_margins in margins.items()
    }


def test_condition_coverage_matches_contract() -> None:
    assert len(evaluator.EVAL_CONDITIONS) == 9
    assert "native" in evaluator.EVAL_CONDITIONS
    assert "oracle_code_coupled" in evaluator.EVAL_CONDITIONS
    for arm in ("A_k1_nooledger", "B_k1_ledger", "C_k4_nooledger", "D_k4_ledger"):
        assert arm in evaluator.EVAL_CONDITIONS
    assert "B_k1_ledger_reset" in evaluator.EVAL_CONDITIONS
    assert "C_k4_nooledger_reset" not in evaluator.EVAL_CONDITIONS
    assert "D_k4_ledger_shuffled" in evaluator.EVAL_CONDITIONS


def test_contrast_series_computes_paired_differences() -> None:
    rows = _rows_by_condition(
        {
            "native": [0.0, 0.0],
            "oracle_code_coupled": [1.0, 1.0],
            "A_k1_nooledger": [0.1, 0.1],
            "B_k1_ledger": [0.2, 0.2],
            "C_k4_nooledger": [0.3, 0.3],
            "D_k4_ledger": [0.5, 0.5],
            "B_k1_ledger_reset": [0.1, 0.1],
            "D_k4_ledger_reset": [0.2, 0.2],
            "D_k4_ledger_shuffled": [-0.1, -0.1],
        }
    )
    result = evaluator.contrast_series(
        rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["conditions"]["D_k4_ledger"]["mean"] == pytest.approx(0.5)
    assert result["contrasts"]["D_minus_A"]["mean"] == pytest.approx(0.4)
    assert result["contrasts"]["B_minus_A"]["mean"] == pytest.approx(0.1)
    assert result["contrasts"]["D_reset_minus_D"]["mean"] == pytest.approx(-0.3)
    assert result["contrasts"]["D_shuffled_minus_D"]["mean"] == pytest.approx(-0.6)
    assert result["contrasts"]["oracle_minus_native"]["mean"] == pytest.approx(1.0)


def test_contrast_series_rejects_misaligned_rows() -> None:
    rows = _rows_by_condition(
        {condition: [0.0] for condition in evaluator.EVAL_CONDITIONS}
    )
    rows["D_k4_ledger"][0]["pair_id"] = "different"
    with pytest.raises(AssertionError):
        evaluator.contrast_series(
            rows, metric="sampled_margin", seed=1, resamples=200
        )


def test_manifest_includes_evaluator_and_trainer_chain() -> None:
    paths = evaluator.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_evaluate.py"
        in paths
    )
    assert (
        "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_train.py"
        in paths
    )


def test_far_reweighting_returns_basal_when_no_future_modes_exist() -> None:
    basal = np.array([0.25, -0.5])
    result = evaluator.far_reweighted_candidate_mean(
        (basal,),
        np.array([1.0]),
        np.eye(2),
        np.zeros(2),
        far_tau=1.0,
    )
    assert np.array_equal(result, basal)
    assert result is not basal


def test_far_reweighting_excludes_basal_from_competing_futures() -> None:
    result = evaluator.far_reweighted_candidate_mean(
        (np.array([100.0, 100.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        np.array([0.8, 0.1, 0.1]),
        np.eye(2),
        np.array([1.0, 0.0]),
        far_tau=0.1,
    )
    assert result == pytest.approx(np.array([1.0, 0.0]))


def test_far_reweighting_rejects_misaligned_probabilities() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        evaluator.far_reweighted_candidate_mean(
            (np.zeros(2), np.ones(2)),
            np.array([1.0]),
            np.eye(2),
            np.zeros(2),
            far_tau=1.0,
        )
