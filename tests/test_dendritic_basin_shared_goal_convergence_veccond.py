"""Focused tests for the vector-conditioned convergence run."""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_veccond as veccond,
)
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


class _FakeCortex:
    """Minimal byte cortex: uniform distribution, stateless enough for tests."""

    def __init__(self) -> None:
        self.resets = 0

    def reset_episode(self) -> None:
        self.resets += 1

    def observe_chunk_probabilities(self, byte_ids, *, learn: bool):
        return np.full(len(byte_ids), 1.0 / 256.0)

    def predict_next_distribution(self) -> np.ndarray:
        return np.full(256, 1.0 / 256.0)

    def advance_observation(self, byte_id: int, *, learn: bool = False) -> None:
        return None


def test_generate_unprimed_runs_horizon_bytes_and_resets() -> None:
    cortex = _FakeCortex()
    data, digest = veccond.generate_unprimed(
        cortex, horizon=16, uniform_draws=np.full(16, 0.5)
    )
    assert len(data) == 16
    assert len(digest) == 64
    assert cortex.resets == 2


def test_contrast_series_covers_all_named_contrasts() -> None:
    rows = _rows_by_condition(
        {
            "native": [0.0, 0.0],
            "oracle": [1.0, 1.0],
            "arm": [0.5, 0.5],
            "arm_reset": [0.2, 0.2],
            "arm_shuffled": [-0.1, -0.1],
        }
    )
    result = veccond.contrast_series(
        rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert set(result["contrasts"]) == set(veccond.CONTRAST_NAMES)
    assert result["contrasts"]["arm_minus_native"]["mean"] == pytest.approx(0.5)
    assert result["contrasts"]["arm_reset_minus_arm"]["mean"] == pytest.approx(-0.3)
    assert result["contrasts"]["arm_minus_shuffled"]["mean"] == pytest.approx(0.6)


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
    assert result["contrasts"]["D_minus_C"]["mean"] == pytest.approx(0.4)
    assert result["contrasts"]["B_minus_A"]["mean"] == pytest.approx(0.1)


def test_manifest_includes_veccond_chain() -> None:
    paths = veccond.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_veccond.py"
        in paths
    )
    assert "src/manifoldflow/dendritic_basin/vector_conditioned_base.py" in paths
