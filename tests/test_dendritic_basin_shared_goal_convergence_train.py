"""Focused tests for the convergence trainer."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_convergence_train as trainer,
)


def test_delta_organ_near_update_reduces_error_on_repeated_target() -> None:
    organ = trainer.DeltaOrgan(
        feature_dim=3,
        code_dim=2,
        near_rate=0.5,
        far_rate=0.5,
        far_enabled=False,
        near_seed=np.zeros((2, 3)),
        far_seed=None,
    )
    phi = np.array([1.0, 0.0, 0.0])
    target = np.array([1.0, -1.0])
    first = organ.update_near(phi, target)
    second = organ.update_near(phi, target)
    assert second < first
    assert organ.near_updates == 2
    prediction = organ.predict_near(phi)
    assert np.allclose(prediction, target * (1.0 - 0.5**2), atol=1e-12)


def test_delta_organ_far_head_requires_far_enabled() -> None:
    organ = trainer.DeltaOrgan(
        feature_dim=2,
        code_dim=2,
        near_rate=0.1,
        far_rate=0.1,
        far_enabled=False,
        near_seed=None,
        far_seed=None,
    )
    with pytest.raises(ValueError):
        organ.predict_far(np.zeros(2))


def test_delta_organ_rejects_bad_seed_shape() -> None:
    with pytest.raises(ValueError):
        trainer.DeltaOrgan(
            feature_dim=3,
            code_dim=2,
            near_rate=0.1,
            far_rate=0.1,
            far_enabled=True,
            near_seed=np.zeros((2, 4)),
            far_seed=None,
        )


def test_ridge_seed_recovers_a_linear_map_exactly() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 4))
    true_map = rng.normal(size=(3, 4))
    targets = true_map @ features.T
    seed = trainer.ridge_seed(features, targets.T, alpha=1e-8)
    assert np.allclose(seed @ features.T, targets, atol=1e-6)


def test_pass_marks_match_runbook_schedule() -> None:
    marks = trainer.pass_marks(8.0, updates_per_pass=1000)
    assert marks == [0, 250, 500, 1000, 2000, 4000, 8000]
    short = trainer.pass_marks(1.5, updates_per_pass=1000)
    assert short == [0, 250, 500, 1000, 1500]


def test_snapshot_payload_round_trip() -> None:
    organ = trainer.DeltaOrgan(
        feature_dim=2,
        code_dim=2,
        near_rate=0.1,
        far_rate=0.1,
        far_enabled=True,
        near_seed=np.ones((2, 2)),
        far_seed=np.ones((2, 2)) * 2.0,
    )
    receivers = {arm: object() for arm in trainer.ARM_NAMES}
    payload = trainer.snapshot_payload(
        {arm: organ for arm in trainer.ARM_NAMES}, receivers, 500, 1000
    )
    assert payload["updates_done"] == 500
    assert payload["passes"] == 0.5
    restored = pickle.loads(pickle.dumps(payload))
    assert np.allclose(restored["organs"]["B_k1_ledger"]["near"], 1.0)
    assert np.allclose(restored["organs"]["B_k1_ledger"]["far"], 2.0)


def test_manifest_includes_trainer_chain() -> None:
    paths = trainer.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/dendritic_basin_shared_goal_convergence_train.py"
        in paths
    )
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_sweep_ablation.py"
        in paths
    )
