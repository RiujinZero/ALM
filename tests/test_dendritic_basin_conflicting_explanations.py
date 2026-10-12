from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_competitive_resonance_routing_smoke import (  # noqa: E501
    FAMILIES,
)
from manifoldflow.experiments.dendritic_basin_conflicting_explanations_smoke import (
    generate_conflict_batch,
)


def test_conflict_batch_hides_route_identity_and_balances_reliable_organs() -> None:
    batch = generate_conflict_batch(512, seed=3, decoy_scale=0.7)
    assert batch.sequences.shape == (512, 8, 24)
    assert batch.reliable_organs.shape == (512,)
    assert set(batch.reliable_organs) == set(range(len(FAMILIES)))


def test_three_competing_organs_share_a_wrong_but_valid_proposal() -> None:
    batch = generate_conflict_batch(128, seed=5, decoy_scale=0.7)
    for target, reliable, proposals in zip(
        batch.targets, batch.reliable_organs, batch.organ_targets, strict=True
    ):
        assert proposals[reliable] == target
        decoys = np.delete(proposals, reliable)
        assert np.all(decoys == decoys[0])
        assert decoys[0] != target


def test_conflict_batch_activity_remains_bounded() -> None:
    batch = generate_conflict_batch(64, seed=7, decoy_scale=0.7)
    assert np.all(np.isfinite(batch.sequences))
    assert float(batch.sequences.min()) >= 0.0
    assert float(batch.sequences.max()) <= 1.0
