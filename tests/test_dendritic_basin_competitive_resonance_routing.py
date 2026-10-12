from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_competitive_resonance_routing_smoke import (  # noqa: E501
    FAMILIES,
    generate_process_batch,
    generate_sequential_rollouts,
    organ_evidence,
)


def test_interleaved_process_batch_does_not_append_family_label() -> None:
    batch = generate_process_batch(8, seed=3)
    assert batch.sequences.shape == (32, 8, 24)
    assert set(batch.families) == set(range(len(FAMILIES)))
    assert batch.targets.shape == (32,)


def test_differentiated_organs_receive_distinct_generic_geometries() -> None:
    batch = generate_process_batch(2, seed=5)
    widths = {
        name: organ_evidence(batch.sequences, name, np).shape[1]
        for name in FAMILIES
    }
    assert widths == {
        "invariant": 24,
        "conjunctive": 88,
        "sequential": 80,
        "regulatory": 16,
    }


def test_sequential_rollout_prefix_encodes_direction_without_route_label() -> None:
    sequences, directions = generate_sequential_rollouts(32, seed=7)
    previous = np.argmax(sequences[:, -2, :8], axis=1)
    current = np.argmax(sequences[:, -1, :8], axis=1)
    np.testing.assert_array_equal((current - previous + 4) % 8 - 4, directions)
