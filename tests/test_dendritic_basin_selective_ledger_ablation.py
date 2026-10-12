"""Focused tests for the selective predictive-ledger assay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_selective_ledger_ablation as assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)


def _example(conversation: int, transition: int, token: tuple[float, float]):
    vector = np.asarray((*token, 0.0, 0.0), dtype=np.float64)
    target = np.asarray([float(conversation), float(transition)])
    turn = SimpleNamespace(role_id=1)
    return surprise_assay.TransitionExample(
        conversation_index=conversation,
        transition_index=transition,
        episode_id=f"episode-{conversation}",
        context_state=vector,
        basal_prediction=np.zeros(2),
        target=target,
        target_turn=turn,
    )


def test_bank_fit_is_target_free_and_traces_persist() -> None:
    examples = (
        _example(0, 0, (0.0, 0.0)),
        _example(0, 1, (3.0, 0.0)),
        _example(1, 0, (0.0, 0.0)),
    )
    projector = surprise_assay.ContextProjector.fit(
        np.asarray([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]]),
        dimensions=2,
        scale_floor=0.1,
        clip=10.0,
    )
    args = SimpleNamespace(
        ledger_capacity=4,
        ledger_birth_distance=0.5,
        ledger_match_distance=10.0,
        ledger_prototype_learning_rate=0.05,
        ledger_growth_order_seed=1,
    )
    bank, persistent, current, audit = assay.fit_prototype_bank_and_traces(
        examples,
        projector,
        conversations=2,
        token_dimensions=2,
        args=args,
    )
    assert bank.count == 2
    assert persistent[0] == (current[0],)
    assert len(persistent[1]) == 2
    assert persistent[2] == (current[2],)
    assert audit["successor_targets_used_for_birth_or_matching"] == 0


def test_implementation_manifest_includes_mechanism_and_runner() -> None:
    paths = assay.implementation_hashes()
    assert "src/manifoldflow/dendritic_basin/selective_predictive_ledger.py" in paths
    assert (
        "src/manifoldflow/experiments/dendritic_basin_selective_ledger_ablation.py"
        in paths
    )
