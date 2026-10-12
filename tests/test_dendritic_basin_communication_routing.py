from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.communication_routing import (
    RoutingMode,
    SelectiveCommunicationLearner,
)
from manifoldflow.dendritic_basin.tasks.communication_routing import (
    HiddenCommunicationTask,
)
from manifoldflow.experiments.dendritic_basin_selective_communication_probe import (
    run_probe,
)


def test_hidden_route_task_can_make_every_irrelevant_source_adversarial() -> None:
    task = HiddenCommunicationTask(seed=3)
    batch = task.sample(100, adversarial_irrelevant=True)
    for index, outcome in enumerate(batch.outcomes):
        selected = batch.addressed_sources[index]
        assert batch.source_states[index, selected] == outcome
        irrelevant = np.delete(batch.source_states[index], selected)
        assert np.all(irrelevant == 1 - outcome)


def test_learned_router_rejects_hidden_destination_argument() -> None:
    learner = SelectiveCommunicationLearner(4, routing_mode=RoutingMode.LEARNED)
    with pytest.raises(ValueError, match="must not receive"):
        learner.observe(
            np.asarray((0, 1, 0, 1), dtype=np.uint8),
            np.asarray((1, 0, 1, 0), dtype=np.uint8),
            1,
            addressed_source=2,
        )


def test_local_route_competition_discovers_the_causal_source() -> None:
    report = run_probe(
        source_count=16,
        train_trials=4_000,
        eval_trials=4_000,
        seed=5,
    )
    conditions = report["conditions"]
    global_condition = conditions["global"]
    addressed = conditions["addressed"]
    learned = conditions["learned"]

    assert addressed["adversarial_irrelevant_bits_per_outcome"] < 0.01
    assert learned["adversarial_irrelevant_bits_per_outcome"] < 0.02
    assert learned["adversarial_irrelevant_accuracy"] == 1.0
    assert learned["learned_route_accuracy"] == 1.0
    assert learned["adversarial_irrelevant_bits_per_outcome"] < global_condition[
        "adversarial_irrelevant_bits_per_outcome"
    ]
