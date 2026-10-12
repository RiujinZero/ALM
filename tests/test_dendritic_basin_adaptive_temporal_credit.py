from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.delayed_credit import (
    CompetingEligibilityTraceLearner,
)
from manifoldflow.dendritic_basin.tasks.delayed_credit import (
    OverlappingDelayedCausalTask,
)
from manifoldflow.experiments.dendritic_basin_adaptive_temporal_credit_probe import (
    run_probe,
)


def test_overlapping_task_points_to_the_selected_receivers_past() -> None:
    task = OverlappingDelayedCausalTask(8, causal_delay=5, seed=2)
    stream = task.sample(100)
    valid = np.arange(5, 100)
    expected = stream.basal_states[
        valid - 5,
        stream.apical_receivers[valid],
    ]

    np.testing.assert_array_equal(stream.outcomes[valid], expected)


def test_trace_bank_updates_only_the_apically_selected_receiver() -> None:
    learner = CompetingEligibilityTraceLearner(4)
    learner.observe_sources(np.asarray((0, 1, 0, 1), dtype=np.uint8))
    learner.observe_consequence(2, 1)

    assert np.count_nonzero(learner.outcome_counts[2]) == len(
        learner.trace_decays
    )
    assert np.count_nonzero(learner.outcome_counts[:2]) == 0
    assert np.count_nonzero(learner.outcome_counts[3:]) == 0


def test_local_trace_competition_rejects_too_short_and_too_long_memory() -> None:
    report = run_probe(
        receiver_count=8,
        causal_delay=8,
        train_steps=50_000,
        eval_steps=20_000,
        seed=3,
    )
    scores = {
        row["decay"]: row["eval_bits_per_outcome"]
        for row in report["per_decay"]
    }

    assert report["selected_eval_bits_per_outcome"] < 0.99
    assert scores[0.97] < scores[0.0]
    assert scores[0.97] < scores[0.995]
    assert set(report["selected_decays"]) <= {0.9, 0.97}
