from __future__ import annotations

from manifoldflow.experiments.dendritic_basin_temporal_credit_probe import (
    run_probe,
)


def test_nondecaying_trace_is_delay_invariant_without_interference() -> None:
    report = run_probe(
        receiver_count=8,
        train_trials=500,
        eval_trials=1_000,
        delays=(0, 8, 128),
        decays=(1.0,),
        seed=3,
    )
    conditions = report["conditions"]
    scores = [condition["eval_bits_per_outcome"] for condition in conditions]

    assert scores[0] == scores[1] == scores[2]
    assert all(condition["local_updates"] == 500 for condition in conditions)


def test_short_trace_expires_before_a_long_delayed_consequence() -> None:
    report = run_probe(
        receiver_count=8,
        train_trials=500,
        eval_trials=1_000,
        delays=(0, 128),
        decays=(0.5,),
        seed=4,
    )
    immediate, delayed = report["conditions"]

    assert immediate["eval_bits_per_outcome"] < 0.1
    assert delayed["eval_bits_per_outcome"] == 1.0
    assert immediate["local_updates"] == 500
    assert delayed["local_updates"] == 0
