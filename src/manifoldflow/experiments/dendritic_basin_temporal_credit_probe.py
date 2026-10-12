"""Sweep local eligibility lifetime across delayed causal consequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.delayed_credit import (
    DendriticDelayedCreditLearner,
    TeachingMode,
)
from manifoldflow.dendritic_basin.tasks.delayed_credit import DelayedCausalTask

DEFAULT_DELAYS = (0, 2, 8, 32, 128)
DEFAULT_DECAYS = (0.5, 0.9, 0.97, 0.995, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receivers", type=int, default=16)
    parser.add_argument("--train-trials", type=int, default=1_000)
    parser.add_argument("--eval-trials", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _bits(probability: np.ndarray) -> float:
    return float(np.mean(-np.log2(np.maximum(probability, 1e-300))))


def _evaluate_condition(
    *,
    receiver_count: int,
    train_trials: int,
    eval_trials: int,
    delay_steps: int,
    eligibility_decay: float,
    seed: int,
) -> dict[str, float | int]:
    task = DelayedCausalTask(
        receiver_count,
        delay_steps=delay_steps,
        seed=seed,
    )
    train = task.sample(train_trials)
    evaluation = task.sample(eval_trials)
    learner = DendriticDelayedCreditLearner(
        receiver_count,
        teaching_mode=TeachingMode.APICAL,
        eligibility_decay=eligibility_decay,
    )
    for index, outcome in enumerate(train.outcomes):
        learner.run_trial(
            train.basal_states[index],
            int(train.apical_receivers[index]),
            train.delay_steps,
            int(outcome),
            learn=True,
        )
    probability = np.empty(eval_trials, dtype=np.float64)
    correct = 0
    for index, outcome in enumerate(evaluation.outcomes):
        step = learner.run_trial(
            evaluation.basal_states[index],
            int(evaluation.apical_receivers[index]),
            evaluation.delay_steps,
            int(outcome),
            learn=False,
        )
        probability[index] = step.observed_probability
        correct += int(step.predicted_outcome == outcome)
    return {
        "delay_steps": delay_steps,
        "eligibility_decay": eligibility_decay,
        "eligibility_at_consequence": eligibility_decay**delay_steps,
        "eval_bits_per_outcome": _bits(probability),
        "eval_accuracy": correct / eval_trials,
        "local_updates": learner.local_updates,
        "embodied_bytes": learner.embodied_bytes,
        "peak_transient_bytes": learner.transient_bytes,
    }


def run_probe(
    *,
    receiver_count: int = 16,
    train_trials: int = 1_000,
    eval_trials: int = 4_000,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    decays: tuple[float, ...] = DEFAULT_DECAYS,
    seed: int = 0,
) -> dict[str, object]:
    conditions = []
    for decay in decays:
        for delay in delays:
            conditions.append(
                _evaluate_condition(
                    receiver_count=receiver_count,
                    train_trials=train_trials,
                    eval_trials=eval_trials,
                    delay_steps=delay,
                    eligibility_decay=decay,
                    seed=seed,
                )
            )
    return {
        "task": "local_eligibility_delay_sweep",
        "receiver_count": receiver_count,
        "train_trials": train_trials,
        "eval_trials": eval_trials,
        "delays": delays,
        "decays": decays,
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    report = run_probe(
        receiver_count=args.receivers,
        train_trials=args.train_trials,
        eval_trials=args.eval_trials,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
