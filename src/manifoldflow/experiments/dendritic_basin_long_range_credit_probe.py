"""Compare absent, broadcast, and apically addressed delayed local credit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.delayed_credit import (
    DendriticDelayedCreditLearner,
    TeachingMode,
)
from manifoldflow.dendritic_basin.tasks.delayed_credit import (
    DelayedCausalBatch,
    DelayedCausalTask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receivers", type=int, default=16)
    parser.add_argument("--train-trials", type=int, default=4_000)
    parser.add_argument("--eval-trials", type=int, default=4_000)
    parser.add_argument("--delay-steps", type=int, default=8)
    parser.add_argument("--eligibility-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _run_batch(
    learner: DendriticDelayedCreditLearner,
    batch: DelayedCausalBatch,
    *,
    learn: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    probability = np.empty(len(batch.outcomes), dtype=np.float64)
    predictions = np.empty(len(batch.outcomes), dtype=np.uint8)
    started = time.perf_counter()
    for index, outcome in enumerate(batch.outcomes):
        step = learner.run_trial(
            batch.basal_states[index],
            int(batch.apical_receivers[index]),
            batch.delay_steps,
            int(outcome),
            learn=learn,
        )
        probability[index] = step.observed_probability
        predictions[index] = step.predicted_outcome
    seconds = time.perf_counter() - started
    return probability, predictions, seconds


def _bits(probability: np.ndarray) -> float:
    return float(np.mean(-np.log2(np.maximum(probability, 1e-300))))


def _mapping_accuracy(
    learner: DendriticDelayedCreditLearner, hidden_polarities: np.ndarray
) -> float:
    correct = 0
    total = 0
    for receiver, polarity in enumerate(hidden_polarities):
        for basal_state in (0, 1):
            expected = basal_state ^ int(polarity)
            predicted = int(learner.probability(receiver, basal_state) >= 0.5)
            correct += int(predicted == expected)
            total += 1
    return correct / total


def run_probe(
    *,
    receiver_count: int = 16,
    train_trials: int = 4_000,
    eval_trials: int = 4_000,
    delay_steps: int = 8,
    eligibility_decay: float = 0.97,
    seed: int = 0,
) -> dict[str, object]:
    task = DelayedCausalTask(
        receiver_count,
        delay_steps=delay_steps,
        seed=seed,
    )
    train = task.sample(train_trials)
    evaluation = task.sample(eval_trials)
    result: dict[str, object] = {
        "task": "modality_neutral_delayed_opposing_causal_roles",
        "receiver_count": receiver_count,
        "train_trials": train_trials,
        "eval_trials": eval_trials,
        "delay_steps": delay_steps,
        "eligibility_decay": eligibility_decay,
        "hidden_positive_roles": int(np.sum(task.hidden_polarities == 0)),
        "hidden_negative_roles": int(np.sum(task.hidden_polarities == 1)),
        "conditions": {},
    }
    conditions: dict[str, object] = result["conditions"]  # type: ignore[assignment]
    for mode in TeachingMode:
        learner = DendriticDelayedCreditLearner(
            receiver_count,
            teaching_mode=mode,
            eligibility_decay=eligibility_decay,
        )
        train_probability, train_predictions, train_seconds = _run_batch(
            learner, train, learn=True
        )
        learner.reset_episode()
        eval_probability, eval_predictions, eval_seconds = _run_batch(
            learner, evaluation, learn=False
        )
        conditions[mode.value] = {
            "train_bits_per_outcome": _bits(train_probability),
            "train_late_bits_per_outcome": _bits(
                train_probability[-min(1000, len(train_probability)) :]
            ),
            "train_accuracy": float(np.mean(train_predictions == train.outcomes)),
            "eval_bits_per_outcome": _bits(eval_probability),
            "eval_accuracy": float(np.mean(eval_predictions == evaluation.outcomes)),
            "mapping_accuracy": _mapping_accuracy(learner, task.hidden_polarities),
            "local_updates": learner.local_updates,
            "off_route_updates": learner.off_route_updates,
            "off_route_update_fraction": (
                learner.off_route_updates / max(1, learner.local_updates)
            ),
            "train_trials_per_second": train_trials / max(train_seconds, 1e-12),
            "eval_trials_per_second": eval_trials / max(eval_seconds, 1e-12),
            "embodied_bytes": learner.embodied_bytes,
            "peak_transient_bytes": learner.transient_bytes,
        }
    return result


def main() -> None:
    args = parse_args()
    report = run_probe(
        receiver_count=args.receivers,
        train_trials=args.train_trials,
        eval_trials=args.eval_trials,
        delay_steps=args.delay_steps,
        eligibility_decay=args.eligibility_decay,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
