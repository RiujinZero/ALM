"""Compare global, addressed, and locally learned communication routes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.communication_routing import (
    RoutingMode,
    SelectiveCommunicationLearner,
)
from manifoldflow.dendritic_basin.tasks.communication_routing import (
    CommunicationBatch,
    HiddenCommunicationTask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--train-trials", type=int, default=4_000)
    parser.add_argument("--eval-trials", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _run(
    learner: SelectiveCommunicationLearner,
    batch: CommunicationBatch,
    *,
    learn: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    probabilities = np.empty(len(batch.outcomes), dtype=np.float64)
    predictions = np.empty(len(batch.outcomes), dtype=np.uint8)
    started = time.perf_counter()
    for index, outcome in enumerate(batch.outcomes):
        addressed = (
            int(batch.addressed_sources[index])
            if learner.routing_mode is RoutingMode.ADDRESSED
            else None
        )
        step = learner.observe(
            batch.source_states[index],
            batch.context_receptors[index],
            int(outcome),
            addressed_source=addressed,
            learn=learn,
        )
        probabilities[index] = step.observed_probability
        predictions[index] = step.predicted_outcome
    return probabilities, predictions, time.perf_counter() - started


def _bits(probability: np.ndarray) -> float:
    return float(np.mean(-np.log2(np.maximum(probability, 1e-300))))


def run_probe(
    *,
    source_count: int = 16,
    train_trials: int = 4_000,
    eval_trials: int = 4_000,
    seed: int = 0,
) -> dict[str, object]:
    task = HiddenCommunicationTask(source_count, seed=seed)
    train = task.sample(train_trials)
    natural = task.sample(eval_trials)
    adversarial = task.sample(eval_trials, adversarial_irrelevant=True)
    report: dict[str, object] = {
        "task": "distributed_context_hidden_source_routing",
        "source_count": source_count,
        "context_receptor_count": task.context_receptor_count,
        "train_trials": train_trials,
        "eval_trials": eval_trials,
        "conditions": {},
    }
    conditions: dict[str, object] = report["conditions"]  # type: ignore[assignment]
    for mode in RoutingMode:
        learner = SelectiveCommunicationLearner(source_count, routing_mode=mode)
        train_probability, train_prediction, train_seconds = _run(
            learner, train, learn=True
        )
        natural_probability, natural_prediction, natural_seconds = _run(
            learner, natural, learn=False
        )
        adversarial_probability, adversarial_prediction, adversarial_seconds = _run(
            learner, adversarial, learn=False
        )
        learned_route_accuracy: float | None = None
        if mode is RoutingMode.LEARNED:
            learned = np.asarray(
                [
                    learner.learned_route(pattern)
                    for pattern in task.context_codebook
                ]
            )
            learned_route_accuracy = float(
                np.mean(learned == task.context_to_source)
            )
        conditions[mode.value] = {
            "train_bits_per_outcome": _bits(train_probability),
            "train_accuracy": float(np.mean(train_prediction == train.outcomes)),
            "natural_eval_bits_per_outcome": _bits(natural_probability),
            "natural_eval_accuracy": float(
                np.mean(natural_prediction == natural.outcomes)
            ),
            "adversarial_irrelevant_bits_per_outcome": _bits(
                adversarial_probability
            ),
            "adversarial_irrelevant_accuracy": float(
                np.mean(adversarial_prediction == adversarial.outcomes)
            ),
            "learned_route_accuracy": learned_route_accuracy,
            "local_updates": learner.local_updates,
            "train_trials_per_second": train_trials / max(train_seconds, 1e-12),
            "natural_eval_trials_per_second": eval_trials
            / max(natural_seconds, 1e-12),
            "adversarial_eval_trials_per_second": eval_trials
            / max(adversarial_seconds, 1e-12),
            "embodied_bytes": learner.embodied_bytes,
        }
    return report


def main() -> None:
    args = parse_args()
    report = run_probe(
        source_count=args.sources,
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
