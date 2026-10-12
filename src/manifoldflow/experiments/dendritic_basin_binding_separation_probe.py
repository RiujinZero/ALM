"""Test role-preserving recombination and refusal of false binding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.role_binding import BindingMode, RoleBindingLearner
from manifoldflow.dendritic_basin.tasks.role_binding import (
    CompositionalRoleTask,
    RoleBindingBatch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=16)
    parser.add_argument("--train-trials", type=int, default=4_000)
    parser.add_argument("--eval-trials", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _run(
    learner: RoleBindingLearner, batch: RoleBindingBatch, *, learn: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    probability = np.empty(len(batch.outcomes), dtype=np.float64)
    prediction = np.empty(len(batch.outcomes), dtype=np.uint8)
    started = time.perf_counter()
    for index, outcome in enumerate(batch.outcomes):
        step = learner.observe(
            batch.actor_receptors[index],
            batch.patient_receptors[index],
            int(outcome),
            learn=learn,
        )
        probability[index] = step.observed_probability
        prediction[index] = step.predicted_outcome
    return probability, prediction, time.perf_counter() - started


def _bits(probability: np.ndarray) -> float:
    return float(np.mean(-np.log2(np.maximum(probability, 1e-300))))


def run_probe(
    *,
    entity_count: int = 16,
    train_trials: int = 4_000,
    eval_trials: int = 4_000,
    seed: int = 0,
) -> dict[str, object]:
    task = CompositionalRoleTask(entity_count, seed=seed)
    train = task.sample_train(train_trials)
    unseen = task.sample_unseen(eval_trials)
    actor_properties = task.hidden_properties[unseen.actor_ids]
    patient_properties = task.hidden_properties[unseen.patient_ids]
    differing_roles = actor_properties != patient_properties
    report: dict[str, object] = {
        "task": "distributed_content_role_binding_and_separation",
        "entity_count": entity_count,
        "receptor_count": task.receptor_count,
        "active_receptors_per_entity": task.active_receptors,
        "train_trials": train_trials,
        "eval_trials": eval_trials,
        "seen_ordered_pairs": len(task.train_pairs),
        "held_out_ordered_pairs": len(task.eval_pairs),
        "conditions": {},
    }
    conditions: dict[str, object] = report["conditions"]  # type: ignore[assignment]
    for mode in BindingMode:
        learner = RoleBindingLearner(binding_mode=mode)
        train_probability, train_prediction, train_seconds = _run(
            learner, train, learn=True
        )
        eval_probability, eval_prediction, eval_seconds = _run(
            learner, unseen, learn=False
        )
        conditions[mode.value] = {
            "train_bits_per_outcome": _bits(train_probability),
            "train_accuracy": float(np.mean(train_prediction == train.outcomes)),
            "unseen_pair_bits_per_outcome": _bits(eval_probability),
            "unseen_pair_accuracy": float(np.mean(eval_prediction == unseen.outcomes)),
            "opposed_role_accuracy": float(
                np.mean(
                    eval_prediction[differing_roles]
                    == unseen.outcomes[differing_roles]
                )
            ),
            "local_updates": learner.local_updates,
            "off_role_updates": learner.off_role_updates,
            "off_role_update_fraction": (
                learner.off_role_updates / max(1, learner.local_updates)
            ),
            "train_trials_per_second": train_trials / max(train_seconds, 1e-12),
            "eval_trials_per_second": eval_trials / max(eval_seconds, 1e-12),
            "embodied_bytes": learner.embodied_bytes,
        }
    return report


def main() -> None:
    args = parse_args()
    report = run_probe(
        entity_count=args.entities,
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
