"""Unlabeled differentiated organs learn routing by predictive resonance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_echolocation_smoke import (
    GrownRelations,
    _cupy,
    grow_relations,
    relation_activations,
)
from manifoldflow.experiments.dendritic_basin_modular_arithmetic_smoke import (
    _fit_ridge_adaptive,
)
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
    _relation_bytes,
)

FAMILIES = ("invariant", "conjunctive", "sequential", "regulatory")
CONTENT = slice(0, 8)
SECONDARY = slice(8, 16)
CONTEXT = slice(16, 24)
CHANNELS = 24
STEPS = 8


@dataclass(frozen=True)
class ProcessBatch:
    sequences: np.ndarray
    targets: np.ndarray
    families: np.ndarray


@dataclass(frozen=True)
class Organ:
    name: str
    relations: GrownRelations
    continuation_beta: Any
    resonance_beta: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-per-family", type=int, default=1200)
    parser.add_argument("--candidate-per-family", type=int, default=300)
    parser.add_argument("--test-per-family", type=int, default=600)
    parser.add_argument("--relation-capacity", type=int, default=512)
    parser.add_argument("--growth-rounds", type=int, default=8)
    parser.add_argument("--nursery-size", type=int, default=256)
    parser.add_argument("--hill-steps", type=int, default=64)
    parser.add_argument("--fit-samples", type=int, default=6000)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.columns != 1 or args.max_body_bytes != DEFAULT_BODY_BYTES:
        parser.error("this smoke is fixed to the ALM-64MB-1C body class")
    return args


def _growth_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        relation_capacity=args.relation_capacity,
        growth_rounds=args.growth_rounds,
        nursery_size=args.nursery_size,
        hill_steps=args.hill_steps,
        fit_samples=args.fit_samples,
    )


def generate_process_batch(per_family: int, seed: int) -> ProcessBatch:
    """Generate interleaved processes without exposing family IDs as inputs."""
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    families: list[int] = []
    for family in range(len(FAMILIES)):
        for _ in range(per_family):
            frames = np.zeros((STEPS, CHANNELS), dtype=np.float32)
            if family == 0:
                identity = int(rng.integers(8))
                frames[:6, identity] = 1.0
                noise = rng.integers(8, 16, size=STEPS)
                frames[np.arange(STEPS), noise] = 1.0
                target = identity
            elif family == 1:
                left = int(rng.integers(8))
                right = int(rng.integers(8))
                frames[:-1, rng.integers(8)] = 0.2
                frames[-1, left] = 1.0
                frames[-1, 8 + right] = 1.0
                target = (left + right) % 8
            elif family == 2:
                start = int(rng.integers(8))
                direction = -1 if rng.random() < 0.5 else 1
                positions = [(start + direction * step) % 8 for step in range(STEPS)]
                frames[np.arange(STEPS), positions] = 1.0
                target = (positions[-1] + direction) % 8
            else:
                current = int(rng.integers(8))
                direction = int(rng.integers(2))
                frames[:, current] = 0.15
                frames[-1, CONTENT] = 0.0
                frames[-1, current] = 1.0
                frames[:4, 16 + direction] = 1.0
                target = (current + (1 if direction == 0 else -1)) % 8
            sequences.append(frames)
            targets.append(target)
            families.append(family)
    order = rng.permutation(len(sequences))
    return ProcessBatch(
        np.asarray(sequences)[order],
        np.asarray(targets, dtype=np.int64)[order],
        np.asarray(families, dtype=np.int64)[order],
    )


def organ_evidence(sequences: np.ndarray, organ: str, cp: Any) -> Any:
    """Apply innate receptor geometry without using process-family labels."""
    values = cp.asarray(sequences, dtype=cp.float32)
    if organ == "invariant":
        mean = values.mean(axis=1)
        persistence = cp.mean(values > 0.1, axis=1)
        return cp.where(persistence >= 0.75, mean, 0.0)
    if organ == "conjunctive":
        current = values[:, -1]
        conjunction = (
            current[:, CONTENT, None] * current[:, None, SECONDARY]
        ).reshape(len(values), -1)
        return cp.concatenate((current, conjunction), axis=1)
    if organ == "sequential":
        previous = values[:, -2, CONTENT]
        current = values[:, -1, CONTENT]
        eligibility = (previous[:, :, None] * current[:, None, :]).reshape(
            len(values), -1
        )
        return cp.concatenate((previous, current, eligibility), axis=1)
    if organ == "regulatory":
        weights = cp.asarray(
            [0.5 ** ((STEPS - 1 - step) / 8.0) for step in range(STEPS)],
            dtype=cp.float32,
        )
        slow_context = cp.sum(values[:, :, CONTEXT] * weights[None, :, None], axis=1)
        slow_context /= cp.sum(weights)
        current = values[:, -1, CONTENT]
        return cp.concatenate((current, slow_context), axis=1)
    raise ValueError(f"unknown organ geometry: {organ}")


def _state(evidence: Any, relations: GrownRelations, cp: Any) -> Any:
    activation = relation_activations(evidence, relations, cp)
    return cp.concatenate((evidence, activation), axis=1)


def _fit_ridge(x: Any, y: Any, alpha: float, cp: Any) -> Any:
    return _fit_ridge_adaptive(x, y, alpha, cp)


def _fit_organ(
    name: str,
    train: ProcessBatch,
    candidate: ProcessBatch,
    args: argparse.Namespace,
    seed: int,
    cp: Any,
) -> Organ:
    train_evidence = organ_evidence(train.sequences, name, cp)
    relations = grow_relations(train_evidence, _growth_args(args), seed, cp)
    train_state = _state(train_evidence, relations, cp)
    candidate_state = _state(
        organ_evidence(candidate.sequences, name, cp), relations, cp
    )
    train_target = cp.eye(8, dtype=cp.float32)[cp.asarray(train.targets)]
    candidate_target = cp.eye(8, dtype=cp.float32)[cp.asarray(candidate.targets)]
    best: tuple[float, Any] | None = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        beta = _fit_ridge(train_state, train_target, alpha, cp)
        prediction = candidate_state @ beta[:-1] + beta[-1]
        loss = float(cp.mean((prediction - candidate_target) ** 2).get())
        if best is None or loss < best[0]:
            best = (loss, beta)
    assert best is not None
    continuation_beta = best[1]

    candidate_prediction = (
        candidate_state @ continuation_beta[:-1] + continuation_beta[-1]
    )
    observed_disagreement = cp.mean(
        (candidate_prediction - candidate_target) ** 2, axis=1, keepdims=True
    )
    # A local metaplastic trace: the organ learns which of its own states tend
    # to be confirmed by the next external observation.
    resonance_beta = _fit_ridge(candidate_state, observed_disagreement, 10.0, cp)
    return Organ(name, relations, continuation_beta, resonance_beta)


def _organ_outputs(
    organ: Organ, sequences: np.ndarray, cp: Any
) -> tuple[Any, Any]:
    state = _state(organ_evidence(sequences, organ.name, cp), organ.relations, cp)
    prediction = state @ organ.continuation_beta[:-1] + organ.continuation_beta[-1]
    disagreement = state @ organ.resonance_beta[:-1] + organ.resonance_beta[-1]
    return prediction, cp.maximum(disagreement[:, 0], 0.0)


def evaluate(
    organs: list[Organ],
    batch: ProcessBatch,
    cp: Any,
    *,
    shuffle_resonance_seed: int | None = None,
) -> dict[str, Any]:
    predictions = []
    disagreements = []
    for organ in organs:
        prediction, disagreement = _organ_outputs(organ, batch.sequences, cp)
        predictions.append(prediction)
        disagreements.append(disagreement)
    prediction_stack = cp.stack(predictions, axis=1)
    disagreement_stack = cp.stack(disagreements, axis=1)
    if shuffle_resonance_seed is not None:
        rng = cp.random.RandomState(shuffle_resonance_seed)
        for column in range(len(organs)):
            disagreement_stack[:, column] = disagreement_stack[
                rng.permutation(len(batch.targets)), column
            ]
    winner = cp.argmin(disagreement_stack, axis=1)
    selected = prediction_stack[cp.arange(len(batch.targets)), winner]
    selected_class = cp.argmax(selected, axis=1)
    uniform_class = cp.argmax(prediction_stack.mean(axis=1), axis=1)
    target = cp.asarray(batch.targets)
    family = cp.asarray(batch.families)
    actual_losses = cp.mean(
        (
            prediction_stack
            - cp.eye(8, dtype=cp.float32)[target][:, None, :]
        )
        ** 2,
        axis=2,
    )
    oracle = cp.argmin(actual_losses, axis=1)
    oracle_class = cp.argmax(
        prediction_stack[cp.arange(len(batch.targets)), oracle], axis=1
    )
    per_family = {}
    route_matrix = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.int64)
    winner_cpu = cp.asnumpy(winner)
    for expected in range(len(FAMILIES)):
        mask = batch.families == expected
        counts = np.bincount(winner_cpu[mask], minlength=len(FAMILIES))
        route_matrix[expected] = counts
        per_family[FAMILIES[expected]] = {
            "examples": int(mask.sum()),
            "continuation_accuracy": float(
                cp.mean(
                    selected_class[cp.asarray(mask)] == target[cp.asarray(mask)]
                ).get()
            ),
            "expected_organ_route_fraction": float(
                cp.mean(winner[cp.asarray(mask)] == expected).get()
            ),
        }
    return {
        "examples": len(batch.targets),
        "continuation_accuracy": float(cp.mean(selected_class == target).get()),
        "expected_organ_route_accuracy": float(cp.mean(winner == family).get()),
        "uniform_pool_accuracy": float(cp.mean(uniform_class == target).get()),
        "oracle_resonance_accuracy": float(cp.mean(oracle_class == target).get()),
        "route_matrix": route_matrix.tolist(),
        "per_family": per_family,
    }


def generate_sequential_rollouts(
    count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sequences = np.zeros((count, STEPS, CHANNELS), dtype=np.float32)
    directions = rng.choice(np.asarray((-1, 1)), size=count)
    starts = rng.integers(8, size=count)
    for row, (start, direction) in enumerate(zip(starts, directions, strict=True)):
        positions = [(int(start) + int(direction) * step) % 8 for step in range(STEPS)]
        sequences[row, np.arange(STEPS), positions] = 1.0
    return sequences, directions


def evaluate_autonomous_rollout(
    organs: list[Organ], count: int, steps: int, seed: int, cp: Any
) -> dict[str, Any]:
    sequences, directions = generate_sequential_rollouts(count, seed)
    exact = np.ones(count, dtype=bool)
    sequential_routes = 0
    total_routes = 0
    for _ in range(steps):
        predictions = []
        disagreements = []
        for organ in organs:
            prediction, disagreement = _organ_outputs(organ, sequences, cp)
            predictions.append(prediction)
            disagreements.append(disagreement)
        winner = cp.argmin(cp.stack(disagreements, axis=1), axis=1)
        selected = cp.stack(predictions, axis=1)[cp.arange(count), winner]
        generated = cp.asnumpy(cp.argmax(selected, axis=1))
        current = np.argmax(sequences[:, -1, CONTENT], axis=1)
        expected = (current + directions) % 8
        exact &= generated == expected
        sequential_routes += int(cp.sum(winner == FAMILIES.index("sequential")).get())
        total_routes += count
        sequences[:, :-1] = sequences[:, 1:]
        sequences[:, -1] = 0.0
        sequences[np.arange(count), -1, generated] = 1.0
    return {
        "rollouts": count,
        "generated_steps": steps,
        "exact_trajectory_fraction": float(exact.mean()),
        "sequential_route_fraction": sequential_routes / total_routes,
    }


def run_seed(args: argparse.Namespace, seed: int, cp: Any) -> dict[str, Any]:
    train = generate_process_batch(args.train_per_family, seed * 1009 + 11)
    candidate = generate_process_batch(args.candidate_per_family, seed * 1009 + 23)
    test = generate_process_batch(args.test_per_family, seed * 1009 + 37)
    started = time.perf_counter()
    organs = [
        _fit_organ(name, train, candidate, args, seed * 101 + index * 17, cp)
        for index, name in enumerate(FAMILIES)
    ]
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started
    body_bytes = int(
        sum(
            _relation_bytes(organ.relations)
            + organ.continuation_beta.nbytes
            + organ.resonance_beta.nbytes
            for organ in organs
        )
    )
    if body_bytes > args.max_body_bytes:
        raise MemoryError("competitive resonance organism exceeds ALM-64MB ceiling")
    return {
        "seed": seed,
        "held_out": evaluate(organs, test, cp),
        "shuffled_resonance": evaluate(
            organs, test, cp, shuffle_resonance_seed=seed + 701
        ),
        "sequential_autonomous_rollout": evaluate_autonomous_rollout(
            organs, 512, 8, seed * 1009 + 53, cp
        ),
        "grown_body_bytes": body_bytes,
        "organ_bytes": {
            organ.name: int(
                _relation_bytes(organ.relations)
                + organ.continuation_beta.nbytes
                + organ.resonance_beta.nbytes
            )
            for organ in organs
        },
        "development_seconds": seconds,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "held_out_continuation_accuracy": ("held_out", "continuation_accuracy"),
        "held_out_route_accuracy": ("held_out", "expected_organ_route_accuracy"),
        "uniform_pool_accuracy": ("held_out", "uniform_pool_accuracy"),
        "oracle_resonance_accuracy": ("held_out", "oracle_resonance_accuracy"),
        "shuffled_resonance_accuracy": (
            "shuffled_resonance",
            "continuation_accuracy",
        ),
        "autonomous_exact_trajectory_fraction": (
            "sequential_autonomous_rollout",
            "exact_trajectory_fraction",
        ),
    }
    result = {}
    for name, (group, metric) in paths.items():
        values = np.asarray([row[group][metric] for row in rows])
        result[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    return result


def main() -> None:
    args = parse_args()
    cp = _cupy()
    rows = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(f"competitive resonance seed {seed}", file=sys.stderr, flush=True)
        rows.append(run_seed(args, seed, cp))
    report = {
        "experiment": "alm_64mb_1c_competitive_resonance_routing",
        "alm_name": "ALM-64MB-1C",
        "organs": list(FAMILIES),
        "organ_contract": (
            "all organs use the same local relation growth rule but receive innate "
            "persistent, simultaneous, eligibility, or slow-trace evidence geometry"
        ),
        "routing_contract": (
            "no process-family identity or addressed organ enters development or "
            "inference; each organ predicts its own future disagreement and divisive "
            "competition selects the lowest-disagreement resonance"
        ),
        "confirmation_contract": (
            "only later external observations train route confidence; generated "
            "states do not confirm their own route"
        ),
        "body_class_bytes": args.max_body_bytes,
        "developmental_columns": args.columns,
        "args": vars(args) | {"report": str(args.report) if args.report else None},
        "aggregate": aggregate(rows),
        "seeds": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
