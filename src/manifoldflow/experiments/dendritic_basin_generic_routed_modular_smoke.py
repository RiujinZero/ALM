"""Generic competitive-resonance routing on modular arithmetic.

Unlike the successful manually composed modular smoke, no organ is assigned an
arithmetic role and no operand, modulus, register, or execution route is built
into the anatomy.  Four generically differentiated organs receive the same raw
ASCII stream and compete according to learned future agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    ModularSplit,
    build_modular_ladder,
    modular_expression,
    unfolded_modular_lesson,
)
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

GEOMETRIES = ("persistent", "coincident", "ordered", "slow")
RECEPTORS = 128
HASH_RECEPTORS = 256


@dataclass(frozen=True)
class GenericOrgan:
    geometry: str
    relations: GrownRelations
    continuation_beta: Any
    resonance_beta: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--operand-limit", type=int, default=32)
    parser.add_argument("--seen-moduli", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--unseen-modulus", type=int, default=11)
    parser.add_argument(
        "--curricula",
        nargs="+",
        choices=("endpoint", "mixed"),
        default=["endpoint", "mixed"],
    )
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


def _compact_prefixes(split: ModularSplit) -> list[bytes]:
    return [
        modular_expression(int(a), int(b), int(n))
        for (a, b), n in zip(split.operands, split.moduli, strict=True)
    ]


def _unfolded_prefix(a: int, b: int, n: int) -> bytes:
    residue = (a + b) % n
    lesson = unfolded_modular_lesson(a, b, n, residue)
    return lesson[: -(len(str(residue)) + 1)]


def development_examples(
    split: ModularSplit, curriculum: str
) -> tuple[list[bytes], np.ndarray]:
    compact = _compact_prefixes(split)
    if curriculum == "endpoint":
        return compact, split.residues.copy()
    unfolded = [
        _unfolded_prefix(int(a), int(b), int(n))
        for (a, b), n in zip(split.operands, split.moduli, strict=True)
    ]
    return compact + unfolded, np.concatenate((split.residues, split.residues))


def generic_byte_evidence(
    expressions: list[bytes], geometry: str, cp: Any
) -> Any:
    """Apply a task-agnostic receptor geometry to raw character events."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown generic geometry: {geometry}")
    if geometry == "persistent":
        result = np.zeros((len(expressions), RECEPTORS * 2), dtype=np.float32)
        for row, expression in enumerate(expressions):
            values = np.frombuffer(expression, dtype=np.uint8)
            counts = np.bincount(values, minlength=RECEPTORS).astype(np.float32)
            result[row, :RECEPTORS] = counts / max(len(values), 1)
            result[row, RECEPTORS:] = counts > 0
        return cp.asarray(result)

    if geometry in ("coincident", "ordered"):
        result = np.zeros((len(expressions), HASH_RECEPTORS), dtype=np.float32)
        for row, expression in enumerate(expressions):
            values = np.frombuffer(expression, dtype=np.uint8).astype(np.int64)
            if geometry == "coincident":
                events = 0
                for right in range(len(values)):
                    for left in range(max(0, right - 4), right):
                        low, high = sorted((int(values[left]), int(values[right])))
                        bucket = (low * 131 + high * 17) % HASH_RECEPTORS
                        result[row, bucket] += 1.0
                        events += 1
            else:
                events = 0
                for index in range(1, len(values)):
                    bucket = (
                        int(values[index - 1]) * 131 + int(values[index]) * 17
                    ) % HASH_RECEPTORS
                    result[row, bucket] += 1.0
                    events += 1
                for index in range(2, len(values)):
                    bucket = (
                        int(values[index - 2]) * 197
                        + int(values[index - 1]) * 67
                        + int(values[index]) * 13
                    ) % HASH_RECEPTORS
                    result[row, bucket] += 1.0
                    events += 1
            result[row] /= max(events, 1)
        return cp.asarray(result)

    timescales = (2, 4, 8, 16)
    lambdas = np.asarray(
        [np.exp(np.log(0.5) / scale) for scale in timescales], dtype=np.float32
    )
    result = np.zeros((len(expressions), RECEPTORS * len(timescales)), dtype=np.float32)
    for row, expression in enumerate(expressions):
        traces = np.zeros((len(timescales), RECEPTORS), dtype=np.float32)
        for byte_id in expression:
            traces *= lambdas[:, None]
            traces[:, byte_id] += 1.0
        traces *= (1.0 - lambdas)[:, None]
        result[row] = traces.reshape(-1)
    return cp.asarray(result)


def _state(evidence: Any, relations: GrownRelations, cp: Any) -> Any:
    activation = relation_activations(evidence, relations, cp)
    return cp.concatenate((evidence, activation), axis=1)


def _select_ridge(
    train_x: Any, train_y: Any, candidate_x: Any, candidate_y: Any, cp: Any
) -> tuple[Any, float, float]:
    best: tuple[float, float, Any] | None = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        beta = _fit_ridge_adaptive(train_x, train_y, alpha, cp)
        prediction = candidate_x @ beta[:-1] + beta[-1]
        loss = float(cp.mean((prediction - candidate_y) ** 2).get())
        if best is None or loss < best[0]:
            best = (loss, alpha, beta)
    assert best is not None
    return best[2], best[1], best[0]


def _fit_organ(
    geometry: str,
    train_expressions: list[bytes],
    train_residues: np.ndarray,
    candidate: ModularSplit,
    max_modulus: int,
    args: argparse.Namespace,
    seed: int,
    cp: Any,
) -> tuple[GenericOrgan, dict[str, float]]:
    train_evidence = generic_byte_evidence(train_expressions, geometry, cp)
    relations = grow_relations(train_evidence, _growth_args(args), seed, cp)
    train_state = _state(train_evidence, relations, cp)
    candidate_state = _state(
        generic_byte_evidence(_compact_prefixes(candidate), geometry, cp), relations, cp
    )
    train_target = cp.eye(max_modulus, dtype=cp.float32)[cp.asarray(train_residues)]
    candidate_target = cp.eye(max_modulus, dtype=cp.float32)[
        cp.asarray(candidate.residues)
    ]
    continuation, alpha, candidate_mse = _select_ridge(
        train_state, train_target, candidate_state, candidate_target, cp
    )
    prediction = candidate_state @ continuation[:-1] + continuation[-1]
    disagreement = cp.mean((prediction - candidate_target) ** 2, axis=1, keepdims=True)
    resonance = _fit_ridge_adaptive(candidate_state, disagreement, 10.0, cp)
    return GenericOrgan(geometry, relations, continuation, resonance), {
        "continuation_ridge": alpha,
        "candidate_mse": candidate_mse,
    }


def _organ_outputs(
    organ: GenericOrgan, expressions: list[bytes], cp: Any
) -> tuple[Any, Any]:
    evidence = generic_byte_evidence(expressions, organ.geometry, cp)
    state = _state(evidence, organ.relations, cp)
    prediction = state @ organ.continuation_beta[:-1] + organ.continuation_beta[-1]
    disagreement = state @ organ.resonance_beta[:-1] + organ.resonance_beta[-1]
    return prediction, cp.maximum(disagreement[:, 0], 0.0)


def evaluate(
    organs: list[GenericOrgan],
    split: ModularSplit,
    cp: Any,
    *,
    fixed_organ: int,
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    expressions = _compact_prefixes(split)
    predictions = []
    disagreements = []
    for organ in organs:
        prediction, disagreement = _organ_outputs(organ, expressions, cp)
        predictions.append(prediction)
        disagreements.append(disagreement)
    prediction_stack = cp.stack(predictions, axis=1)
    disagreement_stack = cp.stack(disagreements, axis=1)
    if shuffle_seed is not None:
        rng = cp.random.RandomState(shuffle_seed)
        for column in range(len(organs)):
            disagreement_stack[:, column] = disagreement_stack[
                rng.permutation(len(split.residues)), column
            ]
    winner = cp.argmin(disagreement_stack, axis=1)
    row = cp.arange(len(split.residues))
    target = cp.asarray(split.residues)
    valid = cp.arange(prediction_stack.shape[2])[None, :] < cp.asarray(
        split.moduli
    )[:, None]

    def accuracy(logits: Any) -> float:
        guessed = cp.argmax(cp.where(valid, logits, -cp.inf), axis=1)
        return float(cp.mean(guessed == target).get())

    selected = prediction_stack[row, winner]
    one_hot = cp.eye(prediction_stack.shape[2], dtype=cp.float32)[target]
    actual_loss = cp.mean((prediction_stack - one_hot[:, None, :]) ** 2, axis=2)
    oracle = cp.argmin(actual_loss, axis=1)
    route_counts = cp.asnumpy(cp.bincount(winner, minlength=len(organs)))
    return {
        "examples": len(split.residues),
        "routed_accuracy": accuracy(selected),
        "uniform_pool_accuracy": accuracy(prediction_stack.mean(axis=1)),
        "oracle_router_accuracy": accuracy(prediction_stack[row, oracle]),
        "fixed_candidate_selected_organ_accuracy": accuracy(
            prediction_stack[:, fixed_organ]
        ),
        "per_organ_accuracy": {
            organ.geometry: accuracy(prediction_stack[:, index])
            for index, organ in enumerate(organs)
        },
        "route_fraction": {
            organ.geometry: float(route_counts[index] / len(split.residues))
            for index, organ in enumerate(organs)
        },
    }


def _candidate_best_organ(
    organs: list[GenericOrgan], split: ModularSplit, cp: Any
) -> int:
    expressions = _compact_prefixes(split)
    target = cp.asarray(split.residues)
    valid = cp.arange(max(split.moduli) + 1)[None, :] < cp.asarray(
        split.moduli
    )[:, None]
    scores = []
    for organ in organs:
        prediction, _ = _organ_outputs(organ, expressions, cp)
        prediction = prediction[:, : valid.shape[1]]
        guessed = cp.argmax(cp.where(valid, prediction, -cp.inf), axis=1)
        scores.append(float(cp.mean(guessed == target).get()))
    return int(np.argmax(scores))


def run_seed(
    args: argparse.Namespace, seed: int, curriculum: str, cp: Any
) -> dict[str, Any]:
    ladder = build_modular_ladder(
        seen_moduli=tuple(args.seen_moduli),
        unseen_modulus=args.unseen_modulus,
        operand_limit=args.operand_limit,
        seed=1_000_003 * (seed + 1),
        partition_by_operand_pair=True,
    )
    max_modulus = max(*ladder.seen_moduli, ladder.unseen_modulus_value)
    train_expressions, train_residues = development_examples(ladder.train, curriculum)
    started = time.perf_counter()
    fitted = [
        _fit_organ(
            geometry,
            train_expressions,
            train_residues,
            ladder.candidate,
            max_modulus,
            args,
            seed * 101 + index * 17,
            cp,
        )
        for index, geometry in enumerate(GEOMETRIES)
    ]
    organs = [item[0] for item in fitted]
    diagnostics = {
        organ.geometry: item[1]
        for organ, item in zip(organs, fitted, strict=True)
    }
    fixed_organ = _candidate_best_organ(organs, ladder.candidate, cp)
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
        raise MemoryError("generic routed organism exceeds ALM-64MB ceiling")

    def score(split: ModularSplit, shuffle_seed: int | None = None) -> dict[str, Any]:
        return evaluate(
            organs,
            split,
            cp,
            fixed_organ=fixed_organ,
            shuffle_seed=shuffle_seed,
        )

    held_out = score(ladder.held_out_pairs)
    per_modulus = {}
    for modulus in ladder.seen_moduli:
        mask = ladder.held_out_pairs.moduli == modulus
        subset = ModularSplit(
            ladder.held_out_pairs.inputs[mask],
            ladder.held_out_pairs.residues[mask],
            ladder.held_out_pairs.moduli[mask],
            ladder.held_out_pairs.operands[mask],
        )
        per_modulus[str(modulus)] = score(subset)
    return {
        "seed": seed,
        "curriculum": curriculum,
        "training_examples": len(train_residues),
        "candidate": score(ladder.candidate),
        "held_out_operand_pairs": held_out,
        "held_out_shuffled_resonance": score(
            ladder.held_out_pairs, shuffle_seed=seed + 701
        ),
        "held_out_by_modulus": per_modulus,
        "unseen_modulus": score(ladder.unseen_modulus),
        "candidate_selected_organ": organs[fixed_organ].geometry,
        "organ_diagnostics": diagnostics,
        "grown_body_bytes": body_bytes,
        "organ_bytes": {
            organ.geometry: int(
                _relation_bytes(organ.relations)
                + organ.continuation_beta.nbytes
                + organ.resonance_beta.nbytes
            )
            for organ in organs
        },
        "development_seconds": seconds,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for curriculum in sorted({row["curriculum"] for row in rows}):
        selected = [row for row in rows if row["curriculum"] == curriculum]
        metrics = {
            "routed_held_out": ("held_out_operand_pairs", "routed_accuracy"),
            "uniform_pool_held_out": (
                "held_out_operand_pairs",
                "uniform_pool_accuracy",
            ),
            "oracle_router_held_out": (
                "held_out_operand_pairs",
                "oracle_router_accuracy",
            ),
            "fixed_organ_held_out": (
                "held_out_operand_pairs",
                "fixed_candidate_selected_organ_accuracy",
            ),
            "shuffled_resonance_held_out": (
                "held_out_shuffled_resonance",
                "routed_accuracy",
            ),
            "routed_unseen_modulus": ("unseen_modulus", "routed_accuracy"),
        }
        result[curriculum] = {}
        for name, (group, metric) in metrics.items():
            values = np.asarray([row[group][metric] for row in selected])
            result[curriculum][name] = {
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
    for curriculum in args.curricula:
        for offset in range(args.seeds):
            seed = args.seed_offset + offset
            print(
                f"generic routed modular {curriculum} seed {seed}",
                file=sys.stderr,
                flush=True,
            )
            rows.append(run_seed(args, seed, curriculum, cp))
    report = {
        "experiment": "alm_64mb_1c_generic_routed_modular_arithmetic",
        "alm_name": "ALM-64MB-1C",
        "organs": list(GEOMETRIES),
        "organ_contract": (
            "all organs receive the same raw ASCII stream and use the same local "
            "relation rule; only generic persistence, local coincidence, ordered "
            "transition, and slow-trace receptor geometries differ"
        ),
        "non_specialization_contract": (
            "no organ receives an arithmetic role, operand field, modulus field, "
            "numeric register, residue, cyclic-process identity, or addressed route"
        ),
        "routing_contract": (
            "candidate episodes teach each organ to predict its own future "
            "disagreement; lowest predicted disagreement wins without an organ label"
        ),
        "curriculum_contract": (
            "endpoint uses only compact expressions; mixed duplicates each training "
            "combination as a compact expression and a raw unfolded ring lesson; all "
            "candidate and held-out prompts remain compact equations"
        ),
        "split_contract": (
            "held-out operand pairs are absent under every seen modulus; modulus 11 "
            "is absent from training and candidate selection"
        ),
        "body_class_bytes": args.max_body_bytes,
        "developmental_columns": args.columns,
        "args": vars(args) | {"report": str(args.report) if args.report else None},
        "aggregate": aggregate(rows),
        "runs": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
