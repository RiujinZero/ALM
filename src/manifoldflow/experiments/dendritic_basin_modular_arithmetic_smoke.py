"""ALM-64MB-1C raw-vector modular-arithmetic generalization ladder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    ModularSplit,
    build_modular_ladder,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import (
    GrownRelations,
    _cupy,
    fit_ridge,
    grow_relations,
    relation_activations,
)
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    DEFAULT_BODY_BYTES,
    _relation_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--operand-limit", type=int, default=32)
    parser.add_argument("--seen-moduli", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--unseen-modulus", type=int, default=11)
    parser.add_argument("--first-capacity", type=int, default=512)
    parser.add_argument("--second-capacity", type=int, default=1024)
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


def _growth_args(args: argparse.Namespace, capacity: int) -> argparse.Namespace:
    return argparse.Namespace(
        relation_capacity=capacity,
        growth_rounds=args.growth_rounds,
        nursery_size=args.nursery_size,
        hill_steps=args.hill_steps,
        fit_samples=args.fit_samples,
    )


def _states(
    inputs: np.ndarray,
    first: GrownRelations,
    second: GrownRelations,
    cp: Any,
) -> Any:
    raw = cp.asarray(inputs, dtype=cp.float32)
    first_state = relation_activations(raw, first, cp)
    second_input = cp.concatenate((raw, first_state), axis=1)
    second_state = relation_activations(second_input, second, cp)
    return cp.concatenate((raw, first_state, second_state), axis=1)


def _fit_ridge_adaptive(x: Any, y: Any, alpha: float, cp: Any) -> Any:
    """Use the smaller exact ridge system as growing anatomy exceeds samples."""
    if x.shape[1] + 1 <= x.shape[0]:
        return fit_ridge(x, y, alpha, cp)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x @ centered_x.T
    dual = cp.linalg.solve(
        gram + alpha * cp.eye(len(centered_x), dtype=centered_x.dtype),
        centered_y,
    )
    weights = centered_x.T @ dual
    bias = y_mean - x_mean @ weights
    return cp.concatenate((weights, bias[None, ...]), axis=0)


def _decode(prediction: Any, moduli: np.ndarray, max_modulus: int, cp: Any) -> Any:
    """Decode a fixed population readout while masking invalid residues."""
    classes = cp.arange(max_modulus)[None, :]
    valid = classes < cp.asarray(moduli, dtype=cp.int64)[:, None]
    return cp.argmax(cp.where(valid, prediction, -cp.inf), axis=1)


def _evaluate(
    split: ModularSplit,
    first: GrownRelations,
    second: GrownRelations,
    beta: Any,
    max_modulus: int,
    cp: Any,
) -> dict[str, Any]:
    state = _states(split.inputs, first, second, cp)
    prediction = state @ beta[:-1] + beta[-1]
    decoded = _decode(prediction, split.moduli, max_modulus, cp)
    truth = cp.asarray(split.residues, dtype=cp.int64)
    error = decoded - truth
    return {
        "examples": int(len(truth)),
        "exact_accuracy": float(cp.mean(decoded == truth).get()),
        "integer_mae": float(cp.mean(cp.abs(error)).get()),
        "integer_mse": float(cp.mean(error.astype(cp.float32) ** 2).get()),
    }


def _control_metrics(split: ModularSplit, train: ModularSplit) -> dict[str, Any]:
    truth = split.residues.astype(np.int64)
    moduli = split.moduli.astype(np.int64)
    exact_keys = {
        (int(a), int(b), int(n))
        for (a, b), n in zip(train.operands, train.moduli, strict=True)
    }
    lookup_coverage = np.asarray(
        [
            (int(a), int(b), int(n)) in exact_keys
            for (a, b), n in zip(split.operands, split.moduli, strict=True)
        ]
    )

    global_counts = np.bincount(train.residues.astype(np.int64))
    global_mode = int(np.argmax(global_counts))
    modes: dict[int, int] = {}
    for modulus in np.unique(train.moduli):
        values = train.residues[train.moduli == modulus].astype(np.int64)
        modes[int(modulus)] = int(np.argmax(np.bincount(values)))
    mode_prediction = np.asarray(
        [modes.get(int(n), global_mode) for n in moduli], dtype=np.int64
    )

    unwrapped = np.minimum(
        split.operands[:, 0].astype(np.int64)
        + split.operands[:, 1].astype(np.int64),
        moduli - 1,
    )
    # A raw-space nearest neighbor is a deliberately strong interpolation
    # control. It has no arithmetic rule and cannot retrieve an exact held-out
    # triple because the partitions are disjoint.
    nearest = np.empty(len(split.inputs), dtype=np.int64)
    for start in range(0, len(split.inputs), 256):
        query = split.inputs[start : start + 256]
        distance = np.sum(
            (query[:, None, :] - train.inputs[None, :, :]) ** 2, axis=2
        )
        nearest[start : start + len(query)] = train.residues[
            np.argmin(distance, axis=1)
        ]
    return {
        "empirical_chance_accuracy": float(
            np.mean([1.0 / int(n) for n in moduli])
        ),
        "exact_lookup_coverage": float(np.mean(lookup_coverage)),
        "residue_mode_accuracy": float(np.mean(mode_prediction == truth)),
        "nearest_raw_neighbor_accuracy": float(np.mean(nearest == truth)),
        "addition_without_wrap_accuracy": float(np.mean(unwrapped == truth)),
    }


def run_seed(args: argparse.Namespace, seed: int, cp: Any) -> dict[str, Any]:
    ladder = build_modular_ladder(
        seen_moduli=tuple(args.seen_moduli),
        unseen_modulus=args.unseen_modulus,
        operand_limit=args.operand_limit,
        seed=1_000_003 * (seed + 1),
    )
    max_modulus = max(*ladder.seen_moduli, ladder.unseen_modulus_value)
    start = time.perf_counter()
    raw_train = cp.asarray(ladder.train.inputs, dtype=cp.float32)
    first = grow_relations(
        raw_train, _growth_args(args, args.first_capacity), seed + 101, cp
    )
    first_train = relation_activations(raw_train, first, cp)
    second_train_input = cp.concatenate((raw_train, first_train), axis=1)
    second = grow_relations(
        second_train_input,
        _growth_args(args, args.second_capacity),
        seed + 211,
        cp,
    )
    train_state = cp.concatenate(
        (raw_train, first_train, relation_activations(second_train_input, second, cp)),
        axis=1,
    )
    target = cp.eye(max_modulus, dtype=cp.float32)[
        cp.asarray(ladder.train.residues, dtype=cp.int64)
    ]
    candidate_state = _states(ladder.candidate.inputs, first, second, cp)
    candidate_target = cp.eye(max_modulus, dtype=cp.float32)[
        cp.asarray(ladder.candidate.residues, dtype=cp.int64)
    ]
    best: tuple[float, float, Any] | None = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        beta = _fit_ridge_adaptive(train_state, target, alpha, cp)
        prediction = candidate_state @ beta[:-1] + beta[-1]
        decoded = _decode(prediction, ladder.candidate.moduli, max_modulus, cp)
        accuracy = float(
            cp.mean(decoded == cp.asarray(ladder.candidate.residues)).get()
        )
        mse = float(cp.mean((prediction - candidate_target) ** 2).get())
        key = (-accuracy, mse)
        if best is None or key < (best[0], best[1]):
            best = (-accuracy, mse, beta)
            selected_alpha = alpha
    assert best is not None
    beta = best[2]
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - start

    grown_body_bytes = (
        _relation_bytes(first) + _relation_bytes(second) + int(beta.nbytes)
    )
    if grown_body_bytes > args.max_body_bytes:
        raise MemoryError("grown arithmetic cortex exceeds ALM-64MB ceiling")

    per_modulus = {}
    for modulus in ladder.seen_moduli:
        mask = ladder.held_out_pairs.moduli == modulus
        subset = ModularSplit(
            ladder.held_out_pairs.inputs[mask],
            ladder.held_out_pairs.residues[mask],
            ladder.held_out_pairs.moduli[mask],
            ladder.held_out_pairs.operands[mask],
        )
        per_modulus[str(modulus)] = _evaluate(
            subset, first, second, beta, max_modulus, cp
        )
    return {
        "seed": seed,
        "candidate_selected_ridge": selected_alpha,
        "candidate_exact_accuracy": -best[0],
        "training_fit": _evaluate(
            ladder.train, first, second, beta, max_modulus, cp
        ),
        "held_out_operand_combinations": _evaluate(
            ladder.held_out_pairs, first, second, beta, max_modulus, cp
        ),
        "held_out_operand_controls": _control_metrics(
            ladder.held_out_pairs, ladder.train
        ),
        "seen_modulus_ladder": per_modulus,
        "unseen_modulus": _evaluate(
            ladder.unseen_modulus, first, second, beta, max_modulus, cp
        ),
        "unseen_modulus_controls": _control_metrics(
            ladder.unseen_modulus, ladder.train
        ),
        "chance_accuracy_unseen_modulus": 1.0 / ladder.unseen_modulus_value,
        "train_examples": len(ladder.train.residues),
        "candidate_examples": len(ladder.candidate.residues),
        "grown_body_bytes": grown_body_bytes,
        "training_seconds": seconds,
        "training_relationships_per_second": len(ladder.train.residues) / seconds,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("held_out_operand_combinations", "unseen_modulus"):
        result[group] = {}
        for metric in ("exact_accuracy", "integer_mae", "integer_mse"):
            values = np.asarray([row[group][metric] for row in rows])
            result[group][metric] = {
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
        print(f"ALM-64MB-1C modular seed {seed}", file=sys.stderr, flush=True)
        rows.append(run_seed(args, seed, cp))
    report = {
        "experiment": "alm_64mb_1c_raw_vector_modular_arithmetic",
        "alm_name": "ALM-64MB-1C",
        "input_contract": "three simultaneous float32 channels: (a, b, n)",
        "target_contract": "c = (a + b) mod n; exact integer decoding is primary",
        "forbidden_inputs": ["text", "tokens", "expression strings", "lookup IDs"],
        "selection_contract": (
            "relation growth and readout fitting use train only; ridge selection uses "
            "candidate only; held-out pairs and unseen modulus are untouched"
        ),
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
