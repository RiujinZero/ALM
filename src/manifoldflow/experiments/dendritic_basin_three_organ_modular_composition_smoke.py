"""Three-organ ALM smoke for process-to-symbol modular composition.

One organ develops a reusable one-step cyclic transition, one develops on the
raw character stream of conventional equations, and an integration organ sees
their synchronized states.  The integration association never receives the
residue: it restores a process start state and movement count, after which the
process organ must recurrently produce the answer.  Shuffled synchronization
and a direct symbolic endpoint head are reported as causal controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    ModularSplit,
    build_modular_ladder,
    modular_expression,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--operand-limit", type=int, default=32)
    parser.add_argument("--seen-moduli", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--unseen-modulus", type=int, default=11)
    parser.add_argument(
        "--split-granularity",
        choices=("triple", "operand_pair"),
        default="triple",
    )
    parser.add_argument("--symbol-capacity", type=int, default=512)
    parser.add_argument("--process-capacity", type=int, default=512)
    parser.add_argument("--integration-capacity", type=int, default=1024)
    parser.add_argument(
        "--symbol-topology",
        choices=("mixed_trace", "role_gated"),
        default="mixed_trace",
    )
    parser.add_argument(
        "--bridge-target",
        choices=("process_activity", "afferent_register"),
        default="process_activity",
    )
    parser.add_argument(
        "--execution-protocol",
        choices=("reduced_start", "origin_two_phase"),
        default="reduced_start",
    )
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


def ascii_trace_features(
    expressions: list[bytes],
    timescales: tuple[int, ...],
    cp: Any,
    *,
    receptor_count: int = 128,
) -> Any:
    """Transduce raw characters into generic decaying receptor traces."""
    lambdas = np.asarray(
        [math.exp(math.log(0.5) / scale) for scale in timescales],
        dtype=np.float32,
    )
    result = np.zeros(
        (len(expressions), receptor_count * (1 + len(timescales))),
        dtype=np.float32,
    )
    for row, expression in enumerate(expressions):
        traces = np.zeros((len(timescales), receptor_count), dtype=np.float32)
        current = np.zeros(receptor_count, dtype=np.float32)
        for byte_id in expression:
            if byte_id >= receptor_count:
                raise ValueError("the symbolic organ accepts 7-bit ASCII only")
            current.fill(0.0)
            current[byte_id] = 1.0
            traces *= lambdas[:, None]
            traces[:, byte_id] += 1.0
        traces *= (1.0 - lambdas)[:, None]
        result[row] = np.concatenate((current, traces.reshape(-1)))
    return cp.asarray(result)


def role_gated_ascii_features(expressions: list[bytes], cp: Any) -> Any:
    """Route digit events into innate operand/modulus working registers.

    The transducer knows the public grammar boundaries but neither converts a
    field to a number nor computes an answer.  Each field has two ordered digit
    receptor banks; incoming digits shift the prior event into the tens bank.
    """
    result = np.zeros((len(expressions), 3, 2, 10), dtype=np.float32)
    for row, expression in enumerate(expressions):
        role = 0
        for byte_id in expression:
            if byte_id == ord("+"):
                role = 1
            elif byte_id == ord("m"):
                role = 2
            elif ord("0") <= byte_id <= ord("9"):
                digit = byte_id - ord("0")
                result[row, role, 0] = result[row, role, 1]
                result[row, role, 1].fill(0.0)
                result[row, role, 1, digit] = 1.0
    return cp.asarray(result.reshape(len(expressions), -1))


def _symbol_features(split: ModularSplit, cp: Any, topology: str) -> Any:
    expressions = [
        modular_expression(int(a), int(b), int(n))
        for (a, b), n in zip(split.operands, split.moduli, strict=True)
    ]
    if topology == "role_gated":
        return role_gated_ascii_features(expressions, cp)
    return ascii_trace_features(expressions, (2, 4, 8, 16), cp)


def _process_raw(position: Any, modulus: Any, max_modulus: int, cp: Any) -> Any:
    return cp.column_stack(
        (
            cp.asarray(position, dtype=cp.float32) / max_modulus,
            cp.asarray(modulus, dtype=cp.float32) / max_modulus,
        )
    )


def _process_state(raw: Any, relations: GrownRelations, cp: Any) -> Any:
    return cp.concatenate((raw, relation_activations(raw, relations, cp)), axis=1)


def _symbol_state(raw: Any, relations: GrownRelations, cp: Any) -> Any:
    return cp.concatenate((raw, relation_activations(raw, relations, cp)), axis=1)


def _integration_state(symbol: Any, relations: GrownRelations, cp: Any) -> Any:
    return cp.concatenate((symbol, relation_activations(symbol, relations, cp)), axis=1)


def _transition_examples(
    moduli: tuple[int, ...], max_modulus: int, cp: Any
) -> tuple[Any, Any]:
    rows = [(position, n) for n in moduli for position in range(n)]
    raw = _process_raw(
        [position for position, _n in rows],
        [n for _position, n in rows],
        max_modulus,
        cp,
    )
    target = cp.zeros((len(rows), max_modulus), dtype=cp.float32)
    target[cp.arange(len(rows)), cp.asarray([(p + 1) % n for p, n in rows])] = 1.0
    return raw, target


def _select_ridge(
    train_x: Any,
    train_y: Any,
    candidate_x: Any,
    candidate_y: Any,
    cp: Any,
) -> tuple[Any, float, float]:
    best: tuple[float, float, Any] | None = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        beta = _fit_ridge_adaptive(train_x, train_y, alpha, cp)
        prediction = candidate_x @ beta[:-1] + beta[-1]
        error = float(cp.mean((prediction - candidate_y) ** 2).get())
        if best is None or error < best[0]:
            best = (error, alpha, beta)
    assert best is not None
    return best[2], best[1], best[0]


def _cue_targets(
    split: ModularSplit,
    process_relations: GrownRelations,
    max_modulus: int,
    operand_limit: int,
    cp: Any,
    target_mode: str = "process_activity",
    execution_protocol: str = "reduced_start",
) -> Any:
    if execution_protocol == "origin_two_phase":
        start = np.zeros(len(split.operands), dtype=np.int16)
    else:
        start = np.mod(split.operands[:, 0], split.moduli)
    raw = _process_raw(start, split.moduli, max_modulus, cp)
    if execution_protocol == "origin_two_phase":
        movements = cp.asarray(split.operands, dtype=cp.float32)
        movements /= max(operand_limit - 1, 1)
        return cp.concatenate((raw, movements), axis=1)
    moves = cp.asarray(split.operands[:, 1], dtype=cp.float32)[:, None]
    moves /= max(operand_limit - 1, 1)
    if target_mode == "afferent_register":
        return cp.concatenate((raw, moves), axis=1)
    state = _process_state(raw, process_relations, cp)
    return cp.concatenate((state, moves), axis=1)


def _decode_cues(
    prediction: Any,
    process_relations: GrownRelations,
    max_modulus: int,
    operand_limit: int,
    cp: Any,
    target_mode: str = "process_activity",
    execution_protocol: str = "reduced_start",
) -> tuple[Any, Any, Any]:
    if target_mode == "afferent_register":
        positions = cp.rint(prediction[:, 0] * max_modulus).astype(cp.int64)
        moduli = cp.rint(prediction[:, 1] * max_modulus).astype(cp.int64)
        if execution_protocol == "origin_two_phase":
            first = cp.rint(
                prediction[:, 2] * max(operand_limit - 1, 1)
            ).astype(cp.int64)
            second = cp.rint(
                prediction[:, 3] * max(operand_limit - 1, 1)
            ).astype(cp.int64)
            moves = cp.clip(first, 0, operand_limit - 1) + cp.clip(
                second, 0, operand_limit - 1
            )
        else:
            moves = cp.rint(
                prediction[:, 2] * max(operand_limit - 1, 1)
            ).astype(cp.int64)
        return (
            cp.clip(positions, 0, max_modulus - 1),
            cp.clip(moduli, 2, max_modulus),
            cp.clip(moves, 0, 2 * (operand_limit - 1)),
        )
    prototype_rows = [
        (position, modulus)
        for modulus in range(2, max_modulus + 1)
        for position in range(modulus)
    ]
    prototype_raw = _process_raw(
        [position for position, _modulus in prototype_rows],
        [modulus for _position, modulus in prototype_rows],
        max_modulus,
        cp,
    )
    prototypes = _process_state(prototype_raw, process_relations, cp)
    process_prediction = prediction[:, :-1]
    distance = cp.sum(
        (process_prediction[:, None, :] - prototypes[None, :, :]) ** 2,
        axis=2,
    )
    selected = cp.argmin(distance, axis=1)
    positions = cp.asarray([row[0] for row in prototype_rows], dtype=cp.int64)[selected]
    moduli = cp.asarray([row[1] for row in prototype_rows], dtype=cp.int64)[selected]
    moves = cp.rint(prediction[:, -1] * max(operand_limit - 1, 1)).astype(cp.int64)
    return positions, moduli, cp.clip(moves, 0, operand_limit - 1)


def _rollout(
    positions: Any,
    moduli: Any,
    moves: Any,
    process_relations: GrownRelations,
    transition_beta: Any,
    max_modulus: int,
    cp: Any,
) -> Any:
    current = positions.copy()
    classes = cp.arange(max_modulus)[None, :]
    for step in range(int(moves.max().get()) if len(moves) else 0):
        active = moves > step
        raw = _process_raw(current, moduli, max_modulus, cp)
        state = _process_state(raw, process_relations, cp)
        logits = state @ transition_beta[:-1] + transition_beta[-1]
        valid = classes < moduli[:, None]
        following = cp.argmax(cp.where(valid, logits, -cp.inf), axis=1)
        current = cp.where(active, following, current)
    return current


def _evaluate_bridge(
    split: ModularSplit,
    symbol_relations: GrownRelations,
    integration_relations: GrownRelations,
    process_relations: GrownRelations,
    bridge_beta: Any,
    transition_beta: Any,
    max_modulus: int,
    operand_limit: int,
    cp: Any,
    symbol_topology: str = "mixed_trace",
    bridge_target: str = "process_activity",
    execution_protocol: str = "reduced_start",
) -> dict[str, float | int]:
    symbol = _symbol_state(
        _symbol_features(split, cp, symbol_topology), symbol_relations, cp
    )
    integration = _integration_state(symbol, integration_relations, cp)
    cue_prediction = integration @ bridge_beta[:-1] + bridge_beta[-1]
    position, modulus, moves = _decode_cues(
        cue_prediction,
        process_relations,
        max_modulus,
        operand_limit,
        cp,
        bridge_target,
        execution_protocol,
    )
    answer = _rollout(
        position,
        modulus,
        moves,
        process_relations,
        transition_beta,
        max_modulus,
        cp,
    )
    truth = cp.asarray(split.residues, dtype=cp.int64)
    if execution_protocol == "origin_two_phase":
        true_start = cp.zeros(len(split.operands), dtype=cp.int64)
        true_moves = cp.asarray(split.operands.sum(axis=1))
    else:
        true_start = cp.asarray(np.mod(split.operands[:, 0], split.moduli))
        true_moves = cp.asarray(split.operands[:, 1])
    true_modulus = cp.asarray(split.moduli)
    return {
        "examples": len(split.residues),
        "exact_accuracy": float(cp.mean(answer == truth).get()),
        "start_reinstatement_accuracy": float(cp.mean(position == true_start).get()),
        "modulus_reinstatement_accuracy": float(cp.mean(modulus == true_modulus).get()),
        "movement_reinstatement_accuracy": float(cp.mean(moves == true_moves).get()),
        "complete_cue_reinstatement_accuracy": float(
            cp.mean(
                (position == true_start)
                & (modulus == true_modulus)
                & (moves == true_moves)
            ).get()
        ),
    }


def _oracle_process_accuracy(
    split: ModularSplit,
    process_relations: GrownRelations,
    transition_beta: Any,
    max_modulus: int,
    cp: Any,
    execution_protocol: str = "reduced_start",
) -> float:
    if execution_protocol == "origin_two_phase":
        position = cp.zeros(len(split.operands), dtype=cp.int64)
        moves = cp.asarray(split.operands.sum(axis=1))
    else:
        position = cp.asarray(np.mod(split.operands[:, 0], split.moduli))
        moves = cp.asarray(split.operands[:, 1])
    modulus = cp.asarray(split.moduli)
    answer = _rollout(
        position,
        modulus,
        moves,
        process_relations,
        transition_beta,
        max_modulus,
        cp,
    )
    return float(cp.mean(answer == cp.asarray(split.residues)).get())


def _evaluate_direct_endpoint(
    split: ModularSplit,
    symbol_relations: GrownRelations,
    integration_relations: GrownRelations,
    beta: Any,
    max_modulus: int,
    cp: Any,
    symbol_topology: str = "mixed_trace",
) -> float:
    symbol = _symbol_state(
        _symbol_features(split, cp, symbol_topology), symbol_relations, cp
    )
    state = _integration_state(symbol, integration_relations, cp)
    logits = state @ beta[:-1] + beta[-1]
    classes = cp.arange(max_modulus)[None, :]
    valid = classes < cp.asarray(split.moduli)[:, None]
    prediction = cp.argmax(cp.where(valid, logits, -cp.inf), axis=1)
    return float(cp.mean(prediction == cp.asarray(split.residues)).get())


def run_seed(args: argparse.Namespace, seed: int, cp: Any) -> dict[str, Any]:
    ladder = build_modular_ladder(
        seen_moduli=tuple(args.seen_moduli),
        unseen_modulus=args.unseen_modulus,
        operand_limit=args.operand_limit,
        seed=1_000_003 * (seed + 1),
        partition_by_operand_pair=args.split_granularity == "operand_pair",
    )
    max_modulus = max(*ladder.seen_moduli, ladder.unseen_modulus_value)
    started = time.perf_counter()

    transition_raw, transition_target = _transition_examples(
        ladder.seen_moduli, max_modulus, cp
    )
    process_relations = grow_relations(
        transition_raw,
        _growth_args(args, args.process_capacity),
        seed + 101,
        cp,
    )
    transition_state = _process_state(transition_raw, process_relations, cp)
    transition_beta, transition_alpha, transition_mse = _select_ridge(
        transition_state,
        transition_target,
        transition_state,
        transition_target,
        cp,
    )
    transition_rng = cp.random.RandomState(seed + 151)
    shuffled_transition_target = transition_target[
        transition_rng.permutation(len(transition_target))
    ]
    shuffled_transition_beta, _, _ = _select_ridge(
        transition_state,
        shuffled_transition_target,
        transition_state,
        transition_target,
        cp,
    )

    train_symbol_raw = _symbol_features(ladder.train, cp, args.symbol_topology)
    symbol_relations = grow_relations(
        train_symbol_raw,
        _growth_args(args, args.symbol_capacity),
        seed + 211,
        cp,
    )
    train_symbol = _symbol_state(train_symbol_raw, symbol_relations, cp)
    candidate_symbol = _symbol_state(
        _symbol_features(ladder.candidate, cp, args.symbol_topology),
        symbol_relations,
        cp,
    )
    integration_relations = grow_relations(
        train_symbol,
        _growth_args(args, args.integration_capacity),
        seed + 307,
        cp,
    )
    train_integration = _integration_state(
        train_symbol, integration_relations, cp
    )
    candidate_integration = _integration_state(
        candidate_symbol, integration_relations, cp
    )
    train_cues = _cue_targets(
        ladder.train,
        process_relations,
        max_modulus,
        args.operand_limit,
        cp,
        args.bridge_target,
        args.execution_protocol,
    )
    candidate_cues = _cue_targets(
        ladder.candidate,
        process_relations,
        max_modulus,
        args.operand_limit,
        cp,
        args.bridge_target,
        args.execution_protocol,
    )
    bridge_beta, bridge_alpha, bridge_candidate_mse = _select_ridge(
        train_integration,
        train_cues,
        candidate_integration,
        candidate_cues,
        cp,
    )

    train_residue = cp.eye(max_modulus, dtype=cp.float32)[
        cp.asarray(ladder.train.residues)
    ]
    candidate_residue = cp.eye(max_modulus, dtype=cp.float32)[
        cp.asarray(ladder.candidate.residues)
    ]
    direct_beta, direct_alpha, direct_candidate_mse = _select_ridge(
        train_integration,
        train_residue,
        candidate_integration,
        candidate_residue,
        cp,
    )

    rng = cp.random.RandomState(seed + 401)
    shuffled_cues = train_cues[rng.permutation(len(train_cues))]
    shuffled_beta, shuffled_alpha, _ = _select_ridge(
        train_integration,
        shuffled_cues,
        candidate_integration,
        candidate_cues,
        cp,
    )
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started

    body_bytes = int(
        _relation_bytes(process_relations)
        + _relation_bytes(symbol_relations)
        + _relation_bytes(integration_relations)
        + transition_beta.nbytes
        + bridge_beta.nbytes
    )
    if body_bytes > args.max_body_bytes:
        raise MemoryError("grown composition organism exceeds ALM-64MB ceiling")

    def evaluate(split: ModularSplit, beta: Any) -> dict[str, Any]:
        return _evaluate_bridge(
            split,
            symbol_relations,
            integration_relations,
            process_relations,
            beta,
            transition_beta,
            max_modulus,
            args.operand_limit,
            cp,
            args.symbol_topology,
            args.bridge_target,
            args.execution_protocol,
        )

    held_out = evaluate(ladder.held_out_pairs, bridge_beta)
    held_out_shuffled = evaluate(ladder.held_out_pairs, shuffled_beta)
    unseen = evaluate(ladder.unseen_modulus, bridge_beta)
    held_out_shuffled_process = _evaluate_bridge(
        ladder.held_out_pairs,
        symbol_relations,
        integration_relations,
        process_relations,
        bridge_beta,
        shuffled_transition_beta,
        max_modulus,
        args.operand_limit,
        cp,
        args.symbol_topology,
        args.bridge_target,
        args.execution_protocol,
    )
    per_modulus = {}
    for n in ladder.seen_moduli:
        mask = ladder.held_out_pairs.moduli == n
        subset = ModularSplit(
            ladder.held_out_pairs.inputs[mask],
            ladder.held_out_pairs.residues[mask],
            ladder.held_out_pairs.moduli[mask],
            ladder.held_out_pairs.operands[mask],
        )
        per_modulus[str(n)] = evaluate(subset, bridge_beta)
    return {
        "seed": seed,
        "training": evaluate(ladder.train, bridge_beta),
        "candidate": evaluate(ladder.candidate, bridge_beta),
        "held_out_operand_combinations": held_out,
        "held_out_shuffled_synchronization": held_out_shuffled,
        "held_out_shuffled_process_transition": held_out_shuffled_process,
        "held_out_oracle_process_accuracy": _oracle_process_accuracy(
            ladder.held_out_pairs,
            process_relations,
            transition_beta,
            max_modulus,
            cp,
            args.execution_protocol,
        ),
        "held_out_direct_symbol_accuracy": _evaluate_direct_endpoint(
            ladder.held_out_pairs,
            symbol_relations,
            integration_relations,
            direct_beta,
            max_modulus,
            cp,
            args.symbol_topology,
        ),
        "held_out_by_modulus": per_modulus,
        "unseen_modulus": unseen,
        "unseen_modulus_chance": 1.0 / ladder.unseen_modulus_value,
        "process_transition_ridge": transition_alpha,
        "process_transition_mse": transition_mse,
        "bridge_ridge": bridge_alpha,
        "bridge_candidate_mse": bridge_candidate_mse,
        "direct_symbol_ridge": direct_alpha,
        "direct_symbol_candidate_mse": direct_candidate_mse,
        "shuffled_bridge_ridge": shuffled_alpha,
        "grown_body_bytes": body_bytes,
        "organ_bytes": {
            "process": _relation_bytes(process_relations) + int(transition_beta.nbytes),
            "symbol": _relation_bytes(symbol_relations),
            "integration": _relation_bytes(integration_relations)
            + int(bridge_beta.nbytes),
        },
        "development_seconds": seconds,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in (
        "held_out_operand_combinations",
        "held_out_shuffled_synchronization",
        "held_out_shuffled_process_transition",
        "unseen_modulus",
    ):
        values = np.asarray([row[group]["exact_accuracy"] for row in rows])
        result[group] = {
            "mean_exact_accuracy": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    oracle = np.asarray([row["held_out_oracle_process_accuracy"] for row in rows])
    result["held_out_oracle_process_accuracy"] = {
        "mean": float(oracle.mean()),
        "min": float(oracle.min()),
        "max": float(oracle.max()),
    }
    direct = np.asarray([row["held_out_direct_symbol_accuracy"] for row in rows])
    result["held_out_direct_symbol_accuracy"] = {
        "mean": float(direct.mean()),
        "min": float(direct.min()),
        "max": float(direct.max()),
        "std": float(direct.std()),
    }
    return result


def main() -> None:
    args = parse_args()
    cp = _cupy()
    rows = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(
            f"three-organ modular composition seed {seed}",
            file=sys.stderr,
            flush=True,
        )
        rows.append(run_seed(args, seed, cp))
    report = {
        "experiment": "alm_64mb_1c_three_organ_modular_composition",
        "alm_name": "ALM-64MB-1C",
        "organs": ["cyclic_process", "symbol_stream", "integration"],
        "input_contract": (
            "equations arrive as raw ASCII receptor events; the process organ sees "
            "only cyclic positions and ring identity"
        ),
        "association_contract": (
            "the integration association restores process start state and movement "
            "count from synchronized organ activity; it never receives the residue"
        ),
        "execution_contract": (
            "the independently learned one-step process transition is recurrently "
            f"applied after restoration using {args.execution_protocol}"
        ),
        "controls": (
            "held-out operand combinations, completely unseen modulus, shuffled "
            "cross-organ synchronization, shuffled process transitions, and oracle "
            "process cues"
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
