"""Test local terminal-phase release through learned episodic successors."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path

from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import CASES
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _prime,
)
from manifoldflow.experiments.dendritic_basin_phase99_expression_ceiling import (
    _schedule,
)


def _learned_chain(
    arm: object,
    engram: int,
    pool: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Let asymmetric local successor current order the retrieved pool."""
    transitions = arm.field.engram_sequence_weights[engram]
    pool_set = set(pool)
    entry_drive = {}
    for item in pool:
        outgoing = sum(
            transitions.get(item, {}).get(other, 0.0)
            for other in pool_set - {item}
        )
        incoming = sum(
            transitions.get(other, {}).get(item, 0.0)
            for other in pool_set - {item}
        )
        entry_drive[item] = outgoing - incoming
    current = max(pool, key=lambda item: (entry_drive[item], -item[0], -item[1]))
    chain = [current]
    remaining = pool_set - {current}
    while remaining:
        row = transitions.get(current, {})
        successor = max(
            remaining,
            key=lambda item: (row.get(item, 0.0), -item[0], -item[1]),
        )
        if row.get(successor, 0.0) <= 0.0:
            break
        chain.append(successor)
        remaining.remove(successor)
        current = successor
    return tuple(chain)


def _terminal(arm: object, item: tuple[int, int]) -> bool:
    active = arm.recalled_expression.active.get(item)
    if active is None:
        return False
    phase = active[1]
    key = item[0], item[1], phase
    return arm.recalled_expression.evidence.get(key, 0) < (
        arm.recalled_expression.minimum_evidence
    )


def _trajectory(
    prepared: object,
    expected: bytes,
    chain: tuple[tuple[int, int], ...],
    *,
    mode: str,
    generate: bool,
) -> dict[str, object]:
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.field.reset_episode()
    arm.relational_access.reset_episode()
    arm.recalled_expression.reset_episode()
    chain_index = 0
    emitted = bytearray()
    bits = 0.0
    margins = []
    handoffs = 0
    for expected_byte in expected:
        baseline = arm.learner.predict_next_distribution()
        if mode == "off":
            active = ()
        elif mode == "simultaneous":
            active = chain
        else:
            current = chain[chain_index]
            if _terminal(arm, current) and chain_index + 1 < len(chain):
                chain_index += 1
                current = chain[chain_index]
                handoffs += 1
            active = (current,)
        arm.recalled_expression.activate(active)
        expression = arm.recalled_expression.predict(
            baseline,
            maximum_gain=arm.expression_gain,
            shared_cancellation=arm.expression_cancellation_mode,
            maximum_current_norm=arm.expression_current_limit,
        )
        distribution = expression.distribution
        observed = int(distribution.argmax()) if generate else expected_byte
        emitted.append(observed)
        bits -= log2(max(float(distribution[expected_byte]), 1e-300))
        best_other = max(
            float(probability)
            for byte_id, probability in enumerate(distribution)
            if byte_id != expected_byte
        )
        margins.append(
            log2(max(float(distribution[expected_byte]), 1e-300))
            - log2(max(best_other, 1e-300))
        )
        _observe(
            arm,
            observed,
            float(distribution[observed]),
            external=False,
        )
    return {
        "mode": mode,
        "generate": generate,
        "bits": bits,
        "bpb": bits / len(expected),
        "positive_margins": sum(margin > 0.0 for margin in margins),
        "margins": margins,
        "emitted": emitted.decode("utf-8", errors="backslashreplace"),
        "exact": bytes(emitted) == expected,
        "handoffs": handoffs,
    }


def run(
    state: Path,
    data_path: Path,
    *,
    development_bytes: int,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    developed = phase85._prepare(
        payload,
        data_path.read_bytes()[:development_bytes],
        access_learning_mode="prediction_error",
    )
    case = CASES[0]
    # The ordinary cortex supplies the separator. Retrieval begins from the
    # resulting state, so no semantic timer is needed by the return pathway.
    prepared = _prime(
        developed, case.prompt.encode() + b" ", return_mode="direct"
    )
    _predict(prepared, live=False)
    query_code = prepared.field.cortical_code(prepared.pending_access_context)
    exact_items, _, _ = _schedule(prepared, case.expected.encode())
    common_engrams = set(prepared.field.engrams_containing(exact_items[0]))
    for item in exact_items[1:]:
        common_engrams.intersection_update(prepared.field.engrams_containing(item))
    if not common_engrams:
        raise RuntimeError("exact identity assemblies share no episodic engram")
    cortical_support = prepared.field.engram_cortical_support(
        prepared.pending_access_context, tuple(common_engrams)
    )
    taught_engram = max(
        common_engrams,
        key=lambda value: (cortical_support[value], -value),
    )
    observations = tuple(
        (item, (taught_engram,), ()) for item in exact_items
    )
    prepared.relational_access.reset_episode()
    prepared.relational_access.propose(query_code)
    prepared.relational_access.confirm_many(observations, external=True)
    signal = prepared.relational_access.activate(
        query_code, len(prepared.field.engram_units)
    )
    correct = {item for item, _, _ in observations}
    eligible = [
        (engram, tuple(item for item in row if item in correct))
        for engram, row in signal.content_support.items()
        if all(item in row for item in correct)
    ]
    if not eligible:
        raise RuntimeError("one-shot access did not recover the identity pool")
    engram, pool = max(
        eligible,
        key=lambda row: (
            sum(signal.content_support[row[0]][item] for item in row[1]),
            -row[0],
        ),
    )
    chain = _learned_chain(prepared, engram, pool)
    if len(chain) != len(pool):
        raise RuntimeError("learned episodic edges do not connect the access pool")
    expected = case.expected.encode().lstrip()
    results = []
    for mode in ("off", "simultaneous", "terminal_successor"):
        results.append(
            {
                "mode": mode,
                "forced": _trajectory(
                    prepared, expected, chain, mode=mode, generate=False
                ),
                "greedy": _trajectory(
                    prepared, expected, chain, mode=mode, generate=True
                ),
            }
        )
    return {
        "task": "phase100_local_terminal_successor_release",
        "state": str(state),
        "data_path": str(data_path),
        "development_bytes": development_bytes,
        "controlled_case": case.name,
        "oracle_scope": (
            "Evaluator supplies one correct external access consequence. "
            "Order and duration come only from learned episodic/expression tissue."
        ),
        "engram": engram,
        "pool": [
            prepared.learner.word_forms[item[0]].decode(
                "utf-8", errors="backslashreplace"
            )
            for item in pool
        ],
        "chain": [
            prepared.learner.word_forms[item[0]].decode(
                "utf-8", errors="backslashreplace"
            )
            for item in chain
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--development-bytes", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.data,
        development_bytes=args.development_bytes,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
