"""Measure signed query-to-memory learning across delay and repetition."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import _predict
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import CASES
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _prime,
    _score_suffix,
)


def _correct_observations(
    arm: object, expected: bytes
) -> tuple[
    tuple[tuple[int, int], tuple[int, ...], tuple[tuple[int, int], ...]], ...
]:
    words = tuple(word.lower() for word in expected.strip().split())
    observations = []
    for word in words:
        candidates = [
            item
            for item in arm.field.output_items
            if 0 <= item[0] < len(arm.learner.word_forms)
            and arm.learner.word_forms[item[0]].strip().lower() == word
        ]
        if not candidates:
            continue
        item = min(candidates, key=lambda value: (value[1], value[0]))
        engrams = arm.field.engrams_containing(item)
        if not engrams:
            continue
        support = arm.field.engram_cortical_support(
            arm.pending_access_context, engrams
        )
        winner = max(engrams, key=lambda engram: (support[engram], -engram))
        observations.append((item, (winner,), ()))
    return tuple(observations)


def _access_diagnostics(
    arm: object,
    query_code: np.ndarray,
    correct_items: set[tuple[int, int]],
) -> dict[str, object]:
    signal = arm.relational_access.activate(
        query_code, len(arm.field.engram_units)
    )
    rows = [
        (item, value)
        for content in signal.content_support.values()
        for item, value in content.items()
    ]
    total = sum(value for _, value in rows)
    correct = sum(value for item, value in rows if item in correct_items)
    ranked = sorted(rows, key=lambda row: (-row[1], row[0]))
    correct_ranks = [
        index + 1
        for index, (item, _) in enumerate(ranked)
        if item in correct_items
    ]
    return {
        "active_rows": len(rows),
        "total_support": total,
        "correct_support": correct,
        "correct_share": correct / total if total > 0.0 else 0.0,
        "best_correct_rank": min(correct_ranks, default=None),
        "active_items": [
            {
                "form": arm.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": item[1],
                "support": value,
                "correct": item in correct_items,
            }
            for item, value in ranked[:16]
        ],
    }


def run(
    state: Path,
    data_path: Path,
    *,
    development_bytes: int,
    delays: tuple[int, ...],
    repetitions: tuple[int, ...],
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    stream = data_path.read_bytes()[:development_bytes]
    developed = phase85._prepare(
        payload, stream, access_learning_mode="prediction_error"
    )
    case = CASES[0]
    prepared = _prime(developed, case.prompt.encode(), return_mode="direct")
    # Form the actual query without observing its future answer.
    _predict(prepared, live=False)
    query_code = prepared.field.cortical_code(prepared.pending_access_context)
    observations = _correct_observations(prepared, case.expected.encode())
    correct_items = {item for item, _, _ in observations}
    if not observations:
        raise RuntimeError("controlled identity is absent from episodic tissue")
    rows = []
    for delay in delays:
        for repeat in repetitions:
            arm = pickle.loads(
                pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL)
            )
            for _ in range(repeat):
                arm.relational_access.reset_episode()
                arm.relational_access.propose(query_code)
                for _ in range(delay):
                    arm.relational_access.advance()
                arm.relational_access.confirm_many(
                    observations, external=True
                )
            arm.relational_access.reset_episode()
            arm.relational_access.propose(query_code)
            diagnostics = _access_diagnostics(arm, query_code, correct_items)
            off_bits = _score_suffix(
                arm, case.expected.encode(), recall=False
            )
            on_bits = _score_suffix(
                arm, case.expected.encode(), recall=True
            )
            rows.append(
                {
                    "delay_steps": delay,
                    "repetitions": repeat,
                    "saved_bits": off_bits - on_bits,
                    "off_bits": off_bits,
                    "on_bits": on_bits,
                    "access": diagnostics,
                }
            )
    return {
        "task": "phase98_signed_access_delay_repetition_curve",
        "state": str(state),
        "data_path": str(data_path),
        "development_bytes": development_bytes,
        "controlled_case": case.name,
        "correct_items": [
            {
                "form": prepared.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": item[1],
                "engram": engrams[0],
            }
            for item, engrams, _ in observations
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--development-bytes", type=int, default=512)
    parser.add_argument("--delays", type=int, nargs="+", default=(0, 4, 16, 64))
    parser.add_argument(
        "--repetitions", type=int, nargs="+", default=(0, 1, 4, 16)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.data,
        development_bytes=args.development_bytes,
        delays=tuple(args.delays),
        repetitions=tuple(args.repetitions),
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
