"""Sweep surprise-opened, first-consequence-closed access plateaus."""

from __future__ import annotations

import argparse
import json
import pickle
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
    _clear_fast_state,
    _prime,
)
from manifoldflow.experiments.dendritic_basin_phase99_expression_ceiling import (
    _schedule,
)


def _teach_answer(arm: object, answer: bytes) -> list[float]:
    boundary = []
    for byte_id in answer:
        distribution = _predict(arm, live=False)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)
        boundary.append(arm.last_boundary_signal)
    return boundary


def _probe(learned: object, prompt: bytes) -> dict[str, object]:
    arm = pickle.loads(pickle.dumps(learned, protocol=pickle.HIGHEST_PROTOCOL))
    _clear_fast_state(arm)
    arm.recall_enabled = True
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.access_return_mode = "direct"
    arm.access_recruitment_mode = "none"
    for byte_id in prompt:
        distribution = _predict(arm, live=False)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)
    _predict(arm, live=False)
    code = arm.field.cortical_code(arm.pending_access_context)
    signal = arm.relational_access.activate(code, len(arm.field.engram_units))
    rows = sorted(
        (
            (strength, item, engram)
            for engram, content in signal.content_support.items()
            for item, strength in content.items()
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    return {
        "active_rows": len(rows),
        "rows": [
            {
                "support": strength,
                "form": arm.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": item[1],
                "engram": engram,
            }
            for strength, item, engram in rows[:32]
        ],
    }


def run(
    state: Path,
    data_path: Path,
    *,
    development_bytes: int,
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    developed = phase85._prepare(
        payload,
        data_path.read_bytes()[:development_bytes],
        access_learning_mode="prediction_error",
    )
    target = CASES[0]
    unrelated = CASES[2]
    rows = []
    for threshold in thresholds:
        organism = pickle.loads(
            pickle.dumps(developed, protocol=pickle.HIGHEST_PROTOCOL)
        )
        organism.relational_access.consume_on_confirmation = True
        organism.relational_access.latch_until_confirmation = True
        organism.access_proposal_threshold = threshold
        demonstration = _prime(
            organism,
            target.prompt.encode() + b" ",
            return_mode="direct",
        )
        exact_items, _, _ = _schedule(
            demonstration, target.expected.encode()
        )
        exact_forms = {
            demonstration.learner.word_forms[item[0]].decode(
                "utf-8", errors="backslashreplace"
            )
            for item in exact_items
        }
        answer_boundaries = _teach_answer(
            demonstration, target.expected.encode().lstrip()
        )
        repeated = _probe(demonstration, target.prompt.encode() + b" ")
        unrelated_probe = _probe(
            demonstration, unrelated.prompt.encode() + b" "
        )
        repeated_forms = {row["form"] for row in repeated["rows"]}
        rows.append(
            {
                "threshold": threshold,
                "access_confirmations": demonstration.access_confirmations,
                "answer_boundary_signals": answer_boundaries,
                "target": repeated,
                "target_exact_forms": sorted(exact_forms),
                "target_exact_recovered": sorted(
                    exact_forms & repeated_forms
                ),
                "unrelated": unrelated_probe,
            }
        )
    return {
        "task": "phase101_surprise_opened_access_plateau",
        "state": str(state),
        "data_path": str(data_path),
        "development_bytes": development_bytes,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--development-bytes", type=int, default=512)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0, 4.0)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.data,
        development_bytes=args.development_bytes,
        thresholds=tuple(args.thresholds),
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
