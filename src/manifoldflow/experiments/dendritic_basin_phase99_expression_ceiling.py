"""Test whether phase-correct serialized recall can express a known identity."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path

from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import _observe
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import CASES
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _prime,
)


def _item_for_form(arm: object, form: bytes) -> tuple[int, int]:
    candidates = [
        item
        for item in arm.field.output_items
        if 0 <= item[0] < len(arm.learner.word_forms)
        and arm.learner.word_forms[item[0]].lower() == form.lower()
    ]
    if not candidates:
        candidates = [
            item
            for item in arm.field.output_items
            if 0 <= item[0] < len(arm.learner.word_forms)
            and arm.learner.word_forms[item[0]].strip().lower()
            == form.strip().lower()
        ]
    if not candidates:
        raise RuntimeError(f"no identity assembly found for {form!r}")
    return min(candidates, key=lambda item: (item[1], item[0]))


def _schedule(
    arm: object, phrase: bytes
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], int]:
    words = phrase.strip().split()
    forms = tuple(
        word + (b" " if index < len(words) - 1 else b"")
        for index, word in enumerate(words)
    )
    items = tuple(_item_for_form(arm, form) for form in forms)
    lengths = tuple(len(arm.learner.word_forms[item[0]]) for item in items)
    prefix = len(phrase) - sum(lengths)
    if prefix < 0 or phrase[:prefix] + b"".join(
        arm.learner.word_forms[item[0]] for item in items
    ) != phrase:
        raise RuntimeError(f"identity assemblies do not tile {phrase!r}")
    return items, lengths, prefix


def _active_items(
    mode: str,
    items: tuple[tuple[int, int], ...],
    lengths: tuple[int, ...],
    prefix: int,
    position: int,
) -> tuple[tuple[int, int], ...]:
    if mode == "off":
        return ()
    if mode == "simultaneous":
        return items
    if position < prefix:
        return ()
    cursor = prefix
    for item, length in zip(items, lengths, strict=True):
        if position < cursor + length:
            return (item,)
        cursor += length
    return ()


def _run_trajectory(
    prepared: object,
    phrase: bytes,
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
    items, lengths, prefix = _schedule(arm, phrase)
    emitted = bytearray()
    bits = 0.0
    margins = []
    for position, expected_byte in enumerate(phrase):
        baseline = arm.learner.predict_next_distribution()
        active = _active_items(mode, items, lengths, prefix, position)
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
        "scored_phrase": phrase.decode("utf-8", errors="backslashreplace"),
        "bits": bits,
        "bpb": bits / len(phrase),
        "positive_margins": sum(margin > 0.0 for margin in margins),
        "margins": margins,
        "emitted": emitted.decode("utf-8", errors="backslashreplace"),
        "exact": bytes(emitted) == phrase,
        "items": [
            {
                "form": arm.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": item[1],
                "duration": length,
            }
            for item, length in zip(items, lengths, strict=True)
        ],
    }


def run(
    state: Path,
    data_path: Path,
    *,
    development_bytes: int,
    case_index: int = 0,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    stream = data_path.read_bytes()[:development_bytes]
    developed = phase85._prepare(
        payload, stream, access_learning_mode="prediction_error"
    )
    case = CASES[case_index]
    prepared = _prime(developed, case.prompt.encode(), return_mode="direct")
    results = []
    for phrase_name, phrase in (("expected", case.expected.encode()),):
        for mode in ("off", "simultaneous", "serialized"):
            try:
                forced = _run_trajectory(
                    prepared, phrase, mode=mode, generate=False
                )
                generated = _run_trajectory(
                    prepared, phrase, mode=mode, generate=True
                )
            except RuntimeError as error:
                results.append(
                    {
                        "phrase": phrase_name,
                        "mode": mode,
                        "available": False,
                        "error": str(error),
                    }
                )
                continue
            results.append(
                {
                    "phrase": phrase_name,
                    "mode": mode,
                    "available": True,
                    "forced": forced,
                    "greedy": generated,
                }
            )
    return {
        "task": "phase99_phase_correct_expression_ceiling",
        "state": str(state),
        "data_path": str(data_path),
        "development_bytes": development_bytes,
        "case_index": case_index,
        "controlled_case": case.name,
        "oracle_scope": (
            "Evaluator supplies known content order and handoff duration; "
            "this is an expression ceiling, not an online ALM mechanism."
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--development-bytes", type=int, default=512)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)), default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.data,
        development_bytes=args.development_bytes,
        case_index=args.case_index,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
