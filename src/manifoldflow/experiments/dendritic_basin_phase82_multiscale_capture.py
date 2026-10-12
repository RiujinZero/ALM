"""Test whether overlapping cortical assemblies survive episodic capture."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.multiscale_assembly import MultiscaleAssemblyField
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _new_arm,
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    GenerationCase,
)
from manifoldflow.experiments.dendritic_basin_phase80_expression_ceiling import (
    _common_prefix,
    _first_word,
)
from manifoldflow.experiments.dendritic_basin_phase81_cortical_expression import (
    _calibrated_path,
    _evaluate,
    _natural_signal,
)


def _build_multiscale(
    learner: object, *, trace_half_life_bytes: float = 32.0
) -> MultiscaleAssemblyField:
    field = MultiscaleAssemblyField(
        activation_threshold=1.0,
        maximum_active=64,
        trace_half_life_bytes=trace_half_life_bytes,
    )
    for assembly, form in enumerate(learner.word_forms):
        field.bind(assembly, form)
    return field


def _prime(
    payload: dict[str, object],
    multiscale: MultiscaleAssemblyField,
    prefix: bytes,
    *,
    frames_per_capsule: int,
    items_per_frame: int,
) -> tuple[object, set[int]]:
    arm = _new_arm(payload)
    arm.event_memory.frame_compaction = "diverse"
    arm.event_memory.frames_per_capsule = frames_per_capsule
    arm.event_memory.items_per_frame = items_per_frame
    multiscale.reset_episode()
    recognized: set[int] = set()
    for byte_id in prefix:
        distribution = _predict(arm, live=False)
        activations = multiscale.observe(byte_id)
        recognized.update(item.assembly for item in activations)
        extra = tuple(
            (
                0,
                item.assembly,
                len(arm.learner.word_forms[item.assembly]) * item.support,
            )
            for item in activations
        )
        _observe(
            arm,
            byte_id,
            float(distribution[byte_id]),
            extra_active_items=extra,
        )
    return arm, recognized


def _run_case(
    payload: dict[str, object],
    multiscale: MultiscaleAssemblyField,
    case: GenerationCase,
    *,
    frames_per_capsule: int,
    items_per_frame: int,
    expression_gain: float,
) -> dict[str, object]:
    expected = case.expected.encode()
    contrast = case.contrast.encode()
    shared = _common_prefix(expected, contrast)
    expected_word = _first_word(expected[len(shared) :])
    contrast_word = _first_word(contrast[len(shared) :])
    learner = payload["learner"]
    expected_id = learner.word_ids.get(expected_word)
    contrast_id = learner.word_ids.get(contrast_word)
    if expected_id is None or contrast_id is None:
        return {"case": asdict(case), "available": False}
    arm, recognized = _prime(
        payload,
        multiscale,
        case.prompt.encode() + shared,
        frames_per_capsule=frames_per_capsule,
        items_per_frame=items_per_frame,
    )
    correct = int(expected_id), 0
    swapped = int(contrast_id), 0
    path = _calibrated_path(learner, (correct, swapped), repetitions=24)
    _, signal = _natural_signal(
        pickle.loads(pickle.dumps(arm, protocol=pickle.HIGHEST_PROTOCOL))
    )
    returned = () if signal is None else signal.items
    correct_bound = bool(arm.field.engrams_containing(correct))
    oracle = _evaluate(
        arm,
        path,
        target=expected_word,
        contrast=contrast_word,
        mode="oracle",
        item=correct,
        gain=expression_gain,
        generate=True,
    )
    natural = _evaluate(
        arm,
        path,
        target=expected_word,
        contrast=contrast_word,
        mode="natural",
        item=None,
        gain=expression_gain,
        generate=True,
    )
    return {
        "case": asdict(case),
        "available": True,
        "expected_identity": expected_word.decode(),
        "target_recognized_during_prompt": int(expected_id) in recognized,
        "target_bound_to_event_field": correct_bound,
        "natural_first_return": [
            {
                "assembly": assembly,
                "receptor": receptor,
                "form": arm.learner.word_forms[assembly].decode(
                    "utf-8", errors="backslashreplace"
                ),
            }
            for assembly, receptor in returned
        ],
        "natural_returns_target": correct in returned,
        "oracle_expression": oracle,
        "natural_expression": natural,
        "captures": arm.event_memory.captures,
        "field_output_items": len(arm.field.output_items),
    }


def run_probe(
    state: Path,
    *,
    frames_per_capsule: int = 24,
    items_per_frame: int = 16,
    expression_gain: float = 0.5,
    identity_receptor_limit: int = -1,
    trace_half_life_bytes: float = 32.0,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    payload["field"].sequence_settling_steps = 4
    payload["field"].sequence_compatibility_gain = 2.0
    payload["field"].sequence_stop_compatibility = 0.5
    payload["identity_context"] = True
    payload["identity_receptor_limit"] = identity_receptor_limit
    multiscale = _build_multiscale(
        payload["learner"], trace_half_life_bytes=trace_half_life_bytes
    )
    results = [
        _run_case(
            payload,
            multiscale,
            case,
            frames_per_capsule=frames_per_capsule,
            items_per_frame=items_per_frame,
            expression_gain=expression_gain,
        )
        for case in CASES
    ]
    available = [result for result in results if result["available"]]
    return {
        "task": "phase82_multiscale_event_capture",
        "diagnostic_only": True,
        "state": str(state),
        "frames_per_capsule": frames_per_capsule,
        "items_per_frame": items_per_frame,
        "expression_gain": expression_gain,
        "identity_receptor_limit": identity_receptor_limit,
        "trace_half_life_bytes": trace_half_life_bytes,
        "multiscale_body_bytes": multiscale.embodied_bytes,
        "summary": {
            "available_cases": len(available),
            "recognized_fraction": float(
                np.mean(
                    [result["target_recognized_during_prompt"] for result in available]
                )
            )
            if available
            else 0.0,
            "bound_fraction": float(
                np.mean([result["target_bound_to_event_field"] for result in available])
            )
            if available
            else 0.0,
            "natural_target_return_fraction": float(
                np.mean([result["natural_returns_target"] for result in available])
            )
            if available
            else 0.0,
            "oracle_exact_fraction": float(
                np.mean([result["oracle_expression"]["exact"] for result in available])
            )
            if available
            else 0.0,
            "natural_exact_fraction": float(
                np.mean([result["natural_expression"]["exact"] for result in available])
            )
            if available
            else 0.0,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--frames-per-capsule", type=int, default=24)
    parser.add_argument("--items-per-frame", type=int, default=16)
    parser.add_argument("--expression-gain", type=float, default=0.5)
    parser.add_argument("--identity-receptor-limit", type=int, default=-1)
    parser.add_argument("--trace-half-life-bytes", type=float, default=32.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        args.state,
        frames_per_capsule=args.frames_per_capsule,
        items_per_frame=args.items_per_frame,
        expression_gain=args.expression_gain,
        identity_receptor_limit=args.identity_receptor_limit,
        trace_half_life_bytes=args.trace_half_life_bytes,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
