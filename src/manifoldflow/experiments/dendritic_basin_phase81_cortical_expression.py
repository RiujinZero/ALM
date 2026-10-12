"""Test locally learned cortical expression of a settled episodic assembly.

Phase 80 showed that the old receptor route projects what follows a recalled
motif and therefore reaches identity generation one cortical phase too late.
This diagnostic leaves memory selection and settling frozen.  It calibrates a
sparse receiving-cortex pathway from already observed motif assemblies and
their sensory traces, then asks whether a correct assembly can express its own
ordered bytes.  Replay calibration is diagnostic and is not counted as online
learning or as an ALM score.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.recalled_expression import (
    RecalledCorticalExpression,
    RecallItem,
)
from manifoldflow.experiments.dendritic_basin_phase73_shadow_capsules import (
    _content_match,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _new_arm,
    _observe,
)
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _sensory_context,
    _temporal_cortical_context,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    GenerationCase,
)
from manifoldflow.experiments.dendritic_basin_phase80_expression_ceiling import (
    _best_route_to,
    _common_prefix,
    _first_word,
    _margin,
)


def _prime(payload: dict[str, object], prefix: bytes) -> object:
    arm = _new_arm(payload)
    for byte_id in prefix:
        off = arm.learner.predict_next_distribution()
        _observe(arm, byte_id, float(off[byte_id]))
    return arm


def _natural_signal(arm: object) -> tuple[np.ndarray, object | None]:
    """Run the frozen Phase 79 field and expose its pre-cortical return."""
    off = arm.learner.predict_next_distribution()
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    context = _temporal_cortical_context(
        tuple(arm.route_history),
        include_identity=getattr(arm, "identity_context", False),
        identity_receptor_limit=getattr(arm, "identity_receptor_limit", 8),
    ) + _sensory_context(
        int.from_bytes(arm.raw_window, "big") if arm.raw_window else None
    )
    sequence_cue_items = (
        arm.route_history[-1]
        if getattr(arm, "identity_context", False) and arm.route_history
        else tuple(
            (assembly, receptor)
            for receptor, assembly in arm.learner._active_route_keys()
        )
    )
    compatibility = {
        item: _content_match(arm.learner, item) for item in arm.field.output_items
    }
    completion = arm.field.retrieve(
        context,
        context_id=basin,
        sequence_cue_items=sequence_cue_items,
        content_gain=compatibility,
    )
    if completion is not None and completion.items:
        arm.field.reinstate(completion)
    return off, arm.field.return_signal(completion)


def _calibrated_path(
    learner: object,
    items: tuple[RecallItem, ...],
    *,
    repetitions: int,
) -> RecalledCorticalExpression:
    path = RecalledCorticalExpression(minimum_evidence=1)
    for item in dict.fromkeys(items):
        assembly, _ = item
        if 0 <= assembly < len(learner.word_forms):
            path.learn_sequence(
                item, learner.word_forms[assembly], repetitions=repetitions
            )
    return path


def _incoming_observations(learner: object, item: RecallItem) -> int:
    target, receptor = item
    return sum(
        row.get(target, 0)
        for (route_receptor, _), row in learner.route_counts.items()
        if route_receptor == receptor
    )


def _evaluate(
    primed_arm: object,
    path: RecalledCorticalExpression,
    *,
    target: bytes,
    contrast: bytes,
    mode: str,
    item: RecallItem | None,
    gain: float,
    generate: bool,
) -> dict[str, object]:
    arm = pickle.loads(pickle.dumps(primed_arm, protocol=pickle.HIGHEST_PROTOCOL))
    path.reset_episode()
    emitted = bytearray()
    positions = []
    for index, target_byte in enumerate(target):
        recalled_items: tuple[RecallItem, ...] = ()
        if mode == "natural":
            off, signal = _natural_signal(arm)
            if signal is None:
                path.activate(())
            else:
                recalled_items = signal.items
                path.activate(signal.items, signal.item_strengths)
        else:
            off = arm.learner.predict_next_distribution()
            path.activate(()) if item is None else path.activate((item,))
        expression = path.predict(off, maximum_gain=gain)
        distribution = expression.distribution
        chosen = int(np.argmax(distribution)) if generate else int(target_byte)
        emitted.append(chosen)
        probability = max(float(distribution[int(target_byte)]), 1e-300)
        margin, rank, ties = _margin(distribution, int(target_byte))
        contrast_byte = int(contrast[index]) if index < len(contrast) else None
        positions.append(
            {
                "target_byte": int(target_byte),
                "chosen_byte": chosen,
                "target_probability": probability,
                "target_log2_gain_vs_off": log2(
                    probability / max(float(off[int(target_byte)]), 1e-300)
                ),
                "greedy_margin_bits": margin,
                "target_rank": rank,
                "winner_ties": ties,
                "contrast_margin_bits": (
                    None
                    if contrast_byte is None or contrast_byte == target_byte
                    else log2(probability)
                    - log2(max(float(distribution[contrast_byte]), 1e-300))
                ),
                "expression_gain": expression.gain,
                "active_expression_rows": expression.active_rows,
                "recalled_items": [
                    {
                        "assembly": assembly,
                        "receptor": receptor,
                        "form": (
                            arm.learner.word_forms[assembly].decode(
                                "utf-8", errors="backslashreplace"
                            )
                            if 0 <= assembly < len(arm.learner.word_forms)
                            else ""
                        ),
                    }
                    for assembly, receptor in recalled_items
                ],
            }
        )
        # A generated byte is allowed to become cortex's next sensory state,
        # but it is not an external teaching signal for expression synapses.
        path.observe(chosen, external=False)
        _observe(arm, chosen, float(distribution[chosen]))
    return {
        "generated": bytes(emitted).decode("utf-8", errors="backslashreplace"),
        "target": target.decode("utf-8", errors="backslashreplace"),
        "exact": bytes(emitted).lower() == target.lower(),
        "positions": positions,
    }


def _run_case(
    payload: dict[str, object],
    case: GenerationCase,
    *,
    gains: tuple[float, ...],
    calibration_repetitions: int,
) -> dict[str, object]:
    learner = payload["learner"]
    expected = case.expected.encode()
    contrast = case.contrast.encode()
    shared = _common_prefix(expected, contrast)
    expected_word = _first_word(expected[len(shared) :])
    contrast_word = _first_word(contrast[len(shared) :])
    expected_id = learner.word_ids.get(expected_word)
    contrast_id = learner.word_ids.get(contrast_word)
    if expected_id is None or contrast_id is None:
        return {"case": asdict(case), "available": False}
    expected_route = _best_route_to(learner, int(expected_id))
    if expected_route is None:
        return {"case": asdict(case), "available": False}
    receptor = int(expected_route[0])
    correct = int(expected_id), receptor
    swapped = int(contrast_id), receptor
    path = _calibrated_path(
        learner, (correct, swapped), repetitions=calibration_repetitions
    )
    prefix = case.prompt.encode() + shared
    primed_arm = _prime(payload, prefix)
    conditions = {}
    for gain in gains:
        gain_results = {}
        for name, mode, item in (
            ("off", "oracle", None),
            ("natural", "natural", None),
            ("correct_identity", "oracle", correct),
            ("identity_swapped", "oracle", swapped),
        ):
            gain_results[name] = {
                "teacher_forced": _evaluate(
                    primed_arm,
                    path,
                    target=expected_word,
                    contrast=contrast_word,
                    mode=mode,
                    item=item,
                    gain=0.0 if name == "off" else gain,
                    generate=False,
                ),
                "greedy": _evaluate(
                    primed_arm,
                    path,
                    target=expected_word,
                    contrast=contrast_word,
                    mode=mode,
                    item=item,
                    gain=0.0 if name == "off" else gain,
                    generate=True,
                ),
            }
        conditions[str(gain)] = gain_results
    return {
        "case": asdict(case),
        "available": True,
        "expected_identity": expected_word.decode(),
        "contrast_identity": contrast_word.decode(),
        "correct_item": correct,
        "swapped_item": swapped,
        "correct_incoming_observations": _incoming_observations(learner, correct),
        "swapped_incoming_observations": _incoming_observations(learner, swapped),
        "expression_body_bytes": path.embodied_bytes,
        "conditions": conditions,
    }


def run_probe(
    state: Path,
    *,
    cases: tuple[GenerationCase, ...] = CASES,
    gains: tuple[float, ...] = (0.02, 0.1, 0.25, 0.5, 1.0),
    calibration_repetitions: int = 24,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    payload["field"].sequence_settling_steps = 4
    payload["field"].sequence_compatibility_gain = 2.0
    payload["field"].sequence_stop_compatibility = 0.5
    results = [
        _run_case(
            payload,
            case,
            gains=gains,
            calibration_repetitions=calibration_repetitions,
        )
        for case in cases
    ]
    available = [result for result in results if result["available"]]
    summary = {}
    for gain in gains:
        key = str(gain)
        summary[key] = {}
        for condition in ("off", "natural", "correct_identity", "identity_swapped"):
            rows = [result["conditions"][key][condition] for result in available]
            first = [row["teacher_forced"]["positions"][0] for row in rows]
            summary[key][condition] = {
                "cases": len(rows),
                "mean_first_byte_margin_bits": (
                    float(np.mean([row["greedy_margin_bits"] for row in first]))
                    if first
                    else 0.0
                ),
                "mean_first_byte_rank": (
                    float(np.mean([row["target_rank"] for row in first]))
                    if first
                    else 0.0
                ),
                "mean_target_log2_gain_vs_off": (
                    float(np.mean([row["target_log2_gain_vs_off"] for row in first]))
                    if first
                    else 0.0
                ),
                "exact_greedy_identity_fraction": (
                    float(np.mean([row["greedy"]["exact"] for row in rows]))
                    if rows
                    else 0.0
                ),
            }
    return {
        "task": "phase81_local_cortical_expression",
        "diagnostic_only": True,
        "state": str(state),
        "gains": gains,
        "calibration_repetitions": calibration_repetitions,
        "calibration_uses_only_checkpoint_motif_forms": True,
        "persistent_learning_during_evaluation": False,
        "self_sampled_bytes_teach_expression": False,
        "results": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--calibration-repetitions", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (CASES[args.case_index],) if args.case_index is not None else CASES
    report = run_probe(
        args.state,
        cases=cases,
        calibration_repetitions=args.calibration_repetitions,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
