"""Localize whether settled ALM content can win cortical output competition.

This is an evaluator-supplied oracle diagnostic.  It never contributes to a
reported online score and never exposes a future byte to the learner before the
corresponding prediction.  Correct and identity-swapped returns use already
learned cortical assemblies and receptors at matched injected variation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.recurrent_episodic_field import FieldReturnSignal
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _new_arm,
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _receptor_route_distribution,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    GenerationCase,
)

_WORD = re.compile(rb"[A-Za-z]+")


def _first_word(value: bytes) -> bytes:
    match = _WORD.search(value)
    if match is None:
        raise ValueError("controlled continuation does not contain a word")
    return match.group(0).lower()


def _common_prefix(left: bytes, right: bytes) -> bytes:
    length = 0
    for left_byte, right_byte in zip(left, right, strict=False):
        if left_byte != right_byte:
            break
        length += 1
    return left[:length]


def _route_information(learner: object, key: tuple[int, int], target: int) -> float:
    probability = learner._route_probability(key, target)
    prior = learner._target_prior(target)
    return log2(max(probability, 1e-300) / max(prior, 1e-300))


def _best_route_to(learner: object, target: int) -> tuple[int, int] | None:
    candidates = [
        key
        for key, row in learner.route_counts.items()
        if row.get(target, 0) > 0 and key[1] != target
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda key: (
            _route_information(learner, key, target),
            learner.route_utility.get(key, 0.0),
            learner.route_totals.get(key, 0),
        ),
    )


def _strongest_native_route(
    learner: object, source: int, fallback_receptor: int
) -> tuple[int, int]:
    candidates = [key for key in learner.route_counts if key[1] == source]
    if not candidates:
        return fallback_receptor, source
    return max(
        candidates,
        key=lambda key: (
            learner.route_utility.get(key, 0.0),
            learner.route_totals.get(key, 0),
        ),
    )


def _phase_routes(
    learner: object, target: int
) -> dict[str, tuple[int, int] | None]:
    predecessor = _best_route_to(learner, target)
    if predecessor is None:
        return {"pre_pre_target": None, "pre_target": None, "post_target": None}
    receptor, source = predecessor
    pre_predecessor = _best_route_to(learner, source)
    post_target = _strongest_native_route(learner, target, receptor)
    return {
        "pre_pre_target": pre_predecessor,
        "pre_target": predecessor,
        "post_target": post_target,
    }


def _item_from_key(key: tuple[int, int] | None) -> tuple[int, int] | None:
    if key is None:
        return None
    receptor, source = key
    return source, receptor


def _route_description(
    learner: object, key: tuple[int, int] | None, target: int
) -> dict[str, object] | None:
    if key is None:
        return None
    receptor, source = key
    form = (
        learner.word_forms[source]
        if 0 <= source < len(learner.word_forms)
        else b""
    )
    row = learner.route_counts.get(key, {})
    return {
        "source_assembly": int(source),
        "source_form": form.decode("utf-8", errors="backslashreplace"),
        "receptor": int(receptor),
        "target_observations": int(row.get(target, 0)),
        "route_total": int(learner.route_totals.get(key, sum(row.values()))),
        "route_utility_bits": float(learner.route_utility.get(key, 0.0)),
        "target_information_bits": (
            _route_information(learner, key, target) if row.get(target, 0) else None
        ),
    }


def _oracle_distribution(
    learner: object,
    off: np.ndarray,
    item: tuple[int, int] | None,
    *,
    maximum_gain: float,
    target_variation: float | None = None,
) -> tuple[np.ndarray, float, float]:
    if item is None:
        return off, 0.0, 0.0
    signal = FieldReturnSignal((item,), 1.0, (1.0,))
    recalled = _receptor_route_distribution(
        learner,
        signal.items,
        predictive_only=False,
        item_strengths=signal.item_strengths,
    )
    if recalled is None:
        return off, 0.0, 0.0
    raw_variation = 0.5 * float(np.sum(np.abs(recalled - off)))
    gain = maximum_gain
    if target_variation is not None:
        gain = (
            0.0
            if raw_variation <= 0.0
            else min(1.0, target_variation / raw_variation)
        )
    on = (1.0 - gain) * off + gain * recalled
    on /= float(np.sum(on))
    variation = 0.5 * float(np.sum(np.abs(on - off)))
    return on, gain, variation


def _margin(distribution: np.ndarray, target: int) -> tuple[float, int, int]:
    target_probability = max(float(distribution[target]), 1e-300)
    competitors = distribution.copy()
    competitors[target] = -np.inf
    competitor = max(float(np.max(competitors)), 1e-300)
    rank = 1 + int(np.sum(distribution > distribution[target]))
    winner_count = int(np.count_nonzero(distribution == np.max(distribution)))
    return log2(target_probability) - log2(competitor), rank, winner_count


def _prime(payload: dict[str, object], prefix: bytes) -> object:
    arm = _new_arm(payload)
    for byte_id in prefix:
        distribution = _predict(arm, live=False)
        _observe(arm, byte_id, float(distribution[byte_id]))
    return arm


def _inject_temporal_item(
    learner: object, item: tuple[int, int] | None, *, replace: bool = False
) -> None:
    """Place recalled activity on its native temporal wire, not sensory input."""
    if item is None:
        return
    assembly, receptor = item
    if replace:
        learner.temporal_last_seen.clear()
        learner.temporal_recalled_sources.clear()
    learner.temporal_last_seen[assembly] = learner.temporal_clock - receptor + 1
    learner.temporal_recalled_sources.add(assembly)
    learner._goal_route_projection_cache = None


def _predict_condition(
    arm: object,
    condition: str,
    item: tuple[int, int] | None,
    *,
    maximum_gain: float,
    target_variation: float | None = None,
) -> tuple[np.ndarray, float, float]:
    if condition == "natural":
        distribution = _predict(arm, live=True)
        return distribution, 0.0, 0.0
    off = _predict(arm, live=False)
    if condition == "off":
        return off, 0.0, 0.0
    return _oracle_distribution(
        arm.learner,
        off,
        item,
        maximum_gain=maximum_gain,
        target_variation=target_variation,
    )


def _score_condition(
    payload: dict[str, object],
    prefix: bytes,
    observed: bytes,
    contrast: bytes,
    *,
    condition: str,
    item: tuple[int, int] | None,
    maximum_gain: float,
    matched_variations: tuple[float, ...] | None = None,
    temporal_injection_item: tuple[int, int] | None = None,
    replace_temporal_state: bool = False,
    reset_cortical_phase: bool = False,
) -> dict[str, object]:
    arm = _prime(payload, prefix)
    if reset_cortical_phase:
        arm.learner.current_word.clear()
        arm.learner.active_word_candidates = None
    _inject_temporal_item(
        arm.learner, temporal_injection_item, replace=replace_temporal_state
    )
    bits = 0.0
    positions = []
    greedy_correct = 0
    variations = []
    gains = []
    for index, byte_id in enumerate(observed):
        target_variation = (
            matched_variations[index]
            if matched_variations is not None and index < len(matched_variations)
            else None
        )
        distribution, gain, variation = _predict_condition(
            arm,
            condition,
            item,
            maximum_gain=maximum_gain,
            target_variation=target_variation,
        )
        probability = max(float(distribution[byte_id]), 1e-300)
        bits += -log2(probability)
        margin, rank, winner_count = _margin(distribution, byte_id)
        contrast_byte = int(contrast[index]) if index < len(contrast) else None
        contrast_margin = (
            None
            if contrast_byte is None or contrast_byte == byte_id
            else log2(probability)
            - log2(max(float(distribution[contrast_byte]), 1e-300))
        )
        positions.append(
            {
                "index": index,
                "target_byte": int(byte_id),
                "contrast_byte": contrast_byte,
                "target_probability": probability,
                "greedy_margin_bits": margin,
                "target_rank": rank,
                "winner_ties": winner_count,
                "contrast_margin_bits": contrast_margin,
                "injected_gain": gain,
                "injected_variation": variation,
            }
        )
        gains.append(gain)
        variations.append(variation)
        greedy_correct += int(int(np.argmax(distribution)) == int(byte_id))
        _observe(arm, int(byte_id), probability)
    return {
        "bits": bits,
        "bpb": bits / max(1, len(observed)),
        "greedy_byte_accuracy": greedy_correct / max(1, len(observed)),
        "positions": positions,
        "injected_gains": gains,
        "injected_variations": variations,
    }


def _greedy_expression(
    payload: dict[str, object],
    prefix: bytes,
    target: bytes,
    *,
    condition: str,
    item: tuple[int, int] | None,
    maximum_gain: float,
    matched_variation: float | None = None,
    temporal_injection_item: tuple[int, int] | None = None,
    replace_temporal_state: bool = False,
    reset_cortical_phase: bool = False,
) -> dict[str, object]:
    arm = _prime(payload, prefix)
    if reset_cortical_phase:
        arm.learner.current_word.clear()
        arm.learner.active_word_candidates = None
    _inject_temporal_item(
        arm.learner, temporal_injection_item, replace=replace_temporal_state
    )
    generated = bytearray()
    variations = []
    for _ in target:
        distribution, _, variation = _predict_condition(
            arm,
            condition,
            item,
            maximum_gain=maximum_gain,
            target_variation=matched_variation,
        )
        byte_id = int(np.argmax(distribution))
        generated.append(byte_id)
        variations.append(variation)
        _observe(arm, byte_id, float(distribution[byte_id]))
    return {
        "target": target.decode("utf-8", errors="backslashreplace"),
        "generated": bytes(generated).decode("utf-8", errors="backslashreplace"),
        "exact": bytes(generated).lower() == target.lower(),
        "injected_variations": variations,
    }


def _first_discriminating_position(expected: bytes, contrast: bytes) -> int | None:
    for index, (left, right) in enumerate(zip(expected, contrast, strict=False)):
        if left != right:
            return index
    return None


def _run_case(
    payload: dict[str, object],
    case: GenerationCase,
    *,
    maximum_gain: float,
) -> dict[str, object]:
    learner = payload["learner"]
    expected = case.expected.encode()
    contrast = case.contrast.encode()
    shared = _common_prefix(expected, contrast)
    expected_remainder = expected[len(shared) :]
    contrast_remainder = contrast[len(shared) :]
    expected_word = _first_word(expected_remainder)
    contrast_word = _first_word(contrast_remainder)
    expected_target = learner.word_ids.get(expected_word)
    contrast_target = learner.word_ids.get(contrast_word)
    if expected_target is None or contrast_target is None:
        return {
            "case": asdict(case),
            "available": False,
            "reason": "a controlled identity is absent from cortical assemblies",
        }
    phases = _phase_routes(learner, int(expected_target))
    swapped_key = _best_route_to(learner, int(contrast_target))
    prefix = case.prompt.encode() + shared
    conditions: dict[str, dict[str, object]] = {}
    for name, condition, key in (
        ("off", "off", None),
        ("natural", "natural", None),
        ("oracle_pre_pre_target", "oracle", phases["pre_pre_target"]),
        ("oracle_pre_target", "oracle", phases["pre_target"]),
        ("oracle_post_target", "oracle", phases["post_target"]),
    ):
        item = _item_from_key(key)
        scored = _score_condition(
            payload,
            prefix,
            expected_remainder,
            contrast_remainder,
            condition=condition,
            item=item,
            maximum_gain=maximum_gain,
        )
        conditions[name] = scored

    for name, key in (
        ("phase_reset_pre_pre_target", phases["pre_pre_target"]),
        ("phase_reset_pre_target", phases["pre_target"]),
        ("phase_reset_post_target", phases["post_target"]),
    ):
        conditions[name] = _score_condition(
            payload,
            prefix,
            expected_remainder,
            contrast_remainder,
            condition="oracle",
            item=_item_from_key(key),
            maximum_gain=maximum_gain,
            reset_cortical_phase=True,
        )

    for name, key in (
        ("native_pre_pre_target", phases["pre_pre_target"]),
        ("native_pre_target", phases["pre_target"]),
        ("native_post_target", phases["post_target"]),
    ):
        conditions[name] = _score_condition(
            payload,
            prefix,
            expected_remainder,
            contrast_remainder,
            condition="off",
            item=None,
            maximum_gain=maximum_gain,
            temporal_injection_item=_item_from_key(key),
            replace_temporal_state=True,
        )
    conditions["native_pre_target_competing"] = _score_condition(
        payload,
        prefix,
        expected_remainder,
        contrast_remainder,
        condition="off",
        item=None,
        maximum_gain=maximum_gain,
        temporal_injection_item=_item_from_key(phases["pre_target"]),
    )

    correct_variations = tuple(
        float(value)
        for value in conditions["oracle_pre_target"]["injected_variations"]
    )
    swapped = _score_condition(
        payload,
        prefix,
        expected_remainder,
        contrast_remainder,
        condition="oracle",
        item=_item_from_key(swapped_key),
        maximum_gain=maximum_gain,
        matched_variations=correct_variations,
    )
    conditions["identity_swapped"] = swapped
    conditions["identity_swapped_phase_reset"] = _score_condition(
        payload,
        prefix,
        expected_remainder,
        contrast_remainder,
        condition="oracle",
        item=_item_from_key(swapped_key),
        maximum_gain=maximum_gain,
        matched_variations=tuple(
            float(value)
            for value in conditions["phase_reset_pre_target"][
                "injected_variations"
            ]
        ),
        reset_cortical_phase=True,
    )
    conditions["identity_swapped_native"] = _score_condition(
        payload,
        prefix,
        expected_remainder,
        contrast_remainder,
        condition="off",
        item=None,
        maximum_gain=maximum_gain,
        temporal_injection_item=_item_from_key(swapped_key),
        replace_temporal_state=True,
    )

    first_difference = _first_discriminating_position(
        expected_remainder, contrast_remainder
    )
    target_word_remainder = expected_word
    expression = {}
    initial_correct_variation = (
        correct_variations[0] if correct_variations else None
    )
    phase_reset_variations = conditions["phase_reset_pre_target"][
        "injected_variations"
    ]
    initial_phase_reset_variation = (
        float(phase_reset_variations[0]) if phase_reset_variations else None
    )
    for name, condition, key in (
        ("off", "off", None),
        ("natural", "natural", None),
        ("oracle_pre_pre_target", "oracle", phases["pre_pre_target"]),
        ("oracle_pre_target", "oracle", phases["pre_target"]),
        ("oracle_post_target", "oracle", phases["post_target"]),
        ("identity_swapped", "oracle", swapped_key),
        ("phase_reset_pre_pre_target", "oracle", phases["pre_pre_target"]),
        ("phase_reset_pre_target", "oracle", phases["pre_target"]),
        ("phase_reset_post_target", "oracle", phases["post_target"]),
        ("identity_swapped_phase_reset", "oracle", swapped_key),
        ("native_pre_pre_target", "off", phases["pre_pre_target"]),
        ("native_pre_target", "off", phases["pre_target"]),
        ("native_post_target", "off", phases["post_target"]),
        ("native_pre_target_competing", "off", phases["pre_target"]),
        ("identity_swapped_native", "off", swapped_key),
    ):
        expression[name] = _greedy_expression(
            payload,
            prefix,
            target_word_remainder,
            condition=condition,
            item=_item_from_key(key),
            maximum_gain=maximum_gain,
            matched_variation=(
                initial_correct_variation
                if name == "identity_swapped"
                else initial_phase_reset_variation
                if name == "identity_swapped_phase_reset"
                else None
            ),
            temporal_injection_item=(
                _item_from_key(key)
                if name.startswith("native_") or name.endswith("_native")
                else None
            ),
            replace_temporal_state=(
                name.startswith("native_")
                and name != "native_pre_target_competing"
                or name == "identity_swapped_native"
            ),
            reset_cortical_phase=("phase_reset" in name),
        )

    forced_entry_expression = {}
    if len(expected_word) > 1:
        forced_prefix = prefix + expected_word[:1]
        forced_target = expected_word[1:]
        for name, condition, key in (
            ("off", "off", None),
            ("natural", "natural", None),
            ("oracle_pre_target", "oracle", phases["pre_target"]),
            ("identity_swapped", "oracle", swapped_key),
        ):
            forced_entry_expression[name] = _greedy_expression(
                payload,
                forced_prefix,
                forced_target,
                condition=condition,
                item=_item_from_key(key),
                maximum_gain=maximum_gain,
            )

    decision = {}
    if first_difference is not None:
        for name, scored in conditions.items():
            decision[name] = scored["positions"][first_difference]
        off_probability = float(decision["off"]["target_probability"])
        for item in decision.values():
            item["target_log2_probability_gain_vs_off"] = log2(
                max(float(item["target_probability"]), 1e-300)
                / max(off_probability, 1e-300)
            )
    return {
        "case": asdict(case),
        "available": True,
        "shared_prefix": shared.decode("utf-8", errors="backslashreplace"),
        "expected_identity": expected_word.decode(),
        "contrast_identity": contrast_word.decode(),
        "expected_target_assembly": int(expected_target),
        "contrast_target_assembly": int(contrast_target),
        "first_discriminating_position": first_difference,
        "phase_routes": {
            name: _route_description(learner, key, int(expected_target))
            for name, key in phases.items()
        },
        "identity_swapped_route": _route_description(
            learner, swapped_key, int(contrast_target)
        ),
        "conditions": conditions,
        "discriminating_decision": decision,
        "greedy_identity_expression": expression,
        "greedy_after_forced_entry": forced_entry_expression,
    }


def run_expression_ceiling(
    state: Path,
    *,
    cases: tuple[GenerationCase, ...] = CASES,
    maximum_gain: float = 0.02,
    sequence_settling_steps: int = 4,
    sequence_compatibility_gain: float = 2.0,
    sequence_stop_compatibility: float = 0.5,
) -> dict[str, object]:
    if not 0.0 <= maximum_gain <= 1.0:
        raise ValueError("maximum gain must be in [0, 1]")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    field = payload["field"]
    field.sequence_settling_steps = sequence_settling_steps
    field.sequence_compatibility_gain = sequence_compatibility_gain
    field.sequence_stop_compatibility = sequence_stop_compatibility
    payload["return_path"] = "all_receptor_routes"
    payload["maximum_return_weight"] = maximum_gain
    results = [
        _run_case(payload, case, maximum_gain=maximum_gain) for case in cases
    ]
    available = [result for result in results if result["available"]]
    condition_names = (
        "off",
        "natural",
        "oracle_pre_pre_target",
        "oracle_pre_target",
        "oracle_post_target",
        "identity_swapped",
        "phase_reset_pre_pre_target",
        "phase_reset_pre_target",
        "phase_reset_post_target",
        "identity_swapped_phase_reset",
        "native_pre_pre_target",
        "native_pre_target",
        "native_post_target",
        "native_pre_target_competing",
        "identity_swapped_native",
    )
    summary = {}
    for name in condition_names:
        decisions = [result["discriminating_decision"][name] for result in available]
        expression = [
            result["greedy_identity_expression"][name] for result in available
        ]
        summary[name] = {
            "cases": len(decisions),
            "mean_discriminating_greedy_margin_bits": (
                float(np.mean([item["greedy_margin_bits"] for item in decisions]))
                if decisions
                else 0.0
            ),
            "positive_discriminating_margin_fraction": (
                float(np.mean([item["greedy_margin_bits"] > 0 for item in decisions]))
                if decisions
                else 0.0
            ),
            "mean_discriminating_target_rank": (
                float(np.mean([item["target_rank"] for item in decisions]))
                if decisions
                else 0.0
            ),
            "mean_target_log2_probability_gain_vs_off": (
                float(
                    np.mean(
                        [
                            item["target_log2_probability_gain_vs_off"]
                            for item in decisions
                        ]
                    )
                )
                if decisions
                else 0.0
            ),
            "mean_winner_ties": (
                float(np.mean([item["winner_ties"] for item in decisions]))
                if decisions
                else 0.0
            ),
            "exact_greedy_identity_fraction": (
                float(np.mean([item["exact"] for item in expression]))
                if expression
                else 0.0
            ),
        }
    summary["forced_entry"] = {
        name: {
            "cases": len(available),
            "exact_remaining_identity_fraction": float(
                np.mean(
                    [
                        result["greedy_after_forced_entry"][name]["exact"]
                        for result in available
                    ]
                )
            )
            if available
            else 0.0,
        }
        for name in ("off", "natural", "oracle_pre_target", "identity_swapped")
    }
    return {
        "task": "phase80_expression_ceiling",
        "diagnostic_only": True,
        "state": str(state),
        "maximum_gain": maximum_gain,
        "sequence_settling_steps": sequence_settling_steps,
        "sequence_compatibility_gain": sequence_compatibility_gain,
        "sequence_stop_compatibility": sequence_stop_compatibility,
        "future_bytes_used_for_oracle_selection": True,
        "persistent_learning": False,
        "results": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--maximum-gain", type=float, default=0.02)
    parser.add_argument("--sequence-settling-steps", type=int, default=4)
    parser.add_argument("--sequence-compatibility-gain", type=float, default=2.0)
    parser.add_argument("--sequence-stop-compatibility", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (CASES[args.case_index],) if args.case_index is not None else CASES
    report = run_expression_ceiling(
        args.state,
        cases=cases,
        maximum_gain=args.maximum_gain,
        sequence_settling_steps=args.sequence_settling_steps,
        sequence_compatibility_gain=args.sequence_compatibility_gain,
        sequence_stop_compatibility=args.sequence_stop_compatibility,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
