"""Controlled behavioral generation gates for ALM V1 episodic return."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from math import log2
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    _first_divergence,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _generate,
    _new_arm,
    _observe,
    _predict,
)


@dataclass(frozen=True)
class GenerationCase:
    name: str
    capability: str
    prompt: str
    expected: str
    contrast: str


CASES = (
    GenerationCase(
        "identity_after_distractor",
        "identity retention",
        (
            "Lily had a red ball. Tom had a blue kite. Lily put her red ball "
            "in a small box. Tom flew his blue kite outside. They ate lunch "
            "and talked for a long time. When Lily opened her box again, she "
            "found her"
        ),
        " red ball",
        " blue kite",
    ),
    GenerationCase(
        "reassigned_owner",
        "binding reassignment",
        (
            "Lily had a red ball. Tom had a blue kite. Lily gave the red ball "
            "to Tom, and Tom gave the blue kite to Lily. They played in the "
            "garden until sunset. When Lily went home, she carried the"
        ),
        " blue kite",
        " red ball",
    ),
    GenerationCase(
        "separated_locations",
        "role separation",
        (
            "Mia put the puppy in the garden. Ben put the kitten in the "
            "kitchen. Mia washed her hands and read a book. Ben made lunch. "
            "Later Mia returned to the garden and saw the"
        ),
        " puppy",
        " kitten",
    ),
    GenerationCase(
        "irrelevant_event",
        "irrelevant-memory rejection",
        (
            "Lily's favorite toy was her red ball. She played with it every "
            "morning. Across town, Tom saw a blue kite near a tall tree and "
            "Ben ate an apple in the kitchen. When Lily was asked about her "
            "favorite toy, she said it was the"
        ),
        " red ball",
        " blue kite",
    ),
)


def _score_suffix(
    payload: dict[str, object], prompt: bytes, suffix: bytes, *, live: bool
) -> dict[str, float | int]:
    arm = _new_arm(payload)
    for byte_id in prompt:
        distribution = _predict(arm, live=live)
        _observe(arm, byte_id, float(distribution[byte_id]))
    bits = 0.0
    for byte_id in suffix:
        distribution = _predict(arm, live=live)
        bits += -log2(max(float(distribution[byte_id]), 1e-300))
        _observe(arm, byte_id, float(distribution[byte_id]))
    return {
        "bits": bits,
        "bpb": bits / max(1, len(suffix)),
        "completions": arm.completions,
        "effective_returns": arm.effective_returns,
        "transient_captures": arm.transient_captures,
    }


def _lexical_outcome(generated: bytes, expected: bytes, contrast: bytes) -> str:
    generated = generated.lower()
    expected_at = generated.find(expected.strip().lower())
    contrast_at = generated.find(contrast.strip().lower())
    if expected_at < 0 and contrast_at < 0:
        return "neither"
    if expected_at >= 0 and (contrast_at < 0 or expected_at < contrast_at):
        return "expected"
    if contrast_at >= 0 and (expected_at < 0 or contrast_at < expected_at):
        return "contrast"
    return "tie"


def run_quality_gate(
    state: Path,
    *,
    cases: tuple[GenerationCase, ...] = CASES,
    samples_per_case: int = 3,
    max_new_bytes: int = 300,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    seed: int = 7700,
    return_buffer_mode: str | None = None,
    return_buffer_half_life: float | None = None,
    return_buffer_half_life_ms: float | None = None,
    output_mode: str = "unordered",
    output_delay_bytes: int = 16,
    return_path: str | None = None,
    maximum_return_weight: float | None = None,
    sequence_settling_steps: int | None = None,
    sequence_compatibility_gain: float | None = None,
    sequence_stop_compatibility: float | None = None,
) -> dict[str, object]:
    started = perf_counter()
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    field = payload["field"]
    if not hasattr(field, "return_buffer_time"):
        field.return_buffer_time = {}
    if not hasattr(field, "return_buffer_mode"):
        field.return_buffer_mode = "bytes"
    if not hasattr(field, "return_buffer_half_life_ms"):
        field.return_buffer_half_life_ms = 250.0
    if return_buffer_mode is not None:
        if return_buffer_mode not in {"bytes", "milliseconds", "either"}:
            raise ValueError("unknown return buffer mode")
        field.return_buffer_mode = return_buffer_mode
    if return_buffer_half_life is not None:
        if return_buffer_half_life < 0.0:
            raise ValueError("byte half-life must be non-negative")
        field.return_buffer_half_life = return_buffer_half_life
    if return_buffer_half_life_ms is not None:
        if return_buffer_half_life_ms < 0.0:
            raise ValueError("millisecond half-life must be non-negative")
        field.return_buffer_half_life_ms = return_buffer_half_life_ms
    if output_mode not in {"unordered", "fifo", "delayed_latest", "recurrent_fifo"}:
        raise ValueError("unknown temporal output mode")
    if output_delay_bytes < 1:
        raise ValueError("output delay must be positive")
    payload["output_mode"] = output_mode
    payload["output_delay_bytes"] = output_delay_bytes
    if return_path is not None:
        if return_path not in {
            "temporal_trace",
            "receptor_routes",
            "all_receptor_routes",
        }:
            raise ValueError("unknown return path")
        payload["return_path"] = return_path
    if maximum_return_weight is not None:
        if not 0.0 <= maximum_return_weight <= 1.0:
            raise ValueError("maximum return weight must be in [0, 1]")
        payload["maximum_return_weight"] = maximum_return_weight
    if sequence_settling_steps is not None:
        if sequence_settling_steps < 0:
            raise ValueError("sequence settling steps must be non-negative")
        field.sequence_settling_steps = sequence_settling_steps
    if sequence_compatibility_gain is not None:
        if sequence_compatibility_gain < 0.0:
            raise ValueError("sequence compatibility gain must be non-negative")
        field.sequence_compatibility_gain = sequence_compatibility_gain
    if sequence_stop_compatibility is not None:
        if not 0.0 <= sequence_stop_compatibility <= 1.0:
            raise ValueError("sequence stop compatibility must be in [0, 1]")
        field.sequence_stop_compatibility = sequence_stop_compatibility
    boundary = getattr(payload["learner"], "story_boundary", b"<|endoftext|>\n")
    results = []
    for case_index, case in enumerate(cases):
        prompt = case.prompt.encode()
        expected = case.expected.encode()
        contrast = case.contrast.encode()
        forced = {}
        for live, label in ((True, "live"), (False, "shadow")):
            expected_score = _score_suffix(payload, prompt, expected, live=live)
            contrast_score = _score_suffix(payload, prompt, contrast, live=live)
            forced[label] = {
                "expected": expected_score,
                "contrast": contrast_score,
                "expected_margin_bits_per_byte": (
                    float(contrast_score["bpb"]) - float(expected_score["bpb"])
                ),
            }
        samples = []
        for sample_index in range(samples_per_case):
            sample_seed = seed + case_index * samples_per_case + sample_index
            live_bytes, live_arm = _generate(
                payload,
                prompt,
                live=True,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            shadow_bytes, shadow_arm = _generate(
                payload,
                prompt,
                live=False,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            samples.append(
                {
                    "seed": sample_seed,
                    "byte_identical": live_bytes == shadow_bytes,
                    "first_divergence_byte": _first_divergence(
                        live_bytes, shadow_bytes
                    ),
                    "live_outcome": _lexical_outcome(
                        live_bytes, expected, contrast
                    ),
                    "shadow_outcome": _lexical_outcome(
                        shadow_bytes, expected, contrast
                    ),
                    "live_effective_returns": live_arm.effective_returns,
                    "live_completions": live_arm.completions,
                    "live_sequence_completions": live_arm.sequence_completions,
                    "live_fresh_effective_returns": (
                        live_arm.fresh_effective_returns
                    ),
                    "live_buffered_effective_returns": (
                        live_arm.buffered_effective_returns
                    ),
                    "live_completed_forms": sorted(
                        form.decode("utf-8", errors="backslashreplace")
                        for form in live_arm.completed_forms
                    ),
                    "live_effective_return_forms": sorted(
                        form.decode("utf-8", errors="backslashreplace")
                        for form in live_arm.effective_return_forms
                    ),
                    "live_transient_captures": live_arm.transient_captures,
                    "live_text": (prompt + live_bytes).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "shadow_text": (prompt + shadow_bytes).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "live_diagnostics": _generation_diagnostics(
                        live_bytes, boundary
                    ),
                    "shadow_diagnostics": _generation_diagnostics(
                        shadow_bytes, boundary
                    ),
                }
            )
        results.append(
            {
                "case": asdict(case),
                "forced_choice": forced,
                "memory_margin_delta_bits_per_byte": (
                    forced["live"]["expected_margin_bits_per_byte"]
                    - forced["shadow"]["expected_margin_bits_per_byte"]
                ),
                "samples": samples,
            }
        )
    samples = [sample for result in results for sample in result["samples"]]
    margins = [result["memory_margin_delta_bits_per_byte"] for result in results]
    processed_byte_steps = sum(
        2 * (len(case.prompt.encode()) + len(case.expected.encode()))
        + 2 * (len(case.prompt.encode()) + len(case.contrast.encode()))
        + 2 * samples_per_case * (len(case.prompt.encode()) + max_new_bytes)
        for case in cases
    )
    elapsed_seconds = perf_counter() - started
    return {
        "task": "phase77_controlled_generation_quality",
        "state": str(state),
        "persistent_learning_during_generation": False,
        "transient_episode_learning_during_generation": True,
        "return_buffer_mode": field.return_buffer_mode,
        "return_buffer_half_life": field.return_buffer_half_life,
        "return_buffer_half_life_ms": field.return_buffer_half_life_ms,
        "output_mode": output_mode,
        "output_delay_bytes": output_delay_bytes,
        "order_sensitive_binding": bool(
            getattr(field, "order_sensitive_binding", False)
        ),
        "return_path": payload["return_path"],
        "maximum_return_weight": payload["maximum_return_weight"],
        "sequence_settling_steps": field.sequence_settling_steps,
        "sequence_compatibility_gain": field.sequence_compatibility_gain,
        "sequence_stop_compatibility": field.sequence_stop_compatibility,
        "cases": results,
        "summary": {
            "cases": len(results),
            "samples": len(samples),
            "changed_samples": sum(not sample["byte_identical"] for sample in samples),
            "live_expected": sum(
                sample["live_outcome"] == "expected" for sample in samples
            ),
            "live_contrast": sum(
                sample["live_outcome"] == "contrast" for sample in samples
            ),
            "shadow_expected": sum(
                sample["shadow_outcome"] == "expected" for sample in samples
            ),
            "shadow_contrast": sum(
                sample["shadow_outcome"] == "contrast" for sample in samples
            ),
            "effective_returns": sum(
                sample["live_effective_returns"] for sample in samples
            ),
            "transient_captures": sum(
                sample["live_transient_captures"] for sample in samples
            ),
            "mean_memory_margin_delta_bits_per_byte": float(np.mean(margins)),
            "positive_memory_margin_cases": sum(margin > 0.0 for margin in margins),
            "processed_byte_steps": processed_byte_steps,
            "elapsed_seconds": elapsed_seconds,
            "processed_byte_steps_per_second": (
                processed_byte_steps / max(elapsed_seconds, 1e-12)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--samples-per-case", type=int, default=3)
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7700)
    parser.add_argument(
        "--return-buffer-mode",
        choices=("bytes", "milliseconds", "either"),
    )
    parser.add_argument("--return-buffer-half-life", type=float)
    parser.add_argument("--return-buffer-half-life-ms", type=float)
    parser.add_argument(
        "--output-mode",
        choices=("unordered", "fifo", "delayed_latest", "recurrent_fifo"),
        default="unordered",
    )
    parser.add_argument("--output-delay-bytes", type=int, default=16)
    parser.add_argument(
        "--return-path",
        choices=("temporal_trace", "receptor_routes", "all_receptor_routes"),
    )
    parser.add_argument("--maximum-return-weight", type=float)
    parser.add_argument("--sequence-settling-steps", type=int)
    parser.add_argument("--sequence-compatibility-gain", type=float)
    parser.add_argument("--sequence-stop-compatibility", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (CASES[args.case_index],) if args.case_index is not None else CASES
    report = run_quality_gate(
        args.state,
        cases=cases,
        samples_per_case=args.samples_per_case,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        return_buffer_mode=args.return_buffer_mode,
        return_buffer_half_life=args.return_buffer_half_life,
        return_buffer_half_life_ms=args.return_buffer_half_life_ms,
        output_mode=args.output_mode,
        output_delay_bytes=args.output_delay_bytes,
        return_path=args.return_path,
        maximum_return_weight=args.maximum_return_weight,
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
