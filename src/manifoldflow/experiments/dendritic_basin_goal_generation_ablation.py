"""Paired free-generation ablation for a goal-conditioned ALM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _goal_route_telemetry,
    _read_checkpoint,
    _summarize_samples,
    generate_bytes,
)

DEFAULT_PROMPTS = (
    "Once upon a time",
    "Lily had a red ball. She",
    "Tom was afraid of the dark, but",
    "Mia wanted to bring the lost puppy home. First,",
)


def _first_divergence(left: bytes, right: bytes) -> int | None:
    for index, (left_byte, right_byte) in enumerate(zip(left, right, strict=False)):
        if left_byte != right_byte:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def run_ablation(
    checkpoint: Path,
    *,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    samples_per_prompt: int = 3,
    max_new_bytes: int = 1_000,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    seed: int = 0,
    minimum_route_utility_bits: float | None = None,
    maximum_goal_route_bonus: float | None = None,
    permit_all_goal_routes: bool = False,
) -> dict[str, object]:
    if samples_per_prompt < 1 or not prompts:
        raise ValueError("prompts and samples per prompt must be non-empty")
    goal_on = _read_checkpoint(checkpoint)
    goal_off = _read_checkpoint(checkpoint)
    learned_gain = getattr(goal_on, "goal_apical_weight", None)
    if learned_gain is None:
        raise ValueError("checkpoint does not contain a goal-conditioned cortex")
    checkpoint_minimum_route_utility_bits = goal_on.minimum_route_utility_bits
    checkpoint_maximum_goal_route_bonus = goal_on.maximum_goal_route_bonus
    if minimum_route_utility_bits is not None:
        if minimum_route_utility_bits < 0.0:
            raise ValueError("minimum route utility bits must be non-negative")
        goal_on.minimum_route_utility_bits = minimum_route_utility_bits
        goal_off.minimum_route_utility_bits = minimum_route_utility_bits
    if maximum_goal_route_bonus is not None:
        if not 0.0 <= maximum_goal_route_bonus <= 1.0:
            raise ValueError("maximum goal route bonus must be in [0, 1]")
        goal_on.maximum_goal_route_bonus = maximum_goal_route_bonus
        goal_off.maximum_goal_route_bonus = maximum_goal_route_bonus
    clamped_goal_permission_paths = 0
    if permit_all_goal_routes:
        for key, restoration in goal_on.goal_permission_restoration_sum.items():
            goal_on.goal_permission_restoration_sum[key] = abs(restoration) + 1e-12
            clamped_goal_permission_paths += 1
    goal_off.goal_apical_weight = 0.0
    boundary = getattr(goal_on, "story_boundary", b"<|endoftext|>\n")
    pairs = []
    on_samples = []
    off_samples = []
    for prompt_index, prompt_text in enumerate(prompts):
        prompt = prompt_text.encode("utf-8")
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            telemetry_before = _goal_route_telemetry(goal_on)
            generated_on = generate_bytes(
                goal_on,
                prompt,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=sample_seed,
                stop_at_story_boundary=True,
            )
            telemetry = {
                name: value - telemetry_before[name]
                for name, value in _goal_route_telemetry(goal_on).items()
            }
            switches = telemetry["goal_route_switches"]
            generated_off = generate_bytes(
                goal_off,
                prompt,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=sample_seed,
                stop_at_story_boundary=True,
            )
            on_record = {
                "generated_bytes": len(generated_on),
                "generated_hex": generated_on.hex(),
                "text": (prompt + generated_on).decode(
                    "utf-8", errors="backslashreplace"
                ),
                "diagnostics": _generation_diagnostics(generated_on, boundary),
                "goal_route_switches": switches,
                "goal_route_telemetry": telemetry,
            }
            off_record = {
                "generated_bytes": len(generated_off),
                "generated_hex": generated_off.hex(),
                "text": (prompt + generated_off).decode(
                    "utf-8", errors="backslashreplace"
                ),
                "diagnostics": _generation_diagnostics(generated_off, boundary),
                "goal_route_switches": 0,
                "goal_route_telemetry": {
                    name: 0 for name in telemetry
                },
            }
            on_samples.append(on_record)
            off_samples.append(off_record)
            pairs.append(
                {
                    "prompt": prompt_text,
                    "seed": sample_seed,
                    "byte_identical": generated_on == generated_off,
                    "first_divergence_byte": _first_divergence(
                        generated_on, generated_off
                    ),
                    "goal_route_switches": switches,
                    "goal_on": on_record,
                    "goal_off": off_record,
                }
            )
    changed = np.asarray([not pair["byte_identical"] for pair in pairs], dtype=bool)
    switched = np.asarray([pair["goal_route_switches"] > 0 for pair in pairs])
    return {
        "task": "paired_goal_generation_ablation",
        "checkpoint": str(checkpoint),
        "learned_goal_apical_weight": learned_gain,
        "ablation_goal_apical_weight": 0.0,
        "checkpoint_minimum_route_utility_bits": (
            checkpoint_minimum_route_utility_bits
        ),
        "effective_minimum_route_utility_bits": goal_on.minimum_route_utility_bits,
        "checkpoint_maximum_goal_route_bonus": checkpoint_maximum_goal_route_bonus,
        "effective_maximum_goal_route_bonus": goal_on.maximum_goal_route_bonus,
        "permit_all_goal_routes": permit_all_goal_routes,
        "clamped_goal_permission_paths": clamped_goal_permission_paths,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_bytes": max_new_bytes,
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "changed_pair_fraction": float(np.mean(changed)),
            "switched_pair_fraction": float(np.mean(switched)),
            "changed_with_switch_pair_fraction": float(np.mean(changed & switched)),
            "changed_without_switch_count": int(np.sum(changed & ~switched)),
            "goal_on": _summarize_samples(on_samples),
            "goal_off": _summarize_samples(off_samples),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-new-bytes", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-route-utility-bits", type=float, default=None)
    parser.add_argument("--maximum-goal-route-bonus", type=float, default=None)
    parser.add_argument("--permit-all-goal-routes", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_ablation(
        args.checkpoint,
        prompts=tuple(args.prompts),
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        minimum_route_utility_bits=args.minimum_route_utility_bits,
        maximum_goal_route_bonus=args.maximum_goal_route_bonus,
        permit_all_goal_routes=args.permit_all_goal_routes,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
