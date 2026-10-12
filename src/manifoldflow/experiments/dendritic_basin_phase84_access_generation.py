"""Paired free generation with online relational access live or shadowed."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    DEFAULT_PROMPTS,
    _first_divergence,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import _generate
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    _lexical_outcome,
)


def run_generation_probe(
    state: Path,
    *,
    samples_per_prompt: int = 2,
    max_new_bytes: int = 300,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    expression_gain: float = 0.5,
    seed: int = 8400,
    prompt_limit: int | None = None,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    payload["field"].sequence_settling_steps = 4
    prompts = tuple(DEFAULT_PROMPTS) + tuple(case.prompt for case in CASES)
    if prompt_limit is not None:
        prompts = prompts[:prompt_limit]
    boundary = getattr(payload["learner"], "story_boundary", b"<|endoftext|>\n")
    rows = []
    started = perf_counter()
    processed_bytes = 0
    for prompt_index, prompt in enumerate(prompts):
        controlled = (
            CASES[prompt_index - len(DEFAULT_PROMPTS)]
            if prompt_index >= len(DEFAULT_PROMPTS)
            else None
        )
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            live, live_arm = _generate(
                payload,
                prompt.encode(),
                live=True,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                relational_access=True,
                expression_gain=expression_gain,
            )
            shadow, shadow_arm = _generate(
                payload,
                prompt.encode(),
                live=False,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                relational_access=True,
                expression_gain=expression_gain,
            )
            processed_bytes += 2 * (len(prompt.encode()) + max_new_bytes)
            rows.append(
                {
                    "prompt": prompt,
                    "seed": sample_seed,
                    "controlled_case": None if controlled is None else controlled.name,
                    "byte_identical": live == shadow,
                    "first_divergence_byte": _first_divergence(live, shadow),
                    "live_access_confirmations": live_arm.access_confirmations,
                    "shadow_access_confirmations": shadow_arm.access_confirmations,
                    "live_access_completions": live_arm.access_completions,
                    "live_field_completions": live_arm.completions,
                    "live_effective_returns": live_arm.effective_returns,
                    "live_relational_access_bytes": (
                        live_arm.relational_access.embodied_bytes
                    ),
                    "live_expression_bytes": (
                        live_arm.recalled_expression.embodied_bytes
                    ),
                    "live_multiscale_bytes": live_arm.multiscale.embodied_bytes,
                    "live_diagnostics": _generation_diagnostics(live, boundary),
                    "shadow_diagnostics": _generation_diagnostics(shadow, boundary),
                    "live_outcome": (
                        None
                        if controlled is None
                        else _lexical_outcome(
                            live,
                            controlled.expected.encode(),
                            controlled.contrast.encode(),
                        )
                    ),
                    "shadow_outcome": (
                        None
                        if controlled is None
                        else _lexical_outcome(
                            shadow,
                            controlled.expected.encode(),
                            controlled.contrast.encode(),
                        )
                    ),
                    "live_text": (prompt.encode() + live).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "shadow_text": (prompt.encode() + shadow).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                }
            )
    elapsed = perf_counter() - started
    controlled_rows = [row for row in rows if row["controlled_case"] is not None]
    def controlled_fraction(outcome: str, *, live: bool) -> float:
        key = "live_outcome" if live else "shadow_outcome"
        return (
            float(np.mean([row[key] == outcome for row in controlled_rows]))
            if controlled_rows
            else 0.0
        )

    return {
        "task": "phase84_online_access_free_generation",
        "state": str(state),
        "access_learning_from_prompt_only": True,
        "generated_bytes_can_teach_access": False,
        "diagnostic_lazy_expression_replay": True,
        "samples_per_prompt": samples_per_prompt,
        "max_new_bytes": max_new_bytes,
        "expression_gain": expression_gain,
        "elapsed_seconds": elapsed,
        "processed_bytes_per_second": processed_bytes / max(elapsed, 1e-12),
        "summary": {
            "pairs": len(rows),
            "different_fraction": float(
                np.mean([not row["byte_identical"] for row in rows])
            ),
            "mean_access_confirmations": float(
                np.mean([row["live_access_confirmations"] for row in rows])
            ),
            "mean_access_completions": float(
                np.mean([row["live_access_completions"] for row in rows])
            ),
            "controlled_live_expected_fraction": controlled_fraction(
                "expected", live=True
            ),
            "controlled_shadow_expected_fraction": controlled_fraction(
                "expected", live=False
            ),
            "controlled_live_contrast_fraction": controlled_fraction(
                "contrast", live=True
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--expression-gain", type=float, default=0.5)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_generation_probe(
        args.state,
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        expression_gain=args.expression_gain,
        prompt_limit=args.prompt_limit,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
