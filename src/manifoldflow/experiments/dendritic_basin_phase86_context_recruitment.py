"""Test context-conditioned recruitment of already learned access branches."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    DEFAULT_PROMPTS,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    _lexical_outcome,
)

CONDITIONS = {
    "recall_disabled": None,
    "adapted_unconditioned": "none",
    "adapted_basal_context": "basal",
    "adapted_apical_context": "apical",
    "adapted_branch_recruitment": "branch_peak",
    "adapted_branch_successor": "branch_peak",
    "adapted_branch_aligned_successor": "branch_peak",
    "adapted_branch_content_successor": "branch_peak",
    "adapted_contextual_successor": "branch_peak",
}


def _generate(
    prepared: object,
    condition: str,
    *,
    seed: int,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[bytes, object, bool, int]:
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.lazy_expression_enabled = False
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.return_adaptation_half_life = 8.0
    mode = CONDITIONS[condition]
    if mode is None:
        arm.recall_enabled = False
    else:
        arm.access_recruitment_mode = mode
        if condition == "adapted_branch_successor":
            arm.access_return_mode = "successor"
        elif condition == "adapted_branch_aligned_successor":
            arm.access_return_mode = "aligned_successor"
        elif condition == "adapted_branch_content_successor":
            arm.access_return_mode = "content_successor"
        elif condition == "adapted_contextual_successor":
            arm.access_return_mode = "contextual_successor"
    before = phase85._association_fingerprint(arm)
    initial_completions = arm.access_completions
    rng = np.random.default_rng(seed)
    generated = bytearray()
    for _ in range(max_new_bytes):
        distribution = _predict(arm, live=arm.recall_enabled)
        byte_id = _sample_byte(
            distribution,
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        _observe(arm, byte_id, float(distribution[byte_id]), external=False)
    return (
        bytes(generated),
        arm,
        before == phase85._association_fingerprint(arm),
        arm.access_completions - initial_completions,
    )


def run_ablation(
    state: Path,
    *,
    samples_per_prompt: int = 1,
    max_new_bytes: int = 300,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    seed: int = 8600,
    prompt_limit: int | None = None,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
    contextual_successor_binding: bool = False,
) -> dict[str, object]:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown or "recall_disabled" not in conditions:
        raise ValueError("conditions must be known and include recall_disabled")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    payload["field"].sequence_settling_steps = 4
    payload["field"].contextual_successor_binding = contextual_successor_binding
    prompts = tuple(DEFAULT_PROMPTS) + tuple(case.prompt for case in CASES)
    if prompt_limit is not None:
        prompts = prompts[:prompt_limit]
    boundary = getattr(payload["learner"], "story_boundary", b"<|endoftext|>\n")
    rows = []
    scored_bytes = 0
    started = perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        controlled = (
            CASES[prompt_index - len(DEFAULT_PROMPTS)]
            if prompt_index >= len(DEFAULT_PROMPTS)
            else None
        )
        prepared = phase85._prepare(payload, prompt.encode())
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            outputs = {}
            for condition in conditions:
                generated, arm, frozen, generation_completions = _generate(
                    prepared,
                    condition,
                    seed=sample_seed,
                    max_new_bytes=max_new_bytes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                scored_bytes += len(generated)
                outputs[condition] = {
                    "generated": generated,
                    "associations_unchanged": frozen,
                    "access_completions": generation_completions,
                    "recruitment_events": arm.recruitment_events,
                    "recruitment_suppressed": arm.recruitment_suppressed,
                    "recruitment_support_values": arm.recruitment_support_values,
                    "access_sequence_attempts": arm.access_sequence_attempts,
                    "access_sequence_source_hits": arm.access_sequence_source_hits,
                    "diagnostics": _generation_diagnostics(generated, boundary),
                    "outcome": (
                        None
                        if controlled is None
                        else _lexical_outcome(
                            generated,
                            controlled.expected.encode(),
                            controlled.contrast.encode(),
                        )
                    ),
                    "text": generated.decode("utf-8", errors="backslashreplace"),
                }
            baseline = outputs["recall_disabled"]["generated"]
            rows.append(
                {
                    "prompt": prompt,
                    "seed": sample_seed,
                    "controlled_case": None if controlled is None else controlled.name,
                    "prompt_access_paths": len(prepared.relational_access.weights),
                    "conditions": {
                        name: {
                            key: value
                            for key, value in result.items()
                            if key != "generated"
                        }
                        | {"equals_baseline": result["generated"] == baseline}
                        for name, result in outputs.items()
                    },
                }
            )
    elapsed = perf_counter() - started
    summary = {}
    for condition in conditions:
        values = [row["conditions"][condition] for row in rows]
        controlled_values = [
            row["conditions"][condition]
            for row in rows
            if row["controlled_case"] is not None
        ]
        proposed = sum(value["recruitment_events"] for value in values)
        suppressed = sum(value["recruitment_suppressed"] for value in values)
        support_values = [
            support
            for value in values
            for support in value["recruitment_support_values"]
        ]
        sequence_attempts = sum(
            value["access_sequence_attempts"] for value in values
        )
        sequence_hits = sum(
            value["access_sequence_source_hits"] for value in values
        )
        summary[condition] = {
            "equals_baseline_fraction": float(
                np.mean([value["equals_baseline"] for value in values])
            ),
            "mean_access_completions": float(
                np.mean([value["access_completions"] for value in values])
            ),
            "recruitment_proposals": int(proposed),
            "recruitment_suppressed_fraction": (
                float(suppressed / proposed) if proposed else 0.0
            ),
            "recruitment_support_quantiles": (
                {
                    "p10": float(np.quantile(support_values, 0.10)),
                    "p50": float(np.quantile(support_values, 0.50)),
                    "p90": float(np.quantile(support_values, 0.90)),
                }
                if support_values
                else {"p10": 0.0, "p50": 0.0, "p90": 0.0}
            ),
            "access_sequence_source_hit_fraction": (
                float(sequence_hits / sequence_attempts) if sequence_attempts else 0.0
            ),
            "mean_distinct_4gram_fraction": float(
                np.mean(
                    [
                        value["diagnostics"]["distinct_ngram_fraction"]["4"]
                        for value in values
                    ]
                )
            ),
            "mean_zlib_ratio": float(
                np.mean(
                    [value["diagnostics"]["zlib_compression_ratio"] for value in values]
                )
            ),
            "maximum_repeated_bytes": int(
                max(
                    value["diagnostics"]["longest_repeated_block"]["repeated_bytes"]
                    for value in values
                )
            ),
            "controlled_expected_fraction": (
                float(
                    np.mean(
                        [value["outcome"] == "expected" for value in controlled_values]
                    )
                )
                if controlled_values
                else 0.0
            ),
            "associations_unchanged_fraction": float(
                np.mean([value["associations_unchanged"] for value in values])
            ),
        }
    return {
        "task": "phase86_context_conditioned_recruitment",
        "state": str(state),
        "conditions": conditions,
        "generated_activity_is_external": False,
        "adaptation_half_life_bytes": 8.0,
        "samples_per_prompt": samples_per_prompt,
        "max_new_bytes": max_new_bytes,
        "elapsed_seconds": elapsed,
        "throughput_bytes_per_second": scored_bytes / max(elapsed, 1e-12),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=tuple(CONDITIONS),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_ablation(
        args.state,
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        prompt_limit=args.prompt_limit,
        conditions=tuple(args.conditions),
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
