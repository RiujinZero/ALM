"""Isolate acquisition, self-credit, and retriggering in access generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    DEFAULT_PROMPTS,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _enable_relational_access,
    _new_arm,
    _observe,
    _predict,
    _predict_encoding_only,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    _lexical_outcome,
)

CONDITIONS = (
    "recall_disabled",
    "frozen_associations",
    "protected_online_updates",
    "frozen_with_return_adaptation",
)


def _association_fingerprint(arm: object) -> str:
    anatomy = (
        arm.relational_access.weights,
        arm.relational_access.source_weights,
        arm.recalled_expression.weights,
        arm.recalled_expression.evidence,
    )
    return hashlib.sha256(
        pickle.dumps(anatomy, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _calibrate_expression(arm: object) -> None:
    """Calibrate existing cortical return wires from external experience."""
    # Give every externally acquired access path the already established Phase
    # 81 expression ceiling before freezing. No generation byte participates.
    learned_items = {item for _, item in arm.relational_access.weights}
    learned_items.update(item for _, item in arm.relational_access.branches)
    # Successor experiments must not confuse a correct temporal return with an
    # uncalibrated receiving pathway.  Add only targets already reachable from
    # predecessor phases learned through external access confirmation.
    for (engram, _item), sources in arm.relational_access.source_weights.items():
        if engram >= len(arm.field.engram_sequence_weights):
            continue
        transitions = arm.field.engram_sequence_weights[engram]
        for source in sources:
            learned_items.update(transitions.get(source, {}))
    for branches in arm.field.engram_contextual_successors:
        for branch in branches:
            learned_items.update(branch.targets)
    learned_items = sorted(learned_items)
    for item in learned_items:
        assembly, _ = item
        if 0 <= assembly < len(arm.learner.word_forms):
            arm.recalled_expression.learn_sequence(
                item, arm.learner.word_forms[assembly], repetitions=24
            )
            arm.expression_calibrated_items.add(item)
    arm.recalled_expression.reset_episode()


def _prepare(
    payload: dict[str, object],
    prompt: bytes,
    *,
    access_learning_mode: str = "hebbian",
    frames_per_capsule: int = 24,
    access_proposal_threshold: float = 0.0,
) -> object:
    arm = _new_arm(payload)
    _enable_relational_access(
        arm,
        expression_gain=0.5,
        frames_per_capsule=frames_per_capsule,
        access_proposal_threshold=access_proposal_threshold,
    )
    arm.relational_access.learning_mode = access_learning_mode
    arm.lazy_expression_enabled = False
    for byte_id in prompt:
        distribution = _predict_encoding_only(arm)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)
    _calibrate_expression(arm)
    return arm


def _generate_condition(
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
    if condition == "recall_disabled":
        arm.recall_enabled = False
    if condition in {
        "frozen_associations",
        "frozen_with_return_adaptation",
    }:
        arm.access_learning_enabled = False
        arm.expression_learning_enabled = False
    elif condition == "protected_online_updates":
        arm.access_learning_enabled = True
        arm.expression_learning_enabled = True
    if condition == "frozen_with_return_adaptation":
        arm.return_adaptation_half_life = 8.0
    before = _association_fingerprint(arm)
    initial_completions = arm.access_completions
    rng = np.random.default_rng(seed)
    generated = bytearray()
    for _ in range(max_new_bytes):
        distribution = _predict(arm, live=condition != "recall_disabled")
        byte_id = _sample_byte(
            distribution,
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        # Generated and recalled activity are state, never external evidence.
        _observe(arm, byte_id, float(distribution[byte_id]), external=False)
    return (
        bytes(generated),
        arm,
        before == _association_fingerprint(arm),
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
    seed: int = 8500,
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
    for prompt_index, prompt in enumerate(prompts):
        controlled = (
            CASES[prompt_index - len(DEFAULT_PROMPTS)]
            if prompt_index >= len(DEFAULT_PROMPTS)
            else None
        )
        prepared = _prepare(payload, prompt.encode())
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            outputs = {}
            for condition in CONDITIONS:
                generated, arm, frozen, generation_completions = _generate_condition(
                    prepared,
                    condition,
                    seed=sample_seed,
                    max_new_bytes=max_new_bytes,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                outputs[condition] = {
                    "generated": generated,
                    "associations_unchanged": frozen,
                    "access_confirmations_after_prompt": (
                        arm.access_confirmations - prepared.access_confirmations
                    ),
                    "access_completions": generation_completions,
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
                    "prompt_access_confirmations": prepared.access_confirmations,
                    "prompt_access_paths": len(prepared.relational_access.weights),
                    "conditions": {
                        name: {
                            key: value
                            for key, value in result.items()
                            if key != "generated"
                        }
                        | {
                            "equals_baseline": result["generated"] == baseline,
                            "first_divergence": next(
                                (
                                    index
                                    for index, (left, right) in enumerate(
                                        zip(
                                            baseline,
                                            result["generated"],
                                            strict=True,
                                        )
                                    )
                                    if left != right
                                ),
                                None,
                            ),
                        }
                        for name, result in outputs.items()
                    },
                }
            )
    summaries = {}
    for condition in CONDITIONS:
        values = [row["conditions"][condition] for row in rows]
        controlled_values = [
            row["conditions"][condition]
            for row in rows
            if row["controlled_case"] is not None
        ]
        summaries[condition] = {
            "equals_baseline_fraction": float(
                np.mean([value["equals_baseline"] for value in values])
            ),
            "associations_unchanged_fraction": float(
                np.mean([value["associations_unchanged"] for value in values])
            ),
            "post_prompt_confirmations": int(
                sum(value["access_confirmations_after_prompt"] for value in values)
            ),
            "mean_access_completions": float(
                np.mean([value["access_completions"] for value in values])
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
        }
    return {
        "task": "phase85_generation_credit_and_adaptation_ablation",
        "state": str(state),
        "conditions": CONDITIONS,
        "generated_activity_is_external": False,
        "adaptation_half_life_bytes": 8.0,
        "samples_per_prompt": samples_per_prompt,
        "max_new_bytes": max_new_bytes,
        "elapsed_seconds": perf_counter() - started,
        "summary": summaries,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=300)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_ablation(
        args.state,
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        prompt_limit=args.prompt_limit,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
