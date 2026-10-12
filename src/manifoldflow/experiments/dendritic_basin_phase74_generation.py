"""Paired autonomous generation with frozen Phase 74 recall permission."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    DEFAULT_PROMPTS,
    _first_divergence,
)
from manifoldflow.experiments.dendritic_basin_phase74_risk_permission import (
    _select_proposal,
)


@dataclass
class _Arm:
    learner: object
    memory: object
    permission: object
    feature_context: str
    candidate_limit: int
    veto_winner_change: bool
    recency_bonus: float
    recency_half_life: float
    recall_recency_bonus: float
    recall_recency_half_life: float
    raw_window: bytearray
    previous_routes: set[tuple[int, int]]
    previous_surprise: float = 0.0
    proposals: int = 0
    permitted: int = 0
    permitted_total_variation: float = 0.0
    maximum_permitted_total_variation: float = 0.0


def _new_arm(payload: dict[str, object]) -> _Arm:
    clone = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    learner = clone["learner"]
    memory = clone["memory"]
    learner.reset_episode()
    memory.reset_episode()
    return _Arm(
        learner,
        memory,
        clone["permission"],
        str(clone["feature_context"]),
        int(clone["candidate_limit"]),
        bool(clone["veto_winner_change"]),
        float(clone.get("recency_bonus", 0.0)),
        float(clone.get("recency_half_life", 256.0)),
        float(clone.get("recall_recency_bonus", 0.0)),
        float(clone.get("recall_recency_half_life", 256.0)),
        bytearray(),
        set(),
    )


def _predict(arm: _Arm, *, live: bool) -> np.ndarray:
    off = arm.learner.predict_next_distribution()
    query = int.from_bytes(arm.raw_window, "big") if len(arm.raw_window) == 3 else None
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    selected = _select_proposal(
        arm.learner,
        arm.memory,
        arm.permission,
        query=query,
        basin=basin,
        off=off,
        candidate_limit=arm.candidate_limit,
        feature_context=arm.feature_context,
        veto_winner_change=arm.veto_winner_change,
        recency_bonus=arm.recency_bonus,
        recency_half_life=arm.recency_half_life,
        recall_recency_bonus=arm.recall_recency_bonus,
        recall_recency_half_life=arm.recall_recency_half_life,
    )
    if selected is None:
        return off
    arm.proposals += 1
    *_, read, on, _disruption, _overturn, _key, decision, _age, _recall_age = selected
    if live and decision.permitted:
        arm.permitted += 1
        arm.memory.reinstate(read)
        total_variation = 0.5 * float(np.sum(np.abs(on - off)))
        arm.permitted_total_variation += total_variation
        arm.maximum_permitted_total_variation = max(
            arm.maximum_permitted_total_variation, total_variation
        )
        return on
    arm.memory.touch(read)
    return off


def _observe(arm: _Arm, byte_id: int, probability: float) -> None:
    arm.learner.advance_observation(byte_id, learn=False)
    surprise = -log2(max(probability, 1e-300))
    routes = set(arm.learner._active_route_keys())
    union = len(routes | arm.previous_routes)
    turnover = 0.0 if not union else 1.0 - len(routes & arm.previous_routes) / union
    disagreement = arm.learner._temporal_disagreement(tuple(routes))
    boundary_signal = abs(surprise - arm.previous_surprise) + turnover + disagreement
    arm.raw_window.append(byte_id)
    if len(arm.raw_window) > 3:
        del arm.raw_window[0]
    query = int.from_bytes(arm.raw_window, "big") if len(arm.raw_window) == 3 else None
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    active_items = tuple(
        (
            receptor,
            assembly,
            len(arm.learner.word_forms[assembly])
            / max(float(arm.learner.motif_activity[assembly]), 1e-12) ** 0.5,
        )
        for receptor, assembly in routes
        if 0 <= assembly < len(arm.learner.word_forms)
        and assembly not in arm.learner.temporal_recalled_sources
    )
    arm.memory.observe_boundary(
        boundary_signal,
        query=query,
        active_items=active_items,
        basin=basin,
    )
    arm.previous_routes = routes
    arm.previous_surprise = surprise


def _generate(
    payload: dict[str, object],
    prompt: bytes,
    *,
    live: bool,
    seed: int,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[bytes, _Arm]:
    arm = _new_arm(payload)
    for byte_id in prompt:
        probability = arm.learner.predict_next_distribution()
        _observe(arm, byte_id, float(probability[byte_id]))
    rng = np.random.default_rng(seed)
    generated = bytearray()
    for _ in range(max_new_bytes):
        probability = _predict(arm, live=live)
        byte_id = _sample_byte(
            probability,
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        _observe(arm, byte_id, float(probability[byte_id]))
    return bytes(generated), arm


def run_ablation(
    state_path: Path,
    *,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    samples_per_prompt: int = 3,
    max_new_bytes: int = 1_000,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    seed: int = 0,
) -> dict[str, object]:
    with state_path.open("rb") as handle:
        payload = pickle.load(handle)
    boundary = getattr(payload["learner"], "story_boundary", b"<|endoftext|>\n")
    pairs = []
    for prompt_index, prompt_text in enumerate(prompts):
        prompt = prompt_text.encode()
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            on, on_arm = _generate(
                payload,
                prompt,
                live=True,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            shadow, shadow_arm = _generate(
                payload,
                prompt,
                live=False,
                seed=sample_seed,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            pairs.append(
                {
                    "prompt": prompt_text,
                    "seed": sample_seed,
                    "byte_identical": on == shadow,
                    "first_divergence_byte": _first_divergence(on, shadow),
                    "live_proposals": on_arm.proposals,
                    "live_permitted_reads": on_arm.permitted,
                    "live_permitted_total_variation": (
                        on_arm.permitted_total_variation
                    ),
                    "maximum_permitted_total_variation": (
                        on_arm.maximum_permitted_total_variation
                    ),
                    "shadow_proposals": shadow_arm.proposals,
                    "permission_on_text": (prompt + on).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "shadow_only_text": (prompt + shadow).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "permission_on_diagnostics": _generation_diagnostics(on, boundary),
                    "shadow_only_diagnostics": _generation_diagnostics(
                        shadow, boundary
                    ),
                }
            )
    changed = [not pair["byte_identical"] for pair in pairs]
    return {
        "task": "phase74_paired_autonomous_generation",
        "state": str(state_path),
        "persistent_permission_updates_during_generation": False,
        "pairs": pairs,
        "summary": {
            "pairs": len(pairs),
            "changed_pairs": sum(changed),
            "changed_fraction": float(np.mean(changed)),
            "live_permitted_reads": sum(pair["live_permitted_reads"] for pair in pairs),
            "live_permitted_total_variation": sum(
                pair["live_permitted_total_variation"] for pair in pairs
            ),
            "maximum_permitted_total_variation": max(
                pair["maximum_permitted_total_variation"] for pair in pairs
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--prompt-index",
        type=int,
        choices=range(len(DEFAULT_PROMPTS)),
        help="run one default prompt so independent prompt groups can be sharded",
    )
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-new-bytes", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prompts = (
        (DEFAULT_PROMPTS[args.prompt_index],)
        if args.prompt_index is not None
        else tuple(args.prompts)
    )
    report = run_ablation(
        args.state,
        prompts=prompts,
        samples_per_prompt=args.samples_per_prompt,
        max_new_bytes=args.max_new_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
