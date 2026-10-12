"""Develop bounded contextual successor tissue before frozen generation probes."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments import (
    dendritic_basin_phase86_context_recruitment as phase86,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _observe,
    _predict,
    _predict_encoding_only,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    _lexical_outcome,
)


def _development_stream(
    tiny: bytes,
    smol: bytes,
    *,
    mode: str,
    byte_count: int,
    chunk_bytes: int,
) -> bytes:
    if mode == "tiny":
        return tiny[:byte_count]
    output = bytearray()
    offsets = [0, 0]
    corpora = (tiny, smol)
    turn = 0
    while len(output) < byte_count:
        corpus = corpora[turn % 2]
        offset = offsets[turn % 2]
        if offset >= len(corpus):
            raise ValueError("development corpus is shorter than requested exposure")
        take = min(chunk_bytes, byte_count - len(output), len(corpus) - offset)
        output.extend(corpus[offset : offset + take])
        offsets[turn % 2] += take
        turn += 1
    return bytes(output)


def _clear_fast_state(arm: object) -> None:
    """Start a new episode while retaining developed long-term tissue."""
    arm.learner.reset_episode()
    arm.field.reset_episode()
    arm.event_memory.capsules.clear()
    arm.event_memory.reset_episode()
    arm.route_history.clear()
    arm.raw_window.clear()
    arm.previous_routes.clear()
    arm.delayed_frames.clear()
    arm.fifo_replay.clear()
    arm.engram_frames.clear()
    arm.multiscale.reset_episode()
    arm.relational_access.reset_episode()
    arm.recalled_expression.reset_episode()
    if getattr(arm, "temporal_relational_eligibility", None) is not None:
        arm.temporal_relational_eligibility.reset_episode()
    arm.pending_access_context = ()
    arm.pending_access_sequence_cue = ()
    arm.return_adaptation.clear()
    arm.previous_surprise = 0.0
    arm.last_boundary_signal = 0.0


def _prime(
    developed: object,
    prompt: bytes,
    *,
    return_mode: str = "contextual_successor",
) -> object:
    arm = pickle.loads(pickle.dumps(developed, protocol=pickle.HIGHEST_PROTOCOL))
    _clear_fast_state(arm)
    arm.recall_enabled = True
    arm.access_learning_enabled = True
    arm.expression_learning_enabled = True
    arm.access_recruitment_mode = (
        "branch_peak" if return_mode == "contextual_successor" else "none"
    )
    arm.access_return_mode = return_mode
    for byte_id in prompt:
        distribution = _predict_encoding_only(arm)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)
    phase85._calibrate_expression(arm)
    return arm


def _score_suffix(prepared: object, suffix: bytes, *, recall: bool) -> float:
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    arm.recall_enabled = recall
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    arm.return_adaptation_half_life = 8.0
    bits = 0.0
    for byte_id in suffix:
        distribution = _predict(arm, live=recall)
        bits -= log2(max(float(distribution[byte_id]), 1e-300))
        _observe(arm, byte_id, float(distribution[byte_id]), external=False)
    return bits


def _branch_statistics(field: object) -> dict[str, object]:
    counts = [len(branches) for branches in field.engram_contextual_successors]
    return {
        "engrams": len(counts),
        "branches": int(sum(counts)),
        "maximum_per_engram": int(max(counts, default=0)),
        "mean_per_engram": float(np.mean(counts)) if counts else 0.0,
        "births": int(field.contextual_successor_births),
        "deaths": int(field.contextual_successor_deaths),
        "capacity": int(field.contextual_successor_capacity),
    }


def run(
    state: Path,
    tiny_path: Path,
    smol_path: Path,
    *,
    mode: str,
    development_bytes: int,
    chunk_bytes: int,
    max_new_bytes: int,
    seed: int,
    competition_gain: float = 0.0,
    case_limit: int | None = None,
    access_learning_mode: str = "hebbian",
    return_mode: str = "contextual_successor",
    return_cancellation: str = "none",
    return_current_limit: float = 0.0,
) -> tuple[dict[str, object], object]:
    started = perf_counter()
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    field = payload["field"]
    field.contextual_successor_binding = True
    field.contextual_successor_branches_per_source = 4
    field.contextual_successor_capacity = 256
    field.contextual_successor_competition_gain = competition_gain
    tiny = tiny_path.read_bytes()
    smol = smol_path.read_bytes()
    stream = _development_stream(
        tiny,
        smol,
        mode=mode,
        byte_count=development_bytes,
        chunk_bytes=chunk_bytes,
    )
    development_started = perf_counter()
    developed = phase85._prepare(
        payload, stream, access_learning_mode=access_learning_mode
    )
    developed.expression_cancellation_mode = return_cancellation
    developed.expression_current_limit = return_current_limit
    development_elapsed = perf_counter() - development_started
    rows = []
    cases = CASES if case_limit is None else CASES[:case_limit]
    for case_index, case in enumerate(cases):
        prepared = _prime(
            developed, case.prompt.encode(), return_mode=return_mode
        )
        expected = case.expected.encode()
        contrast = case.contrast.encode()
        forced = {}
        for label, suffix in (("expected", expected), ("contrast", contrast)):
            off_bits = _score_suffix(prepared, suffix, recall=False)
            on_bits = _score_suffix(prepared, suffix, recall=True)
            forced[label] = {
                "off_bits": off_bits,
                "on_bits": on_bits,
                "saved_bits": off_bits - on_bits,
            }
        generated = {}
        live_condition = (
            "adapted_contextual_successor"
            if return_mode == "contextual_successor"
            else "adapted_unconditioned"
        )
        for condition in ("recall_disabled", live_condition):
            output, arm, frozen, completions = phase86._generate(
                prepared,
                condition,
                seed=seed + case_index,
                max_new_bytes=max_new_bytes,
                temperature=0.8,
                top_k=32,
                top_p=0.9,
            )
            generated[condition] = {
                "text": output.decode("utf-8", errors="backslashreplace"),
                "outcome": _lexical_outcome(output, expected, contrast),
                "access_completions": completions,
                "associations_unchanged": frozen,
                "equals_baseline": False,
            }
        generated[live_condition]["equals_baseline"] = (
            generated[live_condition]["text"]
            == generated["recall_disabled"]["text"]
        )
        rows.append(
            {
                "case": case.name,
                "forced": forced,
                "generation": generated,
                "post_prompt_branches": _branch_statistics(prepared.field),
                "live_condition": live_condition,
            }
        )
    elapsed = perf_counter() - started
    report = {
        "task": "phase91_developed_contextual_successor",
        "mode": mode,
        "state": str(state),
        "development_bytes": development_bytes,
        "chunk_bytes": chunk_bytes,
        "competition_gain": competition_gain,
        "access_learning_mode": access_learning_mode,
        "return_mode": return_mode,
        "return_cancellation": return_cancellation,
        "return_current_limit": return_current_limit,
        "development_seconds": development_elapsed,
        "development_bytes_per_second": (
            development_bytes / development_elapsed
            if development_elapsed > 0.0
            else 0.0
        ),
        "developed_tissue": _branch_statistics(developed.field),
        "controlled_expected_generation_fraction": float(
            np.mean(
                [
                    row["generation"][row["live_condition"]]["outcome"]
                    == "expected"
                    for row in rows
                ]
            )
        ),
        "mean_expected_saved_bits": float(
            np.mean([row["forced"]["expected"]["saved_bits"] for row in rows])
        ),
        "mean_contrast_saved_bits": float(
            np.mean([row["forced"]["contrast"]["saved_bits"] for row in rows])
        ),
        "elapsed_seconds": elapsed,
        "rows": rows,
    }
    return report, developed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--tiny-data", type=Path, required=True)
    parser.add_argument("--smol-data", type=Path, required=True)
    parser.add_argument("--mode", choices=("tiny", "mixed"), required=True)
    parser.add_argument("--development-bytes", type=int, default=2_000)
    parser.add_argument("--chunk-bytes", type=int, default=256)
    parser.add_argument("--max-new-bytes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--competition-gain", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument(
        "--access-learning-mode",
        choices=("hebbian", "prediction_error"),
        default="hebbian",
    )
    parser.add_argument(
        "--return-mode",
        choices=("direct", "contextual_successor"),
        default="contextual_successor",
    )
    parser.add_argument(
        "--return-cancellation", choices=("none", "pre", "post"), default="none"
    )
    parser.add_argument("--return-current-limit", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--developed-state", type=Path)
    args = parser.parse_args()
    report, developed = run(
        args.state,
        args.tiny_data,
        args.smol_data,
        mode=args.mode,
        development_bytes=args.development_bytes,
        chunk_bytes=args.chunk_bytes,
        max_new_bytes=args.max_new_bytes,
        seed=args.seed,
        competition_gain=args.competition_gain,
        case_limit=args.case_limit,
        access_learning_mode=args.access_learning_mode,
        return_mode=args.return_mode,
        return_cancellation=args.return_cancellation,
        return_current_limit=args.return_current_limit,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.developed_state is not None:
        args.developed_state.parent.mkdir(parents=True, exist_ok=True)
        with args.developed_state.open("wb") as handle:
            pickle.dump(developed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(encoded, end="")


if __name__ == "__main__":
    main()
