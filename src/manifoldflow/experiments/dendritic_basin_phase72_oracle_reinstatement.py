"""Test whether the existing cortex can use an evaluator-supplied memory.

This is deliberately an oracle diagnostic, not part of the online algorithm.
The evaluator uses the observed next motif to select an already learned,
positive-utility route, then reinstates its filler through its recorded
receptor from the same pre-word state.
"""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)


def _is_letter(byte_id: int) -> bool:
    return 65 <= byte_id <= 90 or 97 <= byte_id <= 122


def _effect_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    positive = array[array > 0.0]
    negative = array[array < 0.0]
    gross_positive = float(positive.sum())
    return {
        "events": len(values),
        "total_bits_saved": float(array.sum()),
        "gross_positive_bits": gross_positive,
        "gross_negative_bits": float(negative.sum()),
        "win_fraction": float(np.mean(array > 0.0)) if len(array) else 0.0,
        "median_bits_saved": float(np.median(array)) if len(array) else 0.0,
        "largest_rescue_bits": float(np.max(array)) if len(array) else 0.0,
        "largest_rescue_share_of_gross_positive": (
            float(np.max(positive) / gross_positive) if gross_positive else 0.0
        ),
    }


def _candidate_index(learner: object) -> dict[int, list[tuple[int, int]]]:
    by_target: dict[int, list[tuple[int, int]]] = {}
    for key, row in learner.route_counts.items():
        if learner.route_utility.get(key, 0.0) <= learner.minimum_route_utility_bits:
            continue
        for target, observations in row.items():
            if observations > 0:
                by_target.setdefault(int(target), []).append(key)
    return by_target


def _route_score(learner: object, key: tuple[int, int], target: int) -> float:
    probability = learner._route_probability(key, target)
    prior = learner._target_prior(target)
    return log2(max(probability, 1e-300) / max(prior, 1e-300))


def _select_correct_route(
    learner: object,
    candidates: list[tuple[int, int]],
    target: int,
) -> tuple[int, int] | None:
    absent = [
        key
        for key in candidates
        if key[1] != target
        and key[1] not in learner.temporal_last_seen
        and _route_score(learner, key, target) > 0.0
    ]
    if not absent:
        return None
    return max(
        absent,
        key=lambda key: (
            _route_score(learner, key, target),
            learner.route_utility[key],
            learner.route_totals[key],
        ),
    )


def _alternative_filler(
    learner: object,
    receptor: int,
    correct_filler: int,
    target: int,
) -> int | None:
    candidates = [
        source
        for candidate_receptor, source in learner.route_counts
        if candidate_receptor == receptor
        and source != correct_filler
        and source != target
        and source not in learner.temporal_last_seen
        and learner.route_utility.get((receptor, source), 0.0)
        > learner.minimum_route_utility_bits
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda source: _route_score(learner, (receptor, source), target),
    )


def _inject(learner: object, filler: int | None, receptor: int | None) -> None:
    if filler is None or receptor is None:
        return
    learner.temporal_last_seen[filler] = learner.temporal_clock - receptor + 1
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    learner.temporal_recalled_sources.add(filler)
    learner._goal_route_projection_cache = None


def _score_bytes(learner: object, observed: bytes) -> float:
    bits = 0.0
    for byte_id in observed:
        probability = learner._observe_probability_only(byte_id, learn=False)
        if probability is not None:
            bits += -log2(max(probability, 1e-300))
    return bits


def run_oracle_probe(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    eval_offset: int = 0,
    max_events: int = 64,
) -> dict[str, object]:
    learner = _read_checkpoint(checkpoint)
    # Natural binder activity would confound this receiving-path assay.
    learner.binding_index = None
    learner.reset_episode()
    by_target = _candidate_index(learner)
    with eval_path.open("rb") as handle:
        handle.seek(eval_offset)
        stream = np.fromfile(handle, dtype=np.uint8, count=eval_bytes)
    events: list[dict[str, object]] = []
    for position, byte_id in enumerate(stream):
        byte = int(byte_id)
        word_start = _is_letter(byte) and (
            position == 0 or not _is_letter(int(stream[position - 1]))
        )
        if word_start and len(events) < max_events:
            end = position
            while end < len(stream) and _is_letter(int(stream[end])):
                end += 1
            form = bytes(stream[position:end]).lower()
            target = learner.word_ids.get(form)
            if target is not None:
                key = _select_correct_route(
                    learner, by_target.get(int(target), []), int(target)
                )
                if key is not None:
                    receptor, filler = key
                    alternative = _alternative_filler(
                        learner, receptor, filler, int(target)
                    )
                    receptors = tuple(learner.temporal_bins)
                    wrong_receptor = receptors[
                        (receptors.index(receptor) + 1) % len(receptors)
                    ]
                    snapshot = pickle.dumps(learner, protocol=pickle.HIGHEST_PROTOCOL)
                    branches = {
                        name: pickle.loads(snapshot)
                        for name in (
                            "off",
                            "correct",
                            "wrong_receptor",
                            "alternative_filler",
                        )
                    }
                    _inject(branches["correct"], filler, receptor)
                    _inject(branches["wrong_receptor"], filler, wrong_receptor)
                    _inject(branches["alternative_filler"], alternative, receptor)
                    # Include the following delimiter so event completion is scored.
                    finish = min(end + 1, len(stream))
                    observed = bytes(stream[position:finish])
                    bits = {
                        name: _score_bytes(branch, observed)
                        for name, branch in branches.items()
                    }
                    events.append(
                        {
                            "position": eval_offset + position,
                            "form_hex": form.hex(),
                            "target": int(target),
                            "correct_filler": int(filler),
                            "recorded_receptor": int(receptor),
                            "wrong_receptor": int(wrong_receptor),
                            "alternative_filler": alternative,
                            "route_utility": float(learner.route_utility[key]),
                            "route_information_bits": _route_score(
                                learner, key, int(target)
                            ),
                            "scored_bytes": len(observed),
                            "correct_bits_saved": bits["off"] - bits["correct"],
                            "wrong_receptor_bits_saved": (
                                bits["off"] - bits["wrong_receptor"]
                            ),
                            "alternative_filler_bits_saved": (
                                None
                                if alternative is None
                                else bits["off"] - bits["alternative_filler"]
                            ),
                        }
                    )
        learner._observe_probability_only(byte, learn=False)
    correct = [float(event["correct_bits_saved"]) for event in events]
    wrong = [float(event["wrong_receptor_bits_saved"]) for event in events]
    alternative = [
        float(event["alternative_filler_bits_saved"])
        for event in events
        if event["alternative_filler_bits_saved"] is not None
    ]
    return {
        "task": "phase72_oracle_reinstatement_receiving_path",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_offset": eval_offset,
        "eval_bytes": len(stream),
        "events": events,
        "correct_memory": _effect_summary(correct),
        "same_filler_wrong_receptor": _effect_summary(wrong),
        "alternative_filler_same_receptor": _effect_summary(alternative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_oracle_probe(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        eval_offset=args.eval_offset,
        max_events=args.max_events,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
