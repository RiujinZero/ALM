"""Measure exact final-probability effects of transient cortical reinstatement."""

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


def sampling_distribution(
    distribution: np.ndarray, *, temperature: float, top_k: int, top_p: float
) -> np.ndarray:
    """Apply the actual generation transforms without drawing a sample."""
    probability = np.asarray(distribution, dtype=np.float64).copy()
    if temperature == 0.0 or top_k == 1:
        result = np.zeros(256, dtype=np.float64)
        result[int(np.argmax(probability))] = 1.0
        return result
    logits = np.log(np.maximum(probability, 1e-300)) / temperature
    if 0 < top_k < 256:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full(256, -np.inf, dtype=np.float64)
        masked[keep] = logits[keep]
        logits = masked
    logits -= np.max(logits)
    probability = np.exp(logits)
    probability /= probability.sum()
    if top_p < 1.0:
        order = np.argsort(probability)[::-1]
        keep_count = int(
            np.searchsorted(np.cumsum(probability[order]), top_p, side="left")
        ) + 1
        result = np.zeros(256, dtype=np.float64)
        result[order[:keep_count]] = probability[order[:keep_count]]
        probability = result / result.sum()
    return probability


def _alternative_filler(
    learner: object, basin: int, role: int, selected: int
) -> int | None:
    index = learner.binding_index
    cells = np.flatnonzero(
        (index.basin == basin)
        & (index.role == role)
        & (index.filler >= 0)
        & (index.filler != selected)
    )
    if not len(cells):
        return None
    winner = int(max(cells, key=lambda cell: float(index.activity[cell])))
    filler = int(index.filler[winner])
    return filler if filler < len(learner.word_forms) else None


def _intervene(
    learner: object,
    recall: tuple[int, int, int, int],
    mode: str,
) -> tuple[int | None, int | None]:
    basin, role, filler, receptor = recall
    learner.temporal_last_seen.pop(filler, None)
    getattr(learner, "temporal_recalled_sources", set()).discard(filler)
    learner._binding_pending_credit = None
    inserted_filler: int | None = None
    inserted_receptor: int | None = None
    if mode == "selected":
        inserted_filler, inserted_receptor = filler, receptor
    elif mode == "wrong_receptor":
        receptors = tuple(learner.temporal_bins)
        position = receptors.index(receptor)
        inserted_filler = filler
        inserted_receptor = receptors[(position + 1) % len(receptors)]
    elif mode == "alternative_filler":
        inserted_filler = _alternative_filler(learner, basin, role, filler)
        inserted_receptor = receptor if inserted_filler is not None else None
    elif mode != "off":
        raise ValueError(f"unknown read intervention: {mode}")
    if inserted_filler is not None and inserted_receptor is not None:
        learner.temporal_last_seen[inserted_filler] = (
            learner.temporal_clock - inserted_receptor + 1
        )
        if not hasattr(learner, "temporal_recalled_sources"):
            learner.temporal_recalled_sources = set()
        learner.temporal_recalled_sources.add(inserted_filler)
    learner._goal_route_projection_cache = None
    return inserted_filler, inserted_receptor


def _branch(
    snapshot: bytes,
    recall: tuple[int, int, int, int],
    mode: str,
) -> tuple[object, int | None, int | None]:
    learner = pickle.loads(snapshot)
    filler, receptor = _intervene(learner, recall, mode)
    return learner, filler, receptor


def _score_horizon(learner: object, future: np.ndarray) -> float:
    bits = 0.0
    for byte_id in future:
        probability = learner._observe_probability_only(int(byte_id), learn=False)
        if probability is not None:
            bits += -log2(max(probability, 1e-300))
    return bits


def _effect_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    positive = array[array > 0.0]
    negative = array[array < 0.0]
    gross_positive = float(np.sum(positive))
    return {
        "events": len(values),
        "total_bits_saved": float(np.sum(array)),
        "gross_positive_bits": gross_positive,
        "gross_negative_bits": float(np.sum(negative)),
        "win_fraction": float(np.mean(array > 0.0)) if len(array) else 0.0,
        "median_bits_saved": float(np.median(array)) if len(array) else 0.0,
        "largest_rescue_bits": float(np.max(array)) if len(array) else 0.0,
        "largest_rescue_share_of_gross_positive": (
            float(np.max(positive) / gross_positive) if gross_positive else 0.0
        ),
    }


def run_readout(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    max_events: int = 32,
    horizon_bytes: int = 16,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
) -> dict[str, object]:
    learner = _read_checkpoint(checkpoint)
    learner.reset_episode()
    stream = np.fromfile(eval_path, dtype=np.uint8, count=eval_bytes)
    events: list[dict[str, object]] = []
    for position, byte_id in enumerate(stream[:-1]):
        before = int(getattr(learner, "binding_reinstatements", 0))
        learner._observe_probability_only(int(byte_id), learn=False)
        after = int(getattr(learner, "binding_reinstatements", 0))
        recall = getattr(learner, "_binding_last_reinstatement", None)
        if after == before or recall is None or len(events) >= max_events:
            continue
        snapshot = pickle.dumps(learner, protocol=pickle.HIGHEST_PROTOCOL)
        branches = {
            mode: _branch(snapshot, recall, mode)
            for mode in ("selected", "off", "wrong_receptor", "alternative_filler")
        }
        distributions = {
            mode: branch[0].predict_next_distribution()
            for mode, branch in branches.items()
        }
        observed = int(stream[position + 1])
        off_probability = float(distributions["off"][observed])
        selected_probability = float(distributions["selected"][observed])
        wrong_probability = float(distributions["wrong_receptor"][observed])
        alternative = branches["alternative_filler"][1]
        alternative_probability = float(
            distributions["alternative_filler"][observed]
        )
        selected_sampling = sampling_distribution(
            distributions["selected"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        off_sampling = sampling_distribution(
            distributions["off"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        end = min(len(stream), position + 1 + horizon_bytes)
        future = stream[position + 1 : end]
        horizon_selected = _score_horizon(branches["selected"][0], future)
        horizon_off = _score_horizon(branches["off"][0], future)
        role = int(recall[1])
        proxy = float(learner._binding_role_utility(role))
        selected_route = int(recall[3]), int(recall[2])
        wrong_route = branches["wrong_receptor"][2], int(recall[2])
        selected_route_utility = learner.route_utility.get(selected_route)
        wrong_route_utility = learner.route_utility.get(wrong_route)
        selected_effect = log2(
            max(selected_probability, 1e-300) / max(off_probability, 1e-300)
        )
        events.append(
            {
                "position": position + 1,
                "basin": int(recall[0]),
                "query_role": role,
                "selected_filler": int(recall[2]),
                "recorded_receptor": int(recall[3]),
                "selected_route_utility": selected_route_utility,
                "selected_route_output_eligible": bool(
                    selected_route_utility is not None
                    and selected_route_utility > learner.minimum_route_utility_bits
                ),
                "wrong_receptor_route_utility": wrong_route_utility,
                "alternative_filler": alternative,
                "observed_byte": observed,
                "selected_bits_saved": selected_effect,
                "wrong_receptor_bits_saved": log2(
                    max(wrong_probability, 1e-300)
                    / max(off_probability, 1e-300)
                ),
                "alternative_filler_bits_saved": (
                    None
                    if alternative is None
                    else log2(
                        max(alternative_probability, 1e-300)
                        / max(off_probability, 1e-300)
                    )
                ),
                "sampling_total_variation": float(
                    0.5 * np.sum(np.abs(selected_sampling - off_sampling))
                ),
                "proxy_utility": proxy,
                "proxy_and_exact_sign_agree": bool(
                    (proxy > 0.0) == (selected_effect > 0.0)
                ),
                "horizon_bytes": len(future),
                "horizon_bits_saved": horizon_off - horizon_selected,
            }
        )
    selected = [float(event["selected_bits_saved"]) for event in events]
    wrong = [float(event["wrong_receptor_bits_saved"]) for event in events]
    alternatives = [
        float(event["alternative_filler_bits_saved"])
        for event in events
        if event["alternative_filler_bits_saved"] is not None
    ]
    horizon = [float(event["horizon_bits_saved"]) for event in events]
    return {
        "task": "phase72_exact_causal_reinstatement_readout",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": len(stream),
        "max_events": max_events,
        "selected_read": _effect_summary(selected),
        "wrong_receptor": _effect_summary(wrong),
        "alternative_filler": _effect_summary(alternatives),
        "short_horizon": _effect_summary(horizon),
        "mean_sampling_total_variation": float(
            np.mean([event["sampling_total_variation"] for event in events])
        )
        if events
        else 0.0,
        "proxy_exact_sign_agreement_fraction": float(
            np.mean([event["proxy_and_exact_sign_agree"] for event in events])
        )
        if events
        else 0.0,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--horizon-bytes", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_readout(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        max_events=args.max_events,
        horizon_bytes=args.horizon_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
