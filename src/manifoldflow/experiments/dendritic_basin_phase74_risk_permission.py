"""Calibrate risk-sensitive permission for entropy-restorative capsule recalls."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
    CapsuleRead,
)
from manifoldflow.dendritic_basin.risk_permission import RiskSensitivePermissionBank
from manifoldflow.experiments.dendritic_basin_generate import _read_checkpoint
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_VALID_PATH,
)
from manifoldflow.experiments.dendritic_basin_phase73_shadow_capsules import (
    _content_match,
    _NoopBindingIndex,
    _temporary_distribution,
)


def _bucket(value: float, thresholds: tuple[float, ...]) -> int:
    return sum(value > threshold for threshold in thresholds)


def _feature_key(
    *,
    receptor: int,
    restoration: float,
    disruption: float,
    overturn: int,
    context: str,
) -> tuple[int, ...]:
    restoration_bin = _bucket(restoration, (0.001, 0.01, 0.05, 0.1, 0.25, 0.5))
    disruption_bin = _bucket(disruption, (0.001, 0.01, 0.05, 0.1, 0.25))
    base = (restoration_bin, disruption_bin, overturn)
    return (receptor, *base) if context == "receptor" else base


def _calibration_bins(
    events: list[dict[str, float]], *, catastrophic_threshold: float
) -> list[dict[str, float | int]]:
    if not events:
        return []
    order = np.argsort([event["permission_score"] for event in events])
    result: list[dict[str, float | int]] = []
    for indices in np.array_split(order, min(5, len(order))):
        selected = [events[int(index)] for index in indices]
        utility = np.asarray(
            [event["utility_bits"] for event in selected], dtype=np.float64
        )
        tail_count = max(1, int(np.ceil(0.01 * len(utility))))
        result.append(
            {
                "reads": len(selected),
                "minimum_score": min(event["permission_score"] for event in selected),
                "maximum_score": max(event["permission_score"] for event in selected),
                "mean_score": float(
                    np.mean([event["permission_score"] for event in selected])
                ),
                "positive_fraction": float(np.mean(utility > 0.0)),
                "mean_utility_bits": float(np.mean(utility)),
                "catastrophic_fraction": float(
                    np.mean(utility < -catastrophic_threshold)
                ),
                "worst_one_percent_mean_bits": float(
                    np.mean(np.sort(utility)[:tail_count])
                ),
            }
        )
    return result


def _utility_summary(
    events: list[dict[str, float]], *, catastrophic_threshold: float
) -> dict[str, float | int]:
    utility = np.asarray([event["utility_bits"] for event in events], dtype=np.float64)
    if not len(utility):
        return {
            "reads": 0,
            "total_utility_bits": 0.0,
            "mean_utility_bits": 0.0,
            "positive_fraction": 0.0,
            "catastrophic_fraction": 0.0,
            "worst_one_percent_mean_bits": 0.0,
        }
    tail_count = max(1, int(np.ceil(0.01 * len(utility))))
    return {
        "reads": len(utility),
        "total_utility_bits": float(utility.sum()),
        "mean_utility_bits": float(utility.mean()),
        "positive_fraction": float(np.mean(utility > 0.0)),
        "catastrophic_fraction": float(np.mean(utility < -catastrophic_threshold)),
        "worst_one_percent_mean_bits": float(np.mean(np.sort(utility)[:tail_count])),
    }


def _select_proposal(
    learner: object,
    memory: BoundaryEventCapsuleMemory,
    permission: RiskSensitivePermissionBank,
    *,
    query: int | None,
    basin: int | None,
    off: np.ndarray,
    candidate_limit: int,
    feature_context: str,
    veto_winner_change: bool,
    recency_bonus: float = 0.0,
    recency_half_life: float = 256.0,
    recall_recency_bonus: float = 0.0,
    recall_recency_half_life: float = 256.0,
) -> (
    tuple[
        float,
        float,
        CapsuleRead,
        np.ndarray,
        float,
        int,
        tuple[int, ...],
        object,
        float,
        float | None,
    ]
    | None
):
    if query is None:
        return None
    off_entropy = -float(np.sum(off * np.log2(np.maximum(off, 1e-300))))
    options: list[tuple[float, float, float | None, CapsuleRead]] = []
    for candidate in memory.candidates(query, basin, limit=8):
        capsule = memory.capsules[candidate.capsule]
        created_at = capsule.__dict__.get("created_at", capsule.last_used)
        age = max(0.0, float(memory.clock - created_at))
        recency = 2.0 ** (-age / recency_half_life)
        last_recalled = capsule.__dict__.get("last_recalled")
        recall_age = (
            None
            if last_recalled is None
            else max(0.0, float(memory.clock - last_recalled))
        )
        recall_recency = (
            0.0
            if recall_age is None
            else 2.0 ** (-recall_age / recall_recency_half_life)
        )
        for item in candidate.items:
            if (
                item[0] in learner.temporal_last_seen
                or learner.route_utility.get((item[1], item[0]), 0.0)
                <= learner.minimum_route_utility_bits
            ):
                continue
            match = _content_match(learner, item)
            if match > 0.0:
                options.append(
                    (
                        match
                        * (
                            1.0
                            + recency_bonus * recency
                            + recall_recency_bonus * recall_recency
                        ),
                        age,
                        recall_age,
                        CapsuleRead(
                            candidate.capsule,
                            candidate.query,
                            item[1],
                            (item,),
                            0.0,
                            0,
                        ),
                    )
                )
    options.sort(key=lambda pair: pair[0], reverse=True)
    proposals = []
    seen: set[tuple[int, int]] = set()
    for proposal_match, age, recall_age, read in options:
        if read.items[0] in seen:
            continue
        seen.add(read.items[0])
        on = _temporary_distribution(learner, read)
        on_entropy = -float(np.sum(on * np.log2(np.maximum(on, 1e-300))))
        restoration = off_entropy - on_entropy
        if restoration <= 0.0:
            continue
        disruption = 0.5 * float(np.sum(np.abs(on - off)))
        winner_changed = int(np.argmax(on) != np.argmax(off))
        overturn = 2 if winner_changed and float(np.max(off)) >= 0.5 else winner_changed
        if veto_winner_change and overturn:
            continue
        key = _feature_key(
            receptor=read.receptor,
            restoration=restoration,
            disruption=disruption,
            overturn=overturn,
            context=feature_context,
        )
        proposals.append(
            (
                restoration,
                proposal_match,
                read,
                on,
                disruption,
                overturn,
                key,
                permission.decide(key),
                age,
                recall_age,
            )
        )
        if len(proposals) >= candidate_limit:
            break
    return max(proposals, key=lambda value: value[0]) if proposals else None


def run_probe(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 1_000_000,
    eval_offset: int = 0,
    calibration_bytes: int | None = None,
    candidate_limit: int = 4,
    feature_context: str = "pooled",
    minimum_evidence: int = 16,
    confidence_width: float = 2.0,
    catastrophic_threshold: float = 0.1,
    tail_risk_penalty: float = 0.5,
    maximum_catastrophic_rate: float = 0.05,
    veto_winner_change: bool = False,
    recency_bonus: float = 0.0,
    recency_half_life: float = 256.0,
    recall_recency_bonus: float = 0.0,
    recall_recency_half_life: float = 256.0,
    state_output: Path | None = None,
) -> dict[str, object]:
    if (
        recency_bonus < 0.0
        or recency_half_life <= 0.0
        or recall_recency_bonus < 0.0
        or recall_recency_half_life <= 0.0
    ):
        raise ValueError("recency bonus and half-life must be non-negative/positive")
    learner = _read_checkpoint(checkpoint)
    learner.reset_episode()
    learner.binding_index = _NoopBindingIndex()
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    memory = BoundaryEventCapsuleMemory(permission_context="query_receptor")
    permission = RiskSensitivePermissionBank(
        minimum_evidence=minimum_evidence,
        confidence_width=confidence_width,
        catastrophic_threshold=catastrophic_threshold,
        tail_risk_penalty=tail_risk_penalty,
        maximum_catastrophic_rate=maximum_catastrophic_rate,
    )
    if eval_offset < 0:
        raise ValueError("eval_offset must be non-negative")
    with eval_path.open("rb") as handle:
        handle.seek(eval_offset)
        stream = np.frombuffer(handle.read(eval_bytes), dtype=np.uint8)
    calibration_bytes = (
        len(stream) // 2 if calibration_bytes is None else calibration_bytes
    )
    raw_window = bytearray()
    previous_routes: set[tuple[int, int]] = set()
    previous_surprise = 0.0
    baseline_bits = live_bits = 0.0
    calibration_events: list[dict[str, float]] = []
    evaluation_events: list[dict[str, float]] = []

    for position, byte_value in enumerate(stream):
        if position == calibration_bytes and state_output is not None:
            state_output.parent.mkdir(parents=True, exist_ok=True)
            with state_output.open("wb") as handle:
                pickle.dump(
                    {
                        "learner": learner,
                        "memory": memory,
                        "permission": permission,
                        "feature_context": feature_context,
                        "candidate_limit": candidate_limit,
                        "veto_winner_change": veto_winner_change,
                        "recency_bonus": recency_bonus,
                        "recency_half_life": recency_half_life,
                        "recall_recency_bonus": recall_recency_bonus,
                        "recall_recency_half_life": recall_recency_half_life,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        byte_id = int(byte_value)
        evaluating = position >= calibration_bytes
        off = learner.predict_next_distribution()
        off_probability = float(off[byte_id])
        query = int.from_bytes(raw_window, "big") if len(raw_window) == 3 else None
        basin_organ = getattr(learner, "predictive_goal_organ", None)
        basin = None if basin_organ is None else basin_organ.current_basin
        selected = _select_proposal(
            learner,
            memory,
            permission,
            query=query,
            basin=basin,
            off=off,
            candidate_limit=candidate_limit,
            feature_context=feature_context,
            veto_winner_change=veto_winner_change,
            recency_bonus=recency_bonus,
            recency_half_life=recency_half_life,
            recall_recency_bonus=recall_recency_bonus,
            recall_recency_half_life=recall_recency_half_life,
        )
        live_probability = off_probability
        if selected is not None:
            (
                restoration,
                match,
                read,
                on,
                disruption,
                overturn,
                key,
                decision,
                age,
                recall_age,
            ) = selected
            utility = log2(
                max(float(on[byte_id]), 1e-300) / max(off_probability, 1e-300)
            )
            event = {
                "utility_bits": utility,
                "permission_score": decision.conservative_value,
                "permission_evidence": float(decision.observations),
                "catastrophic_rate": decision.catastrophic_rate,
                "restoration": restoration,
                "disruption": disruption,
                "overturn": float(overturn),
                "content_match": match,
                "capsule_age": age,
                "recall_age": -1.0 if recall_age is None else recall_age,
                "permitted": float(decision.permitted),
            }
            if evaluating:
                evaluation_events.append(event)
                if decision.permitted:
                    live_probability = float(on[byte_id])
                    memory.reinstate(read)
                else:
                    memory.touch(read)
            else:
                calibration_events.append(event)
                permission.update(key, utility)
                memory.touch(read)

        if evaluating:
            baseline_bits += -log2(max(off_probability, 1e-300))
            live_bits += -log2(max(live_probability, 1e-300))
        learner._observe_probability_only(byte_id, learn=False)
        surprise = -log2(max(off_probability, 1e-300))
        routes = set(learner._active_route_keys())
        union = len(routes | previous_routes)
        turnover = 0.0 if not union else 1.0 - len(routes & previous_routes) / union
        disagreement = learner._temporal_disagreement(tuple(routes))
        boundary_signal = abs(surprise - previous_surprise) + turnover + disagreement
        raw_window.append(byte_id)
        if len(raw_window) > 3:
            del raw_window[0]
        current_query = (
            int.from_bytes(raw_window, "big") if len(raw_window) == 3 else None
        )
        active_items = tuple(
            (
                receptor,
                assembly,
                len(learner.word_forms[assembly])
                / max(float(learner.motif_activity[assembly]), 1e-12) ** 0.5,
            )
            for receptor, assembly in routes
            if 0 <= assembly < len(learner.word_forms)
            and assembly not in learner.temporal_recalled_sources
        )
        memory.observe_boundary(
            boundary_signal,
            query=current_query,
            active_items=active_items,
            basin=basin,
        )
        previous_routes = routes
        previous_surprise = surprise

    utility = np.asarray(
        [event["utility_bits"] for event in evaluation_events], dtype=np.float64
    )
    permitted = [event for event in evaluation_events if event["permitted"] > 0.0]
    permitted_utility = np.asarray(
        [event["utility_bits"] for event in permitted], dtype=np.float64
    )
    heldout_bytes = max(1, len(stream) - calibration_bytes)
    return {
        "task": "phase74_risk_sensitive_restorative_permission",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": len(stream),
        "eval_offset": eval_offset,
        "calibration_bytes": calibration_bytes,
        "feature_context": feature_context,
        "minimum_evidence": minimum_evidence,
        "confidence_width": confidence_width,
        "catastrophic_threshold": catastrophic_threshold,
        "tail_risk_penalty": tail_risk_penalty,
        "maximum_catastrophic_rate": maximum_catastrophic_rate,
        "veto_winner_change": veto_winner_change,
        "recency_bonus": recency_bonus,
        "recency_half_life": recency_half_life,
        "recall_recency_bonus": recall_recency_bonus,
        "recall_recency_half_life": recall_recency_half_life,
        "learned_permission_classes": len(permission.statistics),
        "calibration_proposals": len(calibration_events),
        "heldout_proposals": len(evaluation_events),
        "heldout_proposal_utility_bits": float(utility.sum()),
        "heldout_proposal_positive_fraction": (
            float(np.mean(utility > 0.0)) if len(utility) else 0.0
        ),
        "heldout_permitted_reads": len(permitted),
        "heldout_permitted_utility_bits": float(permitted_utility.sum()),
        "heldout_permitted_positive_fraction": (
            float(np.mean(permitted_utility > 0.0)) if len(permitted_utility) else 0.0
        ),
        "baseline_bpb": baseline_bits / heldout_bytes,
        "risk_permitted_bpb": live_bits / heldout_bytes,
        "bpb_improvement": (baseline_bits - live_bits) / heldout_bytes,
        "permission_calibration": _calibration_bins(
            evaluation_events,
            catastrophic_threshold=catastrophic_threshold,
        ),
        "proposal_summary": _utility_summary(
            evaluation_events, catastrophic_threshold=catastrophic_threshold
        ),
        "permitted_summary": _utility_summary(
            permitted, catastrophic_threshold=catastrophic_threshold
        ),
        "overturn_summaries": {
            str(overturn): _utility_summary(
                [event for event in evaluation_events if event["overturn"] == overturn],
                catastrophic_threshold=catastrophic_threshold,
            )
            for overturn in (0.0, 1.0, 2.0)
        },
        "permitted_overturn_summaries": {
            str(overturn): _utility_summary(
                [event for event in permitted if event["overturn"] == overturn],
                catastrophic_threshold=catastrophic_threshold,
            )
            for overturn in (0.0, 1.0, 2.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--calibration-bytes", type=int)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument(
        "--feature-context", choices=("pooled", "receptor"), default="pooled"
    )
    parser.add_argument("--minimum-evidence", type=int, default=16)
    parser.add_argument("--confidence-width", type=float, default=2.0)
    parser.add_argument("--catastrophic-threshold", type=float, default=0.1)
    parser.add_argument("--tail-risk-penalty", type=float, default=0.5)
    parser.add_argument("--maximum-catastrophic-rate", type=float, default=0.05)
    parser.add_argument("--veto-winner-change", action="store_true")
    parser.add_argument("--recency-bonus", type=float, default=0.0)
    parser.add_argument("--recency-half-life", type=float, default=256.0)
    parser.add_argument("--recall-recency-bonus", type=float, default=0.0)
    parser.add_argument("--recall-recency-half-life", type=float, default=256.0)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        eval_offset=args.eval_offset,
        calibration_bytes=args.calibration_bytes,
        candidate_limit=args.candidate_limit,
        feature_context=args.feature_context,
        minimum_evidence=args.minimum_evidence,
        confidence_width=args.confidence_width,
        catastrophic_threshold=args.catastrophic_threshold,
        tail_risk_penalty=args.tail_risk_penalty,
        maximum_catastrophic_rate=args.maximum_catastrophic_rate,
        veto_winner_change=args.veto_winner_change,
        recency_bonus=args.recency_bonus,
        recency_half_life=args.recency_half_life,
        recall_recency_bonus=args.recall_recency_bonus,
        recall_recency_half_life=args.recall_recency_half_life,
        state_output=args.state_output,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
