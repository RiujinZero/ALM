"""Causal utility gate for a sparse recurrent episodic field."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import deque
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
    CapsuleRead,
)
from manifoldflow.dendritic_basin.recurrent_episodic_field import (
    SparseRecurrentEpisodicField,
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
from manifoldflow.experiments.dendritic_basin_phase74_risk_permission import (
    _bucket,
    _calibration_bins,
    _utility_summary,
)


def _field_permission_key(
    *,
    strength: float,
    coherence: float,
    cue_overlap: float,
    cortical_compatibility: float,
    disruption: float,
    winner_changed: bool,
) -> tuple[int, ...]:
    return (
        _bucket(strength, (0.05, 0.1, 0.15, 0.2, 0.3)),
        _bucket(coherence, (0.5, 0.75, 0.9)),
        _bucket(cue_overlap, (0.1, 0.25, 0.5)),
        _bucket(cortical_compatibility, (0.1, 0.25, 0.5, 0.75)),
        _bucket(disruption, (0.001, 0.01, 0.05, 0.1)),
        int(winner_changed),
    )


def _temporal_cortical_context(
    history: tuple[tuple[tuple[int, int], ...], ...],
    *,
    include_identity: bool = False,
    identity_receptor_limit: int = 8,
) -> tuple[tuple[int, int], ...]:
    """Route activity delivered through distinct decaying temporal receptors."""
    context: list[tuple[int, int]] = []
    for lag, frame in enumerate(reversed(history)):
        for assembly, receptor in frame:
            context.append((assembly, receptor + lag * 65_536))
            if include_identity and (
                receptor == 0 or receptor <= identity_receptor_limit
            ):
                context.append((assembly, 2_000_000))
    return tuple(context)


def _sensory_context(
    query: int | None, *, maximum_bytes: int = 16
) -> tuple[tuple[int, int], ...]:
    """Expose recent raw sensory activity through lag-specific receptors."""
    if query is None:
        return ()
    width = max(1, (query.bit_length() + 7) // 8)
    values = query.to_bytes(width, "big")[-maximum_bytes:]
    return tuple(
        (1_000_000 + value, 1_000_000 + lag)
        for lag, value in enumerate(reversed(values))
    )


def _attractor_summary(probes: list[dict[str, object]]) -> dict[str, object]:
    def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {"count": 0, "mean_target": 0.0, "success_fraction": 0.0}
        return {
            "count": len(rows),
            "mean_target": float(np.mean([row["target"] for row in rows])),
            "success_fraction": float(
                np.mean(
                    [
                        row["target"] >= 0.8 and row["target"] > row["competitor"]
                        for row in rows
                    ]
                )
            ),
        }

    if not probes:
        return {"unique_candidates": 0, "natural": summarize([]), "seed": {}}
    seed_keys = tuple((probes[0]["seed"]).keys())
    return {
        "unique_candidates": len(probes),
        "natural": summarize([probe["natural"] for probe in probes]),
        "seed": {
            key: summarize([probe["seed"][key] for probe in probes])
            for key in seed_keys
        },
        "reconstruction": summarize([probe["reconstruction"] for probe in probes]),
        "persistent": summarize([probe["persistent"] for probe in probes]),
    }


def _return_path_summary(events: list[dict[str, float]]) -> dict[str, float | int]:
    if not events:
        return {
            "reads": 0,
            "mean_items": 0.0,
            "existing_route_fraction": 0.0,
            "predictive_route_fraction": 0.0,
            "nonzero_distribution_fraction": 0.0,
            "total_variation": 0.0,
            "maximum_variation": 0.0,
        }
    total_items = sum(event["return_items"] for event in events)
    return {
        "reads": len(events),
        "mean_items": float(np.mean([event["return_items"] for event in events])),
        "existing_route_fraction": sum(
            event["existing_return_routes"] for event in events
        )
        / max(1.0, total_items),
        "predictive_route_fraction": sum(
            event["predictive_return_routes"] for event in events
        )
        / max(1.0, total_items),
        "nonzero_distribution_fraction": float(
            np.mean([event["disruption"] > 1e-12 for event in events])
        ),
        "total_variation": float(sum(event["disruption"] for event in events)),
        "maximum_variation": float(max(event["disruption"] for event in events)),
    }


def _receptor_route_distribution(
    learner: object,
    items: tuple[tuple[int, int], ...],
    *,
    predictive_only: bool = True,
    item_strengths: tuple[float, ...] = (),
) -> np.ndarray | None:
    """Project recalled assemblies through their learned cortical receptors.

    A field return is ``(assembly, receptor)``.  The receptor-specific route is
    already a learned cortical association from that source assembly to later
    motif assemblies.  Projecting that row through the current raw-byte prefix
    lets recall enter by the same dendritic pathway without pretending it was a
    new sensory observation.
    """
    prefix = bytes(learner.current_word)
    projected = np.zeros(256, dtype=np.float64)
    strengths = item_strengths or tuple(1.0 for _ in items)
    for (assembly, receptor), item_strength in zip(items, strengths, strict=True):
        key = (receptor, assembly)
        row = learner.route_counts.get(key)
        if not row:
            continue
        utility = float(learner.route_utility.get(key, 0.0))
        if predictive_only and utility <= learner.minimum_route_utility_bits:
            continue
        route_total = float(learner.route_totals.get(key, sum(row.values())))
        if route_total <= 0.0:
            continue
        route_weight = (utility if predictive_only else 1.0) * item_strength
        route_weight = max(route_weight, 1e-12)
        for target, observations in row.items():
            if observations <= 0 or not 0 <= target < len(learner.word_forms):
                continue
            form = learner.word_forms[target]
            if prefix:
                if not form.startswith(prefix) or len(form) <= len(prefix):
                    continue
                byte_id = form[len(prefix)]
            else:
                if not form:
                    continue
                byte_id = form[0]
            projected[byte_id] += route_weight * observations / route_total
    total = float(np.sum(projected))
    if total <= 0.0:
        return None
    projected /= total
    return projected


def _return_distribution(
    learner: object,
    completion: object,
    off: np.ndarray,
    *,
    return_path: str,
    maximum_return_weight: float,
) -> tuple[np.ndarray, float]:
    """Open one provenance-preserving return path into the byte cortex."""
    if return_path == "temporal_trace":
        read = CapsuleRead(
            0,
            0,
            completion.items[0][1],
            completion.items,
            completion.strength,
            0,
        )
        recalled = _temporary_distribution(learner, read)
    else:
        recalled = _receptor_route_distribution(
            learner,
            completion.items,
            predictive_only=return_path == "receptor_routes",
            item_strengths=getattr(completion, "item_strengths", ()),
        )
        if recalled is None:
            return off, 0.0
    return_gain = min(
        maximum_return_weight,
        max(0.0, float(completion.commitment)) * maximum_return_weight,
    )
    on = (1.0 - return_gain) * off + return_gain * recalled
    on /= float(np.sum(on))
    return on, return_gain


def run_probe(
    checkpoint: Path,
    eval_path: Path,
    *,
    eval_bytes: int = 100_000,
    eval_offset: int = 0,
    memory_units: int = 4_096,
    assembly_size: int = 8,
    settling_steps: int = 6,
    capture_recency_gain: float = 0.01,
    recall_recency_gain: float = 0.01,
    calibration_bytes: int | None = None,
    minimum_permission_evidence: int = 8,
    learned_risk_permission: bool = False,
    cortical_context_depth: int = 4,
    branch_count: int = 10,
    heterosynaptic_decay: float = 0.30,
    branch_survival_threshold: float = 0.12,
    branch_input_mode: str = "raw",
    background_learning_rate: float = 0.01,
    residual_seed_gain: float = 0.0,
    dendritic_buffer_half_life: float = 0.0,
    return_buffer_half_life: float = 16.0,
    return_buffer_half_life_ms: float = 250.0,
    return_buffer_mode: str = "bytes",
    order_sensitive_binding: bool = False,
    sequence_trace_decay: float = 0.5,
    sequence_learning_rate: float = 1.0,
    sequence_settling_steps: int = 0,
    sequence_compatibility_gain: float = 2.0,
    sequence_stop_compatibility: float = 0.5,
    return_path: str = "temporal_trace",
    maximum_return_weight: float = 1.0,
    attractor_diagnostic: bool = False,
    state_output: Path | None = None,
) -> dict[str, object]:
    if return_path not in {
        "temporal_trace",
        "receptor_routes",
        "all_receptor_routes",
    }:
        raise ValueError("unknown recurrent field return path")
    if not 0.0 <= maximum_return_weight <= 1.0:
        raise ValueError("maximum return weight must be in [0, 1]")
    learner = _read_checkpoint(checkpoint)
    learner.reset_episode()
    learner.binding_index = _NoopBindingIndex()
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    boundary_memory = BoundaryEventCapsuleMemory(permission_context="query_receptor")
    field = SparseRecurrentEpisodicField(
        memory_units=memory_units,
        assembly_size=assembly_size,
        settling_steps=settling_steps,
        capture_recency_gain=capture_recency_gain,
        recall_recency_gain=recall_recency_gain,
        branch_count=branch_count,
        heterosynaptic_decay=heterosynaptic_decay,
        branch_survival_threshold=branch_survival_threshold,
        branch_input_mode=branch_input_mode,
        background_learning_rate=background_learning_rate,
        residual_seed_gain=residual_seed_gain,
        dendritic_buffer_half_life=dendritic_buffer_half_life,
        return_buffer_half_life=return_buffer_half_life,
        return_buffer_half_life_ms=return_buffer_half_life_ms,
        return_buffer_mode=return_buffer_mode,
        order_sensitive_binding=order_sensitive_binding,
        sequence_trace_decay=sequence_trace_decay,
        sequence_learning_rate=sequence_learning_rate,
        sequence_settling_steps=sequence_settling_steps,
        sequence_compatibility_gain=sequence_compatibility_gain,
        sequence_stop_compatibility=sequence_stop_compatibility,
    )
    permission = RiskSensitivePermissionBank(
        minimum_evidence=minimum_permission_evidence
    )
    with eval_path.open("rb") as handle:
        handle.seek(eval_offset)
        stream = np.frombuffer(handle.read(eval_bytes), dtype=np.uint8)
    raw_window = bytearray()
    previous_routes: set[tuple[int, int]] = set()
    route_history: deque[tuple[tuple[int, int], ...]] = deque(
        maxlen=cortical_context_depth
    )
    previous_surprise = 0.0
    baseline_bits = 0.0
    live_bits = 0.0
    calibration_events: list[dict[str, float]] = []
    evaluation_events: list[dict[str, float]] = []
    field_diagnostics: list[dict[str, float]] = []
    attractor_probes: list[dict[str, object]] = []
    calibration_bytes = (
        len(stream) // 2 if calibration_bytes is None else calibration_bytes
    )

    for position, value in enumerate(stream):
        byte_id = int(value)
        evaluating = position >= calibration_bytes
        off = learner.predict_next_distribution()
        off_probability = float(off[byte_id])
        live_probability = off_probability
        organ = getattr(learner, "predictive_goal_organ", None)
        basin = None if organ is None else organ.current_basin
        cortical_context = _temporal_cortical_context(tuple(route_history)) + (
            _sensory_context(int.from_bytes(raw_window, "big") if raw_window else None)
        )
        if attractor_diagnostic:
            probe = field.probe_unique_attractor(
                cortical_context,
                context_id=basin,
            )
            if probe is not None:
                attractor_probes.append(probe)
        compatibility = None
        if field.sequence_settling_steps > 0:
            compatibility = {
                item: _content_match(learner, item) for item in field.output_items
            }
        completion = field.retrieve(
            cortical_context,
            context_id=basin,
            sequence_cue_items=tuple(
                (assembly, receptor)
                for receptor, assembly in learner._active_route_keys()
            ),
            content_gain=compatibility,
        )
        if field.last_diagnostics:
            field_diagnostics.append(dict(field.last_diagnostics))
        if completion is not None and completion.items:
            on, return_gain = _return_distribution(
                learner,
                completion,
                off,
                return_path=return_path,
                maximum_return_weight=maximum_return_weight,
            )
            disruption = 0.5 * float(np.sum(np.abs(on - off)))
            winner_changed = bool(np.argmax(on) != np.argmax(off))
            cortical_compatibility = max(
                (_content_match(learner, item) for item in completion.items),
                default=0.0,
            )
            existing_return_routes = sum(
                (receptor, assembly) in learner.route_counts
                for assembly, receptor in completion.items
            )
            predictive_return_routes = sum(
                (receptor, assembly) in learner.route_counts
                and learner.route_utility[(receptor, assembly)]
                > learner.minimum_route_utility_bits
                for assembly, receptor in completion.items
            )
            key = _field_permission_key(
                strength=completion.strength,
                coherence=completion.coherence,
                cue_overlap=completion.cue_overlap,
                cortical_compatibility=cortical_compatibility,
                disruption=disruption,
                winner_changed=winner_changed,
            )
            decision = permission.decide(key)
            utility = log2(
                max(float(on[byte_id]), 1e-300) / max(off_probability, 1e-300)
            )
            event = {
                "utility_bits": utility,
                "permission_score": decision.conservative_value,
                "field_strength": completion.strength,
                "coherence": completion.coherence,
                "cue_overlap": completion.cue_overlap,
                "commitment": completion.commitment,
                "apical_support": completion.apical_support,
                "committed_assemblies": float(completion.committed_assemblies),
                "cortical_compatibility": cortical_compatibility,
                "return_gain": return_gain,
                "disruption": disruption,
                "winner_changed": float(winner_changed),
                "return_items": float(len(completion.items)),
                "existing_return_routes": float(existing_return_routes),
                "predictive_return_routes": float(predictive_return_routes),
                "permitted": float(
                    decision.permitted if learned_risk_permission else True
                ),
                "position": float(position),
            }
            if evaluating:
                evaluation_events.append(event)
                permitted_now = decision.permitted if learned_risk_permission else True
                if permitted_now:
                    live_probability = float(on[byte_id])
                    field.reinstate(completion)
            else:
                calibration_events.append(event)
                permission.update(key, utility)

        learner._observe_probability_only(byte_id, learn=False)
        if evaluating:
            baseline_bits += -log2(max(off_probability, 1e-300))
            live_bits += -log2(max(live_probability, 1e-300))
        surprise = -log2(max(off_probability, 1e-300))
        routes = set(learner._active_route_keys())
        union = len(routes | previous_routes)
        turnover = 0.0 if not union else 1.0 - len(routes & previous_routes) / union
        disagreement = learner._temporal_disagreement(tuple(routes))
        boundary_signal = abs(surprise - previous_surprise) + turnover + disagreement
        raw_window.append(byte_id)
        if len(raw_window) > 16:
            del raw_window[0]
        query = int.from_bytes(raw_window, "big") if len(raw_window) >= 3 else None
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
        captured = boundary_memory.observe_boundary(
            boundary_signal,
            query=query,
            active_items=active_items,
            basin=basin,
        )
        if captured and boundary_memory.capsules:
            capsule = max(boundary_memory.capsules, key=lambda item: item.last_used)
            event_history: deque[tuple[tuple[int, int], ...]] = deque(
                maxlen=cortical_context_depth
            )
            event_units = None
            sequence_started = False
            for frame_query, address in capsule.frames:
                event_history.append(tuple(address))
                if order_sensitive_binding and not address:
                    continue
                event_units = field.encode(
                    _temporal_cortical_context(tuple(event_history))
                    + _sensory_context(frame_query),
                    (
                        tuple(address)
                        if order_sensitive_binding
                        else capsule.items
                    ),
                    units=event_units,
                    context_id=capsule.basin,
                    reset_sequence=not sequence_started,
                )
                sequence_started = sequence_started or bool(event_units)
        field.advance()
        previous_routes = routes
        route_history.append(
            tuple((assembly, receptor) for receptor, assembly in routes)
        )
        previous_surprise = surprise

    catastrophic_threshold = 0.1
    winner_preserving = [
        event for event in evaluation_events if not event["winner_changed"]
    ]
    winner_changing = [event for event in evaluation_events if event["winner_changed"]]
    permitted = [event for event in evaluation_events if event["permitted"]]
    field_bytes = len(pickle.dumps(field, protocol=pickle.HIGHEST_PROTOCOL))
    heldout_bytes = max(1, len(stream) - calibration_bytes)
    diagnostic_summary = {
        name: {
            "maximum": float(np.max(values)),
            "median": float(np.median(values)),
            "p99": float(np.quantile(values, 0.99)),
        }
        for name in (
            "maximum_commitment",
            "maximum_apical_support",
            "maximum_instantaneous_support",
            "maximum_completion",
            "maximum_drive",
            "cue_pair_conflict",
            "cue_veto",
            "assembly_conflict",
            "eligible_memories",
        )
        if (
            values := np.asarray(
                [row[name] for row in field_diagnostics], dtype=np.float64
            )
        ).size
    }
    if state_output is not None:
        state_output.parent.mkdir(parents=True, exist_ok=True)
        with state_output.open("wb") as handle:
            pickle.dump(
                {
                    "learner": learner,
                    "field": field,
                    "return_path": return_path,
                    "maximum_return_weight": maximum_return_weight,
                    "cortical_context_depth": cortical_context_depth,
                    "order_sensitive_binding": order_sensitive_binding,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return {
        "task": "phase76_sparse_recurrent_episodic_field",
        "checkpoint": str(checkpoint),
        "eval_path": str(eval_path),
        "eval_bytes": len(stream),
        "eval_offset": eval_offset,
        "memory_units": memory_units,
        "assembly_size": assembly_size,
        "settling_steps": settling_steps,
        "capture_recency_gain": capture_recency_gain,
        "recall_recency_gain": recall_recency_gain,
        "calibration_bytes": calibration_bytes,
        "minimum_permission_evidence": minimum_permission_evidence,
        "learned_risk_permission": learned_risk_permission,
        "cortical_context_depth": cortical_context_depth,
        "branch_count": branch_count,
        "heterosynaptic_decay": heterosynaptic_decay,
        "branch_survival_threshold": branch_survival_threshold,
        "branch_input_mode": branch_input_mode,
        "background_learning_rate": background_learning_rate,
        "residual_seed_gain": residual_seed_gain,
        "dendritic_buffer_half_life": dendritic_buffer_half_life,
        "return_buffer_half_life": return_buffer_half_life,
        "return_buffer_half_life_ms": return_buffer_half_life_ms,
        "return_buffer_mode": return_buffer_mode,
        "order_sensitive_binding": order_sensitive_binding,
        "sequence_trace_decay": sequence_trace_decay,
        "sequence_learning_rate": sequence_learning_rate,
        "sequence_settling_steps": sequence_settling_steps,
        "sequence_compatibility_gain": sequence_compatibility_gain,
        "sequence_stop_compatibility": sequence_stop_compatibility,
        "sequence_updates": field.sequence_updates,
        "sequence_returns": field.sequence_returns,
        "sequence_dynamics_returns": field.sequence_dynamics_returns,
        "return_path": return_path,
        "maximum_return_weight": maximum_return_weight,
        "attractor_diagnostic": attractor_diagnostic,
        "attractor_summary": _attractor_summary(attractor_probes),
        "permission_classes": len(permission.statistics),
        "calibration_completions": len(calibration_events),
        "field_encodings": field.encodings,
        "field_live_units": int(np.count_nonzero(field.use_count)),
        "field_maximum_unit_uses": int(np.max(field.use_count)),
        "field_mean_live_unit_uses": float(
            np.mean(field.use_count[field.use_count > 0])
        ),
        "field_serialized_bytes": field_bytes,
        "field_live_branches": sum(
            branch.strength > 0.0
            for branches in field.engram_branches
            for branch in branches
        ),
        "field_branch_depressions": field.branch_depressions,
        "field_branch_deaths": field.branch_deaths,
        "field_branch_reassignments": field.branch_reassignments,
        "field_diagnostics": diagnostic_summary,
        "baseline_bpb": baseline_bits / heldout_bytes,
        "risk_permitted_bpb": live_bits / heldout_bytes,
        "bpb_improvement": (baseline_bits - live_bits) / heldout_bytes,
        "completions": len(evaluation_events),
        "permitted_completions": len(permitted),
        "completion_summary": _utility_summary(
            evaluation_events, catastrophic_threshold=catastrophic_threshold
        ),
        "return_path_summary": _return_path_summary(evaluation_events),
        "permitted_summary": _utility_summary(
            permitted, catastrophic_threshold=catastrophic_threshold
        ),
        "winner_preserving_summary": _utility_summary(
            winner_preserving, catastrophic_threshold=catastrophic_threshold
        ),
        "winner_changing_summary": _utility_summary(
            winner_changing, catastrophic_threshold=catastrophic_threshold
        ),
        "strength_calibration": _calibration_bins(
            evaluation_events, catastrophic_threshold=catastrophic_threshold
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--memory-units", type=int, default=4_096)
    parser.add_argument("--assembly-size", type=int, default=8)
    parser.add_argument("--settling-steps", type=int, default=6)
    parser.add_argument("--capture-recency-gain", type=float, default=0.01)
    parser.add_argument("--recall-recency-gain", type=float, default=0.01)
    parser.add_argument("--calibration-bytes", type=int)
    parser.add_argument("--minimum-permission-evidence", type=int, default=8)
    parser.add_argument("--learned-risk-permission", action="store_true")
    parser.add_argument("--cortical-context-depth", type=int, default=4)
    parser.add_argument("--branch-count", type=int, default=10)
    parser.add_argument("--heterosynaptic-decay", type=float, default=0.30)
    parser.add_argument("--branch-survival-threshold", type=float, default=0.12)
    parser.add_argument(
        "--branch-input-mode",
        choices=("raw", "running_residual", "context_residual"),
        default="raw",
    )
    parser.add_argument("--background-learning-rate", type=float, default=0.01)
    parser.add_argument("--residual-seed-gain", type=float, default=0.0)
    parser.add_argument("--dendritic-buffer-half-life", type=float, default=0.0)
    parser.add_argument("--return-buffer-half-life", type=float, default=16.0)
    parser.add_argument("--return-buffer-half-life-ms", type=float, default=250.0)
    parser.add_argument(
        "--return-buffer-mode",
        choices=("bytes", "milliseconds", "either"),
        default="bytes",
    )
    parser.add_argument("--order-sensitive-binding", action="store_true")
    parser.add_argument("--sequence-trace-decay", type=float, default=0.5)
    parser.add_argument("--sequence-learning-rate", type=float, default=1.0)
    parser.add_argument("--sequence-settling-steps", type=int, default=0)
    parser.add_argument("--sequence-compatibility-gain", type=float, default=2.0)
    parser.add_argument("--sequence-stop-compatibility", type=float, default=0.5)
    parser.add_argument(
        "--return-path",
        choices=("temporal_trace", "receptor_routes", "all_receptor_routes"),
        default="temporal_trace",
    )
    parser.add_argument("--maximum-return-weight", type=float, default=1.0)
    parser.add_argument("--attractor-diagnostic", action="store_true")
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        args.checkpoint,
        args.eval_path,
        eval_bytes=args.eval_bytes,
        eval_offset=args.eval_offset,
        memory_units=args.memory_units,
        assembly_size=args.assembly_size,
        settling_steps=args.settling_steps,
        capture_recency_gain=args.capture_recency_gain,
        recall_recency_gain=args.recall_recency_gain,
        calibration_bytes=args.calibration_bytes,
        minimum_permission_evidence=args.minimum_permission_evidence,
        learned_risk_permission=args.learned_risk_permission,
        cortical_context_depth=args.cortical_context_depth,
        branch_count=args.branch_count,
        heterosynaptic_decay=args.heterosynaptic_decay,
        branch_survival_threshold=args.branch_survival_threshold,
        branch_input_mode=args.branch_input_mode,
        background_learning_rate=args.background_learning_rate,
        residual_seed_gain=args.residual_seed_gain,
        dendritic_buffer_half_life=args.dendritic_buffer_half_life,
        return_buffer_half_life=args.return_buffer_half_life,
        return_buffer_half_life_ms=args.return_buffer_half_life_ms,
        return_buffer_mode=args.return_buffer_mode,
        order_sensitive_binding=args.order_sensitive_binding,
        sequence_trace_decay=args.sequence_trace_decay,
        sequence_learning_rate=args.sequence_learning_rate,
        sequence_settling_steps=args.sequence_settling_steps,
        sequence_compatibility_gain=args.sequence_compatibility_gain,
        sequence_stop_compatibility=args.sequence_stop_compatibility,
        return_path=args.return_path,
        maximum_return_weight=args.maximum_return_weight,
        attractor_diagnostic=args.attractor_diagnostic,
        state_output=args.state_output,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
