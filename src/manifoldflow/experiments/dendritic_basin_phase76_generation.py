"""Paired autonomous generation through the Phase 76 recurrent return path."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import BoundaryEventCapsuleMemory
from manifoldflow.dendritic_basin.multiscale_assembly import MultiscaleAssemblyField
from manifoldflow.dendritic_basin.recalled_expression import (
    RecalledCorticalExpression,
)
from manifoldflow.dendritic_basin.recurrent_episodic_field import FieldReturnSignal
from manifoldflow.dendritic_basin.relational_access import (
    RelationalAccessField,
    RelationalAccessSignal,
)
from manifoldflow.dendritic_basin.temporal_relational_eligibility import (
    TemporalRelationalEligibility,
)
from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    DEFAULT_PROMPTS,
    _first_divergence,
)
from manifoldflow.experiments.dendritic_basin_phase73_shadow_capsules import (
    _content_match,
    _NoopBindingIndex,
)
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _return_distribution,
    _sensory_context,
    _temporal_cortical_context,
)


@dataclass
class _Arm:
    learner: object
    field: object
    return_path: str
    maximum_return_weight: float
    event_memory: BoundaryEventCapsuleMemory
    route_history: deque[tuple[tuple[int, int], ...]]
    raw_window: bytearray
    previous_routes: set[tuple[int, int]]
    output_mode: str
    output_delay_bytes: int
    identity_context: bool = False
    identity_receptor_limit: int = 8
    previous_surprise: float = 0.0
    transient_captures: int = 0
    completions: int = 0
    sequence_completions: int = 0
    effective_returns: int = 0
    fresh_effective_returns: int = 0
    buffered_effective_returns: int = 0
    total_variation: float = 0.0
    maximum_variation: float = 0.0
    completed_forms: set[bytes] | None = None
    effective_return_forms: set[bytes] | None = None
    delayed_frames: deque[tuple[tuple[int, int], ...]] | None = None
    fifo_replay: deque[tuple[tuple[int, int], ...]] | None = None
    engram_frames: dict[
        frozenset[int], tuple[tuple[tuple[int, int], ...], ...]
    ] | None = None
    multiscale: MultiscaleAssemblyField | None = None
    relational_access: RelationalAccessField | None = None
    recalled_expression: RecalledCorticalExpression | None = None
    expression_gain: float = 0.5
    expression_repetitions: int = 24
    pending_access_context: tuple[tuple[int, int], ...] = ()
    pending_access_sequence_cue: tuple[tuple[int, int], ...] = ()
    expression_calibrated_items: set[tuple[int, int]] | None = None
    access_confirmations: int = 0
    access_completions: int = 0
    recall_enabled: bool = True
    access_learning_enabled: bool = True
    expression_learning_enabled: bool = True
    lazy_expression_enabled: bool = True
    return_adaptation_half_life: float = 0.0
    return_adaptation: dict[tuple[int, int], float] | None = None
    access_recruitment_mode: str = "none"
    recruitment_events: int = 0
    recruitment_suppressed: int = 0
    recruitment_support_values: list[float] | None = None
    access_return_mode: str = "direct"
    access_sequence_attempts: int = 0
    access_sequence_source_hits: int = 0
    expression_cancellation_mode: str = "none"
    expression_current_limit: float = 0.0
    access_proposal_threshold: float = 0.0
    last_boundary_signal: float = 0.0
    temporal_relational_eligibility: TemporalRelationalEligibility | None = None
    pending_temporal_population: dict[tuple[int, int, int], float] | None = None
    pending_temporal_predictions: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray]
    ] | None = None
    pending_temporal_global_prediction: tuple[np.ndarray, np.ndarray] | None = None
    pending_relational_receptors: dict[tuple[int, int], float] | None = None
    pending_receptor_relational_population: dict[
        tuple[int, int, int], float
    ] | None = None
    temporal_credit_mode: str = "relation_local"
    relational_route_codevelopment: bool = False
    relational_route_learning_events: int = 0
    relational_residual_codevelopment: bool = False
    relational_residual_routes: dict[
        tuple[int, int], dict[int, float]
    ] | None = None
    relational_residual_learning_events: int = 0

    def __post_init__(self) -> None:
        self.completed_forms = set()
        self.effective_return_forms = set()
        self.delayed_frames = deque(maxlen=self.output_delay_bytes + 1)
        self.fifo_replay = deque()
        self.engram_frames = {}
        self.expression_calibrated_items = set()
        self.return_adaptation = {}
        self.recruitment_support_values = []
        self.pending_temporal_population = {}
        self.pending_temporal_predictions = {}
        self.pending_temporal_global_prediction = None
        self.pending_relational_receptors = {}
        self.pending_receptor_relational_population = {}
        self.relational_residual_routes = {}


def _new_arm(payload: dict[str, object]) -> _Arm:
    clone = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    learner = clone["learner"]
    field = clone["field"]
    if not hasattr(field, "return_buffer_time"):
        field.return_buffer_time = {}
    if not hasattr(field, "return_buffer_mode"):
        field.return_buffer_mode = "bytes"
    if not hasattr(field, "return_buffer_half_life_ms"):
        field.return_buffer_half_life_ms = 250.0
    learner.reset_episode()
    learner.binding_index = _NoopBindingIndex()
    if not hasattr(learner, "temporal_recalled_sources"):
        learner.temporal_recalled_sources = set()
    field.reset_episode()
    depth = int(clone["cortical_context_depth"])
    output_mode = str(clone.get("output_mode", "unordered"))
    if output_mode not in {"unordered", "fifo", "delayed_latest", "recurrent_fifo"}:
        raise ValueError("unknown temporal output mode")
    output_delay_bytes = int(clone.get("output_delay_bytes", 16))
    if output_delay_bytes < 1:
        raise ValueError("output delay must be positive")
    return _Arm(
        learner,
        field,
        str(clone["return_path"]),
        float(clone["maximum_return_weight"]),
        BoundaryEventCapsuleMemory(permission_context="query_receptor"),
        deque(maxlen=depth),
        bytearray(),
        set(),
        output_mode,
        output_delay_bytes,
        bool(clone.get("identity_context", False)),
        int(clone.get("identity_receptor_limit", 8)),
    )


def _enable_relational_access(
    arm: _Arm,
    *,
    expression_gain: float = 0.5,
    expression_repetitions: int = 24,
    frames_per_capsule: int = 24,
    access_proposal_threshold: float = 0.0,
) -> None:
    """Install Phase 83 tissue for online prompt learning and free recall."""
    arm.identity_context = True
    arm.identity_receptor_limit = -1
    arm.event_memory.frame_compaction = "diverse"
    if frames_per_capsule < 2:
        raise ValueError("frames per capsule must be at least two")
    if access_proposal_threshold < 0.0:
        raise ValueError("access proposal threshold must be non-negative")
    arm.event_memory.frames_per_capsule = frames_per_capsule
    arm.access_proposal_threshold = float(access_proposal_threshold)
    arm.event_memory.items_per_frame = 16
    arm.multiscale = MultiscaleAssemblyField(
        activation_threshold=1.0,
        maximum_active=64,
        trace_half_life_bytes=32.0,
    )
    for assembly, form in enumerate(arm.learner.word_forms):
        arm.multiscale.bind(assembly, form)
    arm.relational_access = RelationalAccessField(
        arm.field.cortical_dimensions,
        trace_decay=0.95,
        activation_threshold=0.3,
    )
    arm.recalled_expression = RecalledCorticalExpression(minimum_evidence=1)
    arm.temporal_relational_eligibility = TemporalRelationalEligibility()
    arm.expression_gain = expression_gain
    # Kept on the arm so lazy calibration has no hidden global state.
    arm.expression_repetitions = expression_repetitions


def _predict(arm: _Arm, *, live: bool) -> np.ndarray:
    arm.pending_temporal_population = {}
    arm.pending_temporal_predictions = {}
    arm.pending_temporal_global_prediction = None
    arm.pending_relational_receptors = {}
    arm.pending_receptor_relational_population = {}
    off = arm.learner.predict_next_distribution()
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    context = _temporal_cortical_context(
        tuple(arm.route_history),
        include_identity=arm.identity_context,
        identity_receptor_limit=arm.identity_receptor_limit,
    ) + _sensory_context(
        int.from_bytes(arm.raw_window, "big") if arm.raw_window else None
    )
    sequence_cue_items = (
        arm.route_history[-1]
        if arm.identity_context and arm.route_history
        else tuple(
            (assembly, receptor)
            for receptor, assembly in arm.learner._active_route_keys()
        )
    )
    if not arm.recall_enabled:
        if arm.recalled_expression is not None:
            arm.recalled_expression.activate(())
        arm.pending_access_context = ()
        return off
    access_signal = None
    if arm.relational_access is not None:
        access_code = arm.field.cortical_code(context)
        if arm.last_boundary_signal >= arm.access_proposal_threshold:
            arm.relational_access.propose(
                access_code, salience=arm.last_boundary_signal
            )
        access_signal = arm.relational_access.activate(
            access_code, len(arm.field.engram_units)
        )
        if arm.access_recruitment_mode != "none":
            basal, apical, branch_peak = arm.field.contextual_recruitment(
                context, context_id=basin
            )
            if arm.access_recruitment_mode == "basal":
                # Existing x->h current is bounded locally rather than ranked
                # globally.  A strong access proposal still cannot manufacture
                # contextual evidence that is absent from the present cortex.
                recruitment = basal / (basal + 0.10)
            elif arm.access_recruitment_mode == "apical":
                recruitment = np.clip(apical, 0.0, 1.0)
            elif arm.access_recruitment_mode == "branch_peak":
                recruitment = np.clip(branch_peak, 0.0, 1.0)
            elif arm.access_recruitment_mode == "combined":
                basal_recruitment = basal / (basal + 0.10)
                recruitment = np.sqrt(
                    basal_recruitment * np.clip(apical, 0.0, 1.0)
                )
            else:
                raise ValueError("unknown access recruitment mode")
            proposed = int(np.count_nonzero(access_signal.engram_support > 0.0))
            arm.recruitment_support_values.extend(
                float(recruitment[index])
                for index in np.flatnonzero(access_signal.engram_support > 0.0)
            )
            conditioned = access_signal.engram_support * recruitment
            retained = int(np.count_nonzero(conditioned > 0.15))
            arm.recruitment_events += proposed
            arm.recruitment_suppressed += max(0, proposed - retained)
            content = {
                engram: {
                    item: strength * float(recruitment[engram])
                    for item, strength in row.items()
                    if strength * float(recruitment[engram]) > 0.15
                }
                for engram, row in access_signal.content_support.items()
                if engram < len(conditioned) and conditioned[engram] > 0.15
            }
            source_content = {
                engram: {
                    item: strength * float(recruitment[engram])
                    for item, strength in row.items()
                }
                for engram, row in access_signal.source_support.items()
                if engram < len(conditioned) and conditioned[engram] > 0.15
            }
            access_signal = RelationalAccessSignal(
                conditioned, content, source_content
            )
        arm.pending_access_context = context
        arm.pending_access_sequence_cue = sequence_cue_items
        if arm.access_return_mode in {
            "successor",
            "aligned_successor",
            "content_successor",
            "contextual_successor",
        }:
            active_access = tuple(
                int(index)
                for index in np.flatnonzero(access_signal.engram_support > 0.15)
            )
            arm.access_sequence_attempts += len(active_access)
            if arm.access_return_mode == "successor":
                arm.access_sequence_source_hits += sum(
                    any(
                        arm.field.engram_sequence_weights[engram].get(item)
                        for item in sequence_cue_items
                    )
                    for engram in active_access
                )
            elif arm.access_return_mode == "aligned_successor":
                arm.access_sequence_source_hits += sum(
                    bool(access_signal.source_support.get(engram))
                    for engram in active_access
                )
            else:
                arm.access_sequence_source_hits += sum(
                    (
                        any(
                            arm.field.engram_sequence_weights[engram].get(item)
                            for item in access_signal.content_support.get(engram, {})
                        )
                        if arm.access_return_mode == "content_successor"
                        else any(
                            branch.source
                            in access_signal.content_support.get(engram, {})
                            for branch in arm.field.engram_contextual_successors[engram]
                        )
                    )
                    for engram in active_access
                )
    compatibility = None
    if getattr(arm.field, "sequence_settling_steps", 0) > 0:
        compatibility = {
            item: _content_match(arm.learner, item)
            for item in arm.field.output_items
        }
    temporal_population: dict[tuple[int, int, int], float] = {}
    completion = arm.field.retrieve(
        context,
        context_id=basin,
        sequence_cue_items=sequence_cue_items,
        content_gain=compatibility,
        external_apical_support=(
            None if access_signal is None else access_signal.engram_support
        ),
        access_content=(
            None if access_signal is None else access_signal.content_support
        ),
        access_return_mode=arm.access_return_mode,
        access_sequence_seed=(
            None if access_signal is None else access_signal.source_support
        ),
    )
    if completion is not None and completion.items:
        arm.completions += 1
        arm.sequence_completions += int(completion.sequence_used)
        arm.completed_forms.update(
            arm.learner.word_forms[assembly]
            for assembly, _ in completion.items
            if 0 <= assembly < len(arm.learner.word_forms)
        )
        if live:
            arm.field.reinstate(completion)
        if access_signal is not None:
            completed = set(completion.items)
            temporal_population = {
                (engram, item[0], item[1]): float(access_strength)
                * float(completion.commitment)
                for engram, content in access_signal.content_support.items()
                for item, access_strength in content.items()
                if item in completed and access_strength > 0.0
            }
            relational_receptors: dict[tuple[int, int], float] = {}
            receptor_population: dict[tuple[int, int, int], float] = {}
            for engram, content in access_signal.content_support.items():
                sources = access_signal.source_support.get(engram, {})
                for item, content_strength in content.items():
                    if item not in completed or content_strength <= 0.0:
                        continue
                    for source, source_strength in sources.items():
                        if source_strength <= 0.0:
                            continue
                        returned = item[0], source[1]
                        strength = (
                            float(content_strength)
                            * float(source_strength)
                            * float(completion.commitment)
                        )
                        relational_receptors[returned] = max(
                            relational_receptors.get(returned, 0.0), strength
                        )
                        # Preserve the completed identity while retaining the
                        # predecessor pathway that requested it.  This may be a
                        # silent cross-organ synapse until cortex co-develops a
                        # corresponding target-on-receptor route.
                        relation = engram, item[0], source[1]
                        receptor_population[relation] = max(
                            receptor_population.get(relation, 0.0), strength
                        )
            arm.pending_relational_receptors = relational_receptors
            arm.pending_receptor_relational_population = receptor_population
    signal = None
    if arm.output_mode == "unordered":
        signal = arm.field.return_signal(completion)
    elif arm.output_mode == "delayed_latest":
        if len(arm.delayed_frames) >= arm.output_delay_bytes:
            items = arm.delayed_frames[-arm.output_delay_bytes]
            if items:
                signal = FieldReturnSignal(items, 1.0)
    else:
        if arm.output_mode == "recurrent_fifo" and completion is not None:
            frames = arm.engram_frames.get(frozenset(completion.units))
            if frames and not arm.fifo_replay:
                arm.fifo_replay.extend(frames)
        if arm.fifo_replay:
            items = arm.fifo_replay.popleft()
            if items:
                signal = FieldReturnSignal(items, 1.0)
    if signal is None or not signal.items:
        if arm.recalled_expression is not None:
            arm.recalled_expression.activate(())
        return off
    if arm.recalled_expression is not None and arm.return_adaptation_half_life > 0.0:
        original_strengths = signal.item_strengths or tuple(
            1.0 for _ in signal.items
        )
        adapted = tuple(
            strength * (1.0 - arm.return_adaptation.get(item, 0.0))
            for item, strength in zip(
                signal.items, original_strengths, strict=True
            )
        )
        retained = tuple(
            (item, strength)
            for item, strength in zip(signal.items, adapted, strict=True)
            if strength >= 0.01
        )
        if not retained:
            arm.recalled_expression.activate(())
            return off
        signal = FieldReturnSignal(
            tuple(item for item, _ in retained),
            max(strength for _, strength in retained),
            tuple(strength for _, strength in retained),
        )
    if arm.recalled_expression is None:
        on, gain = _return_distribution(
            arm.learner,
            signal,
            off,
            return_path=arm.return_path,
            maximum_return_weight=arm.maximum_return_weight,
        )
    else:
        for item in signal.items:
            assembly, _ = item
            if (
                arm.lazy_expression_enabled
                and item not in arm.expression_calibrated_items
                and 0 <= assembly < len(arm.learner.word_forms)
            ):
                arm.recalled_expression.learn_sequence(
                    item,
                    arm.learner.word_forms[assembly],
                    repetitions=arm.expression_repetitions,
                )
                arm.expression_calibrated_items.add(item)
        arm.recalled_expression.activate(signal.items, signal.item_strengths)
        expression = arm.recalled_expression.predict(
            off,
            maximum_gain=arm.expression_gain,
            shared_cancellation=arm.expression_cancellation_mode,
            maximum_current_norm=arm.expression_current_limit,
        )
        on, gain = expression.distribution, expression.gain
        arm.access_completions += int(gain > 0.0)
        if (
            live
            and gain > 0.0
            and temporal_population
            and arm.temporal_relational_eligibility is not None
        ):
            population = {
                relation: strength * gain
                for relation, strength in temporal_population.items()
            }
            arm.pending_temporal_population = population
            predictions = {}
            for relation, strength in population.items():
                item = relation[1], relation[2]
                active = arm.recalled_expression.active.get(item)
                if active is None:
                    continue
                phase = active[1]
                key = item[0], item[1], phase
                if arm.recalled_expression.evidence.get(key, 0) < (
                    arm.recalled_expression.minimum_evidence
                ):
                    continue
                recalled = arm.recalled_expression._row_distribution(
                    arm.recalled_expression.weights.get(key, {})
                )
                local_gain = min(
                    arm.expression_gain,
                    arm.expression_gain * max(0.0, float(strength)),
                )
                candidate = (1.0 - local_gain) * off + local_gain * recalled
                candidate /= float(np.sum(candidate))
                predictions[relation] = (off, candidate)
            arm.pending_temporal_predictions = predictions
            arm.pending_temporal_global_prediction = (off, on)
        if live and gain > 0.0 and arm.return_adaptation_half_life > 0.0:
            for item in signal.items:
                arm.return_adaptation[item] = 1.0
    variation = 0.5 * float(np.sum(np.abs(on - off)))
    if variation > 1e-12:
        arm.effective_returns += 1
        if completion is None:
            arm.buffered_effective_returns += 1
        else:
            arm.fresh_effective_returns += 1
        arm.effective_return_forms.update(
            arm.learner.word_forms[assembly]
            for assembly, _ in signal.items
            if 0 <= assembly < len(arm.learner.word_forms)
        )
        arm.total_variation += variation
        arm.maximum_variation = max(arm.maximum_variation, variation)
        if live and gain > 0.0:
            return on
    return off


def _predict_encoding_only(arm: _Arm) -> np.ndarray:
    """Predict and expose an external write trace without attempting recall.

    Development needs the cortex's prediction plus a locally available access
    trace.  Running attractor completion before every externally supplied byte
    cannot affect that prediction when ``live=False`` and is metabolically
    wasteful.  Keep proposal/confirmation anatomy identical while leaving the
    episodic field quiescent until it is actually queried.
    """
    off = arm.learner.predict_next_distribution()
    context = _temporal_cortical_context(
        tuple(arm.route_history),
        include_identity=arm.identity_context,
        identity_receptor_limit=arm.identity_receptor_limit,
    ) + _sensory_context(
        int.from_bytes(arm.raw_window, "big") if arm.raw_window else None
    )
    sequence_cue_items = (
        arm.route_history[-1]
        if arm.identity_context and arm.route_history
        else tuple(
            (assembly, receptor)
            for receptor, assembly in arm.learner._active_route_keys()
        )
    )
    if arm.relational_access is not None:
        access_code = arm.field.cortical_code(context)
        if arm.last_boundary_signal >= arm.access_proposal_threshold:
            arm.relational_access.propose(
                access_code, salience=arm.last_boundary_signal
            )
        arm.pending_access_context = context
        arm.pending_access_sequence_cue = sequence_cue_items
    return off


def _external_access_observations(
    arm: _Arm, activations: tuple[object, ...]
) -> tuple[tuple[tuple[int, int], tuple[int, ...], tuple[tuple[int, int], ...]], ...]:
    """Bind externally observed content to its learned predecessor wires."""
    if (
        arm.relational_access is None
        or not arm.access_learning_enabled
        or not arm.pending_access_context
    ):
        return ()
    observations = []
    for activation in activations:
        if activation.matched_synapses <= 0:
            continue
        item = (activation.assembly, 0)
        engrams = arm.field.engrams_containing(item)
        if not engrams:
            continue
        compatibility = arm.field.engram_cortical_support(
            arm.pending_access_context, engrams
        )
        winner = max(
            engrams,
            key=lambda engram: (compatibility[engram], -engram),
        )
        transitions = arm.field.engram_sequence_weights[winner]
        predecessor = tuple(
            source
            for source in arm.pending_access_sequence_cue
            if item in transitions.get(source, {})
        )
        if not predecessor:
            predecessor = tuple(
                source for source, targets in transitions.items() if item in targets
            )
        observations.append((item, (winner,), predecessor))
    return tuple(observations)


def _observe(
    arm: _Arm,
    byte_id: int,
    probability: float,
    *,
    extra_active_items: tuple[tuple[int, int, float], ...] = (),
    external: bool = True,
) -> None:
    if arm.recalled_expression is not None:
        arm.recalled_expression.observe(
            byte_id,
            external=external and arm.expression_learning_enabled,
        )
    codevelopment_keys: tuple[tuple[int, int], ...] = ()
    if (
        external
        and (
            arm.relational_route_codevelopment
            or arm.relational_residual_codevelopment
        )
        and arm.temporal_relational_eligibility is not None
    ):
        # Recalled identity is only a presynaptic cause.  It can nominate an
        # identity-on-receptor route, but the subsequent external word remains
        # the sole target allowed to grow that cortical route.
        routed = {}
        for relation, activity in (
            arm.pending_receptor_relational_population or {}
        ).items():
            _engram, assembly, receptor = relation
            permission = arm.temporal_relational_eligibility.permission(relation)
            support = float(activity) * (
                1.0 if arm.relational_residual_codevelopment else permission
            )
            key = receptor, assembly
            routed[key] = max(routed.get(key, 0.0), support)
        codevelopment_keys = tuple(
            key for key, support in routed.items() if support > 0.0
        )
    baseline_event_row = (
        dict(arm.learner._active_event_row() or {})
        if codevelopment_keys and arm.relational_residual_codevelopment
        else {}
    )
    temporal_clock_before = getattr(arm.learner, "temporal_clock", None)
    arm.learner._observe_probability_only(byte_id, learn=False)
    if codevelopment_keys and temporal_clock_before is not None:
        temporal_clock = arm.learner.temporal_clock
        if temporal_clock != temporal_clock_before:
            targets = tuple(
                assembly
                for assembly, last_seen in arm.learner.temporal_last_seen.items()
                if last_seen == temporal_clock
                and assembly not in arm.learner.temporal_recalled_sources
            )
            for target in targets:
                if arm.relational_route_codevelopment:
                    arm.learner._update_routes(codevelopment_keys, target)
                    arm.relational_route_learning_events += len(codevelopment_keys)
                if arm.relational_residual_codevelopment and baseline_event_row:
                    total = float(sum(baseline_event_row.values()))
                    if total > 0.0:
                        for key in codevelopment_keys:
                            row = arm.relational_residual_routes.setdefault(key, {})
                            row[target] = row.get(target, 0.0) + 1.0
                            for candidate, observations in baseline_event_row.items():
                                row[candidate] = row.get(candidate, 0.0) - (
                                    float(observations) / total
                                )
                            arm.relational_residual_learning_events += 1
    routes = set(arm.learner._active_route_keys())
    surprise = -float(np.log2(max(probability, 1e-300)))
    union = len(routes | arm.previous_routes)
    turnover = 0.0 if not union else 1.0 - len(routes & arm.previous_routes) / union
    disagreement = arm.learner._temporal_disagreement(tuple(routes))
    boundary_signal = (
        abs(surprise - arm.previous_surprise) + turnover + disagreement
    )
    arm.last_boundary_signal = boundary_signal
    arm.raw_window.append(byte_id)
    if len(arm.raw_window) > 16:
        del arm.raw_window[0]
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    query = int.from_bytes(arm.raw_window, "big") if arm.raw_window else None
    multiscale_items: tuple[tuple[int, int, float], ...] = ()
    activations = ()
    if arm.multiscale is not None:
        activations = arm.multiscale.observe(byte_id)
        multiscale_items = tuple(
            (
                0,
                item.assembly,
                len(arm.learner.word_forms[item.assembly]) * item.support,
            )
            for item in activations
        )
    sensory_items = tuple(
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
    active_items = sensory_items + extra_active_items + multiscale_items
    frame_items = tuple(
        (assembly, receptor)
        for receptor, assembly, _ in sorted(
            active_items, key=lambda item: (-item[2], item[0], item[1])
        )[: arm.event_memory.items_per_frame]
    )
    access_observations = (
        _external_access_observations(arm, activations) if external else ()
    )
    if arm.temporal_relational_eligibility is not None:
        population = (
            arm.pending_receptor_relational_population
            if arm.temporal_credit_mode.startswith("receptor_relation")
            else arm.pending_temporal_population
        )
        preparticipated = False
        if (
            arm.temporal_credit_mode == "receptor_relation_residual_utility"
            and population
        ):
            known = tuple(arm.temporal_relational_eligibility.traces)
            incompatible = {
                relation: tuple(
                    other
                    for other in known
                    if other[2] == relation[2] and other != relation
                )
                for relation in population
            }
            arm.temporal_relational_eligibility.participate_population(
                population, incompatible=incompatible
            )
            preparticipated = True
        if external:
            if arm.temporal_credit_mode == "broadcast":
                if arm.pending_temporal_global_prediction is None:
                    utility = {}
                else:
                    baseline, candidate = arm.pending_temporal_global_prediction
                    global_utility = float(
                        np.clip(
                            np.log2(max(candidate[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300)),
                            -8.0,
                            8.0,
                        )
                    )
                    utility = {
                        relation: global_utility
                        for relation in arm.temporal_relational_eligibility.traces
                    }
            elif arm.temporal_credit_mode == "contextual":
                if (
                    arm.pending_temporal_global_prediction is None
                    or arm.relational_access is None
                ):
                    utility = {}
                else:
                    baseline, candidate = arm.pending_temporal_global_prediction
                    global_utility = float(
                        np.clip(
                            np.log2(max(candidate[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300)),
                            -8.0,
                            8.0,
                        )
                    )
                    utility = {}
                    for relation in arm.temporal_relational_eligibility.traces:
                        engram, assembly, receptor = relation
                        branches = arm.relational_access.branches.get(
                            (engram, (assembly, receptor)), ()
                        )
                        response = max(
                            (
                                arm.relational_access._branch_response(
                                    branch,
                                    arm.relational_access.current_query,
                                )
                                for branch in branches
                            ),
                            default=0.0,
                        )
                        if response >= arm.relational_access.activation_threshold:
                            utility[relation] = global_utility * response
            elif arm.temporal_credit_mode == "relation_local":
                utility = {
                    relation: float(
                        np.clip(
                            np.log2(max(candidate[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300)),
                            -8.0,
                            8.0,
                        )
                    )
                    for relation, (
                        baseline,
                        candidate,
                    ) in (arm.pending_temporal_predictions or {}).items()
                }
            elif arm.temporal_credit_mode in {
                "receptor_relation",
                "receptor_relation_scrambled",
                "receptor_relation_identity_scrambled",
            }:
                observed_relations = {
                    (engram, item[0], source[1])
                    for item, engrams, predecessor in access_observations
                    for engram in engrams
                    for source in predecessor
                }
                if arm.temporal_credit_mode == "receptor_relation_scrambled":
                    bins = tuple(arm.learner.temporal_bins)
                    observed_relations = {
                        (
                            engram,
                            assembly,
                            bins[(bins.index(receptor) + 1) % len(bins)],
                        )
                        for engram, assembly, receptor in observed_relations
                        if receptor in bins
                    }
                elif arm.temporal_credit_mode == (
                    "receptor_relation_identity_scrambled"
                ):
                    vocabulary = max(1, len(arm.learner.word_forms))
                    observed_relations = {
                        (engram, (assembly + 7919) % vocabulary, receptor)
                        for engram, assembly, receptor in observed_relations
                    }
                utility = {
                    relation: 1.0
                    for relation in observed_relations
                    if relation in arm.temporal_relational_eligibility.traces
                }
            elif arm.temporal_credit_mode == "receptor_relation_residual_utility":
                utility = {
                    relation: float(
                        np.clip(
                            np.log2(max(candidate[byte_id], 1e-300))
                            - np.log2(max(baseline[byte_id], 1e-300)),
                            -8.0,
                            8.0,
                        )
                    )
                    for relation, (
                        baseline,
                        candidate,
                    ) in (arm.pending_temporal_predictions or {}).items()
                }
            else:
                raise ValueError("unknown temporal credit mode")
            # Credit the old trace before this occurrence refreshes it. Thus
            # the measured age is the delay between useful recruitments, not
            # an artifact of the current byte resetting its own clock.
            arm.temporal_relational_eligibility.confirm(utility, external=True)
        if population and not preparticipated:
            known = tuple(arm.temporal_relational_eligibility.traces)
            incompatible = {
                relation: tuple(
                    other
                    for other in known
                    if other[2] == relation[2] and other != relation
                )
                for relation in population
            }
            arm.temporal_relational_eligibility.participate_population(
                population, incompatible=incompatible
            )
        arm.temporal_relational_eligibility.advance()
        arm.pending_temporal_population = {}
        arm.pending_temporal_predictions = {}
        arm.pending_temporal_global_prediction = None
        arm.pending_relational_receptors = {}
        arm.pending_receptor_relational_population = {}
    arm.delayed_frames.append(frame_items)
    captured = arm.event_memory.observe_boundary(
        boundary_signal,
        query=query,
        active_items=active_items,
        basin=basin,
    )
    if captured and arm.event_memory.capsules:
        capsule = max(arm.event_memory.capsules, key=lambda item: item.last_used)
        event_history: deque[tuple[tuple[int, int], ...]] = deque(
            maxlen=arm.route_history.maxlen
        )
        event_units = None
        order_sensitive = bool(
            getattr(arm.field, "order_sensitive_binding", False)
        )
        sequence_started = False
        frames_to_encode = tuple(capsule.frames)
        for frame_index, (frame_query, address) in enumerate(frames_to_encode):
            event_history.append(tuple(address))
            if order_sensitive and not address:
                continue
            event_units = arm.field.encode(
                _temporal_cortical_context(
                    tuple(event_history),
                    include_identity=arm.identity_context,
                    identity_receptor_limit=arm.identity_receptor_limit,
                )
                + _sensory_context(frame_query),
                tuple(address) if order_sensitive else capsule.items,
                units=event_units,
                context_id=capsule.basin,
                reset_sequence=not sequence_started,
                update_access_branches=frame_index == len(frames_to_encode) - 1,
            )
            sequence_started = sequence_started or bool(event_units)
        frames = tuple(address for _, address in capsule.frames if address)
        if event_units and frames:
            arm.engram_frames[frozenset(event_units)] = frames
            if arm.output_mode == "fifo":
                arm.fifo_replay.extend(frames)
        arm.transient_captures += 1
    if external and arm.relational_access is not None and access_observations:
        arm.access_confirmations += arm.relational_access.confirm_many(
            access_observations,
            external=True,
        )
    history_items = (
        tuple((assembly, receptor) for receptor, assembly, _ in active_items)
        if extra_active_items
        else tuple((assembly, receptor) for receptor, assembly in routes)
    )
    arm.route_history.append(history_items)
    arm.field.advance()
    if arm.relational_access is not None:
        arm.relational_access.advance()
    if arm.return_adaptation_half_life > 0.0:
        decay = 2.0 ** (-1.0 / arm.return_adaptation_half_life)
        arm.return_adaptation = {
            item: value * decay
            for item, value in arm.return_adaptation.items()
            if value * decay >= 0.01
        }
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
    relational_access: bool = False,
    expression_gain: float = 0.5,
) -> tuple[bytes, _Arm]:
    arm = _new_arm(payload)
    if relational_access:
        _enable_relational_access(arm, expression_gain=expression_gain)
    for byte_id in prompt:
        distribution = _predict(arm, live=live)
        _observe(arm, byte_id, float(distribution[byte_id]), external=True)
    rng = np.random.default_rng(seed)
    generated = bytearray()
    for _ in range(max_new_bytes):
        distribution = _predict(arm, live=live)
        byte_id = _sample_byte(
            distribution,
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        _observe(arm, byte_id, float(distribution[byte_id]), external=False)
    return bytes(generated), arm


def run_generation(
    state: Path,
    *,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    samples_per_prompt: int = 3,
    max_new_bytes: int = 600,
    temperature: float = 0.8,
    top_k: int = 32,
    top_p: float = 0.9,
    seed: int = 7600,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    boundary = getattr(payload["learner"], "story_boundary", b"<|endoftext|>\n")
    pairs = []
    for prompt_index, prompt_text in enumerate(prompts):
        prompt = prompt_text.encode()
        for sample_index in range(samples_per_prompt):
            sample_seed = seed + prompt_index * samples_per_prompt + sample_index
            live, live_arm = _generate(
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
                    "byte_identical": live == shadow,
                    "first_divergence_byte": _first_divergence(live, shadow),
                    "live_completions": live_arm.completions,
                    "live_effective_returns": live_arm.effective_returns,
                    "live_fresh_effective_returns": (
                        live_arm.fresh_effective_returns
                    ),
                    "live_buffered_effective_returns": (
                        live_arm.buffered_effective_returns
                    ),
                    "live_completed_forms": sorted(
                        form.decode("utf-8", errors="backslashreplace")
                        for form in live_arm.completed_forms
                    ),
                    "live_effective_return_forms": sorted(
                        form.decode("utf-8", errors="backslashreplace")
                        for form in live_arm.effective_return_forms
                    ),
                    "live_transient_captures": live_arm.transient_captures,
                    "live_total_variation": live_arm.total_variation,
                    "live_maximum_variation": live_arm.maximum_variation,
                    "shadow_completions": shadow_arm.completions,
                    "live_text": (prompt + live).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "shadow_text": (prompt + shadow).decode(
                        "utf-8", errors="backslashreplace"
                    ),
                    "live_diagnostics": _generation_diagnostics(live, boundary),
                    "shadow_diagnostics": _generation_diagnostics(shadow, boundary),
                }
            )
    return {
        "task": "phase76_paired_recurrent_field_generation",
        "state": str(state),
        "return_path": payload["return_path"],
        "maximum_return_weight": payload["maximum_return_weight"],
        "persistent_learning_during_generation": False,
        "transient_episode_learning_during_generation": True,
        "pairs": pairs,
        "summary": {
            "pairs": len(pairs),
            "changed_pairs": sum(not pair["byte_identical"] for pair in pairs),
            "live_completions": sum(pair["live_completions"] for pair in pairs),
            "live_effective_returns": sum(
                pair["live_effective_returns"] for pair in pairs
            ),
            "live_transient_captures": sum(
                pair["live_transient_captures"] for pair in pairs
            ),
            "live_total_variation": sum(
                pair["live_total_variation"] for pair in pairs
            ),
            "maximum_variation": max(
                (pair["live_maximum_variation"] for pair in pairs), default=0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-index", type=int, choices=range(len(DEFAULT_PROMPTS)))
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-new-bytes", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prompts = (
        (DEFAULT_PROMPTS[args.prompt_index],)
        if args.prompt_index is not None
        else tuple(args.prompts)
    )
    report = run_generation(
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
