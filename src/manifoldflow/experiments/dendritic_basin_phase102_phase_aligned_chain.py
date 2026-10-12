"""Test phase-aligned local sequence completion after one demonstration."""

from __future__ import annotations

import argparse
import json
import pickle
from math import log2
from pathlib import Path
from time import perf_counter

import numpy as np

from manifoldflow.dendritic_basin.reciprocal_relational_coupling import (
    settle_relational_returns,
)
from manifoldflow.dendritic_basin.transient_commitment import (
    TransientRecallCommitment,
)
from manifoldflow.experiments import (
    dendritic_basin_phase85_generation_credit_ablation as phase85,
)
from manifoldflow.experiments.dendritic_basin_phase76_generation import (
    _enable_relational_access,
    _new_arm,
    _observe,
    _predict,
    _predict_encoding_only,
    _sensory_context,
    _temporal_cortical_context,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import CASES
from manifoldflow.experiments.dendritic_basin_phase91_developmental_successor import (
    _clear_fast_state,
    _development_stream,
    _prime,
)


def _matched_phase(form: bytes, history: bytes) -> tuple[int, float] | None:
    """Return the phase after the longest proper prefix present in cortex."""
    maximum = min(len(form) - 1, len(history))
    for length in range(maximum, 0, -1):
        if history.endswith(form[:length]):
            return int(form[length - 1]), length / len(form)
    return None


def _transition_novelty(
    form: bytes,
    history: bytes,
    generated: bytes,
    *,
    matched_bytes: int,
    window_bytes: int,
    order: int = 3,
) -> float:
    """Return the fraction of proposed local transitions not recently expressed.

    This is a short-lived sensory refractory trace, not a semantic repetition
    rule.  A partially matched form may still finish through a new transition
    (``ball`` -> ``all.``), while a synonymous assembly proposing only recently
    traversed transitions receives no current.
    """
    continuation = form[matched_bytes:]
    if not continuation or not generated or window_bytes <= 0:
        return 1.0
    recent = generated[-window_bytes:]
    if len(recent) < order:
        return 1.0
    seen = {
        recent[index : index + order]
        for index in range(len(recent) - order + 1)
    }
    prefix = bytearray(history[-(order - 1) :])
    novel = 0
    tested = 0
    for byte_id in continuation:
        prefix.append(byte_id)
        if len(prefix) >= order:
            transition = bytes(prefix[-order:])
            novel += transition not in seen
            tested += 1
        if len(prefix) > order - 1:
            del prefix[: len(prefix) - (order - 1)]
    return novel / tested if tested else 1.0


def _access_candidates(
    arm: object,
    history: bytes,
    *,
    minimum_bridge_prefix: int,
    context: tuple[tuple[int, int], ...] | None = None,
    allow_onset: bool = False,
    onset_weight: float = 1.0,
    excluded: frozenset[tuple[int, int]] = frozenset(),
    generated: bytes = b"",
    adaptation_window_bytes: int = 0,
) -> list[tuple[float, float, tuple[int, int], int, int | None, float]]:
    code = arm.field.cortical_code(
        arm.pending_access_context if context is None else context
    )
    signal = arm.relational_access.activate(code, len(arm.field.engram_units))
    rows = []
    conjunctive_rows = []
    for engram, content in signal.content_support.items():
        for item, strength in content.items():
            if item in excluded or not 0 <= item[0] < len(arm.learner.word_forms):
                continue
            form = arm.learner.word_forms[item[0]]
            matched = _matched_phase(form, history)
            if matched is None:
                if not allow_onset:
                    continue
                phase = None
                overlap = onset_weight
                matched_bytes = 0
            else:
                phase, overlap = matched
                matched_bytes = round(overlap * len(form))
                if matched_bytes != 1 and matched_bytes < minimum_bridge_prefix:
                    continue
                learned_phase = arm.recalled_expression.phase_after_prefix(
                    item, form[:matched_bytes]
                )
                if learned_phase is None:
                    continue
                phase = learned_phase
            novelty = _transition_novelty(
                form,
                history,
                generated,
                matched_bytes=matched_bytes,
                window_bytes=adaptation_window_bytes,
            )
            if novelty <= 0.0:
                continue
            row = (
                strength * overlap * novelty,
                strength,
                item,
                engram,
                phase,
                overlap,
            )
            rows.append(row)
            if matched is not None and matched_bytes >= minimum_bridge_prefix:
                conjunctive_rows.append(row)
    if conjunctive_rows:
        rows = conjunctive_rows
    return sorted(
        rows,
        key=lambda row: (-row[0], -row[1], row[2], row[3]),
    )


def _access_entry(
    candidates: list[
        tuple[float, float, tuple[int, int], int, int | None, float]
    ],
) -> tuple[int, tuple[int, int], float, int | None, float] | None:
    if not candidates:
        return None
    commitment, strength, item, engram, phase, _ = candidates[0]
    return engram, item, strength, phase, commitment


def _access_signature(arm: object, engram: int, item: tuple[int, int]) -> np.ndarray:
    """Return the strongest learned dendritic relation for the current query."""
    dimensions = arm.relational_access.dimensions
    signature = np.zeros(dimensions, dtype=np.float64)
    branches = arm.relational_access.branches.get((engram, item), ())
    if not branches:
        return signature
    query = arm.relational_access.current_query
    branch = max(
        branches,
        key=lambda candidate: arm.relational_access._branch_response(
            candidate, query
        ),
    )
    for dimension, weight in branch.weights.items():
        signature[dimension] = weight
    return signature


def _reciprocal_access_entry(
    arm: object,
    candidates: list[
        tuple[float, float, tuple[int, int], int, int | None, float]
    ],
    baseline: np.ndarray,
    *,
    selective_competition: bool,
    relation_scramble_seed: int | None,
    settling_steps: int,
    maximum_traces: int = 64,
) -> tuple[
    tuple[int, tuple[int, int], float, int | None, float] | None,
    dict[str, object],
    tuple[tuple[tuple[int, int], float, int], ...],
]:
    """Let access and native cortical returns settle before commitment."""
    if not candidates:
        return None, {"candidate_count": 0, "active_traces": 0}, ()
    rows = candidates[:maximum_traces]
    returns = []
    signatures = []
    retained_rows = []
    for row in rows:
        _, _, item, engram, phase, _ = row
        resolved_phase = 256 if phase is None else phase
        key = item[0], item[1], resolved_phase
        if arm.recalled_expression.evidence.get(key, 0) < (
            arm.recalled_expression.minimum_evidence
        ):
            continue
        returns.append(
            arm.recalled_expression._row_distribution(
                arm.recalled_expression.weights.get(key, {})
            )
        )
        signatures.append(_access_signature(arm, engram, item))
        retained_rows.append(row)
    if not retained_rows:
        fallback = _access_entry(candidates)
        activations = (
            ()
            if fallback is None
            else ((fallback[1], 1.0, 256 if fallback[3] is None else fallback[3]),)
        )
        return fallback, {
            "candidate_count": len(candidates),
            "eligible_returns": 0,
            "active_traces": 1,
            "fallback": "one_way",
        }, activations
    return_rows = retained_rows
    return_matrix = np.asarray(returns)
    if relation_scramble_seed is not None and len(retained_rows) > 1:
        permutation = np.random.default_rng(relation_scramble_seed).permutation(
            len(retained_rows)
        )
        return_matrix = return_matrix[permutation]
        return_rows = [retained_rows[int(index)] for index in permutation]
    result = settle_relational_returns(
        np.asarray([row[0] for row in retained_rows]),
        return_matrix,
        baseline,
        access_signatures=np.asarray(signatures),
        steps=settling_steps,
        selective_competition=selective_competition,
    )
    winner = int(np.argmax(result.activity))
    selected = return_rows[winner]
    commitment, strength, item, _, phase, _ = selected
    # The memory population and receptor-bound return remain distinct: the
    # winning access engram supplies context, while a scrambled control may
    # deliberately place another learned return on that relationship.
    access_engram = retained_rows[winner][3]
    entry = (access_engram, item, strength, phase, commitment)
    population: dict[tuple[tuple[int, int], int], float] = {}
    for index in np.flatnonzero(result.activity > 1e-3):
        row = return_rows[int(index)]
        returned_item = row[2]
        returned_phase = 256 if row[4] is None else row[4]
        key = returned_item, returned_phase
        population[key] = population.get(key, 0.0) + float(
            result.activity[int(index)]
        )
    activations = tuple(
        (returned_item, value, returned_phase)
        for (returned_item, returned_phase), value in sorted(
            population.items(), key=lambda row: (-row[1], row[0])
        )
    )
    return entry, {
        "candidate_count": len(candidates),
        "eligible_returns": len(retained_rows),
        "active_traces": result.active_traces,
        "entropy": result.entropy,
        "winner_activity": float(result.activity[winner]),
        "selective_competition": selective_competition,
        "relation_scrambled": relation_scramble_seed is not None,
    }, activations


def _settled_access_entry(
    arm: object,
    candidates: list[
        tuple[float, float, tuple[int, int], int, int | None, float]
    ],
    baseline: np.ndarray,
    *,
    mode: str,
    selective_competition: bool,
    relation_scramble_seed: int | None,
    settling_steps: int,
) -> tuple[
    tuple[int, tuple[int, int], float, int | None, float] | None,
    dict[str, object],
    tuple[tuple[tuple[int, int], float, int], ...],
]:
    if mode == "one_way":
        entry = _access_entry(candidates)
        activations = (
            ()
            if entry is None
            else ((entry[1], 1.0, 256 if entry[3] is None else entry[3]),)
        )
        return entry, {
            "candidate_count": len(candidates),
            "active_traces": int(bool(candidates)),
            "mode": "one_way",
        }, activations
    if mode != "reciprocal":
        raise ValueError("unknown access settling mode")
    return _reciprocal_access_entry(
        arm,
        candidates,
        baseline,
        selective_competition=selective_competition,
        relation_scramble_seed=relation_scramble_seed,
        settling_steps=settling_steps,
    )


def _successor(
    arm: object,
    engram: int,
    item: tuple[int, int],
    history: bytes,
    *,
    excluded: frozenset[tuple[int, int]] = frozenset(),
    generated: bytes = b"",
    adaptation_window_bytes: int = 0,
) -> tuple[tuple[int, int], int, float] | None:
    transitions = arm.field.engram_sequence_weights[engram].get(item, {})
    candidates = []
    for target, weight in transitions.items():
        if target in excluded:
            continue
        assembly, _ = target
        if not 0 <= assembly < len(arm.learner.word_forms):
            continue
        form = arm.learner.word_forms[assembly]
        matched = _matched_phase(form, history)
        if matched is None:
            continue
        phase, overlap = matched
        matched_bytes = round(overlap * len(form))
        learned_phase = arm.recalled_expression.phase_after_prefix(
            target, form[:matched_bytes]
        )
        if learned_phase is None:
            continue
        phase = learned_phase
        key = assembly, target[1], phase
        if arm.recalled_expression.evidence.get(key, 0) < (
            arm.recalled_expression.minimum_evidence
        ):
            continue
        novelty = _transition_novelty(
            form,
            history,
            generated,
            matched_bytes=matched_bytes,
            window_bytes=adaptation_window_bytes,
        )
        if novelty <= 0.0:
            continue
        candidates.append(
            (weight * overlap * novelty, overlap, weight, target, phase)
        )
    if not candidates:
        return None
    score, overlap, _, target, phase = max(
        candidates,
        key=lambda row: (row[0], row[1], row[2], -row[3][0], -row[3][1]),
    )
    return target, phase, score


def _advance_generated_state(
    arm: object,
    byte_id: int,
    probability: float,
) -> None:
    """Advance transient state without treating generated bytes as evidence.

    Emitted bytes must remain available to the current trajectory and its
    closed-loop query.  They must not create durable capsules, access branches,
    or external confirmation.  This is the generated provenance compartment,
    not a reduced sensory-learning rule.
    """
    if arm.recalled_expression is not None:
        arm.recalled_expression.observe(byte_id, external=False)
    advance_generated = getattr(
        arm.learner, "advance_generated_observation", None
    )
    if advance_generated is None:
        arm.learner._observe_probability_only(byte_id, learn=False)
    else:
        advance_generated(byte_id)
    routes = set(arm.learner._active_route_keys())
    surprise = -float(np.log2(max(probability, 1e-300)))
    union = len(routes | arm.previous_routes)
    turnover = 0.0 if not union else 1.0 - len(routes & arm.previous_routes) / union
    disagreement = arm.learner._temporal_disagreement(tuple(routes))
    arm.last_boundary_signal = (
        abs(surprise - arm.previous_surprise) + turnover + disagreement
    )
    arm.raw_window.append(byte_id)
    if len(arm.raw_window) > 16:
        del arm.raw_window[0]
    arm.route_history.append(
        tuple((assembly, receptor) for receptor, assembly in routes)
    )
    arm.field.advance()
    if arm.relational_access is not None:
        if getattr(arm, "access_learning_enabled", True):
            arm.relational_access.advance()
        else:
            arm.relational_access.clock += 1
    if arm.return_adaptation_half_life > 0.0:
        decay = 2.0 ** (-1.0 / arm.return_adaptation_half_life)
        arm.return_adaptation = {
            item: value * decay
            for item, value in arm.return_adaptation.items()
            if value * decay >= 0.01
        }
    if getattr(arm, "temporal_relational_eligibility", None) is not None:
        arm.temporal_relational_eligibility.advance()
    arm.previous_routes = routes
    arm.previous_surprise = surprise


def _generate(
    prepared: object,
    expected: bytes,
    *,
    recall: bool,
    initial_history_bytes: int,
    minimum_bridge_prefix: int,
    closed_loop_access: bool,
    minimum_access_handoff_interval: int,
    initial_allow_onset: bool,
    onset_weight: float,
    generation_bytes: int | None = None,
    adaptation_mode: str = "none",
    adaptation_window_bytes: int = 64,
    maximum_recall_handoffs: int | None = None,
    commitment_half_life_bytes: float = 0.0,
    commitment_extinction: float = 0.25,
    baseline_learner: object | None = None,
    baseline_prompt: bytes = b"",
    access_settling_mode: str = "one_way",
    selective_access_competition: bool = True,
    relation_scramble_seed: int | None = None,
    access_settling_steps: int = 1,
) -> dict[str, object]:
    setup_started = perf_counter()
    arm = pickle.loads(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
    cortical = (
        arm.learner
        if baseline_learner is None
        else pickle.loads(
            pickle.dumps(baseline_learner, protocol=pickle.HIGHEST_PROTOCOL)
        )
    )
    if baseline_learner is not None:
        cortical.reset_episode()
        cortical.observe_chunk_probabilities(
            np.frombuffer(baseline_prompt, dtype=np.uint8), learn=False
        )
    arm.access_learning_enabled = False
    arm.expression_learning_enabled = False
    if arm.relational_access is not None:
        # Frozen autonomous retrieval cannot use training eligibility. Clear
        # it once instead of metabolizing dead traces for every emitted byte.
        # The biological clock still advances in _advance_generated_state.
        arm.relational_access.reset_episode()
    sensory_seed = bytes(arm.raw_window[-initial_history_bytes:])
    candidates = (
        _access_candidates(
            arm,
            sensory_seed,
            minimum_bridge_prefix=minimum_bridge_prefix,
            allow_onset=initial_allow_onset,
            onset_weight=onset_weight,
        )
        if recall
        else []
    )
    initial_baseline = cortical.predict_next_distribution()
    entry, initial_settling, active_returns = _settled_access_entry(
        arm,
        candidates,
        initial_baseline,
        mode=access_settling_mode,
        selective_competition=selective_access_competition,
        relation_scramble_seed=relation_scramble_seed,
        settling_steps=access_settling_steps,
    )
    arm.recalled_expression.reset_episode()
    current = None if entry is None else entry[1]
    engram = None if entry is None else entry[0]
    initial_phase = None if entry is None else entry[3]
    emitted = bytearray()
    trajectory_history = bytearray(sensory_seed)
    bits = 0.0
    margins = []
    chain = [] if current is None else [arm.learner.word_forms[current[0]]]
    visited_items = set() if current is None else {current}
    handoffs = 0
    access_handoffs = 0
    last_access_handoff = 0
    handoff_trace = []
    step_trace = []
    commitment = (
        TransientRecallCommitment(
            half_life_steps=commitment_half_life_bytes,
            extinction=commitment_extinction,
        )
        if commitment_half_life_bytes > 0.0
        else None
    )
    if commitment is not None and current is not None:
        commitment.recruit(external=True)
    steps = len(expected) if generation_bytes is None else generation_bytes
    setup_elapsed = perf_counter() - setup_started
    started = perf_counter()
    for position in range(steps):
        expected_byte = expected[position] if position < len(expected) else None
        baseline = cortical.predict_next_distribution()
        if current is not None and current in arm.recalled_expression.active:
            phase = arm.recalled_expression.active[current][1]
            key = current[0], current[1], phase
            terminal = arm.recalled_expression.evidence.get(key, 0) < (
                arm.recalled_expression.minimum_evidence
            )
            if terminal:
                handoff_budget_exhausted = (
                    maximum_recall_handoffs is not None
                    and handoffs + access_handoffs >= maximum_recall_handoffs
                )
                commitment_exhausted = (
                    commitment is not None and not commitment.active
                )
                excluded = (
                    frozenset(visited_items)
                    if adaptation_mode == "item"
                    else frozenset()
                )
                generated_context = (
                    bytes(emitted) if adaptation_mode == "transition" else b""
                )
                next_item = (
                    None
                    if handoff_budget_exhausted or commitment_exhausted
                    else _successor(
                        arm,
                        int(engram),
                        current,
                        bytes(trajectory_history),
                        excluded=excluded,
                        generated=generated_context,
                        adaptation_window_bytes=adaptation_window_bytes,
                    )
                )
                access_next = None
                if (
                    not handoff_budget_exhausted
                    and not commitment_exhausted
                    and closed_loop_access
                    and (
                    len(emitted) - last_access_handoff
                    >= minimum_access_handoff_interval
                    )
                ):
                    context = _temporal_cortical_context(
                        tuple(arm.route_history),
                        include_identity=arm.identity_context,
                        identity_receptor_limit=arm.identity_receptor_limit,
                    ) + _sensory_context(
                        int.from_bytes(arm.raw_window, "big")
                        if arm.raw_window
                        else None
                    )
                    (
                        access_next,
                        access_settling,
                        next_active_returns,
                    ) = _settled_access_entry(
                        arm,
                        _access_candidates(
                            arm,
                            bytes(trajectory_history),
                            minimum_bridge_prefix=minimum_bridge_prefix,
                            context=context,
                            allow_onset=True,
                            onset_weight=onset_weight,
                            excluded=excluded,
                            generated=generated_context,
                            adaptation_window_bytes=adaptation_window_bytes,
                        ),
                        baseline,
                        mode=access_settling_mode,
                        selective_competition=selective_access_competition,
                        relation_scramble_seed=relation_scramble_seed,
                        settling_steps=access_settling_steps,
                    )
                else:
                    access_settling = None
                    next_active_returns = ()
                access_score = -1.0 if access_next is None else access_next[4]
                local_score = -1.0 if next_item is None else next_item[2]
                source_form = arm.learner.word_forms[current[0]]
                if access_score > local_score:
                    assert access_next is not None
                    engram, current, _, initial_phase, _ = access_next
                    access_handoffs += 1
                    last_access_handoff = len(emitted)
                    visited_items.add(current)
                    chain.append(arm.learner.word_forms[current[0]])
                    selected_path = "access"
                    active_returns = next_active_returns
                else:
                    if next_item is None:
                        current = None
                        initial_phase = None
                        selected_path = "release"
                        active_returns = ()
                    else:
                        current, initial_phase, _ = next_item
                        visited_items.add(current)
                        chain.append(arm.learner.word_forms[current[0]])
                        handoffs += 1
                        selected_path = "local"
                        active_returns = (
                            (current, 1.0, initial_phase),
                        )
                handoff_trace.append(
                    {
                        "position": len(emitted),
                        "source": source_form.decode(
                            "utf-8", errors="backslashreplace"
                        ),
                        "selected_path": selected_path,
                        "target": (
                            None
                            if current is None
                            else arm.learner.word_forms[current[0]].decode(
                                "utf-8", errors="backslashreplace"
                            )
                        ),
                        "local_score": local_score,
                        "access_score": access_score,
                        "baseline_max": float(np.max(baseline)),
                        "baseline_entropy": float(
                            -np.sum(
                                baseline
                                * np.log2(np.maximum(baseline, 1e-300))
                            )
                        ),
                        "commitment": (
                            None if commitment is None else commitment.level
                        ),
                        "access_settling": access_settling,
                    }
                )
        if current is None:
            arm.recalled_expression.activate(())
        else:
            if not active_returns:
                active_returns = (
                    (current, 1.0, 256 if initial_phase is None else initial_phase),
                )
            arm.recalled_expression.activate(
                tuple(row[0] for row in active_returns),
                tuple(row[1] for row in active_returns),
                tuple(row[2] for row in active_returns),
            )
            initial_phase = None
        expression = arm.recalled_expression.predict(
            baseline, maximum_gain=arm.expression_gain
        )
        distribution = expression.distribution
        byte_id = int(distribution.argmax())
        if len(step_trace) < 32:
            baseline_byte = int(baseline.argmax())
            recalled_byte = (
                None
                if expression.recalled_distribution is None
                else int(expression.recalled_distribution.argmax())
            )
            step_trace.append(
                {
                    "position": position,
                    "active": (
                        None
                        if current is None
                        else arm.learner.word_forms[current[0]].decode(
                            "utf-8", errors="backslashreplace"
                        )
                    ),
                    "baseline_byte": chr(baseline_byte),
                    "recalled_byte": (
                        None if recalled_byte is None else chr(recalled_byte)
                    ),
                    "emitted_byte": chr(byte_id),
                    "agreement": recalled_byte == baseline_byte,
                    "baseline_selected_probability": float(baseline[byte_id]),
                    "gain": expression.gain,
                }
            )
        emitted.append(byte_id)
        trajectory_history.append(byte_id)
        if expected_byte is not None:
            bits -= log2(max(float(distribution[expected_byte]), 1e-300))
            other = max(
                float(probability)
                for index, probability in enumerate(distribution)
                if index != expected_byte
            )
            margins.append(
                log2(max(float(distribution[expected_byte]), 1e-300))
                - log2(max(other, 1e-300))
            )
        if baseline_learner is None:
            _observe(arm, byte_id, float(distribution[byte_id]), external=False)
        else:
            _advance_generated_state(arm, byte_id, float(distribution[byte_id]))
        if baseline_learner is not None:
            cortical._observe_probability_only(byte_id, learn=False)
        if commitment is not None:
            commitment.advance()
    elapsed = perf_counter() - started
    return {
        "access_entry": (
            None
            if entry is None
            else {
                "form": prepared.learner.word_forms[entry[1][0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": entry[1][1],
                "engram": entry[0],
                "support": entry[2],
                "phase": entry[3],
                "coincidence": entry[4],
            }
        ),
        "compatible_access": [
            {
                "form": prepared.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                ),
                "receptor": item[1],
                "engram": engram,
                "support": support,
                "phase": phase,
                "prefix_overlap": overlap,
                "coincidence": commitment,
                "branches": len(
                    getattr(arm.relational_access, "branches", {}).get(
                        (engram, item), ()
                    )
                ),
            }
            for commitment, support, item, engram, phase, overlap in candidates[:24]
        ],
        "access_settling": initial_settling,
        "chain": [
            form.decode("utf-8", errors="backslashreplace") for form in chain
        ],
        "emitted": emitted.decode("utf-8", errors="backslashreplace"),
        "exact": bytes(emitted) == expected,
        "expected_prefix_exact": bytes(emitted[: len(expected)]) == expected,
        "bits": bits,
        "bpb": bits / len(expected),
        "positive_margins": sum(margin > 0.0 for margin in margins),
        "margins": margins,
        "handoffs": handoffs,
        "access_handoffs": access_handoffs,
        "handoff_trace": handoff_trace,
        "step_trace": step_trace,
        "setup_elapsed_seconds": setup_elapsed,
        "elapsed_seconds": elapsed,
        "bytes_per_second": steps / max(elapsed, 1e-12),
    }


def _access_anatomy(arm: object) -> dict[str, int]:
    branches = getattr(arm.relational_access, "branches", {})
    return {
        "rows": len(branches),
        "branches": sum(len(row) for row in branches.values()),
        "branch_births": int(
            getattr(arm.relational_access, "branch_births", 0)
        ),
        "branch_deaths": int(
            getattr(arm.relational_access, "branch_deaths", 0)
        ),
        "embodied_bytes": int(arm.relational_access.embodied_bytes),
    }


def _temporal_eligibility_statistics(arm: object) -> dict[str, object]:
    tissue = getattr(arm, "temporal_relational_eligibility", None)
    if tissue is None:
        return {"enabled": False}
    ages = tuple(int(age) for age in tissue.confirmation_ages)
    strengths = (
        np.stack(tuple(tissue.strengths.values()))
        if tissue.strengths
        else np.zeros((0, len(tissue.half_lives)), dtype=np.float64)
    )
    return {
        "enabled": True,
        "participations": int(tissue.participations),
        "external_updates": int(tissue.external_updates),
        "positive_updates": int(tissue.positive_updates),
        "negative_updates": int(tissue.negative_updates),
        "positive_utility": float(tissue.positive_utility),
        "negative_utility": float(tissue.negative_utility),
        "learned_relations": len(tissue.strengths),
        "active_traces": len(tissue.traces),
        "embodied_bytes": int(tissue.embodied_bytes),
        "positive_strength": float(np.sum(np.maximum(0.0, strengths))),
        "negative_strength": float(np.sum(np.maximum(0.0, -strengths))),
        "timescale_strength": (
            np.sum(strengths, axis=0).tolist() if len(strengths) else []
        ),
        "confirmation_age_mean": float(np.mean(ages)) if ages else 0.0,
        "confirmation_age_maximum": max(ages, default=0),
        "confirmation_age_histogram": {
            "0_2": sum(age <= 2 for age in ages),
            "3_7": sum(3 <= age <= 7 for age in ages),
            "8_31": sum(8 <= age <= 31 for age in ages),
            "32_127": sum(32 <= age <= 127 for age in ages),
            "128_511": sum(128 <= age <= 511 for age in ages),
            "512_plus": sum(age >= 512 for age in ages),
        },
    }


def _branch_probe(arm: object, expected: bytes) -> dict[str, object]:
    """Expose whether expected content is absent, weak, or below threshold."""
    access = arm.relational_access
    code = arm.field.cortical_code(arm.pending_access_context)
    maximum = float(np.max(code, initial=0.0))
    normalized = code / maximum if maximum > 0.0 else code
    query = normalized - access.background_activity
    expected_words = tuple(expected.strip().lower().split())
    rows = []
    for (engram, item), branches in access.branches.items():
        if not branches or not 0 <= item[0] < len(arm.learner.word_forms):
            continue
        form = arm.learner.word_forms[item[0]]
        winner = max(
            branches,
            key=lambda branch: access._branch_response(branch, query),
        )
        response = access._branch_response(winner, query)
        rows.append(
            {
                "form": form.decode("utf-8", errors="backslashreplace"),
                "receptor": item[1],
                "engram": engram,
                "response": response,
                "strength": winner.strength,
                "false_responses": winner.false_responses,
                "capture_salience": winner.capture_salience,
                "branches": len(branches),
                "expected_related": any(
                    word in form.lower() for word in expected_words
                ),
            }
        )
    rows.sort(key=lambda row: (-float(row["response"]), str(row["form"])))
    return {
        "activation_threshold": access.activation_threshold,
        "top": rows[:32],
        "expected_related": [row for row in rows if row["expected_related"]],
    }


def _prepare_development(
    payload: dict[str, object],
    stream: bytes,
    *,
    mode: str,
    access_learning_mode: str,
    frames_per_capsule: int,
    access_proposal_threshold: float,
    temporal_credit_mode: str,
    relational_route_codevelopment: bool = False,
    relational_residual_codevelopment: bool = False,
) -> object:
    """Develop staged or with live memory/cortex interaction.

    The coupled arm still receives teaching only from the subsequent external
    byte. Recall can participate in the presynaptic state but cannot serve as
    its own observation.
    """
    if mode == "staged":
        arm = phase85._prepare(
            payload,
            stream,
            access_learning_mode=access_learning_mode,
            frames_per_capsule=frames_per_capsule,
            access_proposal_threshold=access_proposal_threshold,
        )
        arm.temporal_credit_mode = temporal_credit_mode
        arm.relational_route_codevelopment = relational_route_codevelopment
        arm.relational_residual_codevelopment = relational_residual_codevelopment
        return arm
    if mode != "coupled":
        raise ValueError("unknown development mode")
    arm = _new_arm(payload)
    _enable_relational_access(
        arm,
        expression_gain=0.5,
        frames_per_capsule=frames_per_capsule,
        access_proposal_threshold=access_proposal_threshold,
    )
    arm.relational_access.learning_mode = access_learning_mode
    arm.temporal_credit_mode = temporal_credit_mode
    arm.relational_route_codevelopment = relational_route_codevelopment
    arm.relational_residual_codevelopment = relational_residual_codevelopment
    arm.lazy_expression_enabled = False
    for byte_id in stream:
        distribution = _predict(arm, live=True)
        _observe(
            arm,
            byte_id,
            float(distribution[byte_id]),
            external=True,
        )
    phase85._calibrate_expression(arm)
    return arm


def run(
    state: Path,
    data_path: Path,
    *,
    cortical_state: Path | None = None,
    secondary_data_path: Path | None,
    interleave_chunk_bytes: int,
    development_bytes: int,
    frames_per_capsule: int,
    access_proposal_threshold: float,
    eligibility_mode: str,
    access_learning_mode: str,
    consume_on_confirmation: bool,
    minimum_confirmation_age: int,
    case_index: int,
    initial_history_bytes: int,
    minimum_bridge_prefix: int,
    closed_loop_access: bool,
    minimum_access_handoff_interval: int,
    initial_allow_onset: bool,
    branch_false_response_grace: int,
    branch_survival_mode: str,
    onset_weight: float,
    generation_bytes: int | None,
    order_sensitive_expression: bool,
    adaptation_mode: str,
    adaptation_window_bytes: int,
    maximum_recall_handoffs: int | None,
    commitment_half_life_bytes: float,
    commitment_extinction: float,
    access_settling_mode: str = "one_way",
    selective_access_competition: bool = True,
    relation_scramble_seed: int | None = None,
    development_mode: str = "staged",
    access_settling_steps: int = 1,
    temporal_credit_mode: str = "relation_local",
) -> dict[str, object]:
    if maximum_recall_handoffs is not None and maximum_recall_handoffs < 0:
        raise ValueError("maximum recall handoffs must be non-negative")
    if frames_per_capsule < 2:
        raise ValueError("frames per capsule must be at least two")
    if access_proposal_threshold < 0.0:
        raise ValueError("access proposal threshold must be non-negative")
    if commitment_half_life_bytes < 0.0:
        raise ValueError("commitment half-life must be non-negative")
    if not 0.0 < commitment_extinction < 1.0:
        raise ValueError("commitment extinction must be in (0, 1)")
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    cortical_learner = None
    if cortical_state is not None:
        with cortical_state.open("rb") as handle:
            cortical_payload = pickle.load(handle)
        cortical_learner = cortical_payload["learner"]
        if not hasattr(cortical_learner, "max_sequence_nursery_paths"):
            cortical_learner.max_sequence_nursery_paths = 50_000
        if not hasattr(cortical_learner, "sequence_nursery_evictions"):
            cortical_learner.sequence_nursery_evictions = 0
    primary_data = data_path.read_bytes()
    development_stream = (
        primary_data[:development_bytes]
        if secondary_data_path is None
        else _development_stream(
            primary_data,
            secondary_data_path.read_bytes(),
            mode="mixed",
            byte_count=development_bytes,
            chunk_bytes=interleave_chunk_bytes,
        )
    )
    development_started = perf_counter()
    developed = _prepare_development(
        payload,
        development_stream,
        mode=development_mode,
        access_learning_mode=access_learning_mode,
        frames_per_capsule=frames_per_capsule,
        access_proposal_threshold=access_proposal_threshold,
        temporal_credit_mode=temporal_credit_mode,
    )
    development_elapsed = perf_counter() - development_started
    developed.relational_access.eligibility_mode = eligibility_mode
    developed.relational_access.consume_on_confirmation = consume_on_confirmation
    developed.relational_access.minimum_confirmation_age = minimum_confirmation_age
    developed.relational_access.branch_false_response_grace = (
        branch_false_response_grace
    )
    developed.relational_access.branch_survival_mode = branch_survival_mode
    if order_sensitive_expression:
        developed.recalled_expression.order_sensitive_phases = True
        phase85._calibrate_expression(developed)
    case = CASES[case_index]
    demonstration_started = perf_counter()
    demonstration = _prime(
        developed, case.prompt.encode() + b" ", return_mode="direct"
    )
    demonstration_trace = []
    for byte_id in case.expected.encode().lstrip():
        distribution = _predict(demonstration, live=False)
        births_before = int(demonstration.relational_access.branch_births)
        _observe(
            demonstration,
            byte_id,
            float(distribution[byte_id]),
            external=True,
        )
        created_at = demonstration.relational_access.clock - 1
        created = sorted(
            {
                demonstration.learner.word_forms[item[0]].decode(
                    "utf-8", errors="backslashreplace"
                )
                for (_, item), row in demonstration.relational_access.branches.items()
                if 0 <= item[0] < len(demonstration.learner.word_forms)
                if any(branch.created_at == created_at for branch in row)
            }
        )
        demonstration_trace.append(
            {
                "byte": bytes((byte_id,)).decode(
                    "utf-8", errors="backslashreplace"
                ),
                "surprise_bits": -log2(
                    max(float(distribution[byte_id]), 1e-300)
                ),
                "boundary_signal": float(demonstration.last_boundary_signal),
                "branch_births": int(
                    demonstration.relational_access.branch_births - births_before
                ),
                "created_forms": created,
            }
        )
    demonstration_elapsed = perf_counter() - demonstration_started
    probe_started = perf_counter()
    probe = pickle.loads(
        pickle.dumps(demonstration, protocol=pickle.HIGHEST_PROTOCOL)
    )
    _clear_fast_state(probe)
    probe.recall_enabled = True
    probe.access_learning_enabled = False
    probe.expression_learning_enabled = False
    probe.access_return_mode = "direct"
    probe.access_recruitment_mode = "none"
    for byte_id in case.prompt.encode() + b" ":
        distribution = _predict_encoding_only(probe)
        _observe(probe, byte_id, float(distribution[byte_id]), external=True)
    _predict(probe, live=False)
    probe_elapsed = perf_counter() - probe_started
    expected = case.expected.encode().lstrip()
    return {
        "task": "phase102_phase_aligned_local_chain",
        "state": str(state),
        "cortical_state": None if cortical_state is None else str(cortical_state),
        "data_path": str(data_path),
        "secondary_data_path": (
            None if secondary_data_path is None else str(secondary_data_path)
        ),
        "interleave_chunk_bytes": interleave_chunk_bytes,
        "development_bytes": development_bytes,
        "development_mode": development_mode,
        "temporal_credit_mode": temporal_credit_mode,
        "frames_per_capsule": frames_per_capsule,
        "access_proposal_threshold": access_proposal_threshold,
        "timing": {
            "development_seconds": development_elapsed,
            "development_bytes_per_second": development_bytes
            / max(development_elapsed, 1e-12),
            "demonstration_seconds": demonstration_elapsed,
            "probe_seconds": probe_elapsed,
        },
        "development_activity": {
            "field_completions": int(developed.completions),
            "effective_cortical_returns": int(developed.effective_returns),
            "fresh_effective_returns": int(developed.fresh_effective_returns),
            "buffered_effective_returns": int(developed.buffered_effective_returns),
            "access_confirmations": int(developed.access_confirmations),
            "access_completions": int(developed.access_completions),
            "temporal_relational_eligibility": (
                _temporal_eligibility_statistics(developed)
            ),
        },
        "eligibility_mode": eligibility_mode,
        "access_learning_mode": access_learning_mode,
        "consume_on_confirmation": consume_on_confirmation,
        "minimum_confirmation_age": minimum_confirmation_age,
        "demonstrations": 1,
        "case_index": case_index,
        "case": case.name,
        "initial_history_bytes": initial_history_bytes,
        "minimum_bridge_prefix": minimum_bridge_prefix,
        "closed_loop_access": closed_loop_access,
        "minimum_access_handoff_interval": minimum_access_handoff_interval,
        "initial_allow_onset": initial_allow_onset,
        "branch_false_response_grace": branch_false_response_grace,
        "branch_survival_mode": branch_survival_mode,
        "onset_weight": onset_weight,
        "generation_bytes": generation_bytes,
        "order_sensitive_expression": order_sensitive_expression,
        "adaptation_mode": adaptation_mode,
        "adaptation_window_bytes": adaptation_window_bytes,
        "maximum_recall_handoffs": maximum_recall_handoffs,
        "commitment_half_life_bytes": commitment_half_life_bytes,
        "commitment_extinction": commitment_extinction,
        "access_settling_mode": access_settling_mode,
        "selective_access_competition": selective_access_competition,
        "relation_scramble_seed": relation_scramble_seed,
        "access_settling_steps": access_settling_steps,
        "access_anatomy": _access_anatomy(demonstration),
        "demonstration_trace": demonstration_trace,
        "branch_probe": _branch_probe(probe, expected),
        "recall_off": _generate(
            probe,
            expected,
            recall=False,
            initial_history_bytes=initial_history_bytes,
            minimum_bridge_prefix=minimum_bridge_prefix,
            closed_loop_access=closed_loop_access,
            minimum_access_handoff_interval=minimum_access_handoff_interval,
            initial_allow_onset=initial_allow_onset,
            onset_weight=onset_weight,
            generation_bytes=generation_bytes,
            adaptation_mode=adaptation_mode,
            adaptation_window_bytes=adaptation_window_bytes,
            maximum_recall_handoffs=maximum_recall_handoffs,
            commitment_half_life_bytes=commitment_half_life_bytes,
            commitment_extinction=commitment_extinction,
            baseline_learner=cortical_learner,
            baseline_prompt=case.prompt.encode() + b" ",
            access_settling_mode=access_settling_mode,
            selective_access_competition=selective_access_competition,
            relation_scramble_seed=relation_scramble_seed,
            access_settling_steps=access_settling_steps,
        ),
        "phase_aligned_recall": _generate(
            probe,
            expected,
            recall=True,
            initial_history_bytes=initial_history_bytes,
            minimum_bridge_prefix=minimum_bridge_prefix,
            closed_loop_access=closed_loop_access,
            minimum_access_handoff_interval=minimum_access_handoff_interval,
            initial_allow_onset=initial_allow_onset,
            onset_weight=onset_weight,
            generation_bytes=generation_bytes,
            adaptation_mode=adaptation_mode,
            adaptation_window_bytes=adaptation_window_bytes,
            maximum_recall_handoffs=maximum_recall_handoffs,
            commitment_half_life_bytes=commitment_half_life_bytes,
            commitment_extinction=commitment_extinction,
            baseline_learner=cortical_learner,
            baseline_prompt=case.prompt.encode() + b" ",
            access_settling_mode=access_settling_mode,
            selective_access_competition=selective_access_competition,
            relation_scramble_seed=relation_scramble_seed,
            access_settling_steps=access_settling_steps,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--cortical-state", type=Path)
    parser.add_argument("--secondary-data", type=Path)
    parser.add_argument("--interleave-chunk-bytes", type=int, default=256)
    parser.add_argument("--development-bytes", type=int, default=64)
    parser.add_argument(
        "--development-mode",
        choices=("staged", "coupled"),
        default="staged",
    )
    parser.add_argument(
        "--temporal-credit-mode",
        choices=(
            "relation_local",
            "contextual",
            "broadcast",
            "receptor_relation",
            "receptor_relation_scrambled",
            "receptor_relation_identity_scrambled",
            "receptor_relation_residual_utility",
        ),
        default="relation_local",
    )
    parser.add_argument("--frames-per-capsule", type=int, default=24)
    parser.add_argument("--access-proposal-threshold", type=float, default=0.0)
    parser.add_argument(
        "--eligibility-mode",
        choices=("additive", "parallel"),
        default="additive",
    )
    parser.add_argument(
        "--access-learning-mode",
        choices=("prediction_error", "branches"),
        default="prediction_error",
    )
    parser.add_argument(
        "--eligibility-consumption",
        choices=("first", "decay"),
        default="first",
        help="Consume a query at its first confirmation or let it decay naturally.",
    )
    parser.add_argument("--minimum-confirmation-age", type=int, default=0)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)), default=0)
    parser.add_argument("--initial-history-bytes", type=int, default=1)
    parser.add_argument("--minimum-bridge-prefix", type=int, default=3)
    parser.add_argument("--closed-loop-access", action="store_true")
    parser.add_argument("--minimum-access-handoff-interval", type=int, default=0)
    parser.add_argument("--initial-allow-onset", action="store_true")
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--generation-bytes", type=int)
    parser.add_argument("--order-sensitive-expression", action="store_true")
    parser.add_argument(
        "--adaptation-mode", choices=("none", "item", "transition"), default="none"
    )
    parser.add_argument("--adaptation-window-bytes", type=int, default=64)
    parser.add_argument("--maximum-recall-handoffs", type=int)
    parser.add_argument("--commitment-half-life-bytes", type=float, default=0.0)
    parser.add_argument("--commitment-extinction", type=float, default=0.25)
    parser.add_argument(
        "--access-settling-mode",
        choices=("one_way", "reciprocal"),
        default="one_way",
    )
    parser.add_argument("--disable-selective-access-competition", action="store_true")
    parser.add_argument("--relation-scramble-seed", type=int)
    parser.add_argument("--access-settling-steps", type=int, default=1)
    parser.add_argument("--branch-false-response-grace", type=int, default=0)
    parser.add_argument(
        "--branch-survival-mode",
        choices=("uniform", "surprise"),
        default="uniform",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.state,
        args.data,
        cortical_state=args.cortical_state,
        secondary_data_path=args.secondary_data,
        interleave_chunk_bytes=args.interleave_chunk_bytes,
        development_bytes=args.development_bytes,
        frames_per_capsule=args.frames_per_capsule,
        access_proposal_threshold=args.access_proposal_threshold,
        eligibility_mode=args.eligibility_mode,
        access_learning_mode=args.access_learning_mode,
        consume_on_confirmation=args.eligibility_consumption == "first",
        minimum_confirmation_age=args.minimum_confirmation_age,
        case_index=args.case_index,
        initial_history_bytes=args.initial_history_bytes,
        minimum_bridge_prefix=args.minimum_bridge_prefix,
        closed_loop_access=args.closed_loop_access,
        minimum_access_handoff_interval=args.minimum_access_handoff_interval,
        initial_allow_onset=args.initial_allow_onset,
        branch_false_response_grace=args.branch_false_response_grace,
        branch_survival_mode=args.branch_survival_mode,
        onset_weight=args.onset_weight,
        generation_bytes=args.generation_bytes,
        order_sensitive_expression=args.order_sensitive_expression,
        adaptation_mode=args.adaptation_mode,
        adaptation_window_bytes=args.adaptation_window_bytes,
        maximum_recall_handoffs=args.maximum_recall_handoffs,
        commitment_half_life_bytes=args.commitment_half_life_bytes,
        commitment_extinction=args.commitment_extinction,
        access_settling_mode=args.access_settling_mode,
        selective_access_competition=(
            not args.disable_selective_access_competition
        ),
        relation_scramble_seed=args.relation_scramble_seed,
        development_mode=args.development_mode,
        access_settling_steps=args.access_settling_steps,
        temporal_credit_mode=args.temporal_credit_mode,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
