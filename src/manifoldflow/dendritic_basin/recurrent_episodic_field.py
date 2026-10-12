"""Sparse recurrent episodic field with local Hebbian plasticity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import monotonic_ns

import numpy as np

from manifoldflow.dendritic_basin.resonant_capsule import (
    SparseResonantCapsuleIndex,
)


@dataclass(frozen=True)
class FieldCompletion:
    units: tuple[int, ...]
    items: tuple[tuple[int, int], ...]
    strength: float
    coherence: float
    cue_overlap: float
    commitment: float
    apical_support: float
    committed_assemblies: int
    iterations: int
    sequence_used: bool = False
    item_strengths: tuple[float, ...] = ()


@dataclass(frozen=True)
class FieldReturnSignal:
    items: tuple[tuple[int, int], ...]
    commitment: float
    item_strengths: tuple[float, ...] = ()


@dataclass
class DendriticBranch:
    synapses: frozenset[int]
    strength: float = 1.0
    false_rate: float = 0.0


@dataclass
class ContextualSuccessorBranch:
    """An occurrence-sensitive local transition dendrite."""

    source: tuple[int, int]
    targets: tuple[tuple[int, int], ...]
    synapses: frozenset[int]
    context: frozenset[int]
    target_strengths: tuple[float, ...] = ()
    strength: float = 1.0
    false_rate: float = 0.0


class SparseRecurrentEpisodicField:
    """Bind cortical states to self-completing sparse memory assemblies."""

    def __getstate__(self) -> dict[str, object]:
        """Exclude rebuildable metabolic execution mirrors from body bytes."""
        state = self.__dict__.copy()
        for name in (
            "_input_matrix",
            "_code_matrix",
            "_code_owners",
            "_code_count",
        ):
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Migrate pre-sequence checkpoints without changing their behavior."""
        self.__dict__.update(state)
        self.order_sensitive_binding = getattr(
            self, "order_sensitive_binding", False
        )
        self.sequence_trace_decay = getattr(self, "sequence_trace_decay", 0.5)
        self.sequence_learning_rate = getattr(self, "sequence_learning_rate", 1.0)
        self.sequence_settling_steps = getattr(self, "sequence_settling_steps", 0)
        self.sequence_compatibility_gain = getattr(
            self, "sequence_compatibility_gain", 2.0
        )
        self.sequence_stop_compatibility = getattr(
            self, "sequence_stop_compatibility", 0.5
        )
        self.engram_sequence_traces = getattr(
            self,
            "engram_sequence_traces",
            [{} for _ in self.engram_units],
        )
        self.engram_sequence_weights = getattr(
            self,
            "engram_sequence_weights",
            [{} for _ in self.engram_units],
        )
        self.sequence_updates = getattr(self, "sequence_updates", 0)
        self.sequence_returns = getattr(self, "sequence_returns", 0)
        self.sequence_dynamics_returns = getattr(
            self, "sequence_dynamics_returns", 0
        )
        self.contextual_successor_binding = getattr(
            self, "contextual_successor_binding", False
        )
        self.contextual_successor_branches_per_source = getattr(
            self, "contextual_successor_branches_per_source", 4
        )
        self.contextual_successor_capacity = getattr(
            self, "contextual_successor_capacity", 256
        )
        self.contextual_successor_competition_gain = getattr(
            self, "contextual_successor_competition_gain", 0.0
        )
        self.engram_sequence_context_traces = getattr(
            self,
            "engram_sequence_context_traces",
            [{} for _ in self.engram_units],
        )
        self.engram_contextual_successors = getattr(
            self,
            "engram_contextual_successors",
            [[] for _ in self.engram_units],
        )
        self.contextual_successor_births = getattr(
            self, "contextual_successor_births", 0
        )
        self.contextual_successor_deaths = getattr(
            self, "contextual_successor_deaths", 0
        )
        self._rebuild_execution_mirrors()

    def __init__(
        self,
        *,
        cortical_dimensions: int = 2_048,
        memory_units: int = 512,
        assembly_size: int = 8,
        settling_steps: int = 6,
        input_learning_rate: float = 1.0,
        recurrent_learning_rate: float = 0.25,
        output_learning_rate: float = 1.0,
        inhibitory_learning_rate: float = 0.05,
        recurrent_gain: float = 0.75,
        learned_inhibition_gain: float = 0.5,
        global_inhibition: float = 0.1,
        adaptation_rate: float = 0.05,
        activation_threshold: float = 0.05,
        content_threshold: float = 0.25,
        branch_count: int = 10,
        branch_candidates: int = 80,
        branch_width: int = 3,
        branch_hits: int = 2,
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
        contextual_successor_binding: bool = False,
        contextual_successor_branches_per_source: int = 4,
        contextual_successor_capacity: int = 256,
        contextual_successor_competition_gain: float = 0.0,
        commitment_rate: float = 0.24,
        commitment_substeps: int = 6,
        commitment_self_gain: float = 2.2,
        commitment_bias: float = -4.5,
        coincidence_gain: float = 5.5,
        cue_bound_gain: float = 1.5,
        commitment_lateral_gain: float = 2.4,
        assembly_conflict_gain: float = 40.0,
        cue_conflict_gain: float = 14.0,
        commitment_threshold: float = 0.7,
        capture_recency_gain: float = 0.01,
        recall_recency_gain: float = 0.01,
        capture_half_life: float = 1_024.0,
        recall_half_life: float = 1_024.0,
        seed: int = 76,
    ) -> None:
        if memory_units < assembly_size or assembly_size < 1:
            raise ValueError("memory field must fit at least one sparse assembly")
        if (
            cortical_dimensions < 8
            or settling_steps < 1
            or branch_count < 1
            or branch_width < 2
            or not 1 <= branch_hits <= branch_width
            or commitment_substeps < 1
        ):
            raise ValueError("field dimensions and settling steps must be positive")
        if branch_input_mode not in {"raw", "running_residual", "context_residual"}:
            raise ValueError("unknown branch input mode")
        if not 0.0 < background_learning_rate <= 1.0:
            raise ValueError("background learning rate must be in (0, 1]")
        if residual_seed_gain < 0.0:
            raise ValueError("residual seed gain must be non-negative")
        if dendritic_buffer_half_life < 0.0:
            raise ValueError("dendritic buffer half-life must be non-negative")
        if return_buffer_half_life < 0.0:
            raise ValueError("return buffer half-life must be non-negative")
        if return_buffer_half_life_ms < 0.0:
            raise ValueError("return buffer millisecond half-life must be non-negative")
        if return_buffer_mode not in {"bytes", "milliseconds", "either"}:
            raise ValueError("unknown return buffer mode")
        if not 0.0 <= sequence_trace_decay < 1.0:
            raise ValueError("sequence trace decay must be in [0, 1)")
        if sequence_learning_rate < 0.0:
            raise ValueError("sequence learning rate must be non-negative")
        if sequence_settling_steps < 0:
            raise ValueError("sequence settling steps must be non-negative")
        if sequence_compatibility_gain < 0.0:
            raise ValueError("sequence compatibility gain must be non-negative")
        if not 0.0 <= sequence_stop_compatibility <= 1.0:
            raise ValueError("sequence stop compatibility must be in [0, 1]")
        if contextual_successor_branches_per_source < 1:
            raise ValueError(
                "contextual successor branches per source must be positive"
            )
        if contextual_successor_capacity < 1:
            raise ValueError("contextual successor capacity must be positive")
        if contextual_successor_competition_gain < 0.0:
            raise ValueError(
                "contextual successor competition gain must be non-negative"
            )
        self.cortical_dimensions = cortical_dimensions
        self.memory_units = memory_units
        self.assembly_size = assembly_size
        self.settling_steps = settling_steps
        self.input_learning_rate = input_learning_rate
        self.recurrent_learning_rate = recurrent_learning_rate
        self.output_learning_rate = output_learning_rate
        self.inhibitory_learning_rate = inhibitory_learning_rate
        self.recurrent_gain = recurrent_gain
        self.learned_inhibition_gain = learned_inhibition_gain
        self.global_inhibition = global_inhibition
        self.adaptation_rate = adaptation_rate
        self.activation_threshold = activation_threshold
        self.content_threshold = content_threshold
        self.branch_count = branch_count
        self.branch_candidates = branch_candidates
        self.branch_width = branch_width
        self.branch_hits = branch_hits
        self.heterosynaptic_decay = heterosynaptic_decay
        self.branch_survival_threshold = branch_survival_threshold
        self.branch_input_mode = branch_input_mode
        self.background_learning_rate = background_learning_rate
        self.residual_seed_gain = residual_seed_gain
        self.dendritic_buffer_half_life = dendritic_buffer_half_life
        self.return_buffer_half_life = return_buffer_half_life
        self.return_buffer_half_life_ms = return_buffer_half_life_ms
        self.return_buffer_mode = return_buffer_mode
        self.order_sensitive_binding = order_sensitive_binding
        self.sequence_trace_decay = sequence_trace_decay
        self.sequence_learning_rate = sequence_learning_rate
        self.sequence_settling_steps = sequence_settling_steps
        self.sequence_compatibility_gain = sequence_compatibility_gain
        self.sequence_stop_compatibility = sequence_stop_compatibility
        self.contextual_successor_binding = contextual_successor_binding
        self.contextual_successor_branches_per_source = (
            contextual_successor_branches_per_source
        )
        self.contextual_successor_capacity = contextual_successor_capacity
        self.contextual_successor_competition_gain = (
            contextual_successor_competition_gain
        )
        self.commitment_rate = commitment_rate
        self.commitment_substeps = commitment_substeps
        self.commitment_self_gain = commitment_self_gain
        self.commitment_bias = commitment_bias
        self.coincidence_gain = coincidence_gain
        self.cue_bound_gain = cue_bound_gain
        self.commitment_lateral_gain = commitment_lateral_gain
        self.assembly_conflict_gain = assembly_conflict_gain
        self.cue_conflict_gain = cue_conflict_gain
        self.commitment_threshold = commitment_threshold
        self.capture_recency_gain = capture_recency_gain
        self.recall_recency_gain = recall_recency_gain
        self.capture_half_life = capture_half_life
        self.recall_half_life = recall_half_life
        self.projector = SparseResonantCapsuleIndex(
            dimensions=cortical_dimensions,
            include_pair_hashes=False,
        )
        self.input_weights: list[dict[int, float]] = [{} for _ in range(memory_units)]
        self.input_incoming: list[dict[int, float]] = [
            {} for _ in range(cortical_dimensions)
        ]
        self._input_matrix = np.zeros(
            (memory_units, cortical_dimensions), dtype=np.float64
        )
        self._code_matrix = np.zeros(
            (256, cortical_dimensions), dtype=np.bool_
        )
        self._code_owners = np.full(256, -1, dtype=np.int32)
        self._code_count = 0
        self.recurrent_weights: list[dict[int, float]] = [
            {} for _ in range(memory_units)
        ]
        self.inhibitory_weights: list[dict[int, float]] = [
            {} for _ in range(memory_units)
        ]
        self.output_weights: list[dict[tuple[int, int], float]] = [
            {} for _ in range(memory_units)
        ]
        self._output_items: set[tuple[int, int]] = set()
        self.capture_clock = np.full(memory_units, -1, dtype=np.int64)
        self.recall_clock = np.full(memory_units, -1, dtype=np.int64)
        self.use_count = np.zeros(memory_units, dtype=np.uint32)
        self.engram_units: list[tuple[int, ...]] = []
        self.engram_branches: list[list[DendriticBranch]] = []
        self.engram_codes: list[list[frozenset[int]]] = []
        self.engram_sequence_traces: list[dict[tuple[int, int], float]] = []
        self.engram_sequence_weights: list[
            dict[tuple[int, int], dict[tuple[int, int], float]]
        ] = []
        self.engram_sequence_context_traces: list[
            dict[tuple[int, int], frozenset[int]]
        ] = []
        self.engram_contextual_successors: list[
            list[ContextualSuccessorBranch]
        ] = []
        self._engram_by_units: dict[tuple[int, ...], int] = {}
        self.clock = 0
        self.encodings = 0
        self.reinstatements = 0
        self.sequence_updates = 0
        self.sequence_returns = 0
        self.sequence_dynamics_returns = 0
        self.contextual_successor_births = 0
        self.contextual_successor_deaths = 0
        self.branch_depressions = 0
        self.branch_deaths = 0
        self.branch_reassignments = 0
        self.background_activity = np.zeros(cortical_dimensions, dtype=np.float32)
        self.context_background_activity: dict[int, np.ndarray] = {}
        self.engram_facilitation: list[float] = []
        self.return_buffer: dict[tuple[int, int], float] = {}
        self.return_buffer_time: dict[tuple[int, int], tuple[float, int]] = {}
        self.last_diagnostics: dict[str, float] = {}
        self._rng = np.random.default_rng(seed)

    def _sparsify(self, membrane: np.ndarray) -> np.ndarray:
        activity = np.maximum(0.0, membrane - self.activation_threshold)
        maximum_active = min(2 * self.assembly_size, self.memory_units)
        active = np.flatnonzero(activity > 0.0)
        if len(active) > maximum_active:
            inhibitory_threshold = np.partition(activity[active], -maximum_active)[
                -maximum_active
            ]
            activity[activity < inhibitory_threshold] = 0.0
        return activity

    @property
    def output_items(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._output_items)

    def engrams_containing(
        self, item: tuple[int, int]
    ) -> tuple[int, ...]:
        """Return engrams with an existing local h->receptor binding."""
        matches = []
        for engram, units in enumerate(self.engram_units):
            if any(item in self.output_weights[unit] for unit in units):
                matches.append(engram)
        return tuple(matches)

    def engram_cortical_support(
        self,
        cortical_items: tuple[tuple[int, int], ...],
        engrams: tuple[int, ...],
    ) -> dict[int, float]:
        """Measure local x->h compatibility for simultaneous engram competition."""
        drive = self._drive(self._code(cortical_items))
        support = {}
        for engram in engrams:
            units = np.asarray(self.engram_units[engram], dtype=np.int64)
            support[engram] = float(np.mean(np.maximum(0.0, drive[units])))
        return support

    def contextual_recruitment(
        self,
        cortical_items: tuple[tuple[int, int], ...],
        *,
        context_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read existing basal and apical context support without learning.

        The basal value is ordinary learned cortical-to-engram current.  The
        apical value is support from surviving discriminative conjunction
        branches.  Neither channel advances the background estimate: this is
        a recruitment signal for the present prediction, not an observation.
        """
        count = len(self.engram_units)
        if not cortical_items or count == 0:
            empty = np.zeros(count, dtype=np.float64)
            return empty, empty.copy(), empty.copy()
        code = self._code(cortical_items)
        drive = self._drive(code)
        basal = np.zeros(count, dtype=np.float64)
        for engram, units in enumerate(self.engram_units):
            indices = np.asarray(units, dtype=np.int64)
            basal[engram] = float(np.mean(np.maximum(0.0, drive[indices])))
        branch_input = self._branch_input(code, context_id)
        apical = self._apical_support(branch_input)
        branch_peak = np.zeros(count, dtype=np.float64)
        for engram, branches in enumerate(self.engram_branches):
            branch_peak[engram] = max(
                (
                    branch.strength * self._branch_response(branch, branch_input)
                    for branch in branches
                ),
                default=0.0,
            )
        return basal, apical, branch_peak

    def _code(self, items: tuple[tuple[int, int], ...]) -> np.ndarray:
        return self.projector.sparse_code(items).astype(np.float32, copy=False)

    def cortical_code(self, items: tuple[tuple[int, int], ...]) -> np.ndarray:
        """Expose the shared sparse cortical substrate to adjacent dendrites."""
        return self._code(items)

    def _grow_contextual_successor(
        self,
        engram: int,
        source: tuple[int, int],
        targets: tuple[tuple[int, int], ...],
        source_context: frozenset[int],
        target_context: np.ndarray,
        eligibility: float,
    ) -> None:
        """Grow and prune local context branches for one temporal occurrence."""
        if len(source_context) < self.branch_width or not targets:
            return
        branches = self.engram_contextual_successors[engram]
        target_set = frozenset(targets)
        for branch in branches:
            if branch.source != source or frozenset(branch.targets) == target_set:
                continue
            response = self._branch_response_synapses(
                branch.synapses, source_context
            )
            if response <= 0.0:
                continue
            branch.false_rate = 0.95 * branch.false_rate + 0.05 * response
            branch.strength *= 1.0 - self.heterosynaptic_decay * eligibility
        retained = [
            branch
            for branch in branches
            if branch.strength >= self.branch_survival_threshold
        ]
        self.contextual_successor_deaths += len(branches) - len(retained)
        branches[:] = retained

        matching = [
            branch
            for branch in branches
            if branch.source == source
            and frozenset(branch.targets) == target_set
            and self._branch_response_synapses(branch.synapses, source_context) > 0.0
        ]
        if matching:
            for branch in matching:
                observed_strengths = self._target_residual_strengths(
                    targets, target_context
                )
                branch.target_strengths = tuple(
                    max(old, new)
                    for old, new in zip(
                        branch.target_strengths or observed_strengths,
                        observed_strengths,
                        strict=True,
                    )
                )
                branch.strength = min(
                    4.0,
                    branch.strength + self.sequence_learning_rate * eligibility,
                )
            return

        candidates: set[frozenset[int]] = set()
        dimensions = np.asarray(sorted(source_context), dtype=np.int64)
        maximum_candidates = min(16, len(dimensions) * (len(dimensions) - 1))
        attempts = 0
        while len(candidates) < maximum_candidates and attempts < 1_000:
            sampled = self._rng.choice(
                dimensions, self.branch_width, replace=False
            )
            candidates.add(frozenset(int(value) for value in sampled))
            attempts += 1
        alternatives = [
            branch.context
            for branch in branches
            if branch.source == source and frozenset(branch.targets) != target_set
        ]
        scored = []
        for candidate in candidates:
            false_rate = sum(
                self._branch_response_synapses(candidate, context) > 0.0
                for context in alternatives
            ) / max(1, len(alternatives))
            scored.append((false_rate, tuple(sorted(candidate)), candidate))
        for false_rate, _order, candidate in sorted(scored)[:2]:
            branches.append(
                ContextualSuccessorBranch(
                    source,
                    targets,
                    candidate,
                    source_context,
                    self._target_residual_strengths(targets, target_context),
                    float(np.exp(-4.0 * false_rate)),
                    false_rate,
                )
            )
            self.contextual_successor_births += 1
        same_source = [branch for branch in branches if branch.source == source]
        if len(same_source) > self.contextual_successor_branches_per_source:
            survivors = set(
                id(branch)
                for branch in sorted(
                    same_source,
                    key=lambda branch: (
                        branch.false_rate,
                        -branch.strength,
                        tuple(sorted(branch.synapses)),
                    ),
                )[: self.contextual_successor_branches_per_source]
            )
            before = len(branches)
            branches[:] = [
                branch
                for branch in branches
                if branch.source != source or id(branch) in survivors
            ]
            self.contextual_successor_deaths += before - len(branches)
        if len(branches) > self.contextual_successor_capacity:
            survivors = set(
                id(branch)
                for branch in sorted(
                    branches,
                    key=lambda branch: (
                        branch.false_rate,
                        -branch.strength,
                        tuple(sorted(branch.synapses)),
                        branch.source,
                    ),
                )[: self.contextual_successor_capacity]
            )
            before = len(branches)
            branches[:] = [branch for branch in branches if id(branch) in survivors]
            self.contextual_successor_deaths += before - len(branches)

    def _settle_contextual_successor(
        self,
        engram: int,
        cue_strength: dict[tuple[int, int], float],
        branch_input: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        """Recruit occurrence-specific outgoing branches in parallel."""
        active_context = frozenset(
            int(value) for value in np.flatnonzero(branch_input)
        )
        content: defaultdict[tuple[int, int], float] = defaultdict(float)
        for branch in self.engram_contextual_successors[engram]:
            source_gain = max(0.0, cue_strength.get(branch.source, 0.0))
            if source_gain <= 0.0:
                continue
            response = self._branch_response_synapses(
                branch.synapses, active_context
            )
            if response <= 0.0:
                continue
            branch_gain = source_gain * branch.strength * response
            target_strengths = getattr(branch, "target_strengths", ()) or tuple(
                1.0 for _ in branch.targets
            )
            for target, target_strength in zip(
                branch.targets, target_strengths, strict=True
            ):
                content[target] += branch_gain * target_strength
        maximum = max(content.values(), default=0.0)
        if maximum <= 0.0:
            return {}
        if self.contextual_successor_competition_gain > 0.0:
            items = tuple(content)
            logits = np.asarray([content[item] for item in items], dtype=np.float64)
            logits /= maximum
            logits *= self.contextual_successor_competition_gain
            logits -= float(np.max(logits))
            normalized = np.exp(logits)
            normalized /= max(float(np.sum(normalized)), 1e-12)
            content = defaultdict(
                float,
                {
                    item: float(value)
                    for item, value in zip(items, normalized, strict=True)
                },
            )
            maximum = max(content.values(), default=0.0)
        return {
            item: value / maximum
            for item, value in content.items()
            if value >= self.content_threshold * maximum
        }

    def _target_residual_strengths(
        self,
        targets: tuple[tuple[int, int], ...],
        target_context: np.ndarray,
    ) -> tuple[float, ...]:
        """Measure which coactive targets account for surprising frame activity."""
        strengths = []
        for target in targets:
            target_code = self._code((target,))
            dimensions = np.flatnonzero(target_code)
            strength = (
                float(np.mean(target_context[dimensions]))
                if len(dimensions)
                else 0.0
            )
            strengths.append(max(strength, 1e-6))
        maximum = max(strengths, default=0.0)
        if maximum <= 0.0:
            return tuple(1.0 for _ in targets)
        return tuple(strength / maximum for strength in strengths)

    def _settle_sequence(
        self,
        engram: int,
        cue_items: tuple[tuple[int, int], ...],
        compatibility: dict[tuple[int, int], float] | None,
        *,
        maximum_steps: int | None = None,
        cue_strength: dict[tuple[int, int], float] | None = None,
    ) -> tuple[dict[tuple[int, int], float], int]:
        """Release a cue into locally learned directed successor dynamics."""
        transitions = self.engram_sequence_weights[engram]
        activity = {
            item: (
                1.0 if cue_strength is None else max(0.0, cue_strength.get(item, 0.0))
            )
            for item in cue_items
            if item in transitions
        }
        activity = {item: value for item, value in activity.items() if value > 0.0}
        if not activity:
            return {}, 0
        last_nonempty: dict[tuple[int, int], float] = {}
        iterations = 0
        steps = self.sequence_settling_steps if maximum_steps is None else maximum_steps
        for iterations in range(1, steps + 1):
            propagated: defaultdict[tuple[int, int], float] = defaultdict(float)
            for source, source_activity in activity.items():
                row = transitions.get(source, {})
                row_total = sum(row.values())
                if row_total <= 0.0:
                    continue
                for target, weight in row.items():
                    propagated[target] += source_activity * weight / row_total
            maximum = max(propagated.values(), default=0.0)
            if maximum <= 0.0:
                break
            activity = {
                item: value / maximum
                for item, value in propagated.items()
                if value >= self.content_threshold * maximum
            }
            if not activity:
                break
            last_nonempty = activity
            if compatibility is not None:
                compatible = {
                    item: value
                    * (
                        1.0
                        + self.sequence_compatibility_gain
                        * max(0.0, compatibility.get(item, 0.0))
                    )
                    for item, value in activity.items()
                    if compatibility.get(item, 0.0)
                    >= self.sequence_stop_compatibility
                }
                if compatible:
                    return compatible, iterations
        return last_nonempty, iterations

    def _allocate(self) -> np.ndarray:
        jitter = self._rng.random(self.memory_units) * 1e-6
        allocation = self.use_count.astype(np.float64) + jitter
        return np.argpartition(allocation, self.assembly_size - 1)[: self.assembly_size]

    def _drive(self, code: np.ndarray) -> np.ndarray:
        return self._input_matrix @ code

    def _rebuild_execution_mirrors(self) -> None:
        """Recreate unembodied execution mirrors from sparse anatomy."""
        matrix = np.zeros(
            (self.memory_units, self.cortical_dimensions), dtype=np.float64
        )
        for unit, row in enumerate(self.input_weights):
            if not row:
                continue
            dimensions = np.fromiter(row, dtype=np.intp, count=len(row))
            matrix[unit, dimensions] = np.fromiter(
                row.values(), dtype=np.float64, count=len(row)
            )
        self._input_matrix = matrix
        frames = [
            (engram, frame)
            for engram, rows in enumerate(self.engram_codes)
            for frame in rows
        ]
        capacity = max(256, len(frames))
        self._code_matrix = np.zeros(
            (capacity, self.cortical_dimensions), dtype=np.bool_
        )
        self._code_owners = np.full(capacity, -1, dtype=np.int32)
        self._code_count = 0
        for engram, frame in frames:
            self._append_code_mirror(engram, frame)

    def _append_code_mirror(self, engram: int, frame: frozenset[int]) -> None:
        """Add one sparse cortical frame to the metabolic incidence bank."""
        if self._code_count >= len(self._code_owners):
            capacity = max(256, 2 * len(self._code_owners))
            matrix = np.zeros(
                (capacity, self.cortical_dimensions), dtype=np.bool_
            )
            matrix[: self._code_count] = self._code_matrix[: self._code_count]
            owners = np.full(capacity, -1, dtype=np.int32)
            owners[: self._code_count] = self._code_owners[: self._code_count]
            self._code_matrix = matrix
            self._code_owners = owners
        if frame:
            self._code_matrix[self._code_count, tuple(frame)] = True
        self._code_owners[self._code_count] = engram
        self._code_count += 1

    def _branch_input(self, code: np.ndarray, context_id: int | None) -> np.ndarray:
        """Expose locally surprising positive activity to apical branches."""
        active = code > 0.0
        if self.branch_input_mode == "raw":
            return active.astype(np.float32)
        expected = self.background_activity
        if self.branch_input_mode == "context_residual" and context_id is not None:
            expected = self.context_background_activity.get(
                int(context_id), self.background_activity
            )
        residual = np.zeros_like(code, dtype=np.float32)
        residual[active] = np.maximum(
            0.0,
            (code[active] - expected[active]) / np.maximum(code[active], 1e-12),
        )
        return residual

    def _observe_background(self, code: np.ndarray, context_id: int | None) -> None:
        if self.branch_input_mode == "raw":
            return
        rate = self.background_learning_rate
        self.background_activity *= 1.0 - rate
        self.background_activity += rate * code
        if self.branch_input_mode != "context_residual" or context_id is None:
            return
        context = self.context_background_activity.setdefault(
            int(context_id), np.zeros(self.cortical_dimensions, dtype=np.float32)
        )
        context *= 1.0 - rate
        context += rate * code

    def _facilitate_apical_support(self, support: np.ndarray) -> np.ndarray:
        """Boost repeated selective evidence without creating a candidate."""
        if self.dendritic_buffer_half_life <= 0.0:
            return support
        decay = 2.0 ** (-1.0 / self.dendritic_buffer_half_life)
        active = support > 0.15
        for engram in range(len(self.engram_facilitation)):
            trace = decay * self.engram_facilitation[engram]
            if active[engram]:
                trace = min(1.0, trace + float(support[engram]))
            self.engram_facilitation[engram] = trace
        facilitated = np.zeros_like(support)
        traces = np.asarray(self.engram_facilitation, dtype=np.float64)
        facilitated[active] = np.maximum(support[active], traces[active])
        return facilitated

    @staticmethod
    def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

    def _bind_branches(self, engram: int, branch_input: np.ndarray) -> None:
        dimensions = np.flatnonzero(branch_input)
        if len(dimensions) < self.branch_width:
            return
        candidates: set[frozenset[int]] = set()
        maximum_candidates = min(
            self.branch_candidates,
            len(dimensions) * (len(dimensions) - 1),
        )
        attempts = 0
        while len(candidates) < maximum_candidates and attempts < 10_000:
            sampled = self._rng.choice(dimensions, self.branch_width, replace=False)
            candidates.add(frozenset(int(value) for value in sampled))
            attempts += 1
        ordered_candidates = sorted(
            candidates, key=lambda candidate: tuple(sorted(candidate))
        )
        false_rates = np.zeros(len(ordered_candidates), dtype=np.float64)
        alternative_mask = (
            self._code_owners[: self._code_count] != engram
        )
        alternative_count = int(np.count_nonzero(alternative_mask))
        if alternative_count and ordered_candidates:
            candidate_indices = np.asarray(
                [tuple(sorted(candidate)) for candidate in ordered_candidates],
                dtype=np.intp,
            )
            # All provisional branches confront all previous sparse events in
            # one batched local coincidence operation.  Integer hit counts are
            # exactly the old frozenset-intersection response criterion.
            responses = np.sum(
                self._code_matrix[: self._code_count][alternative_mask][
                    :, candidate_indices
                ],
                axis=2,
                dtype=np.int16,
            )
            false_rates = np.mean(
                responses >= self.branch_hits, axis=0, dtype=np.float64
            )
        scored = [
            (float(false_rate), tuple(sorted(candidate)), candidate)
            for candidate, false_rate in zip(
                ordered_candidates, false_rates, strict=True
            )
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        nursery = [
            DendriticBranch(
                candidate,
                float(np.exp(-4.0 * false_rate)),
                false_rate,
            )
            for false_rate, _order, candidate in scored
        ]
        previous = self.engram_branches[engram]
        combined = {
            branch.synapses: branch for branch in previous + nursery if branch.synapses
        }
        survivors = sorted(
            combined.values(),
            key=lambda branch: (
                branch.false_rate,
                -branch.strength,
                tuple(sorted(branch.synapses)),
            ),
        )[: self.branch_count]
        retained = sum(branch in previous for branch in survivors)
        self.branch_reassignments += max(0, len(previous) - retained)
        self.engram_branches[engram] = survivors

    def _branch_response_synapses(
        self,
        synapses: frozenset[int],
        branch_input: np.ndarray | set[int] | frozenset[int],
    ) -> float:
        if not synapses:
            return 0.0
        if isinstance(branch_input, np.ndarray):
            response = sum(float(branch_input[index]) for index in synapses)
            return 0.0 if response < self.branch_hits else response / len(synapses)
        hits = len(synapses & branch_input)
        return 0.0 if hits < self.branch_hits else hits / len(synapses)

    def _branch_response(
        self, branch: DendriticBranch, branch_input: np.ndarray
    ) -> float:
        if not branch.synapses or branch.strength <= 0.0:
            return 0.0
        return self._branch_response_synapses(branch.synapses, branch_input)

    def _depress_competing_branches(
        self, engram: int, branch_input: np.ndarray
    ) -> None:
        """Remove conjunctions that remain active during other events."""
        for other, branches in enumerate(self.engram_branches):
            if other == engram:
                continue
            for branch in branches:
                response = self._branch_response(branch, branch_input)
                if response <= 0.0:
                    continue
                previous = branch.strength
                branch.strength *= 1.0 - self.heterosynaptic_decay
                if branch.strength < previous:
                    self.branch_depressions += 1
                if branch.strength < self.branch_survival_threshold and branch.synapses:
                    branch.synapses = frozenset()
                    branch.strength = 0.0
                    self.branch_deaths += 1

    def _apical_support(self, branch_input: np.ndarray) -> np.ndarray:
        """Direct cortical conjunction support, isolated from recurrence."""
        support = np.zeros(len(self.engram_units), dtype=np.float64)
        for engram, branches in enumerate(self.engram_branches):
            live = [branch for branch in branches if branch.strength > 0.0]
            if not live:
                continue
            numerator = sum(
                branch.strength * self._branch_response(branch, branch_input)
                for branch in live
            )
            support[engram] = numerator / max(
                sum(branch.strength for branch in live), 1e-12
            )
        return support

    def _autonomous_settle(
        self,
        initial_activity: np.ndarray,
        target_engram: int,
        *,
        persistent_clamp: bool = False,
    ) -> tuple[float, float]:
        """Release a brief cue and measure recurrent completion without input."""
        activity = initial_activity.astype(np.float64, copy=True)
        target_units = np.asarray(self.engram_units[target_engram], dtype=np.int64)
        adaptation = np.zeros(self.memory_units, dtype=np.float64)
        for _ in range(self.settling_steps):
            recurrent = np.zeros(self.memory_units, dtype=np.float64)
            learned_inhibition = np.zeros(self.memory_units, dtype=np.float64)
            for source in np.flatnonzero(activity):
                source_activity = float(activity[source])
                for target, weight in self.recurrent_weights[int(source)].items():
                    recurrent[target] += weight * source_activity
                for target, weight in self.inhibitory_weights[int(source)].items():
                    learned_inhibition[target] += weight * source_activity
            excess = max(0.0, float(np.sum(activity)) - self.assembly_size)
            membrane = (
                self.recurrent_gain * recurrent
                - self.learned_inhibition_gain * learned_inhibition
                - self.global_inhibition * excess
                - adaptation
            )
            activity = self._sparsify(membrane)
            divisor = max(1.0, float(np.sum(activity)) / self.assembly_size)
            activity /= divisor
            if persistent_clamp:
                activity[target_units] = np.maximum(activity[target_units], 1.0)
            adaptation = 0.8 * adaptation + self.adaptation_rate * activity
        scale = max(float(np.max(activity, initial=0.0)), 1e-12)
        completion = np.asarray(
            [
                float(np.mean(activity[np.asarray(units, dtype=np.int64)]) / scale)
                for units in self.engram_units
            ],
            dtype=np.float64,
        )
        target = float(completion[target_engram])
        competitors = np.delete(completion, target_engram)
        return target, float(np.max(competitors, initial=0.0))

    def probe_unique_attractor(
        self,
        cortical_items: tuple[tuple[int, int], ...],
        *,
        context_id: int | None = None,
        seed_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
    ) -> dict[str, object] | None:
        """Probe the sole residual candidate without changing field state."""
        if not cortical_items or not self.encodings:
            return None
        code = self._code(cortical_items)
        branch_input = self._branch_input(code, context_id)
        support = self._apical_support(branch_input)
        eligible = np.flatnonzero(support > 0.15)
        if len(eligible) != 1:
            return None
        target = int(eligible[0])
        target_units = np.asarray(self.engram_units[target], dtype=np.int64)

        natural = self._sparsify(self._drive(code) + self._recency_bias())
        natural_target, natural_competitor = self._autonomous_settle(natural, target)

        seed_results: dict[str, dict[str, float]] = {}
        for fraction in seed_fractions:
            count = max(1, int(np.ceil(fraction * len(target_units))))
            initial = np.zeros(self.memory_units, dtype=np.float64)
            initial[target_units[:count]] = 1.0
            target_score, competitor_score = self._autonomous_settle(initial, target)
            seed_results[f"{fraction:.2f}"] = {
                "target": target_score,
                "competitor": competitor_score,
            }

        stored_dimensions = max(
            self.engram_codes[target],
            key=lambda stored: len(stored & set(np.flatnonzero(code))),
        )
        reconstruction = np.zeros(self.cortical_dimensions, dtype=np.float32)
        reconstruction[list(stored_dimensions)] = 1.0 / max(
            1.0, len(stored_dimensions) ** 0.5
        )
        reconstructed_activity = self._sparsify(self._drive(reconstruction))
        reconstruction_target, reconstruction_competitor = self._autonomous_settle(
            reconstructed_activity, target
        )

        persistent = np.zeros(self.memory_units, dtype=np.float64)
        persistent[target_units] = 1.0
        persistent_target, persistent_competitor = self._autonomous_settle(
            persistent, target, persistent_clamp=True
        )
        return {
            "target_engram": target,
            "apical_support": float(support[target]),
            "natural": {
                "target": natural_target,
                "competitor": natural_competitor,
            },
            "seed": seed_results,
            "reconstruction": {
                "target": reconstruction_target,
                "competitor": reconstruction_competitor,
            },
            "persistent": {
                "target": persistent_target,
                "competitor": persistent_competitor,
            },
        }

    def encode(
        self,
        cortical_items: tuple[tuple[int, int], ...],
        output_items: tuple[tuple[int, int], ...],
        *,
        units: tuple[int, ...] | None = None,
        context_id: int | None = None,
        reset_sequence: bool = False,
        update_access_branches: bool = True,
    ) -> tuple[int, ...]:
        """Apply local x→h, h→h, and h→receptor Hebbian updates."""
        if not cortical_items or not output_items:
            return ()
        code = self._code(cortical_items)
        branch_input = self._branch_input(code, context_id)
        frame_code = frozenset(
            int(value) for value in np.flatnonzero(branch_input)
        )
        allocated = (
            self._allocate() if units is None else np.asarray(units, dtype=np.int64)
        )
        unit_key = tuple(sorted(int(unit) for unit in allocated))
        engram = self._engram_by_units.get(unit_key)
        if engram is None:
            engram = len(self.engram_units)
            self._engram_by_units[unit_key] = engram
            self.engram_units.append(unit_key)
            self.engram_branches.append([])
            self.engram_codes.append([])
            self.engram_facilitation.append(0.0)
            self.engram_sequence_traces.append({})
            self.engram_sequence_weights.append({})
            self.engram_sequence_context_traces.append({})
            self.engram_contextual_successors.append([])
        while len(self.engram_sequence_context_traces) <= engram:
            self.engram_sequence_context_traces.append({})
        while len(self.engram_contextual_successors) <= engram:
            self.engram_contextual_successors.append([])
        if self.order_sensitive_binding:
            trace = self.engram_sequence_traces[engram]
            context_trace = self.engram_sequence_context_traces[engram]
            if reset_sequence:
                trace.clear()
                context_trace.clear()
            else:
                for source in tuple(trace):
                    trace[source] *= self.sequence_trace_decay
                    if trace[source] < 1e-6:
                        del trace[source]
                        context_trace.pop(source, None)
            transitions = self.engram_sequence_weights[engram]
            for source, eligibility in trace.items():
                targets = transitions.setdefault(source, {})
                for target in output_items:
                    if target == source:
                        continue
                    targets[target] = min(
                        4.0,
                        targets.get(target, 0.0)
                        + self.sequence_learning_rate * eligibility,
                    )
                    self.sequence_updates += 1
                if self.contextual_successor_binding:
                    self._grow_contextual_successor(
                        engram,
                        source,
                        tuple(item for item in output_items if item != source)[:8],
                        context_trace.get(source, frozenset()),
                        branch_input,
                        eligibility,
                    )
            for item in output_items:
                trace[item] = max(1.0, trace.get(item, 0.0))
                context_trace[item] = frame_code
        if update_access_branches:
            self._bind_branches(engram, branch_input)
        if frame_code not in self.engram_codes[engram]:
            self.engram_codes[engram].append(frame_code)
            self._append_code_mirror(engram, frame_code)
        if update_access_branches:
            self._depress_competing_branches(engram, branch_input)
        drive = self._drive(code)
        competitors = np.argpartition(
            drive, -min(2 * self.assembly_size, self.memory_units)
        )[-min(2 * self.assembly_size, self.memory_units) :]
        competitors = np.setdiff1d(competitors, allocated, assume_unique=False)
        for unit in allocated:
            unit_id = int(unit)
            row = self.input_weights[unit_id]
            for dimension in np.flatnonzero(code):
                dimension_id = int(dimension)
                row[dimension_id] = row.get(dimension_id, 0.0) + (
                    self.input_learning_rate * float(code[dimension])
                )
            norm = sum(weight * weight for weight in row.values()) ** 0.5
            if norm > 0.0:
                for dimension in tuple(row):
                    row[dimension] /= norm
                    self.input_incoming[dimension][unit_id] = row[dimension]
                    self._input_matrix[unit_id, dimension] = row[dimension]
            self.use_count[unit] += 1
            self.capture_clock[unit] = self.clock
            for item in output_items:
                self._output_items.add(item)
                weights = self.output_weights[unit]
                weights[item] = weights.get(item, 0.0) + self.output_learning_rate
        for source in allocated:
            row = self.recurrent_weights[int(source)]
            for target in allocated:
                if target != source:
                    target_id = int(target)
                    row[target_id] = min(
                        1.0,
                        row.get(target_id, 0.0) + self.recurrent_learning_rate,
                    )
        if len(competitors):
            for source in allocated:
                row = self.inhibitory_weights[int(source)]
                for target in competitors:
                    target_id = int(target)
                    row[target_id] = min(
                        1.0,
                        row.get(target_id, 0.0) + self.inhibitory_learning_rate,
                    )
        self.encodings += 1
        return tuple(int(unit) for unit in allocated)

    def advance(self) -> None:
        self.clock += 1
        if (
            self.return_buffer_mode in {"bytes", "either"}
            and self.return_buffer_half_life > 0.0
        ):
            decay = 2.0 ** (-1.0 / self.return_buffer_half_life)
            self.return_buffer = {
                item: strength * decay
                for item, strength in self.return_buffer.items()
                if strength * decay >= 0.01
            }
        elif self.return_buffer_mode in {"bytes", "either"}:
            self.return_buffer.clear()

    def reset_episode(self) -> None:
        """Clear fast dendritic state while preserving learned episodic tissue."""
        self.engram_facilitation = [0.0 for _ in self.engram_facilitation]
        self.return_buffer.clear()
        self.return_buffer_time.clear()
        self.recall_clock.fill(-1)
        self.last_diagnostics = {}

    def return_signal(
        self, completion: FieldCompletion | None = None
    ) -> FieldReturnSignal | None:
        """Expose a decaying recalled channel without fabricating observation."""
        content = (
            dict(self.return_buffer)
            if self.return_buffer_mode in {"bytes", "either"}
            else {}
        )
        if self.return_buffer_mode in {"milliseconds", "either"}:
            now = monotonic_ns()
            half_life_ns = self.return_buffer_half_life_ms * 1_000_000.0
            expired = []
            for item, (initial, written_at) in self.return_buffer_time.items():
                strength = (
                    0.0
                    if half_life_ns <= 0.0
                    else initial * 2.0 ** (-(now - written_at) / half_life_ns)
                )
                if strength < 0.01:
                    expired.append(item)
                    continue
                content[item] = max(content.get(item, 0.0), strength)
            for item in expired:
                self.return_buffer_time.pop(item, None)
        if completion is not None:
            strengths = completion.item_strengths or tuple(
                1.0 for _ in completion.items
            )
            for item, strength in zip(completion.items, strengths, strict=True):
                content[item] = max(
                    content.get(item, 0.0),
                    completion.commitment * strength,
                )
        maximum = max(content.values(), default=0.0)
        if maximum <= 0.0:
            return None
        selected = tuple(
            (item, strength)
            for item, strength in sorted(
                content.items(), key=lambda pair: (-pair[1], pair[0])
            )
            if strength >= 0.25 * maximum
        )[: self.assembly_size]
        if not selected:
            return None
        return FieldReturnSignal(
            tuple(item for item, _ in selected),
            maximum,
            tuple(strength / maximum for _, strength in selected),
        )

    def _recency_bias(self) -> np.ndarray:
        capture_age = np.maximum(0, self.clock - self.capture_clock)
        recall_age = np.maximum(0, self.clock - self.recall_clock)
        capture = np.where(
            self.capture_clock >= 0,
            self.capture_recency_gain * 2.0 ** (-capture_age / self.capture_half_life),
            0.0,
        )
        recall = np.where(
            self.recall_clock >= 0,
            self.recall_recency_gain * 2.0 ** (-recall_age / self.recall_half_life),
            0.0,
        )
        return capture + recall

    def retrieve(
        self,
        cortical_items: tuple[tuple[int, int], ...],
        *,
        content_gain: dict[tuple[int, int], float] | None = None,
        context_id: int | None = None,
        sequence_cue_items: tuple[tuple[int, int], ...] = (),
        external_apical_support: np.ndarray | None = None,
        access_content: dict[
            int, dict[tuple[int, int], float]
        ] | None = None,
        access_return_mode: str = "direct",
        access_sequence_seed: dict[
            int, dict[tuple[int, int], float]
        ] | None = None,
    ) -> FieldCompletion | None:
        """Settle recurrently from a partial cortical cue and open local returns."""
        if access_return_mode not in {
            "direct",
            "successor",
            "aligned_successor",
            "content_successor",
            "contextual_successor",
        }:
            raise ValueError("unknown access return mode")
        if not cortical_items or not self.encodings:
            return None
        code = self._code(cortical_items)
        branch_input = self._branch_input(code, context_id)
        self._observe_background(code, context_id)
        drive = self._drive(code)
        instantaneous_support = self._apical_support(branch_input)
        if external_apical_support is not None:
            supplied = np.asarray(external_apical_support, dtype=np.float64)
            if supplied.shape != instantaneous_support.shape:
                raise ValueError("external apical support has the wrong shape")
            # A learned access assembly recruits its own local inhibitory field.
            # This suppresses incompatible raw-branch interpretations before
            # conflict-sensitive commitment, without weakening the veto when
            # multiple learned access assemblies themselves remain active.
            access_peak = float(np.max(supplied, initial=0.0))
            if access_peak > 0.0:
                incompatible = supplied <= 0.15
                instantaneous_support[incompatible] *= 1.0 - min(1.0, access_peak)
            instantaneous_support = np.maximum(instantaneous_support, supplied)
        apical_support = self._facilitate_apical_support(instantaneous_support)
        residual_seed = np.zeros(self.memory_units, dtype=np.float64)
        for engram, units in enumerate(self.engram_units):
            if apical_support[engram] > 0.15:
                residual_seed[np.asarray(units, dtype=np.int64)] = apical_support[
                    engram
                ]
        selective_seed = (
            self.branch_input_mode != "raw" and self.residual_seed_gain > 0.0
        )
        recurrent_feedforward = np.zeros_like(drive) if selective_seed else drive
        active_apical = np.where(apical_support > 0.15, apical_support, 0.0)
        cue_pair_conflict = max(
            0.0,
            (
                float(np.sum(active_apical)) ** 2
                - float(np.dot(active_apical, active_apical))
            )
            / 2.0,
        )
        cue_veto_floor = float(self._sigmoid(-3.0 * 0.35))
        cue_veto = max(
            0.0,
            (float(self._sigmoid(3.0 * (cue_pair_conflict - 0.35))) - cue_veto_floor)
            / (1.0 - cue_veto_floor),
        )
        self.last_diagnostics = {
            "maximum_commitment": 0.0,
            "maximum_apical_support": float(np.max(apical_support, initial=0.0)),
            "maximum_instantaneous_support": float(
                np.max(instantaneous_support, initial=0.0)
            ),
            "eligible_memories": float(np.count_nonzero(active_apical > 0.0)),
            "maximum_completion": 0.0,
            "maximum_drive": float(np.max(drive, initial=0.0)),
            "cue_pair_conflict": cue_pair_conflict,
            "cue_veto": cue_veto,
            "assembly_conflict": 0.0,
        }
        membrane = (
            recurrent_feedforward
            + self._recency_bias()
            + self.residual_seed_gain * residual_seed
        )
        activity = self._sparsify(membrane)
        adaptation = np.zeros(self.memory_units, dtype=np.float64)
        commitment = np.zeros(len(self.engram_units), dtype=np.float64)
        for _ in range(self.settling_steps):
            recurrent = np.zeros(self.memory_units, dtype=np.float64)
            learned_inhibition = np.zeros(self.memory_units, dtype=np.float64)
            for source in np.flatnonzero(activity):
                source_activity = float(activity[source])
                for target, weight in self.recurrent_weights[int(source)].items():
                    recurrent[target] += weight * source_activity
                for target, weight in self.inhibitory_weights[int(source)].items():
                    learned_inhibition[target] += weight * source_activity
            excess = max(0.0, float(np.sum(activity)) - self.assembly_size)
            activity_scale = max(float(np.max(activity, initial=0.0)), 1e-12)
            completion = np.asarray(
                [
                    float(
                        np.mean(activity[np.asarray(units, dtype=np.int64)])
                        / activity_scale
                    )
                    for units in self.engram_units
                ],
                dtype=np.float64,
            )
            cue_support = np.tanh(np.maximum(0.0, drive) / 1.8)
            cue_bound = np.asarray(
                [
                    float(
                        np.mean(
                            activity[np.asarray(units, dtype=np.int64)]
                            * cue_support[np.asarray(units, dtype=np.int64)]
                        )
                        / activity_scale
                    )
                    for units in self.engram_units
                ],
                dtype=np.float64,
            )
            assembly_conflict = max(
                0.0,
                (float(np.sum(completion)) ** 2 - float(np.dot(completion, completion)))
                / 2.0,
            )
            for _substep in range(self.commitment_substeps):
                lateral = float(np.sum(commitment)) - commitment
                commitment_drive = (
                    self.commitment_bias
                    + self.coincidence_gain * completion * apical_support
                    + self.cue_bound_gain * cue_bound
                    + self.commitment_self_gain * commitment
                    - self.commitment_lateral_gain * lateral
                    - self.assembly_conflict_gain * assembly_conflict
                    - self.cue_conflict_gain * cue_veto
                )
                commitment_target = self._sigmoid(commitment_drive)
                commitment = (
                    1.0 - self.commitment_rate
                ) * commitment + self.commitment_rate * commitment_target
            unit_commitment = np.zeros(self.memory_units, dtype=np.float64)
            for engram, units in enumerate(self.engram_units):
                unit_commitment[np.asarray(units, dtype=np.int64)] = commitment[engram]
            local_disinhibition = unit_commitment * recurrent
            membrane = (
                recurrent_feedforward
                + self._recency_bias()
                + self.recurrent_gain * recurrent
                + local_disinhibition
                - self.learned_inhibition_gain * learned_inhibition
                - self.global_inhibition * excess
                - adaptation
            )
            activity = self._sparsify(membrane)
            divisor = max(1.0, float(np.sum(activity)) / self.assembly_size)
            activity /= divisor
            adaptation = 0.8 * adaptation + self.adaptation_rate * activity
        total = float(np.sum(activity))
        if total <= 0.0:
            return None
        active = np.flatnonzero(activity > self.activation_threshold)
        if not len(active):
            return None
        strongest = active[
            np.argpartition(activity[active], -min(self.assembly_size, len(active)))[
                -min(self.assembly_size, len(active)) :
            ]
        ]
        coherent_mass = float(np.sum(activity[strongest]))
        coherence = coherent_mass / total
        cue_overlap = float(np.max(drive[strongest]))
        committed = np.flatnonzero(commitment >= self.commitment_threshold)
        maximum_commitment = float(np.max(commitment, initial=0.0))
        self.last_diagnostics = {
            "maximum_commitment": maximum_commitment,
            "maximum_apical_support": float(np.max(apical_support, initial=0.0)),
            "maximum_instantaneous_support": float(
                np.max(instantaneous_support, initial=0.0)
            ),
            "eligible_memories": float(np.count_nonzero(active_apical > 0.0)),
            "maximum_completion": float(np.max(completion, initial=0.0)),
            "maximum_drive": float(np.max(drive, initial=0.0)),
            "cue_pair_conflict": cue_pair_conflict,
            "cue_veto": cue_veto,
            "assembly_conflict": assembly_conflict,
        }
        if not len(committed):
            return None
        strength = coherence * cue_overlap * maximum_commitment
        content: defaultdict[tuple[int, int], float] = defaultdict(float)
        sequence_used = False
        sequence_iterations = 0
        if access_content and access_return_mode == "direct":
            for engram in committed:
                engram_id = int(engram)
                bound = access_content.get(engram_id, {})
                commitment_gain = float(commitment[engram_id])
                for item, access_strength in bound.items():
                    if any(
                        item in self.output_weights[unit]
                        for unit in self.engram_units[engram_id]
                    ):
                        content[item] += access_strength * commitment_gain
        elif (
            access_content
            and access_return_mode
            in {
                "successor",
                "aligned_successor",
                "content_successor",
                "contextual_successor",
            }
            and self.order_sensitive_binding
            and (
                sequence_cue_items
                or access_return_mode in {"aligned_successor", "content_successor"}
                or access_return_mode == "contextual_successor"
            )
        ):
            for engram in committed:
                engram_id = int(engram)
                if engram_id not in access_content:
                    continue
                cue = sequence_cue_items
                seed_strength = None
                if access_return_mode == "aligned_successor":
                    seed_strength = (access_sequence_seed or {}).get(engram_id, {})
                    cue = tuple(seed_strength)
                elif access_return_mode == "content_successor":
                    seed_strength = access_content.get(engram_id, {})
                    cue = tuple(seed_strength)
                if access_return_mode == "contextual_successor":
                    settled = self._settle_contextual_successor(
                        engram_id,
                        access_content.get(engram_id, {}),
                        branch_input,
                    )
                    iterations = int(bool(settled))
                else:
                    settled, iterations = self._settle_sequence(
                        engram_id,
                        cue,
                        content_gain,
                        maximum_steps=1,
                        cue_strength=seed_strength,
                    )
                sequence_iterations = max(sequence_iterations, iterations)
                commitment_gain = float(commitment[engram_id])
                for item, value in settled.items():
                    content[item] += value * commitment_gain
                    sequence_used = True
        elif (
            self.order_sensitive_binding
            and sequence_cue_items
            and self.sequence_settling_steps > 0
        ):
            for engram in committed:
                settled, iterations = self._settle_sequence(
                    int(engram), sequence_cue_items, content_gain
                )
                sequence_iterations = max(sequence_iterations, iterations)
                commitment_gain = float(commitment[int(engram)])
                for item, value in settled.items():
                    content[item] += value * commitment_gain
                    sequence_used = True
        elif self.order_sensitive_binding and sequence_cue_items:
            for engram in committed:
                commitment_gain = float(commitment[int(engram)])
                transitions = self.engram_sequence_weights[int(engram)]
                for source in sequence_cue_items:
                    for item, weight in transitions.get(source, {}).items():
                        gain = (
                            1.0
                            if content_gain is None
                            else content_gain.get(item, 0.0)
                        )
                        content[item] += weight * gain * commitment_gain
                        sequence_used = True
        elif not self.order_sensitive_binding:
            for engram in committed:
                commitment_gain = float(commitment[int(engram)])
                for unit in self.engram_units[int(engram)]:
                    for item, weight in self.output_weights[unit].items():
                        gain = (
                            1.0
                            if content_gain is None
                            else content_gain.get(item, 0.0)
                        )
                        content[item] += (
                            float(activity[unit]) * weight * gain * commitment_gain
                        )
        maximum = max(content.values(), default=0.0)
        if maximum <= 0.0:
            return None
        completed = [
            (item, value)
            for item, value in content.items()
            if value >= self.content_threshold * maximum
        ]
        completed.sort(key=lambda pair: (-pair[1], pair[0]))
        items = tuple(item for item, _ in completed[: self.assembly_size])
        item_strengths = tuple(
            value / maximum for _, value in completed[: self.assembly_size]
        )
        if sequence_used:
            self.sequence_returns += 1
            if self.sequence_settling_steps > 0:
                self.sequence_dynamics_returns += 1
                completed = completed[:1]
                items = tuple(item for item, _ in completed)
                item_strengths = tuple(value / maximum for _, value in completed)
        return FieldCompletion(
            tuple(int(unit) for unit in strongest),
            items,
            strength,
            coherence,
            cue_overlap,
            maximum_commitment,
            float(np.max(apical_support[committed], initial=0.0)),
            len(committed),
            self.settling_steps + sequence_iterations,
            sequence_used,
            item_strengths,
        )

    def reinstate(self, completion: FieldCompletion) -> None:
        """Refresh recall excitability only after the return path is opened."""
        self.recall_clock[np.asarray(completion.units, dtype=np.int64)] = self.clock
        now = monotonic_ns()
        strengths = completion.item_strengths or tuple(1.0 for _ in completion.items)
        for item, item_strength in zip(completion.items, strengths, strict=True):
            returned_strength = completion.commitment * item_strength
            if self.return_buffer_mode in {"bytes", "either"}:
                self.return_buffer[item] = max(
                    self.return_buffer.get(item, 0.0), returned_strength
                )
            if (
                self.return_buffer_mode in {"milliseconds", "either"}
                and self.return_buffer_half_life_ms > 0.0
            ):
                previous = self.return_buffer_time.get(item, (0.0, now))[0]
                self.return_buffer_time[item] = (
                    max(previous, returned_strength),
                    now,
                )
        self.reinstatements += 1
