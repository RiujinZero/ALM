"""Delayed, locally taught access dendrites for episodic engrams.

The field does not decide at capture time which part of an event will matter.
A cortical query instead leaves a short eligibility trace.  If later external
sensory activity reactivates content already bound to an engram, coincidence
strengthens a query -> engram -> content return path.  Recalled activity is
never accepted as the confirming signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.compiled_kernels import (
    access_branch_survival_compiled,
)

RecallItem = tuple[int, int]


@dataclass
class AccessBranch:
    """One occurrence-specific dendritic conjunction on an access row."""

    weights: dict[int, float]
    strength: float
    created_at: int
    last_used: int
    false_responses: int = 0
    group_targets: tuple[float, ...] = ()
    capture_salience: float = 0.0


@dataclass(frozen=True)
class RelationalAccessSignal:
    """Parallel dendritic support and receptor-bound content preferences."""

    engram_support: np.ndarray
    content_support: dict[int, dict[RecallItem, float]]
    source_support: dict[int, dict[RecallItem, float]]


class RelationalAccessField:
    """Learn delayed heteroassociations from query traces to stored content."""

    def __init__(
        self,
        dimensions: int,
        *,
        trace_decay: float = 0.85,
        learning_rate: float = 1.0,
        activation_threshold: float = 0.35,
        maximum_synapses: int = 96,
        learning_mode: str = "hebbian",
        background_learning_rate: float = 0.001,
        consume_on_confirmation: bool = False,
        latch_until_confirmation: bool = False,
        eligibility_mode: str = "maximum",
        eligibility_bank_size: int = 32,
        minimum_confirmation_age: int = 0,
        maximum_branches_per_row: int = 8,
        maximum_branches: int = 8192,
        branch_false_response_decay: float = 0.25,
        branch_survival_threshold: float = 0.15,
        branch_compartments: int = 4,
        branch_false_response_grace: int = 0,
        branch_survival_mode: str = "uniform",
    ) -> None:
        if dimensions < 1 or maximum_synapses < 1:
            raise ValueError("dimensions and synapse limit must be positive")
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError("trace decay must be in [0, 1)")
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if not 0.0 < activation_threshold <= 1.0:
            raise ValueError("activation threshold must be in (0, 1]")
        if learning_mode not in {"hebbian", "prediction_error", "branches"}:
            raise ValueError("unknown relational access learning mode")
        if not 0.0 < background_learning_rate <= 1.0:
            raise ValueError("background learning rate must be in (0, 1]")
        if eligibility_mode not in {"maximum", "additive", "parallel"}:
            raise ValueError("unknown relational eligibility mode")
        if eligibility_bank_size < 1:
            raise ValueError("eligibility bank size must be positive")
        if minimum_confirmation_age < 0:
            raise ValueError("minimum confirmation age must be nonnegative")
        if maximum_branches_per_row < 1 or maximum_branches < 1:
            raise ValueError("access branch capacities must be positive")
        if not 0.0 <= branch_false_response_decay <= 1.0:
            raise ValueError("branch false-response decay must be in [0, 1]")
        if not 0.0 <= branch_survival_threshold <= 1.0:
            raise ValueError("branch survival threshold must be in [0, 1]")
        if branch_compartments < 1:
            raise ValueError("branch compartments must be positive")
        if branch_false_response_grace < 0:
            raise ValueError("branch false-response grace must be nonnegative")
        if branch_survival_mode not in {"uniform", "surprise"}:
            raise ValueError("unknown branch survival mode")
        self.dimensions = dimensions
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate
        self.activation_threshold = activation_threshold
        self.maximum_synapses = maximum_synapses
        self.learning_mode = learning_mode
        self.background_learning_rate = background_learning_rate
        self.consume_on_confirmation = consume_on_confirmation
        self.latch_until_confirmation = latch_until_confirmation
        self.eligibility_mode = eligibility_mode
        self.eligibility_bank_size = eligibility_bank_size
        self.minimum_confirmation_age = minimum_confirmation_age
        self.maximum_branches_per_row = maximum_branches_per_row
        self.maximum_branches = maximum_branches
        self.branch_false_response_decay = branch_false_response_decay
        self.branch_survival_threshold = branch_survival_threshold
        self.branch_compartments = branch_compartments
        self.branch_false_response_grace = branch_false_response_grace
        self.branch_survival_mode = branch_survival_mode
        self.eligibility = np.zeros(dimensions, dtype=np.float32)
        self.eligibility_bank: list[np.ndarray] = []
        self.eligibility_births: list[int] = []
        self.eligibility_saliences: list[float] = []
        self.current_salience = 0.0
        self.background_activity = np.zeros(dimensions, dtype=np.float32)
        self.current_query = np.zeros(dimensions, dtype=np.float32)
        self.weights: dict[tuple[int, RecallItem], dict[int, float]] = {}
        self.branches: dict[tuple[int, RecallItem], list[AccessBranch]] = {}
        self.source_weights: dict[
            tuple[int, RecallItem], dict[RecallItem, float]
        ] = {}
        self.confirmations = 0
        self.clock = 0
        self.branch_births = 0
        self.branch_deaths = 0

    def __setstate__(self, state: dict[str, object]) -> None:
        """Migrate checkpoints created before signed access correction."""
        self.__dict__.update(state)
        self.learning_mode = getattr(self, "learning_mode", "hebbian")
        self.background_learning_rate = getattr(
            self, "background_learning_rate", 0.001
        )
        self.background_activity = getattr(
            self,
            "background_activity",
            np.zeros(self.dimensions, dtype=np.float32),
        )
        self.current_query = getattr(
            self,
            "current_query",
            np.zeros(self.dimensions, dtype=np.float32),
        )
        self.consume_on_confirmation = getattr(
            self, "consume_on_confirmation", False
        )
        self.latch_until_confirmation = getattr(
            self, "latch_until_confirmation", False
        )
        self.eligibility_mode = getattr(self, "eligibility_mode", "maximum")
        self.eligibility_bank_size = getattr(self, "eligibility_bank_size", 32)
        self.eligibility_bank = getattr(self, "eligibility_bank", [])
        self.minimum_confirmation_age = getattr(
            self, "minimum_confirmation_age", 0
        )
        self.eligibility_births = getattr(
            self,
            "eligibility_births",
            [getattr(self, "clock", 0)] * len(self.eligibility_bank),
        )
        self.eligibility_saliences = getattr(
            self,
            "eligibility_saliences",
            [0.0] * len(self.eligibility_bank),
        )
        self.current_salience = getattr(self, "current_salience", 0.0)
        self.maximum_branches_per_row = getattr(
            self, "maximum_branches_per_row", 8
        )
        self.maximum_branches = getattr(self, "maximum_branches", 8192)
        self.branch_false_response_decay = getattr(
            self, "branch_false_response_decay", 0.25
        )
        self.branch_survival_threshold = getattr(
            self, "branch_survival_threshold", 0.15
        )
        self.branch_compartments = getattr(self, "branch_compartments", 4)
        self.branch_false_response_grace = getattr(
            self, "branch_false_response_grace", 0
        )
        self.branch_survival_mode = getattr(
            self, "branch_survival_mode", "uniform"
        )
        self.branches = getattr(self, "branches", {})
        for row in self.branches.values():
            for branch in row:
                if not hasattr(branch, "capture_salience"):
                    branch.capture_salience = 0.0
        self.clock = getattr(self, "clock", 0)
        self.branch_births = getattr(self, "branch_births", 0)
        self.branch_deaths = getattr(self, "branch_deaths", 0)

    @property
    def embodied_bytes(self) -> int:
        import pickle

        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def reset_episode(self) -> None:
        self.eligibility.fill(0.0)
        self.eligibility_bank.clear()
        self.eligibility_births.clear()
        self.eligibility_saliences.clear()
        self.current_salience = 0.0
        self.current_query.fill(0.0)

    def advance(self) -> None:
        self.clock += 1
        self.eligibility *= self.trace_decay
        self.eligibility[np.abs(self.eligibility) < 1e-5] = 0.0
        if self.eligibility_mode == "parallel":
            retained = []
            retained_births = []
            retained_saliences = []
            for trace, birth, salience in zip(
                self.eligibility_bank,
                self.eligibility_births,
                self.eligibility_saliences,
                strict=True,
            ):
                trace *= self.trace_decay
                trace[np.abs(trace) < 1e-5] = 0.0
                if np.any(trace):
                    retained.append(trace)
                    retained_births.append(birth)
                    retained_saliences.append(salience)
            self.eligibility_bank = retained
            self.eligibility_births = retained_births
            self.eligibility_saliences = retained_saliences

    def propose(self, cortical_code: np.ndarray, *, salience: float = 0.0) -> None:
        """Leave a local trace of an externally present relational cue."""
        if cortical_code.shape != (self.dimensions,):
            raise ValueError("cortical code has the wrong shape")
        if self.latch_until_confirmation and np.any(self.eligibility):
            return
        maximum = float(np.max(cortical_code, initial=0.0))
        normalized = (
            np.asarray(cortical_code, dtype=np.float32) / maximum
            if maximum > 0.0
            else np.asarray(cortical_code, dtype=np.float32)
        )
        if self.learning_mode in {"prediction_error", "branches"}:
            query = normalized - self.background_activity
            if self.eligibility_mode == "parallel":
                self.eligibility_bank.append(query.copy())
                self.eligibility_births.append(self.clock)
                self.eligibility_saliences.append(max(0.0, float(salience)))
                if len(self.eligibility_bank) > self.eligibility_bank_size:
                    del self.eligibility_bank[0]
                    del self.eligibility_births[0]
                    del self.eligibility_saliences[0]
                self.eligibility = query.copy()
            elif self.eligibility_mode == "additive":
                self.eligibility = np.clip(
                    self.eligibility + query, -1.0, 1.0
                )
            else:
                replace = np.abs(query) > np.abs(self.eligibility)
                self.eligibility[replace] = query[replace]
            self.current_query = query
            self.current_salience = max(0.0, float(salience))
            rate = self.background_learning_rate
            self.background_activity *= 1.0 - rate
            self.background_activity += rate * normalized
        else:
            self.eligibility = np.maximum(self.eligibility, normalized)
            self.current_query = normalized

    @staticmethod
    def _prediction(row: dict[int, float], query: np.ndarray) -> float:
        numerator = sum(
            weight * float(query[dimension])
            for dimension, weight in row.items()
        )
        return float(np.clip(numerator, 0.0, 1.0))

    def _correct_row(
        self,
        row: dict[int, float],
        dimensions: tuple[int, ...],
        target: float,
    ) -> None:
        predicted = self._prediction(row, self.eligibility)
        error = target - predicted
        energy = sum(
            float(self.eligibility[dimension]) ** 2
            for dimension in dimensions
        )
        for dimension in dimensions:
            row[dimension] = float(
                np.clip(
                    row.get(dimension, 0.0)
                    + self.learning_rate
                    * float(self.eligibility[dimension])
                    * error
                    / max(energy, 1e-12),
                    -1.0,
                    1.0,
                )
            )
        if len(row) > self.maximum_synapses:
            retained = {
                dimension
                for dimension, _ in sorted(
                    row.items(), key=lambda item: (-abs(item[1]), item[0])
                )[: self.maximum_synapses]
            }
            for dimension in tuple(row):
                if dimension not in retained:
                    del row[dimension]

    def _new_branch(
        self,
        eligibility: np.ndarray,
        dimensions: tuple[int, ...],
        salience: float,
    ) -> AccessBranch:
        energy = sum(float(eligibility[index]) ** 2 for index in dimensions)
        weights = {
            index: float(eligibility[index]) / max(energy, 1e-12)
            for index in dimensions
        }
        targets = []
        for group in range(self.branch_compartments):
            targets.append(
                sum(
                    weights[index] * float(eligibility[index])
                    for index in dimensions
                    if index % self.branch_compartments == group
                )
            )
        return AccessBranch(
            weights,
            1.0,
            self.clock,
            self.clock,
            group_targets=tuple(targets),
            capture_salience=salience,
        )

    def _branch_response(
        self, branch: AccessBranch, query: np.ndarray
    ) -> float:
        if not branch.group_targets:
            return branch.strength * self._prediction(branch.weights, query)
        responses = []
        for group, target in enumerate(branch.group_targets):
            if target <= 1e-12:
                continue
            current = sum(
                weight * float(query[index])
                for index, weight in branch.weights.items()
                if index % self.branch_compartments == group
            )
            responses.append(float(np.clip(current / target, 0.0, 1.0)))
        if not responses:
            return 0.0
        coincidence = float(np.prod(responses) ** (1.0 / len(responses)))
        return branch.strength * coincidence

    def _enforce_branch_capacity(self) -> None:
        for key in tuple(self.branches):
            row = self.branches[key]
            if len(row) > self.maximum_branches_per_row:
                row.sort(
                    key=lambda branch: (
                        -branch.strength,
                        -branch.last_used,
                        -branch.created_at,
                    )
                )
                removed = len(row) - self.maximum_branches_per_row
                del row[self.maximum_branches_per_row :]
                self.branch_deaths += removed
            if not row:
                del self.branches[key]
        total = sum(len(row) for row in self.branches.values())
        while total > self.maximum_branches:
            key, index, _ = min(
                (
                    (key, index, branch)
                    for key, row in self.branches.items()
                    for index, branch in enumerate(row)
                ),
                key=lambda entry: (
                    entry[2].strength,
                    entry[2].last_used,
                    entry[2].created_at,
                    entry[0],
                    entry[1],
                ),
            )
            del self.branches[key][index]
            if not self.branches[key]:
                del self.branches[key]
            self.branch_deaths += 1
            total -= 1

    def _confirm_branches(
        self,
        eligibilities: tuple[tuple[np.ndarray, float], ...],
        positive_keys: set[tuple[int, RecallItem]],
    ) -> None:
        negative_entries = [
            (key, branch)
            for key, row in self.branches.items()
            if key not in positive_keys
            for branch in row
        ]
        if negative_entries and eligibilities:
            maximum = max(len(branch.weights) for _, branch in negative_entries)
            count = len(negative_entries)
            indices = np.zeros((count, maximum), dtype=np.int32)
            weights = np.zeros((count, maximum), dtype=np.float64)
            synapse_counts = np.zeros(count, dtype=np.int32)
            group_targets = np.zeros(
                (count, self.branch_compartments), dtype=np.float64
            )
            group_counts = np.zeros(count, dtype=np.int32)
            strengths = np.empty(count, dtype=np.float64)
            last_used = np.empty(count, dtype=np.int64)
            capture_saliences = np.empty(count, dtype=np.float64)
            for branch_index, (_, branch) in enumerate(negative_entries):
                items = tuple(branch.weights.items())
                synapse_counts[branch_index] = len(items)
                for slot, (dimension, weight) in enumerate(items):
                    indices[branch_index, slot] = dimension
                    weights[branch_index, slot] = weight
                targets = branch.group_targets[: self.branch_compartments]
                group_counts[branch_index] = len(targets)
                group_targets[branch_index, : len(targets)] = targets
                strengths[branch_index] = branch.strength
                last_used[branch_index] = branch.last_used
                capture_saliences[branch_index] = branch.capture_salience
            trace_matrix = np.ascontiguousarray(
                np.stack([trace for trace, _ in eligibilities]),
                dtype=np.float32,
            )
            saliences = np.asarray(
                [salience for _, salience in eligibilities], dtype=np.float64
            )
            strengths, false_responses, alive = access_branch_survival_compiled(
                trace_matrix,
                saliences,
                indices,
                weights,
                synapse_counts,
                group_targets,
                group_counts,
                strengths,
                last_used,
                capture_saliences,
                self.clock,
                self.branch_false_response_grace,
                self.branch_survival_mode == "surprise",
                self.branch_compartments,
                self.activation_threshold,
                self.branch_false_response_decay,
                self.branch_survival_threshold,
            )
            survival_by_identity = {}
            for branch_index, (_, branch) in enumerate(negative_entries):
                branch.strength = float(strengths[branch_index])
                branch.false_responses += int(false_responses[branch_index])
                survival_by_identity[id(branch)] = bool(alive[branch_index])
            for key in tuple(self.branches):
                if key in positive_keys:
                    continue
                row = self.branches[key]
                survivors = [
                    branch
                    for branch in row
                    if survival_by_identity.get(id(branch), True)
                ]
                self.branch_deaths += len(row) - len(survivors)
                self.branches[key] = survivors

        for eligibility, salience in eligibilities:
            active = np.flatnonzero(np.abs(eligibility) > 0.0)
            if not len(active):
                continue
            dimensions = tuple(
                sorted(
                    (int(index) for index in active),
                    key=lambda index: (
                        -abs(float(eligibility[index])),
                        index,
                    ),
                )[: self.maximum_synapses]
            )
            for key in positive_keys:
                row = self.branches.setdefault(key, [])
                matching = [
                    branch
                    for branch in row
                    if self._branch_response(branch, eligibility) >= 0.95
                ]
                if matching:
                    branch = max(matching, key=lambda value: value.strength)
                    branch.strength = min(1.0, branch.strength + self.learning_rate)
                    branch.last_used = self.clock
                    branch.capture_salience = max(
                        branch.capture_salience, salience
                    )
                else:
                    row.append(
                        self._new_branch(eligibility, dimensions, salience)
                    )
                    self.branch_births += 1
        self._enforce_branch_capacity()

    def confirm(
        self,
        item: RecallItem,
        engrams: tuple[int, ...],
        *,
        external: bool,
        source_items: tuple[RecallItem, ...] = (),
    ) -> int:
        """Bind eligible query dendrites only to externally confirmed content."""
        return self.confirm_many(
            ((item, engrams, source_items),), external=external
        )

    def confirm_many(
        self,
        observations: tuple[
            tuple[RecallItem, tuple[int, ...], tuple[RecallItem, ...]], ...
        ],
        *,
        external: bool,
    ) -> int:
        """Correct access once from a simultaneous external assembly vector."""
        observations = tuple(row for row in observations if row[1])
        if not external or not observations:
            return 0
        if self.eligibility_mode == "parallel":
            eligible_rows = tuple(
                (trace, birth, salience)
                for trace, birth, salience in zip(
                    self.eligibility_bank,
                    self.eligibility_births,
                    self.eligibility_saliences,
                    strict=True,
                )
                if self.clock - birth >= self.minimum_confirmation_age
            )
            eligibilities = tuple(
                (trace, salience) for trace, _, salience in eligible_rows
            )
        else:
            eligible_rows = ()
            eligibilities = ((self.eligibility, self.current_salience),)
        if not any(np.any(trace) for trace, _ in eligibilities):
            return 0
        positive_keys = {
            (int(engram), item)
            for item, engrams, _ in observations
            for engram in engrams
        }
        sources_by_key: dict[tuple[int, RecallItem], set[RecallItem]] = {}
        for item, engrams, source_items in observations:
            for engram in engrams:
                key = (int(engram), item)
                sources_by_key.setdefault(key, set()).update(source_items)
        if self.learning_mode == "branches":
            self._confirm_branches(eligibilities, positive_keys)
            for key in positive_keys:
                sources = self.source_weights.setdefault(key, {})
                for source in sources_by_key.get(key, ()):
                    sources[source] = min(
                        4.0,
                        sources.get(source, 0.0) + self.learning_rate,
                    )
            self.confirmations += len(positive_keys)
            if self.consume_on_confirmation:
                if self.eligibility_mode == "parallel":
                    retained = [
                        (trace, birth, salience)
                        for trace, birth, salience in zip(
                            self.eligibility_bank,
                            self.eligibility_births,
                            self.eligibility_saliences,
                            strict=True,
                        )
                        if self.clock - birth < self.minimum_confirmation_age
                    ]
                    self.eligibility_bank = [trace for trace, _, _ in retained]
                    self.eligibility_births = [birth for _, birth, _ in retained]
                    self.eligibility_saliences = [
                        salience for _, _, salience in retained
                    ]
                    if self.eligibility_bank:
                        self.eligibility = self.eligibility_bank[-1].copy()
                    else:
                        self.eligibility.fill(0.0)
                        self.current_query.fill(0.0)
                        self.current_salience = 0.0
                else:
                    self.eligibility.fill(0.0)
                    self.current_query.fill(0.0)
                    self.current_salience = 0.0
            return len(positive_keys)
        for eligibility, _ in eligibilities:
            self.eligibility = eligibility
            active = np.flatnonzero(np.abs(self.eligibility) > 0.0)
            if not len(active):
                continue
            strongest = sorted(
                (int(index) for index in active),
                key=lambda index: (
                    -abs(float(self.eligibility[index])),
                    index,
                ),
            )[: self.maximum_synapses]
            dimensions = tuple(strongest)
            if self.learning_mode == "prediction_error":
                for key, row in self.weights.items():
                    if key in positive_keys:
                        continue
                    if self._prediction(row, self.eligibility) > 0.0:
                        self._correct_row(row, dimensions, 0.0)
            for key in positive_keys:
                row = self.weights.setdefault(key, {})
                if self.learning_mode == "prediction_error":
                    self._correct_row(row, dimensions, 1.0)
                else:
                    for dimension in strongest:
                        row[dimension] = min(
                            1.0,
                            row.get(dimension, 0.0)
                            + self.learning_rate
                            * float(self.eligibility[dimension]),
                        )
                sources = self.source_weights.setdefault(key, {})
                for source in sources_by_key.get(key, ()):
                    sources[source] = min(
                        4.0,
                        sources.get(source, 0.0) + self.learning_rate,
                    )
        self.confirmations += len(positive_keys)
        if self.consume_on_confirmation:
            self.eligibility.fill(0.0)
            self.eligibility_bank.clear()
            self.eligibility_births.clear()
            self.eligibility_saliences.clear()
            self.current_query.fill(0.0)
            self.current_salience = 0.0
        return len(positive_keys)

    def activate(
        self, cortical_code: np.ndarray, engram_count: int
    ) -> RelationalAccessSignal:
        """Let all learned access dendrites integrate the current cue in parallel."""
        if cortical_code.shape != (self.dimensions,):
            raise ValueError("cortical code has the wrong shape")
        maximum = float(np.max(cortical_code, initial=0.0))
        normalized = (
            np.asarray(cortical_code, dtype=np.float32) / maximum
            if maximum > 0.0
            else np.asarray(cortical_code, dtype=np.float32)
        )
        query = (
            normalized - self.background_activity
            if self.learning_mode in {"prediction_error", "branches"}
            else normalized
        )
        support = np.zeros(engram_count, dtype=np.float64)
        content: dict[int, dict[RecallItem, float]] = {}
        sources: dict[int, dict[RecallItem, float]] = {}
        for (engram, item), row in self.weights.items():
            if engram >= engram_count or not row:
                continue
            if self.learning_mode == "prediction_error":
                overlap = self._prediction(row, query)
            else:
                numerator = sum(
                    weight * float(query[dimension])
                    for dimension, weight in row.items()
                )
                denominator = sum(row.values())
                overlap = numerator / max(denominator, 1e-12)
            if overlap < self.activation_threshold:
                continue
            support[engram] = max(support[engram], overlap)
            row_content = content.setdefault(engram, {})
            row_content[item] = max(row_content.get(item, 0.0), overlap)
            source_row = sources.setdefault(engram, {})
            for source, weight in self.source_weights.get((engram, item), {}).items():
                source_row[source] = max(
                    source_row.get(source, 0.0), overlap * weight
                )
        if self.learning_mode == "branches":
            for (engram, item), row in self.branches.items():
                if engram >= engram_count or not row:
                    continue
                overlap = max(
                    self._branch_response(branch, query) for branch in row
                )
                if overlap < self.activation_threshold:
                    continue
                support[engram] = max(support[engram], overlap)
                row_content = content.setdefault(engram, {})
                row_content[item] = max(row_content.get(item, 0.0), overlap)
                source_row = sources.setdefault(engram, {})
                for source, weight in self.source_weights.get(
                    (engram, item), {}
                ).items():
                    source_row[source] = max(
                        source_row.get(source, 0.0), overlap * weight
                    )
        return RelationalAccessSignal(support, content, sources)
