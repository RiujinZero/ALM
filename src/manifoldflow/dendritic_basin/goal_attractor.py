"""Sparse goals learned from restoration of a regulated condition.

The organ receives no desired target or action. Recent event assemblies leave
local eligibility, and a decrease in regulatory deviation reinforces only
those eligible assemblies. Once learned, a restorative predictor can provide a
weaker conditioned signal to earlier eligible events. Competition selects one
such predictor as persistent apical context for specialist organs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoalAttractorStep:
    goal: int | None
    changed: bool
    modulation: float
    local_updates: int


class HomeostaticGoalAttractor:
    """Winner-sized goal state driven by locally eligible restoration value."""

    def __init__(
        self,
        assembly_count: int,
        *,
        eligibility_decay: float = 0.9,
        conditioned_gain: float = 0.5,
        value_prior: float = 1.0,
        minimum_goal_value: float = 0.05,
        failure_patience: int = 4,
        restoration_fraction: float = 0.1,
    ) -> None:
        if assembly_count < 2:
            raise ValueError("assembly_count must be at least two")
        if not 0.0 < eligibility_decay < 1.0:
            raise ValueError("eligibility_decay must be in (0, 1)")
        if not 0.0 <= conditioned_gain < 1.0:
            raise ValueError("conditioned_gain must be in [0, 1)")
        if value_prior <= 0.0:
            raise ValueError("value_prior must be positive")
        if failure_patience < 1:
            raise ValueError("failure_patience must be positive")
        if not 0.0 < restoration_fraction <= 1.0:
            raise ValueError("restoration fraction must be in (0, 1]")
        self.assembly_count = int(assembly_count)
        self.eligibility_decay = float(eligibility_decay)
        self.conditioned_gain = float(conditioned_gain)
        self.value_prior = float(value_prior)
        self.minimum_goal_value = float(minimum_goal_value)
        self.failure_patience = int(failure_patience)
        self.restoration_fraction = float(restoration_fraction)

        # One signed restorative statistic and evidence total per assembly.
        self.restoration_sum = np.zeros(assembly_count, dtype=np.float64)
        self.restoration_evidence = np.zeros(assembly_count, dtype=np.float64)

        # Bounded working state, not durable anatomy.
        self.eligibility = np.zeros(assembly_count, dtype=np.float64)
        self.current_goal: int | None = None
        self.previous_deviation: float | None = None
        self.goal_reference_deviation: float | None = None
        self.failure_count = 0

        self.steps = 0
        self.local_updates = 0
        self.goal_births = 0
        self.goal_dissolutions = 0

    @property
    def embodied_bytes(self) -> int:
        body = (
            self.assembly_count,
            self.eligibility_decay,
            self.conditioned_gain,
            self.value_prior,
            self.minimum_goal_value,
            self.failure_patience,
            self.restoration_fraction,
            self.restoration_sum,
            self.restoration_evidence,
        )
        return len(pickle.dumps(body, protocol=5))

    @property
    def transient_bytes(self) -> int:
        state = (
            self.eligibility,
            self.current_goal,
            self.previous_deviation,
            self.goal_reference_deviation,
            self.failure_count,
        )
        return len(pickle.dumps(state, protocol=5))

    def value(self, assembly: int) -> float:
        assembly = self._validate_assembly(assembly)
        return float(
            self.restoration_sum[assembly]
            / (self.restoration_evidence[assembly] + self.value_prior)
        )

    def values(self) -> np.ndarray:
        return self.restoration_sum / (self.restoration_evidence + self.value_prior)

    def _validate_assembly(self, assembly: int) -> int:
        assembly = int(assembly)
        if not 0 <= assembly < self.assembly_count:
            raise ValueError("assembly is outside the goal population")
        return assembly

    def _apply_modulation(self, modulation: float) -> int:
        modulation = float(np.clip(modulation, -1.0, 1.0))
        active = self.eligibility > 1e-9
        strengths = self.eligibility[active]
        if not np.any(active) or modulation == 0.0:
            return 0
        self.restoration_sum[active] += strengths * modulation
        self.restoration_evidence[active] += strengths * abs(modulation)
        updates = int(np.sum(active))
        self.local_updates += updates
        return updates

    def observe_event(self, assembly: int, *, learn: bool = True) -> GoalAttractorStep:
        """Activate an event and let its learned value teach its predecessors."""
        assembly = self._validate_assembly(assembly)
        self.eligibility *= self.eligibility_decay
        modulation = self.conditioned_gain * max(0.0, self.value(assembly))
        updates = self._apply_modulation(modulation) if learn else 0
        self.eligibility[assembly] = 1.0
        self.steps += 1
        return GoalAttractorStep(self.current_goal, False, modulation, updates)

    def observe_deviation(
        self, deviation: float, *, learn: bool = True
    ) -> GoalAttractorStep:
        """Convert restoration of a declared regulated variable into modulation."""
        deviation = float(deviation)
        if not np.isfinite(deviation) or deviation < 0.0:
            raise ValueError("deviation must be finite and non-negative")
        modulation = 0.0
        if self.previous_deviation is not None:
            scale = max(self.previous_deviation, deviation, 1e-12)
            modulation = (self.previous_deviation - deviation) / scale
        self.previous_deviation = deviation
        updates = self._apply_modulation(modulation) if learn else 0

        changed = False
        if self.current_goal is not None:
            restored = (
                self.goal_reference_deviation is not None
                and deviation
                <= self.goal_reference_deviation * (1.0 - self.restoration_fraction)
            )
            if modulation > 0.0 and restored:
                self.current_goal = None
                self.goal_reference_deviation = None
                self.failure_count = 0
                self.goal_dissolutions += 1
                changed = True
            elif modulation < 0.0:
                self.failure_count += 1
                if self.failure_count >= self.failure_patience:
                    self.current_goal = None
                    self.goal_reference_deviation = None
                    self.failure_count = 0
                    self.goal_dissolutions += 1
                    changed = True
            elif modulation >= 0.0:
                self.failure_count = max(0, self.failure_count - 1)
        return GoalAttractorStep(self.current_goal, changed, modulation, updates)

    def select_goal(
        self,
        candidates: np.ndarray | list[int] | tuple[int, ...],
        *,
        accessibility: np.ndarray | None = None,
    ) -> GoalAttractorStep:
        """Select by learned restoration and accessibility, never target labels."""
        candidate_array = np.asarray(candidates, dtype=np.int64).reshape(-1)
        if len(candidate_array) == 0:
            return GoalAttractorStep(self.current_goal, False, 0.0, 0)
        outside = (candidate_array < 0) | (candidate_array >= self.assembly_count)
        if np.any(outside):
            raise ValueError("candidate is outside the goal population")
        if accessibility is None:
            access = np.ones(len(candidate_array), dtype=np.float64)
        else:
            access = np.asarray(accessibility, dtype=np.float64).reshape(-1)
            if access.shape != candidate_array.shape or np.any(access < 0.0):
                raise ValueError("accessibility must be non-negative per candidate")
        scores = self.values()[candidate_array] * access
        winner_index = int(np.argmax(scores))
        winner = int(candidate_array[winner_index])
        if scores[winner_index] < self.minimum_goal_value:
            winner = None
        changed = winner != self.current_goal
        if changed and winner is not None:
            self.goal_births += 1
        self.current_goal = winner
        self.goal_reference_deviation = (
            self.previous_deviation if winner is not None else None
        )
        self.failure_count = 0
        return GoalAttractorStep(self.current_goal, changed, 0.0, 0)

    def reset_episode(self) -> None:
        self.eligibility.fill(0.0)
        self.current_goal = None
        self.previous_deviation = None
        self.goal_reference_deviation = None
        self.failure_count = 0


class GoalConditionedLocalController:
    """Specialist action counts gated by a sparse goal identity."""

    def __init__(self, module_count: int, goal_count: int, *, prior: float = 0.5):
        if module_count < 1 or goal_count < 2 or prior <= 0.0:
            raise ValueError("invalid controller dimensions or prior")
        self.module_count = int(module_count)
        self.goal_count = int(goal_count)
        self.prior = float(prior)
        self.outcomes = np.zeros((module_count, goal_count, 2, 2), dtype=np.float64)

    @property
    def embodied_bytes(self) -> int:
        return len(pickle.dumps((self.prior, self.outcomes), protocol=5))

    def action(self, module: int, goal: int) -> int:
        rows = self.outcomes[int(module), int(goal)] + self.prior
        success = rows[:, 1] / rows.sum(axis=1)
        return int(np.argmax(success))

    def observe(self, module: int, goal: int, action: int, improved: bool) -> None:
        self.outcomes[int(module), int(goal), int(action), int(bool(improved))] += 1.0


class SparseHomeostaticGoalAttractor:
    """Dictionary-backed version for dynamically born cortical assemblies."""

    def __init__(
        self,
        *,
        eligibility_decay: float = 0.9,
        conditioned_gain: float = 0.5,
        value_prior: float = 1.0,
        minimum_goal_value: float = 0.05,
        max_eligibility: int = 64,
        max_values: int = 256,
        failure_patience: int = 4,
        restoration_fraction: float = 0.1,
        component_count: int = 0,
        minimum_mode_pressure: float = 0.05,
    ) -> None:
        if not 0.0 < eligibility_decay < 1.0:
            raise ValueError("eligibility_decay must be in (0, 1)")
        if not 0.0 <= conditioned_gain < 1.0:
            raise ValueError("conditioned_gain must be in [0, 1)")
        if (
            value_prior <= 0.0
            or max_eligibility < 1
            or max_values < 2
            or failure_patience < 1
            or component_count < 0
            or minimum_mode_pressure < 0.0
        ):
            raise ValueError("invalid sparse goal capacity")
        if not 0.0 < restoration_fraction <= 1.0:
            raise ValueError("restoration fraction must be in (0, 1]")
        self.eligibility_decay = float(eligibility_decay)
        self.conditioned_gain = float(conditioned_gain)
        self.value_prior = float(value_prior)
        self.minimum_goal_value = float(minimum_goal_value)
        self.max_eligibility = int(max_eligibility)
        self.max_values = int(max_values)
        self.failure_patience = int(failure_patience)
        self.restoration_fraction = float(restoration_fraction)
        self.component_count = int(component_count)
        self.minimum_mode_pressure = float(minimum_mode_pressure)
        self.restoration_sum: dict[int, float] = {}
        self.restoration_evidence: dict[int, float] = {}
        self.component_restoration_sum: dict[int, np.ndarray] = {}
        self.component_restoration_evidence: dict[int, np.ndarray] = {}
        self.eligibility: dict[int, float] = {}
        self.current_goal: int | None = None
        self.previous_deviation: float | None = None
        self.goal_reference_deviation: float | None = None
        self.previous_components: np.ndarray | None = None
        self.current_mode: int | None = None
        self.mode_reference_pressure: float | None = None
        self.failure_count = 0
        self.local_updates = 0
        self.goal_births = 0
        self.goal_dissolutions = 0
        self.mode_births = 0
        self.mode_dissolutions = 0
        self._weakest_key: int | None = None

    @property
    def embodied_bytes(self) -> int:
        body = (
            self.eligibility_decay,
            self.conditioned_gain,
            self.value_prior,
            self.minimum_goal_value,
            self.max_eligibility,
            self.max_values,
            self.failure_patience,
            self.restoration_fraction,
            getattr(self, "component_count", 0),
            getattr(self, "minimum_mode_pressure", 0.05),
            self.restoration_sum,
            self.restoration_evidence,
            getattr(self, "component_restoration_sum", {}),
            getattr(self, "component_restoration_evidence", {}),
        )
        return len(pickle.dumps(body, protocol=5))

    @property
    def transient_bytes(self) -> int:
        state = (
            self.eligibility,
            self.current_goal,
            self.previous_deviation,
            self.goal_reference_deviation,
            getattr(self, "previous_components", None),
            getattr(self, "current_mode", None),
            getattr(self, "mode_reference_pressure", None),
            self.failure_count,
            self._weakest_key,
        )
        return len(pickle.dumps(state, protocol=5))

    def value(self, assembly: int) -> float:
        assembly = int(assembly)
        return self.restoration_sum.get(assembly, 0.0) / (
            self.restoration_evidence.get(assembly, 0.0) + self.value_prior
        )

    def restoration_signature(self, assembly: int) -> np.ndarray:
        count = getattr(self, "component_count", 0)
        if count == 0:
            return np.empty(0, dtype=np.float64)
        numerator = getattr(self, "component_restoration_sum", {}).get(assembly)
        evidence = getattr(self, "component_restoration_evidence", {}).get(assembly)
        if numerator is None or evidence is None:
            return np.zeros(count, dtype=np.float64)
        return numerator / (evidence + self.value_prior)

    def _admit_value(self, assembly: int, proposed_strength: float) -> bool:
        if assembly in self.restoration_sum:
            return True
        if len(self.restoration_sum) >= self.max_values:
            weakest = self._weakest_key
            if weakest not in self.restoration_sum:
                weakest = min(
                    self.restoration_sum,
                    key=lambda key: abs(self.restoration_sum[key]),
                )
                self._weakest_key = weakest
            if abs(self.restoration_sum[weakest]) >= proposed_strength:
                return False
            self.restoration_sum.pop(weakest)
            self.restoration_evidence.pop(weakest)
            self.component_restoration_sum.pop(weakest, None)
            self.component_restoration_evidence.pop(weakest, None)
        self.restoration_sum[assembly] = 0.0
        self.restoration_evidence[assembly] = 0.0
        self._weakest_key = None
        return True

    def _modulate(self, signal: float) -> int:
        signal = float(np.clip(signal, -1.0, 1.0))
        if signal == 0.0:
            return 0
        updates = 0
        for assembly, eligibility in self.eligibility.items():
            increment = eligibility * signal
            if not self._admit_value(assembly, abs(increment)):
                continue
            old_strength = abs(self.restoration_sum[assembly])
            self.restoration_sum[assembly] += increment
            self.restoration_evidence[assembly] += eligibility * abs(signal)
            new_strength = abs(self.restoration_sum[assembly])
            if assembly == self._weakest_key and new_strength > old_strength:
                self._weakest_key = None
            elif self._weakest_key is not None:
                weakest_strength = abs(self.restoration_sum[self._weakest_key])
                if new_strength < weakest_strength:
                    self._weakest_key = assembly
            updates += 1
        self.local_updates += updates
        return updates

    def observe_components(
        self, components: tuple[float, ...] | np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        """Learn which eligible events restore each regulated variable."""
        values = np.asarray(components, dtype=np.float64).reshape(-1)
        count = getattr(self, "component_count", 0)
        if (
            values.shape != (count,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("components must match the non-negative regulatory state")
        previous = getattr(self, "previous_components", None)
        signals = np.zeros(count, dtype=np.float64)
        if previous is not None:
            scale = np.maximum(np.maximum(previous, values), 1e-12)
            signals = np.clip((previous - values) / scale, -1.0, 1.0)
        self.previous_components = values.copy()
        if learn and np.any(signals):
            for assembly, eligibility in self.eligibility.items():
                proposed = float(eligibility * np.max(np.abs(signals)))
                if not self._admit_value(assembly, proposed):
                    continue
                sums = self.component_restoration_sum.setdefault(
                    assembly, np.zeros(count, dtype=np.float64)
                )
                evidence = self.component_restoration_evidence.setdefault(
                    assembly, np.zeros(count, dtype=np.float64)
                )
                sums += eligibility * signals
                evidence += eligibility * np.abs(signals)
        mode = getattr(self, "current_mode", None)
        reference = getattr(self, "mode_reference_pressure", None)
        if mode is not None and reference is not None:
            if values[mode] <= reference * (1.0 - self.restoration_fraction):
                self.current_mode = None
                self.mode_reference_pressure = None
                self.mode_dissolutions += 1
        elif len(values):
            winner = int(np.argmax(values))
            pressure = float(values[winner])
            if pressure >= getattr(self, "minimum_mode_pressure", 0.05):
                self.current_mode = winner
                self.mode_reference_pressure = pressure
                self.mode_births += 1
        return signals

    def observe_event(self, assembly: int, *, learn: bool = True) -> float:
        self.eligibility = {
            key: value * self.eligibility_decay
            for key, value in self.eligibility.items()
            if value * self.eligibility_decay > 1e-6
        }
        signal = self.conditioned_gain * max(0.0, self.value(assembly))
        if learn:
            self._modulate(signal)
        at_capacity = len(self.eligibility) >= self.max_eligibility
        if assembly not in self.eligibility and at_capacity:
            del self.eligibility[min(self.eligibility, key=self.eligibility.get)]
        self.eligibility[int(assembly)] = 1.0
        return signal

    def observe_deviation(
        self,
        deviation: float,
        *,
        learn: bool = True,
        dissolve_on_restoration: bool = True,
    ) -> float:
        deviation = float(deviation)
        if not np.isfinite(deviation) or deviation < 0.0:
            raise ValueError("deviation must be finite and non-negative")
        signal = 0.0
        if self.previous_deviation is not None:
            scale = max(self.previous_deviation, deviation, 1e-12)
            signal = (self.previous_deviation - deviation) / scale
        self.previous_deviation = deviation
        if learn:
            self._modulate(signal)
        if self.current_goal is not None:
            restored = (
                self.goal_reference_deviation is not None
                and deviation
                <= self.goal_reference_deviation * (1.0 - self.restoration_fraction)
            )
            if signal > 0.0 and restored and dissolve_on_restoration:
                self.current_goal = None
                self.goal_reference_deviation = None
                self.failure_count = 0
                self.goal_dissolutions += 1
            elif signal < 0.0:
                self.failure_count += 1
                if self.failure_count >= self.failure_patience:
                    self.current_goal = None
                    self.goal_reference_deviation = None
                    self.failure_count = 0
                    self.goal_dissolutions += 1
            elif signal > 0.0:
                self.failure_count = max(0, self.failure_count - 1)
        return signal

    def select_goal(self, candidates: dict[int, float]) -> int | None:
        if self.current_goal is not None:
            return self.current_goal
        scored = [
            (self.value(candidate) * max(0.0, accessibility), candidate)
            for candidate, accessibility in candidates.items()
        ]
        if not scored:
            return None
        score, winner = max(scored)
        if score < self.minimum_goal_value:
            return None
        self.current_goal = int(winner)
        self.goal_reference_deviation = self.previous_deviation
        self.failure_count = 0
        self.goal_births += 1
        return self.current_goal

    def achieve(self, assembly: int) -> bool:
        if self.current_goal != int(assembly):
            return False
        self.current_goal = None
        self.goal_reference_deviation = None
        self.failure_count = 0
        self.goal_dissolutions += 1
        return True

    def reset_episode(self) -> None:
        self.eligibility.clear()
        self.current_goal = None
        self.previous_deviation = None
        self.goal_reference_deviation = None
        self.previous_components = None
        self.current_mode = None
        self.mode_reference_pressure = None
        self.failure_count = 0


class SparsePredictiveBasinAttractor:
    """Competitive fan-in attractor for reusable local future signatures."""

    def __init__(
        self,
        component_count: int = 8,
        *,
        max_basins: int = 16,
        birth_similarity: float = 0.75,
        hysteresis: float = 0.05,
    ) -> None:
        if component_count < 2 or max_basins < 2:
            raise ValueError("predictive basin dimensions and capacity must exceed one")
        if not -1.0 <= birth_similarity <= 1.0:
            raise ValueError("birth similarity must be in [-1, 1]")
        if not 0.0 <= hysteresis <= 2.0:
            raise ValueError("hysteresis must be in [0, 2]")
        self.component_count = int(component_count)
        self.max_basins = int(max_basins)
        self.birth_similarity = float(birth_similarity)
        self.hysteresis = float(hysteresis)
        self.prototype_sum = np.zeros(
            (self.max_basins, self.component_count), dtype=np.float64
        )
        self.evidence = np.zeros(self.max_basins, dtype=np.uint64)
        self.current_basin: int | None = None
        self.basin_births = 0
        self.basin_switches = 0

    @property
    def live_basins(self) -> int:
        return int(np.count_nonzero(self.evidence))

    @property
    def embodied_bytes(self) -> int:
        body = (
            self.component_count,
            self.max_basins,
            self.birth_similarity,
            self.hysteresis,
            self.prototype_sum,
            self.evidence,
        )
        return len(pickle.dumps(body, protocol=5))

    def prototype(self, basin: int) -> np.ndarray:
        vector = self.prototype_sum[int(basin)]
        norm = float(np.linalg.norm(vector))
        return vector.copy() if norm == 0.0 else vector / norm

    def similarities(self, signature: np.ndarray) -> np.ndarray:
        vector = np.asarray(signature, dtype=np.float64).reshape(-1)
        if vector.shape != (self.component_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("signature must match the finite predictive basin state")
        norm = float(np.linalg.norm(vector))
        scores = np.full(self.max_basins, -np.inf, dtype=np.float64)
        if norm == 0.0:
            return scores
        live = np.flatnonzero(self.evidence)
        for basin in live:
            scores[basin] = float(np.dot(self.prototype(int(basin)), vector) / norm)
        return scores

    def observe(self, signature: np.ndarray, *, learn: bool = True) -> int | None:
        vector = np.asarray(signature, dtype=np.float64).reshape(-1)
        scores = self.similarities(vector)
        if float(np.linalg.norm(vector)) == 0.0:
            return self.current_basin
        live = self.live_basins
        winner = None if live == 0 else int(np.argmax(scores))
        born = False
        if (
            learn
            and live < self.max_basins
            and (winner is None or scores[winner] < self.birth_similarity)
        ):
            winner = int(np.flatnonzero(self.evidence == 0)[0])
            self.basin_births += 1
            born = True
        if winner is None:
            return self.current_basin
        if (
            not born
            and self.current_basin is not None
            and self.evidence[self.current_basin] > 0
        ):
            current_score = scores[self.current_basin]
            if current_score + self.hysteresis >= scores[winner]:
                winner = self.current_basin
        if winner != self.current_basin and self.current_basin is not None:
            self.basin_switches += 1
        self.current_basin = winner
        if learn:
            self.prototype_sum[winner] += vector
            self.evidence[winner] += 1
        return winner

    def alignment(self, signature: np.ndarray) -> float:
        if self.current_basin is None:
            return 0.0
        vector = np.asarray(signature, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return 0.0
        return float(np.dot(self.prototype(self.current_basin), vector) / norm)

    def consolidate(self, other: SparsePredictiveBasinAttractor) -> dict[int, int]:
        configuration = (
            "component_count",
            "max_basins",
            "birth_similarity",
            "hysteresis",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate predictive basins with different settings"
            )
        incoming_to_local: dict[int, int] = {}
        for incoming in np.flatnonzero(other.evidence):
            signature = other.prototype(int(incoming))
            scores = self.similarities(signature)
            live = self.live_basins
            target = None if live == 0 else int(np.argmax(scores))
            if live < self.max_basins and (
                target is None or scores[target] < self.birth_similarity
            ):
                target = int(np.flatnonzero(self.evidence == 0)[0])
            if target is None:
                continue
            incoming_to_local[int(incoming)] = target
            self.prototype_sum[target] += other.prototype_sum[incoming]
            self.evidence[target] += other.evidence[incoming]
        self.basin_births = self.live_basins
        self.basin_switches += other.basin_switches
        self.current_basin = None
        return incoming_to_local

    def reset_episode(self) -> None:
        self.current_basin = None
