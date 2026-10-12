"""Multi-timescale local eligibility for cross-organ relationships."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np

Relation = Hashable


class TemporalRelationalEligibility:
    """Keep fading pathway participation without keeping content active."""

    def __init__(
        self,
        *,
        half_lives: tuple[float, ...] = (8.0, 32.0, 128.0, 512.0),
        learning_rate: float = 0.25,
        conflict_decay: float = 0.5,
        maximum_strength: float = 8.0,
    ) -> None:
        if not half_lives or any(value <= 0.0 for value in half_lives):
            raise ValueError("eligibility half-lives must be positive")
        if learning_rate <= 0.0 or maximum_strength <= 0.0:
            raise ValueError("learning rate and strength bound must be positive")
        if not 0.0 <= conflict_decay <= 1.0:
            raise ValueError("conflict decay must be in [0, 1]")
        self.half_lives = np.asarray(half_lives, dtype=np.float64)
        self.decay = 2.0 ** (-1.0 / self.half_lives)
        self.learning_rate = float(learning_rate)
        self.conflict_decay = float(conflict_decay)
        self.maximum_strength = float(maximum_strength)
        self.traces: dict[Relation, np.ndarray] = {}
        self.strengths: dict[Relation, np.ndarray] = {}
        self.last_participation: dict[Relation, int] = {}
        self.confirmation_ages: list[int] = []
        self.clock = 0
        self.participations = 0
        self.external_updates = 0
        self.positive_updates = 0
        self.negative_updates = 0
        self.positive_utility = 0.0
        self.negative_utility = 0.0

    def participate(
        self,
        relation: Relation,
        amount: float,
        *,
        incompatible: tuple[Relation, ...] = (),
    ) -> None:
        """Leave a local trace and weaken mutually exclusive old traces."""
        if amount <= 0.0:
            return
        trace = self.traces.setdefault(
            relation, np.zeros(len(self.half_lives), dtype=np.float64)
        )
        trace += min(1.0, float(amount))
        np.minimum(trace, 1.0, out=trace)
        self.last_participation[relation] = self.clock
        self.participations += 1
        for other in incompatible:
            if other == relation or other not in self.traces:
                continue
            self.traces[other] *= self.conflict_decay

    def participate_population(
        self,
        population: Mapping[Relation, float],
        *,
        incompatible: Mapping[Relation, tuple[Relation, ...]] | None = None,
    ) -> None:
        """Lay down one simultaneous population without order artifacts."""
        active = {relation for relation, amount in population.items() if amount > 0.0}
        if incompatible is not None:
            suppressed = {
                other
                for relation in active
                for other in incompatible.get(relation, ())
                if other not in active
            }
            for other in suppressed:
                if other in self.traces:
                    self.traces[other] *= self.conflict_decay
        for relation, amount in population.items():
            self.participate(relation, amount)

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if not steps:
            return
        self.clock += steps
        decay = self.decay**steps
        for relation in tuple(self.traces):
            self.traces[relation] *= decay
            if float(np.max(self.traces[relation], initial=0.0)) < 1e-8:
                del self.traces[relation]

    def confirm(
        self,
        utility: Mapping[Relation, float],
        *,
        external: bool,
    ) -> None:
        """Apply later utility only when a real external consequence arrived."""
        if not external:
            return
        updated = False
        for relation, value in utility.items():
            trace = self.traces.get(relation)
            if trace is None or value == 0.0:
                continue
            strengths = self.strengths.setdefault(
                relation, np.zeros(len(self.half_lives), dtype=np.float64)
            )
            strengths += self.learning_rate * trace * float(value)
            np.clip(
                strengths,
                -self.maximum_strength,
                self.maximum_strength,
                out=strengths,
            )
            updated = True
            if value > 0.0:
                self.positive_updates += 1
                self.positive_utility += float(value)
            else:
                self.negative_updates += 1
                self.negative_utility += abs(float(value))
            if relation in self.last_participation:
                self.confirmation_ages.append(
                    self.clock - self.last_participation[relation]
                )
        self.external_updates += int(updated)

    def reset_episode(self) -> None:
        """Clear biochemical traces while retaining learned pathway strength."""
        self.traces.clear()
        self.last_participation.clear()

    @property
    def embodied_bytes(self) -> int:
        import pickle

        return len(
            pickle.dumps(
                (self.half_lives, self.strengths),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    def trace(self, relation: Relation) -> np.ndarray:
        return self.traces.get(
            relation, np.zeros(len(self.half_lives), dtype=np.float64)
        ).copy()

    def strength(self, relation: Relation) -> np.ndarray:
        return self.strengths.get(
            relation, np.zeros(len(self.half_lives), dtype=np.float64)
        ).copy()

    def permission(self, relation: Relation) -> float:
        """Conservative learned value; negative timescales cannot vote live."""
        values = np.maximum(0.0, self.strength(relation))
        return float(1.0 - np.exp(-np.sum(values)))
