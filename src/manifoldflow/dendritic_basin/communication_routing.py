"""Selective communication through addressed or locally learned routes."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RoutingMode(str, Enum):
    GLOBAL = "global"
    ADDRESSED = "addressed"
    LEARNED = "learned"


@dataclass(frozen=True)
class CommunicationStep:
    observed_probability: float
    predicted_outcome: int
    update_count: int
    selected_source: int | None


def _context_key(context_receptors: np.ndarray) -> bytes:
    vector = np.asarray(context_receptors, dtype=np.uint8).reshape(-1)
    if np.any(vector > 1):
        raise ValueError("context receptors must be binary")
    return len(vector).to_bytes(2, "little") + np.packbits(vector).tobytes()


class SelectiveCommunicationLearner:
    """Ablate global pooling, fixed addressing, and learned route competition.

    Learned routing keeps one local agreement statistic on every candidate
    context-to-source branch.  Branches compete within the active context; no
    source identity from the environment is passed to that condition.
    """

    def __init__(
        self,
        source_count: int,
        *,
        routing_mode: RoutingMode | str,
        prior: float = 0.5,
    ) -> None:
        if source_count < 2:
            raise ValueError("source_count must be at least two")
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.source_count = int(source_count)
        self.routing_mode = RoutingMode(routing_mode)
        self.prior = float(prior)
        self.outcome_counts = np.zeros((source_count, 2, 2), dtype=np.float64)
        self.route_matches: dict[bytes, np.ndarray] = {}
        self.route_totals: dict[bytes, int] = {}
        self.steps = 0
        self.local_updates = 0

    @property
    def embodied_bytes(self) -> int:
        body = (
            self.source_count,
            self.routing_mode.value,
            self.prior,
            self.outcome_counts,
            self.route_matches,
            self.route_totals,
        )
        return len(pickle.dumps(body, protocol=5))

    def _global_probability(self, source_states: np.ndarray) -> float:
        counts = np.full(2, self.prior, dtype=np.float64)
        for source, state in enumerate(source_states):
            counts += self.outcome_counts[source, int(state)]
        return float(counts[1] / counts.sum())

    def _addressed_probability(self, source_states: np.ndarray, source: int) -> float:
        counts = self.outcome_counts[source, int(source_states[source])] + self.prior
        return float(counts[1] / counts.sum())

    def _learned_probability(
        self, source_states: np.ndarray, context: bytes
    ) -> tuple[float, int | None]:
        matches = self.route_matches.get(context)
        total = self.route_totals.get(context, 0)
        if matches is None or total == 0:
            return 0.5, None
        reliability = (matches + self.prior) / (total + 2.0 * self.prior)
        selected = int(np.argmax(reliability))
        correct_probability = float(reliability[selected])
        if int(source_states[selected]) == 1:
            return correct_probability, selected
        return 1.0 - correct_probability, selected

    def probability(
        self,
        source_states: np.ndarray,
        context_receptors: np.ndarray,
        *,
        addressed_source: int | None = None,
    ) -> tuple[float, int | None]:
        states = np.asarray(source_states, dtype=np.uint8).reshape(-1)
        if states.shape != (self.source_count,) or np.any(states > 1):
            raise ValueError("source_states must contain one bit per source")
        context = _context_key(context_receptors)
        if self.routing_mode is RoutingMode.GLOBAL:
            return self._global_probability(states), None
        if self.routing_mode is RoutingMode.ADDRESSED:
            if addressed_source is None:
                raise ValueError("addressed mode requires a local destination")
            source = int(addressed_source)
            if not 0 <= source < self.source_count:
                raise ValueError("addressed_source is outside the source population")
            return self._addressed_probability(states, source), source
        if addressed_source is not None:
            raise ValueError("learned routes must not receive the hidden destination")
        return self._learned_probability(states, context)

    def observe(
        self,
        source_states: np.ndarray,
        context_receptors: np.ndarray,
        outcome: int,
        *,
        addressed_source: int | None = None,
        learn: bool = True,
    ) -> CommunicationStep:
        states = np.asarray(source_states, dtype=np.uint8).reshape(-1)
        context = _context_key(context_receptors)
        probability_one, selected = self.probability(
            states,
            context_receptors,
            addressed_source=addressed_source,
        )
        outcome = int(outcome)
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        observed = probability_one if outcome else 1.0 - probability_one
        update_count = 0

        if learn:
            if self.routing_mode is RoutingMode.GLOBAL:
                for source, state in enumerate(states):
                    self.outcome_counts[source, int(state), outcome] += 1.0
                update_count = self.source_count
            elif self.routing_mode is RoutingMode.ADDRESSED:
                if selected is None:
                    raise RuntimeError("addressed route disappeared during update")
                self.outcome_counts[selected, int(states[selected]), outcome] += 1.0
                update_count = 1
            else:
                matches = self.route_matches.setdefault(
                    context, np.zeros(self.source_count, dtype=np.float64)
                )
                matches += states == outcome
                self.route_totals[context] = self.route_totals.get(context, 0) + 1
                update_count = self.source_count
            self.steps += 1
            self.local_updates += update_count

        return CommunicationStep(
            observed_probability=observed,
            predicted_outcome=int(probability_one >= 0.5),
            update_count=update_count,
            selected_source=selected,
        )

    def learned_route(self, context_receptors: np.ndarray) -> int | None:
        context = _context_key(context_receptors)
        matches = self.route_matches.get(context)
        if matches is None:
            return None
        return int(np.argmax(matches))
