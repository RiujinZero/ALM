"""Local delayed credit through basal eligibility and apical addressing.

This module is deliberately smaller than the byte cortex.  It isolates one
question: can a long-range signal select a local update without transporting a
gradient or a desired weight?  Every receiver owns two opponent basal
receptors and a local outcome table.  A delayed consequence either reaches no
receiver, is broadcast to every eligible receiver, or is admitted only where
an apical context receptor is active.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TeachingMode(str, Enum):
    """How a delayed consequence is made available to local synapses."""

    NONE = "none"
    BROADCAST = "broadcast"
    APICAL = "apical"


@dataclass(frozen=True)
class DelayedCreditStep:
    """One prediction and its subsequent local-update audit."""

    observed_probability: float
    predicted_outcome: int
    update_count: int
    off_route_updates: int


class DendriticDelayedCreditLearner:
    """Count-based local learner with decaying synaptic eligibility.

    The learner never receives a column's hidden input/output polarity.  Its
    only durable state is a receiver-local ``basal receptor x outcome`` table.
    The apical route says where context arrived, not what outcome is desired.
    """

    def __init__(
        self,
        receiver_count: int,
        *,
        teaching_mode: TeachingMode | str,
        prior: float = 0.5,
        eligibility_decay: float = 0.97,
        minimum_eligibility: float = 1e-6,
    ) -> None:
        if receiver_count < 2:
            raise ValueError("receiver_count must be at least two")
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        if not 0.0 < eligibility_decay <= 1.0:
            raise ValueError("eligibility_decay must be in (0, 1]")
        if minimum_eligibility < 0.0:
            raise ValueError("minimum_eligibility must be non-negative")
        self.receiver_count = int(receiver_count)
        self.teaching_mode = TeachingMode(teaching_mode)
        self.prior = float(prior)
        self.eligibility_decay = float(eligibility_decay)
        self.minimum_eligibility = float(minimum_eligibility)

        # Two mutually exclusive basal receptors and two possible outcomes per
        # receiver.  No dense receiver-to-receiver weights are present.
        self.outcome_counts = np.zeros(
            (self.receiver_count, 2, 2), dtype=np.float64
        )

        # Bounded working state: one local receptor identity and eligibility
        # scalar per receiver, plus the currently active apical route.
        self.basal_trace = np.zeros(self.receiver_count, dtype=np.uint8)
        self.eligibility = np.zeros(self.receiver_count, dtype=np.float64)
        self.apical_receiver = -1
        self.pending = False

        # Audit counters are telemetry, not learned anatomy.
        self.steps = 0
        self.local_updates = 0
        self.off_route_updates = 0

    @property
    def embodied_bytes(self) -> int:
        """Serialized durable body, excluding transient traces and telemetry."""
        body = (
            self.receiver_count,
            self.teaching_mode.value,
            self.prior,
            self.eligibility_decay,
            self.minimum_eligibility,
            self.outcome_counts,
        )
        return len(pickle.dumps(body, protocol=5))

    @property
    def transient_bytes(self) -> int:
        """Serialized active eligibility state at peak trial occupancy."""
        state = (
            self.basal_trace,
            self.eligibility,
            self.apical_receiver,
            self.pending,
        )
        return len(pickle.dumps(state, protocol=5))

    def probability(self, receiver: int, basal_state: int) -> float:
        """Return local P(outcome=1) for one receiver and basal receptor."""
        receiver = int(receiver)
        basal_state = int(basal_state)
        if not 0 <= receiver < self.receiver_count:
            raise ValueError("receiver is outside the local population")
        if basal_state not in (0, 1):
            raise ValueError("basal_state must be binary")
        row = self.outcome_counts[receiver, basal_state] + self.prior
        return float(row[1] / row.sum())

    def begin_trial(self, basal_states: np.ndarray, apical_receiver: int) -> float:
        """Set local traces and predict before any future outcome is known."""
        if self.pending:
            raise RuntimeError("the previous delayed consequence is still pending")
        states = np.asarray(basal_states, dtype=np.uint8).reshape(-1)
        if states.shape != (self.receiver_count,) or np.any(states > 1):
            raise ValueError("basal_states must contain one binary state per receiver")
        apical_receiver = int(apical_receiver)
        if not 0 <= apical_receiver < self.receiver_count:
            raise ValueError("apical_receiver is outside the local population")

        self.basal_trace[...] = states
        self.eligibility.fill(1.0)
        self.apical_receiver = apical_receiver
        self.pending = True
        return self.probability(apical_receiver, int(states[apical_receiver]))

    def advance_delay(self, steps: int = 1) -> None:
        """Let local eligibility decay while the causal consequence is absent."""
        if not self.pending:
            raise RuntimeError("no trial is awaiting a delayed consequence")
        steps = int(steps)
        if steps < 0:
            raise ValueError("delay steps must be non-negative")
        self.eligibility *= self.eligibility_decay**steps

    def observe_consequence(
        self, outcome: int, *, learn: bool = True
    ) -> DelayedCreditStep:
        """Apply only locally available consequence signals, then clear traces."""
        if not self.pending:
            raise RuntimeError("begin_trial must precede the consequence")
        outcome = int(outcome)
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")

        probability_one = self.probability(
            self.apical_receiver,
            int(self.basal_trace[self.apical_receiver]),
        )
        observed_probability = probability_one if outcome else 1.0 - probability_one
        predicted_outcome = int(probability_one >= 0.5)
        update_count = 0
        off_route_updates = 0

        if learn and self.teaching_mode is not TeachingMode.NONE:
            if self.teaching_mode is TeachingMode.APICAL:
                receivers = (self.apical_receiver,)
            else:
                receivers = range(self.receiver_count)
            for receiver in receivers:
                strength = float(self.eligibility[receiver])
                if strength < self.minimum_eligibility:
                    continue
                basal_state = int(self.basal_trace[receiver])
                self.outcome_counts[receiver, basal_state, outcome] += strength
                update_count += 1
                if receiver != self.apical_receiver:
                    off_route_updates += 1

        self.steps += 1
        self.local_updates += update_count
        self.off_route_updates += off_route_updates
        self.pending = False
        self.apical_receiver = -1
        self.eligibility.fill(0.0)
        return DelayedCreditStep(
            observed_probability=observed_probability,
            predicted_outcome=predicted_outcome,
            update_count=update_count,
            off_route_updates=off_route_updates,
        )

    def run_trial(
        self,
        basal_states: np.ndarray,
        apical_receiver: int,
        delay_steps: int,
        outcome: int,
        *,
        learn: bool = True,
    ) -> DelayedCreditStep:
        """Convenience wrapper preserving predict-delay-update ordering."""
        self.begin_trial(basal_states, apical_receiver)
        self.advance_delay(delay_steps)
        return self.observe_consequence(outcome, learn=learn)

    def reset_episode(self) -> None:
        """Clear bounded working state without changing learned anatomy."""
        self.basal_trace.fill(0)
        self.eligibility.fill(0.0)
        self.apical_receiver = -1
        self.pending = False


class CompetingEligibilityTraceLearner:
    """Local bank of dendritic timescales selected by predictive utility.

    Every receiver maintains several exponentially filtered basal traces.
    Delayed apical consequence updates receiver-local outcome counts for each
    trace.  Conditional entropy selects the trace whose sign is most predictive
    without access to the environment's true delay.
    """

    def __init__(
        self,
        receiver_count: int,
        *,
        trace_decays: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.97, 0.995),
        prior: float = 0.5,
    ) -> None:
        if receiver_count < 2:
            raise ValueError("receiver_count must be at least two")
        decays = np.asarray(trace_decays, dtype=np.float64)
        if decays.ndim != 1 or len(decays) < 2:
            raise ValueError("at least two trace decays are required")
        if np.any(decays < 0.0) or np.any(decays >= 1.0):
            raise ValueError("trace decays must be in [0, 1)")
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.receiver_count = int(receiver_count)
        self.trace_decays = decays
        self.prior = float(prior)
        self.traces = np.zeros((receiver_count, len(decays)), dtype=np.float64)
        self.outcome_counts = np.zeros(
            (receiver_count, len(decays), 2, 2), dtype=np.float64
        )
        self.steps = 0
        self.local_updates = 0

    @property
    def embodied_bytes(self) -> int:
        body = (
            self.receiver_count,
            self.trace_decays,
            self.prior,
            self.outcome_counts,
        )
        return len(pickle.dumps(body, protocol=5))

    @property
    def transient_bytes(self) -> int:
        return len(pickle.dumps(self.traces, protocol=5))

    def observe_sources(self, basal_states: np.ndarray) -> None:
        states = np.asarray(basal_states, dtype=np.uint8).reshape(-1)
        if states.shape != (self.receiver_count,) or np.any(states > 1):
            raise ValueError("basal_states must contain one bit per receiver")
        signed = 2.0 * states.astype(np.float64) - 1.0
        self.traces *= self.trace_decays[None, :]
        self.traces += signed[:, None] * (1.0 - self.trace_decays[None, :])
        self.steps += 1

    def _trace_state(self, receiver: int) -> np.ndarray:
        return (self.traces[receiver] >= 0.0).astype(np.int64)

    def _channel_probabilities(self, receiver: int) -> np.ndarray:
        states = self._trace_state(receiver)
        channels = np.arange(len(self.trace_decays))
        rows = self.outcome_counts[receiver, channels, states] + self.prior
        return rows[:, 1] / rows.sum(axis=1)

    def _conditional_entropies(self, receiver: int) -> np.ndarray:
        joint = self.outcome_counts[receiver] + self.prior
        state_totals = joint.sum(axis=2)
        totals = state_totals.sum(axis=1)
        probabilities = joint / state_totals[:, :, None]
        entropy_by_state = -np.sum(probabilities * np.log2(probabilities), axis=2)
        return np.sum((state_totals / totals[:, None]) * entropy_by_state, axis=1)

    def selected_channel(self, receiver: int) -> int:
        receiver = int(receiver)
        if not 0 <= receiver < self.receiver_count:
            raise ValueError("receiver is outside the local population")
        return int(np.argmin(self._conditional_entropies(receiver)))

    def probability_for_channel(self, receiver: int, channel: int) -> float:
        receiver = int(receiver)
        channel = int(channel)
        if not 0 <= channel < len(self.trace_decays):
            raise ValueError("channel is outside the trace bank")
        return float(self._channel_probabilities(receiver)[channel])

    def observe_consequence(
        self, receiver: int, outcome: int, *, learn: bool = True
    ) -> DelayedCreditStep:
        receiver = int(receiver)
        outcome = int(outcome)
        if not 0 <= receiver < self.receiver_count:
            raise ValueError("receiver is outside the local population")
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        selected = self.selected_channel(receiver)
        probability_one = self.probability_for_channel(receiver, selected)
        observed = probability_one if outcome else 1.0 - probability_one
        update_count = 0
        if learn:
            states = self._trace_state(receiver)
            channels = np.arange(len(self.trace_decays))
            self.outcome_counts[receiver, channels, states, outcome] += 1.0
            update_count = len(self.trace_decays)
            self.local_updates += update_count
        return DelayedCreditStep(
            observed_probability=observed,
            predicted_outcome=int(probability_one >= 0.5),
            update_count=update_count,
            off_route_updates=0,
        )

    def reset_traces(self) -> None:
        self.traces.fill(0.0)
