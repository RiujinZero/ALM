"""Modality-neutral delayed causal stream for long-range credit tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedCausalBatch:
    """Independent basal receptors, one apical context route, and an outcome."""

    basal_states: np.ndarray
    apical_receivers: np.ndarray
    outcomes: np.ndarray
    delay_steps: int


class DelayedCausalTask:
    """Generate hidden, opposing receiver roles without language structure.

    Every trial activates one binary basal receptor in every receiver.  A
    context pathway terminates on one receiver's apical compartment.  After a
    distractor delay, the outcome is that receiver's basal state, optionally
    inverted by its fixed hidden polarity.  The learner sees the route and the
    later consequence, but never the polarity table used by the environment.
    """

    def __init__(
        self,
        receiver_count: int = 16,
        *,
        delay_steps: int = 8,
        seed: int = 0,
    ) -> None:
        if receiver_count < 2:
            raise ValueError("receiver_count must be at least two")
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.receiver_count = int(receiver_count)
        self.delay_steps = int(delay_steps)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # Balanced opposing causal roles, randomly assigned to intermingled
        # receivers.  This hidden table is never passed to the learner.
        polarities = np.arange(self.receiver_count, dtype=np.uint8) & 1
        rng.shuffle(polarities)
        self.hidden_polarities = polarities
        self._rng = rng

    def sample(self, trial_count: int) -> DelayedCausalBatch:
        trial_count = int(trial_count)
        if trial_count < 1:
            raise ValueError("trial_count must be positive")
        basal = self._rng.integers(
            0,
            2,
            size=(trial_count, self.receiver_count),
            dtype=np.uint8,
        )
        receivers = self._rng.integers(
            0, self.receiver_count, size=trial_count, dtype=np.int64
        )
        selected = basal[np.arange(trial_count), receivers]
        outcomes = selected ^ self.hidden_polarities[receivers]
        return DelayedCausalBatch(
            basal_states=basal,
            apical_receivers=receivers,
            outcomes=outcomes,
            delay_steps=self.delay_steps,
        )


@dataclass(frozen=True)
class OverlappingCausalStream:
    """Continuous sources with retrospectively routed delayed outcomes."""

    basal_states: np.ndarray
    apical_receivers: np.ndarray
    outcomes: np.ndarray
    causal_delay: int


class OverlappingDelayedCausalTask:
    """Create interference on both sides of a useful eligibility lifetime."""

    def __init__(
        self,
        receiver_count: int = 8,
        *,
        causal_delay: int = 8,
        seed: int = 0,
    ) -> None:
        if receiver_count < 2:
            raise ValueError("receiver_count must be at least two")
        if causal_delay < 1:
            raise ValueError("causal_delay must be positive")
        self.receiver_count = int(receiver_count)
        self.causal_delay = int(causal_delay)
        self._rng = np.random.default_rng(seed)

    def sample(self, step_count: int) -> OverlappingCausalStream:
        step_count = int(step_count)
        if step_count <= self.causal_delay:
            raise ValueError("step_count must exceed the causal delay")
        basal = self._rng.integers(
            0,
            2,
            size=(step_count, self.receiver_count),
            dtype=np.uint8,
        )
        receivers = self._rng.integers(
            0, self.receiver_count, size=step_count, dtype=np.int64
        )
        outcomes = np.full(step_count, -1, dtype=np.int8)
        valid = np.arange(self.causal_delay, step_count)
        outcomes[valid] = basal[
            valid - self.causal_delay,
            receivers[valid],
        ]
        return OverlappingCausalStream(
            basal_states=basal,
            apical_receivers=receivers,
            outcomes=outcomes,
            causal_delay=self.causal_delay,
        )
