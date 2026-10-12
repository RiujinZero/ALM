"""Fast observational hindsight-outcome-to-production interface probe.

This synthetic ablation joins two interfaces without claiming that either one is
already a conversation-level goal system:

* a delayed successor map treats an actually observed terminal vector as a
  hindsight desired-outcome *proxy* and keeps its prediction in an exact,
  supplied fixed-lag register;
* an action-conditioned consequence model predicts which terminal vector each
  coarse act produces, then selects the act most compatible with the proxy;
* :class:`ContextCoupledByteWriter` holds that selected act fixed while emitting
  a multi-symbol command from its own generated suffix.

The first observation is an injective one-hot symbolic cue for the generator's
intention.  It is therefore observed intent information, not latent-intent
inference.  A scripted logger takes the cue-matching action with configurable
probability, and all context/action consequences are observed.  The learners
receive finite observation vectors, logged coarse action IDs, subsequent
observation vectors, and (for the actuator only) engineered command targets.
They never receive reward, valence, success labels, a field naming the correct
action, semantic modality names, or response candidates.  Episode boundaries,
the exact prediction lag, the action alphabet, the observation code, behavior
policy, and command codebook are supplied scaffold.

This is an interface and identifiability assay, not reinforcement learning,
open-ended generation, native ALM evidence, or a claim that predictable futures
are inherently desirable.  A direct cue/context behavior-cloning control and
decision-time jitter diagnostics are reported precisely to prevent the central
100% interface result from being read as evidence for those broader claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    WRITER_KINDS,
    ContextCoupledByteWriter,
    WriterKind,
)

OutcomeModelKind = Literal["local", "additive", "additive_wide", "coupled"]
OUTCOME_MODEL_KINDS: tuple[OutcomeModelKind, ...] = (
    "local",
    "additive",
    "additive_wide",
    "coupled",
)


@dataclass(frozen=True)
class VectorCode:
    """Supplied synthetic encoder; the learning rules only see its vectors."""

    goals: int = 4
    contexts: int = 4
    distractors: int = 8

    def __post_init__(self) -> None:
        if self.goals < 2:
            raise ValueError("goals must be at least two")
        if self.contexts < 2:
            raise ValueError("contexts must be at least two")
        if self.distractors < 2:
            raise ValueError("distractors must be at least two")

    @property
    def dimension(self) -> int:
        return 4 + self.goals + self.contexts + self.distractors + self.goals

    @property
    def cue_slice(self) -> slice:
        return slice(4, 4 + self.goals)

    @property
    def context_slice(self) -> slice:
        start = self.cue_slice.stop
        return slice(start, start + self.contexts)

    @property
    def distractor_slice(self) -> slice:
        start = self.context_slice.stop
        return slice(start, start + self.distractors)

    @property
    def outcome_slice(self) -> slice:
        start = self.distractor_slice.stop
        return slice(start, start + self.goals)

    def cue(self, outcome: int) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[0] = 1.0
        vector[self.cue_slice.start + self._goal(outcome)] = 1.0
        return vector

    def distractor(self, identity: int) -> np.ndarray:
        if not 0 <= int(identity) < self.distractors:
            raise ValueError("distractor identity is outside the code")
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[1] = 1.0
        vector[self.distractor_slice.start + int(identity)] = 1.0
        return vector

    def decision(self, context: int) -> np.ndarray:
        if not 0 <= int(context) < self.contexts:
            raise ValueError("context is outside the code")
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[2] = 1.0
        vector[self.context_slice.start + int(context)] = 1.0
        return vector

    def terminal(self, outcome: int) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[3] = 1.0
        vector[self.outcome_slice.start + self._goal(outcome)] = 1.0
        return vector

    def decode_cue(self, vector: np.ndarray) -> int:
        return int(np.argmax(self._vector(vector)[self.cue_slice]))

    def decode_context(self, vector: np.ndarray) -> int:
        return int(np.argmax(self._vector(vector)[self.context_slice]))

    def decode_outcome(self, vector: np.ndarray) -> int:
        return int(np.argmax(self._vector(vector)[self.outcome_slice]))

    def _goal(self, value: int) -> int:
        if not 0 <= int(value) < self.goals:
            raise ValueError("outcome is outside the code")
        return int(value)

    def _vector(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"observation must have shape ({self.dimension},)")
        return vector


@dataclass(frozen=True)
class ObservedTrajectory:
    """Learner-facing training record: observations and an observed action."""

    key: str
    observations: np.ndarray
    action: int


@dataclass(frozen=True)
class TrainingAudit:
    """Generator-only facts that never enter a learner update."""

    key: str
    context: int
    intended_outcome: int
    observed_action: int
    observed_outcome: int
    cue_followed: bool


@dataclass(frozen=True)
class CrossoverProbe:
    """Held-out decision prefix plus audit-only counterfactual endpoint."""

    key: str
    group: int
    context: int
    prefix: np.ndarray
    desired_outcome: int
    required_action: int


def transition_outcome(context: int, action: int, *, goals: int) -> int:
    """Supplied deterministic world transition, kept outside every learner."""

    if goals < 2:
        raise ValueError("goals must be at least two")
    if not 0 <= int(context) < goals or not 0 <= int(action) < goals:
        raise ValueError("context and action must be valid modular identities")
    return (int(context) + int(action)) % goals


def required_action(context: int, desired_outcome: int, *, goals: int) -> int:
    """Audit-only inverse of :func:`transition_outcome`."""

    if not 0 <= int(desired_outcome) < goals:
        raise ValueError("desired outcome is outside the world")
    return (int(desired_outcome) - int(context)) % goals


def _validate_trajectory(
    trajectory: ObservedTrajectory,
    *,
    code: VectorCode,
    actions: int,
    horizon: int,
) -> None:
    observations = np.asarray(trajectory.observations, dtype=np.float64)
    if observations.shape != (horizon + 1, code.dimension):
        raise ValueError(
            f"trajectory observations must have shape ({horizon + 1}, {code.dimension})"
        )
    if not np.isfinite(observations).all():
        raise ValueError("trajectory observations must be finite")
    if not 0 <= int(trajectory.action) < actions:
        raise ValueError("trajectory action is outside the action alphabet")


def generate_observed_trajectories(
    n_trajectories: int,
    *,
    horizon: int,
    code: VectorCode,
    rng: np.random.Generator,
    local_window: int = 4,
    dominant_probability: float = 0.70,
    cue_follow_probability: float = 0.80,
    key_prefix: str = "train",
) -> tuple[list[ObservedTrajectory], list[TrainingAudit]]:
    """Generate behavior logs without exposing the latent intention to learners.

    Contexts are balanced.  For each context, the outcome produced by action zero
    is the common latent intention; the remaining intentions share the residual
    mass.  A behavior process follows the cue only probabilistically and otherwise
    samples another legal act.  Therefore action zero is the local/marginal mode,
    while mistakes provide coverage of every observed transition.
    """

    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")
    if horizon <= local_window:
        raise ValueError("horizon must exceed local_window")
    if local_window < 1:
        raise ValueError("local_window must be positive")
    if code.contexts != code.goals:
        raise ValueError("this modular assay requires contexts == goals")
    if not 0.0 < dominant_probability < 1.0:
        raise ValueError("dominant_probability must be in (0, 1)")
    if not 0.0 < cue_follow_probability < 1.0:
        raise ValueError("cue_follow_probability must be in (0, 1)")

    context_order = np.resize(np.arange(code.contexts, dtype=np.int64), n_trajectories)
    rng.shuffle(context_order)
    trajectories: list[ObservedTrajectory] = []
    audit: list[TrainingAudit] = []
    for index, context_value in enumerate(context_order):
        context = int(context_value)
        dominant_outcome = transition_outcome(context, 0, goals=code.goals)
        if rng.random() < dominant_probability:
            intention = dominant_outcome
        else:
            alternatives = [
                goal for goal in range(code.goals) if goal != dominant_outcome
            ]
            intention = int(rng.choice(alternatives))
        matching_action = required_action(context, intention, goals=code.goals)
        followed = bool(rng.random() < cue_follow_probability)
        if followed:
            action = matching_action
        else:
            alternatives = [
                candidate
                for candidate in range(code.goals)
                if candidate != matching_action
            ]
            action = int(rng.choice(alternatives))
        observed_outcome = transition_outcome(context, action, goals=code.goals)

        observations = np.empty((horizon + 1, code.dimension), dtype=np.float64)
        observations[0] = code.cue(intention)
        filler_stop = horizon - local_window
        for position in range(1, filler_stop):
            observations[position] = code.distractor(
                int(rng.integers(code.distractors))
            )
        observations[filler_stop:horizon] = code.decision(context)
        observations[horizon] = code.terminal(observed_outcome)
        key = f"{key_prefix}-{index:07d}"
        trajectories.append(
            ObservedTrajectory(key=key, observations=observations, action=action)
        )
        audit.append(
            TrainingAudit(
                key=key,
                context=context,
                intended_outcome=intention,
                observed_action=action,
                observed_outcome=observed_outcome,
                cue_followed=followed,
            )
        )
    return trajectories, audit


def generate_crossover_probes(
    groups: int,
    *,
    horizon: int,
    code: VectorCode,
    rng: np.random.Generator,
    local_window: int = 4,
    key_prefix: str = "eval",
) -> list[CrossoverProbe]:
    """Create matched groups differing only in their first, distant cue."""

    if groups <= 0:
        raise ValueError("groups must be positive")
    if horizon <= local_window or local_window < 1:
        raise ValueError("horizon must exceed a positive local_window")
    if code.contexts != code.goals:
        raise ValueError("this modular assay requires contexts == goals")

    probes: list[CrossoverProbe] = []
    for group in range(groups):
        context = group % code.contexts
        common = np.empty((horizon, code.dimension), dtype=np.float64)
        common[0] = code.cue(0)  # Replaced below; never used as shared evidence.
        filler_stop = horizon - local_window
        for position in range(1, filler_stop):
            common[position] = code.distractor(int(rng.integers(code.distractors)))
        common[filler_stop:horizon] = code.decision(context)
        for outcome in range(code.goals):
            prefix = common.copy()
            prefix[0] = code.cue(outcome)
            probes.append(
                CrossoverProbe(
                    key=f"{key_prefix}-{group:05d}-{outcome:02d}",
                    group=group,
                    context=context,
                    prefix=prefix,
                    desired_outcome=outcome,
                    required_action=required_action(context, outcome, goals=code.goals),
                )
            )
    return probes


def shuffled_terminal_control(
    trajectories: Sequence[ObservedTrajectory],
    *,
    code: VectorCode,
    horizon: int,
    rng: np.random.Generator,
) -> list[ObservedTrajectory]:
    """Balance terminal targets within each cue, destroying cue dependence."""

    grouped: dict[int, list[int]] = {}
    for index, trajectory in enumerate(trajectories):
        _validate_trajectory(trajectory, code=code, actions=code.goals, horizon=horizon)
        cue = code.decode_cue(trajectory.observations[0])
        grouped.setdefault(cue, []).append(index)

    replacements = np.empty(len(trajectories), dtype=np.int64)
    for indices in grouped.values():
        balanced = np.resize(np.arange(code.goals, dtype=np.int64), len(indices))
        rng.shuffle(balanced)
        replacements[np.asarray(indices, dtype=np.int64)] = balanced

    shuffled: list[ObservedTrajectory] = []
    for index, trajectory in enumerate(trajectories):
        observations = np.array(trajectory.observations, copy=True)
        observations[-1] = code.terminal(int(replacements[index]))
        shuffled.append(
            ObservedTrajectory(
                key=trajectory.key,
                observations=observations,
                action=trajectory.action,
            )
        )
    return shuffled


def _linear_fit(inputs: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    if inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets):
        raise ValueError("linear fit expects aligned two-dimensional arrays")
    design = np.concatenate(
        (np.ones((len(inputs), 1), dtype=np.float64), inputs), axis=1
    )
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ targets)


class LaggedHindsightGoal:
    """Linear far-successor map whose prediction survives in a delay queue."""

    def __init__(self, dimension: int, *, lag: int, ridge: float = 1e-3) -> None:
        if dimension < 2:
            raise ValueError("dimension must be at least two")
        if lag < 1:
            raise ValueError("lag must be positive")
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        self.dimension = int(dimension)
        self.lag = int(lag)
        self.ridge = float(ridge)
        self.weights = np.zeros((self.dimension + 1, self.dimension))
        self.fitted_pairs = 0

    def fit(self, trajectories: Sequence[ObservedTrajectory]) -> None:
        sources: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for trajectory in trajectories:
            observations = np.asarray(trajectory.observations, dtype=np.float64)
            if observations.ndim != 2 or observations.shape[1] != self.dimension:
                raise ValueError("trajectory observation dimension mismatch")
            if len(observations) <= self.lag:
                continue
            sources.extend(observations[: -self.lag])
            targets.extend(observations[self.lag :])
        if not sources:
            raise ValueError("no lagged observation pairs were available")
        self.weights = _linear_fit(np.asarray(sources), np.asarray(targets), self.ridge)
        self.fitted_pairs = len(sources)

    def predict_source(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float64)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError("observation has invalid shape or values")
        augmented = np.concatenate((np.ones(1), vector))
        return np.asarray(augmented @ self.weights, dtype=np.float64)

    def forecast(
        self,
        prefix: np.ndarray,
        *,
        reset_index: int | None = None,
    ) -> np.ndarray:
        """Return the prediction due after processing ``prefix``.

        A reset is an intervention on transient state only.  For the standard
        prefix length equal to ``lag``, any reset after index zero and before the
        decision removes the cue's scheduled prediction.
        """

        observations = np.asarray(prefix, dtype=np.float64)
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise ValueError("prefix observation dimension mismatch")
        if reset_index is not None and not 0 < reset_index < len(observations):
            raise ValueError("reset_index must occur after the cue and before decision")
        pending: list[np.ndarray | None] = [None] * self.lag
        for index, observation in enumerate(observations):
            if reset_index is not None and index == reset_index:
                pending = [None] * self.lag
            pending.pop(0)
            pending.append(self.predict_source(observation))
        due = pending[0]
        return np.zeros(self.dimension) if due is None else np.array(due, copy=True)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.weights.view(np.uint8)).hexdigest()


class ActionConditionedOutcomeModel:
    """Predict an observed consequence for each act, then compare it to a proxy."""

    def __init__(
        self,
        kind: OutcomeModelKind,
        *,
        observation_dimension: int,
        actions: int,
        ridge: float = 1e-3,
    ) -> None:
        if kind not in OUTCOME_MODEL_KINDS:
            raise ValueError(f"unknown outcome model kind: {kind!r}")
        if observation_dimension < 2 or actions < 2:
            raise ValueError("observation_dimension and actions must be at least two")
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        self.kind = kind
        self.observation_dimension = int(observation_dimension)
        self.actions = int(actions)
        self.ridge = float(ridge)
        self.weights = np.zeros(
            (self.feature_dimension, self.observation_dimension), dtype=np.float64
        )
        self.fitted_transitions = 0

    @property
    def feature_dimension(self) -> int:
        if self.kind == "local":
            return 1 + self.actions
        base = 1 + self.observation_dimension + self.actions
        if self.kind in ("additive_wide", "coupled"):
            return base + self.observation_dimension * self.actions
        return base

    @property
    def parameter_count(self) -> int:
        return int(self.feature_dimension * self.observation_dimension)

    def features(self, decision: np.ndarray, action: int) -> np.ndarray:
        vector = np.asarray(decision, dtype=np.float64)
        if (
            vector.shape != (self.observation_dimension,)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("decision observation has invalid shape or values")
        if not 0 <= int(action) < self.actions:
            raise ValueError("action is outside the action alphabet")
        action_code = np.zeros(self.actions, dtype=np.float64)
        action_code[int(action)] = 1.0
        if self.kind == "local":
            return np.concatenate((np.ones(1), action_code))
        additive = np.concatenate((np.ones(1), vector, action_code))
        if self.kind == "additive":
            return additive
        if self.kind == "coupled":
            extra = np.outer(vector, action_code).reshape(-1)
        else:
            # Same allocation and per-example nonzero count as the interaction
            # bank, but every added coordinate is a pure duplicate of context.
            extra = np.zeros(self.observation_dimension * self.actions)
            extra[: self.observation_dimension] = vector
        return np.concatenate((additive, extra))

    def active_feature_count(self, decision: np.ndarray, action: int) -> int:
        return int(np.count_nonzero(self.features(decision, action)))

    def fit(
        self,
        trajectories: Sequence[ObservedTrajectory],
        *,
        horizon: int,
    ) -> None:
        features: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []
        for trajectory in trajectories:
            observations = np.asarray(trajectory.observations, dtype=np.float64)
            if observations.shape != (horizon + 1, self.observation_dimension):
                raise ValueError("trajectory shape does not match model")
            features.append(self.features(observations[-2], trajectory.action))
            outcomes.append(observations[-1])
        if not features:
            raise ValueError("at least one trajectory is required")
        # _linear_fit supplies its own intercept; the explicit feature bias is
        # retained to keep allocation/active counts aligned with the mechanism.
        fitted = _linear_fit(np.asarray(features), np.asarray(outcomes), self.ridge)
        self.weights = fitted[1:]
        self.weights[0] += fitted[0]
        self.fitted_transitions = len(features)

    def predict(self, decision: np.ndarray, action: int) -> np.ndarray:
        return self.features(decision, action) @ self.weights

    def values(self, decision: np.ndarray, goal_proxy: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal_proxy, dtype=np.float64)
        if goal.shape != (self.observation_dimension,) or not np.isfinite(goal).all():
            raise ValueError("goal proxy has invalid shape or values")
        predictions = np.asarray(
            [self.predict(decision, action) for action in range(self.actions)]
        )
        # Negative squared distance is a generic compatibility, not reward or
        # valence.  Equal scores deterministically choose the lower action ID.
        return -np.sum((predictions - goal[None, :]) ** 2, axis=1)

    def choose(self, decision: np.ndarray, goal_proxy: np.ndarray) -> int:
        return int(np.argmax(self.values(decision, goal_proxy)))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.weights.view(np.uint8)).hexdigest()


class ObservableActionPrior:
    """Most frequent logged action for an exactly observed decision context."""

    def __init__(self, actions: int) -> None:
        self.actions = int(actions)
        self.counts: dict[bytes, np.ndarray] = {}

    def fit(self, trajectories: Sequence[ObservedTrajectory]) -> None:
        for trajectory in trajectories:
            decision = np.asarray(trajectory.observations[-2], dtype=np.float64)
            row = self.counts.setdefault(
                decision.tobytes(), np.zeros(self.actions, dtype=np.int64)
            )
            row[int(trajectory.action)] += 1

    def choose(self, decision: np.ndarray) -> int:
        row = self.counts.get(np.asarray(decision, dtype=np.float64).tobytes())
        return 0 if row is None else int(np.argmax(row))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for key, counts in sorted(self.counts.items()):
            digest.update(key)
            digest.update(np.asarray(counts, dtype=np.int64).view(np.uint8))
        return digest.hexdigest()


class ObservedCueContextBehaviorClone:
    """Modal logged action for an observed symbolic cue/context pair.

    This is deliberately a strong allowed-input control, not a proposed
    persistence mechanism.  It has oracle access to the first observation when
    scoring a prefix, and it learns only from the same noisy observed action IDs
    available to the consequence model.  If it solves the assay, the dataset
    does not establish that explicit outcome modeling is necessary.
    """

    def __init__(self, *, observation_dimension: int, actions: int) -> None:
        if observation_dimension < 2 or actions < 2:
            raise ValueError("observation_dimension and actions must be at least two")
        self.observation_dimension = int(observation_dimension)
        self.actions = int(actions)
        self.counts: dict[tuple[bytes, bytes], np.ndarray] = {}
        self.fitted_trajectories = 0

    def _key(self, cue: np.ndarray, decision: np.ndarray) -> tuple[bytes, bytes]:
        cue_vector = np.asarray(cue, dtype=np.float64)
        decision_vector = np.asarray(decision, dtype=np.float64)
        expected = (self.observation_dimension,)
        if (
            cue_vector.shape != expected
            or decision_vector.shape != expected
            or not np.isfinite(cue_vector).all()
            or not np.isfinite(decision_vector).all()
        ):
            raise ValueError("cue and decision observations must be finite vectors")
        return cue_vector.tobytes(), decision_vector.tobytes()

    def fit(
        self,
        trajectories: Sequence[ObservedTrajectory],
        *,
        horizon: int,
    ) -> None:
        for trajectory in trajectories:
            observations = np.asarray(trajectory.observations, dtype=np.float64)
            if observations.shape != (horizon + 1, self.observation_dimension):
                raise ValueError("trajectory shape does not match behavior clone")
            if not 0 <= int(trajectory.action) < self.actions:
                raise ValueError("trajectory action is outside the action alphabet")
            row = self.counts.setdefault(
                self._key(observations[0], observations[-2]),
                np.zeros(self.actions, dtype=np.int64),
            )
            row[int(trajectory.action)] += 1
            self.fitted_trajectories += 1
        if self.fitted_trajectories == 0:
            raise ValueError("at least one trajectory is required")

    def choose(self, cue: np.ndarray, decision: np.ndarray) -> int:
        row = self.counts.get(self._key(cue, decision))
        return 0 if row is None else int(np.argmax(row))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for (cue, decision), counts in sorted(self.counts.items()):
            digest.update(cue)
            digest.update(decision)
            digest.update(np.asarray(counts, dtype=np.int64).view(np.uint8))
        return digest.hexdigest()


COMMAND_ALPHABET = 9
COMMAND_STEPS = (1, 2, 4, 5)
COMMAND_PREFIX = (0,)


def action_command(action: int) -> tuple[int, ...]:
    """Equal-multiset transition cycle; ordering alone identifies the action."""

    if not 0 <= int(action) < len(COMMAND_STEPS):
        raise ValueError("action has no command cycle")
    step = COMMAND_STEPS[int(action)]
    return tuple((position * step) % COMMAND_ALPHABET for position in range(1, 10))


def action_feature_ids(action: int) -> tuple[int, ...]:
    if not 0 <= int(action) < len(COMMAND_STEPS):
        raise ValueError("action has no persistent feature")
    return (10_000 + int(action),)


def decode_action_command(command: Sequence[int]) -> int | None:
    value = tuple(int(symbol) for symbol in command)
    for action in range(len(COMMAND_STEPS)):
        if value == action_command(action):
            return action
    return None


def fit_writers(
    trajectories: Sequence[ObservedTrajectory],
    *,
    table_bits: int,
) -> dict[WriterKind, ContextCoupledByteWriter]:
    """Fit identical-allocation actuator writers on logged action channels."""

    writers = {
        kind: ContextCoupledByteWriter(
            kind,
            alphabet_size=COMMAND_ALPHABET,
            table_bits=table_bits,
            learning_rate=0.55,
            local_orders=(1,),
            max_persistent_features=1,
        )
        for kind in WRITER_KINDS
    }
    for trajectory in trajectories:
        command = action_command(trajectory.action)
        state = action_feature_ids(trajectory.action)
        for writer in writers.values():
            writer.fit_episode(command, state, prefix=COMMAND_PREFIX)
    return writers


def generate_action(
    writer: ContextCoupledByteWriter,
    selected_action: int,
    *,
    disconnected: bool = False,
) -> tuple[int | None, tuple[int, ...]]:
    state_action = 0 if disconnected else int(selected_action)
    command = writer.generate(
        action_feature_ids(state_action),
        max_symbols=COMMAND_ALPHABET,
        prefix=COMMAND_PREFIX,
    )
    return decode_action_command(command), command


def _training_coverage(
    audit: Sequence[TrainingAudit], *, goals: int
) -> dict[str, object]:
    action_counts = np.zeros(goals, dtype=np.int64)
    context_action = np.zeros((goals, goals), dtype=np.int64)
    outcome_counts = np.zeros(goals, dtype=np.int64)
    followed = 0
    for row in audit:
        action_counts[row.observed_action] += 1
        context_action[row.context, row.observed_action] += 1
        outcome_counts[row.observed_outcome] += 1
        followed += int(row.cue_followed)
    fractions = action_counts / max(1, int(action_counts.sum()))
    return {
        "action_counts": action_counts.tolist(),
        "action_fractions": fractions.tolist(),
        "action_zero_is_unique_marginal_majority": bool(
            action_counts[0] > int(np.max(action_counts[1:]))
        ),
        "action_zero_is_unique_majority_in_every_context": bool(
            np.all(context_action[:, 0] > np.max(context_action[:, 1:], axis=1))
        ),
        "context_action_counts": context_action.tolist(),
        "minimum_context_action_coverage": int(np.min(context_action)),
        "outcome_counts": outcome_counts.tolist(),
        "cue_follow_fraction": followed / len(audit),
    }


def _goal_vectors(
    probes: Sequence[CrossoverProbe],
    *,
    condition: str,
    code: VectorCode,
    causal: LaggedHindsightGoal,
    shuffled: LaggedHindsightGoal,
    reset_index: int,
) -> list[np.ndarray]:
    causal_values = [causal.forecast(probe.prefix) for probe in probes]
    if condition == "oracle":
        return [code.terminal(probe.desired_outcome) for probe in probes]
    if condition == "causal":
        return causal_values
    if condition == "shuffled_training":
        return [shuffled.forecast(probe.prefix) for probe in probes]
    if condition == "register_reset":
        return [
            causal.forecast(probe.prefix, reset_index=reset_index) for probe in probes
        ]
    if condition == "swapped":
        lookup = {
            (probe.group, probe.desired_outcome): value
            for probe, value in zip(probes, causal_values, strict=True)
        }
        return [
            lookup[(probe.group, (probe.desired_outcome + 1) % code.goals)]
            for probe in probes
        ]
    raise ValueError(f"unknown goal condition: {condition!r}")


def _score_goal_forecasts(
    probes: Sequence[CrossoverProbe],
    vectors: Sequence[np.ndarray],
    *,
    code: VectorCode,
) -> dict[str, float | int]:
    decoded = [code.decode_outcome(vector) for vector in vectors]
    correct = sum(
        int(value == probe.desired_outcome)
        for value, probe in zip(decoded, probes, strict=True)
    )
    return {"episodes": len(probes), "accuracy": correct / len(probes)}


def jittered_decision_prefix(prefix: np.ndarray, offset: int) -> np.ndarray:
    """Move the decision one step early/late without changing its local suffix.

    The intervention removes or duplicates the first post-cue distractor.  The
    cue itself and the complete decision-local suffix are unchanged.  It is an
    audit of the supplied exact-lag queue, not an attempt to repair it.
    """

    observations = np.asarray(prefix, dtype=np.float64)
    if observations.ndim != 2 or len(observations) < 3:
        raise ValueError("prefix must be a two-dimensional sequence of length >= 3")
    if int(offset) == -1:
        return np.concatenate((observations[:1], observations[2:]), axis=0)
    if int(offset) == 0:
        return np.array(observations, copy=True)
    if int(offset) == 1:
        return np.concatenate((observations[:2], observations[1:]), axis=0)
    raise ValueError(
        "only one-step early, exact, and one-step late offsets are defined"
    )


def _selection_metrics(
    probes: Sequence[CrossoverProbe],
    selected: Sequence[int],
    *,
    goals: int,
) -> dict[str, float | int]:
    if len(probes) != len(selected):
        raise ValueError("probes and selected actions must be aligned")
    correct = 0
    outcome_consistent = 0
    minority_correct = 0
    minority_total = 0
    group_success: dict[int, list[bool]] = {}
    for probe, action in zip(probes, selected, strict=True):
        is_correct = int(action) == probe.required_action
        correct += int(is_correct)
        outcome_consistent += int(
            transition_outcome(probe.context, int(action), goals=goals)
            == probe.desired_outcome
        )
        if probe.required_action != 0:
            minority_total += 1
            minority_correct += int(is_correct)
        group_success.setdefault(probe.group, []).append(is_correct)
    return {
        "episodes": len(probes),
        "action_accuracy": correct / len(probes),
        "predicted_outcome_consistency": outcome_consistent / len(probes),
        "less_locally_likely_cases": minority_total,
        "less_locally_likely_action_accuracy": minority_correct
        / max(1, minority_total),
        "complete_crossover_groups": sum(
            int(all(values)) for values in group_success.values()
        ),
        "complete_crossover_group_rate": sum(
            int(all(values)) for values in group_success.values()
        )
        / len(group_success),
    }


def _score_selector(
    probes: Sequence[CrossoverProbe],
    vectors: Sequence[np.ndarray],
    *,
    model: ActionConditionedOutcomeModel,
    prior: ObservableActionPrior,
    use_prior: bool = False,
) -> tuple[dict[str, float | int], list[int]]:
    selected: list[int] = []
    for probe, goal in zip(probes, vectors, strict=True):
        decision = probe.prefix[-1]
        action = prior.choose(decision) if use_prior else model.choose(decision, goal)
        selected.append(action)
    return _selection_metrics(probes, selected, goals=model.actions), selected


def _score_behavior_clone(
    probes: Sequence[CrossoverProbe],
    clone: ObservedCueContextBehaviorClone,
) -> tuple[dict[str, float | int], list[int]]:
    selected = [clone.choose(probe.prefix[0], probe.prefix[-1]) for probe in probes]
    return (
        _selection_metrics(probes, selected, goals=clone.actions),
        selected,
    )


def _score_writer_interface(
    writer: ContextCoupledByteWriter, *, actions: int
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    correct = 0
    for action in range(actions):
        decoded, generated = generate_action(writer, action)
        correct += int(decoded == action)
        rows.append(
            {
                "selected_action": action,
                "decoded_action": decoded,
                "exact": decoded == action,
                "generated": list(generated),
            }
        )
    return {
        "actions": actions,
        "exact_action_rate": correct / actions,
        "rows": rows,
    }


def _score_end_to_end(
    probes: Sequence[CrossoverProbe],
    selected: Sequence[int],
    *,
    writer: ContextCoupledByteWriter,
    goals: int,
    disconnected: bool = False,
) -> dict[str, float | int]:
    valid = 0
    exact_selected = 0
    desired = 0
    minority_desired = 0
    minority_total = 0
    groups: dict[int, list[bool]] = {}
    for probe, selected_action in zip(probes, selected, strict=True):
        executed, _ = generate_action(
            writer, selected_action, disconnected=disconnected
        )
        valid += int(executed is not None)
        exact_selected += int(executed == selected_action)
        success = bool(
            executed is not None
            and transition_outcome(probe.context, executed, goals=goals)
            == probe.desired_outcome
        )
        desired += int(success)
        if probe.required_action != 0:
            minority_total += 1
            minority_desired += int(success)
        groups.setdefault(probe.group, []).append(success)
    return {
        "episodes": len(probes),
        "valid_command_rate": valid / len(probes),
        "realized_selected_action_rate": exact_selected / len(probes),
        "desired_outcome_proxy_consistency": desired / len(probes),
        "less_locally_likely_cases": minority_total,
        "less_locally_likely_outcome_consistency": minority_desired
        / max(1, minority_total),
        "complete_crossover_group_rate": sum(
            int(all(values)) for values in groups.values()
        )
        / len(groups),
    }


def _suffixes_identical(probes: Sequence[CrossoverProbe]) -> bool:
    grouped: dict[int, list[CrossoverProbe]] = {}
    for probe in probes:
        grouped.setdefault(probe.group, []).append(probe)
    return all(
        all(np.array_equal(rows[0].prefix[1:], row.prefix[1:]) for row in rows[1:])
        for rows in grouped.values()
    )


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(seed: int) -> dict[str, object]:
    source = Path(__file__).resolve()
    project_root = source.parents[3]
    focused_test = (
        project_root
        / "tests"
        / "test_dendritic_basin_hindsight_outcome_control_ablation.py"
    )
    writer_source = (
        source.parents[1] / "dendritic_basin" / "context_coupled_byte_writer.py"
    )
    streams = {
        "training_trajectories": seed,
        "evaluation_probes": seed + 1,
        "shuffled_terminal_control": seed + 2,
    }
    return {
        "experiment_source": str(source.relative_to(project_root)).replace("\\", "/"),
        "experiment_source_sha256": _file_sha256(source),
        "focused_test": str(focused_test.relative_to(project_root)).replace("\\", "/"),
        "focused_test_sha256": _file_sha256(focused_test),
        "writer_source": str(writer_source.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "writer_source_sha256": _file_sha256(writer_source),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "rng_constructor": "numpy.random.default_rng",
        "bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
        "independent_seed_streams": streams,
        "determinism_note": (
            "fixed-config metrics are deterministic for these source hashes; "
            "elapsed_seconds is intentionally not deterministic"
        ),
    }


def run_ablation(
    *,
    train_trajectories: int = 4096,
    crossover_groups: int = 64,
    horizon: int = 64,
    local_window: int = 4,
    seed: int = 153,
    dominant_probability: float = 0.70,
    cue_follow_probability: float = 0.80,
    table_bits: int = 12,
) -> dict[str, object]:
    """Fit the self-supervised bridge and execute all predeclared controls."""

    started = time.perf_counter()
    code = VectorCode()
    train, train_audit = generate_observed_trajectories(
        train_trajectories,
        horizon=horizon,
        code=code,
        rng=np.random.default_rng(seed),
        local_window=local_window,
        dominant_probability=dominant_probability,
        cue_follow_probability=cue_follow_probability,
        key_prefix=f"train-{seed}",
    )
    probes = generate_crossover_probes(
        crossover_groups,
        horizon=horizon,
        code=code,
        rng=np.random.default_rng(seed + 1),
        local_window=local_window,
        key_prefix=f"eval-{seed}",
    )
    shuffled_train = shuffled_terminal_control(
        train,
        code=code,
        horizon=horizon,
        rng=np.random.default_rng(seed + 2),
    )

    causal_goal = LaggedHindsightGoal(code.dimension, lag=horizon)
    shuffled_goal = LaggedHindsightGoal(code.dimension, lag=horizon)
    causal_goal.fit(train)
    shuffled_goal.fit(shuffled_train)

    outcome_models = {
        kind: ActionConditionedOutcomeModel(
            kind,
            observation_dimension=code.dimension,
            actions=code.goals,
        )
        for kind in OUTCOME_MODEL_KINDS
    }
    for model in outcome_models.values():
        model.fit(train, horizon=horizon)
    prior = ObservableActionPrior(code.goals)
    prior.fit(train)
    behavior_clone = ObservedCueContextBehaviorClone(
        observation_dimension=code.dimension,
        actions=code.goals,
    )
    behavior_clone.fit(train, horizon=horizon)
    writers = fit_writers(train, table_bits=table_bits)

    fingerprints_before = {
        "causal_goal": causal_goal.fingerprint(),
        "shuffled_goal": shuffled_goal.fingerprint(),
        "outcome_models": {
            kind: model.fingerprint() for kind, model in outcome_models.items()
        },
        "observable_action_prior": prior.fingerprint(),
        "cue_context_behavior_clone": behavior_clone.fingerprint(),
        "writers": {kind: writer.fingerprint() for kind, writer in writers.items()},
    }

    reset_index = max(1, horizon // 2)
    goal_conditions = (
        "oracle",
        "causal",
        "shuffled_training",
        "swapped",
        "register_reset",
    )
    goal_vectors = {
        condition: _goal_vectors(
            probes,
            condition=condition,
            code=code,
            causal=causal_goal,
            shuffled=shuffled_goal,
            reset_index=reset_index,
        )
        for condition in goal_conditions
    }
    goal_metrics = {
        condition: _score_goal_forecasts(probes, vectors, code=code)
        for condition, vectors in goal_vectors.items()
    }

    selector_metrics: dict[str, object] = {}
    selected_actions: dict[tuple[str, str], list[int]] = {}
    for model_kind, model in outcome_models.items():
        for goal_condition, vectors in goal_vectors.items():
            metrics, selected = _score_selector(
                probes, vectors, model=model, prior=prior
            )
            selector_metrics[f"{model_kind}__{goal_condition}"] = metrics
            selected_actions[(model_kind, goal_condition)] = selected
    prior_metrics, prior_selected = _score_selector(
        probes,
        goal_vectors["oracle"],
        model=outcome_models["coupled"],
        prior=prior,
        use_prior=True,
    )
    selector_metrics["observable_action_prior"] = prior_metrics
    clone_metrics, clone_selected = _score_behavior_clone(probes, behavior_clone)
    selector_metrics["cue_context_behavior_clone_oracle_retained_cue"] = clone_metrics

    writer_interface = {
        kind: _score_writer_interface(writer, actions=code.goals)
        for kind, writer in writers.items()
    }
    end_to_end: dict[str, object] = {}
    for writer_kind, writer in writers.items():
        for goal_condition in goal_conditions:
            key = f"{writer_kind}__coupled_selector__{goal_condition}"
            end_to_end[key] = _score_end_to_end(
                probes,
                selected_actions[("coupled", goal_condition)],
                writer=writer,
                goals=code.goals,
            )
    end_to_end["coupled_disconnected__coupled_selector__causal"] = _score_end_to_end(
        probes,
        selected_actions[("coupled", "causal")],
        writer=writers["coupled"],
        goals=code.goals,
        disconnected=True,
    )
    end_to_end["coupled_writer__observable_action_prior"] = _score_end_to_end(
        probes, prior_selected, writer=writers["coupled"], goals=code.goals
    )
    end_to_end["coupled_writer__cue_context_behavior_clone_oracle_retained_cue"] = (
        _score_end_to_end(
            probes, clone_selected, writer=writers["coupled"], goals=code.goals
        )
    )

    timing_jitter: dict[str, object] = {}
    jitter_conditions = (
        (-1, "one_step_early"),
        (0, "exact_lag"),
        (1, "one_step_late"),
    )
    for offset, label in jitter_conditions:
        jittered_prefixes = [
            jittered_decision_prefix(probe.prefix, offset) for probe in probes
        ]
        jittered_vectors = [
            causal_goal.forecast(prefix) for prefix in jittered_prefixes
        ]
        jitter_goal_metrics = _score_goal_forecasts(probes, jittered_vectors, code=code)
        jitter_selector_metrics, jitter_selected = _score_selector(
            probes,
            jittered_vectors,
            model=outcome_models["coupled"],
            prior=prior,
        )
        timing_jitter[label] = {
            "decision_time_offset": offset,
            "prefix_lengths": sorted({len(prefix) for prefix in jittered_prefixes}),
            "cue_and_decision_local_suffix_unchanged": True,
            "goal_forecast": jitter_goal_metrics,
            "coupled_selector": jitter_selector_metrics,
            "coupled_writer_end_to_end": _score_end_to_end(
                probes,
                jitter_selected,
                writer=writers["coupled"],
                goals=code.goals,
            ),
        }

    fingerprints_after = {
        "causal_goal": causal_goal.fingerprint(),
        "shuffled_goal": shuffled_goal.fingerprint(),
        "outcome_models": {
            kind: model.fingerprint() for kind, model in outcome_models.items()
        },
        "observable_action_prior": prior.fingerprint(),
        "cue_context_behavior_clone": behavior_clone.fingerprint(),
        "writers": {kind: writer.fingerprint() for kind, writer in writers.items()},
    }

    exemplar = probes[0].prefix[-1]
    outcome_allocation = {
        kind: {
            "feature_dimension": model.feature_dimension,
            "parameter_count": model.parameter_count,
            "active_feature_count": model.active_feature_count(exemplar, 0),
        }
        for kind, model in outcome_models.items()
    }
    writer_allocation = {
        kind: {
            "allocated_parameters": writer.parameter_count,
            "allocated_bytes": writer.allocated_bytes,
            "raw_active_feature_count": writer.raw_active_feature_count(1),
        }
        for kind, writer in writers.items()
    }
    train_keys = {row.key for row in train}
    eval_keys = {row.key for row in probes}
    coverage = _training_coverage(train_audit, goals=code.goals)
    provenance = _provenance(seed)
    jitter_preserves_endpoints = all(
        np.array_equal(
            jittered_decision_prefix(probe.prefix, offset)[0], probe.prefix[0]
        )
        and np.array_equal(
            jittered_decision_prefix(probe.prefix, offset)[-local_window:],
            probe.prefix[-local_window:],
        )
        for probe in probes
        for offset in (-1, 0, 1)
    )

    return {
        "schema_version": 2,
        "supersedes": "hindsight_outcome_control_synthetic_v1.json",
        "task": "hindsight_outcome_proxy_to_persistent_coarse_action_writer",
        "claim_scope": (
            "synthetic observed-symbolic-cue, full-coverage, exact-lag "
            "mechanism/interface ablation; not latent-intent inference, learned "
            "valence, open-ended language, native ALM, or 500MB-chat evidence"
        ),
        "headline_findings": {
            "causal_bridge_coupled_writer": end_to_end[
                "coupled__coupled_selector__causal"
            ],
            "allowed_input_direct_behavior_clone": end_to_end[
                "coupled_writer__cue_context_behavior_clone_oracle_retained_cue"
            ],
            "decision_time_jitter_goal_accuracy": {
                label: row["goal_forecast"]["accuracy"]
                for label, row in timing_jitter.items()
            },
            "valid_reading": (
                "the three supplied interfaces can be connected on familiar, "
                "fully observed synthetic cells at the exact trained lag"
            ),
            "invalid_reading": (
                "the assay does not show latent goal inference, preference over "
                "likely futures, variable-duration persistence, unseen-effect "
                "identification, or necessity over direct behavior cloning"
            ),
        },
        "scientific_contract": {
            "learner_inputs": [
                "finite observation vectors",
                "injective observed one-hot cue identifying generator intention",
                "coarse action IDs logged by a scripted cue-following policy",
                "subsequent observation vectors",
                "engineered per-action command sequences as actuator targets",
            ],
            "withheld_from_learning": [
                "reward or valence",
                "success labels",
                "audit-only scalar intended-outcome field, but not its cue identity",
                "a field labelling actions as expert or correct",
                "semantic modality names",
                "response candidates",
            ],
            "supplied_scaffold": [
                "episode boundaries",
                "exact fixed far-future lag",
                "finite action alphabet",
                "injective synthetic observation encoder",
                "scripted stochastic policy that usually takes the matching action",
                "fully observed deterministic world-transition family",
                "finite command alphabet and deterministic action-command codebook",
            ],
            "interpretation": (
                "the future observation is the scripted policy's modal consequence; "
                "it is a hindsight compatibility proxy, not learned utility"
            ),
        },
        "limitations": [
            "the proxy has predictability but no learned valence",
            "the intention identity is directly observable as a one-hot cue",
            "the proxy inherits direction from a mostly cue-following scripted policy",
            "an allowed-input cue/context behavior clone also solves the assay",
            "the goal register is an exact timer and fails one-step timing jitter",
            "all context-action cells are observed; unlogged counterfactual effects "
            "are not identified",
            "the finite observation code, behavior process, action set, and command "
            "cycles are engineered",
            "held-out keys reuse the familiar modular transition family",
            "closed-form linear fitting is not a native local-plasticity result",
        ],
        "config": {
            "train_trajectories": train_trajectories,
            "crossover_groups": crossover_groups,
            "evaluation_episodes": len(probes),
            "horizon": horizon,
            "local_window": local_window,
            "seed": seed,
            "dominant_probability": dominant_probability,
            "cue_follow_probability": cue_follow_probability,
            "writer_table_bits": table_bits,
            "observation_dimension": code.dimension,
            "actions": code.goals,
        },
        "provenance": provenance,
        "dataset_audit": {
            "train_key_count": len(train_keys),
            "evaluation_key_count": len(eval_keys),
            "train_evaluation_key_overlap": len(train_keys & eval_keys),
            "matched_post_cue_prefixes_identical": _suffixes_identical(probes),
            "balanced_evaluation_action_zero_wrong_fraction": sum(
                int(probe.required_action != 0) for probe in probes
            )
            / len(probes),
            "training_coverage": coverage,
            "cue_context_behavior_clone": {
                "learner_inputs": [
                    "first observed cue vector",
                    "final decision-context vector",
                    "logged observed action ID",
                ],
                "oracle_retains_first_cue_until_decision": True,
                "observed_cue_context_cells": len(behavior_clone.counts),
                "capacity_matched_to_consequence_model": False,
                "purpose": (
                    "allowed-input solvability control, not a persistence mechanism"
                ),
            },
        },
        "allocation_audit": {
            "outcome_models": outcome_allocation,
            "outcome_additive_wide_matches_coupled_parameters": (
                outcome_models["additive_wide"].parameter_count
                == outcome_models["coupled"].parameter_count
            ),
            "outcome_additive_wide_matches_coupled_active_count": (
                outcome_models["additive_wide"].active_feature_count(exemplar, 0)
                == outcome_models["coupled"].active_feature_count(exemplar, 0)
            ),
            "writers": writer_allocation,
            "writer_allocations_identical": len(
                {writer.parameter_count for writer in writers.values()}
            )
            == 1,
            "writer_additive_wide_matches_coupled_active_count": (
                writers["additive_wide"].raw_active_feature_count(1)
                == writers["coupled"].raw_active_feature_count(1)
            ),
        },
        "goal_forecast": goal_metrics,
        "selector": selector_metrics,
        "writer_interface_oracle_clamped_action": writer_interface,
        "end_to_end": end_to_end,
        "decision_time_jitter_diagnostic": timing_jitter,
        "invariants": {
            "register_reset_index_after_cue_before_decision": 0 < reset_index < horizon,
            "jitter_keeps_cue_and_decision_local_suffix_unchanged": (
                jitter_preserves_endpoints
            ),
            "weights_frozen_during_evaluation": fingerprints_before
            == fingerprints_after,
            "all_command_targets_have_equal_symbol_histograms": len(
                {
                    tuple(np.bincount(action_command(action), minlength=9))
                    for action in range(code.goals)
                }
            )
            == 1,
            "no_train_eval_key_overlap": not bool(train_keys & eval_keys),
            "provenance_hashes_available": all(
                provenance[name] is not None
                for name in (
                    "experiment_source_sha256",
                    "focused_test_sha256",
                    "writer_source_sha256",
                )
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajectories", type=int, default=4096)
    parser.add_argument("--crossover-groups", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--dominant-probability", type=float, default=0.70)
    parser.add_argument("--cue-follow-probability", type=float, default=0.80)
    parser.add_argument("--table-bits", type=int, default=12)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_ablation(
        train_trajectories=args.train_trajectories,
        crossover_groups=args.crossover_groups,
        horizon=args.horizon,
        local_window=args.local_window,
        seed=args.seed,
        dominant_probability=args.dominant_probability,
        cue_follow_probability=args.cue_follow_probability,
        table_bits=args.table_bits,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
