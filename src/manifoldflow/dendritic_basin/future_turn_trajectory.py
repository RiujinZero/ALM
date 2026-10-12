"""Causal recurrent forecasts over already-encoded interaction events.

This module is deliberately below language and modality.  It accepts dense event
vectors plus explicit source and role identities, learns one transition operator,
and recursively reuses that operator to form an ordered future tape.  It does not
score bytes and it does not claim that next-event prediction is itself a goal.

The tape has a real rate bottleneck: every farther slot is projected through a
nested, lower-width subspace learned from the training events.  Multiplying a
vector by a small scalar would be reversible and is therefore not used as a
stand-in for lossiness.  This first baseline fits observed one-step transitions
only.  Recursive rollouts are evaluated but no loss is differentiated or fitted
through them, so successful fitting is not evidence that long rollouts are stable.
The decoded event is a deterministic conditional-mean baseline, not a learned
distribution over competing possible futures.
The width schedule bottlenecks only the exposed tape readouts: transition fitting
and recursive state evolution remain full-width in this first audit baseline.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np

EntityId: TypeAlias = str | int


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _validate_entity_id(value: EntityId, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{field} must be a string or integer ID")
    if isinstance(value, str) and not value:
        raise ValueError(f"{field} must not be empty")


def _readonly_copy(
    value: np.ndarray, *, dtype: np.dtype | type | None = None
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EncodedTurn:
    """One observed event in a modality-neutral interaction sequence."""

    vector: np.ndarray
    source_id: EntityId
    role_id: EntityId

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("vector must be a non-empty one-dimensional array")
        if np.any(~np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")
        _validate_entity_id(self.source_id, field="source_id")
        _validate_entity_id(self.role_id, field="role_id")
        object.__setattr__(self, "vector", _readonly_copy(vector))


@dataclass(frozen=True)
class FutureSlot:
    """One lossy slot in a predicted or oracle future tape.

    ``bottleneck_mask`` indexes coordinates in the learned PCA basis;
    ``vector`` is expressed in the original event space.  The unprojected
    forecast and recurrent state are deliberately absent from this production
    surface so a receiver cannot bypass the bottleneck accidentally.

    ``accumulated_residual_variance_heuristic`` is only a diagonal sum of
    source-conditioned one-step event residuals.  It is not propagated through
    the transition or PCA projector and is not calibrated predictive uncertainty.
    """

    horizon: int
    source_horizon: int
    information_width: int
    bottleneck_mask: np.ndarray
    vector: np.ndarray
    source_scores: np.ndarray
    role_scores: np.ndarray
    source_id: EntityId
    role_id: EntityId
    accumulated_residual_variance_heuristic: np.ndarray | None
    discarded_training_variance: float

    def __post_init__(self) -> None:
        for name in (
            "bottleneck_mask",
            "vector",
            "source_scores",
            "role_scores",
        ):
            value = getattr(self, name)
            dtype = np.bool_ if name == "bottleneck_mask" else np.float64
            object.__setattr__(self, name, _readonly_copy(value, dtype=dtype))
        variance = self.accumulated_residual_variance_heuristic
        if variance is not None:
            object.__setattr__(
                self,
                "accumulated_residual_variance_heuristic",
                _readonly_copy(variance, dtype=np.float64),
            )


@dataclass(frozen=True)
class RawFutureSlotDiagnostic:
    """Unbottlenecked rollout internals for tests and diagnostics only.

    These values must never be supplied to a production receiver: doing so
    would make the future-tape rate ablation meaningless.
    """

    horizon: int
    source_horizon: int
    raw_vector: np.ndarray
    latent_state: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_vector", _readonly_copy(self.raw_vector, dtype=np.float64)
        )
        object.__setattr__(
            self, "latent_state", _readonly_copy(self.latent_state, dtype=np.float64)
        )


@dataclass(frozen=True)
class FutureTape:
    """An ordered forecast surface.

    ``causal`` is true only for a tape rolled out from the current recurrent
    state.  An oracle tape is intentionally and visibly marked non-causal.
    """

    mode: str
    causal: bool
    observed_count: int
    slots: tuple[FutureSlot, ...]


@dataclass(frozen=True)
class FutureTrajectorySnapshot:
    """Immutable transient-state snapshot with a deterministic fingerprint."""

    state: np.ndarray
    observed_count: int
    fitted: bool
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _readonly_copy(self.state))


class FutureTurnTrajectory:
    """Shared-transition, progressively bottlenecked future-event tape.

    The recurrent state is ``[current_token, persistent_history]``.  A token
    contains a multiplicatively bound event block for every source/role pair,
    explicit source and role indicators, and a constant.  On observation,

    ``history <- memory_decay * history + token``.

    Ridge regression learns a *single* map from state to the next token.  That
    map is embedded in a square transition which enforces the history equation
    analytically.  Forecasting horizon ``k`` applies that same matrix exactly
    ``k`` times.  An arriving observation replaces the predicted current token
    and corrects the persistent history before all future slots are rolled out
    again.

    The event readout sums the source/role-bound blocks, yielding a deterministic
    conditional-mean forecast.  It does not retain a multimodal distribution over
    alternative identities or contents; callers must not claim that it does.
    """

    training_objective = "observed_one_step_ridge"
    trains_through_rollout = False
    bottlenecks_transition_dynamics = False

    def __init__(
        self,
        event_dim: int,
        *,
        source_ids: Sequence[EntityId],
        role_ids: Sequence[EntityId],
        horizon_widths: Sequence[int],
        memory_decay: float = 0.8,
        ridge: float = 1e-8,
    ) -> None:
        if isinstance(event_dim, bool) or not isinstance(event_dim, int):
            raise TypeError("event_dim must be an integer")
        if event_dim < 1:
            raise ValueError("event_dim must be positive")
        self.event_dim = int(event_dim)
        self.source_ids = self._validate_id_sequence(source_ids, field="source_ids")
        self.role_ids = self._validate_id_sequence(role_ids, field="role_ids")

        widths = tuple(horizon_widths)
        if not widths:
            raise ValueError("horizon_widths must contain at least one slot")
        if any(
            isinstance(width, bool) or not isinstance(width, int) for width in widths
        ):
            raise TypeError("horizon widths must be integers")
        if any(width < 1 or width > self.event_dim for width in widths):
            raise ValueError("horizon widths must be in [1, event_dim]")
        if any(right > left for left, right in zip(widths, widths[1:], strict=False)):
            raise ValueError("horizon widths must be non-increasing")
        self.horizon_widths = tuple(int(width) for width in widths)

        if not np.isfinite(memory_decay) or not 0.0 < memory_decay <= 1.0:
            raise ValueError("memory_decay must be finite and in (0, 1]")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        self.memory_decay = float(memory_decay)
        self.ridge = float(ridge)

        self._source_index = {
            value: index for index, value in enumerate(self.source_ids)
        }
        self._role_index = {value: index for index, value in enumerate(self.role_ids)}
        self._binding_count = len(self.source_ids) * len(self.role_ids)
        self._bound_dim = self._binding_count * self.event_dim
        self._source_slice = slice(
            self._bound_dim, self._bound_dim + len(self.source_ids)
        )
        role_start = self._source_slice.stop
        self._role_slice = slice(role_start, role_start + len(self.role_ids))
        self.token_dim = self._role_slice.stop + 1
        self.state_dim = 2 * self.token_dim

        self._state = np.zeros(self.state_dim, dtype=np.float64)
        self._observed_count = 0
        self._transition: np.ndarray | None = None
        self._projection_mean: np.ndarray | None = None
        self._projection_basis: np.ndarray | None = None
        self._basis_variance: np.ndarray | None = None
        self._residual_variance: dict[EntityId, np.ndarray] = {}

    @staticmethod
    def _validate_id_sequence(
        values: Sequence[EntityId], *, field: str
    ) -> tuple[EntityId, ...]:
        result = tuple(values)
        if not result:
            raise ValueError(f"{field} must contain at least one ID")
        for value in result:
            _validate_entity_id(value, field=field)
        if len(set(result)) != len(result):
            raise ValueError(f"{field} must not contain duplicate IDs")
        return result

    @property
    def fitted(self) -> bool:
        return self._transition is not None

    @property
    def horizon_count(self) -> int:
        return len(self.horizon_widths)

    @property
    def observed_count(self) -> int:
        return self._observed_count

    @property
    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        assert self._transition is not None
        return self._transition.copy()

    @property
    def residual_variance_by_source(self) -> dict[EntityId, np.ndarray]:
        """One-step event residual variance, separated by target source."""

        self._require_fitted()
        return {key: value.copy() for key, value in self._residual_variance.items()}

    def projection_mask(self, horizon: int) -> np.ndarray:
        """Return the retained-coordinate mask in learned PCA coordinates."""

        index = self._validate_horizon(horizon)
        mask = np.zeros(self.event_dim, dtype=np.bool_)
        mask[: self.horizon_widths[index]] = True
        return mask

    def projection_matrix(self, horizon: int) -> np.ndarray:
        """Return the rank-limited projector used by one tape slot."""

        self._require_fitted()
        index = self._validate_horizon(horizon)
        assert self._projection_basis is not None
        width = self.horizon_widths[index]
        basis = self._projection_basis[:, :width]
        return basis @ basis.T

    def _validate_horizon(self, horizon: int) -> int:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer")
        if not 1 <= horizon <= self.horizon_count:
            raise ValueError("horizon is outside the future tape")
        return horizon - 1

    def _validate_turn(self, turn: EncodedTurn) -> None:
        if not isinstance(turn, EncodedTurn):
            raise TypeError("each turn must be an EncodedTurn")
        if turn.vector.shape != (self.event_dim,):
            raise ValueError("turn vector does not match event_dim")
        if turn.source_id not in self._source_index:
            raise ValueError("turn has an unknown source_id")
        if turn.role_id not in self._role_index:
            raise ValueError("turn has an unknown role_id")

    def _token(self, turn: EncodedTurn) -> np.ndarray:
        self._validate_turn(turn)
        token = np.zeros(self.token_dim, dtype=np.float64)
        source_index = self._source_index[turn.source_id]
        role_index = self._role_index[turn.role_id]
        binding_index = source_index * len(self.role_ids) + role_index
        start = binding_index * self.event_dim
        token[start : start + self.event_dim] = turn.vector
        token[self._source_slice.start + source_index] = 1.0
        token[self._role_slice.start + role_index] = 1.0
        token[-1] = 1.0
        return token

    def _advance_observed(self, state: np.ndarray, turn: EncodedTurn) -> np.ndarray:
        token = self._token(turn)
        history = self.memory_decay * state[self.token_dim :] + token
        return np.concatenate((token, history))

    def fit(self, sequences: Iterable[Sequence[EncodedTurn]]) -> FutureTurnTrajectory:
        """Fit the shared one-step operator from complete training sequences.

        Training may use observed successors, but no training sequence is retained
        in transient state.  ``fit`` always finishes with a reset, so inference
        begins from an explicitly supplied causal prefix.  The closed-form ridge
        objective predicts the next observed token from each adjacent state only;
        it does not train through recursive multi-horizon rollout.
        """

        materialized = [tuple(sequence) for sequence in sequences]
        if not materialized:
            raise ValueError("at least one training sequence is required")
        all_turns: list[EncodedTurn] = []
        predictors: list[np.ndarray] = []
        target_tokens: list[np.ndarray] = []
        target_sources: list[EntityId] = []

        for sequence in materialized:
            if not sequence:
                raise ValueError("training sequences must not be empty")
            for turn in sequence:
                self._validate_turn(turn)
            all_turns.extend(sequence)
            state = np.zeros(self.state_dim, dtype=np.float64)
            previous: np.ndarray | None = None
            for turn in sequence:
                state = self._advance_observed(state, turn)
                if previous is not None:
                    predictors.append(previous)
                    target_tokens.append(state[: self.token_dim])
                    target_sources.append(turn.source_id)
                previous = state

        if not predictors:
            raise ValueError("training data must contain at least one adjacent pair")

        events = np.stack([turn.vector for turn in all_turns])
        self._fit_projection(events)
        x = np.stack(predictors)
        y_token = np.stack(target_tokens)
        if self.ridge == 0.0:
            token_transition = np.linalg.lstsq(x, y_token, rcond=None)[0]
        else:
            gram = x.T @ x
            regularized = gram + self.ridge * np.eye(self.state_dim)
            token_transition = np.linalg.solve(regularized, x.T @ y_token)
        transition = self._assemble_transition(token_transition)
        if np.any(~np.isfinite(transition)):
            raise FloatingPointError("ridge fit produced a non-finite transition")
        self._transition = np.asarray(transition, dtype=np.float64)
        self._fit_residual_variance(x, y_token, target_sources)
        self.reset()
        return self

    def _fit_projection(self, events: np.ndarray) -> None:
        mean = np.mean(events, axis=0)
        centered = events - mean
        # Decomposing the event_dim x event_dim covariance keeps a complete
        # basis even when there are fewer events than dimensions, without the
        # catastrophic N x N left-singular matrix requested by full SVD for a
        # tall natural-turn corpus (for example, 25K x 96).
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues, kind="stable")[::-1]
        basis = eigenvectors[:, order]
        # Eigenvectors have an arbitrary sign.  Canonicalizing it makes snapshots
        # and fingerprints stable across repeated fits in the same numeric stack.
        for column in range(basis.shape[1]):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0.0:
                basis[:, column] *= -1.0
        coordinates = centered @ basis
        self._projection_mean = np.asarray(mean, dtype=np.float64)
        self._projection_basis = np.asarray(basis, dtype=np.float64)
        self._basis_variance = np.mean(coordinates * coordinates, axis=0)

    def _fit_residual_variance(
        self,
        predictors: np.ndarray,
        target_tokens: np.ndarray,
        target_sources: Sequence[EntityId],
    ) -> None:
        assert self._transition is not None
        predicted_tokens = predictors @ self._transition[:, : self.token_dim]
        residuals = np.stack(
            [
                self._decode_token_event(target)
                - self._decode_token_event(estimate)
                for target, estimate in zip(
                    target_tokens, predicted_tokens, strict=True
                )
            ]
        )
        global_variance = np.mean(residuals * residuals, axis=0)
        variances: dict[EntityId, np.ndarray] = {}
        for source_id in self.source_ids:
            selected = np.asarray(
                [value == source_id for value in target_sources], dtype=np.bool_
            )
            if np.any(selected):
                variance = np.mean(residuals[selected] ** 2, axis=0)
            else:
                variance = global_variance
            variances[source_id] = np.asarray(variance, dtype=np.float64)
        self._residual_variance = variances

    def observe(self, turn: EncodedTurn) -> None:
        """Correct the recurrent state with one newly arrived event."""

        self._require_fitted()
        self._state = self._advance_observed(self._state, turn)
        self._observed_count += 1

    def observe_and_rollout(
        self, turn: EncodedTurn, **rollout_controls: object
    ) -> FutureTape:
        """Correct with an observation, then refine every prospective slot."""

        self.observe(turn)
        return self.predicted_tape(**rollout_controls)

    def reset(self) -> None:
        """Clear conversation state without changing learned anatomy."""

        self._state.fill(0.0)
        self._observed_count = 0

    def predicted_tape(
        self,
        *,
        transition_matrix: np.ndarray | None = None,
        reset_state: bool = False,
        horizon_permutation: Sequence[int] | None = None,
    ) -> FutureTape:
        """Roll out a strictly prefix-causal future tape.

        Controls are explicit and non-mutating: callers may supply a shuffled
        transition, start the rollout from a zeroed state, or permute which
        recursively predicted horizon occupies each fixed-rate tape slot.
        Because PCA projection is affine around the training mean, a zero-state
        compressed slot is a reset-history control, not necessarily a zero vector
        or disconnected-code control.
        """

        self._require_fitted()
        transition = self._validated_transition(transition_matrix)
        permutation = self._validated_permutation(horizon_permutation)
        state = (
            np.zeros(self.state_dim, dtype=np.float64)
            if reset_state
            else self._state.copy()
        )
        forecast_states: list[np.ndarray] = []
        forecast_residual_heuristics: list[np.ndarray] = []
        accumulated = np.zeros(self.event_dim, dtype=np.float64)
        for _ in range(self.horizon_count):
            state = state @ transition
            source_id, _, _ = self._decode_identities(state)
            accumulated = accumulated + self._residual_variance[source_id]
            forecast_states.append(state.copy())
            forecast_residual_heuristics.append(accumulated.copy())

        slots = tuple(
            self._make_slot(
                horizon=horizon,
                source_horizon=source_horizon,
                state=forecast_states[source_horizon - 1],
                residual_variance_heuristic=(
                    forecast_residual_heuristics[source_horizon - 1]
                ),
            )
            for horizon, source_horizon in enumerate(permutation, start=1)
        )
        return FutureTape(
            mode="predicted",
            causal=True,
            observed_count=0 if reset_state else self._observed_count,
            slots=slots,
        )

    def diagnostic_raw_rollout(
        self,
        *,
        transition_matrix: np.ndarray | None = None,
        reset_state: bool = False,
        horizon_permutation: Sequence[int] | None = None,
    ) -> tuple[RawFutureSlotDiagnostic, ...]:
        """Return unprojected rollout internals for audit code only.

        Production receivers must consume :meth:`predicted_tape`, whose slots
        cannot expose the information removed by the horizon bottleneck.
        """

        self._require_fitted()
        transition = self._validated_transition(transition_matrix)
        permutation = self._validated_permutation(horizon_permutation)
        state = (
            np.zeros(self.state_dim, dtype=np.float64)
            if reset_state
            else self._state.copy()
        )
        forecast_states: list[np.ndarray] = []
        for _ in range(self.horizon_count):
            state = state @ transition
            forecast_states.append(state.copy())
        return tuple(
            RawFutureSlotDiagnostic(
                horizon=horizon,
                source_horizon=source_horizon,
                raw_vector=self._decode_event(
                    forecast_states[source_horizon - 1]
                ),
                latent_state=forecast_states[source_horizon - 1],
            )
            for horizon, source_horizon in enumerate(permutation, start=1)
        )

    def oracle_tape(
        self,
        future_turns: Sequence[EncodedTurn],
        *,
        horizon_permutation: Sequence[int] | None = None,
    ) -> FutureTape:
        """Encode observed future turns for a diagnostic, explicitly non-causal."""

        self._require_fitted()
        turns = tuple(future_turns)
        if len(turns) != self.horizon_count:
            raise ValueError("oracle future must contain exactly one turn per slot")
        for turn in turns:
            self._validate_turn(turn)
        permutation = self._validated_permutation(horizon_permutation)
        slots: list[FutureSlot] = []
        for horizon, source_horizon in enumerate(permutation, start=1):
            turn = turns[source_horizon - 1]
            oracle_state = np.zeros(self.state_dim, dtype=np.float64)
            oracle_state[: self.token_dim] = self._token(turn)
            slots.append(
                self._make_slot(
                    horizon=horizon,
                    source_horizon=source_horizon,
                    state=oracle_state,
                    residual_variance_heuristic=None,
                )
            )
        return FutureTape(
            mode="oracle",
            causal=False,
            observed_count=self._observed_count,
            slots=tuple(slots),
        )

    def oracle_vs_predicted(
        self, future_turns: Sequence[EncodedTurn]
    ) -> tuple[FutureTape, FutureTape]:
        """Return causal and diagnostic surfaces without conflating their status."""

        return self.predicted_tape(), self.oracle_tape(future_turns)

    def _make_slot(
        self,
        *,
        horizon: int,
        source_horizon: int,
        state: np.ndarray,
        residual_variance_heuristic: np.ndarray | None,
    ) -> FutureSlot:
        raw = self._decode_event(state)
        vector = self._project(raw, horizon)
        source_id, role_id, scores = self._decode_identities(state)
        assert self._basis_variance is not None
        width = self.horizon_widths[horizon - 1]
        discarded = float(np.sum(self._basis_variance[width:]))
        return FutureSlot(
            horizon=horizon,
            source_horizon=source_horizon,
            information_width=width,
            bottleneck_mask=self.projection_mask(horizon),
            vector=vector,
            source_scores=scores[0],
            role_scores=scores[1],
            source_id=source_id,
            role_id=role_id,
            accumulated_residual_variance_heuristic=residual_variance_heuristic,
            discarded_training_variance=discarded,
        )

    def _decode_event(self, state: np.ndarray) -> np.ndarray:
        return self._decode_token_event(state[: self.token_dim])

    def _decode_token_event(self, token: np.ndarray) -> np.ndarray:
        # Ridge regression can activate several bound blocks at once.  Summing
        # them is the conditional-mean baseline; it must not be interpreted as a
        # sampled future or as content conditioned on the separately argmaxed ID.
        bound = token[: self._bound_dim].reshape(
            self._binding_count, self.event_dim
        )
        return np.sum(bound, axis=0)

    def _decode_identities(
        self, state: np.ndarray
    ) -> tuple[EntityId, EntityId, tuple[np.ndarray, np.ndarray]]:
        token = state[: self.token_dim]
        source_scores = token[self._source_slice]
        role_scores = token[self._role_slice]
        source_id = self.source_ids[int(np.argmax(source_scores))]
        role_id = self.role_ids[int(np.argmax(role_scores))]
        return source_id, role_id, (source_scores.copy(), role_scores.copy())

    def _project(self, vector: np.ndarray, horizon: int) -> np.ndarray:
        assert self._projection_mean is not None
        projector = self.projection_matrix(horizon)
        return self._projection_mean + projector @ (vector - self._projection_mean)

    def shuffled_transition(self, *, seed: int) -> np.ndarray:
        """Return a deterministic next-token output-coordinate shuffle control.

        The recurrent history half is reconstructed analytically after the
        shuffle, so this negative control cannot win or lose merely by violating
        the intact model's state-update equation.
        """

        self._require_fitted()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        assert self._transition is not None
        permutation = np.random.default_rng(seed).permutation(self.token_dim)
        token_transition = self._transition[:, : self.token_dim][:, permutation]
        return self._assemble_transition(token_transition)

    def _assemble_transition(self, token_transition: np.ndarray) -> np.ndarray:
        """Embed a learned next-token map in the exact recurrent state update."""

        token_map = np.asarray(token_transition, dtype=np.float64)
        if token_map.shape != (self.state_dim, self.token_dim):
            raise ValueError("next-token transition has the wrong shape")
        transition = np.empty((self.state_dim, self.state_dim), dtype=np.float64)
        transition[:, : self.token_dim] = token_map
        transition[:, self.token_dim :] = token_map
        transition[self.token_dim :, self.token_dim :] += (
            self.memory_decay * np.eye(self.token_dim)
        )
        return transition

    def _validated_transition(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            assert self._transition is not None
            return self._transition
        transition = np.asarray(value, dtype=np.float64)
        if transition.shape != (self.state_dim, self.state_dim):
            raise ValueError("transition_matrix must match state_dim by state_dim")
        if np.any(~np.isfinite(transition)):
            raise ValueError("transition_matrix must contain only finite values")
        expected = self._assemble_transition(transition[:, : self.token_dim])
        if not np.allclose(transition, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(
                "transition_matrix must preserve the analytical history update"
            )
        return transition

    def _validated_permutation(
        self, value: Sequence[int] | None
    ) -> tuple[int, ...]:
        if value is None:
            return tuple(range(1, self.horizon_count + 1))
        permutation = tuple(value)
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in permutation
        ):
            raise TypeError("horizon_permutation must contain integers")
        if sorted(permutation) != list(range(1, self.horizon_count + 1)):
            raise ValueError("horizon_permutation must permute every 1-based horizon")
        return permutation

    def snapshot(self) -> FutureTrajectorySnapshot:
        return FutureTrajectorySnapshot(
            state=self._state,
            observed_count=self._observed_count,
            fitted=self.fitted,
            fingerprint=self.fingerprint(),
        )

    def fingerprint(self, *, include_state: bool = True) -> str:
        """Hash configuration, learned arrays, and optionally transient state."""

        digest = hashlib.sha256()
        digest.update(b"future-turn-trajectory-v1\0")
        digest.update(struct.pack("<q", self.event_dim))
        digest.update(struct.pack("<d", self.memory_decay))
        digest.update(struct.pack("<d", self.ridge))
        for values in (self.source_ids, self.role_ids):
            digest.update(struct.pack("<q", len(values)))
            for value in values:
                encoded = self._encoded_id(value)
                digest.update(struct.pack("<q", len(encoded)))
                digest.update(encoded)
        digest.update(np.asarray(self.horizon_widths, dtype="<i8").tobytes())
        for array in (
            self._transition,
            self._projection_mean,
            self._projection_basis,
            self._basis_variance,
        ):
            self._update_array_digest(digest, array)
        for source_id in self.source_ids:
            self._update_array_digest(digest, self._residual_variance.get(source_id))
        if include_state:
            self._update_array_digest(digest, self._state)
            digest.update(struct.pack("<q", self._observed_count))
        return digest.hexdigest()

    @staticmethod
    def _encoded_id(value: EntityId) -> bytes:
        if isinstance(value, str):
            return b"s:" + value.encode("utf-8")
        return b"i:" + str(value).encode("ascii")

    @staticmethod
    def _update_array_digest(
        digest: _HashWriter, array: np.ndarray | None
    ) -> None:
        if array is None:
            digest.update(b"none\0")
            return
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(struct.pack("<q", contiguous.ndim))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit must be called before forecasting or observation")


__all__ = [
    "EncodedTurn",
    "FutureSlot",
    "FutureTape",
    "FutureTrajectorySnapshot",
    "FutureTurnTrajectory",
    "RawFutureSlotDiagnostic",
]
