"""Causal, surprise-grown prediction of the next vector-valued event.

This module applies the project's local assembly lifecycle to a future-event
transition.  It is intentionally *not* a dialogue component: ``context`` and
``target`` are arbitrary finite vectors, so the same mechanism can operate on
turn codes, visual-event codes, motor episodes, or another modality's event
representation.

The basal prediction is supplied by the caller and is never fitted here.  A
call to :meth:`SurpriseGrownFutureTransition.forecast` records only a causal
pending trace.  When the target later arrives, :meth:`resolve` scores that
pending prediction before structural growth and performs local NumPy updates:

* fixed-floor surprise under the pending conditional components permits a
  nursery proposal only when the existing modes represent the residual poorly;
* recurrent, target-consistent proposals become juvenile residual modes;
* competing modes receive responsibility-weighted residual/variance updates;
* strictly prequential diagonal-Gaussian mixture and leave-one-out log scores
  assign signed predictive gain over the fixed basal control;
* those causal gain EMAs control consolidation, dormancy, and recycling; and
* several residual modes may share one context assembly, preserving distinct
  futures instead of forcing their conditional mean to be the only output.

There is no objective differentiated through time, no autograd framework, and
no backpropagation.  Forecasting never changes learned parameters or anatomy.

For target dimension ``d``, the retained proper score is
``S(q, y) = -log2(sum_k w_k N(y; mean_k, diag(var_k))) / d``.  Mixture gain is
``S(basal, y) - S(full, y)``; a mode or context receives the strictly
prequential marginal gain ``S(mixture_without_it, y) - S(full, y)`` after the
remaining positive weights are renormalized.  Responsibilities are posterior
component probabilities and are used only for local parameter estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .multi_input_interaction_dendrite import MultiInputInteractionDendrite

Lifecycle = Literal["juvenile", "consolidated", "dormant"]


@dataclass(frozen=True)
class SurpriseGrownFutureTransitionConfig:
    """Fixed capacity and local-plasticity constants.

    Inputs are expected to use reasonably stable coordinate scales.  The
    learner estimates a causal residual second moment, but ``residual_floor``
    remains the minimum scale against which surprise and consistency are
    measured.  All predictive-gain fields and lifecycle gain thresholds are
    measured in bits per target coordinate.
    """

    context_dim: int
    target_dim: int
    max_context_assemblies: int = 64
    max_residual_modes: int = 128
    max_modes_per_context: int = 4
    max_nursery_candidates: int = 128
    context_bandwidth: float = 1.0
    min_context_affinity: float = 0.50
    # Dendritic context assembly parameters (set k_inputs_per_branch > 0 to
    # replace linear prototype matching with nonlinear K-input dendrites).
    k_inputs_per_branch: int = 0
    dendrite_initial_sigma: float = 1.0
    dendrite_step_size: float = 0.10
    dendrite_adapt_every: int = 16
    dendrite_buffer_size: int = 64
    residual_floor: float = 0.10
    residual_scale_decay: float = 0.99
    surprise_birth_z: float = 3.0
    poor_representation_z: float = 2.0
    nursery_context_z: float = 1.0
    nursery_residual_z: float = 1.0
    nursery_confirmations: int = 3
    nursery_target_consistency: float = 0.80
    nursery_max_age: int = 256
    residual_learning_rate: float = 0.15
    variance_learning_rate: float = 0.08
    context_learning_rate: float = 0.05
    utility_decay: float = 0.98
    responsibility_fire_threshold: float = 0.20
    basal_mixture_weight: float = 1.0
    juvenile_forecast_weight: float = 0.25
    newborn_exploration_weight: float = 0.25
    causal_support_prior: float = 4.0
    consolidation_min_age: int = 32
    consolidation_min_fires: int = 8
    consolidation_gain_bits: float = 0.02
    dormancy_staleness: int = 256
    prune_gain_bits: float = -0.02
    dormant_min_residence: int = 16
    juvenile_min_recycle_age: int = 64
    dormant_forecast_weight: float = 0.05

    def __post_init__(self) -> None:
        # Comparisons with NaN are false, so reject non-finite values before
        # the range checks below.  Every current field is numeric.
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not finite:
                raise ValueError(f"{name} must be finite")
        for name in (
            "context_dim",
            "target_dim",
            "max_context_assemblies",
            "max_residual_modes",
            "max_modes_per_context",
            "max_nursery_candidates",
            "nursery_confirmations",
            "nursery_max_age",
            "consolidation_min_age",
            "consolidation_min_fires",
            "dormancy_staleness",
            "dormant_min_residence",
            "juvenile_min_recycle_age",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.k_inputs_per_branch < 0:
            raise ValueError("k_inputs_per_branch must be >= 0")
        if self.k_inputs_per_branch > 0 and self.k_inputs_per_branch > self.context_dim:
            raise ValueError("k_inputs_per_branch cannot exceed context_dim")
        if self.max_modes_per_context > self.max_residual_modes:
            raise ValueError("max_modes_per_context cannot exceed max_residual_modes")
        for name in (
            "context_bandwidth",
            "residual_floor",
            "surprise_birth_z",
            "poor_representation_z",
            "nursery_context_z",
            "nursery_residual_z",
            "basal_mixture_weight",
            "juvenile_forecast_weight",
            "newborn_exploration_weight",
            "causal_support_prior",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "min_context_affinity",
            "residual_scale_decay",
            "nursery_target_consistency",
            "residual_learning_rate",
            "variance_learning_rate",
            "context_learning_rate",
            "utility_decay",
            "responsibility_fire_threshold",
            "juvenile_forecast_weight",
            "newborn_exploration_weight",
            "dormant_forecast_weight",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class FutureModeCandidate:
    """One explicit conditional future retained by the transition."""

    mode_id: int
    context_id: int
    prediction: np.ndarray
    weight: float
    lifecycle: Lifecycle


@dataclass(frozen=True)
class FutureTransitionForecast:
    """Point forecast plus the non-averaged future modes behind it."""

    prediction: np.ndarray
    basal_prediction: np.ndarray
    basal_weight: float
    candidates: tuple[FutureModeCandidate, ...]


@dataclass(frozen=True)
class FutureTransitionUpdate:
    """Pre-growth diagnostics for one causally resolved prediction."""

    prediction_loss: float
    basal_loss: float
    basal_nll_bits_per_coordinate: float
    mixture_nll_bits_per_coordinate: float
    mixture_gain_bits_per_coordinate: float
    standardized_surprise: float
    best_representation_z: float
    poor_representation: bool
    scored_mode_ids: tuple[int, ...]
    responsibilities: tuple[float, ...]
    unclaimed_responsibility: float
    mode_marginal_gain_bits_per_coordinate: tuple[float, ...]
    context_marginal_gain_bits_per_coordinate: tuple[tuple[int, float], ...]
    born_mode: int | None
    recycled_modes: tuple[int, ...]

    @property
    def mode_marginal_gain_bits(self) -> tuple[float, ...]:
        """Backward-compatible short name; values are bits/coordinate."""

        return self.mode_marginal_gain_bits_per_coordinate


@dataclass(frozen=True)
class FutureResidualModeState:
    """Read-only public view of one locally learned residual mode."""

    mode_id: int
    context_id: int
    residual_mean: np.ndarray | None
    residual_variance: np.ndarray
    observations: float
    fires: int
    causal_scores: int
    predictive_gain_bits: float
    lifecycle: Lifecycle
    born_step: int
    last_fire_step: int

    @property
    def predictive_gain_bits_per_coordinate(self) -> float:
        return self.predictive_gain_bits


@dataclass(frozen=True)
class FutureContextAssemblyState:
    """Auditable context-level causal utility and owned modes."""

    context_id: int
    prototype: np.ndarray
    mode_ids: tuple[int, ...]
    observations: float
    causal_scores: int
    predictive_gain_bits: float

    @property
    def predictive_gain_bits_per_coordinate(self) -> float:
        return self.predictive_gain_bits


@dataclass
class _ContextAssembly:
    context_id: int
    prototype: np.ndarray
    observations: float
    mode_ids: list[int]
    causal_scores: int = 0
    predictive_gain_bits: float = 0.0
    # Optional: nonlinear K-input dendrite for context matching.
    # When present it replaces the linear Euclidean affinity.
    dendrite: MultiInputInteractionDendrite | None = None
    _obs_since_adapt: int = 0


@dataclass
class _ResidualMode:
    mode_id: int
    context_id: int
    residual_mean: np.ndarray | None  # None when predictor is present (k > 0)
    residual_variance: np.ndarray
    observations: float
    fires: int
    causal_scores: int
    predictive_gain_bits: float
    lifecycle: Lifecycle
    born_step: int
    last_fire_step: int
    dormant_step: int = -1
    predictor: MultiInputInteractionDendrite | None = None
    _obs_since_pred_adapt: int = 0


@dataclass
class _NurseryCandidate:
    context_mean: np.ndarray
    residual_mean: np.ndarray
    residual_m2: np.ndarray
    confirmations: int
    first_seen_step: int
    last_seen_step: int


@dataclass(frozen=True)
class _PendingForecast:
    context: np.ndarray
    basal_prediction: np.ndarray
    prediction: np.ndarray
    mode_ids: tuple[int, ...]
    mode_context_ids: tuple[int, ...]
    basal_weight: float
    mode_weights: np.ndarray
    basal_variance: np.ndarray
    mode_means: np.ndarray
    mode_variances: np.ndarray


class SurpriseGrownFutureTransition:
    """Bounded online residual transition using only local learning rules.

    The causal protocol is deliberately split in two::

        forecast = learner.forecast(context_t, fixed_basal_prediction_t)
        update = learner.resolve(observed_vector_t_plus_1)

    ``resolve`` has no context argument, so a future target cannot influence
    the state that predicted it.  Use ``remember=False`` for inference-only
    forecasts that must not replace or create a pending training trace.
    """

    def __init__(self, config: SurpriseGrownFutureTransitionConfig, seed: int = 0):
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._contexts: list[_ContextAssembly | None] = [
            None
        ] * config.max_context_assemblies
        self._modes: list[_ResidualMode | None] = [None] * config.max_residual_modes
        self._nursery: list[_NurseryCandidate] = []
        self._pending: _PendingForecast | None = None
        self._residual_second_moment = np.full(
            config.target_dim,
            config.residual_floor**2,
            dtype=np.float64,
        )
        self.step = 0
        self.predictions_scored = 0
        self.births = 0
        self.consolidations = 0
        self.dormancies = 0
        self.reactivations = 0
        self.recycles = 0
        self.mixture_predictive_gain_bits = 0.0

    @property
    def pending(self) -> bool:
        """Whether a causal forecast is waiting for its later target."""

        return self._pending is not None

    @property
    def nursery_count(self) -> int:
        return len(self._nursery)

    @property
    def context_count(self) -> int:
        return sum(context is not None for context in self._contexts)

    @property
    def mixture_predictive_gain_bits_per_coordinate(self) -> float:
        """EMA proper-score gain of the emitted mixture over basal."""

        return self.mixture_predictive_gain_bits

    @property
    def context_states(self) -> tuple[FutureContextAssemblyState, ...]:
        """Copies of context state, including prequential marginal utility."""

        return tuple(
            FutureContextAssemblyState(
                context_id=context.context_id,
                prototype=context.prototype.copy(),
                mode_ids=tuple(context.mode_ids),
                observations=context.observations,
                causal_scores=context.causal_scores,
                predictive_gain_bits=context.predictive_gain_bits,
            )
            for context in self._contexts
            if context is not None
        )

    @property
    def mode_states(self) -> tuple[FutureResidualModeState, ...]:
        """Copies of learned mode state, suitable for diagnostics/tests."""

        return tuple(
            FutureResidualModeState(
                mode_id=mode.mode_id,
                context_id=mode.context_id,
                residual_mean=mode.residual_mean.copy() if mode.residual_mean is not None else None,
                residual_variance=mode.residual_variance.copy(),
                observations=mode.observations,
                fires=mode.fires,
                causal_scores=mode.causal_scores,
                predictive_gain_bits=mode.predictive_gain_bits,
                lifecycle=mode.lifecycle,
                born_step=mode.born_step,
                last_fire_step=mode.last_fire_step,
            )
            for mode in self._modes
            if mode is not None
        )

    def modes_for_context(self, context_id: int) -> tuple[int, ...]:
        """Return all residual modes owned by one context assembly."""

        if not 0 <= context_id < len(self._contexts):
            raise ValueError("context_id outside configured capacity")
        context = self._contexts[context_id]
        return () if context is None else tuple(context.mode_ids)

    def _vector(self, value: np.ndarray, dim: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector.copy()

    def _context_z(self, context: np.ndarray, prototype: np.ndarray) -> float:
        return float(
            np.sqrt(np.mean(np.square(context - prototype)))
            / self.config.context_bandwidth
        )

    def _context_affinity(
        self, context: np.ndarray, assembly: _ContextAssembly
    ) -> float:
        if assembly.dendrite is not None:
            return assembly.dendrite.respond(context)
        z = self._context_z(context, assembly.prototype)
        return math.exp(-0.5 * z * z)

    def _mode_representation_z(
        self, residual: np.ndarray, mode: _ResidualMode, mean: np.ndarray
    ) -> float:
        variance = np.maximum(mode.residual_variance, self.config.residual_floor**2)
        return float(
            np.sqrt(np.mean(np.square(residual - mean) / variance))
        )

    def _eligible_modes(
        self, context: np.ndarray
    ) -> tuple[list[_ResidualMode], np.ndarray]:
        modes: list[_ResidualMode] = []
        raw_weights: list[float] = []
        for assembly in self._contexts:
            if assembly is None:
                continue
            affinity = self._context_affinity(context, assembly)
            if affinity < self.config.min_context_affinity:
                continue
            for mode_id in assembly.mode_ids:
                mode = self._modes[mode_id]
                if mode is None:
                    continue
                lifecycle_weight = (
                    self.config.dormant_forecast_weight
                    if mode.lifecycle == "dormant"
                    else (
                        self.config.juvenile_forecast_weight
                        if mode.lifecycle == "juvenile"
                        else 1.0
                    )
                )
                support = mode.observations / (
                    mode.observations + self.config.causal_support_prior
                )
                support = (
                    self.config.newborn_exploration_weight
                    + (1.0 - self.config.newborn_exploration_weight) * support
                )
                raw_weight = affinity * lifecycle_weight * support
                if raw_weight <= 0.0:
                    continue
                modes.append(mode)
                raw_weights.append(raw_weight)
        if not modes:
            return [], np.empty(0, dtype=np.float64)
        return modes, np.asarray(raw_weights, dtype=np.float64)

    def forecast(
        self,
        context: np.ndarray,
        basal_prediction: np.ndarray,
        *,
        remember: bool = True,
    ) -> FutureTransitionForecast:
        """Forecast the next event without learning or changing anatomy.

        ``basal_prediction`` is an externally computed fixed-control output.
        This method only reads learned assemblies.  With ``remember=True`` it
        stores the causal trace needed for a later :meth:`resolve`; storing
        that trace is not a parameter update.  An unresolved trace cannot be
        overwritten accidentally.
        """

        context_vector = self._vector(context, self.config.context_dim, "context")
        basal_vector = self._vector(
            basal_prediction, self.config.target_dim, "basal_prediction"
        )
        if remember and self._pending is not None:
            raise RuntimeError("resolve the pending forecast before replacing it")

        modes, raw_mode_weights = self._eligible_modes(context_vector)
        total_weight = self.config.basal_mixture_weight + float(raw_mode_weights.sum())
        basal_weight = self.config.basal_mixture_weight / total_weight
        mode_weights = raw_mode_weights / total_weight
        def _mode_residual(mode: _ResidualMode) -> np.ndarray:
            if mode.predictor is not None:
                return mode.predictor.predict(context_vector)
            if mode.residual_mean is not None:
                return mode.residual_mean
            return np.zeros(self.config.target_dim, dtype=np.float64)

        if modes:
            residual = sum(
                (
                    weight * _mode_residual(mode)
                    for weight, mode in zip(mode_weights, modes, strict=True)
                ),
                np.zeros(self.config.target_dim, dtype=np.float64),
            )
            prediction = basal_vector + residual
        else:
            prediction = basal_vector.copy()
        candidates = tuple(
            FutureModeCandidate(
                mode_id=mode.mode_id,
                context_id=mode.context_id,
                prediction=(basal_vector + _mode_residual(mode)).copy(),
                weight=float(weight),
                lifecycle=mode.lifecycle,
            )
            for weight, mode in zip(mode_weights, modes, strict=True)
        )
        if remember:
            if modes:
                mode_means = np.stack(
                    [basal_vector + _mode_residual(mode) for mode in modes]
                )
                mode_variances = np.stack(
                    [mode.residual_variance.copy() for mode in modes]
                )
            else:
                mode_means = np.empty((0, self.config.target_dim), dtype=np.float64)
                mode_variances = np.empty_like(mode_means)
            self._pending = _PendingForecast(
                context=context_vector,
                basal_prediction=basal_vector,
                prediction=prediction.copy(),
                mode_ids=tuple(mode.mode_id for mode in modes),
                mode_context_ids=tuple(mode.context_id for mode in modes),
                basal_weight=float(basal_weight),
                mode_weights=mode_weights.copy(),
                basal_variance=np.maximum(
                    self._residual_second_moment,
                    self.config.residual_floor**2,
                ).copy(),
                mode_means=mode_means,
                mode_variances=mode_variances,
            )
        return FutureTransitionForecast(
            prediction=prediction.copy(),
            basal_prediction=basal_vector.copy(),
            basal_weight=float(basal_weight),
            candidates=candidates,
        )

    @staticmethod
    def _logsumexp(values: np.ndarray) -> float:
        maximum = float(values.max())
        return maximum + math.log(float(np.exp(values - maximum).sum()))

    @staticmethod
    def _gaussian_log_density(
        target: np.ndarray, mean: np.ndarray, variance: np.ndarray
    ) -> float:
        return float(
            -0.5
            * np.sum(
                np.log(2.0 * math.pi * variance) + np.square(target - mean) / variance
            )
        )

    def _proper_scores(
        self, pending: _PendingForecast, target: np.ndarray
    ) -> tuple[
        float,
        float,
        np.ndarray,
        float,
        float,
        np.ndarray,
        dict[int, float],
    ]:
        """Score only the components and weights captured at forecast time.

        Returns basal and full-mixture NLL in bits/coordinate, posterior mode
        responsibilities, basal posterior responsibility, best mode
        representation z, leave-one-mode-out gains, and leave-one-context-out
        gains.  Every leave-out mixture is renormalized over its remaining
        positive mass, which always includes the basal component.
        """

        basal_log_density = self._gaussian_log_density(
            target, pending.basal_prediction, pending.basal_variance
        )
        if pending.mode_ids:
            mode_log_densities = np.array(
                [
                    self._gaussian_log_density(target, mean, variance)
                    for mean, variance in zip(
                        pending.mode_means,
                        pending.mode_variances,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            z_values = np.sqrt(
                np.mean(
                    np.square(target[None, :] - pending.mode_means)
                    / pending.mode_variances,
                    axis=1,
                )
            )
            best_z = float(z_values.min())
        else:
            mode_log_densities = np.empty(0, dtype=np.float64)
            best_z = math.inf
        component_log_densities = np.concatenate(
            ([basal_log_density], mode_log_densities)
        )
        component_weights = np.concatenate(
            ([pending.basal_weight], pending.mode_weights)
        )
        log_joint = np.log(component_weights) + component_log_densities
        full_log_density = self._logsumexp(log_joint)
        denominator = math.log(2.0) * self.config.target_dim
        basal_nll = -basal_log_density / denominator
        mixture_nll = -full_log_density / denominator
        posterior = np.exp(log_joint - full_log_density)

        def leave_out(indices: tuple[int, ...]) -> float:
            keep = np.ones(len(component_weights), dtype=bool)
            for index in indices:
                keep[index] = False
            remaining_weights = component_weights[keep]
            remaining_weights /= remaining_weights.sum()
            remaining_log_density = self._logsumexp(
                np.log(remaining_weights) + component_log_densities[keep]
            )
            gain = -remaining_log_density / denominator - mixture_nll
            return 0.0 if abs(gain) < 1e-15 else gain

        mode_marginal_gains = np.array(
            [leave_out((index + 1,)) for index in range(len(pending.mode_ids))],
            dtype=np.float64,
        )
        context_marginal_gains: dict[int, float] = {}
        for context_id in dict.fromkeys(pending.mode_context_ids):
            positions = tuple(
                index + 1
                for index, owner in enumerate(pending.mode_context_ids)
                if owner == context_id
            )
            context_marginal_gains[context_id] = leave_out(positions)
        return (
            basal_nll,
            mixture_nll,
            posterior[1:],
            float(posterior[0]),
            best_z,
            mode_marginal_gains,
            context_marginal_gains,
        )

    def _update_modes(
        self,
        pending: _PendingForecast,
        residual: np.ndarray,
        responsibilities: np.ndarray,
        mixture_gain: float,
        mode_marginal_gains: np.ndarray,
        context_marginal_gains: dict[int, float],
    ) -> None:
        context_credit: dict[int, float] = {}
        decay = self.config.utility_decay
        self.mixture_predictive_gain_bits = (
            decay * self.mixture_predictive_gain_bits + (1.0 - decay) * mixture_gain
        )
        for mode_id, responsibility, marginal_gain in zip(
            pending.mode_ids,
            responsibilities,
            mode_marginal_gains,
            strict=True,
        ):
            mode = self._modes[mode_id]
            assert mode is not None
            mode.predictive_gain_bits = decay * mode.predictive_gain_bits + (
                1.0 - decay
            ) * float(marginal_gain)
            mode.causal_scores += 1
            mode.fires += 1
            mode.last_fire_step = self.step

            # Posterior responsibility is restricted to local parameter
            # estimation.  It never supplies survival credit.
            local_responsibility = (
                float(responsibility)
                if responsibility >= self.config.responsibility_fire_threshold
                else 0.0
            )
            mean_rate = self.config.residual_learning_rate * local_responsibility
            variance_rate = self.config.variance_learning_rate * local_responsibility
            if mode.predictor is not None:
                pred = mode.predictor.predict(pending.context)
                delta = residual - pred
                mode.predictor.adapt_output(pending.context, delta, mean_rate)
                mode.predictor.observe(pending.context)
                mode._obs_since_pred_adapt += 1
                if mode._obs_since_pred_adapt >= self.config.dendrite_adapt_every:
                    mode.predictor.adapt(self.config.dendrite_step_size)
                    mode._obs_since_pred_adapt = 0
            else:
                if mode.residual_mean is not None:
                    delta = residual - mode.residual_mean
                    mode.residual_mean += mean_rate * delta
                else:
                    delta = residual
            mode.residual_variance += variance_rate * (
                np.square(delta) - mode.residual_variance
            )
            mode.residual_variance = np.maximum(
                mode.residual_variance, self.config.residual_floor**2
            )
            mode.observations += local_responsibility
            if (
                mode.lifecycle == "dormant"
                and marginal_gain > self.config.prune_gain_bits
            ):
                mode.lifecycle = "consolidated"
                mode.dormant_step = -1
                self.reactivations += 1
            context_credit[mode.context_id] = (
                context_credit.get(mode.context_id, 0.0) + local_responsibility
            )

        for context_id, raw_credit in context_credit.items():
            assembly = self._contexts[context_id]
            assert assembly is not None
            marginal_gain = context_marginal_gains[context_id]
            assembly.predictive_gain_bits = (
                decay * assembly.predictive_gain_bits + (1.0 - decay) * marginal_gain
            )
            assembly.causal_scores += 1
            credit = min(raw_credit, 1.0)
            if assembly.dendrite is not None:
                assembly.dendrite.observe(pending.context)
                assembly._obs_since_adapt += 1
                if assembly._obs_since_adapt >= self.config.dendrite_adapt_every:
                    assembly.dendrite.adapt(self.config.dendrite_step_size)
                    assembly._obs_since_adapt = 0
            else:
                rate = self.config.context_learning_rate * credit
                assembly.prototype += rate * (pending.context - assembly.prototype)
            assembly.observations += credit

    def _candidate_z(
        self,
        candidate: _NurseryCandidate,
        context: np.ndarray,
        residual: np.ndarray,
    ) -> tuple[float, float]:
        context_z = self._context_z(context, candidate.context_mean)
        residual_z = float(
            np.sqrt(
                np.mean(
                    np.square(residual - candidate.residual_mean)
                    / self.config.residual_floor**2
                )
            )
        )
        return context_z, residual_z

    def _candidate_consistency(self, candidate: _NurseryCandidate) -> float:
        if candidate.confirmations <= 1:
            return 1.0
        variance = candidate.residual_m2 / (candidate.confirmations - 1)
        normalized_spread = float(
            np.sqrt(np.mean(variance / self.config.residual_floor**2))
        )
        return math.exp(-normalized_spread)

    def _mode_is_recyclable(self, mode: _ResidualMode) -> bool:
        if mode.lifecycle == "consolidated":
            return False
        if mode.lifecycle == "dormant":
            return self.step - mode.dormant_step >= self.config.dormant_min_residence
        age = self.step - mode.born_step
        stale = self.step - mode.last_fire_step >= self.config.dormancy_staleness
        weak = mode.predictive_gain_bits < self.config.prune_gain_bits
        return age >= self.config.juvenile_min_recycle_age and (stale or weak)

    def _make_context_assembly(
        self, context_id: int, context_mean: np.ndarray
    ) -> _ContextAssembly:
        dendrite: MultiInputInteractionDendrite | None = None
        k = self.config.k_inputs_per_branch
        if k > 0:
            dendrite = MultiInputInteractionDendrite.from_birth_context(
                context_mean=context_mean,
                k=k,
                initial_sigma=self.config.dendrite_initial_sigma,
                buffer_size=self.config.dendrite_buffer_size,
                rng=self._rng,
            )
        return _ContextAssembly(
            context_id=context_id,
            prototype=context_mean.copy(),
            observations=0.0,
            mode_ids=[],
            dendrite=dendrite,
        )

    def _drop_mode(self, mode_id: int) -> None:
        mode = self._modes[mode_id]
        if mode is None:
            return
        context = self._contexts[mode.context_id]
        if context is not None and mode_id in context.mode_ids:
            context.mode_ids.remove(mode_id)
        self._modes[mode_id] = None

    def _context_for_candidate(
        self, candidate: _NurseryCandidate
    ) -> tuple[int | None, list[int]]:
        best: tuple[float, _ContextAssembly] | None = None
        for assembly in self._contexts:
            if assembly is None:
                continue
            affinity = self._context_affinity(candidate.context_mean, assembly)
            if affinity < self.config.min_context_affinity:
                continue
            if best is None or affinity > best[0]:
                best = (affinity, assembly)
        if best is not None:
            return best[1].context_id, []

        for context_id, assembly in enumerate(self._contexts):
            if assembly is None or not assembly.mode_ids:
                self._contexts[context_id] = self._make_context_assembly(
                    context_id, candidate.context_mean
                )
                return context_id, []

        reclaimable: list[tuple[float, int, list[int]]] = []
        for assembly in self._contexts:
            assert assembly is not None
            modes = [self._modes[mode_id] for mode_id in assembly.mode_ids]
            concrete = [mode for mode in modes if mode is not None]
            if concrete and all(self._mode_is_recyclable(mode) for mode in concrete):
                score = assembly.predictive_gain_bits + sum(
                    mode.predictive_gain_bits for mode in concrete
                )
                reclaimable.append(
                    (score, assembly.context_id, [mode.mode_id for mode in concrete])
                )
        if not reclaimable:
            return None, []
        _score, context_id, recycled = min(reclaimable)
        for mode_id in recycled:
            self._drop_mode(mode_id)
        self._contexts[context_id] = self._make_context_assembly(
            context_id, candidate.context_mean
        )
        return context_id, recycled

    def _mode_slot_for_context(self, context_id: int) -> tuple[int | None, int | None]:
        assembly = self._contexts[context_id]
        assert assembly is not None
        for mode_id, mode in enumerate(self._modes):
            if (
                mode is None
                and len(assembly.mode_ids) < self.config.max_modes_per_context
            ):
                return mode_id, None

        candidates = [
            mode
            for mode in self._modes
            if mode is not None and self._mode_is_recyclable(mode)
        ]
        if len(assembly.mode_ids) >= self.config.max_modes_per_context:
            candidates = [mode for mode in candidates if mode.context_id == context_id]
        if not candidates:
            return None, None
        recycled = min(
            candidates,
            key=lambda mode: (
                0 if mode.lifecycle == "dormant" else 1,
                mode.predictive_gain_bits,
                mode.last_fire_step,
            ),
        )
        mode_id = recycled.mode_id
        self._drop_mode(mode_id)
        return mode_id, mode_id

    def _promote_candidate(
        self, candidate: _NurseryCandidate
    ) -> tuple[int | None, tuple[int, ...]]:
        context_id, context_recycles = self._context_for_candidate(candidate)
        if context_id is None:
            return None, ()
        mode_id, mode_recycle = self._mode_slot_for_context(context_id)
        if mode_id is None:
            # A newly allocated empty context should not consume capacity when
            # no mode can inhabit it.
            context = self._contexts[context_id]
            if context is not None and not context.mode_ids:
                self._contexts[context_id] = None
            return None, tuple(context_recycles)

        confirmations = candidate.confirmations
        variance = candidate.residual_m2 / max(confirmations - 1, 1)
        predictor: MultiInputInteractionDendrite | None = None
        k = self.config.k_inputs_per_branch
        if k > 0:
            predictor = MultiInputInteractionDendrite.from_birth_context(
                context_mean=candidate.context_mean,
                k=k,
                initial_sigma=self.config.dendrite_initial_sigma,
                buffer_size=self.config.dendrite_buffer_size,
                rng=self._rng,
                output_weights=candidate.residual_mean.copy(),
            )
        mode = _ResidualMode(
            mode_id=mode_id,
            context_id=context_id,
            residual_mean=None if predictor is not None else candidate.residual_mean.copy(),
            residual_variance=np.maximum(variance, self.config.residual_floor**2),
            observations=0.0,
            fires=0,
            causal_scores=0,
            predictive_gain_bits=0.0,
            lifecycle="juvenile",
            born_step=self.step,
            last_fire_step=self.step,
            predictor=predictor,
        )
        self._modes[mode_id] = mode
        context = self._contexts[context_id]
        assert context is not None
        context.mode_ids.append(mode_id)
        self.births += 1
        recycled = list(context_recycles)
        if mode_recycle is not None and mode_recycle not in recycled:
            recycled.append(mode_recycle)
        self.recycles += len(recycled)
        return mode_id, tuple(recycled)

    def _nursery_observe(
        self, context: np.ndarray, residual: np.ndarray
    ) -> tuple[int | None, tuple[int, ...]]:
        self._nursery = [
            candidate
            for candidate in self._nursery
            if self.step - candidate.last_seen_step <= self.config.nursery_max_age
        ]
        matches: list[tuple[float, int]] = []
        for index, candidate in enumerate(self._nursery):
            context_z, residual_z = self._candidate_z(candidate, context, residual)
            if (
                context_z <= self.config.nursery_context_z
                and residual_z <= self.config.nursery_residual_z
            ):
                matches.append((context_z + residual_z, index))
        if matches:
            _distance, index = min(matches)
            candidate = self._nursery[index]
            candidate_index = index
            candidate.confirmations += 1
            rate = 1.0 / candidate.confirmations
            context_delta = context - candidate.context_mean
            residual_delta = residual - candidate.residual_mean
            candidate.context_mean += rate * context_delta
            candidate.residual_mean += rate * residual_delta
            candidate.residual_m2 += residual_delta * (
                residual - candidate.residual_mean
            )
            candidate.last_seen_step = self.step
        else:
            if len(self._nursery) >= self.config.max_nursery_candidates:
                evict = min(
                    range(len(self._nursery)),
                    key=lambda item: (
                        self._nursery[item].confirmations,
                        self._nursery[item].last_seen_step,
                    ),
                )
                del self._nursery[evict]
            candidate = _NurseryCandidate(
                context_mean=context.copy(),
                residual_mean=residual.copy(),
                residual_m2=np.zeros(self.config.target_dim, dtype=np.float64),
                confirmations=1,
                first_seen_step=self.step,
                last_seen_step=self.step,
            )
            self._nursery.append(candidate)
            candidate_index = len(self._nursery) - 1

        if candidate.confirmations < self.config.nursery_confirmations:
            return None, ()
        if (
            self._candidate_consistency(candidate)
            < self.config.nursery_target_consistency
        ):
            return None, ()
        born, recycled = self._promote_candidate(candidate)
        if born is not None:
            del self._nursery[candidate_index]
        return born, recycled

    def _sweep_lifecycle(self) -> None:
        for mode in self._modes:
            if mode is None:
                continue
            age = self.step - mode.born_step
            stale = self.step - mode.last_fire_step
            if (
                mode.lifecycle == "juvenile"
                and age >= self.config.consolidation_min_age
                and mode.causal_scores >= 1
                and mode.fires >= self.config.consolidation_min_fires
                and mode.predictive_gain_bits >= self.config.consolidation_gain_bits
            ):
                mode.lifecycle = "consolidated"
                self.consolidations += 1
            elif mode.lifecycle == "consolidated" and (
                stale >= self.config.dormancy_staleness
                or mode.predictive_gain_bits < self.config.prune_gain_bits
            ):
                mode.lifecycle = "dormant"
                mode.dormant_step = self.step
                self.dormancies += 1

    def resolve(
        self, target: np.ndarray, *, learn: bool = True
    ) -> FutureTransitionUpdate:
        """Score the pending forecast, then optionally perform local learning.

        The target is deliberately accepted only after ``forecast`` captured
        its context.  Surprise, representation quality, losses, and mode
        responsibilities are all computed against the pre-growth population.
        A mode born from this target therefore cannot receive credit for
        having predicted the same target.
        """

        if self._pending is None:
            raise RuntimeError("forecast(..., remember=True) must precede resolve")
        pending = self._pending
        self._pending = None
        target_vector = self._vector(target, self.config.target_dim, "target")
        residual = target_vector - pending.basal_prediction
        basal_loss = float(np.mean(np.square(residual)))
        prediction_loss = float(np.mean(np.square(target_vector - pending.prediction)))
        (
            basal_nll,
            mixture_nll,
            responsibilities,
            unclaimed,
            best_z,
            mode_marginal_gains,
            context_marginal_gains,
        ) = self._proper_scores(pending, target_vector)
        # Structural surprise is the fixed-floor distance to the closest
        # component of the *pending conditional predictor*, not distance to a
        # hindsight-fitted component and not globally scaled basal residual.
        # Using components also avoids treating the mean between two narrow
        # future modes as though it were itself a represented future.
        basal_structural_z = float(np.sqrt(basal_loss) / self.config.residual_floor)
        surprise = min(basal_structural_z, best_z)
        mixture_gain = basal_nll - mixture_nll
        if abs(mixture_gain) < 1e-15:
            mixture_gain = 0.0
        poor = best_z > self.config.poor_representation_z
        scored_ids = pending.mode_ids
        born: int | None = None
        recycled: tuple[int, ...] = ()

        self.predictions_scored += 1
        if learn:
            # The resolved event owns this clock tick.  Updating it before
            # causal participants records last_fire_step makes stale == 0 for
            # a mode that just participated.
            self.step += 1
            self._update_modes(
                pending,
                residual,
                responsibilities,
                mixture_gain,
                mode_marginal_gains,
                context_marginal_gains,
            )
            if surprise >= self.config.surprise_birth_z and poor:
                born, recycled = self._nursery_observe(pending.context, residual)
            decay = self.config.residual_scale_decay
            self._residual_second_moment = decay * self._residual_second_moment + (
                1.0 - decay
            ) * np.square(residual)
            self._sweep_lifecycle()

        return FutureTransitionUpdate(
            prediction_loss=prediction_loss,
            basal_loss=basal_loss,
            basal_nll_bits_per_coordinate=basal_nll,
            mixture_nll_bits_per_coordinate=mixture_nll,
            mixture_gain_bits_per_coordinate=mixture_gain,
            standardized_surprise=surprise,
            best_representation_z=best_z,
            poor_representation=poor,
            scored_mode_ids=scored_ids,
            responsibilities=tuple(float(item) for item in responsibilities),
            unclaimed_responsibility=float(unclaimed),
            mode_marginal_gain_bits_per_coordinate=tuple(
                float(item) for item in mode_marginal_gains
            ),
            context_marginal_gain_bits_per_coordinate=tuple(
                (context_id, float(gain))
                for context_id, gain in context_marginal_gains.items()
            ),
            born_mode=born,
            recycled_modes=recycled,
        )
