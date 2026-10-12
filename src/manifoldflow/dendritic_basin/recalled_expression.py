"""Local cortical expression synapses for recalled motif assemblies.

The episodic field decides *which* assembly settles.  This module models a
different operation at the receiving cortex: a recalled assembly arriving on
one apical receptor can bias the next sensory opponent cells.  The bias is
learned only after an external observation arrives.  Internally generated
bytes may advance the local trajectory, but can never teach it.

The previous sensory opponent cell is the legacy smallest phase signal.  The
order-sensitive mode instead grows local phase cells from externally observed
transitions.  A recall-onset cell seeds the trajectory; repeated values then
occupy distinct learned phases and an unseen generated transition terminates
without teaching itself.  This is deliberately modality-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

RecallItem = tuple[int, int]
_ONSET = 256
_TERMINAL = -1


@dataclass(frozen=True)
class ExpressionSignal:
    """One prediction and the local synapses that produced it."""

    distribution: np.ndarray
    recalled_distribution: np.ndarray | None
    active_rows: int
    gain: float


class RecalledCorticalExpression:
    """Sparse apical-to-output associations trained by sensory error.

    Rows are dendritic coincidences ``(assembly, receptor, phase_cell)``.
    Their values are sparse opponent-cell logits.  There is no episodic key,
    candidate ranking, semantic slot, or memory-owned output head here.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        minimum_evidence: int = 2,
        maximum_logit: float = 12.0,
        order_sensitive_phases: bool = False,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if minimum_evidence < 1:
            raise ValueError("minimum evidence must be positive")
        if maximum_logit <= 0.0:
            raise ValueError("maximum logit must be positive")
        self.learning_rate = learning_rate
        self.minimum_evidence = minimum_evidence
        self.maximum_logit = maximum_logit
        self.order_sensitive_phases = order_sensitive_phases
        self.weights: dict[tuple[int, int, int], dict[int, float]] = {}
        self.evidence: dict[tuple[int, int, int], int] = {}
        self.active: dict[RecallItem, tuple[float, int]] = {}
        self._last_rows: tuple[tuple[tuple[int, int, int], float], ...] = ()
        self._background_cache: dict[tuple[int, int], np.ndarray] = {}
        self.last_return_current_norm = 0.0
        self.phase_transitions: dict[tuple[int, int, int, int], int] = {}
        self.next_phase_cell = _ONSET + 1

    def __setstate__(self, state: dict[str, object]) -> None:
        """Migrate checkpoints created before residual return cancellation."""
        self.__dict__.update(state)
        self._background_cache = {}
        self.last_return_current_norm = 0.0
        self.order_sensitive_phases = getattr(
            self, "order_sensitive_phases", False
        )
        self.phase_transitions = getattr(self, "phase_transitions", {})
        self.next_phase_cell = getattr(self, "next_phase_cell", _ONSET + 1)

    @property
    def embodied_bytes(self) -> int:
        """Exact payload bytes are measured by the owning ALM checkpoint."""
        import pickle

        return len(
            pickle.dumps(
                (self.weights, self.evidence, self.phase_transitions),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    def reset_episode(self) -> None:
        self.active.clear()
        self._last_rows = ()
        self.last_return_current_norm = 0.0

    def activate(
        self,
        items: tuple[RecallItem, ...],
        strengths: tuple[float, ...] = (),
        initial_phases: tuple[int, ...] = (),
    ) -> None:
        """Expose all recalled assemblies in parallel on their native wires."""
        strengths = strengths or tuple(1.0 for _ in items)
        if len(strengths) != len(items):
            raise ValueError("each recalled item needs one strength")
        initial_phases = initial_phases or tuple(_ONSET for _ in items)
        if len(initial_phases) != len(items):
            raise ValueError("each recalled item needs one initial phase")
        incoming = {
            item: (
                max(0.0, float(strength)),
                self.active.get(item, (0.0, int(initial_phase)))[1],
            )
            for item, strength, initial_phase in zip(
                items, strengths, initial_phases, strict=True
            )
            if strength > 0.0
        }
        self.active = incoming

    def phase_after_prefix(self, item: RecallItem, prefix: bytes) -> int | None:
        """Follow learned local phase cells after an observed partial cue."""
        if not self.order_sensitive_phases:
            return int(prefix[-1]) if prefix else _ONSET
        phase = _ONSET
        assembly, receptor = item
        for byte_id in prefix:
            next_phase = self.phase_transitions.get(
                (assembly, receptor, phase, byte_id)
            )
            if next_phase is None:
                return None
            phase = next_phase
        return phase

    @staticmethod
    def _row_distribution(row: dict[int, float]) -> np.ndarray:
        """Softmax a sparse row while retaining 256 implicit zero logits."""
        if not row:
            return np.full(256, 1.0 / 256.0, dtype=np.float64)
        maximum = max(0.0, max(row.values()))
        background = exp(-maximum)
        values = np.full(256, background, dtype=np.float64)
        for byte_id, logit in row.items():
            values[byte_id] = exp(logit - maximum)
        values /= float(np.sum(values))
        return values

    @staticmethod
    def _row_current(row: dict[int, float]) -> np.ndarray:
        current = np.zeros(256, dtype=np.float64)
        for byte_id, value in row.items():
            current[byte_id] = value
        return current

    def _background_current(self, receptor: int, phase: int) -> np.ndarray:
        """Estimate shared externally learned current on one receiving wire."""
        key = receptor, phase
        cached = self._background_cache.get(key)
        if cached is not None:
            return cached
        background = np.zeros(256, dtype=np.float64)
        count = 0
        for (assembly, row_receptor, row_phase), row in self.weights.items():
            if (
                row_receptor != receptor
                or row_phase != phase
                or self.evidence.get(
                    (assembly, row_receptor, row_phase), 0
                )
                < self.minimum_evidence
            ):
                continue
            background += self._row_current(row)
            count += 1
        if count:
            background /= count
        self._background_cache[key] = background
        return background

    @staticmethod
    def _softmax_current(current: np.ndarray) -> np.ndarray:
        shifted = current - float(np.max(current))
        distribution = np.exp(shifted)
        distribution /= float(np.sum(distribution))
        return distribution

    def predict(
        self,
        baseline: np.ndarray,
        *,
        maximum_gain: float,
        shared_cancellation: str = "none",
        maximum_current_norm: float = 0.0,
    ) -> ExpressionSignal:
        if not 0.0 <= maximum_gain <= 1.0:
            raise ValueError("maximum gain must be in [0, 1]")
        if shared_cancellation not in {"none", "pre", "post"}:
            raise ValueError("unknown shared return cancellation mode")
        if maximum_current_norm < 0.0:
            raise ValueError("maximum return current norm must be non-negative")
        recalled = np.zeros(256, dtype=np.float64)
        residual_current = np.zeros(256, dtype=np.float64)
        background_distribution = np.zeros(256, dtype=np.float64)
        total_strength = 0.0
        rows: list[tuple[tuple[int, int, int], float]] = []
        for (assembly, receptor), (strength, phase) in self.active.items():
            key = assembly, receptor, phase
            # An immature apical row is a silent synapse, not an absent
            # synapse. It cannot alter cortex before probation evidence, but
            # it remains locally eligible for the next external observation.
            # This breaks the circular dependency in which a row needed
            # evidence before it could ever acquire evidence. Generated or
            # recalled activity still cannot teach it because observe() owns
            # the external-provenance check.
            rows.append((key, strength))
            if self.evidence.get(key, 0) < self.minimum_evidence:
                continue
            row = self.weights.get(key, {})
            row_distribution = self._row_distribution(row)
            recalled += strength * row_distribution
            if shared_cancellation != "none":
                background = self._background_current(receptor, phase)
                residual_current += strength * (
                    self._row_current(row) - background
                )
                background_distribution += (
                    strength * self._softmax_current(background)
                )
            total_strength += strength
        self._last_rows = tuple(rows)
        if total_strength <= 0.0 or maximum_gain <= 0.0:
            self.last_return_current_norm = 0.0
            return ExpressionSignal(baseline, None, 0, 0.0)
        recalled /= total_strength
        residual_current /= total_strength
        raw_current_norm = float(np.linalg.norm(residual_current))
        if (
            shared_cancellation == "pre"
            and maximum_current_norm > 0.0
            and raw_current_norm > maximum_current_norm
        ):
            residual_current *= maximum_current_norm / raw_current_norm
        self.last_return_current_norm = (
            float(np.linalg.norm(residual_current))
            if shared_cancellation != "none"
            else 0.0
        )
        if shared_cancellation == "pre":
            recalled = self._softmax_current(residual_current)
        elif shared_cancellation == "post":
            background_distribution /= total_strength
            positive = np.maximum(0.0, recalled - background_distribution)
            total = float(np.sum(positive))
            recalled = positive / total if total > 0.0 else recalled
        gain = min(maximum_gain, maximum_gain * total_strength)
        distribution = (1.0 - gain) * baseline + gain * recalled
        distribution /= float(np.sum(distribution))
        return ExpressionSignal(distribution, recalled, len(rows), gain)

    def observe(self, byte_id: int, *, external: bool) -> None:
        """Advance phase; update synapses only from an external observation."""
        if not 0 <= byte_id < 256:
            raise ValueError("byte id must be in [0, 255]")
        if external:
            for key, strength in self._last_rows:
                row = self.weights.setdefault(key, {})
                predicted = self._row_distribution(row)
                # Sparse excitatory prediction-error plasticity.  Active wrong
                # alternatives are depressed; unseen alternatives remain the
                # shared inhibitory background represented by zero.
                for target in tuple(row):
                    if target == byte_id:
                        continue
                    error = -predicted[target]
                    row[target] = float(
                        np.clip(
                            row[target] + self.learning_rate * strength * error,
                            -self.maximum_logit,
                            self.maximum_logit,
                        )
                    )
                target_error = 1.0 - predicted[byte_id]
                row[byte_id] = float(
                    np.clip(
                        row.get(byte_id, 0.0)
                        + self.learning_rate * strength * target_error,
                        -self.maximum_logit,
                        self.maximum_logit,
                    )
                )
                self.evidence[key] = self.evidence.get(key, 0) + 1
            self._background_cache.clear()
        advanced = {}
        for item, (strength, phase) in self.active.items():
            next_phase = byte_id
            if self.order_sensitive_phases:
                assembly, receptor = item
                transition = (assembly, receptor, phase, byte_id)
                stored = self.phase_transitions.get(transition)
                if stored is None and external:
                    stored = self.next_phase_cell
                    self.next_phase_cell += 1
                    self.phase_transitions[transition] = stored
                next_phase = _TERMINAL if stored is None else stored
            advanced[item] = (strength, next_phase)
        self.active = advanced
        self._last_rows = ()

    def learn_sequence(
        self,
        item: RecallItem,
        sequence: bytes,
        *,
        repetitions: int = 16,
    ) -> None:
        """Replay an already observed sensory trace through local synapses."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        for _ in range(repetitions):
            self.reset_episode()
            self.activate((item,))
            for byte_id in sequence:
                # Learning rows participate before probation evidence is met.
                assembly, receptor = item
                phase = self.active[item][1]
                key = assembly, receptor, phase
                self.weights.setdefault(key, {})
                self.evidence[key] = max(
                    self.minimum_evidence, self.evidence.get(key, 0)
                )
                self.predict(np.full(256, 1.0 / 256.0), maximum_gain=1.0)
                self.observe(byte_id, external=True)
        self.reset_episode()
