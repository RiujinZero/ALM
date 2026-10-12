"""Causal raw-byte population coding with local opponent receptors.

Every physical bit is represented by a mutually exclusive ON/OFF receptor
pair. A byte therefore arrives as eight simultaneous events in a 16-channel
sensory organ. The code is fixed transduction, not a learned embedding.
"""

from __future__ import annotations

import heapq
import pickle
from dataclasses import dataclass
from math import ceil, exp, log1p, log2

import numpy as np

from manifoldflow.dendritic_basin.causal_attribution import SelfCausalAttribution
from manifoldflow.dendritic_basin.contextual_competition import (
    normalized_competition,
)
from manifoldflow.dendritic_basin.temporal_difference_credit import (
    LocalValueEstimate,
)


class OpponentByteCodec:
    """Lossless 8-bit to 16-receptor opponent population code."""

    bit_count = 8
    receptor_count = 16

    def __init__(self) -> None:
        byte_ids = np.arange(256, dtype=np.uint16)[:, None]
        shifts = np.arange(self.bit_count, dtype=np.uint16)[None, :]
        self.bits = ((byte_ids >> shifts) & 1).astype(np.uint8)
        self.codebook = np.zeros((256, self.receptor_count), dtype=np.float32)
        for bit in range(self.bit_count):
            self.codebook[:, 2 * bit] = 1.0 - self.bits[:, bit]
            self.codebook[:, 2 * bit + 1] = self.bits[:, bit]

    def encode(self, byte_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(byte_ids, dtype=np.uint8)
        return self.codebook[values]

    def decode(self, receptor_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(receptor_vectors)
        if vectors.shape[-1] != self.receptor_count:
            raise ValueError("opponent vectors must have 16 receptors")
        active = vectors[..., 1::2] > vectors[..., 0::2]
        powers = 1 << np.arange(self.bit_count, dtype=np.uint16)
        return np.sum(active.astype(np.uint16) * powers, axis=-1).astype(np.uint8)

    def byte_distribution(self, bit_probabilities: np.ndarray) -> np.ndarray:
        """Decode eight Bernoulli predictions into a 256-byte distribution."""
        probability = np.clip(
            np.asarray(bit_probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12
        )
        if probability.shape != (self.bit_count,):
            raise ValueError("bit probabilities must have shape (8,)")
        log_probability = (
            self.bits * np.log(probability) + (1 - self.bits) * np.log1p(-probability)
        ).sum(axis=1)
        log_probability -= np.max(log_probability)
        distribution = np.exp(log_probability)
        return distribution / distribution.sum()


@dataclass(frozen=True)
class OpponentStep:
    observed_probability: float | None
    prediction: np.ndarray


class OpponentTransitionLearner:
    """Local Hebbian transition circuit over opponent receptor activity.

    Each active receptor owns eight binary outcome synapses. Predictions from
    the eight receptors active in the current byte pool their evidence. This
    is deliberately the basal vector control: dendritic conjunctions must
    improve on it rather than receiving byte identity from a categorical row.
    """

    def __init__(self, *, prior: float = 0.1) -> None:
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.prior = prior
        self.codec = OpponentByteCodec()
        self.unigram_counts = np.zeros((8, 2), dtype=np.float64)
        self.receptor_counts = np.zeros((16, 8, 2), dtype=np.float64)
        self.previous_receptors: np.ndarray | None = None
        self.steps = 0

    @property
    def embodied_bytes(self) -> int:
        return (
            self.codec.codebook.nbytes
            + self.unigram_counts.nbytes
            + self.receptor_counts.nbytes
        )

    def _bit_probabilities(self) -> np.ndarray:
        if self.previous_receptors is None:
            counts = self.unigram_counts + self.prior
        else:
            active = np.flatnonzero(self.previous_receptors)
            counts = self.receptor_counts[active].sum(axis=0) + self.prior
        return counts[:, 1] / counts.sum(axis=1)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        bit_probability = self._bit_probabilities()
        prediction = self.codec.byte_distribution(bit_probability)
        observed = (
            None if self.previous_receptors is None else float(prediction[byte_id])
        )
        target_bits = self.codec.bits[byte_id]
        if learn:
            for bit, target in enumerate(target_bits):
                self.unigram_counts[bit, target] += 1.0
            if self.previous_receptors is not None:
                active = np.flatnonzero(self.previous_receptors)
                for receptor in active:
                    for bit, target in enumerate(target_bits):
                        self.receptor_counts[receptor, bit, target] += 1.0
            self.steps += 1
        self.previous_receptors = self.codec.codebook[byte_id].copy()
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)
        observed = np.full(len(values), np.nan, dtype=np.float64)
        for index, byte_id in enumerate(values):
            step = self.observe(int(byte_id), learn=learn)
            if step.observed_probability is not None:
                observed[index] = step.observed_probability
        return observed

    def reset_episode(self) -> None:
        self.previous_receptors = None


class OpponentPairDendriteLearner:
    """Second cortical stage: pairwise coincidence over opponent receptors.

    A branch joins two receptors belonging to different physical bits. The
    complete bank contains 112 possible local conjunctions; a byte activates
    28. No branch is assigned a character or byte identity.
    """

    def __init__(self, *, prior: float = 0.1) -> None:
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.prior = prior
        self.codec = OpponentByteCodec()
        pairs = []
        for left_bit in range(8):
            for right_bit in range(left_bit + 1, 8):
                for left_state in range(2):
                    for right_state in range(2):
                        pairs.append(
                            (2 * left_bit + left_state, 2 * right_bit + right_state)
                        )
        self.pairs = np.asarray(pairs, dtype=np.uint8)
        self.branch_codebook = (
            self.codec.codebook[:, self.pairs[:, 0]]
            * self.codec.codebook[:, self.pairs[:, 1]]
        ).astype(np.uint8)
        self.unigram_counts = np.zeros((8, 2), dtype=np.float64)
        self.branch_counts = np.zeros((len(self.pairs), 8, 2), dtype=np.float64)
        self.previous_branches: np.ndarray | None = None
        self.steps = 0

    @property
    def embodied_bytes(self) -> int:
        return (
            self.codec.codebook.nbytes
            + self.pairs.nbytes
            + self.branch_codebook.nbytes
            + self.unigram_counts.nbytes
            + self.branch_counts.nbytes
        )

    def _bit_probabilities(self) -> np.ndarray:
        if self.previous_branches is None:
            counts = self.unigram_counts + self.prior
        else:
            active = np.flatnonzero(self.previous_branches)
            counts = self.branch_counts[active].sum(axis=0) + self.prior
        return counts[:, 1] / counts.sum(axis=1)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        prediction = self.codec.byte_distribution(self._bit_probabilities())
        observed = (
            None if self.previous_branches is None else float(prediction[byte_id])
        )
        target_bits = self.codec.bits[byte_id]
        if learn:
            for bit, target in enumerate(target_bits):
                self.unigram_counts[bit, target] += 1.0
            if self.previous_branches is not None:
                for branch in np.flatnonzero(self.previous_branches):
                    for bit, target in enumerate(target_bits):
                        self.branch_counts[branch, bit, target] += 1.0
            self.steps += 1
        self.previous_branches = self.branch_codebook[byte_id].copy()
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)
        observed = np.full(len(values), np.nan, dtype=np.float64)
        for index, byte_id in enumerate(values):
            step = self.observe(int(byte_id), learn=learn)
            if step.observed_probability is not None:
                observed[index] = step.observed_probability
        return observed

    def reset_episode(self) -> None:
        self.previous_branches = None


class OpponentAssemblyTransitionLearner:
    """Recursive coincidence tree that binds distributed byte identity.

    Four receptor-pair assemblies encode two physical bits each; two
    second-order branches bind four bits, and their conjunction forms an
    eight-bit assembly. Assemblies become live only when observed.
    """

    def __init__(self, *, prior: float = 0.1) -> None:
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.prior = prior
        self.codec = OpponentByteCodec()
        self.live_assemblies = np.zeros(256, dtype=np.bool_)
        self.unigram_counts = np.zeros((8, 2), dtype=np.float64)
        self.assembly_counts = np.zeros((256, 8, 2), dtype=np.float64)
        self.previous_assembly: int | None = None
        self.steps = 0

    @property
    def live_assembly_count(self) -> int:
        return int(np.sum(self.live_assemblies))

    @property
    def embodied_bytes(self) -> int:
        return (
            self.codec.codebook.nbytes
            + self.live_assemblies.nbytes
            + self.unigram_counts.nbytes
            + self.assembly_counts.nbytes
        )

    def assembly_path(
        self, byte_id: int
    ) -> tuple[tuple[int, ...], tuple[int, int], int]:
        """Return active two-bit, four-bit, and final assemblies."""
        bits = self.codec.bits[int(byte_id)]
        pairs = tuple(
            int(bits[2 * index] + 2 * bits[2 * index + 1]) for index in range(4)
        )
        left = pairs[0] + 4 * pairs[1]
        right = pairs[2] + 4 * pairs[3]
        final = left + 16 * right
        return pairs, (left, right), final

    def _bit_probabilities(self) -> np.ndarray:
        if self.previous_assembly is None:
            counts = self.unigram_counts + self.prior
        else:
            counts = self.assembly_counts[self.previous_assembly] + self.prior
        return counts[:, 1] / counts.sum(axis=1)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        prediction = self.codec.byte_distribution(self._bit_probabilities())
        observed = (
            None if self.previous_assembly is None else float(prediction[byte_id])
        )
        target_bits = self.codec.bits[byte_id]
        if learn:
            for bit, target in enumerate(target_bits):
                self.unigram_counts[bit, target] += 1.0
            if self.previous_assembly is not None:
                for bit, target in enumerate(target_bits):
                    self.assembly_counts[self.previous_assembly, bit, target] += 1.0
            self.steps += 1
        _pairs, _quartets, assembly = self.assembly_path(byte_id)
        if learn:
            self.live_assemblies[assembly] = True
        self.previous_assembly = assembly
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)
        observed = np.full(len(values), np.nan, dtype=np.float64)
        for index, byte_id in enumerate(values):
            step = self.observe(int(byte_id), learn=learn)
            if step.observed_probability is not None:
                observed[index] = step.observed_probability
        return observed

    def reset_episode(self) -> None:
        self.previous_assembly = None


class OpponentAttractorAssemblyLearner:
    """Sparse transitions between assemblies discovered from receptor trees.

    The output basin competes over whole assemblies and then projects its
    winner distribution back to the opponent receptors. This preserves the
    correlations that independent output-bit synapses necessarily discard.
    """

    def __init__(self, *, prior: float = 0.1) -> None:
        if prior <= 0.0:
            raise ValueError("prior must be positive")
        self.prior = prior
        self.codec = OpponentByteCodec()
        self.live_assemblies = np.zeros(256, dtype=np.bool_)
        self.unigram_counts = np.zeros(256, dtype=np.int64)
        self.transition_counts: list[dict[int, int]] = [dict() for _ in range(256)]
        self.context_totals = np.zeros(256, dtype=np.int64)
        self.previous_assembly: int | None = None
        self.steps = 0

    @property
    def live_assembly_count(self) -> int:
        return int(np.sum(self.live_assemblies))

    @property
    def live_transition_count(self) -> int:
        return sum(len(row) for row in self.transition_counts)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.codec.codebook,
            self.live_assemblies,
            self.unigram_counts,
            self.transition_counts,
            self.context_totals,
            self.previous_assembly,
            self.steps,
        )
        return len(pickle.dumps(payload, protocol=5))

    def _distribution(self) -> np.ndarray:
        if self.previous_assembly is None:
            counts = self.unigram_counts.astype(np.float64) + self.prior
            return counts / counts.sum()
        total = self.context_totals[self.previous_assembly] + 256 * self.prior
        distribution = np.full(256, self.prior / total, dtype=np.float64)
        for target, count in self.transition_counts[self.previous_assembly].items():
            distribution[target] = (count + self.prior) / total
        return distribution

    def predict_next_distribution(self) -> np.ndarray:
        """Return the current 256-way basin state without changing the learner."""
        return self._distribution().copy()

    def _observed_probability(self, byte_id: int) -> float:
        """Read one output basin without broadcasting across all 256 basins."""
        if self.previous_assembly is None:
            total = int(self.unigram_counts.sum()) + 256 * self.prior
            return float((self.unigram_counts[byte_id] + self.prior) / total)
        total = self.context_totals[self.previous_assembly] + 256 * self.prior
        count = self.transition_counts[self.previous_assembly].get(byte_id, 0)
        return float((count + self.prior) / total)

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        observed = (
            None
            if self.previous_assembly is None
            else self._observed_probability(byte_id)
        )
        if learn:
            self.unigram_counts[byte_id] += 1
            if self.previous_assembly is not None:
                row = self.transition_counts[self.previous_assembly]
                row[byte_id] = row.get(byte_id, 0) + 1
                self.context_totals[self.previous_assembly] += 1
            self.live_assemblies[byte_id] = True
            self.steps += 1
        self.previous_assembly = byte_id
        return observed

    def advance_observation(self, byte_id: int, *, learn: bool = False) -> float | None:
        """Advance recurrent state without constructing a full output vector."""
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        return self._observe_probability_only(byte_id, learn=learn)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        prediction = self._distribution()
        observed = (
            None if self.previous_assembly is None else float(prediction[byte_id])
        )
        if learn:
            self.unigram_counts[byte_id] += 1
            if self.previous_assembly is not None:
                row = self.transition_counts[self.previous_assembly]
                row[byte_id] = row.get(byte_id, 0) + 1
                self.context_totals[self.previous_assembly] += 1
            self.live_assemblies[byte_id] = True
            self.steps += 1
        self.previous_assembly = byte_id
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)
        observed = np.full(len(values), np.nan, dtype=np.float64)
        for index, byte_id in enumerate(values):
            probability = self._observe_probability_only(int(byte_id), learn=learn)
            if probability is not None:
                observed[index] = probability
        return observed

    def reset_episode(self) -> None:
        self.previous_assembly = None


class DevelopmentalOpponentAttractorLearner(OpponentAttractorAssemblyLearner):
    """Nursery-grown receptor tree feeding a sparse output attractor."""

    def __init__(self, *, prior: float = 0.1, confirmations: int = 3) -> None:
        super().__init__(prior=prior)
        if confirmations < 1:
            raise ValueError("confirmations must be positive")
        self.confirmations = confirmations
        self.pair_confirmations = np.zeros((4, 4), dtype=np.uint16)
        self.pair_live = np.zeros((4, 4), dtype=np.bool_)
        self.quartet_confirmations = np.zeros((2, 16), dtype=np.uint16)
        self.quartet_live = np.zeros((2, 16), dtype=np.bool_)
        self.assembly_confirmations = np.zeros(256, dtype=np.uint16)
        self.assembly_births = 0

    @property
    def nursery_count(self) -> int:
        return int(
            np.sum((self.pair_confirmations > 0) & ~self.pair_live)
            + np.sum((self.quartet_confirmations > 0) & ~self.quartet_live)
            + np.sum((self.assembly_confirmations > 0) & ~self.live_assemblies)
        )

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.codec.codebook,
            self.live_assemblies,
            self.unigram_counts,
            self.transition_counts,
            self.context_totals,
            self.previous_assembly,
            self.steps,
            self.pair_confirmations,
            self.pair_live,
            self.quartet_confirmations,
            self.quartet_live,
            self.assembly_confirmations,
            self.assembly_births,
        )
        return len(pickle.dumps(payload, protocol=5))

    def _develop(self, byte_id: int) -> bool:
        # A living byte assembly implies that all of its pair and quartet
        # prerequisites are already living. Mature cortex therefore has no
        # developmental work to repeat for this observation.
        if self.live_assemblies[byte_id]:
            return True
        bits = self.codec.bits[byte_id]
        pair_states = tuple(
            int(bits[2 * index] + 2 * bits[2 * index + 1]) for index in range(4)
        )
        for position, state in enumerate(pair_states):
            if not self.pair_live[position, state]:
                self.pair_confirmations[position, state] += 1
                if self.pair_confirmations[position, state] >= self.confirmations:
                    self.pair_live[position, state] = True
        quartet_states = (
            pair_states[0] + 4 * pair_states[1],
            pair_states[2] + 4 * pair_states[3],
        )
        for position, state in enumerate(quartet_states):
            pair_offset = 2 * position
            if (
                self.pair_live[pair_offset, pair_states[pair_offset]]
                and self.pair_live[pair_offset + 1, pair_states[pair_offset + 1]]
                and not self.quartet_live[position, state]
            ):
                self.quartet_confirmations[position, state] += 1
                if self.quartet_confirmations[position, state] >= self.confirmations:
                    self.quartet_live[position, state] = True
        assembly = quartet_states[0] + 16 * quartet_states[1]
        if (
            self.quartet_live[0, quartet_states[0]]
            and self.quartet_live[1, quartet_states[1]]
            and not self.live_assemblies[assembly]
        ):
            self.assembly_confirmations[assembly] += 1
            if self.assembly_confirmations[assembly] >= self.confirmations:
                self.live_assemblies[assembly] = True
                self.assembly_births += 1
        return bool(self.live_assemblies[assembly])

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        prediction = self._distribution()
        observed = None if self.steps == 0 else float(prediction[byte_id])
        if learn:
            is_live = self._develop(byte_id)
            if is_live:
                self.unigram_counts[byte_id] += 1
                if self.previous_assembly is not None:
                    row = self.transition_counts[self.previous_assembly]
                    row[byte_id] = row.get(byte_id, 0) + 1
                    self.context_totals[self.previous_assembly] += 1
            self.steps += 1
            self.previous_assembly = byte_id if is_live else None
        else:
            self.previous_assembly = byte_id if self.live_assemblies[byte_id] else None
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        observed = None if self.steps == 0 else self._observed_probability(byte_id)
        if learn:
            is_live = self._develop(byte_id)
            if is_live:
                self.unigram_counts[byte_id] += 1
                if self.previous_assembly is not None:
                    row = self.transition_counts[self.previous_assembly]
                    row[byte_id] = row.get(byte_id, 0) + 1
                    self.context_totals[self.previous_assembly] += 1
            self.steps += 1
            self.previous_assembly = byte_id if is_live else None
        else:
            self.previous_assembly = byte_id if self.live_assemblies[byte_id] else None
        return observed


class ContextualOpponentAttractorLearner(DevelopmentalOpponentAttractorLearner):
    """Surprise-grown contextual dendrites over living byte assemblies.

    A basal assembly predicts its successor directly. When that prediction is
    repeatedly surprising, the preceding assembly may join it on a dendritic
    branch. The branch stores only observed outputs and backs off to the basal
    basin through a finite prior, so immature context never erases established
    transition knowledge.
    """

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        competition_exponent: float = 1.0,
        adaptive_competition: bool = False,
        trust_gamma: float = 0.5,
        trust_learning_rate: float = 0.2,
        attribution_gate_threshold: float = 0.5,
        burst_surprise_signal: bool = False,
        burst_ema_rate: float = 0.001,
    ) -> None:
        super().__init__(prior=prior, confirmations=confirmations)
        if context_confirmations < 1:
            raise ValueError("context_confirmations must be positive")
        if surprise_bits < 0.0:
            raise ValueError("surprise_bits must be non-negative")
        if basal_strength <= 0.0:
            raise ValueError("basal_strength must be positive")
        if competition_exponent <= 0.0:
            raise ValueError("competition_exponent must be positive")
        if not 0.0 <= trust_gamma <= 1.0:
            raise ValueError("trust_gamma must be in [0, 1]")
        if not 0.0 <= attribution_gate_threshold <= 1.0:
            raise ValueError("attribution_gate_threshold must be in [0, 1]")
        self.context_confirmations = context_confirmations
        self.surprise_bits = surprise_bits
        self.basal_strength = basal_strength
        self.competition_exponent = competition_exponent
        self.older_assembly: int | None = None
        self.context_nursery: dict[tuple[int, int], dict[int, int]] = {}
        self.residual_counts: dict[tuple[int, int], dict[int, int]] = {}
        self.residual_totals: dict[tuple[int, int], int] = {}
        self.context_births = 0

        # Optional, opt-in consequence-sensitive competitive gain (contextual
        # competition's "A_i" term) and its felt-agency gate. Off by default
        # so every pre-existing oracle test stays bit-identical. This state
        # is wake-round-local, not consolidated anatomy: consolidate() and
        # consolidate_delta() only ever merge counts, never context_trust, so
        # a fresh ancestor's (empty) trust table is what every new wake round
        # actually starts from -- trust learned during one round shapes that
        # round's own predictions and is then implicitly discarded, the same
        # provisional standing every nursery path already has before it is
        # confirmed into durable anatomy.
        self.adaptive_competition = adaptive_competition
        self.trust_gamma = trust_gamma
        self.attribution_gate_threshold = attribution_gate_threshold
        self.context_trust: dict[tuple[int, int], float] = {}
        self.context_value = LocalValueEstimate(learning_rate=trust_learning_rate)
        self.context_attribution = SelfCausalAttribution()
        self._pending_context_credit: tuple[tuple[int, int], float] | None = None

        # Opt-in population-level surprise signal (HTM-inspired: "what
        # fraction of recent activity landed on brand-new/immature structure
        # rather than already-confirmed structure", the byte-cortex analog
        # of a minicolumn population's bursting fraction), independent of
        # and complementary to context_trust's existing per-context,
        # TD-bootstrapped value signal above. See
        # docs/HTM_Inspired_Followup_Experiments.md, Idea 1.
        self.burst_surprise_signal = burst_surprise_signal
        self.burst_ema_rate = burst_ema_rate
        self.burst_fraction = 0.0
        self._burst_last_birth_total = 0

    @property
    def live_context_count(self) -> int:
        return len(self.residual_counts)

    @property
    def live_context_synapse_count(self) -> int:
        return sum(len(row) for row in self.residual_counts.values())

    @property
    def nursery_count(self) -> int:
        return super().nursery_count + len(self.context_nursery)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            super().embodied_bytes,
            self.older_assembly,
            self.context_nursery,
            self.residual_counts,
            self.residual_totals,
            self.context_births,
            self.context_confirmations,
            self.surprise_bits,
            self.basal_strength,
            self.competition_exponent,
        )
        # context_trust/context_value/context_attribution are deliberately
        # excluded: they are wake-round-local (see __init__), never survive
        # a sleep boundary, and every pre-existing embodied_bytes oracle
        # test asserts an exact byte count at adaptive_competition=False.
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _context_key(self) -> tuple[int, int] | None:
        if self.older_assembly is None or self.previous_assembly is None:
            return None
        return self.older_assembly, self.previous_assembly

    def _update_burst_fraction(self) -> None:
        """One EMA update per observed byte: was *anything* just born?

        ``context_births``/``sequence_births`` are lifetime cumulative
        counters already maintained elsewhere; this only reads their sum
        and turns "did the total increase this byte" into a 0/1 indicator,
        then folds it into a slow exponential moving average so
        ``burst_fraction`` tracks *recent* structural novelty rather than a
        lifetime average that would just shrink monotonically over a long
        run.
        """
        current_total = self.context_births + getattr(self, "sequence_births", 0)
        indicator = 1.0 if current_total > self._burst_last_birth_total else 0.0
        self._burst_last_birth_total = current_total
        self.burst_fraction += self.burst_ema_rate * (indicator - self.burst_fraction)

    def _burst_damped_gain(self, gain: float) -> float:
        """Damp a child_gain toward the parent estimate while the population
        is bursting -- freshly-forming structure is less trustworthy when a
        lot of *other* structure is also still novel right now. Floored at
        0.05 so a maximal burst never fully zeroes the child out."""
        if not self.burst_surprise_signal:
            return gain
        return gain * max(0.05, 1.0 - self.burst_fraction)

    def _backoff_weights(
        self,
        parent_evidence: float,
        child_evidence: float,
        *,
        child_gain: float = 1.0,
    ) -> tuple[float, float] | None:
        """Shared-pool competition between the coarser and finer prediction.

        This is the contextual-competition equation
        (``contextual_competition.normalized_competition``) with the parent
        prediction and the freshly grown context each treated as one
        competitor: ``kappa = 0`` makes the pair an exact partition.
        ``child_gain`` is the equation's ``A_i`` contextual-gain term -- when
        ``adaptive_competition`` is enabled it is this context's learned
        trust (see ``_credit_context_trust``), otherwise it stays at 1.0. At
        the default ``competition_exponent = 1.0`` and ``child_gain = 1.0``
        this returns ``None`` so every caller falls back to the original
        fused arithmetic verbatim (bit-identical to every pre-existing
        oracle test); a non-default exponent or a learned gain away from 1.0
        shifts which side of the backoff wins.
        """
        if self.competition_exponent == 1.0 and child_gain == 1.0:
            return None
        weights = normalized_competition(
            np.array([parent_evidence, child_evidence], dtype=np.float64),
            np.array([1.0, child_gain], dtype=np.float64),
            exponent=self.competition_exponent,
            kappa=0.0,
        )
        return float(weights[0]), float(weights[1])

    def _credit_context_trust(
        self, key: tuple[int, int], *, basal: float, child_probability: float
    ) -> None:
        """Consequence-sensitive learning plus a felt-agency gate on trust.

        ``utility`` is the same one-step signed log-probability signal used
        throughout this package's other calibrated pathways. It is finalized
        one context-visit late (this method is called once per byte at this
        backoff level, so consecutive calls form a genuine temporal chain),
        giving a real ``delta_t = u + gamma*V(next) - V(current)`` TD credit
        rather than a same-step signal. ``context_attribution`` then asks
        whether this effect size is more consistent with a genuinely useful
        context than with the noise floor already seen for this learner
        before the learned trust (the competition equation's contextual gain
        term) is actually updated -- only externally observed bytes ever
        drive this; nothing generated or recalled can confirm itself.
        """
        utility = log2(max(child_probability, 1e-12)) - log2(max(basal, 1e-12))
        if self._pending_context_credit is not None:
            previous_key, previous_utility = self._pending_context_credit
            delta = self.context_value.update(
                previous_key,
                outcome=previous_utility,
                next_key=key,
                gamma=self.trust_gamma,
            )
            effect = abs(previous_utility)
            self.context_attribution.confirm(
                effect, cause="self" if delta > 0.0 else "other"
            )
            if self.context_attribution.posterior_self(effect) > (
                self.attribution_gate_threshold
            ):
                multiplier = exp(max(-2.0, min(2.0, delta)))
                previous_trust = self.context_trust.get(previous_key, 1.0)
                self.context_trust[previous_key] = min(
                    10.0, max(0.1, previous_trust * multiplier)
                )
        self._pending_context_credit = (key, utility)

    def _observed_probability(self, byte_id: int) -> float:
        basal = super()._observed_probability(byte_id)
        if self.burst_surprise_signal:
            self._update_burst_fraction()
        key = self._context_key()
        if key is None or key not in self.residual_counts:
            return basal
        count = self.residual_counts[key].get(byte_id, 0)
        total = self.residual_totals[key]
        trust = self.context_trust.get(key, 1.0) if self.adaptive_competition else 1.0
        trust = self._burst_damped_gain(trust)
        weights = self._backoff_weights(
            self.basal_strength, float(total), child_gain=trust
        )
        if weights is None:
            result = float(
                (count + self.basal_strength * basal) / (total + self.basal_strength)
            )
        else:
            parent_weight, child_weight = weights
            result = float(parent_weight * basal + child_weight * (count / total))
        if self.adaptive_competition:
            self._credit_context_trust(
                key, basal=basal, child_probability=count / total
            )
        return result

    def _distribution(self) -> np.ndarray:
        basal = super()._distribution()
        key = self._context_key()
        if key is None or key not in self.residual_counts:
            return basal
        row = self.residual_counts[key]
        total = self.residual_totals[key]
        trust = self.context_trust.get(key, 1.0) if self.adaptive_competition else 1.0
        trust = self._burst_damped_gain(trust)
        weights = self._backoff_weights(
            self.basal_strength, float(total), child_gain=trust
        )
        if weights is None:
            distribution = self.basal_strength * basal
            for target, count in row.items():
                distribution[target] += count
            return distribution / (total + self.basal_strength)
        parent_weight, child_weight = weights
        distribution = parent_weight * basal
        for target, count in row.items():
            distribution[target] += child_weight * (count / total)
        return distribution

    def _grow_or_update_context(self, byte_id: int, basal: float) -> None:
        key = self._context_key()
        if key is None:
            return
        tracked = getattr(self, "_lineage_context_keys", None)
        if tracked is not None:
            tracked.add(key)
        if key in self.residual_counts:
            row = self.residual_counts[key]
            row[byte_id] = row.get(byte_id, 0) + 1
            self.residual_totals[key] += 1
            return
        if -log2(max(basal, 1e-300)) < self.surprise_bits:
            return
        nursery = self.context_nursery.setdefault(key, {})
        nursery[byte_id] = nursery.get(byte_id, 0) + 1
        if sum(nursery.values()) >= self.context_confirmations:
            self.residual_counts[key] = nursery
            self.residual_totals[key] = sum(nursery.values())
            del self.context_nursery[key]
            self.context_births += 1

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        is_live = self._develop(byte_id)
        if is_live:
            self._grow_or_update_context(byte_id, basal)
            self.unigram_counts[byte_id] += 1
            if self.previous_assembly is not None:
                row = self.transition_counts[self.previous_assembly]
                row[byte_id] = row.get(byte_id, 0) + 1
                self.context_totals[self.previous_assembly] += 1
            self.older_assembly = self.previous_assembly
            self.previous_assembly = byte_id
        else:
            self.older_assembly = None
            self.previous_assembly = None
        self.steps += 1

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        basal = super()._observed_probability(byte_id)
        prediction = self._distribution()
        observed = None if self.steps == 0 else float(prediction[byte_id])
        if learn:
            self._learn_and_advance(byte_id, basal)
        elif self.live_assemblies[byte_id]:
            self.older_assembly = self.previous_assembly
            self.previous_assembly = byte_id
        else:
            self.older_assembly = None
            self.previous_assembly = None
        return OpponentStep(observed_probability=observed, prediction=prediction)

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        if not 0 <= byte_id < 256:
            raise ValueError("byte_id must be in [0, 255]")
        basal = super()._observed_probability(byte_id)
        observed = None if self.steps == 0 else self._observed_probability(byte_id)
        if learn:
            self._learn_and_advance(byte_id, basal)
        elif self.live_assemblies[byte_id]:
            self.older_assembly = self.previous_assembly
            self.previous_assembly = byte_id
        else:
            self.older_assembly = None
            self.previous_assembly = None
        return observed

    def reset_episode(self) -> None:
        super().reset_episode()
        self.older_assembly = None

    def consolidate(self, other: ContextualOpponentAttractorLearner) -> None:
        """Merge another column's count-based anatomy into this column."""
        configuration = (
            "prior",
            "confirmations",
            "context_confirmations",
            "surprise_bits",
            "basal_strength",
            "competition_exponent",
            "adaptive_competition",
            "trust_gamma",
            "attribution_gate_threshold",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError("cannot consolidate columns with different configuration")

        self.unigram_counts += other.unigram_counts
        for source, row in enumerate(other.transition_counts):
            target_row = self.transition_counts[source]
            for target, count in row.items():
                target_row[target] = target_row.get(target, 0) + count
        self.context_totals = np.asarray(
            [sum(row.values()) for row in self.transition_counts], dtype=np.int64
        )

        for confirmations, incoming in (
            (self.pair_confirmations, other.pair_confirmations),
            (self.quartet_confirmations, other.quartet_confirmations),
            (self.assembly_confirmations, other.assembly_confirmations),
        ):
            total = confirmations.astype(np.uint32) + incoming.astype(np.uint32)
            confirmations[...] = np.minimum(total, np.iinfo(np.uint16).max)
        self.pair_live |= other.pair_live
        self.quartet_live |= other.quartet_live
        self.live_assemblies |= other.live_assemblies
        self.assembly_births = self.live_assembly_count

        live_keys = set(self.residual_counts) | set(other.residual_counts)
        combined: dict[tuple[int, int], dict[int, int]] = {}
        for collection in (
            self.context_nursery,
            other.context_nursery,
            self.residual_counts,
            other.residual_counts,
        ):
            for key, row in collection.items():
                merged_row = combined.setdefault(key, {})
                for target, count in row.items():
                    merged_row[target] = merged_row.get(target, 0) + count
        self.context_nursery = {}
        self.residual_counts = {}
        self.residual_totals = {}
        for key, row in combined.items():
            total = sum(row.values())
            if key in live_keys or total >= self.context_confirmations:
                self.residual_counts[key] = row
                self.residual_totals[key] = total
            else:
                self.context_nursery[key] = row
        self.context_births = self.live_context_count
        self.steps += other.steps
        self.reset_episode()

    @staticmethod
    def _combined_sparse_rows(
        mature: dict[object, dict[int, int]],
        nursery: dict[object, dict[int, int]],
        keys: set[object] | None = None,
    ) -> dict[object, dict[int, int]]:
        selected = set(mature) | set(nursery) if keys is None else keys
        combined = {
            key: nursery[key].copy() for key in selected if key in nursery
        }
        for key in selected:
            row = mature.get(key)
            if row is None:
                continue
            target = combined.setdefault(key, {})
            for output, count in row.items():
                target[output] = target.get(output, 0) + count
        return combined

    def begin_lineage_tracking(self) -> None:
        """Record sparse anatomy touched during one wake generation."""
        self._lineage_context_keys: set[object] = set()

    @staticmethod
    def _sparse_row_delta(
        descendant: dict[object, dict[int, int]],
        ancestor: dict[object, dict[int, int]],
    ) -> dict[object, dict[int, int]]:
        """Compute what each touched key gained, skipping keys a column's
        own nursery FIFO cap (max_sequence_nursery_paths) evicted during its
        wake period.

        Nursery rows only grow monotonically *between* an eviction: once a
        key is evicted, a later observation of that same context within the
        same wake period starts a brand-new row via ``setdefault``, counting
        from zero again rather than continuing where the evicted row left
        off (mirrored natively in cortex.hpp's grow_sequence_nursery). Two
        symptoms follow, both meaning "this column forgot what the ancestor
        remembered about this key, then possibly relearned a fraction of
        it" rather than "this column corrupted its lineage": the key can be
        missing from the descendant entirely, or present with some output's
        count lower than the ancestor's own row had. Either way there is no
        reliable delta to extract for that key from this descendant -- the
        master (or another column) may still carry the ancestor's version,
        so the key is simply skipped rather than raised as an error. This
        was unreachable before the nursery persisted across sleeps (it was
        always wiped empty first), which is why it surfaced only once that
        changed, not because this discipline is new.
        """
        delta: dict[object, dict[int, int]] = {}
        for key, row in descendant.items():
            ancestor_row = ancestor.get(key, {})
            if any(ancestor_row.get(output, 0) > count for output, count in row.items()):
                continue
            increments = {
                output: count - ancestor_row.get(output, 0)
                for output, count in row.items()
                if count > ancestor_row.get(output, 0)
            }
            if increments:
                delta[key] = increments
        return delta

    def consolidate_delta(
        self,
        descendant: ContextualOpponentAttractorLearner,
        ancestor: ContextualOpponentAttractorLearner,
    ) -> None:
        """Merge only learning accrued after a shared cortical ancestor."""
        configuration = (
            "prior",
            "confirmations",
            "context_confirmations",
            "surprise_bits",
            "basal_strength",
            "competition_exponent",
            "adaptive_competition",
            "trust_gamma",
            "attribution_gate_threshold",
        )
        if any(
            getattr(self, name) != getattr(descendant, name)
            or getattr(self, name) != getattr(ancestor, name)
            for name in configuration
        ):
            raise ValueError("cannot consolidate incompatible cortical lineage")
        unigram_delta = descendant.unigram_counts - ancestor.unigram_counts
        if np.any(unigram_delta < 0):
            raise ValueError("descendant anatomy predates its ancestor")
        self.unigram_counts += unigram_delta
        for source, descendant_row in enumerate(descendant.transition_counts):
            ancestor_row = ancestor.transition_counts[source]
            target_row = self.transition_counts[source]
            for target, count in descendant_row.items():
                increment = count - ancestor_row.get(target, 0)
                if increment < 0:
                    raise ValueError("descendant anatomy predates its ancestor")
                if increment:
                    target_row[target] = target_row.get(target, 0) + increment
            if any(
                descendant_row.get(target, 0) < count
                for target, count in ancestor_row.items()
            ):
                raise ValueError("descendant anatomy predates its ancestor")
        self.context_totals = np.asarray(
            [sum(row.values()) for row in self.transition_counts], dtype=np.int64
        )

        for local, incoming, original in (
            (
                self.pair_confirmations,
                descendant.pair_confirmations,
                ancestor.pair_confirmations,
            ),
            (
                self.quartet_confirmations,
                descendant.quartet_confirmations,
                ancestor.quartet_confirmations,
            ),
            (
                self.assembly_confirmations,
                descendant.assembly_confirmations,
                ancestor.assembly_confirmations,
            ),
        ):
            increment = incoming.astype(np.int64) - original.astype(np.int64)
            if np.any(increment < 0):
                raise ValueError("descendant anatomy predates its ancestor")
            total = local.astype(np.int64) + increment
            local[...] = np.minimum(total, np.iinfo(np.uint16).max)
        self.pair_live |= descendant.pair_live
        self.quartet_live |= descendant.quartet_live
        self.live_assemblies |= descendant.live_assemblies
        self.assembly_births = self.live_assembly_count

        touched_contexts = getattr(descendant, "_lineage_context_keys", None)
        descendant_rows = self._combined_sparse_rows(
            descendant.residual_counts,
            descendant.context_nursery,
            touched_contexts,
        )
        ancestor_rows = self._combined_sparse_rows(
            ancestor.residual_counts,
            ancestor.context_nursery,
            touched_contexts,
        )
        delta_rows = self._sparse_row_delta(descendant_rows, ancestor_rows)
        descendant_live = set(descendant.residual_counts)
        for key, increments in delta_rows.items():
            if key in self.residual_counts:
                row = self.residual_counts[key]
            else:
                row = self.context_nursery.setdefault(key, {})
            for target, increment in increments.items():
                row[target] = row.get(target, 0) + increment
            total = sum(row.values())
            if key in descendant_live or total >= self.context_confirmations:
                self.residual_counts[key] = self.context_nursery.pop(key, row)
                self.residual_totals[key] = total
            elif key in self.residual_counts:
                self.residual_totals[key] = total
        self.context_births = self.live_context_count
        step_delta = descendant.steps - ancestor.steps
        if step_delta < 0:
            raise ValueError("descendant anatomy predates its ancestor")
        self.steps += step_delta
        self.reset_episode()


class SequenceCortexOpponentLearner(ContextualOpponentAttractorLearner):
    """Recursive temporal assemblies with hierarchical predictive backoff.

    Byte paths normally use one packed integer axonal address.  Tuple paths
    remain available as a behavioral oracle for compaction tests.
    """

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        competition_exponent: float = 1.0,
        adaptive_competition: bool = False,
        trust_gamma: float = 0.5,
        trust_learning_rate: float = 0.2,
        attribution_gate_threshold: float = 0.5,
        max_context_bytes: int = 6,
        sequence_confirmations: int = 3,
        parent_strength: float = 16.0,
        packed_paths: bool = True,
        max_sequence_nursery_paths: int = 50_000,
        structural_support: bool = False,
        structural_support_rate: float = 0.05,
        connectivity_pruning: bool = False,
        burst_surprise_signal: bool = False,
        burst_ema_rate: float = 0.001,
        long_range_bank: bool = False,
        long_range_context_bytes: int = 24,
        long_range_strength: float = 8.0,
        long_range_gate_on_burst: bool = False,
        long_range_gate_threshold: float = 0.1,
        long_range_max_entries: int = 200_000,
    ) -> None:
        super().__init__(
            prior=prior,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
            competition_exponent=competition_exponent,
            adaptive_competition=adaptive_competition,
            trust_gamma=trust_gamma,
            trust_learning_rate=trust_learning_rate,
            attribution_gate_threshold=attribution_gate_threshold,
            burst_surprise_signal=burst_surprise_signal,
            burst_ema_rate=burst_ema_rate,
        )
        if max_context_bytes < 2:
            raise ValueError("max_context_bytes must be at least two")
        if sequence_confirmations < 1:
            raise ValueError("sequence_confirmations must be positive")
        if parent_strength <= 0.0:
            raise ValueError("parent_strength must be positive")
        if max_sequence_nursery_paths < 1:
            raise ValueError("sequence nursery capacity must be positive")
        if structural_support_rate <= 0.0:
            raise ValueError("structural_support_rate must be positive")
        if long_range_gate_on_burst and not burst_surprise_signal:
            raise ValueError(
                "long_range_gate_on_burst requires burst_surprise_signal=True "
                "(otherwise burst_fraction never leaves 0.0 and every byte is "
                "gated out, silently disabling the long-range bank entirely)"
            )
        self.max_context_bytes = max_context_bytes
        self.sequence_confirmations = sequence_confirmations
        self.parent_strength = parent_strength
        self.packed_paths = packed_paths
        self.max_sequence_nursery_paths = max_sequence_nursery_paths
        self.structural_support = structural_support
        self.structural_support_rate = structural_support_rate
        self.connectivity_pruning = connectivity_pruning

        # HTM-inspired "apical" long-range bank: a second context bank keyed
        # on far more trailing bytes than the ordinary sequence banks (which
        # top out at max_context_bytes, <= 8), blended in as an additive
        # bias the same way sequence banks blend into the basal prediction.
        # This is a deliberately crude first cut at Idea 2 from
        # docs/HTM_Inspired_Followup_Experiments.md: reuse the cortex's own
        # proven context-matching mechanism for long-range prediction
        # instead of building a separate memory organ. Opt-in; pure Python
        # research code only (tuple keys, no native port, no packed-int
        # scheme) -- this is Stage A, a go/no-go probe, not production.
        self.long_range_bank = long_range_bank
        self.long_range_context_bytes = long_range_context_bytes
        self.long_range_strength = long_range_strength
        self.long_range_counts: dict[tuple[int, ...], dict[int, int]] = {}
        self.long_range_totals: dict[tuple[int, ...], int] = {}
        self._long_range_history: list[int] = []
        # Repurposing Idea 1's population-level novelty signal as Idea 2's
        # event-capture trigger (per the user's suggestion): rather than
        # growing the long-range bank on every byte indiscriminately, only
        # strengthen it while burst_fraction signals a genuine novelty
        # event -- the same role the original episodic field's ad hoc
        # "D_t = surprise + route turnover + temporal disagreement" event
        # detector played, but reusing a signal already measured for Idea 1.
        # Requires burst_surprise_signal=True to have any effect (otherwise
        # burst_fraction never leaves 0.0 and every byte is gated out).
        self.long_range_gate_on_burst = long_range_gate_on_burst
        self.long_range_gate_threshold = long_range_gate_threshold
        # No pruning mechanism reads long_range_counts at all (unlike the
        # ordinary sequence banks, which consolidate_to_byte_budget can
        # evict from) -- without a cap this grows without bound over a
        # long run, which conflicts with ALM's whole fixed-body-budget
        # design and could make consolidate_to_byte_budget's target
        # unreachable. FIFO eviction, same idiom as context_nursery/
        # sequence_nursery (dict insertion order is the queue).
        if long_range_max_entries < 1:
            raise ValueError("long_range_max_entries must be positive")
        self.long_range_max_entries = long_range_max_entries
        # The inherited basal and contextual arbors embody depths one and two.
        # These banks begin with three-byte context assemblies.
        self.sequence_counts: list[dict[int | tuple[int, ...], dict[int, int]]] = [
            {} for _ in range(max(0, max_context_bytes - 2))
        ]
        self.sequence_totals: list[dict[int | tuple[int, ...], int]] = [
            {} for _ in self.sequence_counts
        ]
        self.sequence_nursery: list[
            dict[int | tuple[int, ...], dict[int, int]]
        ] = [
            {} for _ in self.sequence_counts
        ]
        # Per-edge structural support s_ij (opt-in, structural_support=True):
        # one float per (context, target byte) connection, separate from the
        # integer counts the Bayesian backoff probability math needs. Only
        # used to rank edges for pruning -- see _credit_structural_support
        # and consolidate_to_byte_budget.
        self.sequence_support: list[
            dict[int | tuple[int, ...], dict[int, float]]
        ] = [{} for _ in self.sequence_counts]
        self.sequence_history: list[int] = []
        self.sequence_births = 0
        self.sequence_nursery_evictions = 0

    @property
    def live_sequence_count(self) -> int:
        return sum(len(bank) for bank in self.sequence_counts)

    @property
    def live_sequence_synapse_count(self) -> int:
        return sum(
            len(row) for bank in self.sequence_counts for row in bank.values()
        )

    @property
    def sequence_nursery_count(self) -> int:
        return sum(len(bank) for bank in self.sequence_nursery)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            super().embodied_bytes,
            self.max_context_bytes,
            self.sequence_confirmations,
            self.parent_strength,
            self.packed_paths,
            self.max_sequence_nursery_paths,
            self.sequence_counts,
            self.sequence_totals,
            self.sequence_nursery,
            self.sequence_support,
            self.sequence_history,
            self.sequence_births,
            self.sequence_nursery_evictions,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _sequence_contexts(self) -> tuple[int | tuple[int, ...], ...]:
        history = tuple(self.sequence_history)
        if self.packed_paths:
            packed = 0
            contexts = []
            for offset, byte_id in enumerate(reversed(history)):
                packed |= byte_id << (8 * offset)
                depth = offset + 1
                if depth >= 3:
                    contexts.append(packed)
            return tuple(contexts)
        contexts = []
        for depth in range(3, self.max_context_bytes + 1):
            if len(history) < depth:
                break
            contexts.append(history[-depth:])
        return tuple(contexts)

    def _observed_probability(self, byte_id: int) -> float:
        probability = super()._observed_probability(byte_id)
        for bank_index, context in enumerate(self._sequence_contexts()):
            row = self.sequence_counts[bank_index].get(context)
            if row is None:
                continue
            total = self.sequence_totals[bank_index][context]
            weights = self._backoff_weights(self.parent_strength, float(total))
            if weights is None:
                probability = float(
                    (row.get(byte_id, 0) + self.parent_strength * probability)
                    / (total + self.parent_strength)
                )
                continue
            parent_weight, child_weight = weights
            probability = float(
                parent_weight * probability
                + child_weight * (row.get(byte_id, 0) / total)
            )
        apical_key = self._long_range_key()
        if apical_key is not None:
            apical_row = self.long_range_counts.get(apical_key)
            if apical_row is not None:
                apical_total = self.long_range_totals[apical_key]
                probability = float(
                    (apical_row.get(byte_id, 0) + self.long_range_strength * probability)
                    / (apical_total + self.long_range_strength)
                )
        return probability

    def _distribution(self) -> np.ndarray:
        distribution = super()._distribution()
        for bank_index, context in enumerate(self._sequence_contexts()):
            row = self.sequence_counts[bank_index].get(context)
            if row is None:
                continue
            total = self.sequence_totals[bank_index][context]
            weights = self._backoff_weights(self.parent_strength, float(total))
            if weights is None:
                child = self.parent_strength * distribution
                for target, count in row.items():
                    child[target] += count
                distribution = child / (total + self.parent_strength)
                continue
            parent_weight, child_weight = weights
            child = parent_weight * distribution
            for target, count in row.items():
                child[target] += child_weight * (count / total)
            distribution = child
        apical_key = self._long_range_key()
        if apical_key is not None:
            apical_row = self.long_range_counts.get(apical_key)
            if apical_row is not None:
                apical_total = self.long_range_totals[apical_key]
                child = self.long_range_strength * distribution
                for target, count in apical_row.items():
                    child[target] += count
                distribution = child / (apical_total + self.long_range_strength)
        return distribution

    def _credit_structural_support(
        self,
        bank_index: int,
        context: int | tuple[int, ...],
        row: dict[int, int],
        byte_id: int,
    ) -> None:
        """Local, per-edge confirmation credit: s_ij <- s_ij + eta*((1{j==y} - b_j) - s_ij).

        Only called while context `i` is active (the caller already gates
        on that -- e_i=0 simply means this function never runs for that
        context this step, so absence of an opportunity to confirm is
        never counted as a failed prediction). Every target `j` this
        context currently tracks gets credited, not just the one that
        fired: the observed outcome `1{j==byte_id}` is compared against
        `j`'s own unigram baseline rate, so an edge only accumulates
        support by predicting its target *better than chance*, not merely
        by being active when the target happens to be common. `byte_id`
        itself is included even if this is its first appearance under this
        context, so a brand-new edge gets an initial support estimate
        rather than starting undefined.
        """
        support = self.sequence_support[bank_index].setdefault(context, {})
        total = float(self.unigram_counts.sum()) + 256.0 * self.prior
        rate = self.structural_support_rate
        targets = set(row)
        targets.add(byte_id)
        for target in targets:
            baseline = (float(self.unigram_counts[target]) + self.prior) / total
            observed = 1.0 if target == byte_id else 0.0
            current = support.get(target, 0.0)
            support[target] = current + rate * ((observed - baseline) - current)

    def _update_sequences(
        self,
        contexts: tuple[int | tuple[int, ...], ...],
        byte_id: int,
    ) -> None:
        for bank_index, context in enumerate(contexts):
            tracked = getattr(self, "_lineage_sequence_keys", None)
            if tracked is not None:
                tracked[bank_index].add(context)
            row = self.sequence_counts[bank_index].get(context)
            if row is not None:
                if self.structural_support:
                    self._credit_structural_support(bank_index, context, row, byte_id)
                row[byte_id] = row.get(byte_id, 0) + 1
                self.sequence_totals[bank_index][context] += 1
                continue
            nursery_bank = self.sequence_nursery[bank_index]
            if (
                context not in nursery_bank
                and len(nursery_bank) >= self.max_sequence_nursery_paths
            ):
                victim = next(iter(nursery_bank))
                del nursery_bank[victim]
                self.sequence_nursery_evictions += 1
                if self.structural_support:
                    # Otherwise this leaks: a FIFO-evicted nursery context's
                    # support entries would never be cleaned up, and with
                    # the nursery persisting across sleeps and churning
                    # constantly, that leak dominates embodied_bytes over a
                    # real run (confirmed directly: ~3x the baseline's
                    # anatomy size for identical input before this fix).
                    self.sequence_support[bank_index].pop(victim, None)
            nursery = nursery_bank.setdefault(context, {})
            if self.structural_support:
                self._credit_structural_support(bank_index, context, nursery, byte_id)
            nursery[byte_id] = nursery.get(byte_id, 0) + 1
            if sum(nursery.values()) >= self.sequence_confirmations:
                self.sequence_counts[bank_index][context] = nursery
                self.sequence_totals[bank_index][context] = sum(nursery.values())
                del self.sequence_nursery[bank_index][context]
                self.sequence_births += 1

    def _advance_sequence_history(self, byte_id: int) -> None:
        self.sequence_history.append(byte_id)
        excess = len(self.sequence_history) - self.max_context_bytes
        if excess > 0:
            del self.sequence_history[:excess]
        if self.long_range_bank:
            self._long_range_history.append(byte_id)
            excess = len(self._long_range_history) - self.long_range_context_bytes
            if excess > 0:
                del self._long_range_history[:excess]

    def _long_range_key(self) -> tuple[int, ...] | None:
        if not self.long_range_bank or len(self._long_range_history) < self.long_range_context_bytes:
            return None
        return tuple(self._long_range_history)

    def _grow_long_range(self, key: tuple[int, ...] | None, byte_id: int) -> None:
        if key is None:
            return
        # The gate applies only to *creating* a new key, not to reinforcing
        # one that already exists. An EMA novelty signal cannot tell
        # "boring because already well-established" apart from "boring
        # because nothing is happening" -- gating reinforcement too meant a
        # real, repeated fact stopped accumulating confidence the moment it
        # stopped being novel (measured: a fact confirmed 40 times in
        # training got reinforced once under gating, recall confidence
        # dropped from 95% to 63% -- see
        # tests/test_dendritic_basin_long_range_bank_recall.py). Gating
        # creation alone keeps repetitive filler from bloating the bank
        # with fresh keys while letting an established fact keep
        # strengthening on every subsequent match.
        is_new_key = key not in self.long_range_counts
        if (
            is_new_key
            and self.long_range_gate_on_burst
            and self.burst_fraction < self.long_range_gate_threshold
        ):
            return
        if is_new_key and len(self.long_range_counts) >= self.long_range_max_entries:
            victim = next(iter(self.long_range_counts))
            del self.long_range_counts[victim]
            del self.long_range_totals[victim]
        row = self.long_range_counts.setdefault(key, {})
        row[byte_id] = row.get(byte_id, 0) + 1
        self.long_range_totals[key] = self.long_range_totals.get(key, 0) + 1

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        contexts = self._sequence_contexts()
        apical_key = self._long_range_key()
        super()._learn_and_advance(byte_id, basal)
        if self.live_assemblies[byte_id]:
            self._update_sequences(contexts, byte_id)
            self._grow_long_range(apical_key, byte_id)
            self._advance_sequence_history(byte_id)
        else:
            self.sequence_history.clear()
            self._long_range_history.clear()

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        step = super().observe(byte_id, learn=learn)
        if not learn:
            if self.live_assemblies[int(byte_id)]:
                self._advance_sequence_history(int(byte_id))
            else:
                self.sequence_history.clear()
        return step

    def _observe_mature_sequence_probability(
        self, byte_id: int, *, learn: bool
    ) -> float | None:
        """Fused exact-class path for a packed mature sequence cortex."""
        previous = self.previous_assembly
        if previous is None:
            basal_total = int(self.unigram_counts.sum()) + 256 * self.prior
            basal = float(
                (self.unigram_counts[byte_id] + self.prior) / basal_total
            )
        else:
            basal_total = self.context_totals[previous] + 256 * self.prior
            basal = float(
                (self.transition_counts[previous].get(byte_id, 0) + self.prior)
                / basal_total
            )
        probability = basal
        context_key = (
            None
            if self.older_assembly is None or previous is None
            else (self.older_assembly, previous)
        )
        if context_key is not None:
            contextual_row = self.residual_counts.get(context_key)
            if contextual_row is not None:
                probability = float(
                    (
                        contextual_row.get(byte_id, 0)
                        + self.basal_strength * probability
                    )
                    / (
                        self.residual_totals[context_key]
                        + self.basal_strength
                    )
                )
        contexts = self._sequence_contexts()
        for bank_index, context in enumerate(contexts):
            row = self.sequence_counts[bank_index].get(context)
            if row is None:
                continue
            probability = float(
                (row.get(byte_id, 0) + self.parent_strength * probability)
                / (
                    self.sequence_totals[bank_index][context]
                    + self.parent_strength
                )
            )
        observed = None if self.steps == 0 else probability
        if learn:
            is_live = self._develop(byte_id)
            if is_live:
                self._grow_or_update_context(byte_id, basal)
                self.unigram_counts[byte_id] += 1
                if previous is not None:
                    row = self.transition_counts[previous]
                    row[byte_id] = row.get(byte_id, 0) + 1
                    self.context_totals[previous] += 1
                self.older_assembly = previous
                self.previous_assembly = byte_id
                self._update_sequences(contexts, byte_id)
                self._advance_sequence_history(byte_id)
            else:
                self.older_assembly = None
                self.previous_assembly = None
                self.sequence_history.clear()
            self.steps += 1
        elif self.live_assemblies[byte_id]:
            self.older_assembly = previous
            self.previous_assembly = byte_id
            self._advance_sequence_history(byte_id)
        else:
            self.older_assembly = None
            self.previous_assembly = None
            self.sequence_history.clear()
        return observed

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        if (
            type(self) is SequenceCortexOpponentLearner
            and self.competition_exponent == 1.0
            and not self.adaptive_competition
            and not self.burst_surprise_signal
            and not self.long_range_bank
        ):
            return self._observe_mature_sequence_probability(
                byte_id, learn=learn
            )
        probability = super()._observe_probability_only(byte_id, learn=learn)
        if not learn:
            if self.live_assemblies[byte_id]:
                self._advance_sequence_history(byte_id)
            else:
                self.sequence_history.clear()
        return probability

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        if (
            type(self) is not SequenceCortexOpponentLearner
            or self.competition_exponent != 1.0
            or self.adaptive_competition
            or self.structural_support
            or self.burst_surprise_signal
            or self.long_range_bank
        ):
            # The fused loop below hardcodes the original fixed backoff
            # arithmetic for speed and never calls _update_sequences (it
            # inlines an equivalent update directly), so it has no hook for
            # _credit_structural_support or _update_burst_fraction. A
            # non-default competition_exponent, adaptive_competition,
            # structural_support, or burst_surprise_signal needs the general
            # contextual-competition path (_backoff_weights/
            # _credit_context_trust/_update_burst_fraction), so fall back to
            # the slower but fully correct per-byte dispatch instead of
            # silently ignoring any of them.
            return super().observe_chunk_probabilities(byte_ids, learn=learn)
        values = np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)
        observed = np.full(len(values), np.nan, dtype=np.float64)

        # Keep the causal byte loop exact while taking invariant Python object
        # traversal out of it. Packed history is one rolling axonal address;
        # progressively wider masks recover the same depth-three-to-six keys
        # without rebuilding and reversing a tuple for every byte.
        unigram_counts = self.unigram_counts
        transition_counts = self.transition_counts
        context_totals = self.context_totals
        residual_counts = self.residual_counts
        residual_totals = self.residual_totals
        context_nursery = self.context_nursery
        sequence_counts = self.sequence_counts
        sequence_totals = self.sequence_totals
        sequence_nursery = self.sequence_nursery
        live_assemblies = self.live_assemblies
        tracked_contexts = getattr(self, "_lineage_context_keys", None)
        tracked_sequences = getattr(self, "_lineage_sequence_keys", None)
        prior = self.prior
        basal_strength = self.basal_strength
        parent_strength = self.parent_strength
        surprise_bits = self.surprise_bits
        context_confirmations = self.context_confirmations
        sequence_confirmations = self.sequence_confirmations
        max_nursery = self.max_sequence_nursery_paths
        max_context = self.max_context_bytes
        packed_paths = self.packed_paths
        history = self.sequence_history
        history_length = len(history)
        packed_history = 0
        if packed_paths:
            for history_byte in history:
                packed_history = (packed_history << 8) | history_byte
        history_mask = (1 << (8 * max_context)) - 1
        context_masks = tuple(
            (1 << (8 * depth)) - 1 for depth in range(3, max_context + 1)
        )
        previous = self.previous_assembly
        older = self.older_assembly
        steps = self.steps
        context_births = self.context_births
        sequence_births = self.sequence_births
        sequence_evictions = self.sequence_nursery_evictions

        for index, byte_id in enumerate(values):
            target = int(byte_id)
            fast_live_update = learn and bool(live_assemblies[target])
            if previous is None:
                basal_total = int(unigram_counts.sum()) + 256 * prior
                basal = float(
                    (unigram_counts[target] + prior) / basal_total
                )
            else:
                basal_total = context_totals[previous] + 256 * prior
                basal = float(
                    (transition_counts[previous].get(target, 0) + prior)
                    / basal_total
                )
            probability = basal
            context_key = (
                None
                if older is None or previous is None
                else (older, previous)
            )
            if context_key is not None:
                contextual_row = residual_counts.get(context_key)
                if contextual_row is not None:
                    contextual_target_count = contextual_row.get(target, 0)
                    probability = float(
                        (
                            contextual_target_count
                            + basal_strength * probability
                        )
                        / (
                            residual_totals[context_key]
                            + basal_strength
                        )
                    )
                    if fast_live_update:
                        contextual_row[target] = contextual_target_count + 1
                        residual_totals[context_key] += 1
                elif fast_live_update and -log2(max(basal, 1e-300)) >= surprise_bits:
                    nursery_row = context_nursery.setdefault(context_key, {})
                    nursery_row[target] = nursery_row.get(target, 0) + 1
                    nursery_total = sum(nursery_row.values())
                    if nursery_total >= context_confirmations:
                        residual_counts[context_key] = nursery_row
                        residual_totals[context_key] = nursery_total
                        del context_nursery[context_key]
                        context_births += 1
                if fast_live_update and tracked_contexts is not None:
                    tracked_contexts.add(context_key)
            if fast_live_update:
                unigram_counts[target] += 1
                if previous is not None:
                    transition_row = transition_counts[previous]
                    transition_row[target] = transition_row.get(target, 0) + 1
                    context_totals[previous] += 1
            if packed_paths:
                if max_context == 6 and history_length >= 6:
                    contexts = (
                        packed_history & context_masks[0],
                        packed_history & context_masks[1],
                        packed_history & context_masks[2],
                        packed_history & context_masks[3],
                    )
                else:
                    available = min(history_length, max_context)
                    contexts = tuple(
                        packed_history & context_masks[index]
                        for index in range(max(0, available - 2))
                    )
            else:
                tuple_contexts: list[tuple[int, ...]] = []
                for depth in range(3, max_context + 1):
                    if history_length < depth:
                        break
                    tuple_contexts.append(tuple(history[-depth:]))
                contexts = tuple(tuple_contexts)
            for bank_index, context in enumerate(contexts):
                row = sequence_counts[bank_index].get(context)
                if row is None:
                    if fast_live_update:
                        if tracked_sequences is not None:
                            tracked_sequences[bank_index].add(context)
                        nursery_bank = sequence_nursery[bank_index]
                        nursery = nursery_bank.get(context)
                        if nursery is None:
                            if len(nursery_bank) >= max_nursery:
                                del nursery_bank[next(iter(nursery_bank))]
                                sequence_evictions += 1
                            nursery = {}
                            nursery_bank[context] = nursery
                        nursery[target] = nursery.get(target, 0) + 1
                        nursery_total = sum(nursery.values())
                        if nursery_total >= sequence_confirmations:
                            sequence_counts[bank_index][context] = nursery
                            sequence_totals[bank_index][context] = nursery_total
                            del nursery_bank[context]
                            sequence_births += 1
                    continue
                row_target_count = row.get(target, 0)
                probability = float(
                    (row_target_count + parent_strength * probability)
                    / (
                        sequence_totals[bank_index][context]
                        + parent_strength
                    )
                )
                if fast_live_update:
                    if tracked_sequences is not None:
                        tracked_sequences[bank_index].add(context)
                    row[target] = row_target_count + 1
                    sequence_totals[bank_index][context] += 1
            if steps != 0:
                observed[index] = probability
            if learn:
                is_live = fast_live_update or self._develop(target)
                if is_live:
                    # A newly born assembly missed the fused updates above;
                    # mature assemblies already changed each active synapse at
                    # the instant its pre-update probability was read.
                    if not fast_live_update:
                        if context_key is not None:
                            if tracked_contexts is not None:
                                tracked_contexts.add(context_key)
                            contextual_row = residual_counts.get(context_key)
                            if contextual_row is not None:
                                contextual_row[target] = (
                                    contextual_row.get(target, 0) + 1
                                )
                                residual_totals[context_key] += 1
                            elif -log2(max(basal, 1e-300)) >= surprise_bits:
                                nursery_row = context_nursery.setdefault(
                                    context_key, {}
                                )
                                nursery_row[target] = nursery_row.get(target, 0) + 1
                                nursery_total = sum(nursery_row.values())
                                if nursery_total >= context_confirmations:
                                    residual_counts[context_key] = nursery_row
                                    residual_totals[context_key] = nursery_total
                                    del context_nursery[context_key]
                                    context_births += 1
                        unigram_counts[target] += 1
                        if previous is not None:
                            transition_row = transition_counts[previous]
                            transition_row[target] = transition_row.get(target, 0) + 1
                            context_totals[previous] += 1
                        for bank_index, context in enumerate(contexts):
                            if tracked_sequences is not None:
                                tracked_sequences[bank_index].add(context)
                            sequence_row = sequence_counts[bank_index].get(context)
                            if sequence_row is not None:
                                sequence_row[target] = sequence_row.get(target, 0) + 1
                                sequence_totals[bank_index][context] += 1
                                continue
                            nursery_bank = sequence_nursery[bank_index]
                            nursery = nursery_bank.get(context)
                            if nursery is None:
                                if len(nursery_bank) >= max_nursery:
                                    del nursery_bank[next(iter(nursery_bank))]
                                    sequence_evictions += 1
                                nursery = {}
                                nursery_bank[context] = nursery
                            nursery[target] = nursery.get(target, 0) + 1
                            nursery_total = sum(nursery.values())
                            if nursery_total >= sequence_confirmations:
                                sequence_counts[bank_index][context] = nursery
                                sequence_totals[bank_index][context] = nursery_total
                                del nursery_bank[context]
                                sequence_births += 1
                    older = previous
                    previous = target
                    if packed_paths:
                        packed_history = (
                            (packed_history << 8) | target
                        ) & history_mask
                        history_length = min(max_context, history_length + 1)
                    else:
                        history.append(target)
                        if len(history) > max_context:
                            del history[0]
                        history_length = len(history)
                else:
                    older = None
                    previous = None
                    packed_history = 0
                    history_length = 0
                    if not packed_paths:
                        history.clear()
                steps += 1
            elif live_assemblies[target]:
                older = previous
                previous = target
                if packed_paths:
                    packed_history = ((packed_history << 8) | target) & history_mask
                    history_length = min(max_context, history_length + 1)
                else:
                    history.append(target)
                    if len(history) > max_context:
                        del history[0]
                    history_length = len(history)
            else:
                older = None
                previous = None
                packed_history = 0
                history_length = 0
                if not packed_paths:
                    history.clear()

        if packed_paths:
            history[:] = [
                (packed_history >> (8 * shift)) & 0xFF
                for shift in range(history_length - 1, -1, -1)
            ]
        self.previous_assembly = previous
        self.older_assembly = older
        self.steps = steps
        self.context_births = context_births
        self.sequence_births = sequence_births
        self.sequence_nursery_evictions = sequence_evictions
        return observed

    def reset_episode(self) -> None:
        super().reset_episode()
        self.sequence_history.clear()
        self._long_range_history.clear()

    def shed_sequence_nursery(self) -> int:
        """Release globally unconfirmed paths after developmental sleep."""
        shed = self.sequence_nursery_count
        self.sequence_nursery = [{} for _ in self.sequence_nursery]
        return shed

    def consolidate_to_byte_budget(self, max_bytes: int) -> dict[str, int]:
        """Sleep weak temporal paths until the mature body fits ``max_bytes``.

        Sequence paths compete by repeated target support first, total use
        second, and depth last. This retains repeatedly reinforced,
        discriminative paths while preferentially shedding weak deep paths.
        Exact embodied size is remeasured after each pruning batch, so the
        byte class is a physical bound rather than a path-count proxy.

        Provisional nursery paths are no longer wiped here. They used to be
        (`shed_sequence_nursery()` was called unconditionally on every sleep,
        including consolidate_delta's own per-round call), but
        consolidate_delta already merges nursery observations across columns
        correctly -- two columns that each see a novel context twice, below
        `sequence_confirmations` individually, jointly promote it to mature
        once merged (verified directly; see
        tests/test_dendritic_basin_opponent_byte_stream.py). Wiping the
        nursery on every sleep threw that capability away before it ever
        had a chance to matter: with sleep firing every `sleep_bytes` chunk
        (far more often than once per epoch), a candidate context almost
        never accumulated enough repeated observations in one window to
        survive. That is the structural reason a long run's
        mature_sequence_paths count would go essentially static soon after
        first reaching the byte cap -- no candidate ever got a real chance
        to compete for a slot, unlike the vision organism's assemblies,
        which are born immediately (no confirmation delay) and keep
        competing for capacity by utility every sleep. Once the nursery
        persists, that same "weakest mature path loses its slot to a
        stronger new candidate" dynamic falls out of the existing
        byte-budget pruning below for free, without inventing a separate
        utility-ranking scheme: a newly-confirmed candidate promotes into
        `sequence_counts`, pushes embodied_bytes over budget, and the loop
        below evicts the weakest existing mature path to make room.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        before = int(self.embodied_bytes)
        self.reset_episode()
        shed_nursery = 0
        after_nursery = int(self.embodied_bytes)
        current = after_nursery
        pruned = 0
        passes = 0
        while current > max_bytes:
            if self.connectivity_pruning:
                # Byte-cortex analog of the vision organism's relation-
                # degree utility (degree*(capacity+1) + observations*
                # confidence in continual_vision.py's sleep()): a well-
                # connected assembly there is one many others co-activate
                # with, so removing it disrupts many relationships at once.
                # The direct equivalent here is a context many longer,
                # deeper mature contexts share as their trailing backoff
                # suffix -- a hub in the backoff chain. Removing it forces
                # every context relying on it to fall back one level
                # further than it should have to. Degree is the primary
                # ranking key (ascending, least-connected pruned first),
                # with the existing count-based ranking as a tiebreaker
                # among equally-connected contexts, mirroring how vision's
                # formula uses observations*confidence only to break ties
                # within the same degree.
                degree: list[dict[object, int]] = [
                    {} for _ in self.sequence_counts
                ]
                for bank_index in range(len(self.sequence_counts) - 1):
                    depth = bank_index + 3
                    mask = (1 << (8 * depth)) - 1
                    parent_degree = degree[bank_index]
                    for child_context in self.sequence_counts[bank_index + 1]:
                        suffix = (
                            child_context & mask
                            if self.packed_paths
                            else child_context[-depth:]
                        )
                        parent_degree[suffix] = parent_degree.get(suffix, 0) + 1

                live = self.live_sequence_count
                if live < 1:
                    raise ValueError(
                        "fixed cortical anatomy exceeds the requested byte budget"
                    )
                mean_path_bytes = max(8.0, current / live)
                needed = max(
                    1,
                    min(
                        live,
                        ceil((current - max_bytes) / mean_path_bytes * 1.05),
                    ),
                )

                def connectivity_candidates():
                    for bank_index, bank in enumerate(self.sequence_counts):
                        parent_degree = degree[bank_index]
                        for context, row in bank.items():
                            strongest = max(row.values())
                            total = self.sequence_totals[bank_index][context]
                            stable_context = (
                                int(context) if isinstance(context, int) else context
                            )
                            yield (
                                parent_degree.get(context, 0),
                                strongest,
                                total,
                                -bank_index,
                                stable_context,
                                bank_index,
                                context,
                            )

                weakest = heapq.nsmallest(needed, connectivity_candidates())
                if not weakest:
                    raise RuntimeError("no temporal anatomy remains available to prune")
                for *_, bank_index, context in weakest:
                    del self.sequence_counts[bank_index][context]
                    del self.sequence_totals[bank_index][context]
                pruned += len(weakest)
                passes += 1
                current = int(self.embodied_bytes)
                continue

            if self.structural_support:
                # Whole-row competition, ranked by structural support
                # instead of raw count. Edge-level pruning (deleting
                # individual weak (context, target) connections out of an
                # otherwise-good row) was tried first and measured worse
                # than the plain count-based baseline even at 3x the
                # retained edges: a row's surviving counts feed directly
                # into the Bayesian backoff formula's denominator
                # (sequence_totals), which implicitly assumes a row
                # represents the *complete* locally observed distribution.
                # A row missing some of its natural targets no longer does,
                # which distorts backoff weighting for every prediction
                # from that context, not just the pruned targets -- a
                # row-level ablation using the same support signal (whole
                # rows kept or dropped, never partially edited) closed most
                # of that gap. So a row's fate is decided by its single
                # strongest edge (does this context have *some* reliable,
                # well-confirmed predictive value), matching the spirit of
                # "does this context still earn a slot" rather than
                # "which specific edges earn a slot."
                #
                # Support alone (no volume term) also measured worse than
                # count-based pruning: a single lucky observation reaches a
                # support level a well-established, high-volume edge would
                # need dozens of confirmations to match, and held-out
                # prediction quality is dominated by the common, high-volume
                # cases. Scaling support by log(1+count) requires both a
                # real predictive edge over baseline *and* enough repeated
                # confirmation to trust that estimate.
                live = self.live_sequence_count
                if live < 1:
                    raise ValueError(
                        "fixed cortical anatomy exceeds the requested byte budget"
                    )
                mean_path_bytes = max(8.0, current / live)
                needed = max(
                    1,
                    min(
                        live,
                        ceil((current - max_bytes) / mean_path_bytes * 1.05),
                    ),
                )

                def support_candidates():
                    for bank_index, bank in enumerate(self.sequence_counts):
                        support_bank = self.sequence_support[bank_index]
                        for context, row in bank.items():
                            support_row = support_bank.get(context, {})
                            utility = max(
                                support_row.get(target, 0.0) * log1p(count)
                                for target, count in row.items()
                            )
                            stable_context = (
                                int(context) if isinstance(context, int) else context
                            )
                            yield (
                                utility,
                                -bank_index,
                                stable_context,
                                bank_index,
                                context,
                            )

                weakest = heapq.nsmallest(needed, support_candidates())
                if not weakest:
                    raise RuntimeError("no temporal anatomy remains available to prune")
                for *_, bank_index, context in weakest:
                    del self.sequence_counts[bank_index][context]
                    del self.sequence_totals[bank_index][context]
                    self.sequence_support[bank_index].pop(context, None)
                pruned += len(weakest)
                passes += 1
                current = int(self.embodied_bytes)
                continue

            live = self.live_sequence_count
            if live < 1:
                raise ValueError(
                    "fixed cortical anatomy exceeds the requested byte budget"
                )
            # Retain only the weakest estimated batch in memory instead of
            # sorting a second copy of the complete million-path anatomy.
            mean_path_bytes = max(8.0, current / live)
            needed = max(
                1,
                min(
                    live,
                    ceil((current - max_bytes) / mean_path_bytes * 1.05),
                ),
            )

            def candidates():
                for bank_index, bank in enumerate(self.sequence_counts):
                    for context, row in bank.items():
                        total = self.sequence_totals[bank_index][context]
                        strongest = max(row.values())
                        # A deeper weak path loses a tie before a shallower one.
                        stable_context = (
                            int(context) if isinstance(context, int) else context
                        )
                        yield (
                            strongest,
                            total,
                            -bank_index,
                            stable_context,
                            bank_index,
                            context,
                        )

            weakest = heapq.nsmallest(needed, candidates())
            if not weakest:
                raise RuntimeError("no temporal anatomy remains available to prune")
            for *_, bank_index, context in weakest:
                del self.sequence_counts[bank_index][context]
                del self.sequence_totals[bank_index][context]
            pruned += len(weakest)
            passes += 1
            current = int(self.embodied_bytes)
        self.sequence_births = self.live_sequence_count
        return {
            "body_bytes_before_sleep": before,
            "body_bytes_after_nursery_shed": after_nursery,
            "body_bytes_after_sleep": current,
            "shed_nursery_paths": shed_nursery,
            "pruned_mature_paths": pruned,
            "competition_passes": passes,
        }

    def begin_lineage_tracking(self) -> None:
        """Record only temporal paths changed by this awake column."""
        super().begin_lineage_tracking()
        self._lineage_sequence_keys: list[set[object]] = [
            set() for _ in self.sequence_counts
        ]

    def consolidate(self, other: SequenceCortexOpponentLearner) -> None:
        """Merge sparse temporal paths from another cortical territory."""
        configuration = (
            "max_context_bytes",
            "sequence_confirmations",
            "parent_strength",
            "packed_paths",
            "max_sequence_nursery_paths",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate sequence cortices with different configuration"
            )
        super().consolidate(other)

        def add_row(target: dict[int, int], source: dict[int, int]) -> None:
            for output, observations in source.items():
                target[output] = target.get(output, 0) + observations

        for bank_index, incoming_bank in enumerate(other.sequence_counts):
            local_bank = self.sequence_counts[bank_index]
            local_totals = self.sequence_totals[bank_index]
            local_nursery = self.sequence_nursery[bank_index]
            for context, incoming in incoming_bank.items():
                row = local_bank.get(context)
                if row is None:
                    row = local_nursery.pop(context, {})
                    local_bank[context] = row
                add_row(row, incoming)
                local_totals[context] = sum(row.values())
            for context, incoming in other.sequence_nursery[bank_index].items():
                row = local_bank.get(context)
                if row is not None:
                    add_row(row, incoming)
                    local_totals[context] += sum(incoming.values())
                    continue
                if (
                    context not in local_nursery
                    and len(local_nursery) >= self.max_sequence_nursery_paths
                ):
                    del local_nursery[next(iter(local_nursery))]
                    self.sequence_nursery_evictions += 1
                row = local_nursery.setdefault(context, {})
                add_row(row, incoming)
                if sum(row.values()) >= self.sequence_confirmations:
                    local_bank[context] = local_nursery.pop(context)
                    local_totals[context] = sum(local_bank[context].values())
        self.sequence_births = self.live_sequence_count
        self.sequence_nursery_evictions += other.sequence_nursery_evictions
        self.reset_episode()

    def consolidate_delta(
        self,
        descendant: SequenceCortexOpponentLearner,
        ancestor: SequenceCortexOpponentLearner,
    ) -> None:
        """Merge one awake column without duplicating its shared ancestry.

        Known gap when structural_support=True: sequence_support is not
        merged here (only sequence_counts/sequence_nursery are). A merged-in
        edge that only exists because of this delta will read back with
        support 0.0 (an edge_candidates() default) until this column's own
        subsequent observations credit it, so it is likelier than it should
        be to be pruned first after a multi-column merge. Acceptable for a
        first, single-process validation of the mechanism; needs a proper
        fix (merging support the same way sequence_nursery's counts already
        are) before this is safe to run through the real multi-column
        wake/sleep harness.
        """
        configuration = (
            "max_context_bytes",
            "sequence_confirmations",
            "parent_strength",
            "packed_paths",
            "max_sequence_nursery_paths",
        )
        if any(
            getattr(self, name) != getattr(descendant, name)
            or getattr(self, name) != getattr(ancestor, name)
            for name in configuration
        ):
            raise ValueError("cannot consolidate incompatible sequence lineage")
        super().consolidate_delta(descendant, ancestor)
        for bank_index in range(len(self.sequence_counts)):
            tracked_banks = getattr(descendant, "_lineage_sequence_keys", None)
            touched = (
                None if tracked_banks is None else tracked_banks[bank_index]
            )
            descendant_rows = self._combined_sparse_rows(
                descendant.sequence_counts[bank_index],
                descendant.sequence_nursery[bank_index],
                touched,
            )
            ancestor_rows = self._combined_sparse_rows(
                ancestor.sequence_counts[bank_index],
                ancestor.sequence_nursery[bank_index],
                touched,
            )
            delta_rows = self._sparse_row_delta(descendant_rows, ancestor_rows)
            descendant_live = set(descendant.sequence_counts[bank_index])
            local_live = self.sequence_counts[bank_index]
            local_nursery = self.sequence_nursery[bank_index]
            local_totals = self.sequence_totals[bank_index]
            for key, increments in delta_rows.items():
                if key in local_live:
                    row = local_live[key]
                else:
                    # _update_sequences enforces max_sequence_nursery_paths
                    # with the same FIFO eviction on every single-byte
                    # nursery insertion; this merge path has to match it --
                    # previously masked by the nursery being wiped on every
                    # sleep regardless (consolidate_to_byte_budget used to
                    # call shed_sequence_nursery() unconditionally), so an
                    # uncapped nursery here never had a chance to be
                    # visible. Now that the nursery persists, a generation's
                    # worth of delta rows can be tens of thousands of novel
                    # contexts across several columns, and without this
                    # check the nursery grows unbounded past its configured
                    # cap.
                    if (
                        key not in local_nursery
                        and len(local_nursery) >= self.max_sequence_nursery_paths
                    ):
                        del local_nursery[next(iter(local_nursery))]
                        self.sequence_nursery_evictions += 1
                    row = local_nursery.setdefault(key, {})
                for target, increment in increments.items():
                    row[target] = row.get(target, 0) + increment
                total = sum(row.values())
                if key in descendant_live or total >= self.sequence_confirmations:
                    local_live[key] = local_nursery.pop(key, row)
                    local_totals[key] = total
                elif key in local_live:
                    local_totals[key] = total
        eviction_delta = (
            descendant.sequence_nursery_evictions
            - ancestor.sequence_nursery_evictions
        )
        if eviction_delta < 0:
            raise ValueError("descendant anatomy predates its ancestor")
        self.sequence_nursery_evictions += eviction_delta
        self.sequence_births = self.live_sequence_count
        if self.long_range_bank:
            self._merge_long_range_delta(descendant, ancestor)
        self.reset_episode()

    def _merge_long_range_delta(
        self,
        descendant: SequenceCortexOpponentLearner,
        ancestor: SequenceCortexOpponentLearner,
    ) -> None:
        """Without this, long_range_bank is silently inert in the real
        multi-column harness: each wake round starts every column from a
        deep copy of the ancestor, so anything a column's long_range_counts
        grew during its own wake period would otherwise simply be discarded
        the moment the round's columns are merged back (only
        sequence_counts/sequence_nursery were merged before this existed --
        found and fixed before this mechanism was ever run through that
        harness for real, not after). Reuses the same ancestor-relative
        delta as sequence_counts's own merge.

        Deliberately does *not* re-apply the burst gate here, even though
        ``_grow_long_range`` does for ordinary per-byte growth: the
        descendant already made that novelty call correctly, using *its
        own* burst_fraction, at the moment it decided to create the key.
        The master's burst_fraction reflects nothing meaningful at merge
        time -- a master never trains directly, only merges -- so gating on
        it here was a real bug: a master seeded from a cold ancestor has
        burst_fraction == 0.0 forever, always below any positive threshold,
        silently discarding every descendant's long-range growth on every
        single merge (caught via a production run where long_range_bank
        stayed inert through the entire multi-column harness despite
        working correctly in every single-process test). The FIFO cap
        below still applies -- that bounds *this* table's size, unlike the
        gate, which was answering "was this novel" with the wrong
        learner's state.
        """
        delta_rows = self._sparse_row_delta(
            descendant.long_range_counts, ancestor.long_range_counts
        )
        for key, increments in delta_rows.items():
            is_new_key = key not in self.long_range_counts
            if is_new_key and len(self.long_range_counts) >= self.long_range_max_entries:
                victim = next(iter(self.long_range_counts))
                del self.long_range_counts[victim]
                del self.long_range_totals[victim]
            row = self.long_range_counts.setdefault(key, {})
            for target, increment in increments.items():
                row[target] = row.get(target, 0) + increment
            self.long_range_totals[key] = sum(row.values())


class PhraseEventOpponentLearner(SequenceCortexOpponentLearner):
    """Emergent word assemblies that return apical byte-completion signals.

    Words are discovered directly between ASCII letter boundaries.  Recent
    normalized word assemblies form sparse event context, while predicted raw
    word forms modulate the lower byte cortex only after a matching prefix has
    begun.  The lower sequence hierarchy therefore remains the prior and an
    incorrect event hypothesis can abstain as soon as its prefix diverges.
    """

    story_boundary = b"<|endoftext|>\n"

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        max_context_bytes: int = 6,
        sequence_confirmations: int = 3,
        parent_strength: float = 16.0,
        packed_paths: bool = True,
        max_sequence_nursery_paths: int = 50_000,
        max_event_words: int = 2,
        event_confirmations: int = 2,
        max_event_nursery_paths: int = 50_000,
        apical_strength: float = 8.0,
        maximum_apical_weight: float = 0.5,
        minimum_prefix_bytes: int = 1,
        max_word_bytes: int = 32,
    ) -> None:
        super().__init__(
            prior=prior,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
            max_context_bytes=max_context_bytes,
            sequence_confirmations=sequence_confirmations,
            parent_strength=parent_strength,
            packed_paths=packed_paths,
            max_sequence_nursery_paths=max_sequence_nursery_paths,
        )
        if max_event_words < 1 or event_confirmations < 1:
            raise ValueError("event depth and confirmations must be positive")
        if max_event_nursery_paths < 1:
            raise ValueError("event nursery capacity must be positive")
        if apical_strength <= 0.0:
            raise ValueError("apical_strength must be positive")
        if not 0.0 < maximum_apical_weight <= 1.0:
            raise ValueError("maximum_apical_weight must be in (0, 1]")
        if minimum_prefix_bytes < 1 or max_word_bytes < minimum_prefix_bytes:
            raise ValueError("word and prefix lengths are inconsistent")
        self.max_event_words = max_event_words
        self.event_confirmations = event_confirmations
        self.max_event_nursery_paths = max_event_nursery_paths
        self.apical_strength = apical_strength
        self.maximum_apical_weight = maximum_apical_weight
        self.minimum_prefix_bytes = minimum_prefix_bytes
        self.max_word_bytes = max_word_bytes
        self.word_forms: list[bytes] = []
        self.word_ids: dict[bytes, int] = {}
        self.event_counts: list[dict[int, dict[int, int]]] = [
            {} for _ in range(max_event_words)
        ]
        self.event_totals: list[dict[int, int]] = [
            {} for _ in self.event_counts
        ]
        self.event_nursery: list[dict[int, dict[int, int]]] = [
            {} for _ in self.event_counts
        ]
        self.event_history: list[int] = []
        self.current_word = bytearray()
        self.active_word_candidates: dict[int, int] | None = None
        self.event_receptor_cache: dict[
            tuple[int, int], tuple[dict[int, int], dict[int, int], int]
        ] = {}
        self.max_event_receptor_cache_entries = 16_384
        self.boundary_window = bytearray()
        self.event_births = 0
        self.event_nursery_evictions = 0

    @property
    def live_event_count(self) -> int:
        return sum(len(bank) for bank in self.event_counts)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["event_receptor_cache"] = {}
        return state

    @property
    def live_event_synapse_count(self) -> int:
        return sum(len(row) for bank in self.event_counts for row in bank.values())

    @property
    def event_nursery_count(self) -> int:
        return sum(len(bank) for bank in self.event_nursery)

    @property
    def live_word_count(self) -> int:
        return len(self.word_forms)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.max_event_words,
            self.event_confirmations,
            self.max_event_nursery_paths,
            self.apical_strength,
            self.maximum_apical_weight,
            self.minimum_prefix_bytes,
            self.max_word_bytes,
            self.word_forms,
            self.word_ids,
            self.event_counts,
            self.event_totals,
            self.event_nursery,
            self.event_history,
            self.current_word,
            self.active_word_candidates,
            self.boundary_window,
            self.event_births,
            self.event_nursery_evictions,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _word_id(self, word: bytes, *, create: bool) -> int | None:
        word_id = self.word_ids.get(word)
        if word_id is not None or not create:
            return word_id
        word_id = len(self.word_forms)
        if word_id >= 1 << 32:
            raise OverflowError("word assembly space exhausted")
        self.word_ids[word] = word_id
        self.word_forms.append(word)
        return word_id

    @staticmethod
    def _pack_event_context(word_ids: tuple[int, ...]) -> int:
        packed = 0
        for word_id in word_ids:
            packed = (packed << 32) | word_id
        return packed

    @staticmethod
    def _unpack_event_context(packed: int, depth: int) -> tuple[int, ...]:
        word_ids = []
        for _ in range(depth):
            word_ids.append(packed & 0xFFFFFFFF)
            packed >>= 32
        return tuple(reversed(word_ids))

    def _active_event_row(self) -> dict[int, int] | None:
        available = min(len(self.event_history), self.max_event_words)
        for depth in range(available, 0, -1):
            word_ids = tuple(self.event_history[-depth:])
            if any(word_id < 0 for word_id in word_ids):
                continue
            context = self._pack_event_context(word_ids)
            row = self.event_counts[depth - 1].get(context)
            if row is not None:
                return row
        return None

    def _apical_byte_counts(self) -> tuple[dict[int, int], int] | None:
        prefix_length = len(self.current_word)
        if prefix_length < self.minimum_prefix_bytes:
            return None
        opened = self.active_word_candidates is None
        if opened:
            row = self._active_event_row()
            if row is None:
                return None
            cache = getattr(self, "event_receptor_cache", None)
            if cache is None:
                cache = self.event_receptor_cache = {}
            cache_key = (id(row), self.current_word[0])
            cached = cache.get(cache_key)
            if cached is not None:
                candidates, cached_counts, cached_total = cached
                if prefix_length == 1:
                    self.active_word_candidates = candidates
                    return cached_counts, cached_total
            else:
                first_byte = self.current_word[0]
                candidates = {
                    word_id: observations
                    for word_id, observations in row.items()
                    if len(self.word_forms[word_id]) > 1
                    and self.word_forms[word_id][0] == first_byte
                }
            if prefix_length > 1:
                prefix = bytes(self.current_word)
                candidates = {
                    word_id: observations
                    for word_id, observations in candidates.items()
                    if len(self.word_forms[word_id]) > prefix_length
                    and self.word_forms[word_id].startswith(prefix)
                }
        else:
            newest_offset = prefix_length - 1
            newest_byte = self.current_word[-1]
            candidates = {
                word_id: observations
                for word_id, observations in self.active_word_candidates.items()
                if len(self.word_forms[word_id]) > prefix_length
                and self.word_forms[word_id][newest_offset] == newest_byte
            }
        self.active_word_candidates = candidates
        counts: dict[int, int] = {}
        for word_id, observations in candidates.items():
            word = self.word_forms[word_id]
            next_byte = word[prefix_length]
            counts[next_byte] = counts.get(next_byte, 0) + observations
        if not counts:
            return None
        total = sum(counts.values())
        if opened and prefix_length == 1:
            capacity = getattr(self, "max_event_receptor_cache_entries", 16_384)
            if len(cache) >= capacity:
                del cache[next(iter(cache))]
            cache[cache_key] = candidates, counts, total
        return counts, total

    def _observed_probability(self, byte_id: int) -> float:
        baseline = super()._observed_probability(byte_id)
        apical = self._apical_byte_counts()
        if apical is None:
            return baseline
        counts, total = apical
        weight = min(
            self.maximum_apical_weight,
            total / (total + self.apical_strength),
        )
        event_probability = counts.get(byte_id, 0) / total
        return float((1.0 - weight) * baseline + weight * event_probability)

    def _distribution(self) -> np.ndarray:
        baseline = super()._distribution()
        apical = self._apical_byte_counts()
        if apical is None:
            return baseline
        counts, total = apical
        weight = min(
            self.maximum_apical_weight,
            total / (total + self.apical_strength),
        )
        event_distribution = np.zeros(256, dtype=np.float64)
        for byte_id, observations in counts.items():
            event_distribution[byte_id] = observations / total
        return (1.0 - weight) * baseline + weight * event_distribution

    def _event_contexts(self) -> tuple[int, ...]:
        available = min(len(self.event_history), self.max_event_words)
        return tuple(
            self._pack_event_context(tuple(self.event_history[-depth:]))
            for depth in range(1, available + 1)
        )

    def _normalize_event_form(self, raw_form: bytes) -> bytes:
        return raw_form.lower()

    def _complete_word(self, *, learn: bool) -> None:
        if not self.current_word:
            return
        raw_word = bytes(self.current_word)
        normalized = self._normalize_event_form(raw_word)
        normalized_id = self._word_id(normalized, create=learn)
        if learn:
            getattr(self, "event_receptor_cache", {}).clear()
            raw_word_id = (
                normalized_id
                if raw_word == normalized
                else self._word_id(raw_word, create=True)
            )
            if raw_word_id is not None:
                for bank_index, context in enumerate(self._event_contexts()):
                    if context < 0:
                        continue
                    row = self.event_counts[bank_index].get(context)
                    if row is not None:
                        row[raw_word_id] = row.get(raw_word_id, 0) + 1
                        self.event_totals[bank_index][context] += 1
                        continue
                    nursery_bank = self.event_nursery[bank_index]
                    if (
                        context not in nursery_bank
                        and len(nursery_bank) >= self.max_event_nursery_paths
                    ):
                        del nursery_bank[next(iter(nursery_bank))]
                        self.event_nursery_evictions += 1
                    nursery = nursery_bank.setdefault(context, {})
                    nursery[raw_word_id] = nursery.get(raw_word_id, 0) + 1
                    if sum(nursery.values()) >= self.event_confirmations:
                        self.event_counts[bank_index][context] = nursery
                        self.event_totals[bank_index][context] = sum(nursery.values())
                        del self.event_nursery[bank_index][context]
                        self.event_births += 1
        self.event_history.append(-1 if normalized_id is None else normalized_id)
        excess = len(self.event_history) - self.max_event_words
        if excess > 0:
            del self.event_history[:excess]
        self.current_word.clear()

    def _advance_event_state(self, byte_id: int, *, learn: bool) -> None:
        is_letter = 65 <= byte_id <= 90 or 97 <= byte_id <= 122
        if is_letter:
            if len(self.current_word) < self.max_word_bytes:
                self.current_word.append(byte_id)
            else:
                self.current_word.clear()
                self.active_word_candidates = None
        else:
            self._complete_word(learn=learn)
            self.active_word_candidates = None
        self.boundary_window.append(byte_id)
        if len(self.boundary_window) > len(self.story_boundary):
            del self.boundary_window[0]
        if bytes(self.boundary_window) == self.story_boundary:
            self.event_history.clear()
            self.current_word.clear()
            self.active_word_candidates = None
            self.boundary_window.clear()

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        super()._learn_and_advance(byte_id, basal)
        self._advance_event_state(byte_id, learn=True)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        step = super().observe(byte_id, learn=learn)
        if not learn:
            self._advance_event_state(int(byte_id), learn=False)
        return step

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        probability = super()._observe_probability_only(byte_id, learn=learn)
        if not learn:
            self._advance_event_state(byte_id, learn=False)
        return probability

    def reset_episode(self) -> None:
        super().reset_episode()
        self.event_history.clear()
        self.current_word.clear()
        self.active_word_candidates = None
        cache = getattr(self, "event_receptor_cache", None)
        if cache is None:
            self.event_receptor_cache = {}
            self.max_event_receptor_cache_entries = 16_384
        else:
            cache.clear()
        self.boundary_window.clear()

    def shed_event_nursery(self) -> int:
        """Release word-event candidates that never became predictive."""
        shed = self.event_nursery_count
        self.event_nursery = [{} for _ in self.event_nursery]
        return shed

    def consolidate_event_anatomy(
        self, *, minimum_observations: int = 3
    ) -> tuple[int, int]:
        """Let weak event dendrites lose the mature branch competition."""
        if minimum_observations < 1:
            raise ValueError("minimum observations must be positive")
        removed_branches = 0
        removed_synapses = 0
        for bank_index, bank in enumerate(self.event_counts):
            totals = self.event_totals[bank_index]
            for context, row in list(bank.items()):
                if totals[context] >= minimum_observations:
                    continue
                removed_branches += 1
                removed_synapses += len(row)
                del bank[context]
                del totals[context]
        getattr(self, "event_receptor_cache", {}).clear()
        self.event_births = self.live_event_count
        return removed_branches, removed_synapses

    def consolidate(self, other: PhraseEventOpponentLearner) -> None:
        """Merge word-event anatomy from another cortical territory."""
        configuration = (
            "max_event_words",
            "event_confirmations",
            "max_event_nursery_paths",
            "apical_strength",
            "maximum_apical_weight",
            "minimum_prefix_bytes",
            "max_word_bytes",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate phrase cortices with different configuration"
            )
        super().consolidate(other)
        incoming_to_local: list[int | None] = []
        for word in other.word_forms:
            word_id = self._word_id(word, create=True)
            incoming_to_local.append(word_id)

        def remap_context(context: int, depth: int) -> int | None:
            incoming_ids = other._unpack_event_context(context, depth)
            if any(word_id >= len(incoming_to_local) for word_id in incoming_ids):
                return None
            mapped = tuple(incoming_to_local[word_id] for word_id in incoming_ids)
            if any(word_id is None for word_id in mapped):
                return None
            return self._pack_event_context(tuple(int(word_id) for word_id in mapped))

        def add_row(target: dict[int, int], source: dict[int, int]) -> int:
            accepted = 0
            for incoming_id, observations in source.items():
                word_id = incoming_to_local[incoming_id]
                if word_id is None:
                    continue
                target[word_id] = target.get(word_id, 0) + observations
                accepted += observations
            return accepted

        for bank_index, incoming_bank in enumerate(other.event_counts):
            live = self.event_counts[bank_index]
            totals = self.event_totals[bank_index]
            nursery = self.event_nursery[bank_index]
            depth = bank_index + 1
            for incoming_context, incoming in incoming_bank.items():
                context = remap_context(incoming_context, depth)
                if context is None:
                    continue
                row = live.get(context)
                if row is None:
                    row = nursery.pop(context, {})
                    live[context] = row
                add_row(row, incoming)
                totals[context] = sum(row.values())
            for incoming_context, incoming in other.event_nursery[bank_index].items():
                context = remap_context(incoming_context, depth)
                if context is None:
                    continue
                row = live.get(context)
                if row is not None:
                    totals[context] += add_row(row, incoming)
                    continue
                if (
                    context not in nursery
                    and len(nursery) >= self.max_event_nursery_paths
                ):
                    del nursery[next(iter(nursery))]
                    self.event_nursery_evictions += 1
                row = nursery.setdefault(context, {})
                add_row(row, incoming)
                if sum(row.values()) >= self.event_confirmations:
                    live[context] = nursery.pop(context)
                    totals[context] = sum(live[context].values())
        self.event_births = self.live_event_count
        self.event_nursery_evictions += other.event_nursery_evictions
        self.reset_episode()


class PredictiveMotifOpponentLearner(PhraseEventOpponentLearner):
    """Modality-neutral motifs segmented by lower-cortex prediction error.

    The inherited event tissue operates on arbitrary byte-valued receptor
    identities.  Unlike the language adapter, no receptor has privileged
    boundary meaning and no ASCII category is inspected.  A large local
    prediction error closes the preceding motif after a minimum duration;
    a maximum duration prevents an indefinitely stable state from consuming
    unbounded working memory.
    """

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        max_context_bytes: int = 6,
        sequence_confirmations: int = 3,
        parent_strength: float = 16.0,
        packed_paths: bool = True,
        max_sequence_nursery_paths: int = 50_000,
        max_event_motifs: int = 2,
        event_confirmations: int = 2,
        max_event_nursery_paths: int = 50_000,
        apical_strength: float = 8.0,
        maximum_apical_weight: float = 0.5,
        minimum_prefix_events: int = 1,
        boundary_surprise_bits: float = 3.0,
        minimum_motif_events: int = 3,
        maximum_motif_events: int = 16,
    ) -> None:
        super().__init__(
            prior=prior,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
            max_context_bytes=max_context_bytes,
            sequence_confirmations=sequence_confirmations,
            parent_strength=parent_strength,
            packed_paths=packed_paths,
            max_sequence_nursery_paths=max_sequence_nursery_paths,
            max_event_words=max_event_motifs,
            event_confirmations=event_confirmations,
            max_event_nursery_paths=max_event_nursery_paths,
            apical_strength=apical_strength,
            maximum_apical_weight=maximum_apical_weight,
            minimum_prefix_bytes=minimum_prefix_events,
            max_word_bytes=maximum_motif_events,
        )
        if boundary_surprise_bits < 0.0:
            raise ValueError("boundary_surprise_bits must be non-negative")
        if minimum_motif_events < 1:
            raise ValueError("minimum_motif_events must be positive")
        if maximum_motif_events < minimum_motif_events:
            raise ValueError("maximum motif length must cover the minimum")
        self.boundary_surprise_bits = boundary_surprise_bits
        self.minimum_motif_events = minimum_motif_events
        self.maximum_motif_events = maximum_motif_events
        self.learned_boundaries = 0

    @property
    def live_motif_count(self) -> int:
        return self.live_word_count

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.boundary_surprise_bits,
            self.minimum_motif_events,
            self.maximum_motif_events,
            self.learned_boundaries,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _normalize_event_form(self, raw_form: bytes) -> bytes:
        return raw_form

    def _advance_motif_state(
        self, byte_id: int, baseline: float, *, learn: bool
    ) -> None:
        length = len(self.current_word)
        boundary = length >= self.maximum_motif_events or (
            length >= self.minimum_motif_events
            and -log2(max(baseline, 1e-300)) >= self.boundary_surprise_bits
        )
        if boundary:
            self._complete_word(learn=learn)
            self.active_word_candidates = None
            if learn:
                self.learned_boundaries += 1
        self.current_word.append(byte_id)

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        baseline = SequenceCortexOpponentLearner._observed_probability(self, byte_id)
        SequenceCortexOpponentLearner._learn_and_advance(self, byte_id, basal)
        self._advance_motif_state(byte_id, baseline, learn=True)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        byte_id = int(byte_id)
        baseline = SequenceCortexOpponentLearner._observed_probability(self, byte_id)
        step = SequenceCortexOpponentLearner.observe(self, byte_id, learn=learn)
        if not learn:
            self._advance_motif_state(byte_id, baseline, learn=False)
        return step

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        baseline = SequenceCortexOpponentLearner._observed_probability(self, byte_id)
        probability = SequenceCortexOpponentLearner._observe_probability_only(
            self, byte_id, learn=learn
        )
        if not learn:
            self._advance_motif_state(byte_id, baseline, learn=False)
        return probability

    def consolidate(self, other: PredictiveMotifOpponentLearner) -> None:
        configuration = (
            "boundary_surprise_bits",
            "minimum_motif_events",
            "maximum_motif_events",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate motif cortices with different configuration"
            )
        super().consolidate(other)
        self.learned_boundaries += other.learned_boundaries


class DelayedOpponentAttractorControl(ContextualOpponentAttractorLearner):
    """Fixed-delay association control; distance alone is not coherence."""

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        fiber_lags: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
        fiber_confirmations: int = 3,
        utility_decay: float = 0.99,
    ) -> None:
        super().__init__(
            prior=prior,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
        )
        if not fiber_lags or any(lag < 3 for lag in fiber_lags):
            raise ValueError("fiber lags must contain values of at least three")
        if fiber_confirmations < 1:
            raise ValueError("fiber_confirmations must be positive")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        self.fiber_lags = tuple(sorted(set(fiber_lags)))
        self.fiber_confirmations = fiber_confirmations
        self.utility_decay = utility_decay
        self.delay_line = np.full(max(self.fiber_lags), -1, dtype=np.int16)
        self.delay_position = 0
        self.delay_count = 0
        self.fiber_nursery: dict[tuple[int, int, int], dict[int, int]] = {}
        self.fiber_counts: dict[tuple[int, int, int], dict[int, int]] = {}
        self.fiber_totals: dict[tuple[int, int, int], int] = {}
        self.fiber_utility: dict[tuple[int, int, int], float] = {}
        self.fiber_births = 0

    @property
    def live_fiber_count(self) -> int:
        return len(self.fiber_counts)

    @property
    def live_fiber_synapse_count(self) -> int:
        return sum(len(row) for row in self.fiber_counts.values())

    @property
    def embodied_bytes(self) -> int:
        payload = (
            super().embodied_bytes,
            self.fiber_lags,
            self.delay_line,
            self.delay_position,
            self.delay_count,
            self.fiber_nursery,
            self.fiber_counts,
            self.fiber_totals,
            self.fiber_utility,
            self.fiber_births,
            self.fiber_confirmations,
            self.utility_decay,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _active_fiber_keys(self) -> tuple[tuple[int, int, int], ...]:
        if self.previous_assembly is None:
            return ()
        keys = []
        size = len(self.delay_line)
        for lag in self.fiber_lags:
            if self.delay_count >= lag:
                distant = int(self.delay_line[(self.delay_position - lag) % size])
                if distant >= 0:
                    keys.append((lag, distant, self.previous_assembly))
        return tuple(keys)

    def _selected_fiber(self) -> tuple[int, int, int] | None:
        live = (key for key in self._active_fiber_keys() if key in self.fiber_counts)
        key = max(live, key=lambda item: self.fiber_utility[item], default=None)
        if key is None or self.fiber_utility[key] <= 0.0:
            return None
        return key

    def _fiber_probability(
        self, key: tuple[int, int, int], byte_id: int, baseline: float
    ) -> float:
        count = self.fiber_counts[key].get(byte_id, 0)
        return float(
            (count + self.basal_strength * baseline)
            / (self.fiber_totals[key] + self.basal_strength)
        )

    def _observed_probability(self, byte_id: int) -> float:
        baseline = super()._observed_probability(byte_id)
        key = self._selected_fiber()
        if key is None:
            return baseline
        return self._fiber_probability(key, byte_id, baseline)

    def _distribution(self) -> np.ndarray:
        baseline = super()._distribution()
        key = self._selected_fiber()
        if key is None:
            return baseline
        distribution = self.basal_strength * baseline
        for target, count in self.fiber_counts[key].items():
            distribution[target] += count
        return distribution / (self.fiber_totals[key] + self.basal_strength)

    def _update_fibers(
        self,
        keys: tuple[tuple[int, int, int], ...],
        byte_id: int,
        baseline: float,
    ) -> None:
        for key in keys:
            if key in self.fiber_counts:
                probability = self._fiber_probability(key, byte_id, baseline)
                gain = log2(max(probability, 1e-300) / max(baseline, 1e-300))
                self.fiber_utility[key] = (
                    self.utility_decay * self.fiber_utility[key]
                    + (1.0 - self.utility_decay) * gain
                )
                row = self.fiber_counts[key]
                row[byte_id] = row.get(byte_id, 0) + 1
                self.fiber_totals[key] += 1
            elif -log2(max(baseline, 1e-300)) >= self.surprise_bits:
                nursery = self.fiber_nursery.setdefault(key, {})
                nursery[byte_id] = nursery.get(byte_id, 0) + 1
                if sum(nursery.values()) >= self.fiber_confirmations:
                    self.fiber_counts[key] = nursery
                    self.fiber_totals[key] = sum(nursery.values())
                    self.fiber_utility[key] = 0.0
                    del self.fiber_nursery[key]
                    self.fiber_births += 1

    def _advance_delay_line(self, byte_id: int) -> None:
        self.delay_line[self.delay_position] = byte_id
        self.delay_position = (self.delay_position + 1) % len(self.delay_line)
        self.delay_count = min(self.delay_count + 1, len(self.delay_line))

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        baseline = super()._observed_probability(byte_id)
        active_keys = self._active_fiber_keys()
        super()._learn_and_advance(byte_id, basal)
        if self.live_assemblies[byte_id]:
            self._update_fibers(active_keys, byte_id, baseline)
            self._advance_delay_line(byte_id)
        else:
            self._reset_delay_line()

    def _reset_delay_line(self) -> None:
        self.delay_line.fill(-1)
        self.delay_position = 0
        self.delay_count = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._reset_delay_line()


class EpisodicOpponentAttractorLearner(SequenceCortexOpponentLearner):
    """Content-addressable working memory above the local byte cortex.

    Recurrent byte paths bounded by non-word receptor events become content
    assemblies. A small habituating population remains active for the current
    story and competes to supply a sparse top-down residual prediction.
    """

    story_boundary = b"<|endoftext|>\n"

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        max_context_bytes: int = 6,
        sequence_confirmations: int = 3,
        parent_strength: float = 16.0,
        content_confirmations: int = 3,
        memory_capacity: int = 1,
        memory_decay: float = 0.95,
        episodic_confirmations: int = 3,
        utility_decay: float = 0.99,
        minimum_utility_bits: float = 0.3,
        max_content_bytes: int = 32,
    ) -> None:
        super().__init__(
            prior=prior,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
            max_context_bytes=max_context_bytes,
            sequence_confirmations=sequence_confirmations,
            parent_strength=parent_strength,
        )
        if content_confirmations < 1 or episodic_confirmations < 1:
            raise ValueError("content and episodic confirmations must be positive")
        if memory_capacity < 1 or max_content_bytes < 1:
            raise ValueError("memory capacity and content length must be positive")
        if not 0.0 < memory_decay <= 1.0:
            raise ValueError("memory_decay must be in (0, 1]")
        if not 0.0 <= utility_decay < 1.0:
            raise ValueError("utility_decay must be in [0, 1)")
        if minimum_utility_bits < 0.0:
            raise ValueError("minimum_utility_bits must be non-negative")
        self.content_confirmations = content_confirmations
        self.memory_capacity = memory_capacity
        self.memory_decay = memory_decay
        self.episodic_confirmations = episodic_confirmations
        self.utility_decay = utility_decay
        self.minimum_utility_bits = minimum_utility_bits
        self.max_content_bytes = max_content_bytes
        self.content_nursery: dict[bytes, int] = {}
        self.content_ids: dict[bytes, int] = {}
        self.content_forms: list[bytes] = []
        self.content_counts: list[int] = []
        self.current_content = bytearray()
        self.boundary_window = bytearray()
        self.working_memory: dict[int, float] = {}
        self.episodic_nursery: dict[tuple[int, int], dict[int, int]] = {}
        self.episodic_counts: dict[tuple[int, int], dict[int, int]] = {}
        self.episodic_totals: dict[tuple[int, int], int] = {}
        self.episodic_utility: dict[tuple[int, int], float] = {}
        self.content_births = 0
        self.episodic_births = 0

    @property
    def live_content_count(self) -> int:
        return len(self.content_forms)

    @property
    def live_episode_count(self) -> int:
        return len(self.episodic_counts)

    @property
    def live_episode_synapse_count(self) -> int:
        return sum(len(row) for row in self.episodic_counts.values())

    def consolidate_episodes(
        self,
        *,
        minimum_observations: int = 8,
        minimum_utility_bits: float | None = None,
    ) -> int:
        """Prune mature episodic branches that cannot pass the recall gate."""
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        threshold = (
            self.minimum_utility_bits
            if minimum_utility_bits is None
            else minimum_utility_bits
        )
        doomed = [
            key
            for key, total in self.episodic_totals.items()
            if total >= minimum_observations and self.episodic_utility[key] <= threshold
        ]
        for key in doomed:
            del self.episodic_counts[key]
            del self.episodic_totals[key]
            del self.episodic_utility[key]
        return len(doomed)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            super().embodied_bytes,
            self.content_nursery,
            self.content_ids,
            self.content_forms,
            self.content_counts,
            self.current_content,
            self.boundary_window,
            self.working_memory,
            self.episodic_nursery,
            self.episodic_counts,
            self.episodic_totals,
            self.episodic_utility,
            self.content_births,
            self.episodic_births,
            self.minimum_utility_bits,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    @staticmethod
    def _is_content_byte(byte_id: int) -> bool:
        return byte_id == 39 or 65 <= byte_id <= 90 or 97 <= byte_id <= 122

    @staticmethod
    def _fold_content_byte(byte_id: int) -> int:
        return byte_id + 32 if 65 <= byte_id <= 90 else byte_id

    def _activate_content(self, content_id: int) -> None:
        for active_id in tuple(self.working_memory):
            self.working_memory[active_id] *= self.memory_decay
        count = self.content_counts[content_id]
        novelty = len(self.content_forms[content_id]) / np.sqrt(max(count, 1))
        self.working_memory[content_id] = max(
            self.working_memory.get(content_id, 0.0), float(novelty)
        )
        if len(self.working_memory) > self.memory_capacity:
            weakest = min(self.working_memory, key=self.working_memory.__getitem__)
            del self.working_memory[weakest]

    def _finish_content(self) -> None:
        if not self.current_content:
            return
        form = bytes(self.current_content)
        self.current_content.clear()
        content_id = self.content_ids.get(form)
        if content_id is None:
            confirmations = self.content_nursery.get(form, 0) + 1
            if confirmations < self.content_confirmations:
                self.content_nursery[form] = confirmations
                return
            content_id = len(self.content_forms)
            self.content_ids[form] = content_id
            self.content_forms.append(form)
            self.content_counts.append(confirmations)
            self.content_births += 1
            self.content_nursery.pop(form, None)
        else:
            self.content_counts[content_id] += 1
        self._activate_content(content_id)

    def _advance_content_stream(self, byte_id: int) -> None:
        if self._is_content_byte(byte_id):
            if len(self.current_content) < self.max_content_bytes:
                self.current_content.append(self._fold_content_byte(byte_id))
        else:
            self._finish_content()
        self.boundary_window.append(byte_id)
        if len(self.boundary_window) > len(self.story_boundary):
            del self.boundary_window[0]
        if bytes(self.boundary_window) == self.story_boundary:
            self.current_content.clear()
            self.working_memory.clear()
            self.boundary_window.clear()

    def _active_episode_keys(self) -> tuple[tuple[int, int], ...]:
        if self.previous_assembly is None:
            return ()
        return tuple(
            (content_id, self.previous_assembly)
            for content_id in self.working_memory
        )

    def _selected_episode(self) -> tuple[int, int] | None:
        live = (
            key for key in self._active_episode_keys() if key in self.episodic_counts
        )
        key = max(live, key=lambda item: self.episodic_utility[item], default=None)
        if key is None or self.episodic_utility[key] <= self.minimum_utility_bits:
            return None
        return key

    def _episode_probability(
        self, key: tuple[int, int], byte_id: int, baseline: float
    ) -> float:
        count = self.episodic_counts[key].get(byte_id, 0)
        return float(
            (count + self.basal_strength * baseline)
            / (self.episodic_totals[key] + self.basal_strength)
        )

    def _observed_probability(self, byte_id: int) -> float:
        baseline = super()._observed_probability(byte_id)
        key = self._selected_episode()
        if key is None:
            return baseline
        return self._episode_probability(key, byte_id, baseline)

    def _distribution(self) -> np.ndarray:
        baseline = super()._distribution()
        key = self._selected_episode()
        if key is None:
            return baseline
        distribution = self.basal_strength * baseline
        for target, count in self.episodic_counts[key].items():
            distribution[target] += count
        return distribution / (self.episodic_totals[key] + self.basal_strength)

    def _update_episodes(
        self, keys: tuple[tuple[int, int], ...], byte_id: int, baseline: float
    ) -> None:
        for key in keys:
            if key in self.episodic_counts:
                probability = self._episode_probability(key, byte_id, baseline)
                gain = log2(max(probability, 1e-300) / max(baseline, 1e-300))
                self.episodic_utility[key] = (
                    self.utility_decay * self.episodic_utility[key]
                    + (1.0 - self.utility_decay) * gain
                )
                row = self.episodic_counts[key]
                row[byte_id] = row.get(byte_id, 0) + 1
                self.episodic_totals[key] += 1
            elif -log2(max(baseline, 1e-300)) >= self.surprise_bits:
                nursery = self.episodic_nursery.setdefault(key, {})
                nursery[byte_id] = nursery.get(byte_id, 0) + 1
                if sum(nursery.values()) >= self.episodic_confirmations:
                    self.episodic_counts[key] = nursery
                    self.episodic_totals[key] = sum(nursery.values())
                    self.episodic_utility[key] = 0.0
                    del self.episodic_nursery[key]
                    self.episodic_births += 1

    def _learn_and_advance(self, byte_id: int, basal: float) -> None:
        baseline = super()._observed_probability(byte_id)
        keys = self._active_episode_keys()
        super()._learn_and_advance(byte_id, basal)
        if self.live_assemblies[byte_id]:
            self._update_episodes(keys, byte_id, baseline)
        self._advance_content_stream(byte_id)

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        step = super().observe(byte_id, learn=learn)
        if not learn:
            self._advance_content_stream(int(byte_id))
        return step

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        probability = super()._observe_probability_only(byte_id, learn=learn)
        if not learn:
            self._advance_content_stream(int(byte_id))
        return probability

    def reset_episode(self) -> None:
        super().reset_episode()
        self.current_content.clear()
        self.boundary_window.clear()
        self.working_memory.clear()

    def consolidate(self, other: EpisodicOpponentAttractorLearner) -> None:
        """Align content paths, then consolidate another episodic column."""
        configuration = (
            "content_confirmations",
            "memory_capacity",
            "memory_decay",
            "episodic_confirmations",
            "utility_decay",
            "minimum_utility_bits",
            "max_content_bytes",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate episodic columns with different configuration"
            )
        super().consolidate(other)

        other_to_self: dict[int, int] = {}
        for other_id, (form, count) in enumerate(
            zip(other.content_forms, other.content_counts, strict=True)
        ):
            content_id = self.content_ids.get(form)
            if content_id is None:
                content_id = len(self.content_forms)
                self.content_ids[form] = content_id
                self.content_forms.append(form)
                self.content_counts.append(0)
            self.content_counts[content_id] += count + self.content_nursery.pop(form, 0)
            other_to_self[other_id] = content_id
        for form, confirmations in other.content_nursery.items():
            content_id = self.content_ids.get(form)
            if content_id is not None:
                self.content_counts[content_id] += confirmations
                continue
            total = self.content_nursery.get(form, 0) + confirmations
            if total >= self.content_confirmations:
                content_id = len(self.content_forms)
                self.content_ids[form] = content_id
                self.content_forms.append(form)
                self.content_counts.append(total)
                self.content_nursery.pop(form, None)
            else:
                self.content_nursery[form] = total
        self.content_births = self.live_content_count

        def add_row(target: dict[int, int], source: dict[int, int]) -> None:
            for output, observations in source.items():
                target[output] = target.get(output, 0) + observations

        # A living incoming branch promotes the aligned local branch directly.
        # This mutates only the touched path instead of rebuilding every branch
        # accumulated from earlier columns.
        for (other_content_id, current), incoming in other.episodic_counts.items():
            key = other_to_self[other_content_id], current
            incoming_total = other.episodic_totals[(other_content_id, current)]
            incoming_utility = other.episodic_utility[(other_content_id, current)]
            if key in self.episodic_counts:
                local_total = self.episodic_totals[key]
                combined_weight = local_total + incoming_total
                self.episodic_utility[key] = (
                    self.episodic_utility[key] * local_total
                    + incoming_utility * incoming_total
                ) / combined_weight
                add_row(self.episodic_counts[key], incoming)
                self.episodic_totals[key] = combined_weight
            else:
                row = self.episodic_nursery.pop(key, {})
                add_row(row, incoming)
                self.episodic_counts[key] = row
                self.episodic_totals[key] = sum(row.values())
                self.episodic_utility[key] = incoming_utility

        # Nursery evidence either joins an already living path or remains
        # sparse until pooled confirmations cross the developmental barrier.
        for (other_content_id, current), incoming in other.episodic_nursery.items():
            key = other_to_self[other_content_id], current
            if key in self.episodic_counts:
                add_row(self.episodic_counts[key], incoming)
                self.episodic_totals[key] += sum(incoming.values())
                continue
            row = self.episodic_nursery.setdefault(key, {})
            add_row(row, incoming)
            total = sum(row.values())
            if total >= self.episodic_confirmations:
                self.episodic_counts[key] = self.episodic_nursery.pop(key)
                self.episodic_totals[key] = total
                self.episodic_utility[key] = 0.0
        self.episodic_births = self.live_episode_count
        self.reset_episode()
