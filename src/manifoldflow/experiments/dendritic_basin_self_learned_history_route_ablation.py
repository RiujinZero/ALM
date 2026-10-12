"""Dev-only convergence assay for a learned history-by-production route.

The fitted object is a conditional density ratio, not a goal or a value:

    p_routed(b) proportional to
        p_native(b) * (q_history(b | persistent, emitted) /
                       q_local(b | emitted)) ** beta

Both ``q`` models learn from externally observed response bytes with the same
softmax cross-entropy update.  The native 500 MB cortex is immutable and is
used only as the deployed support prior.  Free generation is the primary
endpoint; teacher-forced BPB is retained only as a support/collapse diagnostic.

This source intentionally has no configurable evaluation split.  It verifies
and opens only schema-v4 training inputs and the development crossover payloads.
The full-context FIFO-128 motif state is measured for aliasing but never fitted:
on the current crossover construction the long common suffix erases its signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    ContextCoupledByteWriter,
    likelihood_ratio_correction,
)
from manifoldflow.experiments import (
    dendritic_basin_shared_goal_context_coupled_writer_ablation as receiver,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
    _read_jsonl,
    _slice,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)

DEFAULT_PASSES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_STATE_DECAYS = (0.5, 0.85, 0.97)
DEFAULT_BETAS = (0.5, 1.0)
FULL_CONTEXT_ALIAS_CAPACITY = 128
ROLE_PATTERN = re.compile(rb"(?m)^(USER|ASSISTANT):[ \t]*")
CONDITION_NAMES = (
    "native",
    "request_motif_additive_wide",
    "request_motif_coupled",
    "native_trajectory_additive_wide",
    "native_trajectory_coupled",
    "native_trajectory_shuffled_training",
    "native_trajectory_paired_side_swapped",
    "native_trajectory_frozen_joint",
    "native_trajectory_pending_only",
    "native_trajectory_pooled_prior",
)
CONTROLLED_CONTRASTS = {
    "request_motif_coupled_minus_additive_wide": (
        "request_motif_coupled",
        "request_motif_additive_wide",
    ),
    "native_trajectory_coupled_minus_native": (
        "native_trajectory_coupled",
        "native",
    ),
    "native_trajectory_coupled_minus_additive_wide": (
        "native_trajectory_coupled",
        "native_trajectory_additive_wide",
    ),
    "native_trajectory_coupled_minus_shuffled_training": (
        "native_trajectory_coupled",
        "native_trajectory_shuffled_training",
    ),
    "native_trajectory_coupled_minus_paired_side_swapped": (
        "native_trajectory_coupled",
        "native_trajectory_paired_side_swapped",
    ),
    "native_trajectory_coupled_minus_frozen_joint": (
        "native_trajectory_coupled",
        "native_trajectory_frozen_joint",
    ),
    "native_trajectory_coupled_minus_pending_only": (
        "native_trajectory_coupled",
        "native_trajectory_pending_only",
    ),
    "native_trajectory_coupled_minus_pooled_prior": (
        "native_trajectory_coupled",
        "native_trajectory_pooled_prior",
    ),
}


class FrozenCortex(Protocol):
    """Read-only cortex surface used by preprocessing and generation."""

    def reset_episode(self) -> None: ...

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> Any: ...

    def predict_next_distribution(self) -> np.ndarray: ...

    def advance_observation(self, byte_id: int, *, learn: bool) -> Any: ...


@dataclass(frozen=True)
class TrainingEpisode:
    episode_id: str
    prompt: bytes
    target_request: bytes
    response: bytes
    completed_event_ends: tuple[int, ...] = ()


@dataclass(frozen=True)
class DevelopmentPair:
    pair_id: str
    source_label: str
    context_a: bytes
    context_b: bytes
    request_a: bytes
    request_b: bytes
    response_a: bytes
    response_b: bytes


@dataclass(frozen=True)
class NativeTrajectory:
    full_vector: np.ndarray
    pending_vector: np.ndarray
    completed_events: int


@dataclass(frozen=True)
class PairStates:
    request_motif_a: tuple[int, ...]
    request_motif_b: tuple[int, ...]
    native_full_a: tuple[int, ...]
    native_full_b: tuple[int, ...]
    native_pending_a: tuple[int, ...]
    native_pending_b: tuple[int, ...]


@dataclass(frozen=True)
class WriterBank:
    local: ContextCoupledByteWriter
    request_motif_additive_wide: ContextCoupledByteWriter
    request_motif_coupled: ContextCoupledByteWriter
    native_trajectory_additive_wide: ContextCoupledByteWriter
    native_trajectory_coupled: ContextCoupledByteWriter
    native_trajectory_shuffled_training: ContextCoupledByteWriter

    def items(self) -> tuple[tuple[str, ContextCoupledByteWriter], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "local",
                "request_motif_additive_wide",
                "request_motif_coupled",
                "native_trajectory_additive_wide",
                "native_trajectory_coupled",
                "native_trajectory_shuffled_training",
            )
        )


@dataclass(frozen=True)
class RouteAssignment:
    writer: ContextCoupledByteWriter | None
    state_a: tuple[int, ...]
    state_b: tuple[int, ...]
    freeze_joint: bool = False


@dataclass(frozen=True)
class GeneratedTrace:
    data: bytes
    probability_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_passes(values: Sequence[float]) -> tuple[float, ...]:
    passes = tuple(float(value) for value in values)
    if not passes or passes[0] != 0.0:
        raise ValueError("passes must begin at zero")
    if any(not math.isfinite(value) or value < 0.0 for value in passes):
        raise ValueError("passes must be finite and non-negative")
    if any(right <= left for left, right in zip(passes, passes[1:], strict=False)):
        raise ValueError("passes must be strictly increasing")
    return passes


def load_training_episodes(
    dataset_dir: Path,
    *,
    response_bytes: int,
    maximum: int = 0,
) -> list[TrainingEpisode]:
    """Load causal prompts and response windows from schema-v4 training only."""

    if response_bytes < 1:
        raise ValueError("response byte cap must be positive")
    if maximum < 0:
        raise ValueError("maximum training episodes must be non-negative")
    blob = (dataset_dir / "train.bin").read_bytes()
    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    episodes: list[TrainingEpisode] = []
    previous_end = 0
    for row in rows:
        start = int(row.get("start", -1))
        end = int(row.get("end", -1))
        if start != previous_end or not start < end <= len(blob):
            raise ValueError("training rows must contiguously cover train.bin")
        previous_end = end
        if str(row.get("assigned_split", "")) != "train":
            raise ValueError("training index contains a non-training episode")
        episode = blob[start:end]
        if _sha256(episode) != str(row.get("sha256", "")):
            raise ValueError("training episode checksum mismatch")
        target = row.get("target_offsets")
        if not isinstance(target, dict):
            raise ValueError("training row lacks target offsets")
        response_start = int(target.get("score_window_start", -1))
        response_end = min(
            int(target.get("score_window_end", -1)), response_start + response_bytes
        )
        # These two row-level offsets are episode-local; target_offsets below
        # are absolute coordinates in train.bin.
        request_start = start + int(row.get("target_user_content_start", -1))
        request_end = start + int(row.get("target_user_content_end", -1))
        marker_start = int(target.get("role_marker_start", -1))
        if not (
            start
            <= request_start
            < request_end
            <= marker_start
            < response_start
            < response_end
            <= end
        ):
            raise ValueError("training causal/request/response ranges are invalid")
        identifier = str(row.get("episode_id", ""))
        if not identifier:
            raise ValueError("training episode ID is empty")
        raw_turns = row.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError("training episode has no indexed turns")
        event_ends = tuple(
            int(turn["end"]) - start
            for turn in raw_turns
            if int(turn["content_start"]) < int(turn["end"]) <= response_start
        )
        if (
            not event_ends
            or tuple(sorted(event_ends)) != event_ends
            or event_ends[-1] > response_start - start
        ):
            raise ValueError("training indexed event ends are invalid")
        episodes.append(
            TrainingEpisode(
                episode_id=identifier,
                prompt=blob[start:response_start],
                target_request=blob[request_start:request_end],
                response=blob[response_start:response_end],
                completed_event_ends=event_ends,
            )
        )
    if previous_end != len(blob):
        raise ValueError("training index does not cover train.bin")
    if maximum:
        episodes = episodes[:maximum]
    if len(episodes) < 2:
        raise ValueError("at least two training episodes are required")
    return episodes


def _request_from_context(context: bytes, metadata: Mapping[str, Any]) -> bytes:
    start = int(metadata.get("target_user_content_start", -1))
    end = int(metadata.get("target_user_content_end", -1))
    unanswered = int(metadata.get("unanswered_prefix_bytes", -1))
    if not 0 <= start < end <= unanswered <= len(context):
        raise ValueError("development target request range is invalid")
    return context[start:end]


def load_development_pairs(
    dataset_dir: Path,
    *,
    maximum: int = 0,
    excluded_pair_ids: Sequence[str] = (),
) -> list[DevelopmentPair]:
    """Load only the development crossover pair payload."""

    if maximum < 0:
        raise ValueError("maximum development pairs must be non-negative")
    excluded = tuple(str(value) for value in excluded_pair_ids)
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded development pair IDs must be unique")
    blob = (dataset_dir / "dev_pairs.bin").read_bytes()
    rows = _read_jsonl(dataset_dir / "dev_pairs.jsonl")
    pairs: list[DevelopmentPair] = []
    for row in rows:
        if int(row.get("schema_version", -1)) != 4 or row.get("split") != "dev":
            raise ValueError("development sidecar is not schema-v4 dev data")
        invariants = row.get("invariants")
        if not isinstance(invariants, dict) or not all(
            bool(value) for value in invariants.values()
        ):
            raise ValueError("development pair has a false construction invariant")
        ranges = row.get("blob_ranges")
        episode_metadata = row.get("episodes")
        if not isinstance(ranges, dict) or not isinstance(episode_metadata, dict):
            raise ValueError("development pair metadata is incomplete")
        context_a = _slice(blob, ranges["context_a"])
        context_b = _slice(blob, ranges["context_b"])
        response_a = _slice(blob, ranges["response_a"])
        response_b = _slice(blob, ranges["response_b"])
        common = int(row["common_suffix"]["bytes"])
        if common < 1 or context_a[-common:] != context_b[-common:]:
            raise ValueError("development pair common suffix is invalid")
        pairs.append(
            DevelopmentPair(
                pair_id=str(row["pair_id"]),
                source_label=str(row["source_label"]),
                context_a=context_a,
                context_b=context_b,
                request_a=_request_from_context(context_a, episode_metadata["a"]),
                request_b=_request_from_context(context_b, episode_metadata["b"]),
                response_a=response_a,
                response_b=response_b,
            )
        )
    available_ids = {pair.pair_id for pair in pairs}
    missing = sorted(set(excluded) - available_ids)
    if missing:
        raise ValueError("excluded development pair ID was not found")
    if excluded:
        excluded_set = set(excluded)
        pairs = [pair for pair in pairs if pair.pair_id not in excluded_set]
    if maximum:
        pairs = pairs[:maximum]
    if not pairs:
        raise ValueError("development pair payload is empty")
    return pairs


def fifo_fourgram_feature_ids(payload: bytes, capacity: int) -> tuple[int, ...]:
    """REPORT5-style FIFO over first insertion of distinct four-byte motifs."""

    if capacity < 1:
        raise ValueError("motif capacity must be positive")
    memory: OrderedDict[int, None] = OrderedDict()
    for index in range(max(0, len(payload) - 3)):
        motif = int.from_bytes(payload[index : index + 4], "big")
        memory[motif] = None
        if len(memory) > capacity:
            memory.popitem(last=False)
    if memory:
        return tuple(memory)
    fallback = hashlib.blake2b(
        payload or b"empty", digest_size=8, person=b"alm-4gram"
    ).digest()
    return (int.from_bytes(fallback, "little"),)


def full_context_fifo_alias_report(
    pairs: Sequence[DevelopmentPair], *, capacity: int = FULL_CONTEXT_ALIAS_CAPACITY
) -> dict[str, Any]:
    """Measure the untrained online motif state's exact information alias."""

    overlaps: list[float] = []
    equal = 0
    for pair in pairs:
        left = fifo_fourgram_feature_ids(pair.context_a, capacity)
        right = fifo_fourgram_feature_ids(pair.context_b, capacity)
        equal += left == right
        union = set(left) | set(right)
        overlaps.append(len(set(left) & set(right)) / max(len(union), 1))
    return {
        "capacity": capacity,
        "pairs": len(pairs),
        "exactly_equal_pairs": equal,
        "exactly_equal_fraction": equal / len(pairs),
        "mean_jaccard": float(np.mean(overlaps)),
        "writer_fitted": False,
        "interpretation": (
            "an exact-equality state collision makes paired response control "
            "impossible for any downstream readout"
        ),
    }


def _normalized_native_distribution(values: Any) -> np.ndarray:
    distribution = np.asarray(values, dtype=np.float64)
    if distribution.shape != (256,):
        raise ValueError("native distribution must have shape (256,)")
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0.0):
        raise ValueError("native distribution must be finite and non-negative")
    total = float(distribution.sum())
    if total <= 0.0:
        raise ValueError("native distribution must have positive mass")
    return distribution / total


def _activity_vector(distribution: np.ndarray) -> np.ndarray:
    log_probability = np.log(np.maximum(distribution, 1e-300))
    centered = log_probability - float(np.mean(log_probability))
    norm = float(np.linalg.norm(centered))
    return centered if norm == 0.0 else centered / norm


def _completed_event_ends(context: bytes) -> tuple[int, ...]:
    matches = list(ROLE_PATTERN.finditer(context))
    if not matches or matches[0].start() != 0:
        raise ValueError("context must begin with a role-marked event")
    ends: list[int] = []
    for index, match in enumerate(matches):
        end = len(context) if index + 1 == len(matches) else matches[index + 1].start()
        if context[match.end() : end].strip():
            ends.append(end)
    if not ends:
        raise ValueError("context has no completed event")
    return tuple(ends)


def native_predictive_trajectory(
    cortex: FrozenCortex,
    context: bytes,
    *,
    decays: Sequence[float],
    completed_event_ends: Sequence[int] | None = None,
) -> NativeTrajectory:
    """Pool the frozen cortex's causal event-boundary prediction activity."""

    rates = tuple(float(value) for value in decays)
    if not rates or any(not 0.0 < value < 1.0 for value in rates):
        raise ValueError("trajectory decays must lie in (0, 1)")
    ends = (
        _completed_event_ends(context)
        if completed_event_ends is None
        else tuple(int(value) for value in completed_event_ends)
    )
    if (
        not ends
        or tuple(sorted(set(ends))) != ends
        or ends[0] < 1
        or ends[-1] > len(context)
    ):
        raise ValueError("completed event ends must be increasing context offsets")
    cortex.reset_episode()
    cursor = 0
    states = [np.zeros(256, dtype=np.float64) for _ in rates]
    for end in ends:
        if end > cursor:
            cortex.observe_chunk_probabilities(
                np.frombuffer(context[cursor:end], dtype=np.uint8), learn=False
            )
        cursor = end
        activity = _activity_vector(
            _normalized_native_distribution(cortex.predict_next_distribution())
        )
        for index, decay in enumerate(rates):
            states[index] = decay * states[index] + (1.0 - decay) * activity
    if cursor < len(context):
        cortex.observe_chunk_probabilities(
            np.frombuffer(context[cursor:], dtype=np.uint8), learn=False
        )
    pending = _activity_vector(
        _normalized_native_distribution(cortex.predict_next_distribution())
    )
    normalized_states = []
    for state in states:
        norm = float(np.linalg.norm(state))
        normalized_states.append(state if norm == 0.0 else state / norm)
    full = np.concatenate((pending, *normalized_states))
    full.flags.writeable = False
    pending.flags.writeable = False
    return NativeTrajectory(full, pending, len(ends))


def _new_writer(
    kind: str,
    *,
    state_features: int,
    table_bits: int,
    learning_rate: float,
    local_orders: Sequence[int],
) -> ContextCoupledByteWriter:
    return ContextCoupledByteWriter(
        kind,  # type: ignore[arg-type]
        alphabet_size=256,
        table_bits=table_bits,
        learning_rate=learning_rate,
        local_orders=tuple(int(value) for value in local_orders),
        max_persistent_features=state_features,
    )


def make_writer_bank(
    *,
    state_feature_count: int,
    motif_feature_count: int,
    table_bits: int,
    learning_rate: float,
    local_orders: Sequence[int],
) -> WriterBank:
    """Allocate matched control writers; the local conditional is shared."""

    if state_feature_count < 1 or motif_feature_count < 1:
        raise ValueError("persistent feature counts must be positive")
    shared = dict(
        table_bits=table_bits,
        learning_rate=learning_rate,
        local_orders=local_orders,
    )
    local = _new_writer(
        "local",
        state_features=max(state_feature_count, motif_feature_count),
        **shared,
    )
    return WriterBank(
        local=local,
        request_motif_additive_wide=_new_writer(
            "additive_wide", state_features=motif_feature_count, **shared
        ),
        request_motif_coupled=_new_writer(
            "coupled", state_features=motif_feature_count, **shared
        ),
        native_trajectory_additive_wide=_new_writer(
            "additive_wide", state_features=state_feature_count, **shared
        ),
        native_trajectory_coupled=_new_writer(
            "coupled", state_features=state_feature_count, **shared
        ),
        native_trajectory_shuffled_training=_new_writer(
            "coupled", state_features=state_feature_count, **shared
        ),
    )


def _uniform_bases(length: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    if length not in cache:
        values = np.full((length, 256), 1.0 / 256.0, dtype=np.float64)
        values.flags.writeable = False
        cache[length] = values
    return cache[length]


def _accumulate_fit(
    totals: dict[str, dict[str, float]], name: str, diagnostics: Any
) -> None:
    row = totals.setdefault(
        name,
        {
            "episodes": 0.0,
            "symbols": 0.0,
            "base_nll_nats": 0.0,
            "conditional_nll_nats": 0.0,
            "maximum_absolute_update": 0.0,
        },
    )
    row["episodes"] += 1
    row["symbols"] += diagnostics.symbols
    row["base_nll_nats"] += diagnostics.base_nll_nats
    row["conditional_nll_nats"] += diagnostics.anchored_nll_nats
    row["maximum_absolute_update"] = max(
        row["maximum_absolute_update"], diagnostics.maximum_absolute_update
    )


def fit_exposure_interval(
    bank: WriterBank,
    episodes: Sequence[TrainingEpisode],
    request_states: Sequence[tuple[int, ...]],
    native_states: Sequence[tuple[int, ...]],
    shuffled_indices: np.ndarray,
    ordered_indices: np.ndarray,
    *,
    previous_passes: float,
    target_passes: float,
) -> dict[str, Any]:
    """Advance every matched writer through one cumulative exposure interval."""

    count = len(episodes)
    if not (
        len(request_states) == len(native_states) == len(shuffled_indices) == count
    ):
        raise ValueError("training state arrays do not align with episodes")
    start = int(round(previous_passes * count))
    stop = int(round(target_passes * count))
    if stop < start:
        raise ValueError("target passes precede previous passes")
    bases: dict[int, np.ndarray] = {}
    totals: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    response_digest = hashlib.sha256()
    for ordinal in range(start, stop):
        index = int(ordered_indices[ordinal % count])
        episode = episodes[index]
        response_digest.update(len(episode.response).to_bytes(8, "little"))
        response_digest.update(episode.response)
        uniform = _uniform_bases(len(episode.response), bases)
        fits = (
            (
                "local",
                bank.local,
                (),
            ),
            (
                "request_motif_additive_wide",
                bank.request_motif_additive_wide,
                request_states[index],
            ),
            (
                "request_motif_coupled",
                bank.request_motif_coupled,
                request_states[index],
            ),
            (
                "native_trajectory_additive_wide",
                bank.native_trajectory_additive_wide,
                native_states[index],
            ),
            (
                "native_trajectory_coupled",
                bank.native_trajectory_coupled,
                native_states[index],
            ),
            (
                "native_trajectory_shuffled_training",
                bank.native_trajectory_shuffled_training,
                native_states[int(shuffled_indices[index])],
            ),
        )
        for name, writer, state in fits:
            diagnostics = writer.fit_base_anchored_episode(
                episode.response,
                uniform,
                state,
                prefix=b"",
            )
            _accumulate_fit(totals, name, diagnostics)
    for row in totals.values():
        symbols = max(row["symbols"], 1.0)
        row["mean_base_nll_nats"] = row["base_nll_nats"] / symbols
        row["mean_conditional_nll_nats"] = row["conditional_nll_nats"] / symbols
        row["episodes"] = int(row["episodes"])
        row["symbols"] = int(row["symbols"])
    return {
        "previous_passes": previous_passes,
        "target_passes": target_passes,
        "episode_exposures": stop - start,
        "response_sequence_sha256": response_digest.hexdigest(),
        "writers": totals,
        "wall_seconds": time.perf_counter() - started,
    }


def _prime(cortex: FrozenCortex, context: bytes) -> None:
    cortex.reset_episode()
    if context:
        cortex.observe_chunk_probabilities(
            np.frombuffer(context, dtype=np.uint8), learn=False
        )


def _route_distribution(
    native: np.ndarray,
    emitted_prefix: bytes | bytearray,
    local_writer: ContextCoupledByteWriter,
    arm_writer: ContextCoupledByteWriter | None,
    state: tuple[int, ...],
    *,
    beta: float,
    log_ratio_clip: float,
    freeze_joint: bool,
) -> np.ndarray:
    base = _normalized_native_distribution(native)
    if arm_writer is None or beta == 0.0:
        return np.array(base, copy=True)
    local = local_writer.distribution(emitted_prefix, ())
    arm = arm_writer.distribution(emitted_prefix, state, freeze_joint=freeze_joint)
    if np.array_equal(local, arm):
        return np.array(base, copy=True)
    return likelihood_ratio_correction(
        base, local, arm, beta=beta, log_ratio_clip=log_ratio_clip
    )


def generate_route(
    cortex: FrozenCortex,
    context: bytes,
    local_writer: ContextCoupledByteWriter,
    arm_writer: ContextCoupledByteWriter | None,
    state: tuple[int, ...],
    *,
    beta: float,
    horizon: int,
    log_ratio_clip: float,
    uniform_draws: np.ndarray | None,
    freeze_joint: bool = False,
) -> GeneratedTrace:
    """Generate a greedy or common-random-number sampled frozen branch."""

    if horizon < 1:
        raise ValueError("generation horizon must be positive")
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if draws.shape != (horizon,) or np.any((draws < 0.0) | (draws >= 1.0)):
            raise ValueError("uniform draws must have shape (horizon,) in [0,1)")
    else:
        draws = None
    _prime(cortex, context)
    generated = bytearray()
    probability_digest = hashlib.sha256()
    for step in range(horizon):
        effective = _route_distribution(
            cortex.predict_next_distribution(),
            generated,
            local_writer,
            arm_writer,
            state,
            beta=beta,
            log_ratio_clip=log_ratio_clip,
            freeze_joint=freeze_joint,
        )
        probability_digest.update(effective.tobytes(order="C"))
        if draws is None:
            symbol = int(np.argmax(effective))
        else:
            symbol = min(
                int(np.searchsorted(np.cumsum(effective), draws[step], side="right")),
                255,
            )
        generated.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    return GeneratedTrace(bytes(generated), probability_digest.hexdigest())


def teacher_forced_bpb(
    cortex: FrozenCortex,
    context: bytes,
    response: bytes,
    local_writer: ContextCoupledByteWriter,
    arm_writer: ContextCoupledByteWriter | None,
    state: tuple[int, ...],
    *,
    beta: float,
    log_ratio_clip: float,
    freeze_joint: bool = False,
) -> float:
    """Secondary support diagnostic under the identical deployed route."""

    if not response:
        raise ValueError("teacher-forced response must be nonempty")
    _prime(cortex, context)
    prefix = bytearray()
    bits = 0.0
    for symbol in response:
        effective = _route_distribution(
            cortex.predict_next_distribution(),
            prefix,
            local_writer,
            arm_writer,
            state,
            beta=beta,
            log_ratio_clip=log_ratio_clip,
            freeze_joint=freeze_joint,
        )
        bits -= math.log2(max(float(effective[symbol]), 1e-300))
        prefix.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    return bits / len(response)


def _request_obligation_proxy(
    generated: bytes,
    request: bytes,
    own_reference: bytes,
    crossed_reference: bytes,
    *,
    width: int = 4,
) -> float | None:
    def grams(payload: bytes) -> set[bytes]:
        return {
            payload[index : index + width]
            for index in range(max(0, len(payload) - width + 1))
        }

    eligible = grams(request) & (grams(own_reference) - grams(crossed_reference))
    if not eligible:
        return None
    return len(eligible & grams(generated)) / len(eligible)


def _generation_diagnostics(payload: bytes) -> dict[str, Any]:
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in payload) / max(
        len(payload), 1
    )
    maximum_run = 0
    current_run = 0
    previous: int | None = None
    for byte in payload:
        current_run = current_run + 1 if byte == previous else 1
        maximum_run = max(maximum_run, current_run)
        previous = byte
    try:
        payload.decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    return {
        "printable_fraction": printable,
        "valid_utf8": valid_utf8,
        "maximum_identical_byte_run": maximum_run,
        "contains_newline": b"\n" in payload,
        "unique_byte_fraction": len(set(payload)) / max(len(payload), 1),
    }


def _mean_ci(values: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[draws], axis=1)
    return {
        "pairs": len(array),
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _condition_assignments(
    bank: WriterBank, pair: PairStates, pooled: tuple[int, ...]
) -> dict[str, RouteAssignment]:
    return {
        "native": RouteAssignment(None, (), ()),
        "request_motif_additive_wide": RouteAssignment(
            bank.request_motif_additive_wide,
            pair.request_motif_a,
            pair.request_motif_b,
        ),
        "request_motif_coupled": RouteAssignment(
            bank.request_motif_coupled,
            pair.request_motif_a,
            pair.request_motif_b,
        ),
        "native_trajectory_additive_wide": RouteAssignment(
            bank.native_trajectory_additive_wide,
            pair.native_full_a,
            pair.native_full_b,
        ),
        "native_trajectory_coupled": RouteAssignment(
            bank.native_trajectory_coupled,
            pair.native_full_a,
            pair.native_full_b,
        ),
        "native_trajectory_shuffled_training": RouteAssignment(
            bank.native_trajectory_shuffled_training,
            pair.native_full_a,
            pair.native_full_b,
        ),
        "native_trajectory_paired_side_swapped": RouteAssignment(
            bank.native_trajectory_coupled,
            pair.native_full_b,
            pair.native_full_a,
        ),
        "native_trajectory_frozen_joint": RouteAssignment(
            bank.native_trajectory_coupled,
            pair.native_full_a,
            pair.native_full_b,
            True,
        ),
        "native_trajectory_pending_only": RouteAssignment(
            bank.native_trajectory_coupled,
            pair.native_pending_a,
            pair.native_pending_b,
        ),
        "native_trajectory_pooled_prior": RouteAssignment(
            bank.native_trajectory_coupled, pooled, pooled
        ),
    }


def _first_divergence(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def evaluate_checkpoint(
    cortex: FrozenCortex,
    pairs: Sequence[DevelopmentPair],
    pair_states: Sequence[PairStates],
    bank: WriterBank,
    pooled_state: tuple[int, ...],
    *,
    betas: Sequence[float],
    horizon: int,
    log_ratio_clip: float,
    sample_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    expect_zero_route: bool = False,
) -> dict[str, Any]:
    """Run greedy and sampled free generation before any further fitting."""

    if len(pairs) != len(pair_states):
        raise ValueError("development states do not align with pairs")
    before = {name: writer.fingerprint() for name, writer in bank.items()}
    result: dict[str, Any] = {}
    for beta in (float(value) for value in betas):
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("evaluated betas must be finite and positive")
        by_condition: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CONDITION_NAMES
        }
        for pair_index, (pair, states) in enumerate(
            zip(pairs, pair_states, strict=True)
        ):
            assignments = _condition_assignments(bank, states, pooled_state)
            if tuple(assignments) != CONDITION_NAMES:
                raise AssertionError("condition coverage changed")
            draws = {}
            for side_index, side in enumerate(("a", "b")):
                rng = np.random.default_rng(
                    sample_seed + pair_index * 2_003 + side_index
                )
                draws[side] = rng.random(horizon)
            for name, assignment in assignments.items():
                outputs: dict[str, dict[str, Any]] = {}
                bpbs = []
                for side, context, request, response, crossed, state in (
                    (
                        "a",
                        pair.context_a,
                        pair.request_a,
                        pair.response_a,
                        pair.response_b,
                        assignment.state_a,
                    ),
                    (
                        "b",
                        pair.context_b,
                        pair.request_b,
                        pair.response_b,
                        pair.response_a,
                        assignment.state_b,
                    ),
                ):
                    greedy = generate_route(
                        cortex,
                        context,
                        bank.local,
                        assignment.writer,
                        state,
                        beta=beta,
                        horizon=horizon,
                        log_ratio_clip=log_ratio_clip,
                        uniform_draws=None,
                        freeze_joint=assignment.freeze_joint,
                    )
                    sampled = generate_route(
                        cortex,
                        context,
                        bank.local,
                        assignment.writer,
                        state,
                        beta=beta,
                        horizon=horizon,
                        log_ratio_clip=log_ratio_clip,
                        uniform_draws=draws[side],
                        freeze_joint=assignment.freeze_joint,
                    )
                    bpb = teacher_forced_bpb(
                        cortex,
                        context,
                        response,
                        bank.local,
                        assignment.writer,
                        state,
                        beta=beta,
                        log_ratio_clip=log_ratio_clip,
                        freeze_joint=assignment.freeze_joint,
                    )
                    bpbs.append(bpb)
                    outputs[side] = {
                        "greedy": greedy,
                        "sampled": sampled,
                        "greedy_obligation_proxy": _request_obligation_proxy(
                            greedy.data, request, response, crossed
                        ),
                        "sampled_obligation_proxy": _request_obligation_proxy(
                            sampled.data, request, response, crossed
                        ),
                        "teacher_forced_bpb": bpb,
                    }
                greedy_pair = _paired_reference_discrimination(
                    outputs["a"]["greedy"].data,
                    outputs["b"]["greedy"].data,
                    pair.response_a,
                    pair.response_b,
                    dimensions=1024,
                )
                sampled_pair = _paired_reference_discrimination(
                    outputs["a"]["sampled"].data,
                    outputs["b"]["sampled"].data,
                    pair.response_a,
                    pair.response_b,
                    dimensions=1024,
                )
                by_condition[name].append(
                    {
                        "pair_id": pair.pair_id,
                        "greedy_margin": greedy_pair["correct_minus_crossed_margin"],
                        "sampled_margin": sampled_pair["correct_minus_crossed_margin"],
                        "mean_teacher_forced_bpb": float(np.mean(bpbs)),
                        "sides": {
                            side: {
                                "greedy_hex": outputs[side]["greedy"].data.hex(),
                                "greedy_text": outputs[side]["greedy"].data.decode(
                                    "utf-8", errors="backslashreplace"
                                ),
                                "sampled_hex": outputs[side]["sampled"].data.hex(),
                                "sampled_text": outputs[side]["sampled"].data.decode(
                                    "utf-8", errors="backslashreplace"
                                ),
                                "greedy_probability_sha256": outputs[side][
                                    "greedy"
                                ].probability_sha256,
                                "sampled_probability_sha256": outputs[side][
                                    "sampled"
                                ].probability_sha256,
                                "greedy_obligation_proxy": outputs[side][
                                    "greedy_obligation_proxy"
                                ],
                                "sampled_obligation_proxy": outputs[side][
                                    "sampled_obligation_proxy"
                                ],
                                "greedy_diagnostics": _generation_diagnostics(
                                    outputs[side]["greedy"].data
                                ),
                                "sampled_diagnostics": _generation_diagnostics(
                                    outputs[side]["sampled"].data
                                ),
                            }
                            for side in ("a", "b")
                        },
                    }
                )
        summaries: dict[str, Any] = {}
        native_rows = by_condition["native"]
        for condition_index, (name, rows) in enumerate(by_condition.items()):
            greedy = [float(row["greedy_margin"]) for row in rows]
            sampled = [float(row["sampled_margin"]) for row in rows]
            bpb = [float(row["mean_teacher_forced_bpb"]) for row in rows]
            greedy_advantage = [
                value - float(native_rows[index]["greedy_margin"])
                for index, value in enumerate(greedy)
            ]
            sampled_advantage = [
                value - float(native_rows[index]["sampled_margin"])
                for index, value in enumerate(sampled)
            ]
            divergence = []
            for index, row in enumerate(rows):
                for side in ("a", "b"):
                    current = bytes.fromhex(row["sides"][side]["greedy_hex"])
                    native = bytes.fromhex(
                        native_rows[index]["sides"][side]["greedy_hex"]
                    )
                    position = _first_divergence(current, native)
                    divergence.append(horizon if position is None else position)
            obligations = [
                float(value)
                for row in rows
                for side in ("a", "b")
                for value in (row["sides"][side]["sampled_obligation_proxy"],)
                if value is not None
            ]
            summaries[name] = {
                "greedy_free_generation_margin": _mean_ci(
                    greedy,
                    seed=bootstrap_seed + condition_index * 17,
                    resamples=bootstrap_resamples,
                ),
                "sampled_free_generation_margin": _mean_ci(
                    sampled,
                    seed=bootstrap_seed + condition_index * 17 + 1,
                    resamples=bootstrap_resamples,
                ),
                "greedy_advantage_over_native": _mean_ci(
                    greedy_advantage,
                    seed=bootstrap_seed + condition_index * 17 + 2,
                    resamples=bootstrap_resamples,
                ),
                "sampled_advantage_over_native": _mean_ci(
                    sampled_advantage,
                    seed=bootstrap_seed + condition_index * 17 + 3,
                    resamples=bootstrap_resamples,
                ),
                "mean_teacher_forced_bpb": float(np.mean(bpb)),
                "mean_first_greedy_divergence_from_native": float(np.mean(divergence)),
                "sampled_request_reference_proxy": (
                    None
                    if not obligations
                    else {
                        "eligible_sides": len(obligations),
                        "mean": float(np.mean(obligations)),
                    }
                ),
            }
        contrasts: dict[str, Any] = {}
        for contrast_index, (label, (left_name, right_name)) in enumerate(
            CONTROLLED_CONTRASTS.items()
        ):
            left = by_condition[left_name]
            right = by_condition[right_name]
            greedy_difference = [
                float(left[index]["greedy_margin"])
                - float(right[index]["greedy_margin"])
                for index in range(len(pairs))
            ]
            sampled_difference = [
                float(left[index]["sampled_margin"])
                - float(right[index]["sampled_margin"])
                for index in range(len(pairs))
            ]
            bpb_advantage = [
                float(right[index]["mean_teacher_forced_bpb"])
                - float(left[index]["mean_teacher_forced_bpb"])
                for index in range(len(pairs))
            ]
            contrasts[label] = {
                "greedy_free_generation_margin_difference": _mean_ci(
                    greedy_difference,
                    seed=bootstrap_seed + 10_000 + contrast_index * 29,
                    resamples=bootstrap_resamples,
                ),
                "sampled_free_generation_margin_difference": _mean_ci(
                    sampled_difference,
                    seed=bootstrap_seed + 10_001 + contrast_index * 29,
                    resamples=bootstrap_resamples,
                ),
                "teacher_forced_bpb_advantage_secondary": _mean_ci(
                    bpb_advantage,
                    seed=bootstrap_seed + 10_002 + contrast_index * 29,
                    resamples=bootstrap_resamples,
                ),
            }
        result[str(beta)] = {
            "summaries": summaries,
            "controlled_contrasts": contrasts,
            "pair_rows": by_condition,
        }
    beta_keys = tuple(result)
    reference_native = result[beta_keys[0]]["pair_rows"]["native"]
    native_invariant = True
    for beta_key in beta_keys[1:]:
        comparison = result[beta_key]["pair_rows"]["native"]
        for reference, candidate in zip(reference_native, comparison, strict=True):
            native_invariant = native_invariant and (
                reference["mean_teacher_forced_bpb"]
                == candidate["mean_teacher_forced_bpb"]
                and reference["sides"] == candidate["sides"]
            )
    zero_route_exact = None
    if expect_zero_route:
        zero_route_exact = True
        for beta_run in result.values():
            native_rows = beta_run["pair_rows"]["native"]
            for name in CONDITION_NAMES[1:]:
                for native_row, candidate_row in zip(
                    native_rows, beta_run["pair_rows"][name], strict=True
                ):
                    zero_route_exact = zero_route_exact and (
                        native_row["mean_teacher_forced_bpb"]
                        == candidate_row["mean_teacher_forced_bpb"]
                        and native_row["sides"] == candidate_row["sides"]
                    )
        if not zero_route_exact:
            raise AssertionError("checkpoint-zero route is not exactly native")
    after = {name: writer.fingerprint() for name, writer in bank.items()}
    if before != after:
        raise AssertionError("free generation modified a writer")
    if not native_invariant:
        raise AssertionError("native sampled branch changed across beta values")
    return {
        "invariants": {
            "native_greedy_sampled_and_bpb_identical_across_betas": native_invariant,
            "checkpoint_zero_every_route_exactly_native": zero_route_exact,
            "writer_fingerprints_unchanged_by_evaluation": True,
        },
        "beta_runs": result,
    }


def _snapshot_bank(
    bank: WriterBank, destination: Path, *, completed_passes: float
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    metadata: dict[str, Any] = {"completed_passes": completed_passes, "writers": {}}
    for name, writer in bank.items():
        snapshot = writer.snapshot()
        arrays[name] = snapshot.weights
        metadata["writers"][name] = {
            "version": snapshot.version,
            "kind": snapshot.kind,
            "alphabet_size": snapshot.alphabet_size,
            "table_bits": snapshot.table_bits,
            "learning_rate": snapshot.learning_rate,
            "local_orders": list(snapshot.local_orders),
            "max_persistent_features": snapshot.max_persistent_features,
            "episodes_seen": snapshot.episodes_seen,
            "symbols_seen": snapshot.symbols_seen,
        }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(destination, **arrays)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _pass_label(value: float) -> str:
    return str(value).replace(".", "p")


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Fit geometric auxiliary checkpoints and evaluate development only."""

    started = time.perf_counter()
    passes = _strict_passes(args.passes)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if args.wall_limit_seconds <= 0.0:
        raise ValueError("wall limit must be positive")
    if (args.max_train_episodes or args.max_dev_pairs) and not args.smoke:
        raise ValueError("data slicing is permitted only with --smoke")
    verified = receiver._verify_manifest_payloads(
        args.dataset_dir, args.dataset_dir / "manifest.json", "dev"
    )
    episodes = load_training_episodes(
        args.dataset_dir,
        response_bytes=args.response_bytes,
        maximum=args.max_train_episodes,
    )
    pairs = load_development_pairs(
        args.dataset_dir,
        maximum=args.max_dev_pairs,
        excluded_pair_ids=args.exclude_dev_pair_ids,
    )
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    # The native serialized-size estimator includes ordinary rolling episode
    # history.  Capture the learned body before any read-only trajectory is
    # primed so provenance does not mistake transient inference state for new
    # parameters/anatomy.
    native_embodied_bytes_at_load = int(getattr(cortex, "embodied_bytes", -1))

    train_trajectories = [
        native_predictive_trajectory(
            cortex,
            episode.prompt,
            decays=args.state_decays,
            completed_event_ends=episode.completed_event_ends,
        )
        for episode in episodes
    ]
    train_native_ids = [
        receiver.top_signed_feature_ids(row.full_vector, args.state_feature_count)
        for row in train_trajectories
    ]
    train_request_ids = [
        fifo_fourgram_feature_ids(episode.target_request, args.motif_feature_count)
        for episode in episodes
    ]
    pooled_vector = np.mean(
        np.asarray([row.full_vector for row in train_trajectories]), axis=0
    )
    pooled_ids = receiver.top_signed_feature_ids(
        pooled_vector, args.state_feature_count
    )

    dev_trajectories = [
        (
            native_predictive_trajectory(
                cortex, pair.context_a, decays=args.state_decays
            ),
            native_predictive_trajectory(
                cortex, pair.context_b, decays=args.state_decays
            ),
        )
        for pair in pairs
    ]
    pair_states = [
        PairStates(
            request_motif_a=fifo_fourgram_feature_ids(
                pair.request_a, args.motif_feature_count
            ),
            request_motif_b=fifo_fourgram_feature_ids(
                pair.request_b, args.motif_feature_count
            ),
            native_full_a=receiver.top_signed_feature_ids(
                trajectories[0].full_vector, args.state_feature_count
            ),
            native_full_b=receiver.top_signed_feature_ids(
                trajectories[1].full_vector, args.state_feature_count
            ),
            native_pending_a=receiver.top_signed_feature_ids(
                trajectories[0].pending_vector, args.state_feature_count
            ),
            native_pending_b=receiver.top_signed_feature_ids(
                trajectories[1].pending_vector, args.state_feature_count
            ),
        )
        for pair, trajectories in zip(pairs, dev_trajectories, strict=True)
    ]
    bank = make_writer_bank(
        state_feature_count=args.state_feature_count,
        motif_feature_count=args.motif_feature_count,
        table_bits=args.table_bits,
        learning_rate=args.learning_rate,
        local_orders=args.local_orders,
    )
    order = np.random.default_rng(args.training_seed).permutation(len(episodes))
    shuffled = receiver.sattolo_indices(len(episodes), args.training_seed + 1)
    checkpoints = []
    previous = 0.0
    for completed in passes:
        interval = (
            {
                "previous_passes": 0.0,
                "target_passes": 0.0,
                "episode_exposures": 0,
                "response_sequence_sha256": _sha256(b""),
                "writers": {},
                "wall_seconds": 0.0,
            }
            if completed == 0.0
            else fit_exposure_interval(
                bank,
                episodes,
                train_request_ids,
                train_native_ids,
                shuffled,
                order,
                previous_passes=previous,
                target_passes=completed,
            )
        )
        evaluation = evaluate_checkpoint(
            cortex,
            pairs,
            pair_states,
            bank,
            pooled_ids,
            betas=args.beta_values,
            horizon=args.generation_horizon,
            log_ratio_clip=args.log_ratio_clip,
            sample_seed=args.sample_seed,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            expect_zero_route=completed == 0.0,
        )
        snapshot = None
        if args.snapshot_dir is not None:
            snapshot = _snapshot_bank(
                bank,
                args.snapshot_dir / f"pass_{_pass_label(completed)}.npz",
                completed_passes=completed,
            )
        checkpoints.append(
            {
                "completed_passes": completed,
                "cumulative_episode_exposures": int(round(completed * len(episodes))),
                "cumulative_symbol_updates_per_writer": bank.local.symbols_seen,
                "writer_fingerprints": {
                    name: writer.fingerprint() for name, writer in bank.items()
                },
                "training_interval": interval,
                "snapshot": snapshot,
                "evaluation": evaluation,
            }
        )
        previous = completed
        if time.perf_counter() - started > args.wall_limit_seconds:
            raise TimeoutError("wall limit reached after a completed checkpoint")

    native_pair_equal = sum(
        state.native_full_a == state.native_full_b for state in pair_states
    )
    request_pair_equal = sum(
        state.request_motif_a == state.request_motif_b for state in pair_states
    )
    return {
        "schema_version": 1,
        "experiment": "self_learned_history_route_native500_dev",
        "status": "complete_smoke" if args.smoke else "complete_development",
        "claim_scope": {
            "mechanism": (
                "learned conditional likelihood ratio coupling persistent causal "
                "activity to the evolving emitted-symbol prefix"
            ),
            "not_a_goal_or_value": True,
            "native_checkpoint_frozen": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_secondary": True,
            "request_motif_state_is_supplied_receiver_ceiling": True,
            "native_trajectory_is_observable_output_activity_"
            "not_hidden_dendrites": True,
            "natural_reference_proxy_is_not_thoughtful_behavior": True,
        },
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "embodied_bytes": native_embodied_bytes_at_load,
            "embodied_bytes_at_load": native_embodied_bytes_at_load,
            "transient_embodied_bytes_at_report": int(
                getattr(cortex, "embodied_bytes", -1)
            ),
        },
        "config": {
            "passes": list(passes),
            "state_decays": list(args.state_decays),
            "state_feature_count": args.state_feature_count,
            "motif_feature_count": args.motif_feature_count,
            "full_context_alias_capacity": FULL_CONTEXT_ALIAS_CAPACITY,
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "response_bytes": args.response_bytes,
            "beta_values": list(args.beta_values),
            "log_ratio_clip": args.log_ratio_clip,
            "generation_horizon": args.generation_horizon,
            "training_seed": args.training_seed,
            "sample_seed": args.sample_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "smoke": args.smoke,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "unique_training_conversations": len(episodes),
            "unique_development_pairs": len(pairs),
            "response_bytes_one_pass": sum(len(row.response) for row in episodes),
        },
        "representation_preflight": {
            "full_context_fifo128_alias": full_context_fifo_alias_report(pairs),
            "full_context_fifo128_training_skipped": True,
            "request_motif_equal_pairs": request_pair_equal,
            "native_trajectory_equal_pairs": native_pair_equal,
            "pending_only_equal_pairs": sum(
                state.native_pending_a == state.native_pending_b
                for state in pair_states
            ),
            "mean_completed_events_train": float(
                np.mean([row.completed_events for row in train_trajectories])
            ),
            "mean_completed_events_dev_sides": float(
                np.mean(
                    [row.completed_events for pair in dev_trajectories for row in pair]
                )
            ),
        },
        "writer_memory": {
            "allocated_bytes_by_writer": {
                name: writer.allocated_bytes for name, writer in bank.items()
            },
            "total_writer_bytes": sum(
                writer.allocated_bytes for _, writer in bank.items()
            ),
        },
        "checkpoints": checkpoints,
        "wall_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--passes", type=float, nargs="+", default=DEFAULT_PASSES)
    parser.add_argument(
        "--state-decays", type=float, nargs="+", default=DEFAULT_STATE_DECAYS
    )
    parser.add_argument("--state-feature-count", type=int, default=8)
    parser.add_argument("--motif-feature-count", type=int, default=16)
    parser.add_argument("--table-bits", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--response-bytes", type=int, default=128)
    parser.add_argument("--beta-values", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--log-ratio-clip", type=float, default=2.0)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--training-seed", type=int, default=260_920)
    parser.add_argument("--sample-seed", type=int, default=260_921)
    parser.add_argument("--bootstrap-seed", type=int, default=260_922)
    parser.add_argument("--bootstrap-resamples", type=int, default=4_000)
    parser.add_argument("--wall-limit-seconds", type=float, default=7_200.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-episodes", type=int, default=0)
    parser.add_argument("--max-dev-pairs", type=int, default=0)
    parser.add_argument("--exclude-dev-pair-ids", nargs="*", default=())
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_ablation(args)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
