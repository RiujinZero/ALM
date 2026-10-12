"""Ablate a train-only successor-keyed action memory.

The trajectory reranker established that a persistent conversation state can
predict a strictly later continuous successor target.  This experiment asks
the next, deliberately narrower question: can that state retrieve a *useful
action which was actually observed during training*, rather than choose among
incoherent samples from the native byte generator?

Above the byte/text adapters the mechanism is modality-neutral.  A memory is a
packed table of ``(continuous future key, continuous action adapter, opaque
action bytes)``.  Its values are prefixes of complete training ASSISTANT turns;
development and test responses are never inserted.  At inference a separately
learned full-history predictor supplies the future query, while an ordinary
pending-event plausibility map supplies the base score.

Every source stratum is split deterministically into three folds and evaluated
under all three G/M/E rotations:

* G fits the future bottleneck, causal/pending goal maps, plausibility map, and
  equal-width PCA adapters for strong pending/full-history nearest-neighbour
  controls;
* M alone supplies successor keys and action values;
* E alone fits an action-to-successor evaluator used only as a mechanistic
  secondary readout, never for retrieval or hyperparameter selection.

Capacity and causal gain are selected using development raw-byte 2/3/4-gram
pairing separately for a whole-response primary and fixed-prefix sensitivity.
The two nearest-neighbour controls select only their own gain on the same
frozen value bank.  Test pair payloads are not loaded until every development
choice has been frozen.  Rotations are reported separately and must not be
pooled.

This remains an action-relevance falsifier, not evidence of thoughtful output:
raw-byte n-gram pairing is an automated proxy and retrieved text may still be
inappropriate.  Blinded human evaluation is required for the broader claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
    _read_jsonl,
    _slice,
    load_crossover_pairs,
    load_training_examples,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _hashed_byte_ngram_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_DECAYS,
    DEFAULT_SUCCESSOR_GAMMAS,
    RidgeMap,
    fit_ridge,
    hashed_text_vector,
    normalize_rows,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    ContinuousTargetEncoder,
    _clamp_rows,
    _sattolo_permutation,
    action_features,
    fit_target_encoder,
    summarize,
    validate_dataset_semantics,
)

PACK_MAGIC = b"ALMAM02\0"
PACK_HEADER = struct.Struct("<8sIIIIIQ")
CONDITION_NAMES = (
    "goal_off",
    "causal_pending_user_only",
    "causal_full_history",
    "oracle_future_target",
    "shuffled_episode_goal",
    "shuffled_key_response_binding",
    "random_capacity_matched",
    "candidate_blind",
    "pending_key_nearest_neighbor",
    "full_history_nearest_neighbor",
)


@dataclass(frozen=True)
class TrainingAction:
    """One verified full ASSISTANT-turn value from ``train.bin``."""

    episode_id: str
    data: bytes
    pending_cue: bytes
    history_cue: bytes


@dataclass(frozen=True)
class HeldOutReferences:
    """Independent evaluation references, separate from retrieval state."""

    pair_id: str
    preferred_a: bytes
    preferred_b: bytes
    score_window_a: bytes
    score_window_b: bytes
    pending_cue_a: bytes
    pending_cue_b: bytes
    history_cue_a: bytes
    history_cue_b: bytes
    preferred_kind: str


@dataclass(frozen=True)
class PackedActionMemory:
    """Canonical packed operational memory (Python object overhead excluded)."""

    keys: np.ndarray
    cue_vectors: np.ndarray
    action_vectors: np.ndarray
    offsets: np.ndarray
    value_blob: bytes
    key_kind: str

    def __post_init__(self) -> None:
        if (
            self.keys.ndim != 2
            or self.cue_vectors.ndim != 2
            or self.action_vectors.ndim != 2
        ):
            raise ValueError("memory keys, cues, and action vectors must be matrices")
        if not (
            len(self.keys) == len(self.cue_vectors) == len(self.action_vectors)
        ):
            raise ValueError("memory key/value row counts differ")
        if self.keys.dtype != np.dtype("<f4"):
            raise ValueError("memory keys must use canonical little-endian float32")
        if self.action_vectors.dtype != np.dtype("<f4"):
            raise ValueError(
                "memory action vectors must use canonical little-endian float32"
            )
        if self.cue_vectors.dtype != np.dtype("<f4"):
            raise ValueError(
                "memory cue vectors must use canonical little-endian float32"
            )
        if self.offsets.dtype != np.dtype("<u8"):
            raise ValueError("memory offsets must use canonical little-endian uint64")
        if self.offsets.shape != (len(self.keys) + 1,):
            raise ValueError("memory offsets must contain one terminal offset")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(
            self.value_blob
        ):
            raise ValueError("memory offsets do not span the value blob")
        if np.any(np.diff(self.offsets.astype(np.int64)) <= 0):
            raise ValueError("every memory action value must be nonempty")
        for matrix in (self.keys, self.cue_vectors, self.action_vectors):
            if not matrix.flags.c_contiguous or not np.all(np.isfinite(matrix)):
                raise ValueError("memory matrices must be finite and C-contiguous")

    @property
    def entry_count(self) -> int:
        return len(self.keys)

    def value(self, index: int) -> bytes:
        if not 0 <= index < self.entry_count:
            raise IndexError("memory index out of range")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.value_blob[start:end]

    def pack(self) -> bytes:
        """Return the actual canonical serialization whose length is reported."""

        header = PACK_HEADER.pack(
            PACK_MAGIC,
            2,
            self.entry_count,
            int(self.keys.shape[1]),
            int(self.cue_vectors.shape[1]),
            int(self.action_vectors.shape[1]),
            len(self.value_blob),
        )
        return b"".join(
            (
                header,
                self.keys.tobytes(order="C"),
                self.cue_vectors.tobytes(order="C"),
                self.action_vectors.tobytes(order="C"),
                self.offsets.tobytes(order="C"),
                self.value_blob,
            )
        )

    def byte_accounting(self) -> dict[str, Any]:
        packed = self.pack()
        lengths = np.diff(self.offsets.astype(np.int64))
        return {
            "format": "ALMAM02 little-endian packed action memory v2",
            "entries": self.entry_count,
            "key_kind": self.key_kind,
            "key_dimensions": int(self.keys.shape[1]),
            "cue_dimensions": int(self.cue_vectors.shape[1]),
            "action_dimensions": int(self.action_vectors.shape[1]),
            "header_bytes": PACK_HEADER.size,
            "key_float32_bytes": int(self.keys.nbytes),
            "cue_float32_bytes": int(self.cue_vectors.nbytes),
            "action_float32_bytes": int(self.action_vectors.nbytes),
            "offset_uint64_bytes": int(self.offsets.nbytes),
            "opaque_action_value_bytes": len(self.value_blob),
            "opaque_action_value_bytes_min": int(np.min(lengths)),
            "opaque_action_value_bytes_max": int(np.max(lengths)),
            "opaque_action_value_bytes_mean": float(np.mean(lengths)),
            "packed_serialization_bytes": len(packed),
            "packed_serialization_sha256": hashlib.sha256(packed).hexdigest(),
        }


@dataclass(frozen=True)
class GoalRepresentation:
    successor_encoder: ContinuousTargetEncoder
    pending_key_encoder: ContinuousTargetEncoder
    history_key_encoder: ContinuousTargetEncoder
    full_goal_predictor: RidgeMap
    pending_goal_predictor: RidgeMap
    feature_dimensions: int


@dataclass(frozen=True)
class ActionAdapters:
    output_bytes: int
    cue_dimensions: int
    cue_kind: str
    base_mode: str
    learned_base_weight: float
    plausibility_predictor: RidgeMap
    evaluator_action_successor: RidgeMap
    training_cue_vectors: np.ndarray
    training_action_vectors: np.ndarray


@dataclass(frozen=True)
class MemoryBundle:
    successor: PackedActionMemory
    pending_nn: PackedActionMemory
    history_nn: PackedActionMemory
    shuffled_binding: PackedActionMemory
    random_keys: PackedActionMemory
    source_indices: np.ndarray
    binding_permutation: np.ndarray
    requested_capacity: int | None
    output_bytes: int


@dataclass(frozen=True)
class PreparedSplit:
    raw_pairs: tuple[CrossoverPair, ...]
    references: tuple[HeldOutReferences, ...]
    pending_events: np.ndarray
    full_evidence: np.ndarray
    oracle_targets: np.ndarray
    causal_goals: np.ndarray
    pending_goals: np.ndarray
    shuffled_goals: np.ndarray
    pending_nn_queries: np.ndarray
    history_nn_queries: np.ndarray
    cue_queries: np.ndarray
    preferred_reference_vectors: np.ndarray
    score_window_reference_vectors: np.ndarray
    shuffled_goal_permutation: np.ndarray


@dataclass(frozen=True)
class SelectedScope:
    name: str
    requested_capacity: int | None
    output_bytes: int
    causal_gain: float
    pending_nn_gain: float
    history_nn_gain: float
    selection_curve: dict[str, Any]
    pending_nn_gain_curve: dict[str, Any]
    history_nn_gain_curve: dict[str, Any]


@dataclass(frozen=True)
class RotationRuntime:
    name: str
    goal_indices: np.ndarray
    memory_indices: np.ndarray
    evaluator_indices: np.ndarray
    representation: GoalRepresentation
    actions_by_length: dict[int, ActionAdapters]
    selected_scopes: tuple[SelectedScope, ...]


def load_full_training_actions(
    dataset_dir: Path, examples: list[TrainingExample]
) -> list[TrainingAction]:
    """Load full target turns while proving that each is a train ASSISTANT turn."""

    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    blob = (dataset_dir / "train.bin").read_bytes()
    if len(rows) != len(examples):
        raise ValueError("training sidecar and examples have different lengths")
    result: list[TrainingAction] = []
    for index, (row, example) in enumerate(zip(rows, examples, strict=True)):
        if str(row.get("episode_id")) != example.episode_id:
            raise ValueError(f"training sidecar order mismatch at row {index}")
        target_index = int(row["target_turn_index"])
        matching = [turn for turn in row["turns"] if int(turn["index"]) == target_index]
        if len(matching) != 1 or str(matching[0].get("role")) != "ASSISTANT":
            raise ValueError(f"training target {index} is not one ASSISTANT turn")
        offsets = row["target_offsets"]
        start = int(offsets["content_start"])
        end = int(offsets["content_end"])
        if start < int(row["start"]) or end > int(row["end"]) or end <= start:
            raise ValueError(f"training target {index} lies outside its episode")
        if start != int(matching[0]["content_start"]) or end != int(
            matching[0]["end"]
        ):
            raise ValueError(f"training target {index} disagrees with turn metadata")
        goal = row.get("goal_target")
        if not isinstance(goal, dict) or int(goal["start"]) < end:
            raise ValueError(f"training target {index} overlaps its future key target")
        data = blob[start:end]
        expected = row.get("target_full_response_sha256")
        if expected and hashlib.sha256(data).hexdigest() != str(expected):
            raise ValueError(f"training target {index} checksum mismatch")
        if int(row.get("target_full_response_bytes", len(data))) != len(data):
            raise ValueError(f"training target {index} byte count mismatch")
        user_index = int(row["target_user_turn_index"])
        user_turn = [
            turn for turn in row["turns"] if int(turn["index"]) == user_index
        ]
        if len(user_turn) != 1 or str(user_turn[0].get("role")) != "USER":
            raise ValueError(f"training cue {index} is not one USER turn")
        episode_start = int(row["start"])
        cue_start = episode_start + int(row["target_user_content_start"])
        cue_end = episode_start + int(row["target_user_content_end"])
        if (
            cue_start != int(user_turn[0]["content_start"])
            or cue_end != int(user_turn[0]["end"])
        ):
            raise ValueError(f"training cue {index} disagrees with turn metadata")
        result.append(
            TrainingAction(
                example.episode_id,
                data,
                pending_cue=blob[cue_start:cue_end],
                history_cue=blob[int(row["start"]) : cue_end],
            )
        )
    return result


def load_heldout_references(
    dataset_dir: Path,
    split: str,
    pairs: list[CrossoverPair],
) -> list[HeldOutReferences]:
    """Load fixed score windows and optional prospectively stored full turns."""

    rows = _read_jsonl(dataset_dir / f"{split}_pairs.jsonl")
    blob = (dataset_dir / f"{split}_pairs.bin").read_bytes()
    if len(rows) != len(pairs):
        raise ValueError(f"{split} sidecar and loaded pairs have different lengths")
    result = []
    for index, (row, pair) in enumerate(zip(rows, pairs, strict=True)):
        if str(row.get("pair_id")) != pair.pair_id:
            raise ValueError(f"{split} pair order mismatch at row {index}")
        ranges = row["blob_ranges"]
        score_a = _slice(blob, ranges["response_a"])
        score_b = _slice(blob, ranges["response_b"])
        if score_a != pair.response_a or score_b != pair.response_b:
            raise ValueError(f"{split} score-window bytes disagree at pair {index}")
        full_names = None
        for names in (
            ("full_response_a", "full_response_b"),
            ("response_full_a", "response_full_b"),
        ):
            present = (names[0] in ranges, names[1] in ranges)
            if any(present) and not all(present):
                raise ValueError(f"{split} pair {index} has only one full response")
            if all(present):
                full_names = names
                break
        if full_names is None:
            preferred_a, preferred_b = score_a, score_b
            preferred_kind = "fixed_score_window_proxy"
        else:
            preferred_a = _slice(blob, ranges[full_names[0]])
            preferred_b = _slice(blob, ranges[full_names[1]])
            if not preferred_a.startswith(score_a) or not preferred_b.startswith(
                score_b
            ):
                raise ValueError(
                    f"{split} pair {index} score window is not a full-turn prefix"
                )
            preferred_kind = "prospective_full_assistant_turn"
        episode_a = row["episodes"]["a"]
        episode_b = row["episodes"]["b"]

        pair_label = f"{split} pair {index}"

        def cues(
            context: bytes, metadata: dict[str, Any], *, label: str = pair_label
        ) -> tuple[bytes, bytes]:
            cue_start = int(metadata["target_user_content_start"])
            cue_end = int(metadata["target_user_content_end"])
            history_end = int(metadata["unanswered_prefix_bytes"])
            if cue_end != history_end or not 0 <= cue_start < cue_end <= len(context):
                raise ValueError(f"{label} has invalid cue ranges")
            return context[cue_start:cue_end], context[:history_end]

        pending_a, history_a = cues(pair.context_a, episode_a)
        pending_b, history_b = cues(pair.context_b, episode_b)
        result.append(
            HeldOutReferences(
                pair_id=pair.pair_id,
                preferred_a=preferred_a,
                preferred_b=preferred_b,
                score_window_a=score_a,
                score_window_b=score_b,
                pending_cue_a=pending_a,
                pending_cue_b=pending_b,
                history_cue_a=history_a,
                history_cue_b=history_b,
                preferred_kind=preferred_kind,
            )
        )
    return result


def source_stratified_three_way_folds(
    examples: list[TrainingExample], source_labels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make stable, disjoint, nearly balanced folds within every source."""

    if len(examples) != len(source_labels):
        raise ValueError("examples and source labels must have equal lengths")
    if len({row.episode_id for row in examples}) != len(examples):
        raise ValueError("three-way fitting requires unique episode ids")
    by_source: dict[str, list[int]] = {}
    for index, source in enumerate(source_labels):
        by_source.setdefault(source, []).append(index)
    folds: tuple[list[int], list[int], list[int]] = ([], [], [])
    for source in sorted(by_source):
        ordered = sorted(
            by_source[source],
            key=lambda index: (
                hashlib.sha256(
                    (
                        "action-memory-gme-v1\x1f"
                        + source
                        + "\x1f"
                        + examples[index].episode_id
                    ).encode()
                ).digest(),
                examples[index].episode_id,
            ),
        )
        for position, index in enumerate(ordered):
            folds[position % 3].append(index)
    arrays = tuple(np.asarray(sorted(fold), dtype=np.int64) for fold in folds)
    if any(len(fold) < 4 for fold in arrays):
        raise ValueError("each G/M/E fold needs at least four episodes")
    joined = np.concatenate(arrays)
    if len(np.unique(joined)) != len(examples) or sorted(joined.tolist()) != list(
        range(len(examples))
    ):
        raise AssertionError("three-way folds are not an exact disjoint partition")
    return arrays


def fit_goal_representation(
    examples: list[TrainingExample],
    goal_indices: np.ndarray,
    *,
    feature_dimensions: int,
    latent_dimensions: int,
    ridge_alpha: float,
) -> GoalRepresentation:
    """Fit every learned query/key coordinate system from G only."""

    indices = np.asarray(goal_indices, dtype=np.int64)
    evidence = np.asarray([row.evidence for row in examples], dtype=np.float64)
    future = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] < feature_dimensions:
        raise ValueError("training evidence is narrower than the pending event")
    pending = evidence[:, :feature_dimensions]
    successor_encoder = fit_target_encoder(
        future[indices], latent_dimensions=latent_dimensions
    )
    pending_encoder = fit_target_encoder(
        pending[indices], latent_dimensions=latent_dimensions
    )
    history_encoder = fit_target_encoder(
        evidence[indices], latent_dimensions=latent_dimensions
    )
    widths = {
        len(successor_encoder.components),
        len(pending_encoder.components),
        len(history_encoder.components),
    }
    if widths != {latent_dimensions}:
        raise ValueError(
            "G-fold PCA rank is below latent_dimensions; equal-width controls "
            "cannot be capacity matched"
        )
    z = successor_encoder.transform(future[indices])
    return GoalRepresentation(
        successor_encoder=successor_encoder,
        pending_key_encoder=pending_encoder,
        history_key_encoder=history_encoder,
        full_goal_predictor=fit_ridge(evidence[indices], z, ridge_alpha),
        pending_goal_predictor=fit_ridge(pending[indices], z, ridge_alpha),
        feature_dimensions=feature_dimensions,
    )


def _output_value(action: TrainingAction, output_bytes: int) -> bytes:
    if output_bytes < 0:
        raise ValueError("output_bytes must be zero (full) or positive")
    return action.data if output_bytes == 0 else action.data[:output_bytes]


def _action_vectors(
    actions: list[TrainingAction], output_bytes: int, dimensions: int
) -> np.ndarray:
    return np.asarray(
        [
            _cached_action_vector(_output_value(action, output_bytes), dimensions)
            for action in actions
        ],
        dtype=np.float64,
    )


@lru_cache(maxsize=32_768)
def _hashed_cue_ngram_vector(payload: bytes, dimensions: int) -> np.ndarray:
    """High-specificity input adapter, hash-independent from output evaluation."""

    if dimensions < 16:
        raise ValueError("cue dimensions must be at least sixteen")
    vector = np.zeros(dimensions, dtype=np.float64)
    count = 0
    for width in (2, 3, 4):
        for start in range(max(0, len(payload) - width + 1)):
            digest = hashlib.blake2s(
                payload[start : start + width], digest_size=8, person=b"alm-cue"
            ).digest()
            code = int.from_bytes(digest, "little")
            vector[code % dimensions] += 1.0 if code >> 63 else -1.0
            count += 1
    if count == 0:
        digest = hashlib.blake2s(
            payload or b"empty", digest_size=8, person=b"alm-cue"
        ).digest()
        code = int.from_bytes(digest, "little")
        vector[code % dimensions] = 1.0 if code >> 63 else -1.0
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


@lru_cache(maxsize=32_768)
def _cached_action_vector(payload: bytes, dimensions: int) -> np.ndarray:
    return hashed_text_vector(payload.decode("utf-8", errors="replace"), dimensions)


@lru_cache(maxsize=32_768)
def _cached_evaluation_vector(payload: bytes, dimensions: int) -> np.ndarray:
    return _hashed_byte_ngram_vector(payload, dimensions)


def _cue_vectors(
    actions: list[TrainingAction], *, cue_kind: str, dimensions: int
) -> np.ndarray:
    if cue_kind not in {"pending", "full_history"}:
        raise ValueError("cue_kind must be pending or full_history")
    return np.asarray(
        [
            _hashed_cue_ngram_vector(
                action.pending_cue if cue_kind == "pending" else action.history_cue,
                dimensions,
            )
            for action in actions
        ]
    )


def fit_action_adapters(
    examples: list[TrainingExample],
    actions: list[TrainingAction],
    representation: GoalRepresentation,
    goal_indices: np.ndarray,
    evaluator_indices: np.ndarray,
    *,
    output_bytes: int,
    cue_dimensions: int,
    cue_kind: str,
    base_mode: str,
    learned_base_weight: float,
    plausibility_ridge_alpha: float,
    evaluator_ridge_alpha: float,
) -> ActionAdapters:
    """Fit selection base on G and the readout-only action evaluator on E."""

    if [row.episode_id for row in actions] != [row.episode_id for row in examples]:
        raise ValueError("training actions do not align with examples")
    vectors = _action_vectors(actions, output_bytes, representation.feature_dimensions)
    cue_vectors = _cue_vectors(
        actions, cue_kind=cue_kind, dimensions=cue_dimensions
    )
    evidence = np.asarray([row.evidence for row in examples], dtype=np.float64)
    pending = evidence[:, : representation.feature_dimensions]
    future = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    goal_indices = np.asarray(goal_indices, dtype=np.int64)
    evaluator_indices = np.asarray(evaluator_indices, dtype=np.int64)
    if np.intersect1d(goal_indices, evaluator_indices).size:
        raise ValueError("G and E must be disjoint")
    evaluator_targets = representation.successor_encoder.transform(
        future[evaluator_indices]
    )
    evaluator_features = action_features(
        pending[evaluator_indices], vectors[evaluator_indices], interaction=False
    )
    return ActionAdapters(
        output_bytes=output_bytes,
        cue_dimensions=cue_dimensions,
        cue_kind=cue_kind,
        base_mode=base_mode,
        learned_base_weight=learned_base_weight,
        plausibility_predictor=fit_ridge(
            pending[goal_indices], vectors[goal_indices], plausibility_ridge_alpha
        ),
        evaluator_action_successor=fit_ridge(
            evaluator_features, evaluator_targets, evaluator_ridge_alpha
        ),
        training_cue_vectors=cue_vectors,
        training_action_vectors=vectors,
    )


def _pack_memory(
    keys: np.ndarray,
    cue_vectors: np.ndarray,
    action_vectors: np.ndarray,
    values: list[bytes],
    *,
    key_kind: str,
) -> PackedActionMemory:
    key_matrix = np.ascontiguousarray(keys, dtype="<f4")
    cue_matrix = np.ascontiguousarray(cue_vectors, dtype="<f4")
    action_matrix = np.ascontiguousarray(action_vectors, dtype="<f4")
    offsets = np.zeros(len(values) + 1, dtype="<u8")
    for index, value in enumerate(values):
        if not isinstance(value, bytes) or not value:
            raise ValueError("packed action values must be nonempty bytes")
        offsets[index + 1] = offsets[index] + len(value)
    return PackedActionMemory(
        key_matrix,
        cue_matrix,
        action_matrix,
        offsets,
        b"".join(values),
        key_kind,
    )


def _capacity_subset(
    memory_indices: np.ndarray,
    examples: list[TrainingExample],
    requested_capacity: int | None,
) -> np.ndarray:
    indices = np.asarray(memory_indices, dtype=np.int64)
    if requested_capacity is None or requested_capacity >= len(indices):
        return indices.copy()
    if requested_capacity < 2:
        raise ValueError("memory capacity must be at least two")
    ordered = sorted(
        indices.tolist(),
        key=lambda index: (
            hashlib.sha256(
                (
                    "action-memory-capacity-v1\x1f" + examples[index].episode_id
                ).encode()
            ).digest(),
            examples[index].episode_id,
        ),
    )
    return np.asarray(sorted(ordered[:requested_capacity]), dtype=np.int64)


def build_memory_bundle(
    examples: list[TrainingExample],
    actions: list[TrainingAction],
    representation: GoalRepresentation,
    adapters: ActionAdapters,
    memory_indices: np.ndarray,
    *,
    requested_capacity: int | None,
    seed: int,
) -> MemoryBundle:
    """Build main and byte-identical-capacity controls from M only."""

    chosen = _capacity_subset(memory_indices, examples, requested_capacity)
    evidence = np.asarray([row.evidence for row in examples], dtype=np.float64)
    pending = evidence[:, : representation.feature_dimensions]
    future = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    values = [
        _output_value(actions[int(index)], adapters.output_bytes) for index in chosen
    ]
    action_vectors = adapters.training_action_vectors[chosen]
    cue_vectors = adapters.training_cue_vectors[chosen]
    successor_keys = representation.successor_encoder.transform(future[chosen])
    pending_keys = representation.pending_key_encoder.transform(pending[chosen])
    history_keys = representation.history_key_encoder.transform(evidence[chosen])
    successor = _pack_memory(
        successor_keys,
        cue_vectors,
        action_vectors,
        values,
        key_kind="strict_later_successor",
    )
    pending_nn = _pack_memory(
        pending_keys,
        cue_vectors,
        action_vectors,
        values,
        key_kind="pending_event_pca",
    )
    history_nn = _pack_memory(
        history_keys,
        cue_vectors,
        action_vectors,
        values,
        key_kind="full_history_pca",
    )
    rng = np.random.default_rng(seed + 17_111)
    permutation = _sattolo_permutation(len(values), rng)
    shuffled_binding = _pack_memory(
        successor_keys[permutation],
        cue_vectors,
        action_vectors,
        values,
        key_kind="successor_with_shuffled_action_binding",
    )
    random_keys = normalize_rows(rng.normal(size=successor_keys.shape))
    random_memory = _pack_memory(
        random_keys,
        cue_vectors,
        action_vectors,
        values,
        key_kind="random_capacity_matched",
    )
    byte_counts = {
        memory.byte_accounting()["packed_serialization_bytes"]
        for memory in (
            successor,
            pending_nn,
            history_nn,
            shuffled_binding,
            random_memory,
        )
    }
    if len(byte_counts) != 1:
        raise AssertionError("capacity-matched memory variants differ in packed bytes")
    return MemoryBundle(
        successor=successor,
        pending_nn=pending_nn,
        history_nn=history_nn,
        shuffled_binding=shuffled_binding,
        random_keys=random_memory,
        source_indices=chosen,
        binding_permutation=permutation,
        requested_capacity=requested_capacity,
        output_bytes=adapters.output_bytes,
    )


def prepare_split(
    pairs: list[CrossoverPair],
    references: list[HeldOutReferences],
    representation: GoalRepresentation,
    *,
    lexical_dimensions: int,
    cue_dimensions: int,
    cue_kind: str,
    seed: int,
) -> PreparedSplit:
    if len(pairs) != len(references) or not pairs:
        raise ValueError("pairs and references must be equal and nonempty")
    if [pair.pair_id for pair in pairs] != [row.pair_id for row in references]:
        raise ValueError("pair and reference order differs")
    evidence = np.asarray(
        [item for pair in pairs for item in (pair.evidence_a, pair.evidence_b)],
        dtype=np.float64,
    )
    pending = evidence[:, : representation.feature_dimensions]
    raw_targets = []
    for pair in pairs:
        if pair.goal_target_kind != "successor":
            raise ValueError("action memory requires strictly-later successor targets")
        if pair.goal_target_vector_a is None or pair.goal_target_vector_b is None:
            raise ValueError(f"pair {pair.pair_id} lacks future target vectors")
        raw_targets.extend((pair.goal_target_vector_a, pair.goal_target_vector_b))
    oracle = representation.successor_encoder.transform(np.asarray(raw_targets))
    causal = _clamp_rows(representation.full_goal_predictor.predict(evidence))
    pending_goal = _clamp_rows(representation.pending_goal_predictor.predict(pending))
    permutation = _sattolo_permutation(
        len(evidence), np.random.default_rng(seed + 8_291)
    )

    def reference_vectors(kind: str) -> np.ndarray:
        payloads = []
        for row in references:
            if kind == "preferred":
                payloads.extend((row.preferred_a, row.preferred_b))
            else:
                payloads.extend((row.score_window_a, row.score_window_b))
        return np.asarray(
            [_cached_evaluation_vector(value, lexical_dimensions) for value in payloads]
        )

    cue_payloads = []
    for row in references:
        if cue_kind == "pending":
            cue_payloads.extend((row.pending_cue_a, row.pending_cue_b))
        elif cue_kind == "full_history":
            cue_payloads.extend((row.history_cue_a, row.history_cue_b))
        else:
            raise ValueError("cue_kind must be pending or full_history")

    return PreparedSplit(
        raw_pairs=tuple(pairs),
        references=tuple(references),
        pending_events=pending,
        full_evidence=evidence,
        oracle_targets=oracle,
        causal_goals=causal,
        pending_goals=pending_goal,
        shuffled_goals=causal[permutation],
        pending_nn_queries=representation.pending_key_encoder.transform(pending),
        history_nn_queries=representation.history_key_encoder.transform(evidence),
        cue_queries=np.asarray(
            [_hashed_cue_ngram_vector(value, cue_dimensions) for value in cue_payloads]
        ),
        preferred_reference_vectors=reference_vectors("preferred"),
        score_window_reference_vectors=reference_vectors("score_window"),
        shuffled_goal_permutation=permutation,
    )


def _ridge_bytes(model: RidgeMap) -> int:
    return int(
        model.weights.nbytes + model.input_mean.nbytes + model.output_mean.nbytes
    )


def _encoder_bytes(model: ContinuousTargetEncoder) -> int:
    return int(model.mean.nbytes + model.components.nbytes)


def _base_score_matrix(
    prepared: PreparedSplit,
    memory: PackedActionMemory,
    adapters: ActionAdapters,
) -> np.ndarray:
    predicted_action = _clamp_rows(
        adapters.plausibility_predictor.predict(prepared.pending_events)
    )
    learned_base = (
        normalize_rows(predicted_action) @ normalize_rows(memory.action_vectors).T
    )
    cue_base = normalize_rows(prepared.cue_queries) @ normalize_rows(
        memory.cue_vectors
    ).T
    if adapters.base_mode == "cue":
        base = cue_base
    elif adapters.base_mode == "learned":
        base = learned_base
    elif adapters.base_mode == "cue_plus_learned":
        base = cue_base + adapters.learned_base_weight * learned_base
    else:
        raise ValueError(f"unknown base mode: {adapters.base_mode}")
    return base


def _key_score_matrix(
    prepared: PreparedSplit,
    memory: PackedActionMemory,
    queries: np.ndarray | None,
) -> np.ndarray:
    if queries is None:
        return np.zeros((len(prepared.pending_events), memory.entry_count))
    else:
        query = np.asarray(queries, dtype=np.float64)
        if query.shape[0] != len(prepared.pending_events):
            raise ValueError("query count does not match held-out episode count")
        if query.shape[1] != memory.keys.shape[1]:
            raise ValueError("query and memory key widths differ")
        return normalize_rows(query) @ normalize_rows(memory.keys).T


def _stable_argmax(values: np.ndarray) -> int:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not len(vector):
        raise ValueError("stable argmax requires a nonempty vector")
    return int(np.flatnonzero(vector == np.max(vector))[0])


def _condition_specification(
    name: str,
    prepared: PreparedSplit,
    bundle: MemoryBundle,
    *,
    causal_gain: float,
    pending_nn_gain: float,
    history_nn_gain: float,
) -> tuple[PackedActionMemory, np.ndarray | None, float, bool]:
    if name == "goal_off":
        return bundle.successor, None, 0.0, False
    if name == "causal_pending_user_only":
        return bundle.successor, prepared.pending_goals, causal_gain, False
    if name == "causal_full_history":
        return bundle.successor, prepared.causal_goals, causal_gain, False
    if name == "oracle_future_target":
        return bundle.successor, prepared.oracle_targets, causal_gain, False
    if name == "shuffled_episode_goal":
        return bundle.successor, prepared.shuffled_goals, causal_gain, False
    if name == "shuffled_key_response_binding":
        return bundle.shuffled_binding, prepared.causal_goals, causal_gain, False
    if name == "random_capacity_matched":
        return bundle.random_keys, prepared.causal_goals, causal_gain, False
    if name == "candidate_blind":
        return bundle.successor, prepared.causal_goals, causal_gain, True
    if name == "pending_key_nearest_neighbor":
        return bundle.pending_nn, prepared.pending_nn_queries, pending_nn_gain, False
    if name == "full_history_nearest_neighbor":
        return bundle.history_nn, prepared.history_nn_queries, history_nn_gain, False
    raise ValueError(f"unknown action-memory condition: {name}")


def _pair_similarity_readout(
    selected_a: int,
    selected_b: int,
    candidate_vectors: np.ndarray,
    reference_vectors: np.ndarray,
    pair_index: int,
) -> tuple[float, float]:
    offset = 2 * pair_index
    left = candidate_vectors[selected_a]
    right = candidate_vectors[selected_b]
    ref_a = reference_vectors[offset]
    ref_b = reference_vectors[offset + 1]
    aa = float(left @ ref_a)
    ab = float(left @ ref_b)
    ba = float(right @ ref_a)
    bb = float(right @ ref_b)
    # This reassociation is exactly zero when both selected vectors are equal.
    return (aa - ba) + (bb - ab), 0.5 * (aa + bb)


def condition_outcomes(
    prepared: PreparedSplit,
    bundle: MemoryBundle,
    adapters: ActionAdapters,
    name: str,
    *,
    causal_gain: float,
    pending_nn_gain: float,
    history_nn_gain: float,
    lexical_dimensions: int,
    include_pair_audit: bool,
    precomputed_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Freeze outputs for one intervention, then apply independent readouts."""

    memory, queries, gain, candidate_blind = _condition_specification(
        name,
        prepared,
        bundle,
        causal_gain=causal_gain,
        pending_nn_gain=pending_nn_gain,
        history_nn_gain=history_nn_gain,
    )
    if precomputed_scores is None:
        base = _base_score_matrix(prepared, memory, adapters)
        key = _key_score_matrix(prepared, memory, queries)
    else:
        base, key = precomputed_scores
        expected = (len(prepared.pending_events), memory.entry_count)
        if base.shape != expected or key.shape != expected:
            raise ValueError("precomputed score matrices have the wrong shape")
    if candidate_blind:
        key = np.zeros_like(key)
    candidate_lexical = np.asarray(
        [
            _cached_evaluation_vector(memory.value(index), lexical_dimensions)
            for index in range(memory.entry_count)
        ]
    )
    preferred_margin = []
    preferred_matched = []
    score_window_margin = []
    evaluator_margin = []
    selected_key_margin = []
    selections_differ = []
    selected_bytes_differ = []
    audits = []
    for pair_index in range(len(prepared.raw_pairs)):
        offset = 2 * pair_index
        selected_a = _stable_argmax(base[offset] + gain * key[offset])
        selected_b = _stable_argmax(base[offset + 1] + gain * key[offset + 1])
        margin, matched = _pair_similarity_readout(
            selected_a,
            selected_b,
            candidate_lexical,
            prepared.preferred_reference_vectors,
            pair_index,
        )
        fixed_margin, _ = _pair_similarity_readout(
            selected_a,
            selected_b,
            candidate_lexical,
            prepared.score_window_reference_vectors,
            pair_index,
        )
        selected_actions = np.asarray(
            [memory.action_vectors[selected_a], memory.action_vectors[selected_b]],
            dtype=np.float64,
        )
        action_input = action_features(
            prepared.pending_events[offset : offset + 2],
            selected_actions,
            interaction=False,
        )
        predicted_successor = _clamp_rows(
            adapters.evaluator_action_successor.predict(action_input)
        )
        evaluator_pair_margin, _ = _pair_similarity_readout(
            0,
            1,
            normalize_rows(predicted_successor),
            normalize_rows(prepared.oracle_targets),
            pair_index,
        )
        key_pair_margin, _ = _pair_similarity_readout(
            selected_a,
            selected_b,
            normalize_rows(memory.keys),
            normalize_rows(prepared.oracle_targets),
            pair_index,
        )
        value_a = memory.value(selected_a)
        value_b = memory.value(selected_b)
        preferred_margin.append(margin)
        preferred_matched.append(matched)
        score_window_margin.append(fixed_margin)
        evaluator_margin.append(evaluator_pair_margin)
        selected_key_margin.append(key_pair_margin)
        selections_differ.append(selected_a != selected_b)
        selected_bytes_differ.append(value_a != value_b)
        if include_pair_audit:
            audits.append(
                {
                    "pair_id": prepared.raw_pairs[pair_index].pair_id,
                    "selected_memory_index_a": selected_a,
                    "selected_memory_index_b": selected_b,
                    "selected_sha256_a": hashlib.sha256(value_a).hexdigest(),
                    "selected_sha256_b": hashlib.sha256(value_b).hexdigest(),
                    "selected_utf8_preview_a": value_a.decode(
                        "utf-8", errors="replace"
                    )[:160],
                    "selected_utf8_preview_b": value_b.decode(
                        "utf-8", errors="replace"
                    )[:160],
                    "preferred_reference_pair_margin": margin,
                    "fixed_score_window_pair_margin": fixed_margin,
                    "evaluator_action_successor_pair_margin": evaluator_pair_margin,
                }
            )
    result: dict[str, Any] = {
        "preferred_reference_margin": np.asarray(preferred_margin),
        "preferred_reference_matched_mean": np.asarray(preferred_matched),
        "fixed_score_window_margin": np.asarray(score_window_margin),
        "evaluator_action_successor_margin": np.asarray(evaluator_margin),
        "selected_key_oracle_margin": np.asarray(selected_key_margin),
        "selection_differs_between_sides": np.asarray(selections_differ),
        "selected_bytes_differ_between_sides": np.asarray(selected_bytes_differ),
    }
    if include_pair_audit:
        result["pair_audit"] = audits
    return result


def _curve_row(outcomes: dict[str, Any], goal_off: dict[str, Any]) -> dict[str, float]:
    primary = outcomes["preferred_reference_margin"]
    fixed = outcomes["fixed_score_window_margin"]
    delta = primary - goal_off["preferred_reference_margin"]
    return {
        "mean_preferred_reference_pair_margin": float(np.mean(primary)),
        "preferred_reference_pair_win_fraction": float(np.mean(primary > 0.0)),
        "mean_delta_vs_goal_off": float(np.mean(delta)),
        "mean_fixed_score_window_pair_margin": float(np.mean(fixed)),
        "selected_bytes_differ_between_sides_fraction": float(
            np.mean(outcomes["selected_bytes_differ_between_sides"])
        ),
    }


def _score_dev_candidate(
    prepared: PreparedSplit,
    bundle: MemoryBundle,
    adapters: ActionAdapters,
    condition: str,
    gain: float,
    *,
    lexical_dimensions: int,
    precomputed_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    memory, queries, _, blind = _condition_specification(
        condition,
        prepared,
        bundle,
        causal_gain=gain,
        pending_nn_gain=gain,
        history_nn_gain=gain,
    )
    scores = precomputed_scores
    if scores is None:
        scores = (
            _base_score_matrix(prepared, memory, adapters),
            _key_score_matrix(prepared, memory, queries),
        )
    base = scores[0]
    off_scores = (base, np.zeros_like(base))
    goal_off = condition_outcomes(
        prepared,
        bundle,
        adapters,
        "goal_off",
        causal_gain=0.0,
        pending_nn_gain=0.0,
        history_nn_gain=0.0,
        lexical_dimensions=lexical_dimensions,
        include_pair_audit=False,
        precomputed_scores=off_scores,
    )
    outcomes = condition_outcomes(
        prepared,
        bundle,
        adapters,
        condition,
        causal_gain=gain,
        pending_nn_gain=gain,
        history_nn_gain=gain,
        lexical_dimensions=lexical_dimensions,
        include_pair_audit=False,
        precomputed_scores=(scores[0], np.zeros_like(scores[1])) if blind else scores,
    )
    return _curve_row(outcomes, goal_off), outcomes


def select_scopes_on_dev(
    prepared: PreparedSplit,
    examples: list[TrainingExample],
    actions: list[TrainingAction],
    representation: GoalRepresentation,
    actions_by_length: dict[int, ActionAdapters],
    memory_indices: np.ndarray,
    *,
    capacities: tuple[int, ...],
    output_lengths: tuple[int, ...],
    gains: tuple[float, ...],
    lexical_dimensions: int,
    seed: int,
) -> tuple[SelectedScope, ...]:
    """Select global and bounded causal configs, then tune strong baselines."""

    if not capacities or not output_lengths or not gains:
        raise ValueError("capacity, output-length, and gain grids must be nonempty")
    curve_groups: dict[tuple[str, int], dict[str, Any]] = {}
    candidates: dict[
        tuple[str, int], list[tuple[tuple[float, ...], int | None, float]]
    ] = {}
    for output_bytes in output_lengths:
        adapters = actions_by_length[output_bytes]
        for scope, requested_values in (
            ("global", (None,)),
            ("bounded", capacities),
        ):
            group = (scope, output_bytes)
            curve_groups[group] = {}
            candidates[group] = []
            for requested_capacity in requested_values:
                bundle = build_memory_bundle(
                    examples,
                    actions,
                    representation,
                    adapters,
                    memory_indices,
                    requested_capacity=requested_capacity,
                    seed=seed,
                )
                packed_bytes = int(
                    bundle.successor.byte_accounting()["packed_serialization_bytes"]
                )
                causal_scores = (
                    _base_score_matrix(prepared, bundle.successor, adapters),
                    _key_score_matrix(
                        prepared, bundle.successor, prepared.causal_goals
                    ),
                )
                for gain in gains:
                    row, _ = _score_dev_candidate(
                        prepared,
                        bundle,
                        adapters,
                        "causal_full_history",
                        gain,
                        lexical_dimensions=lexical_dimensions,
                        precomputed_scores=causal_scores,
                    )
                    key_name = (
                        f"g={gain}"
                        if requested_capacity is None
                        else f"C={requested_capacity},g={gain}"
                    )
                    curve_groups[group][key_name] = {
                        **row,
                        "effective_entries": bundle.successor.entry_count,
                        "packed_serialization_bytes": packed_bytes,
                    }
                    # Primary proxy only. E's evaluator is intentionally absent.
                    rank = (
                        row["mean_preferred_reference_pair_margin"],
                        row["preferred_reference_pair_win_fraction"],
                        row["mean_delta_vs_goal_off"],
                        -float(packed_bytes),
                        -float(gain),
                    )
                    candidates[group].append((rank, requested_capacity, gain))

    selections = []
    ordered_outputs = sorted(output_lengths, key=lambda value: (value != 0, value))
    for output_bytes in ordered_outputs:
        for scope in ("global", "bounded"):
            group = (scope, output_bytes)
            _, capacity, causal_gain = max(
                candidates[group], key=lambda row: row[0]
            )
            adapters = actions_by_length[output_bytes]
            bundle = build_memory_bundle(
                examples,
                actions,
                representation,
                adapters,
                memory_indices,
                requested_capacity=capacity,
                seed=seed,
            )

            def select_control_gain(
                condition: str,
                *,
                active_bundle: MemoryBundle = bundle,
                active_adapters: ActionAdapters = adapters,
            ) -> tuple[float, dict[str, Any]]:
                curve: dict[str, Any] = {}
                ranked = []
                control_memory, control_query, _, _ = _condition_specification(
                    condition,
                    prepared,
                    active_bundle,
                    causal_gain=0.0,
                    pending_nn_gain=0.0,
                    history_nn_gain=0.0,
                )
                control_scores = (
                    _base_score_matrix(prepared, control_memory, active_adapters),
                    _key_score_matrix(prepared, control_memory, control_query),
                )
                for gain in gains:
                    row, _ = _score_dev_candidate(
                        prepared,
                        active_bundle,
                        active_adapters,
                        condition,
                        gain,
                        lexical_dimensions=lexical_dimensions,
                        precomputed_scores=control_scores,
                    )
                    curve[str(gain)] = row
                    ranked.append(
                        (
                            (
                                row["mean_preferred_reference_pair_margin"],
                                row["preferred_reference_pair_win_fraction"],
                                row["mean_delta_vs_goal_off"],
                                -float(gain),
                            ),
                            gain,
                        )
                    )
                return max(ranked, key=lambda row: row[0])[1], curve

            pending_gain, pending_curve = select_control_gain(
                "pending_key_nearest_neighbor"
            )
            history_gain, history_curve = select_control_gain(
                "full_history_nearest_neighbor"
            )
            output_label = "full_response" if output_bytes == 0 else f"L{output_bytes}"
            selections.append(
                SelectedScope(
                    name=(
                        f"global_all_M_actions_{output_label}"
                        if scope == "global"
                        else f"bounded_M_action_capacity_{output_label}"
                    ),
                    requested_capacity=capacity,
                    output_bytes=output_bytes,
                    causal_gain=causal_gain,
                    pending_nn_gain=pending_gain,
                    history_nn_gain=history_gain,
                    selection_curve=curve_groups[group],
                    pending_nn_gain_curve=pending_curve,
                    history_nn_gain_curve=history_curve,
                )
            )
    return tuple(selections)


def evaluate_scope(
    prepared: PreparedSplit,
    bundle: MemoryBundle,
    adapters: ActionAdapters,
    selected: SelectedScope,
    *,
    lexical_dimensions: int,
    seed: int,
    resamples: int,
    include_pair_audit: bool,
) -> dict[str, Any]:
    common_base = _base_score_matrix(prepared, bundle.successor, adapters)
    raw = {}
    for name in CONDITION_NAMES:
        memory, queries, _, blind = _condition_specification(
            name,
            prepared,
            bundle,
            causal_gain=selected.causal_gain,
            pending_nn_gain=selected.pending_nn_gain,
            history_nn_gain=selected.history_nn_gain,
        )
        keys = _key_score_matrix(prepared, memory, queries)
        if blind:
            keys = np.zeros_like(keys)
        raw[name] = condition_outcomes(
            prepared,
            bundle,
            adapters,
            name,
            causal_gain=selected.causal_gain,
            pending_nn_gain=selected.pending_nn_gain,
            history_nn_gain=selected.history_nn_gain,
            lexical_dimensions=lexical_dimensions,
            include_pair_audit=include_pair_audit,
            precomputed_scores=(common_base, keys),
        )
    off = raw["goal_off"]
    conditions: dict[str, Any] = {}
    for condition_index, name in enumerate(CONDITION_NAMES):
        row = raw[name]
        base_seed = seed + condition_index * 977
        conditions[name] = {
            "independent_preferred_raw_byte_ngram_pairing": summarize(
                row["preferred_reference_margin"],
                seed=base_seed + 1,
                resamples=resamples,
            ),
            "independent_delta_vs_goal_off": summarize(
                row["preferred_reference_margin"]
                - off["preferred_reference_margin"],
                seed=base_seed + 2,
                resamples=resamples,
            ),
            "fixed_score_window_raw_byte_ngram_pairing": summarize(
                row["fixed_score_window_margin"],
                seed=base_seed + 3,
                resamples=resamples,
            ),
            "E_only_action_successor_pairing": summarize(
                row["evaluator_action_successor_margin"],
                seed=base_seed + 4,
                resamples=resamples,
            ),
            "selected_memory_key_oracle_pairing": summarize(
                row["selected_key_oracle_margin"],
                seed=base_seed + 5,
                resamples=resamples,
            ),
            "selection_differs_between_sides_fraction": float(
                np.mean(row["selection_differs_between_sides"])
            ),
            "selected_bytes_differ_between_sides_fraction": float(
                np.mean(row["selected_bytes_differ_between_sides"])
            ),
        }
        if include_pair_audit:
            conditions[name]["pair_audit"] = row["pair_audit"]
    causal = raw["causal_full_history"]
    comparators = (
        "causal_pending_user_only",
        "shuffled_episode_goal",
        "shuffled_key_response_binding",
        "random_capacity_matched",
        "candidate_blind",
        "pending_key_nearest_neighbor",
        "full_history_nearest_neighbor",
    )
    contrasts = {}
    for index, name in enumerate(comparators):
        contrasts[f"causal_full_history_minus_{name}"] = {
            "independent_preferred_pairing": summarize(
                causal["preferred_reference_margin"]
                - raw[name]["preferred_reference_margin"],
                seed=seed + 50_001 + index * 311,
                resamples=resamples,
            ),
            "E_only_action_successor_pairing": summarize(
                causal["evaluator_action_successor_margin"]
                - raw[name]["evaluator_action_successor_margin"],
                seed=seed + 50_002 + index * 311,
                resamples=resamples,
            ),
        }

    def low(summary: dict[str, Any]) -> float:
        return float(summary["mean_bootstrap_ci95_low"])

    causal_condition = conditions["causal_full_history"]
    gates = {
        "causal_independent_pairing_ci_above_zero": low(
            causal_condition["independent_preferred_raw_byte_ngram_pairing"]
        )
        > 0.0,
        "causal_independent_delta_vs_goal_off_ci_above_zero": low(
            causal_condition["independent_delta_vs_goal_off"]
        )
        > 0.0,
        "causal_E_only_action_successor_pairing_ci_above_zero": low(
            causal_condition["E_only_action_successor_pairing"]
        )
        > 0.0,
        "causal_beats_pending_goal_query": low(
            contrasts[
                "causal_full_history_minus_causal_pending_user_only"
            ]["independent_preferred_pairing"]
        )
        > 0.0,
        "causal_beats_shuffled_binding": low(
            contrasts[
                "causal_full_history_minus_shuffled_key_response_binding"
            ]["independent_preferred_pairing"]
        )
        > 0.0,
        "causal_beats_random_keys": low(
            contrasts["causal_full_history_minus_random_capacity_matched"][
                "independent_preferred_pairing"
            ]
        )
        > 0.0,
        "causal_beats_tuned_pending_NN": low(
            contrasts["causal_full_history_minus_pending_key_nearest_neighbor"][
                "independent_preferred_pairing"
            ]
        )
        > 0.0,
        "causal_beats_tuned_full_history_NN": low(
            contrasts["causal_full_history_minus_full_history_nearest_neighbor"][
                "independent_preferred_pairing"
            ]
        )
        > 0.0,
        "candidate_blind_outputs_equal_goal_off": bool(
            np.array_equal(
                raw["candidate_blind"]["preferred_reference_margin"],
                off["preferred_reference_margin"],
            )
        ),
    }
    accounting = bundle.successor.byte_accounting()
    variant_bytes = {
        name: memory.byte_accounting()["packed_serialization_bytes"]
        for name, memory in (
            ("successor", bundle.successor),
            ("pending_nn", bundle.pending_nn),
            ("history_nn", bundle.history_nn),
            ("shuffled_binding", bundle.shuffled_binding),
            ("random_keys", bundle.random_keys),
        )
    }
    return {
        "selected_configuration": {
            "scope": selected.name,
            "requested_capacity": selected.requested_capacity,
            "effective_entries": bundle.successor.entry_count,
            "output_prefix_bytes": selected.output_bytes,
            "emits_complete_training_response": selected.output_bytes == 0,
            "causal_gain": selected.causal_gain,
            "pending_NN_dev_selected_gain": selected.pending_nn_gain,
            "full_history_NN_dev_selected_gain": selected.history_nn_gain,
        },
        "memory_byte_accounting": accounting,
        "capacity_control_packed_bytes": variant_bytes,
        "all_memory_variants_exactly_capacity_matched": len(
            set(variant_bytes.values())
        )
        == 1,
        "conditions": conditions,
        "paired_causal_control_contrasts": contrasts,
        "gate": gates,
    }


def _source_labels(
    dataset_dir: Path, examples: list[TrainingExample]
) -> tuple[str, ...]:
    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    if len(rows) != len(examples):
        raise ValueError("training source labels do not align with examples")
    labels = []
    for index, (row, example) in enumerate(zip(rows, examples, strict=True)):
        if str(row.get("episode_id")) != example.episode_id:
            raise ValueError(f"training source label mismatch at row {index}")
        source = str(row.get("source_label", ""))
        if not source:
            raise ValueError(f"training row {index} has no source label")
        labels.append(source)
    return tuple(labels)


def _fold_report(
    indices: np.ndarray,
    examples: list[TrainingExample],
    source_labels: tuple[str, ...],
) -> dict[str, Any]:
    ids = sorted(examples[int(index)].episode_id for index in indices)
    return {
        "episodes": len(indices),
        "source_counts": {
            source: sum(source_labels[int(index)] == source for index in indices)
            for source in sorted(set(source_labels))
        },
        "sorted_episode_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode()
        ).hexdigest(),
    }


def _reference_report(references: list[HeldOutReferences]) -> dict[str, Any]:
    preferred_lengths = [
        len(value)
        for row in references
        for value in (row.preferred_a, row.preferred_b)
    ]
    fixed_lengths = [
        len(value)
        for row in references
        for value in (row.score_window_a, row.score_window_b)
    ]
    kinds = sorted({row.preferred_kind for row in references})
    return {
        "preferred_reference_kinds": kinds,
        "preferred_full_turn_pair_fraction": float(
            np.mean(
                [
                    row.preferred_kind == "prospective_full_assistant_turn"
                    for row in references
                ]
            )
        ),
        "preferred_reference_bytes_min": min(preferred_lengths),
        "preferred_reference_bytes_max": max(preferred_lengths),
        "fixed_score_window_bytes_min": min(fixed_lengths),
        "fixed_score_window_bytes_max": max(fixed_lengths),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = json.loads(
        (args.dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    target_kind = str(manifest.get("config", {}).get("goal_target_kind", ""))
    if target_kind != "successor":
        raise ValueError("successor-keyed action memory requires successor targets")
    decays = tuple(float(value) for value in args.state_decays)
    gammas = tuple(float(value) for value in args.successor_gammas)
    examples = load_training_examples(
        args.dataset_dir,
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    actions = load_full_training_actions(args.dataset_dir, examples)
    source_labels = _source_labels(args.dataset_dir, examples)
    folds = source_stratified_three_way_folds(examples, source_labels)

    # Only development payloads are available to the selection phase.
    dev_pairs = load_crossover_pairs(
        args.dataset_dir,
        "dev",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    dev_references = load_heldout_references(args.dataset_dir, "dev", dev_pairs)
    runtimes = []
    for rotation in args.rotations:
        if rotation not in (0, 1, 2):
            raise ValueError("rotations must be selected from 0, 1, and 2")
        goal_indices = folds[rotation]
        memory_indices = folds[(rotation + 1) % 3]
        evaluator_indices = folds[(rotation + 2) % 3]
        representation = fit_goal_representation(
            examples,
            goal_indices,
            feature_dimensions=args.feature_dimensions,
            latent_dimensions=args.latent_dimensions,
            ridge_alpha=args.goal_ridge_alpha,
        )
        actions_by_length = {
            output_bytes: fit_action_adapters(
                examples,
                actions,
                representation,
                goal_indices,
                evaluator_indices,
                output_bytes=output_bytes,
                cue_dimensions=args.cue_dimensions,
                cue_kind=args.cue_kind,
                base_mode=args.base_mode,
                learned_base_weight=args.learned_base_weight,
                plausibility_ridge_alpha=args.plausibility_ridge_alpha,
                evaluator_ridge_alpha=args.evaluator_ridge_alpha,
            )
            for output_bytes in args.output_bytes
        }
        dev_prepared = prepare_split(
            dev_pairs,
            dev_references,
            representation,
            lexical_dimensions=args.lexical_dimensions,
            cue_dimensions=args.cue_dimensions,
            cue_kind=args.cue_kind,
            seed=args.seed + rotation * 10_003,
        )
        selected = select_scopes_on_dev(
            dev_prepared,
            examples,
            actions,
            representation,
            actions_by_length,
            memory_indices,
            capacities=tuple(args.memory_capacities),
            output_lengths=tuple(args.output_bytes),
            gains=tuple(args.gain_values),
            lexical_dimensions=args.lexical_dimensions,
            seed=args.seed + rotation * 10_003,
        )
        runtimes.append(
            RotationRuntime(
                name=f"G{rotation}_M{(rotation + 1) % 3}_E{(rotation + 2) % 3}",
                goal_indices=goal_indices,
                memory_indices=memory_indices,
                evaluator_indices=evaluator_indices,
                representation=representation,
                actions_by_length=actions_by_length,
                selected_scopes=selected,
            )
        )

    # This is deliberately after all development choices. It validates test
    # metadata and only then opens the held-out test payload.
    if validate_dataset_semantics(args.dataset_dir) != "successor":
        raise AssertionError("dataset target kind changed during selection")
    test_pairs = load_crossover_pairs(
        args.dataset_dir,
        "test",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    test_references = load_heldout_references(args.dataset_dir, "test", test_pairs)

    rotation_reports = []
    for rotation_index, runtime in enumerate(runtimes):
        dev_prepared = prepare_split(
            dev_pairs,
            dev_references,
            runtime.representation,
            lexical_dimensions=args.lexical_dimensions,
            cue_dimensions=args.cue_dimensions,
            cue_kind=args.cue_kind,
            seed=args.seed + rotation_index * 10_003,
        )
        test_prepared = prepare_split(
            test_pairs,
            test_references,
            runtime.representation,
            lexical_dimensions=args.lexical_dimensions,
            cue_dimensions=args.cue_dimensions,
            cue_kind=args.cue_kind,
            seed=args.seed + rotation_index * 10_003,
        )
        scope_reports = []
        causal_query_bytes = _ridge_bytes(
            runtime.representation.full_goal_predictor
        )
        goal_representation_all_bytes = sum(
            (
                _encoder_bytes(runtime.representation.successor_encoder),
                _encoder_bytes(runtime.representation.pending_key_encoder),
                _encoder_bytes(runtime.representation.history_key_encoder),
                causal_query_bytes,
                _ridge_bytes(runtime.representation.pending_goal_predictor),
            )
        )
        for scope_index, selected in enumerate(runtime.selected_scopes):
            adapters = runtime.actions_by_length[selected.output_bytes]
            bundle = build_memory_bundle(
                examples,
                actions,
                runtime.representation,
                adapters,
                runtime.memory_indices,
                requested_capacity=selected.requested_capacity,
                seed=args.seed + rotation_index * 10_003,
            )
            dev = evaluate_scope(
                dev_prepared,
                bundle,
                adapters,
                selected,
                lexical_dimensions=args.lexical_dimensions,
                seed=args.seed + rotation_index * 10_003 + scope_index * 1_009,
                resamples=args.resamples,
                include_pair_audit=False,
            )
            test = evaluate_scope(
                test_prepared,
                bundle,
                adapters,
                selected,
                lexical_dimensions=args.lexical_dimensions,
                seed=args.seed + rotation_index * 10_003 + scope_index * 1_009,
                resamples=args.resamples,
                include_pair_audit=args.include_pair_audit,
            )
            packed_bytes = int(
                bundle.successor.byte_accounting()["packed_serialization_bytes"]
            )
            plausibility_bytes = _ridge_bytes(adapters.plausibility_predictor)
            evaluator_bytes = _ridge_bytes(adapters.evaluator_action_successor)
            active_base_bytes = (
                0 if adapters.base_mode == "cue" else plausibility_bytes
            )
            scope_reports.append(
                {
                    "scope": selected.name,
                    "auxiliary_byte_accounting": {
                        "packed_action_memory_bytes": packed_bytes,
                        "causal_goal_predictor_numeric_bytes": causal_query_bytes,
                        "active_base_numeric_bytes": active_base_bytes,
                        "learned_plausibility_map_numeric_bytes": plausibility_bytes,
                        "learned_plausibility_map_active": adapters.base_mode != "cue",
                        "E_only_evaluator_numeric_bytes": evaluator_bytes,
                        "deployable_packed_plus_active_selection_numeric_bytes": (
                            packed_bytes + causal_query_bytes + active_base_bytes
                        ),
                        "full_ablation_packed_plus_all_numeric_parameter_bytes": (
                            packed_bytes
                            + goal_representation_all_bytes
                            + plausibility_bytes
                            + evaluator_bytes
                        ),
                        "excludes": (
                            "Python/container overhead, transient matrices, fixed "
                            "hash code, and raw conversation evidence"
                        ),
                    },
                    "selection": {
                        "causal_grid": selected.selection_curve,
                        "pending_NN_gain_grid": selected.pending_nn_gain_curve,
                        "full_history_NN_gain_grid": selected.history_nn_gain_curve,
                        "test_used_for_selection": False,
                    },
                    "dev": dev,
                    "test": test,
                }
            )
        representation_bytes = {
            "successor_encoder": _encoder_bytes(
                runtime.representation.successor_encoder
            ),
            "pending_key_encoder_control": _encoder_bytes(
                runtime.representation.pending_key_encoder
            ),
            "history_key_encoder_control": _encoder_bytes(
                runtime.representation.history_key_encoder
            ),
            "causal_goal_predictor": _ridge_bytes(
                runtime.representation.full_goal_predictor
            ),
            "pending_goal_predictor_control": _ridge_bytes(
                runtime.representation.pending_goal_predictor
            ),
        }
        rotation_reports.append(
            {
                "rotation": runtime.name,
                "role_folds": {
                    "G_goal_and_adapter_fit": _fold_report(
                        runtime.goal_indices, examples, source_labels
                    ),
                    "M_memory_values_only": _fold_report(
                        runtime.memory_indices, examples, source_labels
                    ),
                    "E_action_successor_evaluator_only": _fold_report(
                        runtime.evaluator_indices, examples, source_labels
                    ),
                    "pairwise_overlap": {
                        "G_M": int(
                            np.intersect1d(
                                runtime.goal_indices, runtime.memory_indices
                            ).size
                        ),
                        "G_E": int(
                            np.intersect1d(
                                runtime.goal_indices, runtime.evaluator_indices
                            ).size
                        ),
                        "M_E": int(
                            np.intersect1d(
                                runtime.memory_indices, runtime.evaluator_indices
                            ).size
                        ),
                    },
                },
                "representation_numeric_parameter_bytes": {
                    **representation_bytes,
                    "G_fit_total_including_controls": sum(
                        representation_bytes.values()
                    ),
                },
                "scopes": scope_reports,
            }
        )

    report = {
        "schema_version": 1,
        "experiment": "successor_keyed_train_action_memory",
        "status": "complete",
        "target_kind": target_kind,
        "mechanism": {
            "entry": (
                "G-PCA continuous strictly-later successor key + high-specificity "
                "causal cue + action adapter + an opaque whole or fixed-prefix "
                "M-fold training action"
            ),
            "selection_score": (
                f"{args.base_mode} causal relevance base + dev-selected gain * "
                "query-to-memory-key cosine"
            ),
            "causal_query": "G-fit full-history-to-successor ridge prediction",
            "primary_readout": (
                "frozen signed raw-byte 2/3/4-gram correct-minus-crossed pairing "
                "against held-out responses"
            ),
            "secondary_readout": (
                "E-only additive pending+selected-action ridge prediction of the "
                "strictly-later successor target"
            ),
        },
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "feature_dimensions": args.feature_dimensions,
            "latent_dimensions": args.latent_dimensions,
            "lexical_dimensions": args.lexical_dimensions,
            "cue_dimensions": args.cue_dimensions,
            "cue_kind": args.cue_kind,
            "base_mode": args.base_mode,
            "learned_base_weight": args.learned_base_weight,
            "state_decays": list(decays),
            "successor_gammas": list(gammas),
            "memory_capacities": args.memory_capacities,
            "output_prefix_bytes": args.output_bytes,
            "output_zero_semantics": "zero stores and emits the complete train turn",
            "gain_values": args.gain_values,
            "rotations": args.rotations,
            "seed": args.seed,
            "resamples": args.resamples,
        },
        "counts": {
            "training_episodes": len(examples),
            "training_full_assistant_turn_bytes": sum(
                len(action.data) for action in actions
            ),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
        },
        "provenance": {
            "candidate_values": (
                "only target ASSISTANT content ranges from train.bin; each range "
                "is verified against its train sidecar turn and checksum"
            ),
            "training_action_sequence_sha256": hashlib.sha256(
                b"".join(
                    len(action.data).to_bytes(8, "little") + action.data
                    for action in actions
                )
            ).hexdigest(),
            "test_pair_files_opened_after_all_dev_selections_frozen": True,
            "test_used_for_hyperparameter_selection": False,
        },
        "references": {
            "dev": _reference_report(dev_references),
            "test": _reference_report(test_references),
            "policy": (
                "prefer optional prospectively packed full_response_a/b ranges; "
                "otherwise use the predeclared fixed score-window proxy; always "
                "report the fixed score-window metric separately"
            ),
        },
        "three_way_rotation_semantics": (
            "G/M/E roles rotate over the same held-out dev/test pairs. Rotation "
            "results are reciprocal falsification checks, not independent samples, "
            "and must not be pooled into one pass claim."
        ),
        "conditions": list(CONDITION_NAMES),
        "rotations": rotation_reports,
        "claim_limits": [
            (
                "Retrieved outputs are opaque whole training responses in the "
                "primary L=full scopes and fixed prefixes in sensitivity scopes; "
                "neither is newly composed."
            ),
            (
                "Old schema datasets expose only predeclared response score windows "
                "(normally 128 bytes) for held-out output evaluation; these are "
                "honestly labeled proxies, not reconstructed full responses."
            ),
            (
                "Raw-byte n-gram pairing measures lexical output relevance, not "
                "coherence, instruction following, correctness, or thoughtfulness."
            ),
            (
                "The E-only successor evaluator is mechanistic and shares the "
                "successor coordinate system; it is not an independent quality label."
            ),
            (
                "The dense-vector key/query/memory operations are modality-neutral; "
                "the text hash and byte payload adapters in this assay are not."
            ),
            (
                "A successful retrieval assay would motivate budgeted integration, "
                "not establish that the native 500MB generator itself is goal-directed."
            ),
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-dimensions", type=int, default=96)
    parser.add_argument("--latent-dimensions", type=int, default=32)
    parser.add_argument("--lexical-dimensions", type=int, default=4096)
    parser.add_argument("--cue-dimensions", type=int, default=2048)
    parser.add_argument(
        "--cue-kind", choices=("pending", "full_history"), default="full_history"
    )
    parser.add_argument(
        "--base-mode",
        choices=("cue", "learned", "cue_plus_learned"),
        default="cue",
    )
    parser.add_argument("--learned-base-weight", type=float, default=1.0)
    parser.add_argument("--state-decays", type=float, nargs="+", default=DEFAULT_DECAYS)
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--goal-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--plausibility-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--evaluator-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--memory-capacities", type=int, nargs="+", default=(128, 512)
    )
    parser.add_argument("--output-bytes", type=int, nargs="+", default=(0, 96))
    parser.add_argument(
        "--gain-values", type=float, nargs="+", default=(0.0, 1.0, 4.0, 16.0)
    )
    parser.add_argument("--rotations", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=4_000)
    parser.add_argument("--include-pair-audit", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.feature_dimensions < 8:
        raise ValueError("feature_dimensions must be at least eight")
    if args.latent_dimensions < 1:
        raise ValueError("latent_dimensions must be positive")
    if args.lexical_dimensions < 16:
        raise ValueError("lexical_dimensions must be at least sixteen")
    if args.cue_dimensions < 16:
        raise ValueError("cue_dimensions must be at least sixteen")
    if any(value < 2 for value in args.memory_capacities):
        raise ValueError("memory capacities must be at least two")
    if any(value < 0 for value in args.output_bytes):
        raise ValueError("output byte lengths must be zero (full) or positive")
    if args.learned_base_weight < 0.0:
        raise ValueError("learned_base_weight must be non-negative")
    if any(value < 0.0 for value in args.gain_values):
        raise ValueError("gain values must be non-negative")
    if len(set(args.rotations)) != len(args.rotations):
        raise ValueError("rotations must be unique")
    report = run_ablation(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
