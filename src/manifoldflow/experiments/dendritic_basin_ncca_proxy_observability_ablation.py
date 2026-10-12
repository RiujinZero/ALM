"""Train-only NCCA proxy-observability gate.

This precursor asks one deliberately narrow question: does an anonymous
response action improve compression of the *next external USER event* beyond
what was available in the causal conversation state alone?  It does not select
an action, connect to a native generator, estimate human utility, or identify a
counterfactual response effect.

Only the schema-v4 training binary, its training index, and its manifest are
loadable.  Every assistant-to-next-USER transition is derived inside one
indexed conversation.  Complete conversations are assigned to deterministic
source-stratified folds.  In each of three rotations, compressors fit one fold
and score a disjoint fold.

The response tokenizer is outcome-blind: it sees response bytes and fold IDs,
never later USER bytes.  Two allocation-matched ``ContextCoupledByteWriter``
models receive exactly 17 persistent slots.  ``q_H`` gets eight causal-state
IDs, a NULL action, and eight state/NULL conjunction IDs.  ``q_HA`` replaces
NULL with the observed anonymous action.  A third matched writer is trained
with action rows Sattolo-shuffled inside the exact source/script/horizon/length
stratum.

This module contains no pair-payload loader.  Its command line requires an
explicit dataset directory and an explicit new report path.  Reducing the
14-bit table is a diagnostic-only mode and can never authorize selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    ContextCoupledByteWriter,
    ContextCoupledWriterSnapshot,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    hashed_text_vector,
    normalize,
)

SCHEMA_VERSION = 4
MANIFEST_NAME = "manifest.json"
TRAIN_BINARY_NAME = "train.bin"
TRAIN_INDEX_NAME = "train.index.jsonl"
TRAIN_INPUT_NAMES = (MANIFEST_NAME, TRAIN_BINARY_NAME, TRAIN_INDEX_NAME)

FOLD_NAMES = ("A", "B", "C")
ROTATIONS = (("A", "B"), ("B", "C"), ("C", "A"))

LOCKED_FEATURE_DIMENSIONS = 96
LOCKED_STATE_DECAYS = (0.5, 0.85, 0.97)
LOCKED_STATE_FEATURE_COUNT = 8
LOCKED_RESPONSE_BYTES = 128
LOCKED_NEXT_USER_BYTES = 128
LOCKED_PCA_DIMENSIONS = 32
LOCKED_INITIAL_ACTIONS = 16
LOCKED_MIN_ACTION_SUPPORT = 64
LOCKED_KMEANS_ITERATIONS = 50
LOCKED_TOKENIZER_SEED = 82_972
LOCKED_FOLD_SEED = 260_914
LOCKED_SHUFFLE_SEED = 260_915
LOCKED_BOOTSTRAP_SEED = 260_916
LOCKED_BOOTSTRAP_RESAMPLES = 4_000
LOCKED_MIN_SCORED_CONVERSATIONS = 30
LOCKED_TABLE_BITS = 14
LOCKED_LEARNING_RATE = 0.55
LOCKED_LOCAL_ORDERS = (1, 2, 4, 8)
PERSISTENT_SLOT_COUNT = 17
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

_UINT64_MASK = (1 << 64) - 1
_RAW_ROLE_MARKER = re.compile(rb"(USER|ASSISTANT):[ \t]*")


@dataclass(frozen=True)
class AssayConfig:
    """Configuration with one locked inferential setting.

    Smaller values are useful for synthetic engineering checks, but every
    deviation from the locked values is reported as selection-ineligible.
    """

    feature_dimensions: int = LOCKED_FEATURE_DIMENSIONS
    state_decays: tuple[float, ...] = LOCKED_STATE_DECAYS
    state_feature_count: int = LOCKED_STATE_FEATURE_COUNT
    response_bytes: int = LOCKED_RESPONSE_BYTES
    next_user_bytes: int = LOCKED_NEXT_USER_BYTES
    pca_dimensions: int = LOCKED_PCA_DIMENSIONS
    initial_actions: int = LOCKED_INITIAL_ACTIONS
    min_action_support: int = LOCKED_MIN_ACTION_SUPPORT
    kmeans_iterations: int = LOCKED_KMEANS_ITERATIONS
    tokenizer_seed: int = LOCKED_TOKENIZER_SEED
    fold_seed: int = LOCKED_FOLD_SEED
    shuffle_seed: int = LOCKED_SHUFFLE_SEED
    bootstrap_seed: int = LOCKED_BOOTSTRAP_SEED
    bootstrap_resamples: int = LOCKED_BOOTSTRAP_RESAMPLES
    table_bits: int = LOCKED_TABLE_BITS
    learning_rate: float = LOCKED_LEARNING_RATE
    local_orders: tuple[int, ...] = LOCKED_LOCAL_ORDERS

    def validate(self) -> None:
        if self.feature_dimensions < 8:
            raise ValueError("feature_dimensions must be at least eight")
        if not self.state_decays or any(
            not 0.0 < value < 1.0 for value in self.state_decays
        ):
            raise ValueError("state_decays must contain values in (0, 1)")
        expected_width = self.feature_dimensions * (1 + len(self.state_decays))
        if self.state_feature_count != LOCKED_STATE_FEATURE_COUNT:
            raise ValueError(
                f"state_feature_count must be {LOCKED_STATE_FEATURE_COUNT} "
                "to preserve the 17-slot allocation"
            )
        if self.state_feature_count > expected_width:
            raise ValueError("state feature count exceeds the evidence width")
        if self.response_bytes < 1 or self.next_user_bytes < 1:
            raise ValueError("response and next-USER windows must be positive")
        if self.pca_dimensions < 1:
            raise ValueError("pca_dimensions must be positive")
        if not 2 <= self.initial_actions <= 16:
            raise ValueError("initial_actions must be in [2, 16]")
        if self.min_action_support < 1:
            raise ValueError("min_action_support must be positive")
        if self.kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if any(seed < 0 for seed in self.seed_values()):
            raise ValueError("all seeds must be nonnegative")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 1 <= self.table_bits <= LOCKED_TABLE_BITS:
            raise ValueError(
                f"table_bits must be in [1, {LOCKED_TABLE_BITS}] for this assay"
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not self.local_orders or any(
            value < 1 or value > 64 for value in self.local_orders
        ):
            raise ValueError("local_orders must contain values in [1, 64]")
        if len(set(self.local_orders)) != len(self.local_orders):
            raise ValueError("local_orders must not contain duplicates")

    def seed_values(self) -> tuple[int, ...]:
        return (
            self.tokenizer_seed,
            self.fold_seed,
            self.shuffle_seed,
            self.bootstrap_seed,
        )


@dataclass(frozen=True)
class Transition:
    """One strictly ordered assistant-to-next-USER training transition."""

    decision_id: str
    conversation_id: str
    assistant_turn_index: int
    source_label: str
    script: str
    horizon_bin: int
    response_length_bin: int
    history_start: int
    history_end: int
    response_start: int
    response_end: int
    response_window_end: int
    next_user_start: int
    next_user_end: int
    next_user_window_end: int
    next_history_end: int
    next_assistant_turn_index: int | None
    terminal_after_next_user: bool
    history_sha256: str
    response_sha256: str
    next_user_sha256: str
    response_window_sha256: str
    next_user_window_sha256: str
    next_history_sha256: str
    response: bytes
    next_user: bytes
    evidence: np.ndarray
    declared_source: str | None = None
    source_path: str = ""
    source_line: int = -1
    source_byte_offset: int = -1
    next_response_start: int | None = None
    next_response_end: int | None = None
    next_response_window_end: int | None = None
    next_response_sha256: str | None = None
    next_response_window_sha256: str | None = None
    next_response: bytes | None = None

    @property
    def shuffle_stratum(self) -> tuple[str, str, int, int]:
        return (
            self.source_label,
            self.script,
            self.horizon_bin,
            self.response_length_bin,
        )


@dataclass(frozen=True)
class LoadedTraining:
    transitions: tuple[Transition, ...]
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    extraction: Mapping[str, Any]


@dataclass(frozen=True)
class ActionTokenizer:
    """Deterministic PCA plus spherical prototypes and support merging."""

    vector_dimensions: int
    mean: np.ndarray
    components: np.ndarray
    initial_centroids: np.ndarray
    initial_to_final: np.ndarray
    fold_support: np.ndarray
    merge_history: tuple[Mapping[str, Any], ...]

    @property
    def action_count(self) -> int:
        return int(self.fold_support.shape[1])

    def latent(self, response: bytes) -> np.ndarray:
        vector = _response_vector(response, self.vector_dimensions)
        projected = (vector - self.mean) @ self.components.T
        return normalize(projected)

    def assign(self, response: bytes) -> int:
        raw = int(np.argmax(self.latent(response) @ self.initial_centroids.T))
        return int(self.initial_to_final[raw])


@dataclass(frozen=True)
class ScoredTransition:
    conversation_id: str
    decision_id: str
    action: int
    target_bytes: int
    bits_h: float
    bits_ha: float
    bits_shuffle: float


CheckpointHook = Callable[
    [
        str,
        int,
        int,
        Mapping[str, ContextCoupledWriterSnapshot],
    ],
    None,
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_u64(*parts: object, person: bytes = b"ncca-v1") -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=person).digest()
    return int.from_bytes(digest, "big")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _horizon_bin(value: int) -> int:
    for edge in (4_096, 2_048, 1_024, 512):
        if value >= edge:
            return edge
    return 0


def _response_bin(value: int) -> int:
    for edge in (512, 256, 128, 64):
        if value >= edge:
            return edge
    return 0


def _script_name(character: str) -> str | None:
    code = ord(character)
    if not character.isalpha():
        return None
    if 0x0041 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF or 0xAB30 <= code <= 0xAB6F:
        return "latin"
    if 0x0400 <= code <= 0x052F or 0x2DE0 <= code <= 0x2DFF:
        return "cyrillic"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "han"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "katakana"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return "arabic"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if 0x0590 <= code <= 0x05FF:
        return "hebrew"
    if 0x0E00 <= code <= 0x0E7F:
        return "thai"
    return "other"


def _dominant_script(payload: bytes) -> str:
    counts = Counter(
        name
        for character in payload.decode("utf-8", errors="replace")
        if (name := _script_name(character)) is not None
    )
    if not counts:
        return "none"
    return min(counts, key=lambda name: (-counts[name], name))


def _output_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("manifest outputs must be a list")
    matches = [
        row for row in outputs if isinstance(row, dict) and row.get("path") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one {name!r} output record")
    return matches[0]


def _read_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank training-index row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid training-index JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"training-index row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("training index is empty")
    return rows


def _validated_turns(
    blob: bytes,
    row: Mapping[str, Any],
    *,
    episode_start: int,
    episode_end: int,
) -> list[dict[str, Any]]:
    raw_turns = row.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("training row must contain nonempty turns")
    turns: list[dict[str, Any]] = []
    previous_end = episode_start
    for expected_index, raw in enumerate(raw_turns):
        if not isinstance(raw, dict):
            raise ValueError("turn records must be objects")
        index = int(raw.get("index", -1))
        role = str(raw.get("role", ""))
        separator = int(raw.get("separator_start", -1))
        marker = int(raw.get("marker_start", -1))
        content = int(raw.get("content_start", -1))
        end = int(raw.get("end", -1))
        if index != expected_index:
            raise ValueError("turn indexes must be contiguous and zero based")
        if role not in {"USER", "ASSISTANT"}:
            raise ValueError("turn role must be USER or ASSISTANT")
        if not (episode_start <= separator <= marker < content <= end <= episode_end):
            raise ValueError("turn offsets lie outside their indexed episode")
        if separator != previous_end:
            raise ValueError(
                "first turn must start at the episode boundary and later turn "
                "ranges must be contiguous"
            )
        if expected_index == 0:
            if marker != separator:
                raise ValueError("first role marker must start at the episode boundary")
        elif marker != separator + 2 or blob[separator:marker] != b"\n\n":
            raise ValueError(
                "later role markers must follow exactly one blank separator"
            )
        marker_match = _RAW_ROLE_MARKER.fullmatch(blob[marker:content])
        if marker_match is None:
            raise ValueError("indexed role marker does not match raw training bytes")
        raw_role = marker_match.group(1).decode("ascii")
        if role != raw_role:
            raise ValueError("indexed turn role disagrees with raw role marker")
        if expected_index and role == turns[-1]["role"]:
            raise ValueError("training conversations must alternate roles")
        turns.append(
            {
                "index": index,
                "role": role,
                "separator_start": separator,
                "marker_start": marker,
                "content_start": content,
                "end": end,
            }
        )
        previous_end = end
    if previous_end != episode_end:
        raise ValueError("last turn must end at the episode boundary")
    if turns[0]["role"] != "USER":
        raise ValueError("training conversations must begin with USER")
    return turns


def _event_vector(blob: bytes, turn: Mapping[str, Any], dimensions: int) -> np.ndarray:
    payload = blob[int(turn["content_start"]) : int(turn["end"])]
    return hashed_text_vector(payload.decode("utf-8", errors="replace"), dimensions)


def _causal_evidence(
    vectors: Sequence[np.ndarray], *, decays: tuple[float, ...]
) -> np.ndarray:
    if not vectors:
        raise ValueError("causal evidence needs at least the pending USER event")
    states = [np.zeros_like(vectors[0]) for _ in decays]
    for vector in vectors:
        for index, decay in enumerate(decays):
            states[index] = decay * states[index] + (1.0 - decay) * vector
    return np.concatenate((vectors[-1], *(normalize(state) for state in states)))


def _extract_transitions(
    blob: bytes,
    rows: Sequence[Mapping[str, Any]],
    config: AssayConfig,
) -> tuple[tuple[Transition, ...], dict[str, Any]]:
    transitions: list[Transition] = []
    seen_episodes: set[str] = set()
    previous_end = 0
    candidate_assistant_turns = 0
    excluded_without_later_user = 0
    excluded_empty_window = 0
    terminal_after_next_user = 0
    all_order_invariants = True
    for row_number, row in enumerate(rows, 1):
        start = int(row.get("start", -1))
        end = int(row.get("end", -1))
        if start != previous_end or not start < end <= len(blob):
            raise ValueError("training rows must contiguously cover the binary")
        if str(row.get("assigned_split", "")) != "train":
            raise ValueError("training index contains a non-training row")
        episode_id = str(row.get("episode_id", ""))
        if not episode_id or episode_id in seen_episodes:
            raise ValueError("episode IDs must be nonempty and unique")
        seen_episodes.add(episode_id)
        payload = blob[start:end]
        if int(row.get("episode_bytes", len(payload))) != len(payload):
            raise ValueError("episode byte count disagrees with indexed range")
        if str(row.get("sha256", "")) != _sha256_bytes(payload):
            raise ValueError(f"episode hash mismatch in training row {row_number}")
        turns = _validated_turns(
            blob,
            row,
            episode_start=start,
            episode_end=end,
        )
        vectors = [
            _event_vector(blob, turn, config.feature_dimensions) for turn in turns
        ]
        source_label = str(row.get("source_label", ""))
        if not source_label:
            raise ValueError("source_label must be nonempty")
        declared_source_raw = row.get("declared_source")
        if declared_source_raw is not None and not isinstance(declared_source_raw, str):
            raise ValueError("declared_source must be a string or null")
        declared_source = declared_source_raw
        source_path = str(row.get("source_path", ""))
        source_line = int(row.get("source_line", -1))
        source_byte_offset = int(row.get("source_byte_offset", -1))
        if not source_path or source_line < 1 or source_byte_offset < 0:
            raise ValueError("training row has incomplete source provenance")
        for position, turn in enumerate(turns):
            if turn["role"] != "ASSISTANT":
                continue
            candidate_assistant_turns += 1
            if position == 0 or turns[position - 1]["role"] != "USER":
                raise ValueError("assistant decision lacks a preceding USER turn")
            next_user_position = next(
                (
                    later
                    for later in range(position + 1, len(turns))
                    if turns[later]["role"] == "USER"
                ),
                None,
            )
            if next_user_position is None:
                excluded_without_later_user += 1
                continue
            pending = turns[position - 1]
            later_user = turns[next_user_position]
            next_assistant_position = next(
                (
                    later
                    for later in range(next_user_position + 1, len(turns))
                    if turns[later]["role"] == "ASSISTANT"
                ),
                None,
            )
            next_response_full = (
                None
                if next_assistant_position is None
                else blob[
                    int(turns[next_assistant_position]["content_start"]) : int(
                        turns[next_assistant_position]["end"]
                    )
                ]
            )
            next_response = (
                None
                if next_response_full is None
                else next_response_full[: config.response_bytes]
            )
            response_full = blob[int(turn["content_start"]) : int(turn["end"])]
            next_user_full = blob[
                int(later_user["content_start"]) : int(later_user["end"])
            ]
            response = response_full[: config.response_bytes]
            next_user = next_user_full[: config.next_user_bytes]
            if not response or not next_user:
                excluded_empty_window += 1
                continue
            history_start = start
            history_end = int(pending["end"])
            response_start = int(turn["content_start"])
            response_end = int(turn["end"])
            next_user_start = int(later_user["content_start"])
            next_user_end = int(later_user["end"])
            ordered = (
                history_end
                <= response_start
                < response_end
                <= next_user_start
                < next_user_end
                <= end
            )
            all_order_invariants = all_order_invariants and ordered
            if not ordered:
                raise ValueError("assistant-to-next-USER ranges are not causal")
            evidence = _causal_evidence(vectors[:position], decays=config.state_decays)
            decision_id = _sha256_bytes(
                f"{episode_id}\x1f{int(turn['index'])}".encode()
            )
            pending_payload = blob[int(pending["content_start"]) : int(pending["end"])]
            transitions.append(
                Transition(
                    decision_id=decision_id,
                    conversation_id=episode_id,
                    assistant_turn_index=int(turn["index"]),
                    source_label=source_label,
                    script=_dominant_script(pending_payload),
                    horizon_bin=_horizon_bin(response_start - start),
                    response_length_bin=_response_bin(len(response_full)),
                    history_start=history_start,
                    history_end=history_end,
                    response_start=response_start,
                    response_end=response_end,
                    response_window_end=response_start + len(response),
                    next_user_start=next_user_start,
                    next_user_end=next_user_end,
                    next_user_window_end=next_user_start + len(next_user),
                    next_history_end=next_user_end,
                    next_assistant_turn_index=(
                        None
                        if next_assistant_position is None
                        else int(turns[next_assistant_position]["index"])
                    ),
                    terminal_after_next_user=next_assistant_position is None,
                    history_sha256=_sha256_bytes(blob[history_start:history_end]),
                    response_sha256=_sha256_bytes(response_full),
                    next_user_sha256=_sha256_bytes(next_user_full),
                    response_window_sha256=_sha256_bytes(response),
                    next_user_window_sha256=_sha256_bytes(next_user),
                    next_history_sha256=_sha256_bytes(
                        blob[history_start:next_user_end]
                    ),
                    response=bytes(response),
                    next_user=bytes(next_user),
                    evidence=evidence,
                    declared_source=declared_source,
                    source_path=source_path,
                    source_line=source_line,
                    source_byte_offset=source_byte_offset,
                    next_response_start=(
                        None
                        if next_assistant_position is None
                        else int(turns[next_assistant_position]["content_start"])
                    ),
                    next_response_end=(
                        None
                        if next_assistant_position is None
                        else int(turns[next_assistant_position]["end"])
                    ),
                    next_response_window_end=(
                        None
                        if next_assistant_position is None or next_response is None
                        else int(turns[next_assistant_position]["content_start"])
                        + len(next_response)
                    ),
                    next_response_sha256=(
                        None
                        if next_response_full is None
                        else _sha256_bytes(next_response_full)
                    ),
                    next_response_window_sha256=(
                        None if next_response is None else _sha256_bytes(next_response)
                    ),
                    next_response=(
                        None if next_response is None else bytes(next_response)
                    ),
                )
            )
            terminal_after_next_user += int(next_assistant_position is None)
        previous_end = end
    if previous_end != len(blob):
        raise ValueError("training index does not cover the complete binary")
    if not transitions:
        raise ValueError("no eligible assistant-to-next-USER transitions")
    decision_ids = [row.decision_id for row in transitions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("derived decision IDs are not unique")
    transition_audit = [
        {
            "decision_id": row.decision_id,
            "conversation_id": row.conversation_id,
            "assistant_turn_index": row.assistant_turn_index,
            "source_label": row.source_label,
            "declared_source": row.declared_source,
            "source_path": row.source_path,
            "source_line": row.source_line,
            "source_byte_offset": row.source_byte_offset,
            "history_range": [row.history_start, row.history_end],
            "response_range": [row.response_start, row.response_end],
            "response_window_range": [
                row.response_start,
                row.response_window_end,
            ],
            "next_user_range": [row.next_user_start, row.next_user_end],
            "next_user_window_range": [
                row.next_user_start,
                row.next_user_window_end,
            ],
            "history_sha256": row.history_sha256,
            "response_sha256": row.response_sha256,
            "next_user_sha256": row.next_user_sha256,
            "response_window_sha256": row.response_window_sha256,
            "next_user_window_sha256": row.next_user_window_sha256,
            "next_history_sha256": row.next_history_sha256,
            "next_assistant_turn_index": row.next_assistant_turn_index,
            "next_assistant_response_range": (
                None
                if row.next_response_start is None or row.next_response_end is None
                else [row.next_response_start, row.next_response_end]
            ),
            "next_assistant_response_window_range": (
                None
                if row.next_response_start is None
                or row.next_response_window_end is None
                else [row.next_response_start, row.next_response_window_end]
            ),
            "next_assistant_response_sha256": row.next_response_sha256,
            "next_assistant_response_window_sha256": (row.next_response_window_sha256),
            "terminal_after_next_user": row.terminal_after_next_user,
        }
        for row in transitions
    ]
    return tuple(transitions), {
        "transition_index_schema": "decision-transition-v1-gate-zero",
        "transition_index_persisted": False,
        "materialized_transition_sidecar_is_a_follow_up": True,
        "training_conversations": len(seen_episodes),
        "candidate_assistant_turns": candidate_assistant_turns,
        "eligible_transitions": len(transitions),
        "excluded_without_later_user": excluded_without_later_user,
        "excluded_empty_scoring_window": excluded_empty_window,
        "transitions_terminal_after_next_user": terminal_after_next_user,
        "transitions_with_next_assistant_decision": len(transitions)
        - terminal_after_next_user,
        "response_window_bytes": config.response_bytes,
        "next_user_window_bytes": config.next_user_bytes,
        "scored_next_user_bytes": sum(len(row.next_user) for row in transitions),
        "all_range_order_invariants": all_order_invariants,
        "decision_ids_unique": True,
        "source_provenance_validated": True,
        "raw_role_markers_validated": True,
        "transition_index_hashed_fields": sorted(transition_audit[0]),
        "transition_index_sha256": _canonical_json_sha256(transition_audit),
        "transitions_by_source": dict(
            sorted(Counter(row.source_label for row in transitions).items())
        ),
        "transitions_by_script": dict(
            sorted(Counter(row.script for row in transitions).items())
        ),
    }


def load_training_transitions(dataset_dir: Path, config: AssayConfig) -> LoadedTraining:
    """Load and fully validate only the three hard-coded training inputs."""

    config.validate()
    directory = Path(dataset_dir).resolve()
    paths = {name: directory / name for name in TRAIN_INPUT_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required training input(s): " + ", ".join(sorted(missing))
        )
    manifest_payload = paths[MANIFEST_NAME].read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")

    file_provenance: dict[str, Any] = {}
    for name in (TRAIN_BINARY_NAME, TRAIN_INDEX_NAME):
        record = _output_record(manifest, name)
        path = paths[name]
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if int(record.get("bytes", -1)) != actual_bytes:
            raise ValueError(f"manifest byte count mismatch for {name}")
        if str(record.get("sha256", "")) != actual_sha256:
            raise ValueError(f"manifest SHA-256 mismatch for {name}")
        file_provenance[name] = {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
    blob = paths[TRAIN_BINARY_NAME].read_bytes()
    index_rows = _read_index(paths[TRAIN_INDEX_NAME])
    transitions, extraction = _extract_transitions(blob, index_rows, config)
    return LoadedTraining(
        transitions=transitions,
        manifest=manifest,
        provenance={
            "manifest": {
                "bytes": len(manifest_payload),
                "sha256": _sha256_bytes(manifest_payload),
                "schema_version": SCHEMA_VERSION,
            },
            "training_inputs": file_provenance,
            "opened_input_basenames": list(TRAIN_INPUT_NAMES),
        },
        extraction=extraction,
    )


def assign_conversation_folds(
    transitions: Sequence[Transition], *, seed: int
) -> dict[str, str]:
    """Assign whole conversations to source-stratified deterministic folds."""

    if seed < 0:
        raise ValueError("fold seed must be nonnegative")
    sources: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in transitions:
        sources[row.source_label][row.conversation_id] += 1
    assignment: dict[str, str] = {}
    for source, counts in sorted(sources.items()):
        conversations = sorted(
            counts,
            key=lambda item: (
                _stable_u64(seed, source, item, person=b"ncca-fold"),
                item,
            ),
        )
        for index, conversation_id in enumerate(conversations):
            assignment[conversation_id] = FOLD_NAMES[index % len(FOLD_NAMES)]
    observed = {assignment[row.conversation_id] for row in transitions}
    if observed != set(FOLD_NAMES):
        raise ValueError("every fold must contain at least one conversation")
    return assignment


def _response_vector(response: bytes, dimensions: int) -> np.ndarray:
    return hashed_text_vector(response.decode("utf-8", errors="replace"), dimensions)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.array(components, dtype=np.float64, copy=True)
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


def _spherical_kmeans(
    values: np.ndarray, *, clusters: int, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or not len(values):
        raise ValueError("spherical k-means needs a nonempty matrix")
    if not 2 <= clusters <= len(values):
        raise ValueError("cluster count must be in [2, number of responses]")
    rng = np.random.Generator(np.random.PCG64(seed))
    first = int(rng.integers(len(values)))
    center_rows = [values[first]]
    selected = {first}
    while len(center_rows) < clusters:
        similarity = values @ np.asarray(center_rows).T
        distance = np.maximum(0.0, 1.0 - similarity.max(axis=1))
        for index in selected:
            distance[index] = 0.0
        total = float(distance.sum())
        if total > 0.0:
            chosen = int(rng.choice(len(values), p=distance / total))
        else:
            remaining = [index for index in range(len(values)) if index not in selected]
            chosen = remaining[0]
        center_rows.append(values[chosen])
        selected.add(chosen)
    centers = np.asarray(center_rows, dtype=np.float64)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmax(values @ centers.T, axis=1).astype(np.int64)
        updated = np.zeros_like(centers)
        fit = np.max(values @ centers.T, axis=1)
        reserved: set[int] = set()
        for cluster in range(clusters):
            members = values[labels == cluster]
            if len(members):
                updated[cluster] = normalize(members.mean(axis=0))
            else:
                order = np.lexsort((np.arange(len(values)), fit))
                chosen = next(
                    int(index) for index in order if int(index) not in reserved
                )
                reserved.add(chosen)
                updated[cluster] = values[chosen]
        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break
        centers = updated
    labels = np.argmax(values @ centers.T, axis=1).astype(np.int64)
    return centers, labels


def _fold_support(labels: np.ndarray, folds: Sequence[str], actions: int) -> np.ndarray:
    support = np.zeros((len(FOLD_NAMES), actions), dtype=np.int64)
    for label, fold in zip(labels.tolist(), folds, strict=True):
        support[FOLD_NAMES.index(fold), int(label)] += 1
    return support


def _merge_for_fold_support(
    centroids: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[str],
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, tuple[Mapping[str, Any], ...], np.ndarray]:
    groups: dict[int, set[int]] = {index: {index} for index in range(len(centroids))}
    representatives = {
        index: np.array(row, copy=True) for index, row in enumerate(centroids)
    }
    initial_counts = np.bincount(labels, minlength=len(centroids)).astype(np.float64)
    history: list[Mapping[str, Any]] = []

    def mapping_and_support() -> tuple[np.ndarray, list[int], np.ndarray]:
        active = sorted(groups)
        final_index = {raw: index for index, raw in enumerate(active)}
        raw_to_active: dict[int, int] = {}
        for active_id, members in groups.items():
            for raw in members:
                raw_to_active[raw] = active_id
        mapped = np.asarray(
            [final_index[raw_to_active[int(label)]] for label in labels],
            dtype=np.int64,
        )
        return mapped, active, _fold_support(mapped, folds, len(active))

    while True:
        mapped, active, support = mapping_and_support()
        if np.all(support >= minimum):
            break
        if len(active) <= 2:
            raise ValueError(
                "action support merging would leave fewer than two supported actions"
            )
        totals = support.sum(axis=0)
        victim_final = min(
            range(len(active)),
            key=lambda index: (
                int(support[:, index].min()),
                int(totals[index]),
                active[index],
            ),
        )
        victim = active[victim_final]
        target = max(
            (item for item in active if item != victim),
            key=lambda item: (
                float(representatives[victim] @ representatives[item]),
                -item,
            ),
        )
        victim_weight = float(sum(initial_counts[list(groups[victim])]))
        target_weight = float(sum(initial_counts[list(groups[target])]))
        combined = (
            target_weight * representatives[target]
            + victim_weight * representatives[victim]
        )
        representatives[target] = normalize(combined)
        groups[target].update(groups.pop(victim))
        representatives.pop(victim)
        history.append(
            {
                "victim_initial_cluster": victim,
                "target_initial_cluster": target,
                "victim_min_fold_support": int(support[:, victim_final].min()),
                "victim_total_support": int(totals[victim_final]),
            }
        )
    mapped, active, support = mapping_and_support()
    raw_to_final = np.empty(len(centroids), dtype=np.int64)
    for final, active_id in enumerate(active):
        for raw in groups[active_id]:
            raw_to_final[raw] = final
    if not np.array_equal(mapped, raw_to_final[labels]):
        raise AssertionError("action merge mapping is internally inconsistent")
    return raw_to_final, support, tuple(history), mapped


def fit_action_tokenizer(
    responses: Sequence[bytes],
    fold_names: Sequence[str],
    config: AssayConfig,
) -> tuple[ActionTokenizer, np.ndarray]:
    """Fit using an API that cannot receive any later-interlocutor outcome."""

    if len(responses) != len(fold_names):
        raise ValueError("responses and fold names must have equal length")
    if any(fold not in FOLD_NAMES for fold in fold_names):
        raise ValueError("tokenizer fold names must be A, B, or C")
    if len(responses) < config.initial_actions:
        raise ValueError("too few response rows for the initial action count")
    matrix = np.asarray(
        [
            _response_vector(response, config.feature_dimensions)
            for response in responses
        ],
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    available = min(centered.shape)
    if config.pca_dimensions > available:
        raise ValueError("pca_dimensions exceeds the response matrix rank bound")
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = _canonicalize_component_signs(right[: config.pca_dimensions])
    latent = _normalize_rows(centered @ components.T)
    centroids, initial_labels = _spherical_kmeans(
        latent,
        clusters=config.initial_actions,
        iterations=config.kmeans_iterations,
        seed=config.tokenizer_seed,
    )
    mapping, support, history, labels = _merge_for_fold_support(
        centroids,
        initial_labels,
        fold_names,
        config.min_action_support,
    )
    tokenizer = ActionTokenizer(
        vector_dimensions=config.feature_dimensions,
        mean=mean,
        components=components,
        initial_centroids=centroids,
        initial_to_final=mapping,
        fold_support=support,
        merge_history=history,
    )
    assigned_again = np.asarray(
        [tokenizer.assign(response) for response in responses], dtype=np.int64
    )
    if not np.array_equal(labels, assigned_again):
        raise AssertionError("tokenizer reassignment changed fitted response labels")
    return tokenizer, labels


def _sattolo_indices(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("Sattolo shuffle needs at least two rows")
    permutation = np.arange(count, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for index in range(count - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = (
            permutation[other],
            permutation[index],
        )
    if np.any(permutation == np.arange(count)):
        raise AssertionError("Sattolo shuffle unexpectedly has a fixed point")
    return permutation


def stratified_shuffled_actions(
    transitions: Sequence[Transition],
    actions: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derange rows inside every exact preregistered training stratum."""

    if len(transitions) != len(actions):
        raise ValueError("transitions and actions must have equal length")
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(transitions):
        groups[row.shuffle_stratum].append(index)
    singleton = [stratum for stratum, indexes in groups.items() if len(indexes) < 2]
    if singleton:
        raise ValueError(
            "exact shuffle strata contain singleton rows; protocol amendment "
            "required before real-data execution"
        )
    source_actions = np.asarray(actions, dtype=np.int64)
    result = np.empty_like(source_actions)
    index_fixed_points = 0
    action_equalities = 0
    sizes: list[int] = []
    for stratum, indexes in sorted(groups.items()):
        ordered = sorted(indexes, key=lambda index: transitions[index].decision_id)
        local_seed = _stable_u64(seed, *stratum, person=b"ncca-shuf")
        permutation = _sattolo_indices(len(ordered), local_seed)
        for local_index, destination in enumerate(ordered):
            source = ordered[int(permutation[local_index])]
            result[destination] = source_actions[source]
            index_fixed_points += int(destination == source)
            action_equalities += int(result[destination] == source_actions[destination])
        sizes.append(len(ordered))
    return result, {
        "strata": len(groups),
        "minimum_stratum_rows": min(sizes),
        "maximum_stratum_rows": max(sizes),
        "row_index_fixed_points": index_fixed_points,
        "effective_action_label_equalities": action_equalities,
        "effective_action_label_equality_fraction": action_equalities
        / len(transitions),
    }


def top_signed_feature_ids(vector: np.ndarray, count: int) -> tuple[int, ...]:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("continuous state must be a nonempty finite vector")
    if not 1 <= count <= len(values):
        raise ValueError("state feature count is outside the vector width")
    coordinates = np.arange(len(values), dtype=np.int64)
    order = np.lexsort((coordinates, -np.abs(values)))[:count]
    return tuple(2 * int(index) + int(values[index] < 0.0) for index in order.tolist())


def persistent_slots(state_ids: Sequence[int], action: int | None) -> tuple[int, ...]:
    """Create the allocation-matched 8 + 1 + 8 persistent-slot layout."""

    if len(state_ids) != LOCKED_STATE_FEATURE_COUNT:
        raise ValueError(
            f"persistent state must contain {LOCKED_STATE_FEATURE_COUNT} IDs"
        )
    action_key = "NULL" if action is None else int(action)
    state = tuple(
        _stable_u64("state", int(value), person=b"ncca-feat") for value in state_ids
    )
    action_id = _stable_u64("action", action_key, person=b"ncca-feat")
    joint = tuple(
        _stable_u64("joint", value, action_key, person=b"ncca-feat") for value in state
    )
    slots = (*state, action_id, *joint)
    if len(slots) != PERSISTENT_SLOT_COUNT:
        raise AssertionError("persistent slot layout has the wrong width")
    return slots


def _writer(config: AssayConfig) -> ContextCoupledByteWriter:
    return ContextCoupledByteWriter(
        "coupled",
        alphabet_size=256,
        table_bits=config.table_bits,
        learning_rate=config.learning_rate,
        local_orders=config.local_orders,
        max_persistent_features=PERSISTENT_SLOT_COUNT,
    )


def _state_ids(row: Transition, config: AssayConfig) -> tuple[int, ...]:
    return top_signed_feature_ids(row.evidence, config.state_feature_count)


def _score_episode_bits(
    writer: ContextCoupledByteWriter,
    target: bytes,
    slots: Sequence[int],
) -> float:
    prefix = bytearray()
    bits = 0.0
    for symbol in target:
        probability = float(writer.distribution(prefix, slots)[symbol])
        bits -= math.log2(max(probability, np.finfo(np.float64).tiny))
        prefix.append(symbol)
    return bits


def _fit_and_score_rotation(
    transitions: Sequence[Transition],
    actions: np.ndarray,
    folds: Mapping[str, str],
    *,
    fit_fold: str,
    score_fold: str,
    config: AssayConfig,
    checkpoint_hook: CheckpointHook | None = None,
) -> tuple[list[ScoredTransition], dict[str, Any]]:
    fit_indices = [
        index
        for index, row in enumerate(transitions)
        if folds[row.conversation_id] == fit_fold
    ]
    score_indices = [
        index
        for index, row in enumerate(transitions)
        if folds[row.conversation_id] == score_fold
    ]
    if not fit_indices or not score_indices:
        raise ValueError("rotation fit and score folds must both be nonempty")
    fit_rows = [transitions[index] for index in fit_indices]
    fit_actions = actions[fit_indices]
    shuffled, shuffle_report = stratified_shuffled_actions(
        fit_rows, fit_actions, seed=config.shuffle_seed + FOLD_NAMES.index(fit_fold)
    )
    q_h = _writer(config)
    q_ha = _writer(config)
    q_shuffle = _writer(config)
    training_bytes = 0
    checkpoint_rows = sorted(
        {
            min(
                len(fit_rows),
                max(1, int(math.ceil(len(fit_rows) * fraction))),
            )
            for fraction in CHECKPOINT_FRACTIONS
        }
    )
    checkpoint_records: list[dict[str, Any]] = []
    for local_index, row in enumerate(fit_rows):
        state = _state_ids(row, config)
        action = int(fit_actions[local_index])
        target = row.next_user
        q_h.fit_episode(target, persistent_slots(state, None))
        q_ha.fit_episode(target, persistent_slots(state, action))
        q_shuffle.fit_episode(
            target, persistent_slots(state, int(shuffled[local_index]))
        )
        training_bytes += len(target)
        rows_seen = local_index + 1
        if rows_seen in checkpoint_rows:
            fingerprints = {
                "q_H": q_h.fingerprint(),
                "q_HA": q_ha.fingerprint(),
                "q_stratified_shuffle": q_shuffle.fingerprint(),
            }
            checkpoint_records.append(
                {
                    "fit_transitions_seen": rows_seen,
                    "training_bytes_per_compressor_seen": training_bytes,
                    "fingerprints": fingerprints,
                }
            )
            if checkpoint_hook is not None:
                checkpoint_hook(
                    f"{fit_fold}_to_{score_fold}",
                    rows_seen,
                    training_bytes,
                    {
                        "q_H": q_h.snapshot(),
                        "q_HA": q_ha.snapshot(),
                        "q_stratified_shuffle": q_shuffle.snapshot(),
                    },
                )
    before = {
        "q_H": q_h.fingerprint(),
        "q_HA": q_ha.fingerprint(),
        "q_stratified_shuffle": q_shuffle.fingerprint(),
    }
    scored: list[ScoredTransition] = []
    for index in score_indices:
        row = transitions[index]
        state = _state_ids(row, config)
        action = int(actions[index])
        scored.append(
            ScoredTransition(
                conversation_id=row.conversation_id,
                decision_id=row.decision_id,
                action=action,
                target_bytes=len(row.next_user),
                bits_h=_score_episode_bits(
                    q_h, row.next_user, persistent_slots(state, None)
                ),
                bits_ha=_score_episode_bits(
                    q_ha, row.next_user, persistent_slots(state, action)
                ),
                bits_shuffle=_score_episode_bits(
                    q_shuffle, row.next_user, persistent_slots(state, action)
                ),
            )
        )
    after = {
        "q_H": q_h.fingerprint(),
        "q_HA": q_ha.fingerprint(),
        "q_stratified_shuffle": q_shuffle.fingerprint(),
    }
    if before != after:
        raise AssertionError("evaluation modified a frozen compressor")
    parameter_counts = {
        q_h.parameter_count,
        q_ha.parameter_count,
        q_shuffle.parameter_count,
    }
    allocated_bytes = {
        q_h.allocated_bytes,
        q_ha.allocated_bytes,
        q_shuffle.allocated_bytes,
    }
    active_counts = {
        q_h.raw_active_feature_count(PERSISTENT_SLOT_COUNT),
        q_ha.raw_active_feature_count(PERSISTENT_SLOT_COUNT),
        q_shuffle.raw_active_feature_count(PERSISTENT_SLOT_COUNT),
    }
    if (
        len(parameter_counts) != 1
        or len(allocated_bytes) != 1
        or len(active_counts) != 1
    ):
        raise AssertionError("compressor allocations are not matched")
    return scored, {
        "fit_fold": fit_fold,
        "score_fold": score_fold,
        "fit_conversations": len({row.conversation_id for row in fit_rows}),
        "score_conversations": len(
            {transitions[index].conversation_id for index in score_indices}
        ),
        "fit_transitions": len(fit_rows),
        "score_transitions": len(scored),
        "training_bytes_per_compressor": training_bytes,
        "checkpoint_schedule": checkpoint_records,
        "checkpoint_hook_invoked": checkpoint_hook is not None,
        "scored_bytes_per_compressor": sum(row.target_bytes for row in scored),
        "shuffle": shuffle_report,
        "allocation": {
            "parameter_count_per_compressor": parameter_counts.pop(),
            "allocated_bytes_per_compressor": allocated_bytes.pop(),
            "raw_active_feature_count": active_counts.pop(),
            "persistent_slots": PERSISTENT_SLOT_COUNT,
            "all_three_exactly_matched": True,
        },
        "weights_frozen_during_scoring": True,
        "fit_score_conversation_overlap": len(
            {row.conversation_id for row in fit_rows}
            & {transitions[index].conversation_id for index in score_indices}
        ),
        "fingerprints_after_training": before,
        "scored_transition_sha256": _canonical_json_sha256(
            [asdict(row) for row in scored]
        ),
    }


def conversation_cluster_bootstrap(
    scored: Sequence[ScoredTransition],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Bootstrap whole conversations and retain byte-weighted BPB contrasts."""

    if not scored:
        raise ValueError("cannot bootstrap an empty scored transition set")
    if seed < 0 or resamples < 100:
        raise ValueError("bootstrap needs a nonnegative seed and >=100 resamples")
    grouped: dict[str, np.ndarray] = {}
    conversation_rows: dict[str, list[ScoredTransition]] = {}
    for conversation_id in sorted({row.conversation_id for row in scored}):
        rows = [row for row in scored if row.conversation_id == conversation_id]
        conversation_rows[conversation_id] = rows
        grouped[conversation_id] = np.asarray(
            [
                sum(row.bits_h - row.bits_ha for row in rows),
                sum(row.bits_shuffle - row.bits_ha for row in rows),
                sum(row.target_bytes for row in rows),
                sum(row.bits_h for row in rows),
                sum(row.bits_ha for row in rows),
                sum(row.bits_shuffle for row in rows),
            ],
            dtype=np.float64,
        )
    conversations = list(grouped)
    if len(conversations) < 2:
        raise ValueError("conversation bootstrap needs at least two conversations")
    matrix = np.asarray([grouped[key] for key in conversations])
    totals = matrix.sum(axis=0)
    if totals[2] <= 0.0:
        raise ValueError("bootstrap rows contain no target bytes")
    rng = np.random.Generator(np.random.PCG64(seed))
    indexes = rng.integers(0, len(conversations), size=(resamples, len(conversations)))
    sampled = matrix[indexes].sum(axis=1)
    improvement_ha = sampled[:, 0] / sampled[:, 2]
    improvement_shuffle = sampled[:, 1] / sampled[:, 2]

    def summary(values: np.ndarray, point: float) -> dict[str, Any]:
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        return {
            "point_estimate_bits_per_byte": float(point),
            "bootstrap_ci95": [float(low), float(high)],
            "bootstrap_ci95_above_zero": bool(low > 0.0),
        }

    byte_count = float(totals[2])
    aggregates = [
        {
            "conversation_id": conversation_id,
            "transitions": len(conversation_rows[conversation_id]),
            "target_bytes": int(grouped[conversation_id][2]),
            "bits_q_H": float(grouped[conversation_id][3]),
            "bits_q_HA": float(grouped[conversation_id][4]),
            "bits_q_stratified_shuffle": float(grouped[conversation_id][5]),
            "q_HA_improvement_bits": float(grouped[conversation_id][0]),
            "q_HA_over_shuffle_improvement_bits": float(grouped[conversation_id][1]),
            "action_counts": dict(
                sorted(
                    Counter(
                        row.action for row in conversation_rows[conversation_id]
                    ).items()
                )
            ),
        }
        for conversation_id in conversations
    ]
    return {
        "method": "conversation-clustered nonparametric percentile bootstrap",
        "conversation_count": len(conversations),
        "minimum_inferential_conversations": LOCKED_MIN_SCORED_CONVERSATIONS,
        "enough_conversations_for_gate": (
            len(conversations) >= LOCKED_MIN_SCORED_CONVERSATIONS
        ),
        "transition_count": len(scored),
        "target_bytes": int(totals[2]),
        "bpb": {
            "q_H": float(totals[3] / byte_count),
            "q_HA": float(totals[4] / byte_count),
            "q_stratified_shuffle": float(totals[5] / byte_count),
        },
        "q_HA_improvement_over_q_H": summary(
            improvement_ha, float(totals[0] / byte_count)
        ),
        "q_HA_improvement_over_stratified_shuffle": summary(
            improvement_shuffle, float(totals[1] / byte_count)
        ),
        "conversation_aggregates": aggregates,
        "conversation_aggregates_sha256": _canonical_json_sha256(aggregates),
    }


def _locked_config() -> AssayConfig:
    return AssayConfig()


def protocol_eligibility(config: AssayConfig) -> dict[str, Any]:
    locked = asdict(_locked_config())
    observed = asdict(config)
    deviations = {
        key: {"locked": locked[key], "observed": value}
        for key, value in observed.items()
        if value != locked[key]
    }
    reasons = [f"nonlocked_{key}" for key in sorted(deviations)]
    if config.table_bits < LOCKED_TABLE_BITS:
        reasons.append("diagnostic_small_table_never_selection_eligible")
    return {
        "selection_eligible": not reasons,
        "ineligible_reasons": reasons,
        "deviations": deviations,
        "diagnostic_small_table": config.table_bits < LOCKED_TABLE_BITS,
    }


def run_assay(
    dataset_dir: Path,
    config: AssayConfig | None = None,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Run the train-only gate without loading or connecting a native model."""

    started = time.perf_counter()
    chosen = _locked_config() if config is None else config
    chosen.validate()
    loaded = load_training_transitions(dataset_dir, chosen)
    transitions = loaded.transitions
    folds = assign_conversation_folds(transitions, seed=chosen.fold_seed)
    response_bytes = tuple(row.response for row in transitions)
    response_fold_names = tuple(folds[row.conversation_id] for row in transitions)
    tokenizer, actions = fit_action_tokenizer(
        response_bytes,
        response_fold_names,
        chosen,
    )

    fold_report: dict[str, Any] = {}
    for fold in FOLD_NAMES:
        indexes = [
            index
            for index, row in enumerate(transitions)
            if folds[row.conversation_id] == fold
        ]
        fold_report[fold] = {
            "conversations": len(
                {transitions[index].conversation_id for index in indexes}
            ),
            "transitions": len(indexes),
            "action_counts": np.bincount(
                actions[indexes], minlength=tokenizer.action_count
            ).tolist(),
            "conversations_by_source": dict(
                sorted(
                    Counter(
                        source
                        for source, _ in {
                            (
                                transitions[index].source_label,
                                transitions[index].conversation_id,
                            )
                            for index in indexes
                        }
                    ).items()
                )
            ),
            "transitions_by_source": dict(
                sorted(
                    Counter(
                        transitions[index].source_label for index in indexes
                    ).items()
                )
            ),
        }

    rotations: list[dict[str, Any]] = []
    for rotation_index, (fit_fold, score_fold) in enumerate(ROTATIONS):
        scored, audit = _fit_and_score_rotation(
            transitions,
            actions,
            folds,
            fit_fold=fit_fold,
            score_fold=score_fold,
            config=chosen,
            checkpoint_hook=checkpoint_hook,
        )
        bootstrap = conversation_cluster_bootstrap(
            scored,
            seed=chosen.bootstrap_seed + rotation_index,
            resamples=chosen.bootstrap_resamples,
        )
        rotations.append(
            {
                "rotation": f"{fit_fold}_to_{score_fold}",
                "audit": audit,
                "bootstrap": bootstrap,
                "action_counts_scored": np.bincount(
                    [row.action for row in scored], minlength=tokenizer.action_count
                ).tolist(),
            }
        )

    eligibility = protocol_eligibility(chosen)
    statistical_pass = all(
        row["bootstrap"]["enough_conversations_for_gate"]
        and row["bootstrap"]["q_HA_improvement_over_q_H"]["bootstrap_ci95_above_zero"]
        and row["bootstrap"]["q_HA_improvement_over_stratified_shuffle"][
            "bootstrap_ci95_above_zero"
        ]
        for row in rotations
    )
    source_path = Path(__file__).resolve()
    writer_module_file = sys.modules[ContextCoupledByteWriter.__module__].__file__
    if writer_module_file is None:
        raise RuntimeError("context-coupled writer module has no source path")
    writer_path = Path(writer_module_file).resolve()
    fold_conversation_sets = {
        fold: {
            row.conversation_id
            for row in transitions
            if folds[row.conversation_id] == fold
        }
        for fold in FOLD_NAMES
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "task": "ncca_train_only_proxy_observability_gate_zero",
        "claim_scope": (
            "strictly later USER bytes can reveal predictive dependence on an "
            "anonymous logged response action; not utility, satisfaction, causal "
            "policy benefit, goal inference, or native generation"
        ),
        "config": asdict(chosen),
        "protocol_eligibility": eligibility,
        "provenance": {
            **loaded.provenance,
            "experiment_source": str(source_path),
            "experiment_source_sha256": _sha256_file(source_path),
            "writer_source": str(writer_path),
            "writer_source_sha256": _sha256_file(writer_path),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "rng": "numpy.random.Generator(PCG64)",
        },
        "extraction": loaded.extraction,
        "folds": fold_report,
        "conversation_fold_assignment_sha256": _canonical_json_sha256(
            dict(sorted(folds.items()))
        ),
        "tokenizer": {
            "fit_scope": "all training responses; no next-USER bytes or outcomes",
            "pca_dimensions": chosen.pca_dimensions,
            "initial_actions": chosen.initial_actions,
            "retained_actions": tokenizer.action_count,
            "minimum_required_support_per_fold": chosen.min_action_support,
            "fold_support": tokenizer.fold_support.tolist(),
            "merge_history": list(tokenizer.merge_history),
            "mean_sha256": _sha256_bytes(tokenizer.mean.tobytes()),
            "components_sha256": _sha256_bytes(tokenizer.components.tobytes()),
            "initial_centroids_sha256": _sha256_bytes(
                tokenizer.initial_centroids.tobytes()
            ),
            "mapping_sha256": _sha256_bytes(tokenizer.initial_to_final.tobytes()),
            "action_counts": np.bincount(
                actions, minlength=tokenizer.action_count
            ).tolist(),
            "transition_action_assignment_sha256": _canonical_json_sha256(
                [
                    {
                        "decision_id": row.decision_id,
                        "action": int(actions[index]),
                    }
                    for index, row in enumerate(transitions)
                ]
            ),
            "next_logged_action_assignment_sha256": _canonical_json_sha256(
                [
                    {
                        "decision_id": row.decision_id,
                        "next_action": (
                            None
                            if row.next_response is None
                            else tokenizer.assign(row.next_response)
                        ),
                        "terminal": row.terminal_after_next_user,
                    }
                    for row in transitions
                ]
            ),
            "action_coverage_fraction": float(
                np.mean(
                    np.asarray(
                        [
                            tokenizer.fold_support[
                                FOLD_NAMES.index(folds[row.conversation_id]),
                                int(actions[index]),
                            ]
                            >= chosen.min_action_support
                            for index, row in enumerate(transitions)
                        ]
                    )
                )
            ),
        },
        "rotations": rotations,
        "gate_zero": {
            "statistical_pattern_passed": statistical_pass,
            "selection_eligible": eligibility["selection_eligible"],
            "passed": bool(statistical_pass and eligibility["selection_eligible"]),
            "required": (
                f"each rotation scores at least {LOCKED_MIN_SCORED_CONVERSATIONS} "
                "conversations and all three conversation-bootstrap CI95 intervals "
                "are strictly above zero for q_HA versus q_H and stratified shuffle"
            ),
        },
        "invariants": {
            "only_hard_coded_training_inputs_opened": tuple(
                loaded.provenance["opened_input_basenames"]
            )
            == TRAIN_INPUT_NAMES,
            "manifest_schema_v4": True,
            "every_derived_row_assigned_training": True,
            "strict_response_before_next_user": bool(
                loaded.extraction["all_range_order_invariants"]
            ),
            "raw_role_markers_validated_and_excluded_from_scoring_windows": bool(
                loaded.extraction["raw_role_markers_validated"]
            ),
            "complete_conversation_in_one_fold": all(
                len(
                    {
                        folds[row.conversation_id]
                        for row in transitions
                        if row.conversation_id == conversation
                    }
                )
                == 1
                for conversation in folds
            ),
            "fold_conversation_sets_disjoint": all(
                fold_conversation_sets[left].isdisjoint(fold_conversation_sets[right])
                for left_index, left in enumerate(FOLD_NAMES)
                for right in FOLD_NAMES[left_index + 1 :]
            ),
            "tokenizer_outcome_blind_by_api": True,
            "tokenizer_all_actions_meet_fold_support": bool(
                np.all(tokenizer.fold_support >= chosen.min_action_support)
            ),
            "compressor_allocation_matched_every_rotation": all(
                row["audit"]["allocation"]["all_three_exactly_matched"]
                for row in rotations
            ),
            "compressor_weights_frozen_during_scoring": all(
                row["audit"]["weights_frozen_during_scoring"] for row in rotations
            ),
            "rotation_fit_score_conversation_overlap_zero": all(
                row["audit"]["fit_score_conversation_overlap"] == 0 for row in rotations
            ),
            "shuffle_row_index_fixed_points_zero": all(
                row["audit"]["shuffle"]["row_index_fixed_points"] == 0
                for row in rotations
            ),
            "q_H_receives_null_action_only": True,
            "later_external_bytes_never_enter_action_tokenizer": True,
            "no_native_model_loaded_or_writer_connected": True,
        },
        "scientific_contract": {
            "gate_zero_is_signal_detectability_only": True,
            "observed_action_is_hindsight_tokenized_after_response": True,
            "outcome": (
                "bits(q_H on next USER) minus bits(q_HA on next USER), per byte"
            ),
            "positive_outcome_means_predictive_information_not_desirability": True,
            "alternative_action_outcomes_not_identified": True,
            "mechanism_selection_requires_trained_free_generation_curve": True,
            "frozen_receiver_result_is_interface_diagnostic_not_mechanism_veto": True,
            "downstream_primary_endpoint_is_free_generation_not_this_bpb": True,
            "checkpoint_hook_available_for_later_convergence_curve": True,
            "checkpoint_hook_is_snapshot_only_not_convergence_evidence": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_content_sha256_excluding_self"] = _canonical_json_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-table-bits",
        type=int,
        help=(
            "Use a table smaller than 14 bits for engineering diagnostics. "
            "This always makes the result selection-ineligible."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.diagnostic_table_bits is None:
        config = _locked_config()
    else:
        if not 1 <= args.diagnostic_table_bits < LOCKED_TABLE_BITS:
            raise ValueError(
                f"diagnostic table bits must be in [1, {LOCKED_TABLE_BITS - 1}]"
            )
        config = AssayConfig(table_bits=args.diagnostic_table_bits)
    report_path = Path(args.report).resolve()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report = run_assay(Path(args.dataset_dir), config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate_zero"], sort_keys=True))


if __name__ == "__main__":
    main()
