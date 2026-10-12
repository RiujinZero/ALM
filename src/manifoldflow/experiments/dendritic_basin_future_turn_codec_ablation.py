"""Development-only codec ceiling for a prospective future-turn representation.

This is deliberately the first, diagnostic stage of the receding-horizon idea.
It asks whether a train-fit dense code of an observed assistant turn can
*causally drive* byte production when it is supplied to a prefix-by-latent
decoder.  It also fits a small, explicitly non-recurrent ridge map from causal
conversation/current-turn features to that code.  It does not implement a
multihorizon trajectory, refinement, consequence value, or goal learner.

The oracle condition encodes the held-out answer itself and is therefore a
leakage ceiling, never a deployable score.  Actual bytes are generated directly
from a frozen native cortex at every step.  Free generation is primary;
teacher-forced bits per byte is only a support/collapse diagnostic.

Only schema-4 training conversations and the development crossover payload are
addressable from this source.  There is intentionally no configurable split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from manifoldflow.dendritic_basin.context_coupled_byte_writer import (
    ContextCoupledByteWriter,
)
from manifoldflow.dendritic_basin.vector_conditioned_base import (
    VectorConditionedFitDiagnostics,
    VectorConditionedGeneration,
    fit_vector_conditioned_turn,
    generate_from_vector,
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

ASSISTANT_ROLE = "ASSISTANT"
USER_ROLE = "USER"
ROLE_LABELS = (USER_ROLE, ASSISTANT_ROLE)
ASSISTANT_PROMPT = b"\n\nASSISTANT: "
_TOP_LEVEL_TURN_PATTERN = re.compile(rb"(?:\A|\n\n)(USER|ASSISTANT):[ \t]*")
KNOWN_SMOKE_PAIR_IDS = (
    "8c14b02a3307015f0eb398651cf74ec30e1368f84986c091fc97e114eb56fb31",
    "9bd6656828136ad076e4ea7e070be0b9f04fa723473a5b0ac7f6ab5e3eeefc3f",
)
IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py",
    "tests/test_dendritic_basin_future_turn_codec_ablation.py",
    "src/manifoldflow/dendritic_basin/context_coupled_byte_writer.py",
    "src/manifoldflow/dendritic_basin/vector_conditioned_base.py",
    "src/manifoldflow/dendritic_basin/native_cortex.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_context_coupled_writer_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_crossover_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_shared_goal_generation_ablation.py",
)
CONDITION_NAMES = (
    "native",
    "oracle_code_gain0",
    "local_decoder",
    "oracle_code_coupled",
    "oracle_code_additive",
    "oracle_code_shuffled",
    "oracle_code_paired_swapped",
    "predicted_code_coupled",
    "predicted_code_additive",
    "predicted_code_shuffled",
)
CONTROLLED_CONTRASTS = {
    "oracle_coupled_minus_native": ("oracle_code_coupled", "native"),
    "oracle_coupled_minus_additive": (
        "oracle_code_coupled",
        "oracle_code_additive",
    ),
    "oracle_coupled_minus_shuffled": (
        "oracle_code_coupled",
        "oracle_code_shuffled",
    ),
    "oracle_coupled_minus_paired_swap": (
        "oracle_code_coupled",
        "oracle_code_paired_swapped",
    ),
    "predicted_coupled_minus_native": ("predicted_code_coupled", "native"),
    "predicted_coupled_minus_additive": (
        "predicted_code_coupled",
        "predicted_code_additive",
    ),
    "predicted_coupled_minus_shuffled": (
        "predicted_code_coupled",
        "predicted_code_shuffled",
    ),
    "oracle_coupled_minus_predicted_coupled": (
        "oracle_code_coupled",
        "predicted_code_coupled",
    ),
}


class FrozenCortex(Protocol):
    """Read-only cortex surface used by the byte adapter."""

    def reset_episode(self) -> None: ...

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> Any: ...

    def predict_next_distribution(self) -> np.ndarray: ...

    def advance_observation(self, byte_id: int, *, learn: bool) -> Any: ...


@dataclass(frozen=True)
class AssistantTurn:
    """One completed, observed assistant turn and its causal prefix."""

    episode_id: str
    source_label: str
    turn_index: int
    prompt: bytes
    current_user_turn: bytes
    content: bytes
    content_start: int
    content_end: int


@dataclass(frozen=True)
class TrainingConversation:
    episode_id: str
    source_label: str
    payload: bytes
    assistant_turns: tuple[AssistantTurn, ...]


@dataclass(frozen=True)
class DevelopmentSide:
    interrupted_context: bytes
    original_context: bytes | None
    request: bytes
    response_window: bytes
    full_response: bytes


@dataclass(frozen=True)
class DevelopmentPair:
    pair_id: str
    source_label: str
    partition: str
    side_a: DevelopmentSide
    side_b: DevelopmentSide


@dataclass(frozen=True)
class TurnCode:
    """Dense PCA content code with explicit categorical binding coordinates."""

    vector: np.ndarray
    content_dimensions: int
    role: str
    source: str

    @property
    def content(self) -> np.ndarray:
        return self.vector[: self.content_dimensions]


@dataclass(frozen=True)
class FutureTurnCodec:
    """Train-fit byte adapter for a modality-agnostic dense turn bottleneck."""

    raw_dimensions: int
    content_dimensions: int
    active_content_features: int
    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    magnitude_thresholds: np.ndarray
    source_labels: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        turns: Sequence[AssistantTurn],
        *,
        raw_dimensions: int,
        content_dimensions: int,
        active_content_features: int,
    ) -> FutureTurnCodec:
        """Fit PCA and quantizer statistics using observed training turns only."""

        if len(turns) < 2:
            raise ValueError("at least two observed assistant turns are required")
        if raw_dimensions < 16:
            raise ValueError("raw dimensions must be at least 16")
        if not 1 <= content_dimensions <= raw_dimensions:
            raise ValueError("content dimensions are outside the raw feature width")
        if not 1 <= active_content_features <= content_dimensions:
            raise ValueError("active content features exceed the dense code width")
        if any(not turn.content for turn in turns):
            raise ValueError("completed assistant turns must be nonempty")

        matrix = np.asarray(
            [_hashed_byte_features(turn.content, raw_dimensions) for turn in turns],
            dtype=np.float64,
        )
        mean = np.mean(matrix, axis=0)
        centered = matrix - mean
        denominator = max(len(matrix) - 1, 1)
        covariance = (centered.T @ centered) / denominator
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues, kind="stable")[::-1][:content_dimensions]
        components = np.asarray(eigenvectors[:, order].T, dtype=np.float64)
        # Eigenspaces have an arbitrary sign.  Fix it from the largest loading
        # so repeated fits have stable sparse addresses.
        for row in components:
            pivot = int(np.argmax(np.abs(row)))
            if row[pivot] < 0.0:
                row *= -1.0
        selected_values = np.maximum(eigenvalues[order], 1e-12)
        scales = np.sqrt(selected_values)
        scores = (centered @ components.T) / scales
        magnitude_thresholds = np.quantile(np.abs(scores), (0.50, 0.75, 0.90), axis=0).T
        sources = tuple(sorted({turn.source_label for turn in turns}))
        if not sources:
            raise ValueError("training turns have no source binding")
        for array in (mean, components, scales, magnitude_thresholds):
            array.flags.writeable = False
        return cls(
            raw_dimensions=raw_dimensions,
            content_dimensions=content_dimensions,
            active_content_features=active_content_features,
            mean=mean,
            components=components,
            scales=scales,
            magnitude_thresholds=magnitude_thresholds,
            source_labels=sources,
        )

    @property
    def dense_dimensions(self) -> int:
        """Content plus explicit role/source one-hot coordinates."""

        return self.content_dimensions + len(ROLE_LABELS) + len(self.source_labels) + 1

    @property
    def persistent_feature_count(self) -> int:
        return self.active_content_features + 2

    def encode(
        self,
        payload: bytes,
        *,
        role: str = ASSISTANT_ROLE,
        source: str,
    ) -> TurnCode:
        """Encode bytes without refitting any statistic."""

        raw = _hashed_byte_features(payload, self.raw_dimensions)
        content = ((raw - self.mean) @ self.components.T) / self.scales
        return self.code_from_content(content, role=role, source=source)

    def code_from_content(
        self,
        content: Sequence[float] | np.ndarray,
        *,
        role: str,
        source: str,
    ) -> TurnCode:
        values = np.asarray(content, dtype=np.float64)
        if values.shape != (self.content_dimensions,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("content code has the wrong shape or non-finite values")
        if role not in ROLE_LABELS:
            raise ValueError("unknown role binding")
        dense = np.zeros(self.dense_dimensions, dtype=np.float64)
        dense[: self.content_dimensions] = values
        role_start = self.content_dimensions
        dense[role_start + ROLE_LABELS.index(role)] = 1.0
        source_start = role_start + len(ROLE_LABELS)
        source_index = (
            self.source_labels.index(source)
            if source in self.source_labels
            else len(self.source_labels)
        )
        dense[source_start + source_index] = 1.0
        dense.flags.writeable = False
        return TurnCode(dense, self.content_dimensions, role, source)

    def quantize(self, code: TurnCode) -> tuple[int, ...]:
        """Convert the dense code to fixed-width sparse persistent activity."""

        if code.content_dimensions != self.content_dimensions:
            raise ValueError("turn code belongs to a different codec")
        magnitudes = np.abs(code.content)
        order = np.lexsort((np.arange(len(magnitudes)), -magnitudes))
        selected = order[: self.active_content_features]
        features: list[int] = []
        for index in selected:
            value = float(code.content[index])
            magnitude_bin = int(
                np.searchsorted(
                    self.magnitude_thresholds[index], abs(value), side="right"
                )
            )
            features.append(
                _feature_id("content", int(index), int(value < 0.0), magnitude_bin)
            )
        features.append(_feature_id("role", code.role))
        features.append(_feature_id("source", code.source))
        if len(set(features)) != len(features):
            raise AssertionError("sparse code feature IDs collided")
        return tuple(features)


@dataclass(frozen=True)
class RidgeCodePredictor:
    """One-shot causal feature-to-next-turn-code ridge map; not recurrent."""

    feature_mean: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        alpha: float,
    ) -> RidgeCodePredictor:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("ridge features and targets must be aligned matrices")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("ridge inputs must be finite")
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("ridge alpha must be finite and positive")
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y, axis=0)
        centered_x = x - x_mean
        centered_y = y - y_mean
        gram = centered_x.T @ centered_x
        gram.flat[:: len(gram) + 1] += alpha
        coefficients = np.linalg.solve(gram, centered_x.T @ centered_y)
        for array in (x_mean, y_mean, coefficients):
            array.flags.writeable = False
        return cls(x_mean, y_mean, coefficients, float(alpha))

    def predict(self, features: Sequence[float] | np.ndarray) -> np.ndarray:
        row = np.asarray(features, dtype=np.float64)
        if row.shape != self.feature_mean.shape or not np.all(np.isfinite(row)):
            raise ValueError("ridge feature vector has the wrong shape")
        return np.asarray(
            (row - self.feature_mean) @ self.coefficients + self.target_mean
        )


@dataclass(frozen=True)
class DecoderBank:
    local: ContextCoupledByteWriter
    additive: ContextCoupledByteWriter
    coupled: ContextCoupledByteWriter

    def items(self) -> tuple[tuple[str, ContextCoupledByteWriter], ...]:
        return (
            ("local", self.local),
            ("additive", self.additive),
            ("coupled", self.coupled),
        )


@dataclass(frozen=True)
class RouteAssignment:
    writer: ContextCoupledByteWriter | None
    state: tuple[int, ...]
    gain: float


@dataclass(frozen=True)
class GeneratedBranch:
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


def implementation_hashes() -> dict[str, dict[str, Any]]:
    """Hash this assay, its test, and every directly imported local mechanism."""

    repository = Path(__file__).resolve().parents[3]
    records: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return records


def _feature_id(namespace: str, *parts: object) -> int:
    payload = "|".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"alm-ft-code").digest()
    return int.from_bytes(digest, "little")


def _hashed_byte_features(payload: bytes, dimensions: int) -> np.ndarray:
    """Stable signed byte n-gram sketch used only by this byte codec adapter."""

    if dimensions < 16:
        raise ValueError("feature dimensions must be at least 16")
    values = np.frombuffer(payload or b"\x00", dtype=np.uint8).astype(np.uint64)
    vector = np.zeros(dimensions, dtype=np.float64)
    mask = np.uint64((1 << 64) - 1)
    for width in (1, 2, 3, 4):
        if len(values) < width:
            continue
        count = len(values) - width + 1
        codes = np.full(count, np.uint64(1_000_003 * width), dtype=np.uint64)
        with np.errstate(over="ignore"):
            for offset in range(width):
                codes = (
                    codes * np.uint64(257) + values[offset : offset + count] + 1
                ) & mask
            # SplitMix64 avalanche supplies both a bucket and an independent
            # sign even for short byte n-grams whose polynomial is small.
            codes ^= codes >> np.uint64(30)
            codes *= np.uint64(0xBF58476D1CE4E5B9)
            codes ^= codes >> np.uint64(27)
            codes *= np.uint64(0x94D049BB133111EB)
            codes ^= codes >> np.uint64(31)
        indexes = np.asarray(codes % np.uint64(dimensions), dtype=np.int64)
        signs = np.where((codes >> np.uint64(63)) != 0, 1.0, -1.0)
        vector += np.bincount(indexes, weights=signs, minlength=dimensions)
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _binding_one_hot(
    role: str, source: str, source_labels: Sequence[str]
) -> np.ndarray:
    if role not in ROLE_LABELS:
        raise ValueError("unknown causal role binding")
    labels = tuple(source_labels)
    result = np.zeros(len(ROLE_LABELS) + len(labels) + 1, dtype=np.float64)
    result[ROLE_LABELS.index(role)] = 1.0
    source_offset = len(ROLE_LABELS)
    source_index = labels.index(source) if source in labels else len(labels)
    result[source_offset + source_index] = 1.0
    return result


def causal_turn_features(
    history: bytes,
    current_turn: bytes,
    *,
    role: str,
    source: str,
    source_labels: Sequence[str],
    hash_dimensions: int,
) -> np.ndarray:
    """Causal history/current-turn input for the non-recurrent baseline."""

    if hash_dimensions < 16:
        raise ValueError("causal hash dimensions must be at least 16")
    binding = _binding_one_hot(role, source, source_labels)
    lengths = np.asarray(
        [math.log1p(len(history)) / 16.0, math.log1p(len(current_turn)) / 16.0],
        dtype=np.float64,
    )
    return np.concatenate(
        (
            _hashed_byte_features(history, hash_dimensions),
            _hashed_byte_features(current_turn, hash_dimensions),
            binding,
            lengths,
        )
    )


def last_user_turn_from_context(context: bytes) -> bytes:
    """Return the immediate USER content from a generation-ready transcript.

    Schema-4 chat serialization places real turn markers at the start of the
    payload or after exactly two newlines.  Requiring that framing avoids
    treating a single-newline ``USER:`` quotation inside natural turn content
    as a new event.  A generation context must end in an empty ASSISTANT slot,
    immediately preceded by a nonempty USER turn.
    """

    if not isinstance(context, bytes) or not context:
        raise ValueError("generation context must be nonempty bytes")
    matches = list(_TOP_LEVEL_TURN_PATTERN.finditer(context))
    if not matches or matches[0].start() != 0:
        raise ValueError("generation context must begin with a top-level role marker")
    roles = [match.group(1) for match in matches]
    if roles[0] != USER_ROLE.encode() or any(
        roles[index] == roles[index - 1] for index in range(1, len(roles))
    ):
        raise ValueError("top-level USER and ASSISTANT turns must alternate")
    if len(matches) < 2 or roles[-2:] != [USER_ROLE.encode(), ASSISTANT_ROLE.encode()]:
        raise ValueError(
            "generation context must end with USER followed by an ASSISTANT slot"
        )
    if context[matches[-1].end() :].strip():
        raise ValueError("final ASSISTANT generation slot must be empty")
    current = context[matches[-2].end() : matches[-1].start()]
    if not current.strip():
        raise ValueError("immediate USER turn must be nonempty")
    return current


def _schema4_manifest(dataset_dir: Path) -> Mapping[str, Any]:
    path = dataset_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid dataset manifest: {error}") from error
    if int(manifest.get("schema_version", -1)) != 4:
        raise ValueError("future-turn codec requires a schema-4 dataset")
    return manifest


def load_training_conversations(
    dataset_dir: Path,
    *,
    maximum: int = 0,
) -> list[TrainingConversation]:
    """Extract every completed assistant turn from training conversations."""

    if maximum < 0:
        raise ValueError("maximum training conversations must be non-negative")
    blob = (dataset_dir / "train.bin").read_bytes()
    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    conversations: list[TrainingConversation] = []
    previous_end = 0
    for row in rows:
        start = int(row.get("start", -1))
        end = int(row.get("end", -1))
        if start != previous_end or not start < end <= len(blob):
            raise ValueError("training rows must contiguously cover train.bin")
        previous_end = end
        if row.get("assigned_split") != "train":
            raise ValueError("training index contains a non-training row")
        episode = blob[start:end]
        if _sha256(episode) != str(row.get("sha256", "")):
            raise ValueError("training conversation checksum mismatch")
        episode_id = str(row.get("episode_id", ""))
        source = str(row.get("source_label", ""))
        raw_turns = row.get("turns")
        if not episode_id or not source or not isinstance(raw_turns, list):
            raise ValueError("training conversation metadata is incomplete")
        turns = sorted(raw_turns, key=lambda item: int(item.get("index", -1)))
        assistants: list[AssistantTurn] = []
        previous_index = -1
        for position, turn in enumerate(turns):
            index = int(turn.get("index", -1))
            role = str(turn.get("role", ""))
            content_start = int(turn.get("content_start", -1))
            content_end = int(turn.get("end", -1))
            if index != previous_index + 1:
                raise ValueError("training turn indexes are not contiguous")
            previous_index = index
            if role not in ROLE_LABELS or not (
                start <= content_start <= content_end <= end
            ):
                raise ValueError("training turn role/range is invalid")
            if role != ASSISTANT_ROLE:
                continue
            if content_start == content_end:
                raise ValueError("completed assistant turn is empty")
            current_user = b""
            if position and str(turns[position - 1].get("role")) == USER_ROLE:
                user_start = int(turns[position - 1].get("content_start", -1))
                user_end = int(turns[position - 1].get("end", -1))
                if not start <= user_start <= user_end <= content_start:
                    raise ValueError("causal current-user range is invalid")
                current_user = blob[user_start:user_end]
            local_start = content_start - start
            local_end = content_end - start
            assistants.append(
                AssistantTurn(
                    episode_id=episode_id,
                    source_label=source,
                    turn_index=index,
                    prompt=episode[:local_start],
                    current_user_turn=current_user,
                    content=episode[local_start:local_end],
                    content_start=local_start,
                    content_end=local_end,
                )
            )
        if not assistants:
            raise ValueError("training conversation has no completed assistant turn")
        conversations.append(
            TrainingConversation(episode_id, source, episode, tuple(assistants))
        )
    if previous_end != len(blob):
        raise ValueError("training index does not cover train.bin")
    if maximum:
        conversations = conversations[:maximum]
    if len(conversations) < 2:
        raise ValueError("at least two training conversations are required")
    return conversations


def _request_from_context(context: bytes, metadata: Mapping[str, Any]) -> bytes:
    start = int(metadata.get("target_user_content_start", -1))
    end = int(metadata.get("target_user_content_end", -1))
    unanswered = int(metadata.get("unanswered_prefix_bytes", -1))
    if not 0 <= start < end <= unanswered <= len(context):
        raise ValueError("development target request range is invalid")
    return context[start:end]


def development_partition(pair_id: str) -> str:
    """Order-independent fixed development selection/evaluation assignment."""

    digest = hashlib.sha256(("future-turn-codec-v1|" + pair_id).encode()).digest()
    return "selection" if digest[0] & 1 == 0 else "evaluation"


def _original_context(
    blob: bytes,
    row: Mapping[str, Any],
    side: str,
) -> bytes | None:
    audit = row.get("audit_subranges")
    if not isinstance(audit, Mapping):
        return None
    record = audit.get(f"original_prefix_{side}")
    if not isinstance(record, Mapping):
        return None
    prefix = _slice(blob, record)
    if prefix.endswith(ASSISTANT_PROMPT):
        return prefix
    return prefix + ASSISTANT_PROMPT


def load_development_pairs(
    dataset_dir: Path,
    *,
    maximum: int = 0,
    excluded_pair_ids: Sequence[str] = KNOWN_SMOKE_PAIR_IDS,
) -> list[DevelopmentPair]:
    """Load declared development ranges, including full oracle turns."""

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
            raise ValueError("development sidecar is not schema-4 development data")
        invariants = row.get("invariants")
        if not isinstance(invariants, Mapping) or not all(
            bool(value) for value in invariants.values()
        ):
            raise ValueError("development construction invariant is false")
        ranges = row.get("blob_ranges")
        metadata = row.get("episodes")
        if not isinstance(ranges, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("development pair metadata is incomplete")
        contexts = {
            side: _slice(blob, ranges[f"context_{side}"]) for side in ("a", "b")
        }
        common = int(row.get("common_suffix", {}).get("bytes", -1))
        if common < 1 or contexts["a"][-common:] != contexts["b"][-common:]:
            raise ValueError("development common suffix is invalid")
        sides: dict[str, DevelopmentSide] = {}
        for side in ("a", "b"):
            response = _slice(blob, ranges[f"response_{side}"])
            full_record = ranges.get(f"full_response_{side}")
            full_response = (
                _slice(blob, full_record)
                if isinstance(full_record, Mapping)
                else response
            )
            sides[side] = DevelopmentSide(
                interrupted_context=contexts[side],
                original_context=_original_context(blob, row, side),
                request=_request_from_context(contexts[side], metadata[side]),
                response_window=response,
                full_response=full_response,
            )
        pair_id = str(row.get("pair_id", ""))
        source = str(row.get("source_label", ""))
        if not pair_id or not source:
            raise ValueError("development pair identity is empty")
        pairs.append(
            DevelopmentPair(
                pair_id,
                source,
                development_partition(pair_id),
                sides["a"],
                sides["b"],
            )
        )
    available = {pair.pair_id for pair in pairs}
    missing = sorted(set(excluded) - available)
    if missing:
        raise ValueError("excluded development pair ID was not found")
    excluded_set = set(excluded)
    pairs = [pair for pair in pairs if pair.pair_id not in excluded_set]
    if maximum:
        pairs = pairs[:maximum]
    if not pairs:
        raise ValueError("development pair payload is empty")
    return pairs


def make_decoder_bank(
    *,
    persistent_feature_count: int,
    table_bits: int,
    learning_rate: float,
    local_orders: Sequence[int],
) -> DecoderBank:
    if persistent_feature_count < 1:
        raise ValueError("persistent feature count must be positive")
    shared = dict(
        alphabet_size=256,
        table_bits=table_bits,
        learning_rate=learning_rate,
        local_orders=tuple(int(value) for value in local_orders),
        max_persistent_features=persistent_feature_count,
    )
    return DecoderBank(
        local=ContextCoupledByteWriter("local", **shared),
        additive=ContextCoupledByteWriter("additive_wide", **shared),
        coupled=ContextCoupledByteWriter("coupled", **shared),
    )


def decoder_allocation_audit(bank: DecoderBank, state_width: int) -> dict[str, Any]:
    parameters = {name: writer.parameter_count for name, writer in bank.items()}
    allocated = {name: writer.allocated_bytes for name, writer in bank.items()}
    active = {
        "local": bank.local.raw_active_feature_count(0),
        "additive": bank.additive.raw_active_feature_count(state_width),
        "coupled": bank.coupled.raw_active_feature_count(state_width),
    }
    return {
        "parameter_count_by_writer": parameters,
        "allocated_bytes_by_writer": allocated,
        "all_weight_tables_allocation_matched": len(set(parameters.values())) == 1
        and len(set(allocated.values())) == 1,
        "raw_active_features": active,
        "additive_and_coupled_active_count_matched": active["additive"]
        == active["coupled"],
        "additive_has_no_prefix_by_latent_cross_term": True,
    }


def _normalized_native_distribution(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (256,)
        or not np.all(np.isfinite(result))
        or np.any(result < 0.0)
    ):
        raise ValueError(
            "native distribution must be finite, non-negative, and 256-way"
        )
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("native distribution has no mass")
    return result / total


def _prime(cortex: FrozenCortex, context: bytes) -> None:
    cortex.reset_episode()
    if context:
        cortex.observe_chunk_probabilities(
            np.frombuffer(context, dtype=np.uint8), learn=False
        )


def _advance_chunk(cortex: FrozenCortex, payload: bytes) -> None:
    if payload:
        cortex.observe_chunk_probabilities(
            np.frombuffer(payload, dtype=np.uint8), learn=False
        )


def fit_decoder_bank(
    cortex: FrozenCortex,
    conversations: Sequence[TrainingConversation],
    codec: FutureTurnCodec,
    bank: DecoderBank,
    *,
    response_cap: int,
    training_seed: int,
) -> dict[str, Any]:
    """One chronological pass per conversation with frozen native base laws."""

    if response_cap < 1:
        raise ValueError("response cap must be positive")
    totals: dict[str, dict[str, float]] = {
        name: {
            "episodes": 0.0,
            "symbols": 0.0,
            "base_nll_nats": 0.0,
            "decoder_nll_nats": 0.0,
            "maximum_absolute_update": 0.0,
        }
        for name, _ in bank.items()
    }
    order = np.random.default_rng(training_seed).permutation(len(conversations))
    sequence_digest = hashlib.sha256()
    started = time.perf_counter()
    for conversation_index in order:
        conversation = conversations[int(conversation_index)]
        cortex.reset_episode()
        cursor = 0
        for turn in conversation.assistant_turns:
            if turn.content_start < cursor:
                raise ValueError("assistant training windows overlap")
            _advance_chunk(cortex, conversation.payload[cursor : turn.content_start])
            cursor = turn.content_start
            target = turn.content[:response_cap]
            bases = np.empty((len(target), 256), dtype=np.float64)
            for position, symbol in enumerate(target):
                bases[position] = _normalized_native_distribution(
                    cortex.predict_next_distribution()
                )
                cortex.advance_observation(symbol, learn=False)
                cursor += 1
            state = codec.quantize(
                codec.encode(
                    turn.content,
                    role=ASSISTANT_ROLE,
                    source=turn.source_label,
                )
            )
            sequence_digest.update(turn.episode_id.encode())
            sequence_digest.update(turn.turn_index.to_bytes(4, "little"))
            sequence_digest.update(target)
            for name, writer in bank.items():
                diagnostics = writer.fit_base_anchored_episode(
                    target,
                    bases,
                    () if name == "local" else state,
                    prefix=b"",
                )
                row = totals[name]
                row["episodes"] += 1
                row["symbols"] += diagnostics.symbols
                row["base_nll_nats"] += diagnostics.base_nll_nats
                row["decoder_nll_nats"] += diagnostics.anchored_nll_nats
                row["maximum_absolute_update"] = max(
                    row["maximum_absolute_update"],
                    diagnostics.maximum_absolute_update,
                )
        cortex.reset_episode()
    for row in totals.values():
        symbols = max(row["symbols"], 1.0)
        row["mean_base_nll_nats"] = row["base_nll_nats"] / symbols
        row["mean_decoder_nll_nats"] = row["decoder_nll_nats"] / symbols
        row["mean_residual_improvement_nats"] = (
            row["base_nll_nats"] - row["decoder_nll_nats"]
        ) / symbols
        row["episodes"] = int(row["episodes"])
        row["symbols"] = int(row["symbols"])
    episode_counts = {int(row["episodes"]) for row in totals.values()}
    symbol_counts = {int(row["symbols"]) for row in totals.values()}
    if len(episode_counts) != 1 or len(symbol_counts) != 1:
        raise AssertionError("decoder control training allocations diverged")
    return {
        "passes": 1,
        "conversation_order_sha256": _sha256(order.tobytes()),
        "observed_response_sequence_sha256": sequence_digest.hexdigest(),
        "writers": totals,
        "matched_episode_and_symbol_updates": True,
        "wall_seconds": time.perf_counter() - started,
    }


def fit_vector_conditioned_base(
    cortex: FrozenCortex,
    conversations: Sequence[TrainingConversation],
    codec: FutureTurnCodec,
    *,
    response_cap: int,
    training_seed: int,
) -> dict[str, Any]:
    """Train the base directly on ``dense target vector -> observed bytes``.

    Unlike :func:`fit_decoder_bank`, this path never feeds conversation bytes
    into the base and never constructs or updates a residual receiver.  During
    fitting the completed target turn supplies the oracle dense vector.  At
    deployment the predictor supplies a vector with the identical interface.
    Only the observed response bytes contribute surprise and base plasticity.
    """

    if response_cap < 1:
        raise ValueError("response cap must be positive")
    if not conversations:
        raise ValueError("training conversations must be nonempty")
    order = np.random.default_rng(training_seed).permutation(len(conversations))
    response_digest = hashlib.sha256()
    vector_digest = hashlib.sha256()
    episodes = 0
    symbols = 0
    surprise_bits = 0.0
    minimum_probability = 1.0
    conditioning_bytes = 0
    started = time.perf_counter()
    for raw_index in order:
        conversation = conversations[int(raw_index)]
        for turn in conversation.assistant_turns:
            target = turn.content[:response_cap]
            if not target:
                continue
            code = codec.encode(
                turn.content,
                role=ASSISTANT_ROLE,
                source=turn.source_label,
            )
            diagnostics: VectorConditionedFitDiagnostics = (
                fit_vector_conditioned_turn(cortex, code.vector, target)
            )
            episodes += 1
            symbols += diagnostics.target_bytes
            conditioning_bytes += diagnostics.conditioning_bytes
            surprise_bits += diagnostics.total_surprise_bits
            minimum_probability = min(
                minimum_probability, diagnostics.minimum_observed_probability
            )
            response_digest.update(turn.episode_id.encode("utf-8"))
            response_digest.update(turn.turn_index.to_bytes(4, "little"))
            response_digest.update(target)
            vector_digest.update(np.asarray(code.vector, dtype="<f4").tobytes())
    if episodes == 0:
        raise ValueError("training conversations contain no target bytes")
    return {
        "architecture": "dense_predictor_vector_to_base_to_bytes",
        "receiver_present": False,
        "oracle_target_vectors_during_fit": True,
        "predictor_vectors_at_inference": True,
        "episodes": episodes,
        "symbols": symbols,
        "conditioning_bytes": conditioning_bytes,
        "total_surprise_bits": surprise_bits,
        "mean_surprise_bits": surprise_bits / symbols,
        "minimum_observed_probability": minimum_probability,
        "conversation_order_sha256": _sha256(order.tobytes()),
        "response_sequence_sha256": response_digest.hexdigest(),
        "dense_vector_sequence_sha256": vector_digest.hexdigest(),
        "wall_seconds": time.perf_counter() - started,
    }


def generate_from_turn_vector(
    cortex: FrozenCortex,
    vector: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    uniform_draws: Sequence[float] | np.ndarray | None = None,
) -> VectorConditionedGeneration:
    """Generate natural-language bytes directly from a predictor vector."""

    return generate_from_vector(
        cortex,
        vector,
        horizon=horizon,
        uniform_draws=uniform_draws,
    )


def _route_distribution(
    native: np.ndarray,
    emitted_prefix: bytes | bytearray,
    assignment: RouteAssignment,
    *,
    residual_clip: float,
) -> np.ndarray:
    base = _normalized_native_distribution(native)
    if assignment.writer is None:
        return np.array(base, copy=True)
    return assignment.writer.base_anchored_distribution(
        base,
        emitted_prefix,
        assignment.state,
        gain=assignment.gain,
        residual_clip=residual_clip,
    )


def generate_branch(
    cortex: FrozenCortex,
    context: bytes,
    assignment: RouteAssignment,
    *,
    horizon: int,
    residual_clip: float,
    uniform_draws: np.ndarray | None,
) -> GeneratedBranch:
    """Generate actual bytes; the selected latent acts at every byte."""

    if horizon < 1:
        raise ValueError("generation horizon must be positive")
    draws = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if draws.shape != (horizon,) or np.any((draws < 0.0) | (draws >= 1.0)):
            raise ValueError("common random draws must have shape (horizon,) in [0,1)")
    _prime(cortex, context)
    generated = bytearray()
    digest = hashlib.sha256()
    for position in range(horizon):
        effective = _route_distribution(
            cortex.predict_next_distribution(),
            generated,
            assignment,
            residual_clip=residual_clip,
        )
        digest.update(np.asarray(effective, dtype=np.float64).tobytes())
        if draws is None:
            symbol = int(np.argmax(effective))
        else:
            symbol = min(
                int(
                    np.searchsorted(np.cumsum(effective), draws[position], side="right")
                ),
                255,
            )
        generated.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    return GeneratedBranch(bytes(generated), digest.hexdigest())


def _teacher_forced_native_bases(
    cortex: FrozenCortex, context: bytes, response: bytes
) -> np.ndarray:
    _prime(cortex, context)
    bases = np.empty((len(response), 256), dtype=np.float64)
    for position, symbol in enumerate(response):
        bases[position] = _normalized_native_distribution(
            cortex.predict_next_distribution()
        )
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    return bases


def teacher_forced_bpb_from_bases(
    bases: np.ndarray,
    response: bytes,
    assignment: RouteAssignment,
    *,
    residual_clip: float,
) -> float:
    """Secondary BPB diagnostic under the exact free-generation route."""

    matrix = np.asarray(bases, dtype=np.float64)
    if not response or matrix.shape != (len(response), 256):
        raise ValueError("teacher-forced bases do not align with response bytes")
    prefix = bytearray()
    bits = 0.0
    for position, symbol in enumerate(response):
        effective = _route_distribution(
            matrix[position], prefix, assignment, residual_clip=residual_clip
        )
        bits -= math.log2(max(float(effective[symbol]), 1e-300))
        prefix.append(symbol)
    return bits / len(response)


def sattolo_indices(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("a shuffled code control requires at least two sides")
    result = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for index in range(count - 1, 0, -1):
        other = int(rng.integers(0, index))
        result[index], result[other] = result[other], result[index]
    if np.any(result == np.arange(count)):
        raise AssertionError("Sattolo shuffle unexpectedly retained a code")
    return result


def cross_pair_shuffle_indices(pair_count: int, seed: int) -> np.ndarray:
    """Derange pair blocks while retaining side, distinct from the swap lesion."""

    if pair_count < 2:
        raise ValueError("cross-pair shuffled controls require at least two pairs")
    pair_order = sattolo_indices(pair_count, seed)
    result = np.asarray(
        [
            2 * int(pair_order[index]) + side
            for index in range(pair_count)
            for side in (0, 1)
        ],
        dtype=np.int64,
    )
    if np.any(result // 2 == np.arange(2 * pair_count) // 2):
        raise AssertionError("cross-pair shuffle retained an original pair")
    return result


def code_control_collision_audit(
    codes: Sequence[tuple[int, ...]], shuffle_indices: np.ndarray
) -> dict[str, Any]:
    """Report when a nominal wrong-code control has the same sparse code."""

    values = tuple(tuple(code) for code in codes)
    if not values or len(values) % 2:
        raise ValueError("code controls require two sides for every pair")
    shuffled = np.asarray(shuffle_indices, dtype=np.int64)
    if shuffled.shape != (len(values),) or np.any(
        (shuffled < 0) | (shuffled >= len(values))
    ):
        raise ValueError("shuffle indices do not align with code sides")
    paired_equal = sum(
        values[index] == values[index ^ 1] for index in range(0, len(values), 2)
    )
    shuffled_equal = sum(
        values[index] == values[int(other)]
        for index, other in enumerate(shuffled)
    )
    pair_count = len(values) // 2
    return {
        "paired_swap_pairs": pair_count,
        "paired_swap_identical_code_pairs": paired_equal,
        "paired_swap_distinct_code_fraction": 1.0 - paired_equal / pair_count,
        "cross_pair_shuffled_sides": len(values),
        "cross_pair_shuffle_identical_code_sides": shuffled_equal,
        "cross_pair_shuffle_distinct_code_fraction": (
            1.0 - shuffled_equal / len(values)
        ),
    }


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


def _prefix_accuracy(generated: bytes, reference: bytes) -> float:
    length = min(len(generated), len(reference))
    if length == 0:
        return 0.0
    return sum(generated[index] == reference[index] for index in range(length)) / length


def _first_divergence(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _mean_ci(values: Sequence[float], *, seed: int, resamples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[indexes], axis=1)
    return {
        "pairs": len(array),
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _seeded_draws(identifier: str, *, seed: int, horizon: int) -> np.ndarray:
    digest = hashlib.blake2b(
        identifier.encode(), digest_size=8, person=b"alm-ft-draw"
    ).digest()
    offset = int.from_bytes(digest, "little")
    return np.random.default_rng((seed + offset) % (1 << 64)).random(horizon)


def _assignments(
    bank: DecoderBank,
    *,
    oracle: tuple[int, ...],
    oracle_shuffled: tuple[int, ...],
    oracle_swapped: tuple[int, ...],
    predicted: tuple[int, ...],
    predicted_shuffled: tuple[int, ...],
    gain: float,
) -> dict[str, RouteAssignment]:
    result = {
        "native": RouteAssignment(None, (), 0.0),
        "oracle_code_gain0": RouteAssignment(bank.coupled, oracle, 0.0),
        "local_decoder": RouteAssignment(bank.local, (), gain),
        "oracle_code_coupled": RouteAssignment(bank.coupled, oracle, gain),
        "oracle_code_additive": RouteAssignment(bank.additive, oracle, gain),
        "oracle_code_shuffled": RouteAssignment(bank.coupled, oracle_shuffled, gain),
        "oracle_code_paired_swapped": RouteAssignment(
            bank.coupled, oracle_swapped, gain
        ),
        "predicted_code_coupled": RouteAssignment(bank.coupled, predicted, gain),
        "predicted_code_additive": RouteAssignment(bank.additive, predicted, gain),
        "predicted_code_shuffled": RouteAssignment(
            bank.coupled, predicted_shuffled, gain
        ),
    }
    if tuple(result) != CONDITION_NAMES:
        raise AssertionError("codec condition coverage changed")
    return result


def _summarize_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    *,
    partition: str,
    horizon: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    selected = {
        name: [
            row
            for row in arm_rows
            if partition == "all" or row["partition"] == partition
        ]
        for name, arm_rows in rows.items()
    }
    count = len(selected["native"])
    if count == 0:
        return {"pairs": 0, "arms": {}, "controlled_contrasts": {}}
    native = selected["native"]
    arms: dict[str, Any] = {}
    for condition_index, name in enumerate(CONDITION_NAMES):
        arm_rows = selected[name]
        greedy = [float(row["greedy_margin"]) for row in arm_rows]
        sampled = [float(row["sampled_margin"]) for row in arm_rows]
        greedy_advantage = [
            value - float(native[index]["greedy_margin"])
            for index, value in enumerate(greedy)
        ]
        sampled_advantage = [
            value - float(native[index]["sampled_margin"])
            for index, value in enumerate(sampled)
        ]
        greedy_accuracy = [
            float(
                np.mean(
                    [
                        row["sides"][side]["greedy_prefix_accuracy"]
                        for side in ("a", "b")
                    ]
                )
            )
            for row in arm_rows
        ]
        sampled_accuracy = [
            float(
                np.mean(
                    [
                        row["sides"][side]["sampled_prefix_accuracy"]
                        for side in ("a", "b")
                    ]
                )
            )
            for row in arm_rows
        ]
        divergence = []
        degeneration = []
        for index, row in enumerate(arm_rows):
            for side in ("a", "b"):
                current = bytes.fromhex(row["sides"][side]["greedy_hex"])
                baseline = bytes.fromhex(native[index]["sides"][side]["greedy_hex"])
                position = _first_divergence(current, baseline)
                divergence.append(horizon if position is None else position)
                degeneration.append(row["sides"][side]["sampled_diagnostics"])
        arms[name] = {
            "greedy_free_generation_margin": _mean_ci(
                greedy,
                seed=bootstrap_seed + condition_index * 31,
                resamples=bootstrap_resamples,
            ),
            "sampled_free_generation_margin": _mean_ci(
                sampled,
                seed=bootstrap_seed + condition_index * 31 + 1,
                resamples=bootstrap_resamples,
            ),
            "greedy_advantage_over_native": _mean_ci(
                greedy_advantage,
                seed=bootstrap_seed + condition_index * 31 + 2,
                resamples=bootstrap_resamples,
            ),
            "sampled_advantage_over_native": _mean_ci(
                sampled_advantage,
                seed=bootstrap_seed + condition_index * 31 + 3,
                resamples=bootstrap_resamples,
            ),
            "mean_greedy_reference_prefix_accuracy": float(np.mean(greedy_accuracy)),
            "mean_sampled_reference_prefix_accuracy": float(np.mean(sampled_accuracy)),
            "mean_teacher_forced_bpb_secondary": float(
                np.mean([row["mean_teacher_forced_bpb"] for row in arm_rows])
            ),
            "mean_first_greedy_divergence_from_native": float(np.mean(divergence)),
            "sampled_degeneration": {
                "mean_printable_fraction": float(
                    np.mean([row["printable_fraction"] for row in degeneration])
                ),
                "valid_utf8_fraction": float(
                    np.mean([row["valid_utf8"] for row in degeneration])
                ),
                "maximum_identical_byte_run": int(
                    max(row["maximum_identical_byte_run"] for row in degeneration)
                ),
                "outputs_with_run_at_least_16": int(
                    sum(row["maximum_identical_byte_run"] >= 16 for row in degeneration)
                ),
            },
        }
    contrasts: dict[str, Any] = {}
    for contrast_index, (label, (left_name, right_name)) in enumerate(
        CONTROLLED_CONTRASTS.items()
    ):
        left = selected[left_name]
        right = selected[right_name]
        contrasts[label] = {
            "greedy_free_generation_margin_difference": _mean_ci(
                [
                    float(a["greedy_margin"]) - float(b["greedy_margin"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_000 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
            "sampled_free_generation_margin_difference": _mean_ci(
                [
                    float(a["sampled_margin"]) - float(b["sampled_margin"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_001 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
            "teacher_forced_bpb_advantage_secondary": _mean_ci(
                [
                    float(b["mean_teacher_forced_bpb"])
                    - float(a["mean_teacher_forced_bpb"])
                    for a, b in zip(left, right, strict=True)
                ],
                seed=bootstrap_seed + 10_002 + contrast_index * 37,
                resamples=bootstrap_resamples,
            ),
        }
    return {"pairs": count, "arms": arms, "controlled_contrasts": contrasts}


def evaluate_codec(
    cortex: FrozenCortex,
    pairs: Sequence[DevelopmentPair],
    codec: FutureTurnCodec,
    predictor: RidgeCodePredictor,
    bank: DecoderBank,
    *,
    causal_hash_dimensions: int,
    gain: float,
    residual_clip: float,
    horizon: int,
    sample_seed: int,
    control_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Evaluate oracle codec and non-recurrent causal-code conditions on dev."""

    if len(pairs) < 2:
        raise ValueError("development evaluation requires at least two pairs")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("decoder gain must be finite and positive")
    before = {name: writer.fingerprint() for name, writer in bank.items()}
    flat_sides = [
        (pair, side_name, pair.side_a if side_name == "a" else pair.side_b)
        for pair in pairs
        for side_name in ("a", "b")
    ]
    oracle_codes = [
        codec.quantize(
            codec.encode(
                side.full_response,
                role=ASSISTANT_ROLE,
                source=pair.source_label,
            )
        )
        for pair, _, side in flat_sides
    ]
    shuffle = cross_pair_shuffle_indices(len(pairs), control_seed)
    modes: dict[str, dict[str, Any]] = {}
    for mode in ("original_unanswered_prefix", "long_interruption_crossover"):
        mode_available = all(
            side.original_context is not None for _, _, side in flat_sides
        )
        if mode == "original_unanswered_prefix" and not mode_available:
            modes[mode] = {
                "available": False,
                "reason": "original unanswered prefix metadata unavailable",
            }
            continue
        contexts = [
            (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            for _, _, side in flat_sides
        ]
        if any(context is None for context in contexts):
            raise AssertionError("available context mode contains a missing prefix")
        immediate_user_turns = [
            last_user_turn_from_context(context)  # type: ignore[arg-type]
            for context in contexts
        ]
        predicted_dense = [
            predictor.predict(
                causal_turn_features(
                    context,  # type: ignore[arg-type]
                    immediate_user_turn,
                    role=USER_ROLE,
                    source=pair.source_label,
                    source_labels=codec.source_labels,
                    hash_dimensions=causal_hash_dimensions,
                )
            )
            for context, immediate_user_turn, (pair, _, _) in zip(
                contexts, immediate_user_turns, flat_sides, strict=True
            )
        ]
        predicted_codes = [
            codec.quantize(
                codec.code_from_content(
                    dense,
                    role=ASSISTANT_ROLE,
                    source=pair.source_label,
                )
            )
            for dense, (pair, _, _) in zip(predicted_dense, flat_sides, strict=True)
        ]
        collision_audit = {
            "oracle_sparse_quantized_codes": code_control_collision_audit(
                oracle_codes, shuffle
            ),
            "predicted_sparse_quantized_codes": code_control_collision_audit(
                predicted_codes, shuffle
            ),
        }
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITION_NAMES}
        oracle_cosines = []
        for index, dense in enumerate(predicted_dense):
            oracle = codec.encode(
                flat_sides[index][2].full_response,
                role=ASSISTANT_ROLE,
                source=flat_sides[index][0].source_label,
            ).content
            denominator = float(np.linalg.norm(dense) * np.linalg.norm(oracle))
            oracle_cosines.append(
                0.0 if denominator == 0.0 else float(dense @ oracle / denominator)
            )
        for pair_index, pair in enumerate(pairs):
            pair_outputs: dict[str, dict[str, dict[str, Any]]] = {
                name: {} for name in CONDITION_NAMES
            }
            pair_bpbs: dict[str, list[float]] = {name: [] for name in CONDITION_NAMES}
            for side_offset, side_name in enumerate(("a", "b")):
                flat_index = pair_index * 2 + side_offset
                side = pair.side_a if side_name == "a" else pair.side_b
                context = contexts[flat_index]
                if context is None:
                    raise AssertionError("missing context reached generation")
                swapped_index = pair_index * 2 + (1 - side_offset)
                assignments = _assignments(
                    bank,
                    oracle=oracle_codes[flat_index],
                    oracle_shuffled=oracle_codes[int(shuffle[flat_index])],
                    oracle_swapped=oracle_codes[swapped_index],
                    predicted=predicted_codes[flat_index],
                    predicted_shuffled=predicted_codes[int(shuffle[flat_index])],
                    gain=gain,
                )
                draws = _seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=sample_seed,
                    horizon=horizon,
                )
                reference = side.response_window
                bases = _teacher_forced_native_bases(cortex, context, reference)
                for name, assignment in assignments.items():
                    greedy = generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=horizon,
                        residual_clip=residual_clip,
                        uniform_draws=None,
                    )
                    sampled = generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=horizon,
                        residual_clip=residual_clip,
                        uniform_draws=draws,
                    )
                    pair_bpbs[name].append(
                        teacher_forced_bpb_from_bases(
                            bases,
                            reference,
                            assignment,
                            residual_clip=residual_clip,
                        )
                    )
                    pair_outputs[name][side_name] = {
                        "greedy": greedy,
                        "sampled": sampled,
                        "reference": reference,
                    }
            for name in CONDITION_NAMES:
                outputs = pair_outputs[name]
                greedy_pair = _paired_reference_discrimination(
                    outputs["a"]["greedy"].data,
                    outputs["b"]["greedy"].data,
                    pair.side_a.response_window,
                    pair.side_b.response_window,
                    dimensions=1024,
                )
                sampled_pair = _paired_reference_discrimination(
                    outputs["a"]["sampled"].data,
                    outputs["b"]["sampled"].data,
                    pair.side_a.response_window,
                    pair.side_b.response_window,
                    dimensions=1024,
                )
                rows[name].append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "greedy_margin": greedy_pair["correct_minus_crossed_margin"],
                        "sampled_margin": sampled_pair["correct_minus_crossed_margin"],
                        "mean_teacher_forced_bpb": float(np.mean(pair_bpbs[name])),
                        "sides": {
                            side_name: {
                                "greedy_hex": outputs[side_name]["greedy"].data.hex(),
                                "greedy_text": outputs[side_name]["greedy"].data.decode(
                                    "utf-8", errors="backslashreplace"
                                ),
                                "sampled_hex": outputs[side_name]["sampled"].data.hex(),
                                "sampled_text": outputs[side_name][
                                    "sampled"
                                ].data.decode("utf-8", errors="backslashreplace"),
                                "greedy_probability_sha256": outputs[side_name][
                                    "greedy"
                                ].probability_sha256,
                                "sampled_probability_sha256": outputs[side_name][
                                    "sampled"
                                ].probability_sha256,
                                "greedy_prefix_accuracy": _prefix_accuracy(
                                    outputs[side_name]["greedy"].data,
                                    outputs[side_name]["reference"],
                                ),
                                "sampled_prefix_accuracy": _prefix_accuracy(
                                    outputs[side_name]["sampled"].data,
                                    outputs[side_name]["reference"],
                                ),
                                "greedy_diagnostics": _generation_diagnostics(
                                    outputs[side_name]["greedy"].data
                                ),
                                "sampled_diagnostics": _generation_diagnostics(
                                    outputs[side_name]["sampled"].data
                                ),
                            }
                            for side_name in ("a", "b")
                        },
                    }
                )
        for native_row, gain0_row in zip(
            rows["native"], rows["oracle_code_gain0"], strict=True
        ):
            if native_row != gain0_row:
                raise AssertionError("gain-zero oracle route is not exactly native")
        summaries = {
            partition: _summarize_rows(
                rows,
                partition=partition,
                horizon=horizon,
                bootstrap_seed=bootstrap_seed
                + (0 if mode == "original_unanswered_prefix" else 100_000),
                bootstrap_resamples=bootstrap_resamples,
            )
            for partition in ("selection", "evaluation", "all")
        }
        fixed_ids = {
            pair.pair_id for pair in sorted(pairs, key=lambda row: row.pair_id)[:5]
        }
        modes[mode] = {
            "available": True,
            "code_control_collision_audit": collision_audit,
            "predictor_code_diagnostic": {
                "mean_oracle_cosine": float(np.mean(oracle_cosines)),
                "median_oracle_cosine": float(np.median(oracle_cosines)),
                "not_a_recurrent_transition": True,
                "immediate_user_turns_parsed_from_generation_context": True,
                "immediate_user_turn_sha256": _sha256(
                    b"".join(
                        len(turn).to_bytes(8, "little") + turn
                        for turn in immediate_user_turns
                    )
                ),
                "evaluator_target_request_reinjected_as_current_turn": False,
            },
            "summaries": summaries,
            "pair_rows": rows,
            "fixed_samples": {
                name: [row for row in arm_rows if row["pair_id"] in fixed_ids]
                for name, arm_rows in rows.items()
            },
            "invariants": {
                "gain0_oracle_exactly_native": True,
                "global_shuffle_is_cross_pair_deranged": True,
                "paired_side_swap_is_a_separate_control": True,
                "common_random_temperature_one_draws_shared_across_arms": True,
                "actual_bytes_generated_not_candidate_reranked": True,
                "predictor_uses_immediate_user_turn_from_generation_context": True,
            },
        }
    after = {name: writer.fingerprint() for name, writer in bank.items()}
    if before != after:
        raise AssertionError("development generation modified a decoder writer")
    return {
        "context_modes": modes,
        "writer_fingerprints_unchanged_by_evaluation": True,
        "development_partition_counts": {
            name: sum(pair.partition == name for pair in pairs)
            for name in ("selection", "evaluation")
        },
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Fit on training conversations and evaluate development only."""

    started = time.perf_counter()
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    if args.wall_limit_seconds <= 0.0:
        raise ValueError("wall limit must be positive")
    if (args.max_train_conversations or args.max_dev_pairs) and not args.smoke:
        raise ValueError("data slicing is permitted only with --smoke")
    _schema4_manifest(args.dataset_dir)
    verified = receiver._verify_manifest_payloads(
        args.dataset_dir, args.dataset_dir / "manifest.json", "dev"
    )
    conversations = load_training_conversations(
        args.dataset_dir, maximum=args.max_train_conversations
    )
    pairs = load_development_pairs(
        args.dataset_dir,
        maximum=args.max_dev_pairs,
        excluded_pair_ids=args.exclude_dev_pair_ids,
    )
    turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = FutureTurnCodec.fit(
        turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    target_codes = np.asarray(
        [
            codec.encode(
                turn.content,
                role=ASSISTANT_ROLE,
                source=turn.source_label,
            ).content
            for turn in turns
        ]
    )
    causal_features = np.asarray(
        [
            causal_turn_features(
                turn.prompt,
                turn.current_user_turn,
                role=USER_ROLE,
                source=turn.source_label,
                source_labels=codec.source_labels,
                hash_dimensions=args.causal_hash_dimensions,
            )
            for turn in turns
        ]
    )
    predictor = RidgeCodePredictor.fit(
        causal_features, target_codes, alpha=args.ridge_alpha
    )
    train_prediction = (
        causal_features - predictor.feature_mean
    ) @ predictor.coefficients + predictor.target_mean
    predictor_rmse = float(np.sqrt(np.mean((train_prediction - target_codes) ** 2)))

    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    embodied_at_load = int(getattr(cortex, "embodied_bytes", -1))
    bank = make_decoder_bank(
        persistent_feature_count=codec.persistent_feature_count,
        table_bits=args.table_bits,
        learning_rate=args.learning_rate,
        local_orders=args.local_orders,
    )
    training = fit_decoder_bank(
        cortex,
        conversations,
        codec,
        bank,
        response_cap=args.response_cap,
        training_seed=args.training_seed,
    )
    evaluation = evaluate_codec(
        cortex,
        pairs,
        codec,
        predictor,
        bank,
        causal_hash_dimensions=args.causal_hash_dimensions,
        gain=args.gain,
        residual_clip=args.residual_clip,
        horizon=args.generation_horizon,
        sample_seed=args.sample_seed,
        control_seed=args.control_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after completed codec evaluation")
    return {
        "schema_version": 1,
        "experiment": "future_turn_codec_native500_development",
        "status": "complete_smoke" if args.smoke else "complete_development",
        "claim_scope": {
            "stage": "codec_before_forecasting_horizon",
            "oracle_contains_held_out_answer_content_and_is_leakage_ceiling": True,
            "causal_predictor_is_one_shot_ridge_not_recurrent": True,
            "causal_predictor_uses_immediate_user_turn_parsed_from_context": True,
            "no_evaluator_selected_old_request_is_reinjected_into_predictor": True,
            "no_multihorizon_or_progressive_refinement_claim": True,
            "no_goal_learning_or_consequence_value_claim": True,
            "dense_code_directly_controls_byte_generation_at_every_step": True,
            "candidate_reranking": False,
            "native_checkpoint_frozen": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_secondary_only": True,
            "byte_codec_chat_specific_latent_actuator_interface_generic": True,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "embodied_bytes_at_load": embodied_at_load,
            "transient_embodied_bytes_at_report": int(
                getattr(cortex, "embodied_bytes", -1)
            ),
        },
        "config": {
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "dense_dimensions_with_binding": codec.dense_dimensions,
            "active_content_features": args.active_content_features,
            "persistent_feature_count_with_binding": codec.persistent_feature_count,
            "causal_hash_dimensions_per_timescale": args.causal_hash_dimensions,
            "ridge_alpha": args.ridge_alpha,
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "decoder_training_passes": 1,
            "response_cap": args.response_cap,
            "generation_horizon": args.generation_horizon,
            "gain": args.gain,
            "residual_clip": args.residual_clip,
            "training_seed": args.training_seed,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "smoke": args.smoke,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "completed_training_assistant_turns": len(turns),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
            "decoder_target_bytes_one_pass": sum(
                min(len(turn.content), args.response_cap) for turn in turns
            ),
        },
        "codec": {
            "fit_inputs": "completed training assistant turns only",
            "pca_whitened": True,
            "role_labels": list(ROLE_LABELS),
            "source_labels_fit_on_training": list(codec.source_labels),
            "explicit_role_features_per_code": 1,
            "explicit_source_features_per_code": 1,
            "magnitude_bins": codec.magnitude_thresholds.shape[1] + 1,
            "dense_code_sha256": _sha256(target_codes.tobytes()),
            "sparse_code_sha256": _sha256(
                json.dumps(
                    [
                        codec.quantize(
                            codec.encode(
                                turn.content,
                                role=ASSISTANT_ROLE,
                                source=turn.source_label,
                            )
                        )
                        for turn in turns
                    ],
                    separators=(",", ":"),
                ).encode()
            ),
        },
        "causal_predictor": {
            "kind": "centered_ridge_history_plus_current_turn_to_next_assistant_code",
            "non_recurrent": True,
            "does_not_receive_target_response": True,
            "training_rmse_diagnostic": predictor_rmse,
            "feature_dimensions": int(causal_features.shape[1]),
            "target_dimensions": int(target_codes.shape[1]),
            "parameter_count": int(predictor.coefficients.size),
        },
        "decoder_allocation": decoder_allocation_audit(
            bank, codec.persistent_feature_count
        ),
        "decoder_training": training,
        "evaluation": evaluation,
        "wall_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-dimensions", type=int, default=512)
    parser.add_argument("--content-dimensions", type=int, default=64)
    parser.add_argument("--active-content-features", type=int, default=16)
    parser.add_argument("--causal-hash-dimensions", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=4.0)
    parser.add_argument("--table-bits", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.add_argument("--bootstrap-seed", type=int, default=260_933)
    parser.add_argument("--bootstrap-resamples", type=int, default=4_000)
    parser.add_argument("--wall-limit-seconds", type=float, default=7_200.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-conversations", type=int, default=0)
    parser.add_argument("--max-dev-pairs", type=int, default=0)
    parser.add_argument(
        "--exclude-dev-pair-ids", nargs="*", default=KNOWN_SMOKE_PAIR_IDS
    )
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
