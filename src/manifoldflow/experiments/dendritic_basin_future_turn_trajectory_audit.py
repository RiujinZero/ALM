"""Development-only natural-chat audit of recurrent future-turn trajectories.

This audit deliberately ends at the representation boundary.  A deterministic
text adapter maps each observed turn to a dense event vector while explicit
source and role IDs remain separate.  ``FutureTurnTrajectory`` is fitted on
every adjacent turn transition in complete training conversations.  At
development time it sees only each original, unanswered conversation prefix and
the same learned one-step transition is applied recursively to forecast the
next one through four turns.

The learned objective is one-step ridge regression.  Horizons two through four
are recursive ``F**k`` evaluations, not multi-step-trained predictors.  The
forecast is not a goal, utility, completion probability, language decoder, or
free-generation result.  Observed future turns enter only metrics and an
explicitly non-causal oracle projection ceiling.

The data firewall is intentionally narrow: this module opens exactly the four
locked schema-4 successor development inputs named in ``OPENED_DATA_FILES``.
There is no split selector and no protected-split path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_turn_trajectory import (
    EncodedTurn,
    FutureSlot,
    FutureTurnTrajectory,
)

OPENED_DATA_FILES = (
    "train.bin",
    "train.index.jsonl",
    "dev_pairs.bin",
    "dev_pairs.jsonl",
)
LOCKED_INPUT_SHA256 = {
    "train.bin": "ccb1751d350daee579a37bb9cb974e8b5cce446b958f3ccb7ea4e2ae6814c841",
    "train.index.jsonl": (
        "6efb255f9bd828108f39101afcfaa950fbfcd12b8ce37956ac0c71bc7ad05e5c"
    ),
    "dev_pairs.bin": (
        "789f3209957192594d4d9f2966f1c4040f11c826d4c862c67e129e1a397ca821"
    ),
    "dev_pairs.jsonl": (
        "01893cff764859bbf2d47053b0157f2fb490b34bbad35c1d56ff28e69c3c7681"
    ),
}
ENGINEERING_EXPOSED_PAIR_IDS = (
    "8c14b02a3307015f0eb398651cf74ec30e1368f84986c091fc97e114eb56fb31",
    "9bd6656828136ad076e4ea7e070be0b9f04fa723473a5b0ac7f6ab5e3eeefc3f",
)
DEFAULT_DATASET_DIR = Path(
    "runs/shared_goal/"
    "confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3"
)
DEFAULT_EVENT_DIMENSIONS = 96
DEFAULT_HORIZON_WIDTHS = (64, 32, 16, 8)
DEFAULT_MEMORY_DECAY = 0.8
DEFAULT_RIDGE = 1e-3
DEFAULT_SHUFFLE_SEED = 271_828
DEFAULT_BOOTSTRAP_SEED = 260_923
DEFAULT_BOOTSTRAP_RESAMPLES = 4_000

_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_TOP_LEVEL_TURN_PATTERN = re.compile(
    r"(?:\A|\n\n)(USER|ASSISTANT):[ \t]*"
)


@dataclass(frozen=True)
class RawTurn:
    role: str
    text: str


@dataclass(frozen=True)
class RawConversation:
    episode_id: str
    turns: tuple[RawTurn, ...]


@dataclass(frozen=True)
class RawSide:
    episode_id: str
    prefix: tuple[RawTurn, ...]
    future: tuple[RawTurn, ...]


@dataclass(frozen=True)
class RawPair:
    pair_id: str
    a: RawSide
    b: RawSide


@dataclass(frozen=True)
class LoadedDevelopmentData:
    train: tuple[RawConversation, ...]
    dev: tuple[RawPair, ...]
    locked_hashes: dict[str, str]
    total_dev_pairs: int
    excluded_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class ForecastRecord:
    pair_id: str
    side: str
    horizon: int
    vector: np.ndarray
    target: np.ndarray
    predicted_role: str | None
    target_role: str
    predicted_source: str | None
    target_source: str
    accumulated_residual_variance_heuristic: np.ndarray | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = payload.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {name}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} contains no rows")
    return rows


def _slice_checked(blob: bytes, record: Mapping[str, Any], *, label: str) -> bytes:
    start = int(record["start"])
    end = int(record["end"])
    if start < 0 or end < start or end > len(blob):
        raise ValueError(f"{label} lies outside its blob")
    payload = blob[start:end]
    if "bytes" in record and len(payload) != int(record["bytes"]):
        raise ValueError(f"{label} byte count mismatch")
    if "sha256" in record and _sha256(payload) != str(record["sha256"]):
        raise ValueError(f"{label} checksum mismatch")
    return payload


def _read_locked_inputs(
    dataset_dir: Path, locked_hashes: Mapping[str, str]
) -> tuple[dict[str, bytes], dict[str, str]]:
    if set(locked_hashes) != set(OPENED_DATA_FILES):
        raise ValueError("locked hashes must name exactly the four development files")
    payloads: dict[str, bytes] = {}
    actual: dict[str, str] = {}
    for name in OPENED_DATA_FILES:
        payload = (dataset_dir / name).read_bytes()
        digest = _sha256(payload)
        if digest != str(locked_hashes[name]):
            raise ValueError(f"locked checksum mismatch for {name}")
        payloads[name] = payload
        actual[name] = digest
    return payloads, actual


def _decode_raw_turn(role: Any, payload: bytes) -> RawTurn:
    normalized_role = str(role)
    if normalized_role not in {"USER", "ASSISTANT"}:
        raise ValueError(f"unsupported turn role: {normalized_role}")
    # Empty turns occur in the locked natural corpus and are legitimate observed
    # events. ``lexical_event_vector`` gives them an explicit stable empty code.
    text = payload.decode("utf-8", errors="replace").strip()
    return RawTurn(normalized_role, text)


def _load_training(blob: bytes, sidecar: bytes) -> tuple[RawConversation, ...]:
    conversations: list[RawConversation] = []
    seen_ids: set[str] = set()
    for row in _jsonl(sidecar, name="train.index.jsonl"):
        if row.get("assigned_split") != "train":
            raise ValueError("training sidecar contains a non-training row")
        if row.get("goal_target_kind") != "successor":
            raise ValueError("training sidecar is not a successor dataset")
        episode_id = str(row["episode_id"])
        if episode_id in seen_ids:
            raise ValueError("training episode IDs must be unique")
        seen_ids.add(episode_id)
        episode = _slice_checked(blob, row, label=f"training episode {episode_id}")
        if len(episode) != int(row["episode_bytes"]):
            raise ValueError("training episode byte count mismatch")

        start = int(row["start"])
        end = int(row["end"])
        turn_rows = tuple(row["turns"])
        if len(turn_rows) < 2:
            raise ValueError("every training conversation needs an adjacent turn")
        turns: list[RawTurn] = []
        previous_end = start
        for expected_index, turn_row in enumerate(turn_rows):
            if int(turn_row["index"]) != expected_index:
                raise ValueError("training turn indexes must be contiguous")
            content_start = int(turn_row["content_start"])
            content_end = int(turn_row["end"])
            if content_start < previous_end or content_end < content_start:
                raise ValueError("training turn ranges are not ordered")
            if content_start < start or content_end > end:
                raise ValueError("training turn lies outside its full conversation")
            turns.append(
                _decode_raw_turn(
                    turn_row["role"], blob[content_start:content_end]
                )
            )
            previous_end = content_end
        if previous_end != end:
            raise ValueError("training sidecar does not expose the full conversation")
        if int(row["goal_target_turn_end"]) != len(turns) - 1:
            raise ValueError("training successor range does not reach conversation end")
        conversations.append(RawConversation(episode_id, tuple(turns)))
    return tuple(conversations)


def _turns_from_marked_payload(
    payload: bytes, *, label: str, expected_count: int
) -> tuple[RawTurn, ...]:
    """Parse only double-newline top-level markers from serialized chat bytes.

    Natural turn content can itself contain lines beginning with ``USER:`` or
    ``ASSISTANT:``. The schema-4 serializer separates actual turns with two
    newlines, so the more permissive transcript parser would create false turns.
    """

    text = payload.decode("utf-8", errors="replace").replace("\r\n", "\n")
    matches = list(_TOP_LEVEL_TURN_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"{label} contains no parsed turns")
    turns = tuple(
        RawTurn(
            match.group(1),
            text[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
            ].strip(),
        )
        for index, match in enumerate(matches)
    )
    if len(turns) != expected_count:
        raise ValueError(
            f"{label} parsed {len(turns)} top-level turns, expected {expected_count}"
        )
    return turns


def _load_side(
    blob: bytes, row: Mapping[str, Any], side: str, *, pair_id: str
) -> RawSide:
    metadata = row["episodes"][side]
    if metadata.get("assigned_split") != "dev":
        raise ValueError("development episode metadata has the wrong split")
    if metadata.get("goal_target_kind") != "successor":
        raise ValueError("development episode is not a successor example")

    prefix_record = row["audit_subranges"][f"original_prefix_{side}"]
    prefix_payload = _slice_checked(
        blob, prefix_record, label=f"{pair_id} original prefix {side}"
    )
    if len(prefix_payload) != int(metadata["unanswered_prefix_bytes"]):
        raise ValueError("original prefix length disagrees with episode metadata")
    if _sha256(prefix_payload) != str(metadata["unanswered_prefix_sha256"]):
        raise ValueError("original prefix hash disagrees with episode metadata")
    prefix = _turns_from_marked_payload(
        prefix_payload,
        label=f"{pair_id} original prefix {side}",
        expected_count=int(metadata["target_user_turn_index"]) + 1,
    )
    if prefix[-1].role != "USER":
        raise ValueError("original development prefix must end in USER")

    ranges = row["blob_ranges"]
    response_payload = _slice_checked(
        blob, ranges[f"full_response_{side}"], label=f"{pair_id} response {side}"
    )
    if _sha256(response_payload) != str(metadata["target_full_response_sha256"]):
        raise ValueError("response hash disagrees with episode metadata")
    response = _decode_raw_turn("ASSISTANT", response_payload)

    goal_payload = _slice_checked(
        blob, ranges[f"goal_target_{side}"], label=f"{pair_id} successor {side}"
    )
    if _sha256(goal_payload) != str(metadata["goal_target_sha256"]):
        raise ValueError("successor hash disagrees with episode metadata")
    expected_successor_count = (
        int(metadata["goal_target_turn_end"])
        - int(metadata["goal_target_turn_start"])
        + 1
    )
    successor = _turns_from_marked_payload(
        goal_payload,
        label=f"{pair_id} successor {side}",
        expected_count=expected_successor_count,
    )
    if len(successor) < 2:
        raise ValueError("schema-4 successor examples need at least two later turns")
    return RawSide(
        episode_id=str(metadata["episode_id"]),
        prefix=prefix,
        future=(response, *successor[:3]),
    )


def _load_development(
    blob: bytes, sidecar: bytes
) -> tuple[tuple[RawPair, ...], int, tuple[str, ...]]:
    kept: list[RawPair] = []
    excluded: list[str] = []
    seen: set[str] = set()
    rows = _jsonl(sidecar, name="dev_pairs.jsonl")
    for row in rows:
        if int(row.get("schema_version", -1)) != 4:
            raise ValueError("development audit requires schema version 4")
        if row.get("split") != "dev":
            raise ValueError("development sidecar contains a non-development pair")
        if row.get("goal_target_kind") != "successor":
            raise ValueError("development sidecar is not a successor dataset")
        invariants = dict(row.get("invariants", {}))
        if not invariants or not all(bool(value) for value in invariants.values()):
            raise ValueError("development pair has a false construction invariant")
        pair_id = str(row["pair_id"])
        if pair_id in seen:
            raise ValueError("development pair IDs must be unique")
        seen.add(pair_id)
        side_a = _load_side(blob, row, "a", pair_id=pair_id)
        side_b = _load_side(blob, row, "b", pair_id=pair_id)
        if pair_id in ENGINEERING_EXPOSED_PAIR_IDS:
            excluded.append(pair_id)
            continue
        kept.append(RawPair(pair_id, side_a, side_b))

    if set(excluded) != set(ENGINEERING_EXPOSED_PAIR_IDS):
        raise ValueError("both engineering-exposed development pairs must be present")
    if not kept:
        raise ValueError("no unexposed development pairs remain")
    return tuple(kept), len(rows), tuple(excluded)


def load_development_data(
    dataset_dir: Path,
    *,
    locked_hashes: Mapping[str, str] = LOCKED_INPUT_SHA256,
) -> LoadedDevelopmentData:
    """Load only the locked training and development inputs."""

    payloads, actual = _read_locked_inputs(dataset_dir, locked_hashes)
    train = _load_training(payloads["train.bin"], payloads["train.index.jsonl"])
    dev, total, excluded = _load_development(
        payloads["dev_pairs.bin"], payloads["dev_pairs.jsonl"]
    )
    return LoadedDevelopmentData(train, dev, actual, total, excluded)


def lexical_event_vector(text: str, dimensions: int = 96) -> np.ndarray:
    """Deterministic Unicode lexical event vector for the text adapter only."""

    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise TypeError("dimensions must be an integer")
    if dimensions < 8:
        raise ValueError("dimensions must be at least eight")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = _WORD_PATTERN.findall(normalized)
    features = [f"u:{token}" for token in tokens]
    features.extend(
        f"b:{left}\x1f{right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
    )
    if not features:
        characters = [char for char in normalized if not char.isspace()]
        features = [
            "c:" + "".join(characters[index : index + 3])
            for index in range(max(1, len(characters) - 2))
        ]
    if not features:
        features = ["empty"]

    vector = np.zeros(dimensions, dtype=np.float64)
    for feature in features:
        digest = hashlib.blake2s(
            feature.encode("utf-8"), digest_size=8, person=b"ft-turn1"
        ).digest()
        code = int.from_bytes(digest, "little")
        vector[code % dimensions] += 1.0 if code >> 63 else -1.0
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _encode_turn(turn: RawTurn, dimensions: int) -> EncodedTurn:
    return EncodedTurn(
        lexical_event_vector(turn.text, dimensions),
        source_id="text",
        role_id=turn.role,
    )


def _encode_sequence(
    turns: Sequence[RawTurn], dimensions: int
) -> tuple[EncodedTurn, ...]:
    return tuple(_encode_turn(turn, dimensions) for turn in turns)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def _pair_margin_map(records: Sequence[ForecastRecord]) -> dict[str, float]:
    by_key: dict[tuple[str, str], ForecastRecord] = {}
    for record in records:
        key = (record.pair_id, record.side)
        if key in by_key:
            raise ValueError("duplicate pair side in forecast records")
        by_key[key] = record
    margins: dict[str, float] = {}
    for pair_id in sorted({row.pair_id for row in records}):
        left = by_key.get((pair_id, "a"))
        right = by_key.get((pair_id, "b"))
        if left is None or right is None:
            continue
        margins[pair_id] = (
            _cosine(left.vector, left.target)
            + _cosine(right.vector, right.target)
            - _cosine(left.vector, right.target)
            - _cosine(right.vector, left.target)
        )
    return margins


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not len(samples):
        raise ValueError("bootstrap requires a nonempty one-dimensional sample")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("bootstrap seed must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int):
        raise TypeError("bootstrap resamples must be an integer")
    if resamples < 4_000:
        raise ValueError("paired inference requires at least 4000 bootstrap resamples")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(samples), size=(resamples, len(samples)))
    means = np.mean(samples[indexes], axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _paired_margin_difference(
    left: Sequence[ForecastRecord],
    right: Sequence[ForecastRecord],
    *,
    seed: int,
    resamples: int,
    contrast: str,
    require_same_support: bool = True,
) -> dict[str, Any]:
    left_margins = _pair_margin_map(left)
    right_margins = _pair_margin_map(right)
    pair_ids = sorted(set(left_margins) & set(right_margins))
    if not pair_ids:
        raise ValueError("paired contrast contains no complete shared pairs")
    if require_same_support and (
        set(pair_ids) != set(left_margins) or set(pair_ids) != set(right_margins)
    ):
        raise ValueError("paired contrast conditions have different pair support")
    differences = np.asarray(
        [left_margins[pair_id] - right_margins[pair_id] for pair_id in pair_ids],
        dtype=np.float64,
    )
    lower, upper = _bootstrap_mean_interval(
        differences, seed=seed, resamples=resamples
    )
    return {
        "contrast": contrast,
        "pair_count": len(pair_ids),
        "per_pair_margin_difference": [
            {"pair_id": pair_id, "difference": float(difference)}
            for pair_id, difference in zip(pair_ids, differences, strict=True)
        ],
        "mean_pair_margin_difference": float(np.mean(differences)),
        "ci95_pair_bootstrap": [lower, upper],
        "positive_pair_fraction": float(np.mean(differences > 0.0)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def _records_metrics(records: Sequence[ForecastRecord]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    predictions = np.stack([record.vector for record in records])
    targets = np.stack([record.target for record in records])
    cosines = np.asarray(
        [_cosine(left, right) for left, right in zip(predictions, targets, strict=True)]
    )
    squared = (predictions - targets) ** 2

    target_norms = np.linalg.norm(targets, axis=1)
    prediction_norms = np.linalg.norm(predictions, axis=1)
    denominator = np.maximum(
        prediction_norms[:, None] * target_norms[None, :], 1e-12
    )
    retrieval_scores = predictions @ targets.T / denominator
    ranks: list[float] = []
    for index, row in enumerate(retrieval_scores):
        own = row[index]
        greater = int(np.sum(row > own + 1e-12))
        tied_other = int(np.sum(np.abs(row - own) <= 1e-12)) - 1
        ranks.append(1.0 + greater + 0.5 * max(tied_other, 0))

    margins = _pair_margin_map(records)
    margin_array = np.asarray(list(margins.values()), dtype=np.float64)
    rank_array = np.asarray(ranks, dtype=np.float64)
    result: dict[str, Any] = {
        "count": len(records),
        "cosine": {
            "mean": float(np.mean(cosines)),
            "median": float(np.median(cosines)),
        },
        "mse": {
            "mean_per_coordinate": float(np.mean(squared)),
            "mean_vector_squared_error": float(np.mean(np.sum(squared, axis=1))),
        },
        "retrieval_rank": {
            "candidate_count": len(records),
            "mean": float(np.mean(rank_array)),
            "median": float(np.median(rank_array)),
            "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
            "top1_fraction": float(np.mean(rank_array <= 1.0)),
        },
        "pair_discrimination": {
            "pair_count": len(margin_array),
            "per_pair_matched_minus_crossed_cosine": [
                {"pair_id": pair_id, "margin": margin}
                for pair_id, margin in margins.items()
            ],
            "mean_matched_minus_crossed_cosine": (
                float(np.mean(margin_array)) if len(margin_array) else None
            ),
            "positive_fraction": (
                float(np.mean(margin_array > 0.0)) if len(margin_array) else None
            ),
            "tie_fraction": (
                float(np.mean(np.abs(margin_array) <= 1e-12))
                if len(margin_array)
                else None
            ),
        },
    }
    identity_records = [row for row in records if row.predicted_role is not None]
    if identity_records:
        result["explicit_identity_accuracy"] = {
            "role": float(
                np.mean(
                    [
                        row.predicted_role == row.target_role
                        for row in identity_records
                    ]
                )
            ),
            "source": float(
                np.mean(
                    [
                        row.predicted_source == row.target_source
                        for row in identity_records
                    ]
                )
            ),
        }
    return result


def _inferential_contrasts(
    records: Mapping[str, Mapping[int, Sequence[ForecastRecord]]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    control_names = (
        "mean_event_prior",
        "current_turn_only_reset_history",
        "shuffled_transition",
        "role_swapped_prefix",
        "reversed_horizon_order",
    )
    recurrent = records["recurrent_full_history"]
    versus_controls: dict[str, Any] = {}
    for control_index, control in enumerate(control_names):
        versus_controls[control] = {
            "by_horizon": {
                str(horizon): _paired_margin_difference(
                    recurrent[horizon],
                    records[control][horizon],
                    seed=bootstrap_seed + control_index * 100 + horizon,
                    resamples=bootstrap_resamples,
                    contrast=(
                        "recurrent_full_history minus "
                        f"{control} matched-minus-crossed cosine margin"
                    ),
                )
                for horizon in range(1, 5)
            }
        }

    degradation: dict[str, Any] = {}
    for horizon in range(2, 5):
        degradation[str(horizon)] = _paired_margin_difference(
            recurrent[horizon],
            recurrent[1],
            seed=bootstrap_seed + 10_000 + horizon,
            resamples=bootstrap_resamples,
            contrast=(
                f"recurrent F**{horizon} minus recurrent F**1 "
                "matched-minus-crossed cosine margin"
            ),
            require_same_support=False,
        )
        degradation[str(horizon)]["sign_interpretation"] = (
            "negative means pair discrimination degraded relative to horizon 1"
        )
    return {
        "resampling_unit": "development pair",
        "interval": "percentile 95% pair bootstrap",
        "base_seed": bootstrap_seed,
        "resamples": bootstrap_resamples,
        "recurrent_full_history_minus_controls": versus_controls,
        "recurrent_horizon_k_minus_horizon_1": degradation,
        "oracle_projection_ceiling_excluded_from_causal_contrasts": True,
    }


def _residual_scale_metrics(records: Sequence[ForecastRecord]) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if row.accumulated_residual_variance_heuristic is not None
    ]
    if not selected:
        return {"count": 0}
    totals = np.asarray(
        [
            float(np.sum(row.accumulated_residual_variance_heuristic))
            for row in selected
        ]
    )
    normalized_errors = np.asarray(
        [
            float(np.sum((row.vector - row.target) ** 2)) / max(total, 1e-12)
            for row, total in zip(selected, totals, strict=True)
        ]
    )
    return {
        "count": len(selected),
        "mean_accumulated_residual_variance_sum": float(np.mean(totals)),
        "mean_accumulated_residual_variance_per_coordinate": float(
            np.mean(totals) / selected[0].vector.size
        ),
        "mean_squared_error_over_accumulated_residual_variance": float(
            np.mean(normalized_errors)
        ),
        "interpretation": (
            "residual-scale heuristic formed by summing one-step residual "
            "variances along the recursive rollout; it assumes additive "
            "independent errors and is not calibrated future uncertainty"
        ),
    }


def _record_from_slot(
    *,
    pair_id: str,
    side: str,
    horizon: int,
    slot: FutureSlot,
    target: EncodedTurn,
) -> ForecastRecord:
    return ForecastRecord(
        pair_id=pair_id,
        side=side,
        horizon=horizon,
        vector=slot.vector,
        target=target.vector,
        predicted_role=str(slot.role_id),
        target_role=str(target.role_id),
        predicted_source=str(slot.source_id),
        target_source=str(target.source_id),
        accumulated_residual_variance_heuristic=(
            slot.accumulated_residual_variance_heuristic
        ),
    )


def _constant_record(
    *,
    pair_id: str,
    side: str,
    horizon: int,
    vector: np.ndarray,
    target: EncodedTurn,
) -> ForecastRecord:
    return ForecastRecord(
        pair_id=pair_id,
        side=side,
        horizon=horizon,
        vector=vector,
        target=target.vector,
        predicted_role=None,
        target_role=str(target.role_id),
        predicted_source=None,
        target_source=str(target.source_id),
        accumulated_residual_variance_heuristic=None,
    )


def _role_swapped(turns: Sequence[EncodedTurn]) -> tuple[EncodedTurn, ...]:
    opposite = {"USER": "ASSISTANT", "ASSISTANT": "USER"}
    return tuple(
        EncodedTurn(
            turn.vector,
            source_id=turn.source_id,
            role_id=opposite[str(turn.role_id)],
        )
        for turn in turns
    )


def _observe_prefix(
    model: FutureTurnTrajectory, prefix: Sequence[EncodedTurn]
) -> None:
    model.reset()
    for turn in prefix:
        model.observe(turn)


def _evaluate(
    model: FutureTurnTrajectory,
    pairs: Sequence[RawPair],
    *,
    dimensions: int,
    train_event_mean: np.ndarray,
    shuffle_seed: int,
) -> tuple[dict[str, dict[int, list[ForecastRecord]]], dict[str, Any]]:
    condition_names = (
        "recurrent_full_history",
        "mean_event_prior",
        "current_turn_only_reset_history",
        "shuffled_transition",
        "role_swapped_prefix",
        "reversed_horizon_order",
        "oracle_projection_ceiling",
    )
    records: dict[str, dict[int, list[ForecastRecord]]] = {
        condition: {horizon: [] for horizon in range(1, 5)}
        for condition in condition_names
    }
    shuffled = model.shuffled_transition(seed=shuffle_seed)
    reversed_horizons = (4, 3, 2, 1)
    anatomy_before = model.fingerprint(include_state=False)

    for pair in pairs:
        for side_name, raw_side in (("a", pair.a), ("b", pair.b)):
            prefix = _encode_sequence(raw_side.prefix, dimensions)
            future = _encode_sequence(raw_side.future, dimensions)

            _observe_prefix(model, prefix)
            intact = model.predicted_tape()
            shuffled_tape = model.predicted_tape(transition_matrix=shuffled)
            permuted = model.predicted_tape(
                horizon_permutation=reversed_horizons
            )

            _observe_prefix(model, prefix[-1:])
            current_only = model.predicted_tape()

            _observe_prefix(model, _role_swapped(prefix))
            swapped = model.predicted_tape()

            _observe_prefix(model, prefix)
            padded_future = future + (future[-1],) * (4 - len(future))
            oracle = model.oracle_tape(padded_future)

            tapes = {
                "recurrent_full_history": intact,
                "current_turn_only_reset_history": current_only,
                "shuffled_transition": shuffled_tape,
                "role_swapped_prefix": swapped,
                "reversed_horizon_order": permuted,
                "oracle_projection_ceiling": oracle,
            }
            for horizon, target in enumerate(future, 1):
                for condition, tape in tapes.items():
                    records[condition][horizon].append(
                        _record_from_slot(
                            pair_id=pair.pair_id,
                            side=side_name,
                            horizon=horizon,
                            slot=tape.slots[horizon - 1],
                            target=target,
                        )
                    )
                records["mean_event_prior"][horizon].append(
                    _constant_record(
                        pair_id=pair.pair_id,
                        side=side_name,
                        horizon=horizon,
                        vector=train_event_mean,
                        target=target,
                    )
                )

    if model.fingerprint(include_state=False) != anatomy_before:
        raise AssertionError("evaluation mutated learned trajectory anatomy")
    validation = {
        "learned_anatomy_unchanged_by_evaluation": True,
        "shuffled_transition_sha256": _sha256(
            np.ascontiguousarray(shuffled, dtype="<f8").tobytes()
        ),
        "horizon_permutation": list(reversed_horizons),
        "oracle_marked_noncausal": True,
        "predicted_tapes_receive_observed_future": False,
    }
    return records, validation


def _summarize_conditions(
    records: Mapping[str, Mapping[int, Sequence[ForecastRecord]]],
    widths: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition, by_horizon in records.items():
        result[condition] = {
            "classification": (
                "noncausal_projection_ceiling"
                if condition == "oracle_projection_ceiling"
                else "causal_forecast_or_control"
            ),
            "by_horizon": {
                str(horizon): {
                    "horizon": horizon,
                    "recursive_transition_applications": (
                        0
                        if condition
                        in {"mean_event_prior", "oracle_projection_ceiling"}
                        else (
                            5 - horizon
                            if condition == "reversed_horizon_order"
                            else horizon
                        )
                    ),
                    "information_width": int(widths[horizon - 1]),
                    **_records_metrics(by_horizon[horizon]),
                }
                for horizon in range(1, 5)
            },
        }
    return result


def run_audit(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    locked_hashes: Mapping[str, str] = LOCKED_INPUT_SHA256,
    event_dimensions: int = DEFAULT_EVENT_DIMENSIONS,
    horizon_widths: Sequence[int] = DEFAULT_HORIZON_WIDTHS,
    memory_decay: float = DEFAULT_MEMORY_DECAY,
    ridge: float = DEFAULT_RIDGE,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Fit the one-step transition and audit causal recurrent dev rollouts."""

    started = time.perf_counter()
    widths = tuple(int(width) for width in horizon_widths)
    if len(widths) != 4:
        raise ValueError("the audit requires exactly four ordered horizon widths")
    data = load_development_data(dataset_dir, locked_hashes=locked_hashes)

    encoded_train = tuple(
        _encode_sequence(conversation.turns, event_dimensions)
        for conversation in data.train
    )
    all_train_events = np.stack(
        [turn.vector for sequence in encoded_train for turn in sequence]
    )
    train_event_mean = np.mean(all_train_events, axis=0)
    model = FutureTurnTrajectory(
        event_dimensions,
        source_ids=("text",),
        role_ids=("USER", "ASSISTANT"),
        horizon_widths=widths,
        memory_decay=memory_decay,
        ridge=ridge,
    ).fit(encoded_train)
    fitted_fingerprint = model.fingerprint(include_state=False)
    records, evaluation_validation = _evaluate(
        model,
        data.dev,
        dimensions=event_dimensions,
        train_event_mean=train_event_mean,
        shuffle_seed=shuffle_seed,
    )
    conditions = _summarize_conditions(records, widths)
    inferential_contrasts = _inferential_contrasts(
        records,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    recurrent = records["recurrent_full_history"]
    transition_loss = {
        str(horizon): {
            "recursive_transition_applications": horizon,
            "information_width": widths[horizon - 1],
            "mean_per_coordinate_mse": _records_metrics(recurrent[horizon])[
                "mse"
            ]["mean_per_coordinate"],
            "mean_cosine": _records_metrics(recurrent[horizon])["cosine"]["mean"],
            "evaluated_sides": len(recurrent[horizon]),
        }
        for horizon in range(1, 5)
    }
    residual_scale_heuristic = {
        str(horizon): _residual_scale_metrics(recurrent[horizon])
        for horizon in range(1, 5)
    }
    transitions = sum(len(sequence) - 1 for sequence in encoded_train)
    future_counts = {
        str(horizon): sum(
            len(side.future) >= horizon
            for pair in data.dev
            for side in (pair.a, pair.b)
        )
        for horizon in range(1, 5)
    }
    implementation_path = Path(__file__)
    return {
        "schema_version": 1,
        "task": "development_only_recurrent_future_turn_trajectory_audit",
        "status": "complete_representation_audit",
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "required_schema_version": 4,
            "required_goal_target_kind": "successor",
            "opened_data_files": list(OPENED_DATA_FILES),
            "locked_sha256_verified": data.locked_hashes,
            "train_conversations": len(data.train),
            "train_turns": int(sum(len(row.turns) for row in data.train)),
            "adjacent_training_transitions": transitions,
            "every_adjacent_transition_in_full_training_conversations_used": True,
            "development_pairs_available": data.total_dev_pairs,
            "engineering_exposed_pair_ids": list(ENGINEERING_EXPOSED_PAIR_IDS),
            "engineering_exposed_pair_ids_found_and_excluded": list(
                data.excluded_pair_ids
            ),
            "development_pairs_evaluated": len(data.dev),
            "development_sides_by_observed_future_horizon": future_counts,
            "protected_payload_accessed": False,
            "split_selector_present": False,
        },
        "adapter": {
            "event_dimensions": event_dimensions,
            "kind": "deterministic Unicode signed-hash lexical unigram/bigram",
            "source_id": "text",
            "role_ids": ["USER", "ASSISTANT"],
            "binding": (
                "event vector is passed with explicit source and role IDs; the "
                "generic trajectory core multiplicatively binds source-role blocks"
            ),
            "scope": (
                "chat-specific observation adapter; trajectory mechanism accepts "
                "any modality that supplies event vectors and explicit IDs"
            ),
        },
        "model": {
            "class": "FutureTurnTrajectory",
            "training_objective": model.training_objective,
            "training_target": "the immediately following observed turn token",
            "trains_through_recursive_rollout": model.trains_through_rollout,
            "learned_transition_operators": 1,
            "recurrent_state": "current bound turn plus decayed conversation history",
            "memory_decay": memory_decay,
            "ridge": ridge,
            "horizon_widths": list(widths),
            "nested_information_bottleneck": True,
            "fitted_anatomy_fingerprint": fitted_fingerprint,
        },
        "protocol": {
            "causal_input": (
                "only the original development prefix through its pending USER turn"
            ),
            "one_step": "horizon 1 applies the fitted matrix F once",
            "recursive_rollout": (
                "horizon k applies that exact same fitted matrix recursively as F**k"
            ),
            "training_scope": (
                "closed-form one-step ridge on observed adjacent training states; "
                "horizons 2-4 are evaluated recursively and were not multistep-trained"
            ),
            "targets": (
                "actual next assistant response followed by up to three observed "
                "successor turns; targets are withheld from every causal forecast"
            ),
            "oracle": (
                "observed future vectors passed only through the fixed nested PCA "
                "bottlenecks as a non-causal projection ceiling"
            ),
            "pair_discrimination": (
                "cos(pred_a,target_a)+cos(pred_b,target_b)-"
                "cos(pred_a,target_b)-cos(pred_b,target_a)"
            ),
            "retrieval_rank": (
                "rank of each own observed future among all available development "
                "side targets at the same horizon by cosine"
            ),
            "shuffle_seed": shuffle_seed,
            "paired_bootstrap_seed": bootstrap_seed,
            "paired_bootstrap_resamples": bootstrap_resamples,
            "inferential_unit": (
                "one matched-minus-crossed cosine margin per development pair"
            ),
        },
        "controls": {
            "mean_event_prior": "constant mean of all encoded training turns",
            "current_turn_only_reset_history": (
                "reset conversation state, observe only the pending turn, then roll out"
            ),
            "shuffled_transition": (
                "deterministic output-coordinate shuffle of the fitted transition"
            ),
            "role_swapped_prefix": (
                "preserve every lexical event and order but exchange USER/ASSISTANT IDs"
            ),
            "reversed_horizon_order": (
                "place recursive horizons 4,3,2,1 into ordered width slots 1,2,3,4"
            ),
            "oracle_projection_ceiling": (
                "non-causal observed target after the same slot bottleneck; "
                "not a forecast"
            ),
        },
        "conditions": conditions,
        "inferential_contrasts": inferential_contrasts,
        "transition_loss_by_horizon": transition_loss,
        "accumulated_residual_variance_heuristic_by_horizon": (
            residual_scale_heuristic
        ),
        "validity": evaluation_validation,
        "claim_limitations": [
            (
                "This is a development-only natural-chat representation audit; it "
                "contains no byte likelihood, BPB, decoding, or free-generation claim."
            ),
            (
                "The fitted objective is one-step next-turn regression. Recursive "
                "F**k behavior is measured but never trained directly."
            ),
            (
                "A predictable future turn is not thereby an inferred shared goal, "
                "preference, utility, completion potential, or consequence value."
            ),
            (
                "Lexical hashing is a diagnostic text encoder, not evidence that a "
                "learned multimodal codec retains production-sufficient semantics."
            ),
            (
                "Oracle projection metrics use observed future turns and establish "
                "only the ceiling imposed by the nested width schedule."
            ),
            (
                "Horizon four has fewer natural examples when a conversation records "
                "only the minimum two successor turns; every metric reports its count."
            ),
            (
                "Bootstrap intervals are fixed-seed development-set uncertainty "
                "descriptions. They are not multiplicity-adjusted confirmatory "
                "tests and cannot promote a mechanism without a frozen follow-up."
            ),
        ],
        "implementation": {
            "module": (
                "manifoldflow.experiments."
                "dendritic_basin_future_turn_trajectory_audit"
            ),
            "path": str(implementation_path),
            "sha256": _sha256(implementation_path.read_bytes()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--event-dimensions", type=int, default=96)
    parser.add_argument(
        "--horizon-widths", type=int, nargs=4, default=DEFAULT_HORIZON_WIDTHS
    )
    parser.add_argument("--memory-decay", type=float, default=DEFAULT_MEMORY_DECAY)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_audit(
        args.dataset_dir,
        event_dimensions=args.event_dimensions,
        horizon_widths=tuple(args.horizon_widths),
        memory_decay=args.memory_decay,
        ridge=args.ridge,
        shuffle_seed=args.shuffle_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report_path = args.report
    if report_path is None:
        report_path = (
            args.dataset_dir.parent
            / f"{args.dataset_dir.name}_future_turn_trajectory_audit.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "runtime_seconds": report["runtime_seconds"],
                "development_pairs": report["dataset"][
                    "development_pairs_evaluated"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
