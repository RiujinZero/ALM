"""Audit continuous future-target prediction on held-out crossover pairs.

This runner deliberately stops before byte scoring.  It asks whether a small
continuous representation of an observed future can be predicted from causal
conversation evidence and whether the prediction assigns two held-out futures
to their matching histories more often than to the crossed histories.

The input corpus must be produced by ``build_goal_trajectory_ablation.py`` with
``goal_target_kind`` set to ``successor`` or ``interlocutor``.  Evidence ends at
the pending USER turn.  The scored ASSISTANT response, the synthetic crossover
bridge, and the future target are never part of the predictor input.

For successor corpora, the audit compares a single signed hash of the complete
later-trajectory payload with multi-timescale, per-turn successor signatures.
For interlocutor corpora, the literal next-USER payload is represented by the
single signed hash.  All targets are observed bytes; there are no hand labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    normalize,
    parse_turns,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
DEFAULT_HASH_SALTS = (3, 11, 29)
DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_STATE_DECAYS = (0.5, 0.85)
DEFAULT_SUCCESSOR_GAMMAS = (0.5, 0.85, 0.97)
DEFAULT_SUCCESSOR_BASE_DIMENSIONS = (16, 32)
DEFAULT_CALIBRATION_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


@dataclass(frozen=True)
class RawExample:
    episode_id: str
    causal_turns: tuple[str, ...]
    goal_payload: bytes


@dataclass(frozen=True)
class RawPair:
    pair_id: str
    causal_turns_a: tuple[str, ...]
    causal_turns_b: tuple[str, ...]
    goal_payload_a: bytes
    goal_payload_b: bytes


@dataclass(frozen=True)
class RawDataset:
    goal_target_kind: str
    train: tuple[RawExample, ...]
    dev: tuple[RawPair, ...]
    test: tuple[RawPair, ...]
    manifest: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    adapter: str
    base_dimensions: int
    gammas: tuple[float, ...] = ()

    @property
    def output_dimensions(self) -> int:
        if self.adapter == "successor_signature":
            return self.base_dimensions * len(self.gammas)
        return self.base_dimensions


@dataclass(frozen=True)
class SpectralRidge:
    """Reusable centered ridge decomposition for several continuous targets."""

    train_projection: np.ndarray
    eval_projection: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def from_evidence(
        cls, train_evidence: np.ndarray, eval_evidence: np.ndarray
    ) -> SpectralRidge:
        train = np.asarray(train_evidence, dtype=np.float64)
        evaluate = np.asarray(eval_evidence, dtype=np.float64)
        mean = train.mean(axis=0)
        centered_train = train - mean
        eigenvalues, eigenvectors = np.linalg.eigh(centered_train.T @ centered_train)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return cls(
            train_projection=eigenvectors.T @ centered_train.T,
            eval_projection=(evaluate - mean) @ eigenvectors,
            eigenvalues=eigenvalues,
        )

    def predict(self, train_target: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0.0:
            raise ValueError("ridge alpha must be positive")
        target = np.asarray(train_target, dtype=np.float64)
        target_mean = target.mean(axis=0)
        coefficients = self.train_projection @ (target - target_mean)
        coefficients /= (self.eigenvalues + alpha)[:, None]
        return self.eval_projection @ coefficients + target_mean


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"sidecar row {line_number} is not an object")
            rows.append(row)
    return rows


def _slice(blob: bytes, record: dict[str, Any]) -> bytes:
    start = int(record["start"])
    end = int(record["end"])
    if start < 0 or end < start or end > len(blob):
        raise ValueError("sidecar range lies outside its blob")
    payload = blob[start:end]
    expected = record.get("sha256")
    if expected is not None and _sha256(payload) != str(expected):
        raise ValueError("sidecar range checksum does not match its blob")
    return payload


def salted_text_vector(text: str, dimensions: int, salt: int) -> np.ndarray:
    """Return a deterministic salted signed-hash unigram/bigram vector."""

    if dimensions < 8:
        raise ValueError("feature dimensions must be at least eight")
    if salt < 0 or salt >= 2**64:
        raise ValueError("hash salt must fit an unsigned 64-bit integer")
    tokens = TOKEN_PATTERN.findall(text.casefold())
    features = [f"u:{token}" for token in tokens]
    features.extend(
        f"b:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    )
    if not features:
        features = ["empty"]
    vector = np.zeros(dimensions, dtype=np.float64)
    person = salt.to_bytes(8, "little")
    for feature in features:
        digest = hashlib.blake2s(
            feature.encode("utf-8"), digest_size=8, person=person
        ).digest()
        code = int.from_bytes(digest, "little")
        vector[code % dimensions] += 1.0 if code >> 63 else -1.0
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    return normalize(vector)


def successor_signature(
    payload: bytes,
    *,
    base_dimensions: int,
    gammas: tuple[float, ...],
    salt: int,
) -> np.ndarray:
    """Hash each observable future turn and concatenate discounted summaries."""

    turns = parse_turns(payload.decode("utf-8", errors="replace"))
    if len(turns) < 2:
        raise ValueError("successor target must contain at least two parsed turns")
    # Roles provide observable boundaries only.  The learner receives hashes of
    # turn text, not a role/modality one-hot or a hidden trajectory label.
    events = np.asarray(
        [salted_text_vector(turn.text, base_dimensions, salt) for turn in turns]
    )
    summaries = []
    for gamma in gammas:
        if not 0.0 < gamma < 1.0:
            raise ValueError("successor gammas must lie strictly between zero and one")
        weights = (1.0 - gamma) * gamma ** np.arange(len(events))
        summaries.append(normalize(np.sum(events * weights[:, None], axis=0)))
    if not summaries:
        raise ValueError("at least one successor gamma is required")
    return np.concatenate(summaries)


def target_vector(payload: bytes, spec: TargetSpec, salt: int) -> np.ndarray:
    if spec.adapter == "whole_payload":
        return salted_text_vector(
            payload.decode("utf-8", errors="replace"),
            spec.base_dimensions,
            salt,
        )
    if spec.adapter == "successor_signature":
        return successor_signature(
            payload,
            base_dimensions=spec.base_dimensions,
            gammas=spec.gammas,
            salt=salt,
        )
    raise ValueError(f"unknown target adapter: {spec.adapter}")


def causal_evidence(
    turns: tuple[str, ...],
    *,
    dimensions: int,
    decays: tuple[float, ...],
    salt: int,
) -> np.ndarray:
    """Encode causal turns through and including the pending USER event."""

    if not turns:
        raise ValueError("causal evidence needs at least one event")
    events = [salted_text_vector(text, dimensions, salt) for text in turns]
    states = [np.zeros(dimensions, dtype=np.float64) for _ in decays]
    for event in events:
        for index, decay in enumerate(decays):
            if not 0.0 < decay < 1.0:
                raise ValueError("state decays must lie strictly between zero and one")
            states[index] = decay * states[index] + (1.0 - decay) * event
    return np.concatenate((events[-1], *(normalize(state) for state in states)))


def causal_turns_from_prefix(
    context: bytes, metadata: dict[str, Any]
) -> tuple[str, ...]:
    """Parse only the original prefix, excluding a crossover bridge if present."""

    end = int(metadata["target_user_content_end"])
    if end != int(metadata["unanswered_prefix_bytes"]) or end > len(context):
        raise ValueError("causal prefix does not end at the pending USER turn")
    prefix = context[:end]
    expected = metadata.get("unanswered_prefix_sha256")
    if expected is not None and _sha256(prefix) != str(expected):
        raise ValueError("causal prefix checksum does not match episode metadata")
    turns = parse_turns(prefix.decode("utf-8", errors="replace"))
    if not turns or turns[-1].role != "USER":
        raise ValueError("causal prefix must end in a USER event")
    return tuple(turn.text for turn in turns)


def _validate_target_metadata(metadata: dict[str, Any], expected_kind: str) -> int:
    kind = str(metadata["goal_target_kind"])
    if kind != expected_kind:
        raise ValueError(
            f"mixed goal target kinds: expected {expected_kind}, got {kind}"
        )
    target_end = int(metadata["target_content_end"])
    goal_start = int(metadata["goal_target_local_start"])
    if goal_start < target_end:
        raise ValueError("future target overlaps the scored ASSISTANT response")
    if int(metadata["goal_target_turn_start"]) <= int(metadata["target_turn_index"]):
        raise ValueError("future target does not begin after the response turn")
    if expected_kind == "successor":
        count = (
            int(metadata["goal_target_turn_end"])
            - int(metadata["goal_target_turn_start"])
            + 1
        )
        if count < 2:
            raise ValueError("successor metadata contains fewer than two later turns")
    return goal_start - target_end


def _verify_manifest_files(
    dataset_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for record in manifest.get("outputs", []):
        path = dataset_dir / str(record["path"])
        payload = path.read_bytes()
        actual = _sha256(payload)
        if actual != str(record["sha256"]):
            raise ValueError(f"manifest checksum mismatch for {path.name}")
        if len(payload) != int(record["bytes"]):
            raise ValueError(f"manifest byte count mismatch for {path.name}")
        checked.append({"path": path.name, "bytes": len(payload), "sha256": actual})
    return {"manifest_files_verified": len(checked), "files": checked}


def _load_training(
    dataset_dir: Path, expected_kind: str
) -> tuple[tuple[RawExample, ...], list[int]]:
    blob = (dataset_dir / "train.bin").read_bytes()
    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    examples: list[RawExample] = []
    gaps: list[int] = []
    for row in rows:
        start = int(row["start"])
        end = int(row["end"])
        episode = blob[start:end]
        if _sha256(episode) != str(row["sha256"]):
            raise ValueError(f"episode checksum mismatch for {row['episode_id']}")
        gaps.append(_validate_target_metadata(row, expected_kind))
        goal_record = dict(row["goal_target"])
        if str(goal_record["kind"]) != expected_kind:
            raise ValueError("training goal record has the wrong kind")
        if int(goal_record["start"]) < int(row["target_offsets"]["content_end"]):
            raise ValueError("training future target overlaps the response")
        goal_payload = _slice(blob, goal_record)

        prefix_record = dict(row["unanswered_prefix_range"])
        prefix = _slice(blob, prefix_record)
        if len(prefix) != int(row["unanswered_prefix_bytes"]):
            raise ValueError("training causal prefix has the wrong length")
        if int(row["target_user_content_end"]) != len(prefix):
            raise ValueError("training evidence cutoff is not the pending USER end")

        cutoff = int(prefix_record["end"])
        causal: list[str] = []
        target_user_index = int(row["target_user_turn_index"])
        last_role = ""
        for turn in row["turns"]:
            turn_index = int(turn["index"])
            if turn_index > target_user_index:
                break
            content_start = int(turn["content_start"])
            content_end = int(turn["end"])
            if content_start < start or content_end > cutoff:
                raise ValueError("training causal turn crosses its evidence cutoff")
            causal.append(
                blob[content_start:content_end].decode("utf-8", errors="replace")
            )
            last_role = str(turn["role"])
        if not causal or last_role != "USER":
            raise ValueError("training evidence does not end in the pending USER turn")
        examples.append(RawExample(str(row["episode_id"]), tuple(causal), goal_payload))
    if len(examples) < 4:
        raise ValueError("at least four indexed training episodes are required")
    return tuple(examples), gaps


def _load_pairs(
    dataset_dir: Path, split: str, expected_kind: str
) -> tuple[tuple[RawPair, ...], list[int]]:
    rows = _read_jsonl(dataset_dir / f"{split}_pairs.jsonl")
    blob = (dataset_dir / f"{split}_pairs.bin").read_bytes()
    pairs: list[RawPair] = []
    gaps: list[int] = []
    for row in rows:
        if str(row["goal_target_kind"]) != expected_kind:
            raise ValueError(f"{split} pair has the wrong goal target kind")
        if not all(bool(value) for value in row["invariants"].values()):
            raise ValueError(f"{split} pair has a false construction invariant")
        ranges = row["blob_ranges"]
        context_a = _slice(blob, ranges["context_a"])
        context_b = _slice(blob, ranges["context_b"])
        # Reading these ranges verifies that the scored response windows are
        # present and checksummed, although they are never used as evidence.
        response_a = _slice(blob, ranges["response_a"])
        response_b = _slice(blob, ranges["response_b"])
        if len(response_a) != len(response_b):
            raise ValueError("pair response windows have unequal lengths")
        goal_a = _slice(blob, ranges["goal_target_a"])
        goal_b = _slice(blob, ranges["goal_target_b"])
        metadata_a = row["episodes"]["a"]
        metadata_b = row["episodes"]["b"]
        gaps.extend(
            (
                _validate_target_metadata(metadata_a, expected_kind),
                _validate_target_metadata(metadata_b, expected_kind),
            )
        )
        if _sha256(goal_a) != str(metadata_a["goal_target_sha256"]):
            raise ValueError("pair A goal checksum disagrees with episode metadata")
        if _sha256(goal_b) != str(metadata_b["goal_target_sha256"]):
            raise ValueError("pair B goal checksum disagrees with episode metadata")
        pairs.append(
            RawPair(
                pair_id=str(row["pair_id"]),
                causal_turns_a=causal_turns_from_prefix(context_a, metadata_a),
                causal_turns_b=causal_turns_from_prefix(context_b, metadata_b),
                goal_payload_a=goal_a,
                goal_payload_b=goal_b,
            )
        )
    if not pairs:
        raise ValueError(f"{split} sidecar contains no pairs")
    return tuple(pairs), gaps


def load_raw_dataset(dataset_dir: Path) -> RawDataset:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = str(manifest["config"]["goal_target_kind"])
    if kind not in {"successor", "interlocutor"}:
        raise ValueError("audit requires a successor or interlocutor target corpus")
    file_validation = _verify_manifest_files(dataset_dir, manifest)
    train, train_gaps = _load_training(dataset_dir, kind)
    dev, dev_gaps = _load_pairs(dataset_dir, "dev", kind)
    test, test_gaps = _load_pairs(dataset_dir, "test", kind)
    gaps = train_gaps + dev_gaps + test_gaps
    validation = {
        **file_validation,
        "target_ranges_checked": len(gaps),
        "target_starts_at_or_after_full_response_end": all(gap >= 0 for gap in gaps),
        "minimum_target_gap_bytes": min(gaps),
        "maximum_target_gap_bytes": max(gaps),
        "evidence_cutoff": "end of pending USER content before scored response",
        "crossover_bridge_excluded": True,
        "scored_response_excluded": True,
        "future_target_excluded": True,
    }
    return RawDataset(kind, train, dev, test, manifest, validation)


def sattolo_permutation(length: int, seed: int) -> np.ndarray:
    """Return a deterministic single-cycle derangement."""

    if length < 2:
        raise ValueError("Sattolo shuffle requires at least two rows")
    rng = np.random.default_rng(seed)
    permutation = np.arange(length)
    for index in range(length - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = (
            permutation[other],
            permutation[index],
        )
    if np.any(permutation == np.arange(length)):
        raise AssertionError("Sattolo shuffle unexpectedly produced a fixed point")
    return permutation


def pair_margins(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return matched-minus-crossed cosine assignment scores for each pair."""

    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or predicted.ndim != 3
        or predicted.shape[1] != 2
    ):
        raise ValueError(
            "pair prediction and target must both have shape [pairs, 2, dims]"
        )

    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        numerator = np.sum(left * right, axis=1)
        denominator = np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
        )
        return numerator / denominator

    same = cosine(predicted[:, 0], observed[:, 0]) + cosine(
        predicted[:, 1], observed[:, 1]
    )
    crossed = cosine(predicted[:, 0], observed[:, 1]) + cosine(
        predicted[:, 1], observed[:, 0]
    )
    return same - crossed


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _pair_cross_entropy_bits(margins: np.ndarray, scale: float) -> float:
    return float(np.mean(np.logaddexp(0.0, -scale * margins)) / math.log(2.0))


def choose_calibration_scale(margins: np.ndarray, scales: tuple[float, ...]) -> float:
    if not scales or any(scale < 0.0 for scale in scales):
        raise ValueError("calibration scales must be a nonempty non-negative grid")
    return min(
        scales, key=lambda scale: (_pair_cross_entropy_bits(margins, scale), scale)
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval needs at least one observation")
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(fraction * (1.0 - fraction) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    if resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _pair_metrics(
    margins: np.ndarray,
    *,
    scale: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    values = np.asarray(margins, dtype=np.float64)
    probability = _sigmoid(scale * values)
    wins = int(np.sum(values > 0.0))
    low, high = _wilson_interval(wins, len(values))
    mean_low, mean_high = _bootstrap_mean_interval(
        values, seed=bootstrap_seed, resamples=bootstrap_resamples
    )
    return {
        "pairs": len(values),
        "positive_fraction": float(np.mean(values > 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
        "positive_fraction_wilson_ci95": [low, high],
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "mean_margin_bootstrap_ci95": [mean_low, mean_high],
        "calibration_scale": scale,
        "mean_correct_assignment_probability": float(np.mean(probability)),
        "pair_cross_entropy_bits": _pair_cross_entropy_bits(values, scale),
        "pair_brier": float(np.mean((1.0 - probability) ** 2)),
    }


def _regression_metrics(
    prediction: np.ndarray, target: np.ndarray, prior: np.ndarray
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1, target.shape[-1])
    observed = np.asarray(target, dtype=np.float64).reshape(-1, target.shape[-1])
    baseline = np.broadcast_to(prior, observed.shape)

    def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum(left * right, axis=1) / np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
        )

    model_mse = float(np.mean((predicted - observed) ** 2))
    prior_mse = float(np.mean((baseline - observed) ** 2))
    return {
        "mean_cosine": float(np.mean(row_cosine(predicted, observed))),
        "prior_mean_cosine": float(np.mean(row_cosine(baseline, observed))),
        "mse": model_mse,
        "prior_mse": prior_mse,
        "fractional_mse_improvement_over_prior": (
            (prior_mse - model_mse) / prior_mse if prior_mse > 0.0 else 0.0
        ),
    }


def _evaluate_condition(
    prediction_dev: np.ndarray,
    prediction_test: np.ndarray,
    target_dev: np.ndarray,
    target_test: np.ndarray,
    prior: np.ndarray,
    *,
    calibration_scales: tuple[float, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dev_margins = pair_margins(prediction_dev, target_dev)
    test_margins = pair_margins(prediction_test, target_test)
    scale = choose_calibration_scale(dev_margins, calibration_scales)
    result = {
        "calibration_selected_on": "dev pair cross-entropy",
        "dev": {
            "pair": _pair_metrics(
                dev_margins,
                scale=scale,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "regression": _regression_metrics(prediction_dev, target_dev, prior),
        },
        "test": {
            "pair": _pair_metrics(
                test_margins,
                scale=scale,
                bootstrap_seed=bootstrap_seed + 1,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "regression": _regression_metrics(prediction_test, target_test, prior),
        },
    }
    return result, {"dev": dev_margins, "test": test_margins}


def _target_specs(
    kind: str,
    successor_base_dimensions: tuple[int, ...],
    successor_gammas: tuple[float, ...],
) -> tuple[TargetSpec, ...]:
    specs = [TargetSpec("whole_payload_32", "whole_payload", 32)]
    if kind == "successor":
        specs.extend(
            TargetSpec(
                f"successor_turns_{dimensions}",
                "successor_signature",
                dimensions,
                successor_gammas,
            )
            for dimensions in successor_base_dimensions
        )
    return tuple(specs)


def _evidence_arrays(
    dataset: RawDataset,
    *,
    dimensions: int,
    decays: tuple[float, ...],
    salt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(
        [
            causal_evidence(
                row.causal_turns, dimensions=dimensions, decays=decays, salt=salt
            )
            for row in dataset.train
        ]
    )

    def pairs(rows: tuple[RawPair, ...]) -> np.ndarray:
        return np.asarray(
            [
                [
                    causal_evidence(
                        row.causal_turns_a,
                        dimensions=dimensions,
                        decays=decays,
                        salt=salt,
                    ),
                    causal_evidence(
                        row.causal_turns_b,
                        dimensions=dimensions,
                        decays=decays,
                        salt=salt,
                    ),
                ]
                for row in rows
            ]
        )

    return train, pairs(dataset.dev), pairs(dataset.test)


def _target_arrays(
    dataset: RawDataset, spec: TargetSpec, salt: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(
        [target_vector(row.goal_payload, spec, salt) for row in dataset.train]
    )

    def pairs(rows: tuple[RawPair, ...]) -> np.ndarray:
        return np.asarray(
            [
                [
                    target_vector(row.goal_payload_a, spec, salt),
                    target_vector(row.goal_payload_b, spec, salt),
                ]
                for row in rows
            ]
        )

    return train, pairs(dataset.dev), pairs(dataset.test)


def _numeric_aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for condition in ("full", "pending_user_only", "sattolo_shuffled_target"):
        report[condition] = {}
        for split in ("dev", "test"):
            pair_keys = (
                "positive_fraction",
                "mean_margin",
                "mean_correct_assignment_probability",
                "pair_cross_entropy_bits",
                "pair_brier",
            )
            regression_keys = (
                "mean_cosine",
                "prior_mean_cosine",
                "mse",
                "prior_mse",
                "fractional_mse_improvement_over_prior",
            )
            report[condition][split] = {
                "pair": {
                    key: _numeric_aggregate(
                        [
                            float(run["conditions"][condition][split]["pair"][key])
                            for run in runs
                        ]
                    )
                    for key in pair_keys
                },
                "regression": {
                    key: _numeric_aggregate(
                        [
                            float(
                                run["conditions"][condition][split]["regression"][key]
                            )
                            for run in runs
                        ]
                    )
                    for key in regression_keys
                },
            }
    return report


def _split_columns(
    matrix: np.ndarray, specs: tuple[TargetSpec, ...]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    offset = 0
    for spec in specs:
        end = offset + spec.output_dimensions
        result[spec.name] = matrix[:, offset:end]
        offset = end
    if offset != matrix.shape[1]:
        raise AssertionError("target column split does not cover its matrix")
    return result


def _pair_shape(matrix: np.ndarray, pair_count: int) -> np.ndarray:
    return matrix.reshape(pair_count, 2, matrix.shape[-1])


def _consensus_metrics(
    margins_by_salt: list[dict[str, np.ndarray]],
    *,
    calibration_scales: tuple[float, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    dev = np.mean([row["dev"] for row in margins_by_salt], axis=0)
    test = np.mean([row["test"] for row in margins_by_salt], axis=0)
    scale = choose_calibration_scale(dev, calibration_scales)
    return {
        "definition": "per-pair raw margin averaged across hash salts",
        "dev": _pair_metrics(
            dev,
            scale=scale,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "test": _pair_metrics(
            test,
            scale=scale,
            bootstrap_seed=bootstrap_seed + 1,
            bootstrap_resamples=bootstrap_resamples,
        ),
    }


def run_audit(
    dataset_dir: Path,
    *,
    hash_salts: tuple[int, ...] = DEFAULT_HASH_SALTS,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    evidence_dimensions: int = 32,
    state_decays: tuple[float, ...] = DEFAULT_STATE_DECAYS,
    successor_base_dimensions: tuple[int, ...] = DEFAULT_SUCCESSOR_BASE_DIMENSIONS,
    successor_gammas: tuple[float, ...] = DEFAULT_SUCCESSOR_GAMMAS,
    calibration_scales: tuple[float, ...] = DEFAULT_CALIBRATION_SCALES,
    bootstrap_resamples: int = 2_000,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not hash_salts:
        raise ValueError("at least one hash salt is required")
    if not ridge_alphas:
        raise ValueError("at least one ridge alpha is required")
    dataset = load_raw_dataset(dataset_dir)
    specs = _target_specs(
        dataset.goal_target_kind, successor_base_dimensions, successor_gammas
    )
    grid: dict[str, dict[str, list[dict[str, Any]]]] = {
        spec.name: {str(alpha): [] for alpha in ridge_alphas} for spec in specs
    }
    margin_cache: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}

    for salt in hash_salts:
        evidence_train, evidence_dev, evidence_test = _evidence_arrays(
            dataset,
            dimensions=evidence_dimensions,
            decays=state_decays,
            salt=salt,
        )
        target_by_spec = {
            spec.name: _target_arrays(dataset, spec, salt) for spec in specs
        }
        target_train = np.concatenate(
            [target_by_spec[spec.name][0] for spec in specs], axis=1
        )
        target_dev = {spec.name: target_by_spec[spec.name][1] for spec in specs}
        target_test = {spec.name: target_by_spec[spec.name][2] for spec in specs}
        permutation = sattolo_permutation(len(dataset.train), seed=salt + 104_729)
        permutation_sha256 = _sha256(permutation.astype("<i8").tobytes())

        dev_rows = evidence_dev.reshape(-1, evidence_dev.shape[-1])
        test_rows = evidence_test.reshape(-1, evidence_test.shape[-1])
        eval_full = np.concatenate((dev_rows, test_rows), axis=0)
        full_ridge = SpectralRidge.from_evidence(evidence_train, eval_full)
        pending_ridge = SpectralRidge.from_evidence(
            evidence_train[:, :evidence_dimensions],
            eval_full[:, :evidence_dimensions],
        )
        dev_row_count = len(dev_rows)

        for alpha in ridge_alphas:
            full_prediction = full_ridge.predict(target_train, alpha)
            pending_prediction = pending_ridge.predict(target_train, alpha)
            shuffled_prediction = full_ridge.predict(target_train[permutation], alpha)
            split_predictions = {
                "full": _split_columns(full_prediction, specs),
                "pending_user_only": _split_columns(pending_prediction, specs),
                "sattolo_shuffled_target": _split_columns(shuffled_prediction, specs),
            }
            for spec_index, spec in enumerate(specs):
                run: dict[str, Any] = {
                    "hash_salt": salt,
                    "ridge_alpha": alpha,
                    "sattolo_seed": salt + 104_729,
                    "sattolo_permutation_sha256": permutation_sha256,
                    "conditions": {},
                }
                train_slice = target_by_spec[spec.name][0]
                for condition_index, (condition, predictions) in enumerate(
                    split_predictions.items()
                ):
                    prediction = predictions[spec.name]
                    prediction_dev = _pair_shape(
                        prediction[:dev_row_count], len(dataset.dev)
                    )
                    prediction_test = _pair_shape(
                        prediction[dev_row_count:], len(dataset.test)
                    )
                    evaluated, margins = _evaluate_condition(
                        prediction_dev,
                        prediction_test,
                        target_dev[spec.name],
                        target_test[spec.name],
                        train_slice.mean(axis=0),
                        calibration_scales=calibration_scales,
                        bootstrap_seed=(
                            salt * 10_000 + spec_index * 100 + condition_index * 10
                        ),
                        bootstrap_resamples=bootstrap_resamples,
                    )
                    run["conditions"][condition] = evaluated
                    margin_cache[(spec.name, str(alpha), salt, condition)] = margins
                grid[spec.name][str(alpha)].append(run)

    variants: dict[str, Any] = {}
    for spec in specs:
        alpha_rows = []
        for alpha in ridge_alphas:
            runs = grid[spec.name][str(alpha)]
            aggregate = _aggregate_runs(runs)
            alpha_rows.append(
                {"ridge_alpha": alpha, "by_hash_salt": runs, "aggregate": aggregate}
            )
        selected = min(
            alpha_rows,
            key=lambda row: (
                row["aggregate"]["full"]["dev"]["pair"]["pair_cross_entropy_bits"][
                    "mean"
                ],
                -row["aggregate"]["full"]["dev"]["pair"]["positive_fraction"]["mean"],
                row["ridge_alpha"],
            ),
        )
        selected_alpha = float(selected["ridge_alpha"])
        consensus = {}
        for condition_index, condition in enumerate(
            ("full", "pending_user_only", "sattolo_shuffled_target")
        ):
            consensus[condition] = _consensus_metrics(
                [
                    margin_cache[(spec.name, str(selected_alpha), salt, condition)]
                    for salt in hash_salts
                ],
                calibration_scales=calibration_scales,
                bootstrap_seed=90_000 + condition_index * 100,
                bootstrap_resamples=bootstrap_resamples,
            )
        full_test = selected["aggregate"]["full"]["test"]["pair"]
        pending_test = selected["aggregate"]["pending_user_only"]["test"]["pair"]
        shuffled_test = selected["aggregate"]["sattolo_shuffled_target"]["test"]["pair"]
        variants[spec.name] = {
            "adapter": spec.adapter,
            "base_dimensions": spec.base_dimensions,
            "gammas": list(spec.gammas),
            "output_dimensions": spec.output_dimensions,
            "selected_ridge_alpha": selected_alpha,
            "alpha_selection": (
                "minimum mean dev pair cross-entropy across hash salts for full "
                "evidence; ties prefer higher dev win fraction then smaller alpha"
            ),
            "selected_aggregate": selected["aggregate"],
            "selected_by_hash_salt": selected["by_hash_salt"],
            "cross_salt_consensus": consensus,
            "descriptive_checks": {
                "full_test_positive_fraction_above_half_for_every_salt": (
                    full_test["positive_fraction"]["minimum"] > 0.5
                ),
                "full_test_pair_ce_below_one_bit_for_every_salt": (
                    full_test["pair_cross_entropy_bits"]["maximum"] < 1.0
                ),
                "full_test_ce_better_than_pending_mean": (
                    full_test["pair_cross_entropy_bits"]["mean"]
                    < pending_test["pair_cross_entropy_bits"]["mean"]
                ),
                "full_test_ce_better_than_shuffled_mean": (
                    full_test["pair_cross_entropy_bits"]["mean"]
                    < shuffled_test["pair_cross_entropy_bits"]["mean"]
                ),
            },
            "alpha_grid": alpha_rows,
        }

    implementation_path = Path(__file__)
    return {
        "schema_version": 1,
        "task": "continuous_future_target_pair_discrimination_audit",
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "manifest_schema_version": dataset.manifest.get("schema_version"),
            "goal_target_kind": dataset.goal_target_kind,
            "train_examples": len(dataset.train),
            "dev_pairs": len(dataset.dev),
            "test_pairs": len(dataset.test),
            "validation": dataset.validation,
        },
        "configuration": {
            "hash_salts": list(hash_salts),
            "ridge_alphas": list(ridge_alphas),
            "evidence_event_dimensions": evidence_dimensions,
            "state_decays": list(state_decays),
            "full_evidence_dimensions": evidence_dimensions * (1 + len(state_decays)),
            "pending_user_only_dimensions": evidence_dimensions,
            "successor_base_dimensions": list(successor_base_dimensions),
            "successor_gammas": list(successor_gammas),
            "calibration_scales": list(calibration_scales),
            "bootstrap_resamples": bootstrap_resamples,
        },
        "protocol": {
            "evidence": (
                f"{evidence_dimensions}D salted signed hash of pending USER text "
                f"concatenated with normalized {evidence_dimensions}D EMA states "
                f"at decays {list(state_decays)}, updated only through that pending "
                "USER turn"
            ),
            "pending_user_only_control": (
                f"the exact first {evidence_dimensions}D block of full evidence; "
                "no persistent blocks"
            ),
            "whole_payload_target": (
                "one 32D salted signed unigram/bigram hash of every byte-decoded "
                "future payload; observable role-marker words remain only when the "
                "successor payload itself contains them"
            ),
            "successor_signature_target": (
                "future role markers segment turns; each turn text is signed-hashed "
                "at the stated base dimension without role IDs, then summaries "
                "weighted by (1-gamma)*gamma**turn_offset are normalized and "
                "concatenated"
            ),
            "predictor": "centered linear ridge map with intercept",
            "shuffled_control": (
                "same full evidence and target marginal, but training targets follow "
                "a salt-seeded Sattolo single-cycle derangement with no fixed points"
            ),
            "pair_margin": (
                "cos(pred_a,target_a)+cos(pred_b,target_b)-cos(pred_a,target_b)-"
                "cos(pred_b,target_a)"
            ),
            "calibration": (
                "non-negative sigmoid scale selected on dev from a fixed grid by "
                "pair cross-entropy, then frozen for test"
            ),
            "alpha_selection": "dev only; test does not select alpha or scale",
        },
        "target_representations": variants,
        "implementation": {
            "module": (
                "manifoldflow.experiments.dendritic_basin_future_continuous_pair_audit"
            ),
            "path": str(implementation_path),
            "sha256": _sha256(implementation_path.read_bytes()),
        },
        "claim_limitations": [
            (
                "A positive natural-dialogue crossover result establishes predictable "
                "long-context compatibility, not a uniquely goal-directed state."
            ),
            (
                "Signed lexical hashes are a small text adapter, not a learned "
                "modality-general encoder; hash salts reuse the same corpus pairs and "
                "are sensitivity checks rather than independent samples."
            ),
            (
                "The Sattolo control tests train-row association, but cannot rule out "
                "all corpus, topic, author, or template leakage."
            ),
            (
                "This representation-only audit does not show that a byte head or "
                "frozen cortex can turn the predicted vector into better generation."
            ),
            (
                "Ridge alpha and calibration scale are selected on the same 64-pair "
                "development split; only the 128-pair test split is held out from "
                "those choices."
            ),
        ],
        "runtime_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("runs/shared_goal/goal_trajectory_successor_v1"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hash-salts", type=int, nargs="+", default=DEFAULT_HASH_SALTS)
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=DEFAULT_RIDGE_ALPHAS
    )
    parser.add_argument("--evidence-dimensions", type=int, default=32)
    parser.add_argument(
        "--state-decays", type=float, nargs="+", default=DEFAULT_STATE_DECAYS
    )
    parser.add_argument(
        "--successor-base-dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_SUCCESSOR_BASE_DIMENSIONS,
    )
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument(
        "--calibration-scales",
        type=float,
        nargs="+",
        default=DEFAULT_CALIBRATION_SCALES,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report_path = args.report
    if report_path is None:
        report_path = (
            args.dataset_dir.parent
            / f"{args.dataset_dir.name}_continuous_pair_audit.json"
        )
    report = run_audit(
        args.dataset_dir,
        hash_salts=tuple(args.hash_salts),
        ridge_alphas=tuple(args.ridge_alphas),
        evidence_dimensions=args.evidence_dimensions,
        state_decays=tuple(args.state_decays),
        successor_base_dimensions=tuple(args.successor_base_dimensions),
        successor_gammas=tuple(args.successor_gammas),
        calibration_scales=tuple(args.calibration_scales),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = {
        name: {
            "alpha": row["selected_ridge_alpha"],
            "test_positive_fraction_mean": row["selected_aggregate"]["full"]["test"][
                "pair"
            ]["positive_fraction"]["mean"],
            "test_pair_ce_bits_mean": row["selected_aggregate"]["full"]["test"]["pair"][
                "pair_cross_entropy_bits"
            ]["mean"],
        }
        for name, row in report["target_representations"].items()
    }
    print(
        json.dumps(
            {
                "report": str(report_path),
                "runtime_seconds": report["runtime_seconds"],
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
