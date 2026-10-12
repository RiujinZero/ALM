"""Ablate a continuous shared-goal state as a whole-candidate reranker.

This assay is deliberately separate from byte-level goal forcing.  It asks a
different question: after a candidate response has been proposed, does the
future trajectory implied by taking that action agree with a durable state
inferred from the conversation so far?

The mechanism is modality-neutral once events have been converted to dense
vectors:

* a train-only PCA bottleneck defines a continuous future-state target ``z``;
* a full-history ridge map infers ``z`` causally and clamps it to the unit ball;
* an action-successor ridge map predicts the later state from the pending event,
  candidate response, and a bounded multiplicative interaction;
* cosine agreement between those two states reranks the complete candidate on
  top of a separately fitted, pending-event-only plausibility score.

Only episode-aligned datasets whose target is ``successor`` or ``interlocutor``
are accepted.  Every target must begin after the complete scored assistant turn;
the response itself can never serve as its own goal target.

Evaluation uses exact A/A, A/B, B/A, and B/B scores for every symmetric pair.
Gain is selected on development data using the deployable causal condition and
then frozen for test.  Oracle, pending-only, swapped, shuffled-supervision,
shuffled-episode, and candidate-blind controls separate target inference from
the candidate-conditioned consequence model.

Natural-dialogue success establishes long-context trajectory compatibility, not
autonomous goal formation or human-level response quality.  This reranker can
only choose among candidates supplied by another generator.  The optional
disjoint-supervised-fit protocol puts goal maps and action/plausibility maps on
opposite source-stratified training halves, then swaps them, to test whether a
result depends on shared-example estimator error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
    _load_native,
    _native_trace,
    load_crossover_pairs,
    load_training_examples,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_DECAYS,
    DEFAULT_SUCCESSOR_GAMMAS,
    RidgeMap,
    fit_ridge,
    hashed_text_vector,
    normalize_rows,
)

ALLOWED_TARGET_KINDS = frozenset({"interlocutor", "successor"})
SCORE_KEYS = ("aa", "ab", "ba", "bb")
CONDITION_NAMES = (
    "goal_off",
    "oracle_future_target",
    "causal_full_history",
    "causal_pending_user_only",
    "swapped_oracle",
    "swapped_causal_full_history",
    "shuffled_predictor_target",
    "shuffled_action_target_mapping",
    "shuffled_candidate_action_mapping",
    "shuffled_episode_goal",
    "candidate_blind_action",
)


@dataclass(frozen=True)
class ContinuousTargetEncoder:
    """Train-only linear target bottleneck with deterministic component signs."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_fraction: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("target matrix does not match the fitted encoder")
        return _clamp_rows((matrix - self.mean) @ self.components.T)


@dataclass(frozen=True)
class FittedReranker:
    encoder: ContinuousTargetEncoder
    full_goal_predictor: RidgeMap
    pending_goal_predictor: RidgeMap
    shuffled_goal_predictor: RidgeMap
    action_successor: RidgeMap
    shuffled_action_successor: RidgeMap
    shuffled_candidate_action_successor: RidgeMap
    candidate_blind_successor: RidgeMap
    plausibility_predictor: RidgeMap
    feature_dimensions: int
    predictor_target_permutation: np.ndarray
    action_target_permutation: np.ndarray
    candidate_action_permutation: np.ndarray


@dataclass(frozen=True)
class PairVectors:
    pair_id: str
    pending_a: np.ndarray
    pending_b: np.ndarray
    response_a: np.ndarray
    response_b: np.ndarray
    target_a: np.ndarray
    target_b: np.ndarray


def _clamp_rows(values: np.ndarray) -> np.ndarray:
    """Project rows outside the unit ball onto its boundary."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(1.0, norms)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = np.asarray(left, dtype=np.float64)
    right_matrix = np.asarray(right, dtype=np.float64)
    if left_matrix.shape != right_matrix.shape or left_matrix.ndim != 2:
        raise ValueError("cosine operands must be equal-shaped matrices")
    left_unit = normalize_rows(left_matrix)
    right_unit = normalize_rows(right_matrix)
    return np.clip(np.sum(left_unit * right_unit, axis=1), -1.0, 1.0)


def fit_target_encoder(
    targets: np.ndarray, *, latent_dimensions: int
) -> ContinuousTargetEncoder:
    """Fit a deterministic PCA projection using training targets only."""

    matrix = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("at least two target rows are required")
    if latent_dimensions < 1:
        raise ValueError("latent_dimensions must be positive")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    usable = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-10))
    width = min(latent_dimensions, usable)
    if width < 1:
        raise ValueError("future targets have no non-constant latent dimension")
    components = right[:width].copy()
    # SVD component signs are arbitrary.  Canonicalizing the largest loading
    # makes JSON reports reproducible across conforming linear-algebra builds.
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    energy = singular**2
    explained = float(energy[:width].sum() / max(float(energy.sum()), 1e-300))
    return ContinuousTargetEncoder(mean, components, explained)


def _sattolo_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic-seeded single-cycle derangement."""

    if size < 2:
        raise ValueError("a shuffled control requires at least two examples")
    order = np.arange(size)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        order[index], order[other] = order[other], order[index]
    if np.any(order == np.arange(size)):
        raise AssertionError("Sattolo permutation unexpectedly has a fixed point")
    return order


def action_features(
    pending: np.ndarray, candidate: np.ndarray, *, interaction: bool = True
) -> np.ndarray:
    """Build modality-neutral action features from aligned dense event vectors.

    The normalized Hadamard product is a bounded multiplicative interaction:
    it exposes feature conjunctions without allowing vector magnitude to encode
    the condition.  A bounded cosine scalar preserves a global compatibility
    cue.  Both are omitted by the explicit no-interaction ablation.
    """

    pending_matrix = np.asarray(pending, dtype=np.float64)
    candidate_matrix = np.asarray(candidate, dtype=np.float64)
    if pending_matrix.ndim == 1:
        pending_matrix = pending_matrix[None, :]
    if candidate_matrix.ndim == 1:
        candidate_matrix = candidate_matrix[None, :]
    if pending_matrix.shape != candidate_matrix.shape or pending_matrix.ndim != 2:
        raise ValueError("pending and candidate vectors must have equal shape")
    pending_unit = normalize_rows(pending_matrix)
    candidate_unit = normalize_rows(candidate_matrix)
    if not interaction:
        return np.concatenate((pending_unit, candidate_unit), axis=1)
    product = normalize_rows(pending_unit * candidate_unit)
    cosine = np.sum(pending_unit * candidate_unit, axis=1, keepdims=True)
    return np.concatenate((pending_unit, candidate_unit, product, cosine), axis=1)


def fit_reranker(
    examples: list[TrainingExample],
    *,
    feature_dimensions: int,
    latent_dimensions: int,
    goal_ridge_alpha: float,
    action_ridge_alpha: float,
    plausibility_ridge_alpha: float,
    seed: int,
    interaction: bool = True,
) -> FittedReranker:
    if len(examples) < 4:
        raise ValueError("at least four training episodes are required")
    future = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    encoder = fit_target_encoder(future, latent_dimensions=latent_dimensions)
    all_indices = np.arange(len(examples))
    return _fit_partitioned_reranker(
        examples,
        encoder=encoder,
        goal_indices=all_indices,
        action_indices=all_indices,
        feature_dimensions=feature_dimensions,
        goal_ridge_alpha=goal_ridge_alpha,
        action_ridge_alpha=action_ridge_alpha,
        plausibility_ridge_alpha=plausibility_ridge_alpha,
        seed=seed,
        interaction=interaction,
    )


def _fit_partitioned_reranker(
    examples: list[TrainingExample],
    *,
    encoder: ContinuousTargetEncoder,
    goal_indices: np.ndarray,
    action_indices: np.ndarray,
    feature_dimensions: int,
    goal_ridge_alpha: float,
    action_ridge_alpha: float,
    plausibility_ridge_alpha: float,
    seed: int,
    interaction: bool,
) -> FittedReranker:
    """Fit goal and action maps on explicitly selected training episodes."""

    evidence = np.asarray([row.evidence for row in examples], dtype=np.float64)
    response = np.asarray([row.response_vector for row in examples], dtype=np.float64)
    future = np.asarray([row.goal_target_vector for row in examples], dtype=np.float64)
    if evidence.shape[1] < feature_dimensions:
        raise ValueError("evidence is smaller than one pending-event vector")
    if response.shape[1] != feature_dimensions:
        raise ValueError("response vector width must match feature_dimensions")
    goal_indices = np.asarray(goal_indices, dtype=np.int64)
    action_indices = np.asarray(action_indices, dtype=np.int64)
    for label, indices in (("goal", goal_indices), ("action", action_indices)):
        if indices.ndim != 1 or len(indices) < 4:
            raise ValueError(f"{label} fit requires at least four training episodes")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{label} fit indices must be unique")
        if int(indices.min()) < 0 or int(indices.max()) >= len(examples):
            raise ValueError(f"{label} fit index lies outside the training set")

    pending = evidence[:, :feature_dimensions]
    target_z = encoder.transform(future)
    rng = np.random.default_rng(seed)
    predictor_permutation = _sattolo_permutation(len(goal_indices), rng)
    action_permutation = _sattolo_permutation(len(action_indices), rng)
    candidate_permutation = _sattolo_permutation(len(action_indices), rng)

    goal_evidence = evidence[goal_indices]
    goal_pending = pending[goal_indices]
    goal_targets = target_z[goal_indices]
    action_pending = pending[action_indices]
    action_response = response[action_indices]
    action_targets = target_z[action_indices]
    actions = action_features(action_pending, action_response, interaction=interaction)
    shuffled_candidate_actions = action_features(
        action_pending,
        action_response[candidate_permutation],
        interaction=interaction,
    )
    return FittedReranker(
        encoder=encoder,
        full_goal_predictor=fit_ridge(goal_evidence, goal_targets, goal_ridge_alpha),
        pending_goal_predictor=fit_ridge(goal_pending, goal_targets, goal_ridge_alpha),
        shuffled_goal_predictor=fit_ridge(
            goal_evidence,
            goal_targets[predictor_permutation],
            goal_ridge_alpha,
        ),
        action_successor=fit_ridge(actions, action_targets, action_ridge_alpha),
        shuffled_action_successor=fit_ridge(
            actions, action_targets[action_permutation], action_ridge_alpha
        ),
        shuffled_candidate_action_successor=fit_ridge(
            shuffled_candidate_actions, action_targets, action_ridge_alpha
        ),
        candidate_blind_successor=fit_ridge(
            action_pending, action_targets, action_ridge_alpha
        ),
        plausibility_predictor=fit_ridge(
            action_pending, action_response, plausibility_ridge_alpha
        ),
        feature_dimensions=feature_dimensions,
        predictor_target_permutation=predictor_permutation,
        action_target_permutation=action_permutation,
        candidate_action_permutation=candidate_permutation,
    )


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


def _training_source_labels(
    dataset_dir: Path, examples: list[TrainingExample]
) -> tuple[str, ...]:
    """Load source strata while verifying sidecar/example order exactly."""

    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    if len(rows) != len(examples):
        raise ValueError("training sidecar and loaded examples have different lengths")
    labels = []
    for index, (row, example) in enumerate(zip(rows, examples, strict=True)):
        if str(row.get("episode_id")) != example.episode_id:
            raise ValueError(f"training sidecar order mismatch at row {index}")
        source = str(row.get("source_label", ""))
        if not source:
            raise ValueError(f"training row {index} has no source_label")
        labels.append(source)
    return tuple(labels)


def _source_stratified_disjoint_halves(
    examples: list[TrainingExample], source_labels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Assign stable, nearly balanced A/B halves within every source stratum."""

    if len(examples) != len(source_labels):
        raise ValueError("examples and source labels must have equal lengths")
    if len({row.episode_id for row in examples}) != len(examples):
        raise ValueError("disjoint fitting requires unique episode ids")
    by_source: dict[str, list[int]] = {}
    for index, source in enumerate(source_labels):
        by_source.setdefault(source, []).append(index)

    fold_a: list[int] = []
    fold_b: list[int] = []
    for source in sorted(by_source):
        ordered = sorted(
            by_source[source],
            key=lambda index: (
                hashlib.sha256(
                    (
                        "trajectory-reranker-disjoint-v1\x1f"
                        + source
                        + "\x1f"
                        + examples[index].episode_id
                    ).encode("utf-8")
                ).digest(),
                examples[index].episode_id,
            ),
        )
        fold_a.extend(ordered[::2])
        fold_b.extend(ordered[1::2])
    fold_a_array = np.asarray(sorted(fold_a), dtype=np.int64)
    fold_b_array = np.asarray(sorted(fold_b), dtype=np.int64)
    if len(fold_a_array) < 4 or len(fold_b_array) < 4:
        raise ValueError("each disjoint supervised fit half needs four episodes")
    if np.intersect1d(fold_a_array, fold_b_array).size:
        raise AssertionError("disjoint supervised fit halves overlap")
    if sorted(np.concatenate((fold_a_array, fold_b_array)).tolist()) != list(
        range(len(examples))
    ):
        raise AssertionError("disjoint supervised fit halves do not cover training")
    return fold_a_array, fold_b_array


def _validate_episode_target(metadata: dict[str, Any], *, label: str) -> str:
    kind = str(metadata.get("goal_target_kind", "response"))
    if kind not in ALLOWED_TARGET_KINDS:
        raise ValueError(
            f"{label} uses {kind!r}; reranking requires a strictly-later "
            "successor or interlocutor target"
        )
    goal_start = int(metadata["goal_target_local_start"])
    response_end = int(metadata["target_content_end"])
    goal_turn = int(metadata["goal_target_turn_start"])
    response_turn = int(metadata["target_turn_index"])
    if goal_start < response_end or goal_turn <= response_turn:
        raise ValueError(f"{label} goal target is not strictly after the response")
    return kind


def validate_dataset_semantics(dataset_dir: Path) -> str:
    """Reject response-as-goal or sidecars whose goal can overlap the action."""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_kind = str(manifest.get("config", {}).get("goal_target_kind", "response"))
    if manifest_kind not in ALLOWED_TARGET_KINDS:
        raise ValueError(
            "dataset goal_target_kind must be successor or interlocutor, never response"
        )
    train_rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    if not train_rows:
        raise ValueError("training sidecar contains no episodes")
    for index, row in enumerate(train_rows):
        goal_record = row.get("goal_target")
        if not isinstance(goal_record, dict):
            raise ValueError(
                "every training episode needs an explicit goal_target range"
            )
        kind = _validate_episode_target(row, label=f"train row {index}")
        if str(goal_record.get("kind")) != kind:
            raise ValueError(f"train row {index} has inconsistent goal target kinds")
        if int(goal_record["start"]) < int(row["target_offsets"]["content_end"]):
            raise ValueError(f"train row {index} goal range overlaps its response")
    for split in ("dev", "test"):
        rows = _read_jsonl(dataset_dir / f"{split}_pairs.jsonl")
        if not rows:
            raise ValueError(f"{split} sidecar contains no pairs")
        for index, row in enumerate(rows):
            kind = str(row.get("goal_target_kind", "response"))
            if kind != manifest_kind or kind not in ALLOWED_TARGET_KINDS:
                raise ValueError(
                    f"{split} pair {index} has an invalid goal target kind"
                )
            ranges = row.get("blob_ranges", {})
            if "goal_target_a" not in ranges or "goal_target_b" not in ranges:
                raise ValueError(f"{split} pair {index} lacks explicit future targets")
            for side in ("a", "b"):
                episode_kind = _validate_episode_target(
                    row["episodes"][side], label=f"{split} pair {index}/{side}"
                )
                if episode_kind != kind:
                    raise ValueError(
                        f"{split} pair {index}/{side} has inconsistent target kind"
                    )
    return manifest_kind


def _pair_vectors(
    pairs: list[CrossoverPair], model: FittedReranker
) -> list[PairVectors]:
    result: list[PairVectors] = []
    dimensions = model.feature_dimensions
    for pair in pairs:
        if pair.goal_target_kind not in ALLOWED_TARGET_KINDS:
            raise ValueError(f"pair {pair.pair_id} does not use a later target")
        if pair.goal_target_vector_a is None or pair.goal_target_vector_b is None:
            raise ValueError(f"pair {pair.pair_id} lacks goal target vectors")
        targets = model.encoder.transform(
            np.asarray([pair.goal_target_vector_a, pair.goal_target_vector_b])
        )
        result.append(
            PairVectors(
                pair_id=pair.pair_id,
                pending_a=np.asarray(pair.evidence_a[:dimensions]),
                pending_b=np.asarray(pair.evidence_b[:dimensions]),
                response_a=hashed_text_vector(
                    pair.response_a.decode("utf-8", errors="replace"), dimensions
                ),
                response_b=hashed_text_vector(
                    pair.response_b.decode("utf-8", errors="replace"), dimensions
                ),
                target_a=targets[0],
                target_b=targets[1],
            )
        )
    return result


def _predict_clamped(model: RidgeMap, evidence: np.ndarray) -> np.ndarray:
    return _clamp_rows(model.predict(np.asarray(evidence, dtype=np.float64)))


def _pair_goal_predictions(
    pairs: list[CrossoverPair], model: FittedReranker
) -> dict[str, np.ndarray]:
    evidence = np.asarray(
        [item for pair in pairs for item in (pair.evidence_a, pair.evidence_b)]
    )
    pending = evidence[:, : model.feature_dimensions]
    return {
        "causal_full_history": _predict_clamped(model.full_goal_predictor, evidence),
        "causal_pending_user_only": _predict_clamped(
            model.pending_goal_predictor, pending
        ),
        "shuffled_predictor_target": _predict_clamped(
            model.shuffled_goal_predictor, evidence
        ),
    }


def _action_score_matrix(
    pair: PairVectors,
    model: FittedReranker,
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    action_model: RidgeMap | None = None,
    candidate_blind: bool = False,
    interaction: bool = True,
) -> dict[str, float]:
    pending = np.asarray(
        [pair.pending_a, pair.pending_a, pair.pending_b, pair.pending_b]
    )
    responses = np.asarray(
        [pair.response_a, pair.response_b, pair.response_a, pair.response_b]
    )
    goals = np.asarray([z_a, z_a, z_b, z_b])
    if candidate_blind:
        successor = _predict_clamped(model.candidate_blind_successor, pending)
    else:
        selected = model.action_successor if action_model is None else action_model
        successor = _predict_clamped(
            selected, action_features(pending, responses, interaction=interaction)
        )
    values = _cosine_rows(successor, goals)
    return {key: float(value) for key, value in zip(SCORE_KEYS, values, strict=True)}


def _semantic_base_scores(pair: PairVectors, model: FittedReranker) -> dict[str, float]:
    predicted = _predict_clamped(
        model.plausibility_predictor, np.asarray([pair.pending_a, pair.pending_b])
    )
    left = np.asarray([predicted[0], predicted[0], predicted[1], predicted[1]])
    right = np.asarray(
        [pair.response_a, pair.response_b, pair.response_a, pair.response_b]
    )
    values = _cosine_rows(left, right)
    return {key: float(value) for key, value in zip(SCORE_KEYS, values, strict=True)}


def _native_base_scores(pair: CrossoverPair, cortex: object) -> dict[str, float]:
    result: dict[str, float] = {}
    for context_name, context in (("a", pair.context_a), ("b", pair.context_b)):
        for response_name, response in (("a", pair.response_a), ("b", pair.response_b)):
            trace = _native_trace(cortex, context, response)
            targets = np.frombuffer(response, dtype=np.uint8)
            log_probability = np.log2(
                np.maximum(trace[np.arange(len(targets)), targets], 1e-300)
            )
            result[context_name + response_name] = float(np.mean(log_probability))
    return result


def symmetric_margin(scores: dict[str, float]) -> float:
    if set(scores) != set(SCORE_KEYS):
        raise ValueError("score mapping must contain aa, ab, ba, and bb")
    return (scores["aa"] - scores["ab"]) + (scores["bb"] - scores["ba"])


def _combine_scores(
    base: dict[str, float], alignment: dict[str, float], gain: float
) -> dict[str, float]:
    return {key: base[key] + gain * alignment[key] for key in SCORE_KEYS}


def summarize(
    values: np.ndarray, *, seed: int, resamples: int
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("cannot summarize an empty or non-vector value")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap = values[indices].mean(axis=1)
    observed = float(values.mean())
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(values)))
    null_means = (values[None, :] * signs).mean(axis=1)
    return {
        "observations": int(len(values)),
        "mean": observed,
        "median": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
        "mean_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "mean_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "one_sided_sign_flip_p": float(
            (1 + np.sum(null_means >= observed)) / (resamples + 1)
        ),
    }


def _matched_choice_outcomes(score_rows: list[dict[str, float]]) -> np.ndarray:
    """Return one tie-aware matched-candidate accuracy outcome per pair.

    The two context decisions inside a crossover pair are not independent.  Their
    correctness outcomes are therefore averaged before any bootstrap or sign-flip
    inference so resampling preserves the pair as the experimental unit.
    """

    outcomes = []
    for scores in score_rows:
        margins = np.asarray((scores["aa"] - scores["ab"], scores["bb"] - scores["ba"]))
        correct = (margins > 0.0) + 0.5 * (margins == 0.0)
        outcomes.append(float(np.mean(correct)))
    if not outcomes:
        raise ValueError("cannot score an empty candidate-pair list")
    return np.asarray(outcomes)


def _choice_accuracy(score_rows: list[dict[str, float]]) -> float:
    return float(np.mean(_matched_choice_outcomes(score_rows)))


def _representation_margin(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if predicted.shape != targets.shape or len(predicted) % 2:
        raise ValueError("paired predictions and targets must have equal even shape")
    similarity = normalize_rows(predicted) @ normalize_rows(targets).T
    rows = []
    for start in range(0, len(predicted), 2):
        rows.append(
            (similarity[start, start] - similarity[start, start + 1])
            + (similarity[start + 1, start + 1] - similarity[start + 1, start])
        )
    return np.asarray(rows)


def _condition_alignments(
    pairs: list[PairVectors],
    model: FittedReranker,
    predictions: dict[str, np.ndarray],
    *,
    seed: int,
    interaction: bool,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, np.ndarray]]:
    targets = np.asarray(
        [target for pair in pairs for target in (pair.target_a, pair.target_b)]
    )
    full = predictions["causal_full_history"]
    pending = predictions["causal_pending_user_only"]
    shuffled_predictor = predictions["shuffled_predictor_target"]
    rng = np.random.default_rng(seed + 8_291)
    episode_permutation = _sattolo_permutation(len(full), rng)
    shuffled_episode = full[episode_permutation]

    conditions: dict[str, list[dict[str, float]]] = {
        name: [] for name in CONDITION_NAMES
    }
    for index, pair in enumerate(pairs):
        offset = 2 * index
        goal_pairs = {
            "oracle_future_target": (targets[offset], targets[offset + 1]),
            "causal_full_history": (full[offset], full[offset + 1]),
            "causal_pending_user_only": (pending[offset], pending[offset + 1]),
            "swapped_oracle": (targets[offset + 1], targets[offset]),
            "swapped_causal_full_history": (full[offset + 1], full[offset]),
            "shuffled_predictor_target": (
                shuffled_predictor[offset],
                shuffled_predictor[offset + 1],
            ),
            "shuffled_action_target_mapping": (full[offset], full[offset + 1]),
            "shuffled_candidate_action_mapping": (
                full[offset],
                full[offset + 1],
            ),
            "shuffled_episode_goal": (
                shuffled_episode[offset],
                shuffled_episode[offset + 1],
            ),
            "candidate_blind_action": (full[offset], full[offset + 1]),
        }
        conditions["goal_off"].append({key: 0.0 for key in SCORE_KEYS})
        for name, (z_a, z_b) in goal_pairs.items():
            conditions[name].append(
                _action_score_matrix(
                    pair,
                    model,
                    z_a,
                    z_b,
                    action_model=(
                        model.shuffled_action_successor
                        if name == "shuffled_action_target_mapping"
                        else model.shuffled_candidate_action_successor
                        if name == "shuffled_candidate_action_mapping"
                        else None
                    ),
                    candidate_blind=name == "candidate_blind_action",
                    interaction=interaction,
                )
            )
    representation = {
        "oracle_future_target": _representation_margin(targets, targets),
        "causal_full_history": _representation_margin(full, targets),
        "causal_pending_user_only": _representation_margin(pending, targets),
        "shuffled_predictor_target": _representation_margin(
            shuffled_predictor, targets
        ),
        "shuffled_episode_goal": _representation_margin(shuffled_episode, targets),
        "full_history_minus_pending_user": _representation_margin(full, targets)
        - _representation_margin(pending, targets),
        "causal_minus_shuffled_predictor": _representation_margin(full, targets)
        - _representation_margin(shuffled_predictor, targets),
        "causal_minus_shuffled_episode_goal": _representation_margin(full, targets)
        - _representation_margin(shuffled_episode, targets),
    }
    return conditions, representation


def _evaluate_split(
    raw_pairs: list[CrossoverPair],
    model: FittedReranker,
    *,
    gain: float,
    native_gain: float | None,
    seed: int,
    resamples: int,
    interaction: bool,
    cortex: object | None,
    include_audit_rows: bool,
) -> dict[str, Any]:
    pairs = _pair_vectors(raw_pairs, model)
    predictions = _pair_goal_predictions(raw_pairs, model)
    alignments, representation = _condition_alignments(
        pairs, model, predictions, seed=seed, interaction=interaction
    )
    semantic_bases = [_semantic_base_scores(pair, model) for pair in pairs]
    native_bases = (
        None
        if cortex is None
        else [_native_base_scores(pair, cortex) for pair in raw_pairs]
    )

    def base_report(
        base_rows: list[dict[str, float]], *, base_name: str, base_gain: float
    ) -> dict[str, Any]:
        goal_off = np.asarray([symmetric_margin(row) for row in base_rows])
        goal_off_choices = _matched_choice_outcomes(base_rows)
        conditions: dict[str, Any] = {}
        audit: list[dict[str, Any]] = []
        max_algebra_error = 0.0
        alignment_margins: dict[str, np.ndarray] = {}
        for condition_index, name in enumerate(CONDITION_NAMES):
            alignment_rows = alignments[name]
            combined = [
                _combine_scores(
                    base, alignment, 0.0 if name == "goal_off" else base_gain
                )
                for base, alignment in zip(base_rows, alignment_rows, strict=True)
            ]
            raw = np.asarray([symmetric_margin(row) for row in combined])
            alignment_margin = np.asarray(
                [symmetric_margin(row) for row in alignment_rows]
            )
            alignment_margins[name] = alignment_margin
            adjusted = raw - goal_off
            expected = (
                np.zeros_like(adjusted)
                if name == "goal_off"
                else base_gain * alignment_margin
            )
            max_algebra_error = max(
                max_algebra_error, float(np.max(np.abs(adjusted - expected)))
            )
            choice_margins = np.asarray(
                [
                    value
                    for row in combined
                    for value in (row["aa"] - row["ab"], row["bb"] - row["ba"])
                ]
            )
            matched_choices = _matched_choice_outcomes(combined)
            matched_choice_delta = matched_choices - goal_off_choices
            conditions[name] = {
                "raw_symmetric_pair_margin": summarize(
                    raw,
                    seed=seed + condition_index * 211 + 1,
                    resamples=resamples,
                ),
                "adjusted_effect_vs_goal_off": summarize(
                    adjusted,
                    seed=seed + condition_index * 211 + 2,
                    resamples=resamples,
                ),
                "unscaled_action_alignment_margin": summarize(
                    alignment_margin,
                    seed=seed + condition_index * 211 + 3,
                    resamples=resamples,
                ),
                "within_context_candidate_margin": summarize(
                    choice_margins,
                    seed=seed + condition_index * 211 + 4,
                    resamples=resamples,
                ),
                "matched_candidate_accuracy_ties_half": _choice_accuracy(combined),
                "matched_candidate_accuracy_delta_vs_goal_off": summarize(
                    matched_choice_delta,
                    seed=seed + condition_index * 211 + 5,
                    resamples=resamples,
                ),
            }
            if include_audit_rows:
                for pair_index, scores in enumerate(combined):
                    audit.append(
                        {
                            "pair_id": pairs[pair_index].pair_id,
                            "condition": name,
                            "scores": scores,
                            "goal_off_scores": base_rows[pair_index],
                            "alignment_scores": alignment_rows[pair_index],
                            "raw_symmetric_pair_margin": float(raw[pair_index]),
                            "adjusted_effect_vs_goal_off": float(adjusted[pair_index]),
                            "matched_candidate_accuracy_ties_half": float(
                                matched_choices[pair_index]
                            ),
                            "matched_candidate_accuracy_delta_vs_goal_off": float(
                                matched_choice_delta[pair_index]
                            ),
                        }
                    )
        controls = (
            "causal_pending_user_only",
            "shuffled_predictor_target",
            "shuffled_action_target_mapping",
            "shuffled_candidate_action_mapping",
            "shuffled_episode_goal",
            "candidate_blind_action",
        )
        report: dict[str, Any] = {
            "base": base_name,
            "gain_selected_on_matching_dev_base": base_gain,
            "goal_off_pair_margin": summarize(
                goal_off, seed=seed + 70_001, resamples=resamples
            ),
            "conditions": conditions,
            "paired_causal_control_contrasts": {
                f"causal_full_history_minus_{name}": summarize(
                    alignment_margins["causal_full_history"] - alignment_margins[name],
                    seed=seed + index * 401 + 80_021,
                    resamples=resamples,
                )
                for index, name in enumerate(controls)
            },
            "max_abs_adjusted_equals_gain_times_alignment_error": max_algebra_error,
        }
        if include_audit_rows:
            report["exact_pair_audit"] = audit
        return report

    report = {
        "gain": gain,
        "representation_pair_discrimination": {
            name: summarize(values, seed=seed + index * 307 + 91, resamples=resamples)
            for index, (name, values) in enumerate(representation.items())
        },
        "primary_semantic_base": base_report(
            semantic_bases,
            base_name="pending_user_to_response_ridge_cosine",
            base_gain=gain,
        ),
    }
    if native_bases is not None:
        if native_gain is None:
            raise AssertionError("native evaluation requires a native-calibrated gain")
        report["auxiliary_native_base"] = base_report(
            native_bases,
            base_name="frozen_native_mean_log2_probability_per_byte",
            base_gain=native_gain,
        )
    return report


def _select_gain_from_scores(
    base_rows: list[dict[str, float]],
    alignment_rows: list[dict[str, float]],
    gains: tuple[float, ...],
) -> tuple[float, dict[str, dict[str, float]]]:
    curve: dict[str, dict[str, float]] = {}
    best: tuple[float, float, float] | None = None
    selected = gains[0]
    for gain in gains:
        scores = [
            _combine_scores(base, alignment, gain)
            for base, alignment in zip(base_rows, alignment_rows, strict=True)
        ]
        raw = np.asarray([symmetric_margin(row) for row in scores])
        goal_off = np.asarray([symmetric_margin(row) for row in base_rows])
        accuracy = _choice_accuracy(scores)
        adjusted_mean = float(np.mean(raw - goal_off))
        curve[str(gain)] = {
            "causal_matched_candidate_accuracy_ties_half": accuracy,
            "causal_raw_mean_symmetric_margin": float(np.mean(raw)),
            "causal_adjusted_mean_effect_vs_goal_off": adjusted_mean,
        }
        # Decision accuracy is primary. If several gains make the same choices,
        # retain the least intervention instead of rewarding score inflation.
        key = (accuracy, -gain, adjusted_mean)
        if best is None or key > best:
            best = key
            selected = gain
    return selected, curve


def _select_gains(
    raw_pairs: list[CrossoverPair],
    model: FittedReranker,
    *,
    gains: tuple[float, ...],
    seed: int,
    interaction: bool,
    cortex: object | None,
) -> tuple[
    float,
    dict[str, dict[str, float]],
    float | None,
    dict[str, dict[str, float]] | None,
]:
    """Select gains on dev separately for each incommensurate base scale."""

    pairs = _pair_vectors(raw_pairs, model)
    predictions = _pair_goal_predictions(raw_pairs, model)
    alignments, _ = _condition_alignments(
        pairs, model, predictions, seed=seed, interaction=interaction
    )
    causal = alignments["causal_full_history"]
    semantic_rows = [_semantic_base_scores(pair, model) for pair in pairs]
    semantic_gain, semantic_curve = _select_gain_from_scores(
        semantic_rows, causal, gains
    )
    if cortex is None:
        return semantic_gain, semantic_curve, None, None
    native_rows = [_native_base_scores(pair, cortex) for pair in raw_pairs]
    native_gain, native_curve = _select_gain_from_scores(native_rows, causal, gains)
    return semantic_gain, semantic_curve, native_gain, native_curve


def _select_gain(
    raw_pairs: list[CrossoverPair],
    model: FittedReranker,
    *,
    gains: tuple[float, ...],
    seed: int,
    interaction: bool,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compatibility wrapper for semantic-only development selection."""

    selected, curve, _, _ = _select_gains(
        raw_pairs,
        model,
        gains=gains,
        seed=seed,
        interaction=interaction,
        cortex=None,
    )
    return selected, curve


def _seed_gate(seed_report: dict[str, Any]) -> dict[str, bool]:
    representation = seed_report["test"]["representation_pair_discrimination"]
    primary = seed_report["test"]["primary_semantic_base"]
    conditions = primary["conditions"]
    contrasts = primary["paired_causal_control_contrasts"]

    def lower(summary: dict[str, Any]) -> float:
        return float(summary["mean_bootstrap_ci95_low"])

    causal_action = conditions["causal_full_history"][
        "unscaled_action_alignment_margin"
    ]
    return {
        "causal_representation_pair_discrimination_ci_above_zero": lower(
            representation["causal_full_history"]
        )
        > 0.0,
        "full_history_representation_increment_ci_above_zero": lower(
            representation["full_history_minus_pending_user"]
        )
        > 0.0,
        "causal_representation_exceeds_shuffled_predictor": lower(
            representation["causal_minus_shuffled_predictor"]
        )
        > 0.0,
        "causal_representation_exceeds_shuffled_episode_goal": lower(
            representation["causal_minus_shuffled_episode_goal"]
        )
        > 0.0,
        "oracle_action_alignment_ci_above_zero": lower(
            conditions["oracle_future_target"]["unscaled_action_alignment_margin"]
        )
        > 0.0,
        "causal_action_alignment_ci_above_zero": lower(causal_action) > 0.0,
        "causal_action_pair_win_fraction_above_0_60": float(
            causal_action["positive_fraction"]
        )
        > 0.60,
        "causal_matched_candidate_accuracy_delta_ci_above_zero": lower(
            conditions["causal_full_history"][
                "matched_candidate_accuracy_delta_vs_goal_off"
            ]
        )
        > 0.0,
        "causal_action_full_history_increment_ci_above_zero": lower(
            contrasts["causal_full_history_minus_causal_pending_user_only"]
        )
        > 0.0,
        "causal_action_exceeds_shuffled_predictor": lower(
            contrasts["causal_full_history_minus_shuffled_predictor_target"]
        )
        > 0.0,
        "causal_action_exceeds_shuffled_action_target": lower(
            contrasts["causal_full_history_minus_shuffled_action_target_mapping"]
        )
        > 0.0,
        "causal_action_exceeds_shuffled_candidate_mapping": lower(
            contrasts["causal_full_history_minus_shuffled_candidate_action_mapping"]
        )
        > 0.0,
        "causal_action_exceeds_shuffled_episode_goal": lower(
            contrasts["causal_full_history_minus_shuffled_episode_goal"]
        )
        > 0.0,
        "candidate_blind_action_margin_exactly_zero": float(
            conditions["candidate_blind_action"]["unscaled_action_alignment_margin"][
                "zero_fraction"
            ]
        )
        == 1.0,
        "goal_off_adjusted_effect_exactly_zero": float(
            conditions["goal_off"]["adjusted_effect_vs_goal_off"]["zero_fraction"]
        )
        == 1.0,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    target_kind = validate_dataset_semantics(args.dataset_dir)
    decays = tuple(float(value) for value in args.state_decays)
    gammas = tuple(float(value) for value in args.successor_gammas)
    examples = load_training_examples(
        args.dataset_dir,
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    dev_pairs = load_crossover_pairs(
        args.dataset_dir,
        "dev",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    test_pairs = load_crossover_pairs(
        args.dataset_dir,
        "test",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=gammas,
    )
    cortex = _load_native(args.checkpoint)
    common_encoder: ContinuousTargetEncoder | None = None
    fold_record: dict[str, Any] | None = None
    fit_directions: tuple[tuple[str, np.ndarray, np.ndarray], ...] | None = None
    if args.disjoint_supervised_fit:
        source_labels = _training_source_labels(args.dataset_dir, examples)
        fold_a, fold_b = _source_stratified_disjoint_halves(examples, source_labels)
        common_encoder = fit_target_encoder(
            np.asarray([row.goal_target_vector for row in examples]),
            latent_dimensions=args.latent_dimensions,
        )
        fit_directions = (
            ("goal_A_action_B", fold_a, fold_b),
            ("goal_B_action_A", fold_b, fold_a),
        )

        def fold_summary(indices: np.ndarray) -> dict[str, Any]:
            episode_ids = sorted(examples[int(index)].episode_id for index in indices)
            source_counts = {
                source: int(
                    sum(source_labels[int(index)] == source for index in indices)
                )
                for source in sorted(set(source_labels))
            }
            return {
                "episodes": len(indices),
                "source_counts": source_counts,
                "sorted_episode_ids_sha256": hashlib.sha256(
                    "\n".join(episode_ids).encode("utf-8")
                ).hexdigest(),
            }

        fold_record = {
            "assignment": (
                "within each source, sort by SHA-256 of the versioned protocol "
                "tag, source label, and episode id; alternate into A/B"
            ),
            "fold_A": fold_summary(fold_a),
            "fold_B": fold_summary(fold_b),
            "overlap_episodes": int(np.intersect1d(fold_a, fold_b).size),
            "union_episodes": int(len(np.union1d(fold_a, fold_b))),
        }

    run_reports = []
    for seed in args.seeds:
        directions = (
            (
                (
                    "shared_supervised_fit",
                    np.arange(len(examples)),
                    np.arange(len(examples)),
                ),
            )
            if fit_directions is None
            else fit_directions
        )
        for direction_name, goal_indices, action_indices in directions:
            model = (
                fit_reranker(
                    examples,
                    feature_dimensions=args.feature_dimensions,
                    latent_dimensions=args.latent_dimensions,
                    goal_ridge_alpha=args.goal_ridge_alpha,
                    action_ridge_alpha=args.action_ridge_alpha,
                    plausibility_ridge_alpha=args.plausibility_ridge_alpha,
                    seed=seed,
                    interaction=not args.no_interaction,
                )
                if common_encoder is None
                else _fit_partitioned_reranker(
                    examples,
                    encoder=common_encoder,
                    goal_indices=goal_indices,
                    action_indices=action_indices,
                    feature_dimensions=args.feature_dimensions,
                    goal_ridge_alpha=args.goal_ridge_alpha,
                    action_ridge_alpha=args.action_ridge_alpha,
                    plausibility_ridge_alpha=args.plausibility_ridge_alpha,
                    seed=seed,
                    interaction=not args.no_interaction,
                )
            )
            selected_gain, dev_curve, native_gain, native_dev_curve = _select_gains(
                dev_pairs,
                model,
                gains=tuple(args.gain_values),
                seed=seed,
                interaction=not args.no_interaction,
                cortex=cortex,
            )
            dev = _evaluate_split(
                dev_pairs,
                model,
                gain=selected_gain,
                native_gain=native_gain,
                seed=seed,
                resamples=args.resamples,
                interaction=not args.no_interaction,
                cortex=cortex,
                include_audit_rows=False,
            )
            test = _evaluate_split(
                test_pairs,
                model,
                gain=selected_gain,
                native_gain=native_gain,
                seed=seed,
                resamples=args.resamples,
                interaction=not args.no_interaction,
                cortex=cortex,
                include_audit_rows=args.include_pair_audit,
            )
            run_report: dict[str, Any] = {
                "seed": seed,
                "fit_direction": direction_name,
                "supervised_fit_counts": {
                    "goal_maps": len(goal_indices),
                    "action_and_plausibility_maps": len(action_indices),
                    "shared_episode_overlap": int(
                        np.intersect1d(goal_indices, action_indices).size
                    ),
                },
                "selected_gain_from_dev": selected_gain,
                "dev_gain_curve": dev_curve,
                "native_selected_gain_from_dev": native_gain,
                "native_dev_gain_curve": native_dev_curve,
                "target_encoder": {
                    "input_dimensions": int(len(model.encoder.mean)),
                    "latent_dimensions": int(len(model.encoder.components)),
                    "train_explained_variance_fraction": (
                        model.encoder.explained_variance_fraction
                    ),
                    "fit_episodes": len(examples),
                },
                "shuffle_invariants": {
                    "predictor_target_fixed_points": int(
                        np.sum(
                            model.predictor_target_permutation
                            == np.arange(len(model.predictor_target_permutation))
                        )
                    ),
                    "action_target_fixed_points": int(
                        np.sum(
                            model.action_target_permutation
                            == np.arange(len(model.action_target_permutation))
                        )
                    ),
                    "candidate_action_fixed_points": int(
                        np.sum(
                            model.candidate_action_permutation
                            == np.arange(len(model.candidate_action_permutation))
                        )
                    ),
                },
                "dev": dev,
                "test": test,
            }
            run_report["gate"] = _seed_gate(run_report)
            run_reports.append(run_report)

    report: dict[str, Any] = {
        "experiment": "continuous_shared_goal_trajectory_candidate_reranker",
        "status": "complete",
        "target_kind": target_kind,
        "mechanism": {
            "goal_state": (
                "train-only PCA of strictly-later targets; causal ridge inference; "
                "unit-ball clamp"
            ),
            "action_successor": (
                "ridge map from pending event, candidate response, normalized "
                "Hadamard interaction, and cosine"
                if not args.no_interaction
                else "ridge map from pending event and candidate response"
            ),
            "candidate_score": (
                "goal-off plausibility cosine + selected_gain * cosine("
                "predicted_action_successor, inferred_goal_state)"
            ),
            "primary_effect": (
                "condition symmetric A/A + B/B - A/B - B/A margin minus the "
                "same pair's goal-off margin"
            ),
        },
        "base": {
            "primary": "pending-user-to-response ridge cosine",
            "auxiliary": (
                None
                if args.checkpoint is None
                else "frozen native mean log2 probability per candidate byte"
            ),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        },
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "feature_dimensions": args.feature_dimensions,
            "latent_dimensions": args.latent_dimensions,
            "state_decays": list(decays),
            "successor_gammas": list(gammas),
            "goal_ridge_alpha": args.goal_ridge_alpha,
            "action_ridge_alpha": args.action_ridge_alpha,
            "plausibility_ridge_alpha": args.plausibility_ridge_alpha,
            "gain_values": args.gain_values,
            "seeds": args.seeds,
            "resamples": args.resamples,
            "bounded_interaction": not args.no_interaction,
            "include_pair_audit": args.include_pair_audit,
            "disjoint_supervised_fit": args.disjoint_supervised_fit,
        },
        "counts": {
            "train_episodes": len(examples),
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
        },
        "selection_policy": (
            "maximize causal full-history matched-candidate accuracy on dev; "
            "then prefer smaller gain; then adjusted dev margin; never inspect test"
        ),
        "conditions": list(CONDITION_NAMES),
        "claim_limits": [
            (
                "Natural dialogue measures long-context future-trajectory "
                "compatibility; it does not uniquely identify an inferred goal, "
                "intent, or preference."
            ),
            (
                "The text adapter is a signed-hash lexical vector; modality "
                "neutrality applies to the dense-vector mechanism, not to this "
                "particular adapter's semantic quality."
            ),
            (
                "Oracle future targets are diagnostic and unavailable online; only "
                "causal_full_history is a deployable goal estimate."
            ),
            (
                "The mechanism reranks supplied whole candidates and cannot improve "
                "content absent from the generator's candidate set."
            ),
            (
                "Gain calibrates dimensionless cosine alignment against the chosen "
                "base score and is frozen from development data."
            ),
            (
                "Exact-hash split filtering does not exclude paraphrased templates "
                "or latent user-family overlap."
            ),
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    if args.disjoint_supervised_fit:
        if fold_record is None:
            raise AssertionError("disjoint fit did not record its folds")
        report["disjoint_supervised_fit"] = {
            "target_encoder": "one common PCA fit on all training targets only",
            "supervised_maps": (
                "goal maps fit on one fold; action, candidate-blind, and "
                "plausibility maps fit on the disjoint fold; then roles swap"
            ),
            "folds": fold_record,
            "run_semantics": (
                "A-to-B and B-to-A reuse the same dev/test pairs and are reported "
                "separately; they are reciprocal falsification checks, not "
                "independent samples and must not be pooled"
            ),
            "runs": run_reports,
            "all_run_gates": {
                key: all(bool(row["gate"][key]) for row in run_reports)
                for key in run_reports[0]["gate"]
            },
        }
        report["seed_semantics"] = (
            "Within each fit direction, non-control maps are deterministic and "
            "identical across seed rows. Seeds vary shuffled controls and inferential "
            "resampling; reciprocal fit directions are not independent test sets."
        )
    else:
        report["seeds"] = run_reports
        report["all_seed_gates"] = {
            key: all(bool(row["gate"][key]) for row in run_reports)
            for key in run_reports[0]["gate"]
        }
        report["seed_semantics"] = (
            "The target encoder and non-control ridge maps are deterministic and "
            "identical across seed rows. Seeds vary Sattolo control mappings, "
            "evaluation episode shuffles, and bootstrap/sign-flip draws; they are "
            "not independent model fits."
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--feature-dimensions", type=int, default=96)
    parser.add_argument("--latent-dimensions", type=int, default=32)
    parser.add_argument("--state-decays", type=float, nargs="+", default=DEFAULT_DECAYS)
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--goal-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--action-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--plausibility-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--gain-values", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--resamples", type=int, default=4_000)
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--include-pair-audit", action="store_true")
    parser.add_argument(
        "--disjoint-supervised-fit",
        action="store_true",
        help=(
            "fit goal and action/plausibility maps on disjoint source-stratified "
            "halves and report both reciprocal directions"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.feature_dimensions < 8:
        raise ValueError("feature_dimensions must be at least eight")
    if args.latent_dimensions < 1:
        raise ValueError("latent_dimensions must be positive")
    if any(not 0.0 < value < 1.0 for value in args.state_decays):
        raise ValueError("state decays must lie in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in args.successor_gammas):
        raise ValueError("successor gammas must lie in (0, 1)")
    if any(
        value <= 0.0
        for value in (
            args.goal_ridge_alpha,
            args.action_ridge_alpha,
            args.plausibility_ridge_alpha,
        )
    ):
        raise ValueError("ridge penalties must be positive")
    if not args.gain_values or any(value < 0.0 for value in args.gain_values):
        raise ValueError("gain values must be a nonempty non-negative list")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")
    report = run_ablation(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
