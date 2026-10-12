"""Ablate a variable-delay, modality-neutral predictive-state bank.

The synthetic task is deliberately about trajectory rather than token identity.  A
cue consists of two dense observations.  Reversing their order preserves the exact
set of observable content while changing the consequence at the end of the
episode.  The delay is sampled independently for every episode and is never given
to a learner.  Evaluation additionally replaces the cue's content component with
novel random content while preserving its temporal direction.

The proposed mechanism is a small self-supervised predictive code.  It clusters
observed episode endpoints, learns which code a transition predicts, centers that
forecast against its empirical prior, and stores the evidence in a bank of leaky
timescales.  It receives only dense observations and episode boundaries: no
trajectory labels, content IDs, modality IDs, delays, or generator coordinates.

This is a mechanism screen, not evidence of conversational quality.  The endpoint
is explicit because it plays the role of a candidate response boundary; whether a
useful endpoint code can be inferred in natural conversation is a separate test.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

DEFAULT_HALF_LIVES = (12.0, 48.0, 192.0, 768.0)
DEFAULT_FIXED_LAGS = (16, 48, 96, 160)


@dataclass(frozen=True)
class Episode:
    """The complete learner-facing record; it intentionally has no hidden labels."""

    events: np.ndarray


@dataclass(frozen=True)
class IdentityAudit:
    """Generator-only metadata used for controls and the 2x2 evaluation."""

    group: int
    content: str
    direction: int
    delay: int


@dataclass(frozen=True)
class SyntheticWorld:
    """Parameters retained by the generator, never passed to fitted learners."""

    mixing: np.ndarray
    trajectory_axis: np.ndarray
    train_content_bases: np.ndarray
    query: np.ndarray
    consequences: np.ndarray


@dataclass(frozen=True)
class RidgeMap:
    weights: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values - self.x_mean) @ self.weights + self.y_mean


@dataclass(frozen=True)
class EndpointCodebook:
    centers: np.ndarray

    def assign(self, endpoints: np.ndarray) -> np.ndarray:
        similarity = _normalize_rows(endpoints) @ _normalize_rows(self.centers).T
        return np.argmax(similarity, axis=1)


@dataclass(frozen=True)
class PredictiveCode:
    """A self-supervised transition-to-future-code map."""

    classifier: RidgeMap
    prior: np.ndarray
    temperature: float

    def evidence(self, transition: np.ndarray) -> np.ndarray:
        logits = self.classifier.predict(transition[None, :]) / self.temperature
        return _softmax(logits)[0] - self.prior


@dataclass(frozen=True)
class RandomCode:
    """Capacity-matched, future-agnostic code used as a control."""

    projection: np.ndarray

    def evidence(self, transition: np.ndarray) -> np.ndarray:
        return np.tanh(self.projection @ transition)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _fit_ridge(
    evidence: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> RidgeMap:
    if alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    gram.flat[:: len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, centered_x.T @ centered_y)
    return RidgeMap(weights=weights, x_mean=x_mean, y_mean=y_mean)


def make_world(
    *,
    dimensions: int,
    train_contents: int,
    seed: int,
) -> SyntheticWorld:
    """Create a hidden coordinate system and expose only its dense rotation."""
    if dimensions < 8:
        raise ValueError("dimensions must be at least eight")
    if train_contents < 2:
        raise ValueError("train_contents must be at least two")
    rng = np.random.default_rng(seed)
    mixing, _ = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))

    latent_trajectory = np.zeros(dimensions)
    latent_trajectory[-4] = 1.0
    trajectory_axis = mixing @ latent_trajectory

    latent_contents = rng.normal(size=(train_contents, dimensions))
    latent_contents[:, -4:] = 0.0
    latent_contents = _normalize_rows(latent_contents) * 0.8
    train_content_bases = latent_contents @ mixing.T

    latent_query = np.zeros(dimensions)
    latent_query[-3] = 1.0
    query = mixing @ latent_query
    latent_consequences = np.zeros((2, dimensions))
    latent_consequences[0, -2] = 1.0
    latent_consequences[1, -1] = 1.0
    consequences = latent_consequences @ mixing.T
    return SyntheticWorld(
        mixing=mixing,
        trajectory_axis=trajectory_axis,
        train_content_bases=train_content_bases,
        query=query,
        consequences=consequences,
    )


def _novel_content(world: SyntheticWorld, rng: np.random.Generator) -> np.ndarray:
    latent = rng.normal(size=len(world.query))
    latent[-4:] = 0.0
    latent /= max(float(np.linalg.norm(latent)), 1e-12)
    return (latent * 0.8) @ world.mixing.T


def _nuisance_events(
    world: SyntheticWorld,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate content variation with exactly zero hidden trajectory component."""
    dimensions = len(world.query)
    latent = rng.normal(size=(count, dimensions))
    latent[:, -4:] = 0.0
    latent = _normalize_rows(latent)
    # Sticky nuisance motion makes immediate context useful in the body while
    # remaining independent of the endpoint trajectory.
    for index in range(1, count):
        latent[index] = 0.72 * latent[index - 1] + 0.28 * latent[index]
        latent[index] /= max(float(np.linalg.norm(latent[index])), 1e-12)
    return latent @ world.mixing.T


def make_episode(
    world: SyntheticWorld,
    *,
    content_base: np.ndarray,
    direction: int,
    delay: int,
    nuisance: np.ndarray,
    local_window: int,
    cue_amplitude: float = 1.25,
) -> Episode:
    """Construct an episode whose cue content set is invariant to direction."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    if delay <= local_window + 1:
        raise ValueError("delay is too short for the requested local window")
    nuisance_count = delay - local_window - 1
    if len(nuisance) != nuisance_count:
        raise ValueError("nuisance length does not match the requested delay")
    displacement = cue_amplitude * world.trajectory_axis
    first = content_base - direction * displacement
    second = content_base + direction * displacement
    consequence = world.consequences[int(direction > 0)]
    events = np.vstack(
        (
            first,
            second,
            nuisance,
            np.repeat(world.query[None, :], local_window, axis=0),
            consequence,
        )
    )
    return Episode(events=events)


def generate_training_episodes(
    world: SyntheticWorld,
    *,
    repeats_per_content: int,
    min_delay: int,
    max_delay: int,
    local_window: int,
    seed: int,
) -> list[Episode]:
    """Generate direction-balanced pairs with shared content, delay, and nuisance."""
    if repeats_per_content <= 0:
        raise ValueError("repeats_per_content must be positive")
    if min_delay > max_delay:
        raise ValueError("minimum delay must not exceed maximum delay")
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for content in world.train_content_bases:
        for _ in range(repeats_per_content):
            delay = int(rng.integers(min_delay, max_delay + 1))
            nuisance = _nuisance_events(world, delay - local_window - 1, rng)
            for direction in (-1, 1):
                episodes.append(
                    make_episode(
                        world,
                        content_base=content,
                        direction=direction,
                        delay=delay,
                        nuisance=nuisance,
                        local_window=local_window,
                    )
                )
    rng.shuffle(episodes)
    return episodes


def generate_identity_episodes(
    world: SyntheticWorld,
    *,
    groups: int,
    min_delay: int,
    max_delay: int,
    local_window: int,
    seed: int,
) -> tuple[list[Episode], list[IdentityAudit]]:
    """Generate familiar/novel content crossed with both trajectory directions."""
    if groups <= 0:
        raise ValueError("identity groups must be positive")
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    audits: list[IdentityAudit] = []
    for group in range(groups):
        familiar = world.train_content_bases[
            int(rng.integers(len(world.train_content_bases)))
        ]
        novel = _novel_content(world, rng)
        delay = int(rng.integers(min_delay, max_delay + 1))
        nuisance = _nuisance_events(world, delay - local_window - 1, rng)
        for content_name, content in (("familiar", familiar), ("novel", novel)):
            for direction in (-1, 1):
                episodes.append(
                    make_episode(
                        world,
                        content_base=content,
                        direction=direction,
                        delay=delay,
                        nuisance=nuisance,
                        local_window=local_window,
                    )
                )
                audits.append(
                    IdentityAudit(
                        group=group,
                        content=content_name,
                        direction=direction,
                        delay=delay,
                    )
                )
    return episodes, audits


def shuffled_consequence_control(
    episodes: Sequence[Episode],
    consequences: np.ndarray,
    *,
    seed: int,
) -> list[Episode]:
    """Break cue/consequence dependence while preserving exact endpoint marginals."""
    if len(episodes) % 2:
        raise ValueError("balanced consequence shuffling needs an even episode count")
    rng = np.random.default_rng(seed)
    replacements = np.resize(np.arange(2, dtype=np.int64), len(episodes))
    # Retry a few times to avoid an accidental residual mapping in a finite sample.
    original = np.argmax(
        _normalize_rows(np.asarray([row.events[-1] for row in episodes]))
        @ _normalize_rows(consequences).T,
        axis=1,
    )
    for _ in range(100):
        rng.shuffle(replacements)
        association = abs(float(np.mean((2 * original - 1) * (2 * replacements - 1))))
        if association <= 0.02:
            break
    shuffled: list[Episode] = []
    for episode, replacement in zip(episodes, replacements, strict=True):
        events = episode.events.copy()
        events[-1] = consequences[int(replacement)]
        shuffled.append(Episode(events=events))
    return shuffled


def fit_endpoint_codebook(
    episodes: Sequence[Episode],
    *,
    codes: int,
    iterations: int,
    seed: int,
) -> EndpointCodebook:
    """Cluster raw observed endpoints without generator labels."""
    endpoints = _normalize_rows(
        np.asarray([episode.events[-1] for episode in episodes])
    )
    if not 1 <= codes <= len(endpoints):
        raise ValueError("codes must lie in [1, number of episodes]")
    if codes == 1:
        return EndpointCodebook(centers=np.mean(endpoints, axis=0, keepdims=True))
    rng = np.random.default_rng(seed)
    centers = [endpoints[int(rng.integers(len(endpoints)))]]
    while len(centers) < codes:
        similarity = endpoints @ np.asarray(centers).T
        distance = np.maximum(0.0, 1.0 - np.max(similarity, axis=1))
        total = float(np.sum(distance))
        index = (
            int(rng.integers(len(endpoints)))
            if total <= 1e-12
            else int(rng.choice(len(endpoints), p=distance / total))
        )
        centers.append(endpoints[index])
    matrix = np.asarray(centers)
    for _ in range(iterations):
        labels = np.argmax(endpoints @ _normalize_rows(matrix).T, axis=1)
        updated = matrix.copy()
        for code in range(codes):
            members = endpoints[labels == code]
            if len(members):
                updated[code] = np.mean(members, axis=0)
        updated = _normalize_rows(updated)
        if np.allclose(updated, matrix, atol=1e-9):
            matrix = updated
            break
        matrix = updated
    return EndpointCodebook(centers=matrix)


def _transition_training_rows(
    episodes: Sequence[Episode],
    endpoint_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    evidence: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for episode, label in zip(episodes, endpoint_labels, strict=True):
        prefix = episode.events[:-1]
        transition = prefix[1:] - prefix[:-1]
        evidence.append(transition)
        labels.append(np.full(len(transition), int(label), dtype=np.int64))
    return np.vstack(evidence), np.concatenate(labels)


def fit_predictive_code(
    episodes: Sequence[Episode],
    codebook: EndpointCodebook,
    *,
    ridge_alpha: float,
    temperature: float,
) -> PredictiveCode:
    """Predict a future endpoint code from each observed transition."""
    if temperature <= 0.0:
        raise ValueError("code temperature must be positive")
    endpoint = np.asarray([episode.events[-1] for episode in episodes])
    labels = codebook.assign(endpoint)
    transitions, repeated_labels = _transition_training_rows(episodes, labels)
    codes = len(codebook.centers)
    target = np.eye(codes, dtype=np.float64)[repeated_labels]
    classifier = _fit_ridge(transitions, target, ridge_alpha)
    counts = np.bincount(labels, minlength=codes).astype(np.float64)
    prior = counts / np.sum(counts)
    return PredictiveCode(
        classifier=classifier,
        prior=prior,
        temperature=temperature,
    )


def local_features(prefix: np.ndarray, window: int) -> np.ndarray:
    dimensions = prefix.shape[1]
    selected = prefix[-window:]
    if len(selected) < window:
        selected = np.vstack((np.zeros((window - len(selected), dimensions)), selected))
    return selected.reshape(-1)


def slow_theme_features(prefix: np.ndarray, window: int) -> np.ndarray:
    return np.concatenate((local_features(prefix, window), np.mean(prefix, axis=0)))


def fixed_lag_features(prefix: np.ndarray, window: int, lag: int) -> np.ndarray:
    endpoint = len(prefix) - lag
    dimensions = prefix.shape[1]
    transition = (
        prefix[endpoint] - prefix[endpoint - 1]
        if 0 < endpoint < len(prefix)
        else np.zeros(dimensions)
    )
    return np.concatenate((local_features(prefix, window), transition))


def bank_features(
    prefix: np.ndarray,
    window: int,
    encoder: PredictiveCode | RandomCode,
    decays: np.ndarray,
    *,
    reset_fraction: float | None = None,
) -> np.ndarray:
    code_dimensions = len(encoder.evidence(prefix[1] - prefix[0]))
    states = np.zeros((len(decays), code_dimensions), dtype=np.float64)
    reset_index = (
        max(2, int(len(prefix) * reset_fraction))
        if reset_fraction is not None
        else None
    )
    for endpoint in range(1, len(prefix)):
        if reset_index is not None and endpoint == reset_index:
            states.fill(0.0)
        evidence = encoder.evidence(prefix[endpoint] - prefix[endpoint - 1])
        states *= decays[:, None]
        states += evidence[None, :]
    return np.concatenate((local_features(prefix, window), states.reshape(-1)))


def capacity_matched_recent_features(
    prefix: np.ndarray,
    window: int,
    bank_dimensions: int,
) -> np.ndarray:
    """Match feature count but repeat only current, nonpersistent evidence."""
    transition = prefix[-1] - prefix[-2]
    repeated = np.resize(transition, bank_dimensions)
    return np.concatenate((local_features(prefix, window), repeated))


def collapsed_code_features(
    prefix: np.ndarray,
    window: int,
    bank_dimensions: int,
) -> np.ndarray:
    """Represent the centered evidence from a one-code codebook: identically zero."""
    return np.concatenate(
        (local_features(prefix, window), np.zeros(bank_dimensions, dtype=np.float64))
    )


FeatureFunction = Callable[[np.ndarray], np.ndarray]


def _fit_endpoint_readout(
    episodes: Sequence[Episode],
    feature: FeatureFunction,
    *,
    ridge_alpha: float,
) -> RidgeMap:
    evidence = np.asarray([feature(episode.events[:-1]) for episode in episodes])
    endpoint = np.asarray([episode.events[-1] for episode in episodes])
    return _fit_ridge(evidence, endpoint, ridge_alpha)


def _predict_endpoints(
    model: RidgeMap,
    episodes: Sequence[Episode],
    feature: FeatureFunction,
) -> np.ndarray:
    evidence = np.asarray([feature(episode.events[:-1]) for episode in episodes])
    return model.predict(evidence)


def _candidate_predictions(
    prediction: np.ndarray,
    consequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _normalize_rows(prediction) @ _normalize_rows(consequences).T
    return np.argmax(scores, axis=1), scores


def endpoint_metrics(
    prediction: np.ndarray,
    episodes: Sequence[Episode],
    consequences: np.ndarray,
) -> dict[str, float | int]:
    predicted, scores = _candidate_predictions(prediction, consequences)
    target_vectors = np.asarray([episode.events[-1] for episode in episodes])
    target = np.argmax(
        _normalize_rows(target_vectors) @ _normalize_rows(consequences).T,
        axis=1,
    )
    signed_margin = (
        scores[np.arange(len(target)), target]
        - scores[np.arange(len(target)), 1 - target]
    )
    probability = _softmax(scores * 4.0)
    return {
        "episodes": len(episodes),
        "accuracy": float(np.mean(predicted == target)),
        "cross_entropy_bits": float(
            -np.mean(
                np.log2(np.maximum(probability[np.arange(len(target)), target], 1e-12))
            )
        ),
        "mean_signed_margin": float(np.mean(signed_margin)),
    }


def identity_metrics(
    prediction: np.ndarray,
    consequences: np.ndarray,
    audits: Sequence[IdentityAudit],
) -> dict[str, object]:
    predicted, _ = _candidate_predictions(prediction, consequences)
    lookup = {
        (audit.group, audit.content, audit.direction): int(value)
        for audit, value in zip(audits, predicted, strict=True)
    }
    groups = sorted({audit.group for audit in audits})
    familiar_flip: list[bool] = []
    novel_flip: list[bool] = []
    negative_preserved: list[bool] = []
    positive_preserved: list[bool] = []
    joint: list[bool] = []
    for group in groups:
        familiar_negative = lookup[(group, "familiar", -1)]
        familiar_positive = lookup[(group, "familiar", 1)]
        novel_negative = lookup[(group, "novel", -1)]
        novel_positive = lookup[(group, "novel", 1)]
        checks = (
            familiar_negative != familiar_positive,
            novel_negative != novel_positive,
            familiar_negative == novel_negative,
            familiar_positive == novel_positive,
        )
        familiar_flip.append(checks[0])
        novel_flip.append(checks[1])
        negative_preserved.append(checks[2])
        positive_preserved.append(checks[3])
        joint.append(all(checks))
    quadrant_accuracy: dict[str, float] = {}
    for content in ("familiar", "novel"):
        for direction in (-1, 1):
            rows = [
                index
                for index, audit in enumerate(audits)
                if audit.content == content and audit.direction == direction
            ]
            correct_label = int(direction > 0)
            quadrant_accuracy[f"{content}_{direction:+d}"] = float(
                np.mean(predicted[rows] == correct_label)
            )
    return {
        "groups": len(groups),
        "same_content_opposite_trajectory_flip_rate": {
            "familiar": float(np.mean(familiar_flip)),
            "novel": float(np.mean(novel_flip)),
        },
        "same_trajectory_novel_content_preservation_rate": {
            "negative": float(np.mean(negative_preserved)),
            "positive": float(np.mean(positive_preserved)),
        },
        "joint_structural_pass_rate": float(np.mean(joint)),
        "quadrant_accuracy": quadrant_accuracy,
    }


def _cue_content_set_error(
    episodes: Sequence[Episode], audits: Sequence[IdentityAudit]
) -> float:
    indexed = {
        (audit.group, audit.content, audit.direction): episode
        for episode, audit in zip(episodes, audits, strict=True)
    }
    largest = 0.0
    for group in {audit.group for audit in audits}:
        for content in ("familiar", "novel"):
            negative = indexed[(group, content, -1)].events[:2]
            positive = indexed[(group, content, 1)].events[:2]
            largest = max(
                largest,
                float(np.max(np.abs(negative[0] - positive[1]))),
                float(np.max(np.abs(negative[1] - positive[0]))),
            )
    return largest


def _fit_and_evaluate(
    train: Sequence[Episode],
    evaluation: Sequence[Episode],
    feature: FeatureFunction,
    eval_feature: FeatureFunction,
    consequences: np.ndarray,
    audits: Sequence[IdentityAudit],
    *,
    ridge_alpha: float,
) -> tuple[dict[str, float | int], dict[str, object], np.ndarray]:
    model = _fit_endpoint_readout(train, feature, ridge_alpha=ridge_alpha)
    prediction = _predict_endpoints(model, evaluation, eval_feature)
    return (
        endpoint_metrics(prediction, evaluation, consequences),
        identity_metrics(prediction, consequences, audits),
        prediction,
    )


def run_ablation(
    *,
    dimensions: int = 24,
    train_contents: int = 48,
    repeats_per_content: int = 5,
    identity_groups: int = 96,
    min_delay: int = 24,
    max_delay: int = 192,
    local_window: int = 4,
    half_lives: Sequence[float] = DEFAULT_HALF_LIVES,
    fixed_lags: Sequence[int] = DEFAULT_FIXED_LAGS,
    ridge_alpha: float = 1.0,
    code_temperature: float = 0.18,
    seed: int = 149,
) -> dict[str, object]:
    """Run the proposed bank, identity test, and all requested controls."""
    started = time.perf_counter()
    if min_delay <= local_window + 1:
        raise ValueError("minimum delay must exceed local_window + 1")
    if max_delay <= min_delay:
        raise ValueError("maximum delay must exceed minimum delay")
    if not half_lives or any(value <= 0.0 for value in half_lives):
        raise ValueError("half lives must be positive")
    if not fixed_lags or any(value <= 0 for value in fixed_lags):
        raise ValueError("fixed lags must be positive")

    world = make_world(
        dimensions=dimensions,
        train_contents=train_contents,
        seed=seed,
    )
    train = generate_training_episodes(
        world,
        repeats_per_content=repeats_per_content,
        min_delay=min_delay,
        max_delay=max_delay,
        local_window=local_window,
        seed=seed + 1,
    )
    evaluation, audits = generate_identity_episodes(
        world,
        groups=identity_groups,
        min_delay=min_delay,
        max_delay=max_delay,
        local_window=local_window,
        seed=seed + 2,
    )
    shuffled_train = shuffled_consequence_control(
        train, world.consequences, seed=seed + 3
    )
    codebook = fit_endpoint_codebook(train, codes=2, iterations=20, seed=seed + 4)
    predictive_code = fit_predictive_code(
        train,
        codebook,
        ridge_alpha=ridge_alpha,
        temperature=code_temperature,
    )
    shuffled_codebook = fit_endpoint_codebook(
        shuffled_train, codes=2, iterations=20, seed=seed + 5
    )
    shuffled_code = fit_predictive_code(
        shuffled_train,
        shuffled_codebook,
        ridge_alpha=ridge_alpha,
        temperature=code_temperature,
    )
    decays = np.asarray(
        [math.exp(math.log(0.5) / value) for value in half_lives],
        dtype=np.float64,
    )
    random_projection = np.random.default_rng(seed + 6).normal(size=(2, dimensions))
    random_projection = _normalize_rows(random_projection) * 2.0
    random_code = RandomCode(random_projection)

    local = partial(local_features, window=local_window)
    theme = partial(slow_theme_features, window=local_window)
    bank = partial(
        bank_features,
        window=local_window,
        encoder=predictive_code,
        decays=decays,
    )
    reset_bank = partial(
        bank_features,
        window=local_window,
        encoder=predictive_code,
        decays=decays,
        reset_fraction=0.5,
    )
    shuffled_bank = partial(
        bank_features,
        window=local_window,
        encoder=shuffled_code,
        decays=decays,
    )
    random_bank = partial(
        bank_features,
        window=local_window,
        encoder=random_code,
        decays=decays,
    )
    bank_dimensions = len(decays) * len(codebook.centers)
    matched_recent = partial(
        capacity_matched_recent_features,
        window=local_window,
        bank_dimensions=bank_dimensions,
    )
    collapsed_code = partial(
        collapsed_code_features,
        window=local_window,
        bank_dimensions=bank_dimensions,
    )

    conditions: dict[str, dict[str, object]] = {}

    def add_condition(
        name: str,
        training: Sequence[Episode],
        train_feature: FeatureFunction,
        eval_feature: FeatureFunction | None = None,
    ) -> np.ndarray:
        metrics, identity, prediction = _fit_and_evaluate(
            training,
            evaluation,
            train_feature,
            eval_feature or train_feature,
            world.consequences,
            audits,
            ridge_alpha=ridge_alpha,
        )
        conditions[name] = {"endpoint": metrics, "identity_2x2": identity}
        return prediction

    add_condition("local", train, local)
    add_condition("slow_theme", train, theme)
    bank_prediction = add_condition("multi_timescale_predictive_bank", train, bank)
    add_condition("mid_episode_reset", train, bank, reset_bank)
    add_condition(
        "shuffled_cue_consequence",
        shuffled_train,
        shuffled_bank,
    )
    add_condition("predictive_code_shuffled_only", train, shuffled_bank)
    add_condition("capacity_matched_random_code", train, random_bank)
    add_condition("capacity_matched_recent_only", train, matched_recent)
    add_condition("one_code_collapsed_capacity_padded", train, collapsed_code)

    fixed_lag_results: dict[str, dict[str, object]] = {}
    for lag in fixed_lags:
        lag_feature = partial(
            fixed_lag_features,
            window=local_window,
            lag=lag,
        )
        metrics, identity, _ = _fit_and_evaluate(
            train,
            evaluation,
            lag_feature,
            lag_feature,
            world.consequences,
            audits,
            ridge_alpha=ridge_alpha,
        )
        fixed_lag_results[str(lag)] = {
            "endpoint": metrics,
            "identity_2x2": identity,
        }

    single_timescale: dict[str, dict[str, object]] = {}
    for half_life, decay in zip(half_lives, decays, strict=True):
        one_decay = np.asarray([decay])
        one_bank = partial(
            bank_features,
            window=local_window,
            encoder=predictive_code,
            decays=one_decay,
        )
        metrics, identity, _ = _fit_and_evaluate(
            train,
            evaluation,
            one_bank,
            one_bank,
            world.consequences,
            audits,
            ridge_alpha=ridge_alpha,
        )
        single_timescale[f"{half_life:g}"] = {
            "endpoint": metrics,
            "identity_2x2": identity,
        }

    endpoint_labels = codebook.assign(
        np.asarray([episode.events[-1] for episode in train])
    )
    occupancy = np.bincount(endpoint_labels, minlength=2)
    delays = np.asarray([audit.delay for audit in audits], dtype=np.int64)
    bank_identity = identity_metrics(bank_prediction, world.consequences, audits)
    return {
        "schema_version": 1,
        "experiment": "variable_delay_predictive_state_bank",
        "claim_scope": "synthetic mechanism screen; not a language-quality result",
        "learner_observations": {
            "dense_event_vectors": True,
            "episode_boundaries": True,
            "trajectory_labels": False,
            "content_ids": False,
            "modality_ids": False,
            "delay_or_target_horizon": False,
            "hand_labeled_success": False,
        },
        "task": {
            "train_episodes": len(train),
            "identity_evaluation_episodes": len(evaluation),
            "identity_groups": identity_groups,
            "delay_min": int(np.min(delays)),
            "delay_max": int(np.max(delays)),
            "distinct_delays": int(len(np.unique(delays))),
            "cue_content_set_max_absolute_error": _cue_content_set_error(
                evaluation, audits
            ),
            "chance_accuracy": 0.5,
        },
        "config": {
            "dimensions": dimensions,
            "train_contents": train_contents,
            "repeats_per_content": repeats_per_content,
            "identity_groups": identity_groups,
            "min_delay": min_delay,
            "max_delay": max_delay,
            "local_window": local_window,
            "half_lives": list(half_lives),
            "fixed_lags": list(fixed_lags),
            "ridge_alpha": ridge_alpha,
            "code_temperature": code_temperature,
            "seed": seed,
        },
        "predictive_code": {
            "codes": len(codebook.centers),
            "occupancy": occupancy.tolist(),
            "largest_code_fraction": float(np.max(occupancy) / np.sum(occupancy)),
            "state_dimensions": bank_dimensions,
        },
        "conditions": conditions,
        "single_fixed_lags": fixed_lag_results,
        "single_timescale_predictive_banks": single_timescale,
        "gates": {
            "bank_beats_local": (
                float(
                    conditions["multi_timescale_predictive_bank"]["endpoint"][
                        "accuracy"
                    ]
                )
                > float(conditions["local"]["endpoint"]["accuracy"])
            ),
            "bank_joint_identity_above_90_percent": (
                float(bank_identity["joint_structural_pass_rate"]) >= 0.9
            ),
            "reset_removes_identity": (
                float(
                    conditions["mid_episode_reset"]["identity_2x2"][
                        "joint_structural_pass_rate"
                    ]
                )
                < 0.6
            ),
            "shuffle_removes_identity": (
                float(
                    conditions["shuffled_cue_consequence"]["identity_2x2"][
                        "joint_structural_pass_rate"
                    ]
                )
                < 0.6
            ),
            "predictive_code_beats_capacity_matched_random_code": (
                float(
                    conditions["multi_timescale_predictive_bank"]["endpoint"][
                        "cross_entropy_bits"
                    ]
                )
                < float(
                    conditions["capacity_matched_random_code"]["endpoint"][
                        "cross_entropy_bits"
                    ]
                )
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dimensions", type=int, default=24)
    parser.add_argument("--train-contents", type=int, default=48)
    parser.add_argument("--repeats-per-content", type=int, default=5)
    parser.add_argument("--identity-groups", type=int, default=96)
    parser.add_argument("--min-delay", type=int, default=24)
    parser.add_argument("--max-delay", type=int, default=192)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument(
        "--half-lives", type=float, nargs="+", default=DEFAULT_HALF_LIVES
    )
    parser.add_argument("--fixed-lags", type=int, nargs="+", default=DEFAULT_FIXED_LAGS)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--code-temperature", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=149)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_ablation(
        dimensions=args.dimensions,
        train_contents=args.train_contents,
        repeats_per_content=args.repeats_per_content,
        identity_groups=args.identity_groups,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        local_window=args.local_window,
        half_lives=tuple(args.half_lives),
        fixed_lags=tuple(args.fixed_lags),
        ridge_alpha=args.ridge_alpha,
        code_temperature=args.code_temperature,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
