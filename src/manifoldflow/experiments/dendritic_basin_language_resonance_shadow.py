"""Route a frozen language cortex through learned receptor resonance in shadow.

The experiment does not add anatomy to, or update, the supplied checkpoint.
It exposes four progressively deeper views of its existing cortical hierarchy,
learns each view's future disagreement from an external calibration stream, and
then compares competitive selection with the checkpoint's ordinary deepest
prediction on a separate stream and during autonomous generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _sample_byte,
)
from manifoldflow.experiments.dendritic_basin_phase122_full_corpus_4c import (
    DEFAULT_TINY_TRAIN,
    DEFAULT_TINY_VALID,
)

RECEPTORS = ("persistent", "conjunctive", "sequential", "deep_context")
PROJECTION_BUCKETS = 16
HISTORY_BUCKETS = 32
BRANCH_BUCKETS = 32
RECEPTOR_LOCAL_FEATURES = 7 + 2 * PROJECTION_BUCKETS
SHARED_CONTEXT_FEATURES = HISTORY_BUCKETS + BRANCH_BUCKETS
GENERIC_FEATURES = RECEPTOR_LOCAL_FEATURES + SHARED_CONTEXT_FEATURES
_PROJECTION_INDEX = (
    (np.arange(256, dtype=np.int64) * 2654435761) >> 16
) % PROJECTION_BUCKETS
_PROJECTION_SIGN = np.where(
    ((np.arange(256, dtype=np.uint64) * 11400714819323198485) >> 63) == 0,
    -1.0,
    1.0,
)


def _history_projection(cortex: SequenceCortexOpponentLearner) -> np.ndarray:
    """Sparse position-sensitive expansion of the active cortical trace."""
    projected = np.zeros(HISTORY_BUCKETS, dtype=np.float64)
    history = cortex.sequence_history[-cortex.max_context_bytes :]
    for position, byte_id in enumerate(reversed(history), start=1):
        mixed = (int(byte_id) + 257 * position) * 2654435761
        bucket = (mixed >> 16) % HISTORY_BUCKETS
        sign = -1.0 if ((mixed * 11400714819323198485) >> 63) & 1 else 1.0
        projected[bucket] += sign / position
    return projected


def _branch_projection(
    cortex: SequenceCortexOpponentLearner,
    contexts: tuple[int | tuple[int, ...], ...],
) -> np.ndarray:
    """Expose currently active grown branch identities without a route label."""
    projected = np.zeros(BRANCH_BUCKETS, dtype=np.float64)
    if cortex.older_assembly is not None and cortex.previous_assembly is not None:
        key = (cortex.older_assembly << 8) | cortex.previous_assembly
        if (cortex.older_assembly, cortex.previous_assembly) in cortex.residual_counts:
            mixed = key * 2654435761
            projected[(mixed >> 16) % BRANCH_BUCKETS] += 0.5
    for bank_index, context in enumerate(contexts):
        if context not in cortex.sequence_counts[bank_index]:
            continue
        if isinstance(context, tuple):
            packed = 0
            for byte_id in context:
                packed = (packed << 8) | int(byte_id)
        else:
            packed = int(context)
        mixed = packed ^ ((bank_index + 3) * 11400714819323198485)
        bucket = (mixed >> 16) % BRANCH_BUCKETS
        projected[bucket] += (bank_index + 3) / cortex.max_context_bytes
    return projected


@dataclass(frozen=True)
class ReceptorViews:
    distributions: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class ResonanceModel:
    coefficients: np.ndarray
    threshold: float


def _condition(
    parent: np.ndarray,
    row: dict[int, int] | None,
    total: int,
    strength: float,
) -> np.ndarray:
    if row is None:
        return parent.copy()
    child = strength * parent
    for target, count in row.items():
        child[target] += count
    return child / (total + strength)


def _generic_features(
    distribution: np.ndarray,
    parent: np.ndarray,
    *,
    support: int,
    depth: int,
    max_depth: int,
) -> np.ndarray:
    probability = np.maximum(distribution, 1e-300)
    entropy = -float(np.sum(probability * np.log2(probability))) / 8.0
    top = np.partition(distribution, -2)[-2:]
    margin = float(top[-1] - top[-2])
    divergence = float(
        np.sum(probability * np.log2(probability / np.maximum(parent, 1e-300)))
    )
    scalars = np.asarray(
        (
            entropy,
            float(np.max(distribution)),
            margin,
            math.log1p(support) / 20.0,
            depth / max_depth,
            float(np.sum(np.abs(distribution - parent))) / 2.0,
            divergence / 8.0,
        ),
        dtype=np.float64,
    )
    projected = np.bincount(
        _PROJECTION_INDEX,
        weights=distribution * _PROJECTION_SIGN,
        minlength=PROJECTION_BUCKETS,
    )
    residual = np.bincount(
        _PROJECTION_INDEX,
        weights=(distribution - parent) * _PROJECTION_SIGN,
        minlength=PROJECTION_BUCKETS,
    )
    return np.concatenate((scalars, projected, residual))


def receptor_views(cortex: SequenceCortexOpponentLearner) -> ReceptorViews:
    """Read four nested receptor views without changing recurrent state."""
    base_counts = cortex.unigram_counts.astype(np.float64) + cortex.prior
    persistent = base_counts / base_counts.sum()
    previous = cortex.previous_assembly
    if previous is None:
        conjunctive = persistent.copy()
        conjunctive_support = 0
    else:
        conjunctive_support = int(cortex.context_totals[previous])
        conjunctive = np.full(
            256,
            cortex.prior / (conjunctive_support + 256 * cortex.prior),
            dtype=np.float64,
        )
        for target, count in cortex.transition_counts[previous].items():
            conjunctive[target] = (count + cortex.prior) / (
                conjunctive_support + 256 * cortex.prior
            )

    sequential = conjunctive.copy()
    sequential_support = 0
    sequential_depth = 1
    if cortex.older_assembly is not None and previous is not None:
        key = (cortex.older_assembly, previous)
        row = cortex.residual_counts.get(key)
        if row is not None:
            sequential_support = int(cortex.residual_totals[key])
            sequential = _condition(
                sequential,
                row,
                sequential_support,
                cortex.basal_strength,
            )
            sequential_depth = 2

    contexts = cortex._sequence_contexts()
    if contexts:
        row = cortex.sequence_counts[0].get(contexts[0])
        if row is not None:
            sequential_support = int(cortex.sequence_totals[0][contexts[0]])
            sequential = _condition(
                sequential,
                row,
                sequential_support,
                cortex.parent_strength,
            )
            sequential_depth = 3

    deep = sequential.copy()
    deep_support = sequential_support
    deep_depth = sequential_depth
    for bank_index, context in enumerate(contexts[1:], start=1):
        row = cortex.sequence_counts[bank_index].get(context)
        if row is None:
            continue
        deep_support = int(cortex.sequence_totals[bank_index][context])
        deep = _condition(
            deep,
            row,
            deep_support,
            cortex.parent_strength,
        )
        deep_depth = bank_index + 3

    distributions = np.stack((persistent, conjunctive, sequential, deep))
    max_depth = max(cortex.max_context_bytes, 1)
    features = np.stack(
        (
            _generic_features(
                persistent,
                persistent,
                support=int(cortex.unigram_counts.sum()),
                depth=0,
                max_depth=max_depth,
            ),
            _generic_features(
                conjunctive,
                persistent,
                support=conjunctive_support,
                depth=1,
                max_depth=max_depth,
            ),
            _generic_features(
                sequential,
                conjunctive,
                support=sequential_support,
                depth=sequential_depth,
                max_depth=max_depth,
            ),
            _generic_features(
                deep,
                sequential,
                support=deep_support,
                depth=deep_depth,
                max_depth=max_depth,
            ),
        )
    )
    history = _history_projection(cortex)
    branches = _branch_projection(cortex, contexts)
    shared = np.concatenate((history, branches))
    features = np.concatenate(
        (features, np.broadcast_to(shared, (len(RECEPTORS), len(shared)))),
        axis=1,
    )
    return ReceptorViews(distributions=distributions, features=features)


def _load_cortex(path: Path) -> SequenceCortexOpponentLearner:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local checkpoint
    cortex = payload["learner"] if isinstance(payload, dict) else payload
    if type(cortex) is not SequenceCortexOpponentLearner:
        raise TypeError("shadow experiment requires an exact sequence cortex")
    cortex.reset_episode()
    return cortex


def collect_calibration(
    cortex: SequenceCortexOpponentLearner, path: Path, length: int
) -> tuple[np.ndarray, np.ndarray]:
    cortex.reset_episode()
    features = np.empty((length, len(RECEPTORS), GENERIC_FEATURES), dtype=np.float64)
    realized_bits = np.empty((length, len(RECEPTORS)), dtype=np.float64)
    with path.open("rb") as handle:
        raw = handle.read(length)
    for index, byte_id in enumerate(raw):
        views = receptor_views(cortex)
        features[index] = views.features
        realized_bits[index] = -np.log2(
            np.maximum(views.distributions[:, byte_id], 1e-300)
        )
        cortex.advance_observation(byte_id, learn=False)
    cortex.reset_episode()
    return features[: len(raw)], realized_bits[: len(raw)]


def _permission_features(features: np.ndarray) -> np.ndarray:
    """Compare each alternative receptor with the active deep state."""
    local = features[..., :RECEPTOR_LOCAL_FEATURES]
    alternative = local[..., :-1, :]
    deep = np.broadcast_to(local[..., -1:, :], alternative.shape)
    shared = np.broadcast_to(
        features[..., -1:, RECEPTOR_LOCAL_FEATURES:],
        alternative.shape[:-1] + (SHARED_CONTEXT_FEATURES,),
    )
    joined = np.concatenate((alternative, deep, alternative - deep, shared), axis=-1)
    ones = np.ones(joined.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate((joined, ones), axis=-1)


def predicted_utility(features: np.ndarray, model: ResonanceModel) -> np.ndarray:
    return np.einsum(
        "...rf,rf->...r", _permission_features(features), model.coefficients
    )


def select_routes(features: np.ndarray, model: ResonanceModel) -> np.ndarray:
    utility = predicted_utility(features, model)
    best = np.argmax(utility, axis=-1)
    best_utility = np.take_along_axis(utility, best[..., None], axis=-1)[..., 0]
    return np.where(best_utility > model.threshold, best, len(RECEPTORS) - 1)


def calibration_diagnostics(
    features: np.ndarray, realized_bits: np.ndarray, model: ResonanceModel
) -> dict[str, object]:
    """Measure whether stronger permission predicts larger realized savings."""
    score = predicted_utility(features, model)
    best = np.argmax(score, axis=1)
    predicted = score[np.arange(len(score)), best]
    realized = realized_bits[:, -1] - realized_bits[np.arange(len(realized_bits)), best]
    order = np.argsort(predicted)
    buckets = []
    for indices in np.array_split(order, 10):
        if not len(indices):
            continue
        utility = realized[indices]
        buckets.append(
            {
                "examples": len(indices),
                "mean_permission": float(np.mean(predicted[indices])),
                "mean_realized_savings_bits": float(np.mean(utility)),
                "helpful_fraction": float(np.mean(utility > 0.0)),
                "harmful_fraction": float(np.mean(utility < 0.0)),
                "catastrophic_fraction": float(np.mean(utility < -1.0)),
            }
        )
    routes = select_routes(features, model)
    chosen = realized_bits[np.arange(len(routes)), routes]
    return {
        "baseline_bpb": float(np.mean(realized_bits[:, -1])),
        "selected_bpb": float(np.mean(chosen)),
        "selected_fraction": float(np.mean(routes != len(RECEPTORS) - 1)),
        "permission_buckets": buckets,
    }


def fit_resonance(features: np.ndarray, realized_bits: np.ndarray) -> ResonanceModel:
    """Learn when a receptor reduces surprise relative to the deep attractor."""
    if len(features) < 8:
        raise ValueError("resonance calibration requires at least eight observations")
    split = max(4, int(len(features) * 0.75))
    x = _permission_features(features)
    utility = realized_bits[:, -1, None] - realized_bits[:, :-1]
    fit_x = x[:split]
    fit_y = utility[:split]
    coefficients = []
    for receptor in range(len(RECEPTORS) - 1):
        design = fit_x[:, receptor]
        gram = design.T @ design
        penalty = np.eye(design.shape[1], dtype=np.float64) * 10.0
        penalty[-1, -1] = 0.0
        coefficients.append(
            np.linalg.solve(gram + penalty, design.T @ fit_y[:, receptor])
        )
    model = ResonanceModel(np.stack(coefficients), threshold=0.0)
    tune_score = predicted_utility(features[split:], model)
    tune_bits = realized_bits[split:]
    candidate_thresholds = np.unique(
        np.concatenate(
            (
                np.asarray((0.0,), dtype=np.float64),
                np.quantile(
                    tune_score,
                    (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999),
                ),
            )
        )
    )
    best_total = float(np.sum(tune_bits[:, -1]))
    best_threshold = float("inf")
    for threshold in candidate_thresholds:
        candidate = ResonanceModel(model.coefficients, float(threshold))
        routes = select_routes(features[split:], candidate)
        total = float(np.sum(tune_bits[np.arange(len(routes)), routes]))
        if total < best_total or (
            total == best_total and float(threshold) > best_threshold
        ):
            best_total = total
            best_threshold = float(threshold)
    return ResonanceModel(model.coefficients, best_threshold)


def evaluate(
    cortex: SequenceCortexOpponentLearner,
    path: Path,
    length: int,
    model: ResonanceModel,
    *,
    shuffle_seed: int = 73,
) -> dict[str, object]:
    cortex.reset_episode()
    totals = np.zeros(len(RECEPTORS), dtype=np.float64)
    selected_bits = 0.0
    uniform_bits = 0.0
    oracle_bits = 0.0
    shuffled_bits = 0.0
    selected_routes = np.zeros(len(RECEPTORS), dtype=np.int64)
    helpful = 0
    harmful = 0
    savings = []
    rng = np.random.default_rng(shuffle_seed)
    shuffled_model = ResonanceModel(
        model.coefficients[rng.permutation(len(RECEPTORS) - 1)], model.threshold
    )
    with path.open("rb") as handle:
        raw = handle.read(length)
    for byte_id in raw:
        views = receptor_views(cortex)
        actual = -np.log2(np.maximum(views.distributions[:, byte_id], 1e-300))
        route = int(select_routes(views.features, model))
        shuffled_route = int(select_routes(views.features, shuffled_model))
        baseline = float(actual[-1])
        chosen = float(actual[route])
        totals += actual
        selected_bits += chosen
        uniform_bits += -math.log2(
            max(float(np.mean(views.distributions[:, byte_id])), 1e-300)
        )
        oracle_bits += float(np.min(actual))
        shuffled_bits += float(actual[shuffled_route])
        selected_routes[route] += 1
        delta = baseline - chosen
        savings.append(delta)
        helpful += int(delta > 0.0)
        harmful += int(delta < 0.0)
        cortex.advance_observation(byte_id, learn=False)
    cortex.reset_episode()
    count = len(raw)
    saving_array = np.asarray(savings)
    return {
        "path": str(path),
        "scored_bytes": count,
        "receptor_bpb": {
            name: float(totals[index] / count) for index, name in enumerate(RECEPTORS)
        },
        "baseline_deep_context_bpb": float(totals[-1] / count),
        "competitive_resonance_bpb": float(selected_bits / count),
        "uniform_pool_bpb": float(uniform_bits / count),
        "oracle_receptor_bpb": float(oracle_bits / count),
        "shuffled_resonance_bpb": float(shuffled_bits / count),
        "competitive_savings_bits": float(totals[-1] - selected_bits),
        "helpful_route_fraction": helpful / count,
        "harmful_route_fraction": harmful / count,
        "worst_one_percent_mean_savings_bits": float(
            np.mean(
                np.partition(saving_array, max(0, count // 100 - 1))[: count // 100]
            )
        ),
        "route_fraction": {
            name: float(selected_routes[index] / count)
            for index, name in enumerate(RECEPTORS)
        },
    }


def generate(
    source: SequenceCortexOpponentLearner,
    model: ResonanceModel,
    prompt: bytes,
    *,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    seed: int,
    routing_label: str = "competitive_resonance",
) -> dict[str, object]:
    cortex = copy.deepcopy(source)
    cortex.reset_episode()
    cortex.observe_chunk_probabilities(
        np.frombuffer(prompt, dtype=np.uint8), learn=False
    )
    rng = np.random.default_rng(seed)
    generated = bytearray()
    routes = np.zeros(len(RECEPTORS), dtype=np.int64)
    for _ in range(max_new_bytes):
        views = receptor_views(cortex)
        route = int(select_routes(views.features, model))
        routes[route] += 1
        byte_id = _sample_byte(
            views.distributions[route],
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=0.95,
        )
        generated.append(byte_id)
        cortex.advance_observation(byte_id, learn=False)
    boundary = getattr(cortex, "story_boundary", b"<|endoftext|>\n")
    return {
        "prompt": prompt.decode("utf-8"),
        "routing": routing_label,
        "mode": "greedy" if temperature == 0.0 else "sample",
        "generated_bytes": len(generated),
        "text": (prompt + generated).decode("utf-8", errors="backslashreplace"),
        "generated_hex": generated.hex(),
        "route_fraction": {
            name: float(routes[index] / max(len(generated), 1))
            for index, name in enumerate(RECEPTORS)
        },
        "diagnostics": _generation_diagnostics(bytes(generated), boundary),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source = _load_cortex(args.checkpoint)
    features, bits = collect_calibration(
        source, args.calibration_stream, args.calibration_bytes
    )
    model = fit_resonance(features, bits)
    validation = evaluate(
        source,
        args.validation_stream,
        args.validation_bytes,
        model,
        shuffle_seed=args.seed,
    )
    prompts = (
        b"Once upon a time",
        b"Lily had a red ball. One day, Lily",
        b"USER: Hello! How are you?\nASSISTANT:",
    )
    generations = []
    baseline_model = ResonanceModel(model.coefficients, float("inf"))
    for prompt in prompts:
        for routing_label, generation_model in (
            ("baseline_deep_context", baseline_model),
            ("competitive_resonance", model),
        ):
            generations.append(
                generate(
                    source,
                    generation_model,
                    prompt,
                    max_new_bytes=args.generation_bytes,
                    temperature=0.0,
                    top_k=1,
                    seed=args.seed,
                    routing_label=routing_label,
                )
            )
            generations.append(
                generate(
                    source,
                    generation_model,
                    prompt,
                    max_new_bytes=args.generation_bytes,
                    temperature=0.8,
                    top_k=32,
                    seed=args.seed,
                    routing_label=routing_label,
                )
            )
    report = {
        "experiment": "frozen_language_cortex_resonance_shadow",
        "checkpoint": str(args.checkpoint),
        "checkpoint_body_bytes": source.embodied_bytes,
        "receptors": list(RECEPTORS),
        "receptor_contract": (
            "progressively deeper views of existing cortical anatomy; no organ label "
            "or future byte enters current selection"
        ),
        "learning_contract": (
            "only external calibration bytes train future disagreement; validation "
            "and generated bytes never update cortex or resonance"
        ),
        "calibration_stream": str(args.calibration_stream),
        "calibration_bytes": len(features),
        "resonance_coefficients": model.coefficients.tolist(),
        "permission_threshold_bits": model.threshold,
        "calibration_diagnostics": calibration_diagnostics(features, bits, model),
        "validation": validation,
        "generations": generations,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/alm_phase126_tiny_3epoch_64mb_4c/epoch_3/checkpoint.pkl"),
    )
    parser.add_argument("--calibration-stream", type=Path, default=DEFAULT_TINY_TRAIN)
    parser.add_argument("--validation-stream", type=Path, default=DEFAULT_TINY_VALID)
    parser.add_argument("--calibration-bytes", type=int, default=100_000)
    parser.add_argument("--validation-bytes", type=int, default=200_000)
    parser.add_argument("--generation-bytes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.calibration_bytes, args.validation_bytes, args.generation_bytes) < 1:
        parser.error("byte counts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
