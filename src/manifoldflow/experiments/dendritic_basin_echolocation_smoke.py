"""Native-vector, GPU-resident growing-ALM echolocation smoke.

Relation growth is label-free: a shared bounded nursery proposes arbitrary
pairs from raw channels and generic multiscale traces, then perturb/selects
each local bell by the ALM residual-interaction objective. A backprop-free
linear diagnostic is fit afterward from visible training observations. The
candidate split selects only population size and ridge strength; the held-out
test split is touched once for the causal controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.echolocation import (
    CHANNEL_NAMES,
    EcholocationEpisode,
    dead_reckoning_prediction,
    generate_episodes,
    shuffled_ranging_episodes,
    stack_episode_vectors,
)

EPS = 1e-8


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - optional GPU dependency
        raise RuntimeError("install the 'alm-gpu' extra for this smoke") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("this smoke requires a CUDA device")
    return cp


@dataclass
class GrownRelations:
    parent_a: Any
    parent_b: Any
    w_a: Any
    w_b: Any
    mu: Any
    log_sigma: Any
    score: Any

    def take(self, count: int) -> GrownRelations:
        indices = self.score.argsort()[::-1][:count]
        return self.select(indices)

    def select(self, indices: Any) -> GrownRelations:
        return GrownRelations(
            *(
                getattr(self, name)[indices]
                for name in (
                    "parent_a",
                    "parent_b",
                    "w_a",
                    "w_b",
                    "mu",
                    "log_sigma",
                    "score",
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--candidate-episodes", type=int, default=120)
    parser.add_argument("--test-episodes", type=int, default=240)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hidden-steps", type=int, default=12)
    parser.add_argument("--velocity-wander", type=float, default=0.012)
    parser.add_argument("--relation-capacity", type=int, default=512)
    parser.add_argument("--growth-rounds", type=int, default=8)
    parser.add_argument("--nursery-size", type=int, default=256)
    parser.add_argument("--hill-steps", type=int, default=64)
    parser.add_argument("--fit-samples", type=int, default=6000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def feature_names(timescales: tuple[int, ...]) -> list[str]:
    names = [f"current:{name}" for name in CHANNEL_NAMES]
    names.extend(
        f"trace_{scale}:{name}" for scale in timescales for name in CHANNEL_NAMES
    )
    return names


def temporal_features(vectors: np.ndarray, timescales: tuple[int, ...], cp: Any) -> Any:
    """Build generic normalized decaying traces on-device, reset per episode."""
    values = cp.asarray(vectors, dtype=cp.float32)
    episodes, steps, channels = values.shape
    lambdas = cp.asarray(
        [math.exp(math.log(0.5) / scale) for scale in timescales],
        dtype=cp.float32,
    )
    traces = cp.zeros((episodes, len(timescales), channels), dtype=cp.float32)
    result = cp.empty(
        (episodes, steps, channels * (1 + len(timescales))), dtype=cp.float32
    )
    for index in range(steps):
        traces *= lambdas[None, :, None]
        traces += values[:, index, None, :]
        normalized = traces * (1.0 - lambdas)[None, :, None]
        result[:, index, :channels] = values[:, index]
        result[:, index, channels:] = normalized.reshape(episodes, -1)
    return result


def interaction_scores(
    x: Any, theta: Any, parent_a: Any, parent_b: Any, cp: Any
) -> Any:
    a = x[:, parent_a]
    b = x[:, parent_b]
    sigma = cp.exp(theta[:, 3])[None, :]
    drive = theta[None, :, 0] * a + theta[None, :, 1] * b - theta[None, :, 2]
    activation = cp.exp(-0.5 * (drive / sigma) ** 2)
    dc = activation - activation.mean(axis=0, keepdims=True)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    var_d = (dc * dc).mean(axis=0)
    var_a = (ac * ac).mean(axis=0)
    var_b = (bc * bc).mean(axis=0)
    cov_da = (dc * ac).mean(axis=0)
    cov_db = (dc * bc).mean(axis=0)
    return var_d - cov_da**2 / (var_a + EPS) - cov_db**2 / (var_b + EPS)


def grow_relations(
    features: Any,
    args: argparse.Namespace,
    seed: int,
    cp: Any,
    progress_callback: Callable[[int, GrownRelations], None] | None = None,
) -> GrownRelations:
    """One shared bounded nursery; no modality has reserved slots or status."""
    rng = cp.random.RandomState(seed)
    flat = features.reshape(-1, features.shape[-1])
    if len(flat) > args.fit_samples:
        flat = flat[rng.choice(len(flat), args.fit_samples, replace=False)]
    mature: GrownRelations | None = None
    for growth_round in range(args.growth_rounds):
        parent_a = rng.randint(0, flat.shape[1], args.nursery_size, dtype=cp.int64)
        parent_b = rng.randint(0, flat.shape[1], args.nursery_size, dtype=cp.int64)
        equal = parent_a == parent_b
        parent_b[equal] = (parent_b[equal] + 1) % flat.shape[1]
        theta = cp.column_stack(
            (
                rng.normal(0.0, 2.0, args.nursery_size),
                rng.normal(0.0, 2.0, args.nursery_size),
                rng.uniform(-1.0, 1.0, args.nursery_size),
                rng.uniform(math.log(0.03), math.log(1.0), args.nursery_size),
            )
        ).astype(cp.float32)
        score = interaction_scores(flat, theta, parent_a, parent_b, cp)
        scales = cp.asarray((0.25, 0.25, 0.10, 0.15), dtype=cp.float32)
        for step in range(args.hill_steps):
            proposal = (
                theta + rng.normal(0.0, 1.0, theta.shape).astype(cp.float32) * scales
            )
            proposal_score = interaction_scores(flat, proposal, parent_a, parent_b, cp)
            improved = proposal_score > score
            theta[improved] = proposal[improved]
            score[improved] = proposal_score[improved]
            if (step + 1) % 24 == 0:
                scales *= 0.65
        nursery = GrownRelations(
            parent_a,
            parent_b,
            theta[:, 0],
            theta[:, 1],
            theta[:, 2],
            theta[:, 3],
            score,
        )
        if mature is None:
            mature = nursery.take(args.relation_capacity)
        else:
            mature = GrownRelations(
                *(
                    cp.concatenate((getattr(mature, name), getattr(nursery, name)))
                    for name in (
                        "parent_a",
                        "parent_b",
                        "w_a",
                        "w_b",
                        "mu",
                        "log_sigma",
                        "score",
                    )
                )
            ).take(args.relation_capacity)
        print(
            f"    growth {growth_round + 1}/{args.growth_rounds}: "
            f"mature={len(mature.score)}, best_J={float(mature.score.max()):.6f}",
            file=sys.stderr,
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(growth_round + 1, mature)
    assert mature is not None
    return mature


def relation_activations(features: Any, relations: GrownRelations, cp: Any) -> Any:
    a = features[..., relations.parent_a]
    b = features[..., relations.parent_b]
    drive = relations.w_a * a + relations.w_b * b - relations.mu
    return cp.exp(-0.5 * (drive / cp.exp(relations.log_sigma)) ** 2)


def fit_ridge(x: Any, y: Any, alpha: float, cp: Any) -> Any:
    design = cp.concatenate((x, cp.ones((len(x), 1), dtype=cp.float32)), axis=1)
    gram = design.T @ design
    penalty = cp.eye(gram.shape[0], dtype=cp.float32) * alpha
    penalty[-1, -1] = 0.0
    return cp.linalg.solve(gram + penalty, design.T @ y)


def predict(x: Any, beta: Any, cp: Any) -> Any:
    # Position is a bounded sensory coordinate. This is the same physical
    # support used by the generator and prevents an ill-conditioned visible
    # diagnostic from turning an OOD dark input into an unbounded estimate.
    return cp.clip(x @ beta[:-1] + beta[-1], -1.0, 1.0)


def mse(y: Any, prediction: Any, cp: Any) -> float:
    return float(cp.mean((y - prediction) ** 2).get())


def r2(y: Any, prediction: Any, cp: Any) -> float:
    residual = cp.sum((y - prediction) ** 2)
    total = cp.sum((y - y.mean()) ** 2)
    return float((1.0 - residual / cp.maximum(total, EPS)).get())


def final_features(
    episodes: list[EcholocationEpisode],
    *,
    visible: bool,
    timescales: tuple[int, ...],
    cp: Any,
    audio_mode: str = "real",
) -> Any:
    vectors = stack_episode_vectors(episodes, visible=visible, audio_mode=audio_mode)
    return temporal_features(vectors, timescales, cp)[:, -1]


def relation_report(
    relations: GrownRelations,
    beta: Any,
    names: list[str],
    cp: Any,
    limit: int = 30,
) -> list[dict[str, Any]]:
    order = cp.argsort(cp.abs(beta[:-1]))[::-1][:limit].get()
    rows = []
    for raw_index in order:
        index = int(raw_index)
        a = names[int(relations.parent_a[index].get())]
        b = names[int(relations.parent_b[index].get())]
        rows.append(
            {
                "parent_a": a,
                "parent_b": b,
                "local_J": float(relations.score[index].get()),
                "readout_weight": float(beta[index].get()),
                "audio_relation": "beep" in a
                or "chime" in a
                or "beep" in b
                or "chime" in b,
                "cross_timescale_audio": (
                    ("trace_" in a and "trace_" in b)
                    and (
                        ("beep" in a and "beep" in b) or ("chime" in a and "chime" in b)
                    )
                    and a.split(":")[0] != b.split(":")[0]
                ),
            }
        )
    return rows


def relation_route_indices(
    relations: GrownRelations,
    names: list[str],
    *,
    allowed_sources: set[str],
    require_audio: bool,
    cp: Any,
) -> Any:
    """Select a route by sensory provenance, never by test performance."""
    selected = []
    audio_sources = {"range_beep", "crossing_chime"}
    parent_a = relations.parent_a.get()
    parent_b = relations.parent_b.get()
    for index, (raw_a, raw_b) in enumerate(zip(parent_a, parent_b, strict=True)):
        sources = {
            names[int(raw_a)].split(":", 1)[1],
            names[int(raw_b)].split(":", 1)[1],
        }
        if sources <= allowed_sources and (
            not require_audio or bool(sources & audio_sources)
        ):
            selected.append(index)
    return cp.asarray(selected, dtype=cp.int64)


def select_route_readout(
    relations: GrownRelations,
    indices: Any,
    train_features: Any,
    y_train: Any,
    candidate_features: Any,
    y_candidate: Any,
    cp: Any,
) -> tuple[GrownRelations, Any, float, float]:
    """Fit a scoped diagnostic and tune it on candidate-visible only."""
    if len(indices) == 0:
        raise RuntimeError("shared nursery produced no relations for this route")
    route = relations.select(indices)
    train_x = relation_activations(train_features, route, cp).reshape(-1, len(indices))
    candidate_x = relation_activations(candidate_features, route, cp)
    best: tuple[float, float, Any] | None = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        beta = fit_ridge(train_x, y_train, alpha, cp)
        score = mse(y_candidate, predict(candidate_x, beta, cp), cp)
        if best is None or score < best[0]:
            best = (score, alpha, beta)
    assert best is not None
    score, alpha, beta = best
    return route, beta, alpha, score


def run_seed(args: argparse.Namespace, seed: int, cp: Any) -> dict[str, Any]:
    timescales = (1, 2, 4, 8, 16, 32, 64)
    base = 1_000_003 * (seed + 1)
    kwargs = dict(
        steps=args.steps,
        hidden_steps=args.hidden_steps,
        velocity_wander=args.velocity_wander,
    )
    train = generate_episodes(args.train_episodes, seed=base + 11, **kwargs)
    candidate = generate_episodes(args.candidate_episodes, seed=base + 23, **kwargs)
    test = generate_episodes(args.test_episodes, seed=base + 37, **kwargs)

    start = time.perf_counter()
    train_vectors = stack_episode_vectors(train, visible=True)
    train_features = temporal_features(train_vectors, timescales, cp)
    relations = grow_relations(train_features, args, seed, cp)
    cp.cuda.Stream.null.synchronize()
    growth_seconds = time.perf_counter() - start

    y_train = cp.asarray(np.concatenate([ep.positions for ep in train]))
    candidate_features = final_features(
        candidate, visible=True, timescales=timescales, cp=cp
    )
    y_candidate = cp.asarray([ep.final_position for ep in candidate])
    best: tuple[float, int, float, Any] | None = None
    selection = []
    counts = sorted(
        set(
            (
                max(8, args.relation_capacity // 4),
                max(8, args.relation_capacity // 2),
                args.relation_capacity,
            )
        )
    )
    for count in counts:
        subset = relations.take(count)
        train_x = relation_activations(train_features, subset, cp).reshape(-1, count)
        candidate_x = relation_activations(candidate_features, subset, cp)
        for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            beta = fit_ridge(train_x, y_train, alpha, cp)
            score = mse(y_candidate, predict(candidate_x, beta, cp), cp)
            selection.append(
                {"relations": count, "ridge": alpha, "candidate_visible_mse": score}
            )
            if best is None or score < best[0]:
                best = (score, count, alpha, beta)
    assert best is not None
    _, count, alpha, beta = best
    selected = relations.take(count)
    names = feature_names(timescales)

    visual_indices = relation_route_indices(
        relations,
        names,
        allowed_sources={"dot_position", "target_position", "visible"},
        require_audio=False,
        cp=cp,
    )
    audio_target_indices = relation_route_indices(
        relations,
        names,
        allowed_sources={"target_position", "range_beep", "crossing_chime"},
        require_audio=False,
        cp=cp,
    )
    visual_route, visual_beta, visual_alpha, visual_candidate_mse = (
        select_route_readout(
            relations,
            visual_indices,
            train_features,
            y_train,
            candidate_features,
            y_candidate,
            cp,
        )
    )
    audio_route, audio_beta, audio_alpha, audio_candidate_mse = select_route_readout(
        relations,
        audio_target_indices,
        train_features,
        y_train,
        candidate_features,
        y_candidate,
        cp,
    )

    def evaluate(
        episodes: list[EcholocationEpisode], visible: bool, audio_mode: str = "real"
    ) -> tuple[Any, Any]:
        features = final_features(
            episodes,
            visible=visible,
            timescales=timescales,
            cp=cp,
            audio_mode=audio_mode,
        )
        truth = cp.asarray([ep.final_position for ep in episodes])
        return truth, predict(relation_activations(features, selected, cp), beta, cp)

    multi_y, multi_p = evaluate(test, True)
    dark_y, dark_p = evaluate(test, False)
    shuffled = shuffled_ranging_episodes(test, seed=base + 41)
    shuffled_y, shuffled_p = evaluate(shuffled, False)
    silent_y, silent_p = evaluate(test, False, "silent")
    dark_mse = mse(dark_y, dark_p, cp)
    shuffled_mse = mse(shuffled_y, shuffled_p, cp)
    silent_mse = mse(silent_y, silent_p, cp)
    dead = cp.asarray([dead_reckoning_prediction(ep) for ep in test])
    dead_mse = mse(dark_y, dead, cp)
    multi_r2 = r2(multi_y, multi_p, cp)
    report_rows = relation_report(selected, beta, names, cp)

    def route_prediction(
        episodes: list[EcholocationEpisode],
        route: GrownRelations,
        route_beta: Any,
        *,
        visible: bool,
        audio_mode: str = "real",
    ) -> tuple[Any, Any]:
        features = final_features(
            episodes,
            visible=visible,
            timescales=timescales,
            cp=cp,
            audio_mode=audio_mode,
        )
        truth = cp.asarray([ep.final_position for ep in episodes])
        values = relation_activations(features, route, cp)
        return truth, predict(values, route_beta, cp)

    visual_y, visual_p = route_prediction(test, visual_route, visual_beta, visible=True)
    audio_y, audio_p = route_prediction(test, audio_route, audio_beta, visible=False)
    audio_shuffled_y, audio_shuffled_p = route_prediction(
        shuffled, audio_route, audio_beta, visible=False
    )
    audio_silent_y, audio_silent_p = route_prediction(
        test, audio_route, audio_beta, visible=False, audio_mode="silent"
    )
    audio_probe_mse = mse(audio_y, audio_p, cp)
    audio_probe_shuffled_mse = mse(audio_shuffled_y, audio_shuffled_p, cp)
    audio_probe_silent_mse = mse(audio_silent_y, audio_silent_p, cp)
    causal_pass = dark_mse < dead_mse and dark_mse < shuffled_mse
    return {
        "seed": seed,
        "held_out_multimodal_position_r2": multi_r2,
        "dark_audio_position_mse": dark_mse,
        "dead_reckoning_mse": dead_mse,
        "shuffled_audio_mse": shuffled_mse,
        "silent_audio_mse": silent_mse,
        "audio_vs_dead_reckoning_improvement": dead_mse / max(dark_mse, EPS),
        "real_vs_shuffled_audio_improvement": shuffled_mse / max(dark_mse, EPS),
        "causal_audio_pass": causal_pass,
        "strict_smoke_pass": multi_r2 > 0.80 and causal_pass,
        "growth_seconds": growth_seconds,
        "training_frames_per_second": len(train) * args.steps / growth_seconds,
        "selected_relation_count": count,
        "selected_ridge": alpha,
        "candidate_selection": selection,
        "resident_relation_bytes": int(
            sum(
                getattr(selected, name).nbytes
                for name in (
                    "parent_a",
                    "parent_b",
                    "w_a",
                    "w_b",
                    "mu",
                    "log_sigma",
                    "score",
                )
            )
        ),
        "strongest_relations": report_rows,
        "audio_relations_in_top_report": sum(
            row["audio_relation"] for row in report_rows
        ),
        "cross_timescale_audio_relations_in_top_report": sum(
            row["cross_timescale_audio"] for row in report_rows
        ),
        "separate_routes": {
            "visual_relation_count": len(visual_indices),
            "visual_ridge": visual_alpha,
            "visual_candidate_mse": visual_candidate_mse,
            "visual_held_out_r2": r2(visual_y, visual_p, cp),
            "audio_target_relation_count": len(audio_target_indices),
            "audio_target_ridge": audio_alpha,
            "audio_target_candidate_mse": audio_candidate_mse,
            "audio_target_dark_mse": audio_probe_mse,
            "audio_target_shuffled_mse": audio_probe_shuffled_mse,
            "audio_target_silent_mse": audio_probe_silent_mse,
            "audio_target_vs_dead_reckoning_improvement": (
                dead_mse / max(audio_probe_mse, EPS)
            ),
            "audio_target_real_vs_shuffled_improvement": (
                audio_probe_shuffled_mse / max(audio_probe_mse, EPS)
            ),
        },
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "held_out_multimodal_position_r2",
        "dark_audio_position_mse",
        "dead_reckoning_mse",
        "shuffled_audio_mse",
        "silent_audio_mse",
        "audio_vs_dead_reckoning_improvement",
        "real_vs_shuffled_audio_improvement",
    )
    result = {}
    for name in names:
        values = np.asarray([report[name] for report in reports])
        result[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    result["strict_smoke_pass_count"] = sum(r["strict_smoke_pass"] for r in reports)
    result["all_seeds_pass"] = all(r["strict_smoke_pass"] for r in reports)
    route_names = (
        "visual_held_out_r2",
        "audio_target_dark_mse",
        "audio_target_shuffled_mse",
        "audio_target_silent_mse",
        "audio_target_vs_dead_reckoning_improvement",
        "audio_target_real_vs_shuffled_improvement",
    )
    result["separate_routes"] = {}
    for name in route_names:
        values = np.asarray([report["separate_routes"][name] for report in reports])
        result["separate_routes"][name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    return result


def main() -> None:
    args = parse_args()
    cp = _cupy()
    reports = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(f"native-vector echolocation seed {seed}", file=sys.stderr, flush=True)
        reports.append(run_seed(args, seed, cp))
    report = {
        "experiment": "native_vector_shared_nursery_growing_alm_echolocation",
        "execution": "cupy_cuda_no_autograd",
        "input_shape": [args.steps, len(CHANNEL_NAMES)],
        "input_channels": list(CHANNEL_NAMES),
        "timescales": [1, 2, 4, 8, 16, 32, 64],
        "forbidden_inputs": ["distance", "velocity", "beep frequency", "audio encoder"],
        "selection_contract": (
            "relations grow by label-free local J on train; candidate-visible split "
            "selects only population size and ridge; test is untouched"
        ),
        "args": vars(args) | {"report": str(args.report) if args.report else None},
        "aggregate": aggregate(reports),
        "seeds": reports,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
