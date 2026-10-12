"""ALM-64MB-1C three-organ native-vector echolocation smoke.

Vision and audition develop in independent label-free local-J nurseries.  A
third bounded organ learns relations over auditory-organ activity and the
visual organ's persisted target state.  A reliability gate favors vision while
it has sensory support and delegates to the integration organ in darkness.

This is the arbitrary-vector adaptation of the current bounded ALM body
contract: ``64MB`` is one organism-wide carrying-capacity ceiling and ``1C``
is one developmental lineage.  The three organs are differentiated anatomy
inside that column; they are not three independently named ALMs.  The organism
reports exact grown array bytes separately from its body class and refuses to
cross the class ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.tasks.echolocation import (
    EcholocationEpisode,
    dead_reckoning_prediction,
    generate_episodes,
    retarget_episode,
    shuffled_ranging_episodes,
    stack_episode_vectors,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import (
    EPS,
    GrownRelations,
    _cupy,
    fit_ridge,
    grow_relations,
    mse,
    predict,
    r2,
    relation_activations,
    temporal_features,
)

DEFAULT_BODY_BYTES = 64_000_000
DEFAULT_COLUMNS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--candidate-episodes", type=int, default=120)
    parser.add_argument("--test-episodes", type=int, default=240)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hidden-steps", type=int, default=24)
    parser.add_argument("--velocity-wander", type=float, default=0.03)
    # Vision receives four times the sensory anatomy of audition. Integration
    # remains large because it must bind the two already differentiated senses.
    parser.add_argument("--visual-capacity", type=int, default=1024)
    parser.add_argument("--auditory-capacity", type=int, default=256)
    parser.add_argument("--integration-capacity", type=int, default=1024)
    parser.add_argument("--growth-rounds", type=int, default=8)
    parser.add_argument("--nursery-size", type=int, default=256)
    parser.add_argument("--hill-steps", type=int, default=64)
    parser.add_argument("--fit-samples", type=int, default=6000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.columns != 1:
        parser.error("this smoke is anatomically fixed to one developmental column")
    if args.max_body_bytes != DEFAULT_BODY_BYTES:
        parser.error("this entry point is fixed to the ALM-64MB body class")
    for name in (
        "visual_capacity",
        "auditory_capacity",
        "integration_capacity",
        "growth_rounds",
        "nursery_size",
        "fit_samples",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def _growth_args(args: argparse.Namespace, capacity: int) -> argparse.Namespace:
    return argparse.Namespace(
        relation_capacity=capacity,
        growth_rounds=args.growth_rounds,
        nursery_size=args.nursery_size,
        hill_steps=args.hill_steps,
        fit_samples=args.fit_samples,
    )


def fit_visible_head(
    activations: Any,
    targets: Any,
    candidate_activations: Any,
    candidate_targets: Any,
    cp: Any,
) -> tuple[Any, float, float]:
    flat = activations.reshape(-1, activations.shape[-1])
    target_flat = targets.reshape(-1)
    best: tuple[float, float, Any] | None = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        beta = fit_ridge(flat, target_flat, alpha, cp)
        score = mse(candidate_targets, predict(candidate_activations, beta, cp), cp)
        if best is None or score < best[0]:
            best = (score, alpha, beta)
    assert best is not None
    score, alpha, beta = best
    return beta, alpha, score


def persist_target_state(target_estimate: Any, visibility: Any, cp: Any) -> Any:
    """Hold the last supported visual target estimate inside the visual organ."""
    state = cp.empty_like(target_estimate)
    retained = target_estimate[:, 0]
    for step in range(target_estimate.shape[1]):
        supported = visibility[:, step] > 0.5
        retained = cp.where(supported, target_estimate[:, step], retained)
        state[:, step] = retained
    return state


def _organ_states(
    vectors: np.ndarray,
    visual_relations: GrownRelations,
    auditory_relations: GrownRelations,
    target_beta: Any,
    sensory_timescales: tuple[int, ...],
    cp: Any,
) -> tuple[Any, Any, Any, Any]:
    visual_features = temporal_features(vectors[:, :, :3], sensory_timescales, cp)
    auditory_features = temporal_features(vectors[:, :, 3:], sensory_timescales, cp)
    visual_state = relation_activations(visual_features, visual_relations, cp)
    auditory_state = relation_activations(auditory_features, auditory_relations, cp)
    target_estimate = predict(visual_state, target_beta, cp)
    visibility = cp.asarray(vectors[:, :, 2])
    target_state = persist_target_state(target_estimate, visibility, cp)
    return visual_state, auditory_state, target_state, visibility


def _integration_features(
    auditory_state: Any,
    target_state: Any,
    integration_timescales: tuple[int, ...],
    cp: Any,
) -> Any:
    organ_vector = cp.concatenate((target_state[..., None], auditory_state), axis=-1)
    return temporal_features(organ_vector, integration_timescales, cp)


def _cross_timescale_audio_count(
    relations: GrownRelations, auditory_channels: int, cp: Any
) -> int:
    """Count auditory relations pairing distinct trace banks."""
    a = relations.parent_a.get()
    b = relations.parent_b.get()
    count = 0
    for left, right in zip(a, b, strict=True):
        left_bank = (int(left) - auditory_channels) // auditory_channels
        right_bank = (int(right) - auditory_channels) // auditory_channels
        if int(left) >= auditory_channels and int(right) >= auditory_channels:
            count += int(left_bank != right_bank)
    return count


def _relation_bytes(relations: GrownRelations) -> int:
    """Count the resident device arrays that constitute one grown organ."""
    return int(
        sum(
            getattr(relations, name).nbytes
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


def _readout_bytes(*readouts: Any) -> int:
    return int(sum(readout.nbytes for readout in readouts))


def _visualizer_episode(episode: Any, *, visible: bool) -> dict[str, Any]:
    vectors = stack_episode_vectors([episode], visible=visible)[0]
    return {
        "positions": episode.positions.astype(float).tolist(),
        "target_position": float(episode.target_position),
        "range_beeps": episode.range_beeps.astype(int).tolist(),
        "crossing_chimes": episode.crossing_chimes.astype(int).tolist(),
        "blackout_start": int(episode.blackout_start if not visible else len(vectors)),
        "observations": vectors.astype(float).tolist(),
    }


def _fit_preview_head(activations: Any, targets: Any, cp: Any) -> Any:
    """Fit a small deterministic diagnostic for live developmental previews."""
    flat_x = activations.reshape(-1, activations.shape[-1])
    flat_y = targets.reshape(-1)
    sample_count = min(512, len(flat_x))
    indexes = cp.linspace(0, len(flat_x) - 1, sample_count, dtype=cp.int64)
    x = flat_x[indexes]
    y = flat_y[indexes]
    if x.shape[1] < sample_count:
        return fit_ridge(x, y, 0.1, cp)
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    centered_x = x - x_mean
    centered_y = y - y_mean
    dual = cp.linalg.solve(
        centered_x @ centered_x.T
        + cp.eye(sample_count, dtype=cp.float32) * 0.1,
        centered_y,
    )
    weights = centered_x.T @ dual
    intercept = y_mean - x_mean @ weights
    return cp.concatenate((weights, intercept[None]))


def _dead_reckoning_trajectory(episode: Any) -> list[float]:
    prediction = episode.positions.astype(float).copy()
    stop = episode.blackout_start
    last = float(episode.positions[stop - 1])
    velocity = last - float(episode.positions[stop - 2])
    for index in range(stop, len(prediction)):
        prediction[index] = np.clip(last + (index - stop + 1) * velocity, -1.0, 1.0)
    return prediction.tolist()


def _continuous_episode(
    target_position: float, steps: int = 240, velocity_scale: float = 1.0
) -> Any:
    span = 1.8
    travel = (
        np.arange(steps, dtype=np.float32) * 0.035 * velocity_scale
    ) % (2.0 * span)
    positions = -0.9 + np.where(travel <= span, travel, 2.0 * span - travel)
    blank = np.zeros(steps, dtype=np.float32)
    source = EcholocationEpisode(
        positions=positions.astype(np.float32),
        target_position=float(target_position),
        range_beeps=blank,
        crossing_chimes=blank.copy(),
        blackout_start=16,
    )
    return retarget_episode(source, target_position)


def run_seed(
    args: argparse.Namespace,
    seed: int,
    cp: Any,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    model_callback: (
        Callable[[Callable[[float, str, bool, float], dict[str, Any]]], None] | None
    ) = None,
) -> dict[str, Any]:
    sensory_timescales = (1, 2, 4, 8, 16, 32, 64)
    integration_timescales = (2, 8, 16, 32, 64)
    base = 1_000_003 * (seed + 1)
    kwargs = dict(
        steps=args.steps,
        hidden_steps=args.hidden_steps,
        velocity_wander=args.velocity_wander,
    )
    train = generate_episodes(args.train_episodes, seed=base + 11, **kwargs)
    candidate = generate_episodes(args.candidate_episodes, seed=base + 23, **kwargs)
    test = generate_episodes(args.test_episodes, seed=base + 37, **kwargs)
    dark_example = next(
        (
            episode
            for episode in test
            if np.any(episode.crossing_chimes) and episode.final_position < 0.95
        ),
        test[0],
    )
    if event_callback is not None:
        event_callback(
            {
                "type": "dataset",
                "seed": seed,
                "training": _visualizer_episode(train[0], visible=True),
                "dark_validation": _visualizer_episode(dark_example, visible=False),
                "dead_reckoning_predictions": _dead_reckoning_trajectory(
                    dark_example
                ),
                "training_episode_count": len(train),
                "candidate_episode_count": len(candidate),
                "test_episode_count": len(test),
            }
        )
    train_vectors = stack_episode_vectors(train, visible=True)
    candidate_vectors = stack_episode_vectors(candidate, visible=True)

    def progress(
        organ: str, event_type: str = "growth"
    ) -> Callable[[int, GrownRelations], None] | None:
        if event_callback is None:
            return None

        def emit(round_index: int, relations: GrownRelations) -> None:
            scores = cp.asnumpy(relations.score)
            order = np.argsort(scores)[::-1][: min(48, len(scores))]
            parent_a = cp.asnumpy(relations.parent_a[order])
            parent_b = cp.asnumpy(relations.parent_b[order])
            event_callback(
                {
                    "type": event_type,
                    "organ": organ,
                    "round": round_index,
                    "relation_count": int(len(scores)),
                    "body_bytes": _relation_bytes(relations),
                    "best_score": float(np.max(scores)),
                    "median_score": float(np.median(scores)),
                    "sample_relations": [
                        {
                            "parent_a": int(left),
                            "parent_b": int(right),
                            "score": float(score),
                        }
                        for left, right, score in zip(
                            parent_a, parent_b, scores[order], strict=True
                        )
                    ],
                }
            )

        return emit

    start = time.perf_counter()
    train_visual_features = temporal_features(
        train_vectors[:, :, :3], sensory_timescales, cp
    )
    train_auditory_features = temporal_features(
        train_vectors[:, :, 3:], sensory_timescales, cp
    )
    train_positions = cp.asarray(np.stack([episode.positions for episode in train]))
    train_targets = cp.asarray(
        [[episode.target_position] * args.steps for episode in train]
    )
    example_vectors = {
        "training": stack_episode_vectors([train[0]], visible=True),
        "dark_validation": stack_episode_vectors([dark_example], visible=False),
    }
    example_visual_features = {
        name: temporal_features(vectors[:, :, :3], sensory_timescales, cp)
        for name, vectors in example_vectors.items()
    }

    visual_growth_event = progress("vision")

    def visual_progress(round_index: int, relations: GrownRelations) -> None:
        assert visual_growth_event is not None
        visual_growth_event(round_index, relations)
        preview_activation = relation_activations(
            train_visual_features, relations, cp
        )
        preview_beta = _fit_preview_head(preview_activation, train_positions, cp)
        event_callback(
            {
                "type": "prediction",
                "stage": f"vision growing · round {round_index}/{args.growth_rounds}",
                "predictions": {
                    name: cp.asnumpy(
                        predict(
                            relation_activations(features, relations, cp),
                            preview_beta,
                            cp,
                        )[0]
                    )
                    .astype(float)
                    .tolist()
                    for name, features in example_visual_features.items()
                },
            }
        )

    visual_relations = grow_relations(
        train_visual_features,
        _growth_args(args, args.visual_capacity),
        seed + 101,
        cp,
        progress_callback=visual_progress if event_callback is not None else None,
    )
    auditory_relations = grow_relations(
        train_auditory_features,
        _growth_args(args, args.auditory_capacity),
        seed + 211,
        cp,
        progress_callback=progress("audition"),
    )
    visual_activation = relation_activations(
        train_visual_features, visual_relations, cp
    )
    auditory_activation = relation_activations(
        train_auditory_features, auditory_relations, cp
    )

    candidate_visual_features = temporal_features(
        candidate_vectors[:, :, :3], sensory_timescales, cp
    )
    candidate_visual_activation = relation_activations(
        candidate_visual_features[:, -1], visual_relations, cp
    )
    candidate_positions = cp.asarray([episode.final_position for episode in candidate])
    candidate_targets = cp.asarray([episode.target_position for episode in candidate])
    visual_beta, visual_alpha, visual_candidate_mse = fit_visible_head(
        visual_activation,
        train_positions,
        candidate_visual_activation,
        candidate_positions,
        cp,
    )
    target_beta, target_alpha, target_candidate_mse = fit_visible_head(
        visual_activation,
        train_targets,
        candidate_visual_activation,
        candidate_targets,
        cp,
    )
    if event_callback is not None:
        prediction_sets = {}
        for name, episode, visible in (
            ("training", train[0], True),
            ("dark_validation", dark_example, False),
        ):
            vectors = stack_episode_vectors([episode], visible=visible)
            features = temporal_features(
                vectors[:, :, :3], sensory_timescales, cp
            )
            state = relation_activations(features, visual_relations, cp)
            prediction_sets[name] = cp.asnumpy(
                predict(state, visual_beta, cp)[0]
            ).astype(float).tolist()
        event_callback(
            {
                "type": "prediction",
                "stage": "visual organ calibrated",
                "predictions": prediction_sets,
            }
        )
    train_target_state = persist_target_state(
        predict(visual_activation, target_beta, cp),
        cp.asarray(train_vectors[:, :, 2]),
        cp,
    )
    train_integration_features = _integration_features(
        auditory_activation, train_target_state, integration_timescales, cp
    )
    integration_growth_event = progress("integration")

    def integration_progress(round_index: int, relations: GrownRelations) -> None:
        assert integration_growth_event is not None
        integration_growth_event(round_index, relations)
        preview_activation = relation_activations(
            train_integration_features, relations, cp
        )
        preview_beta = _fit_preview_head(preview_activation, train_positions, cp)
        prediction_sets = {}
        for name, vectors in example_vectors.items():
            visual_state, auditory_state, target_state, visibility = _organ_states(
                vectors,
                visual_relations,
                auditory_relations,
                target_beta,
                sensory_timescales,
                cp,
            )
            visual_estimate = predict(visual_state, visual_beta, cp)
            features = _integration_features(
                auditory_state, target_state, integration_timescales, cp
            )
            integrated_estimate = predict(
                relation_activations(features, relations, cp), preview_beta, cp
            )
            merged = (
                visibility * visual_estimate
                + (1.0 - visibility) * integrated_estimate
            )
            prediction_sets[name] = cp.asnumpy(merged[0]).astype(float).tolist()
        event_callback(
            {
                "type": "prediction",
                "stage": (
                    f"integration growing · round {round_index}/"
                    f"{args.growth_rounds}"
                ),
                "predictions": prediction_sets,
            }
        )

    integration_relations = grow_relations(
        train_integration_features,
        _growth_args(args, args.integration_capacity),
        seed + 307,
        cp,
        progress_callback=(
            integration_progress if event_callback is not None else None
        ),
    )
    integration_activation = relation_activations(
        train_integration_features, integration_relations, cp
    )

    _, candidate_auditory, candidate_target_state, _ = _organ_states(
        candidate_vectors,
        visual_relations,
        auditory_relations,
        target_beta,
        sensory_timescales,
        cp,
    )
    candidate_integration = _integration_features(
        candidate_auditory, candidate_target_state, integration_timescales, cp
    )
    candidate_integration_activation = relation_activations(
        candidate_integration[:, -1], integration_relations, cp
    )
    integration_beta, integration_alpha, integration_candidate_mse = fit_visible_head(
        integration_activation,
        train_positions,
        candidate_integration_activation,
        candidate_positions,
        cp,
    )
    if event_callback is not None:
        prediction_sets = {}
        for name, episode, visible in (
            ("training", train[0], True),
            ("dark_validation", dark_example, False),
        ):
            vectors = stack_episode_vectors([episode], visible=visible)
            visual_state, auditory_state, target_state, visibility = _organ_states(
                vectors,
                visual_relations,
                auditory_relations,
                target_beta,
                sensory_timescales,
                cp,
            )
            visual_estimate = predict(visual_state, visual_beta, cp)
            integration_features = _integration_features(
                auditory_state, target_state, integration_timescales, cp
            )
            integration_state = relation_activations(
                integration_features, integration_relations, cp
            )
            integrated_estimate = predict(
                integration_state, integration_beta, cp
            )
            merged = (
                visibility * visual_estimate
                + (1.0 - visibility) * integrated_estimate
            )
            prediction_sets[name] = cp.asnumpy(merged[0]).astype(float).tolist()
        event_callback(
            {
                "type": "prediction",
                "stage": "three-organ fusion calibrated",
                "predictions": prediction_sets,
            }
        )
    if model_callback is not None:
        adaptive_visual_relations: GrownRelations | None = None
        adaptive_visual_beta: Any | None = None

        def infer_retargeted(
            target_position: float,
            episode_name: str,
            learning_enabled: bool,
            velocity_scale: float = 1.0,
        ) -> dict[str, Any]:
            nonlocal adaptive_visual_beta, adaptive_visual_relations
            is_training = episode_name == "training"
            is_continuous = episode_name == "continuous"
            if is_training:
                episode_name = "training"
                episode = retarget_episode(train[0], target_position)
            elif is_continuous:
                episode_name = "continuous"
                episode = _continuous_episode(
                    target_position, velocity_scale=velocity_scale
                )
            else:
                episode_name = "dark_validation"
                episode = retarget_episode(dark_example, target_position)
            visible = is_training or is_continuous
            vectors = stack_episode_vectors([episode], visible=visible)
            learning_applied = False
            if is_continuous:
                features = temporal_features(
                    vectors[:, :, :3], sensory_timescales, cp
                )
                if learning_enabled and adaptive_visual_relations is None:
                    adaptive_visual_relations = grow_relations(
                        features,
                        _growth_args(args, args.visual_capacity),
                        seed + 419,
                        cp,
                        progress_callback=progress("vision", "continuous_growth"),
                    )
                    adaptive_activation = relation_activations(
                        features, adaptive_visual_relations, cp
                    )
                    adaptive_visual_beta = _fit_preview_head(
                        adaptive_activation,
                        cp.asarray(episode.positions)[None, :],
                        cp,
                    )
                    learning_applied = True
                active_relations = (
                    adaptive_visual_relations
                    if adaptive_visual_relations is not None
                    else visual_relations
                )
                active_beta = (
                    adaptive_visual_beta
                    if adaptive_visual_beta is not None
                    else visual_beta
                )
                visual_state = relation_activations(
                    features, active_relations, cp
                )
                merged = predict(visual_state, active_beta, cp)
                return {
                    "type": "target_update",
                    "episode_name": episode_name,
                    "episode": _visualizer_episode(episode, visible=True),
                    "alm_predictions": cp.asnumpy(merged[0])
                    .astype(float)
                    .tolist(),
                    "dead_reckoning_predictions": [],
                    "learning_enabled": learning_enabled,
                    "learning_applied": learning_applied,
                }
            visual_state, auditory_state, target_state, visibility = _organ_states(
                vectors,
                visual_relations,
                auditory_relations,
                target_beta,
                sensory_timescales,
                cp,
            )
            visual_estimate = predict(visual_state, visual_beta, cp)
            features = _integration_features(
                auditory_state, target_state, integration_timescales, cp
            )
            integration_state = relation_activations(
                features, integration_relations, cp
            )
            integrated_estimate = predict(
                integration_state, integration_beta, cp
            )
            merged = (
                visibility * visual_estimate
                + (1.0 - visibility) * integrated_estimate
            )
            return {
                "type": "target_update",
                "episode_name": episode_name,
                "episode": _visualizer_episode(episode, visible=visible),
                "alm_predictions": cp.asnumpy(merged[0]).astype(float).tolist(),
                "dead_reckoning_predictions": _dead_reckoning_trajectory(episode),
                "learning_enabled": learning_enabled,
                "learning_applied": False,
            }

        model_callback(infer_retargeted)
    cp.cuda.Stream.null.synchronize()
    train_seconds = time.perf_counter() - start

    organ_bytes = {
        "vision": _relation_bytes(visual_relations) + _readout_bytes(
            visual_beta, target_beta
        ),
        "audition": _relation_bytes(auditory_relations),
        "integration": _relation_bytes(integration_relations)
        + _readout_bytes(integration_beta),
    }
    grown_body_bytes = int(sum(organ_bytes.values()))
    if grown_body_bytes > args.max_body_bytes:
        raise MemoryError(
            f"grown anatomy {grown_body_bytes:,} B exceeds ALM-64MB ceiling"
        )

    def evaluate(episodes: list[Any], *, visible: bool, audio_mode: str = "real"):
        vectors = stack_episode_vectors(
            episodes, visible=visible, audio_mode=audio_mode
        )
        visual_state, auditory_state, target_state, visibility = _organ_states(
            vectors,
            visual_relations,
            auditory_relations,
            target_beta,
            sensory_timescales,
            cp,
        )
        visual_estimate = predict(visual_state[:, -1], visual_beta, cp)
        integration_features = _integration_features(
            auditory_state, target_state, integration_timescales, cp
        )
        integration_state = relation_activations(
            integration_features[:, -1], integration_relations, cp
        )
        integrated_estimate = predict(integration_state, integration_beta, cp)
        visual_support = visibility[:, -1]
        merged = (
            visual_support * visual_estimate
            + (1.0 - visual_support) * integrated_estimate
        )
        truth = cp.asarray([episode.final_position for episode in episodes])
        return truth, merged, visual_estimate, integrated_estimate

    multi_y, multi_p, _, _ = evaluate(test, visible=True)
    dark_y, dark_p, _, dark_integration = evaluate(test, visible=False)
    shuffled = shuffled_ranging_episodes(test, seed=base + 41)
    shuffled_y, shuffled_p, _, _ = evaluate(shuffled, visible=False)
    silent_y, silent_p, _, _ = evaluate(test, visible=False, audio_mode="silent")
    dark_mse = mse(dark_y, dark_p, cp)
    shuffled_mse = mse(shuffled_y, shuffled_p, cp)
    silent_mse = mse(silent_y, silent_p, cp)
    dead = cp.asarray([dead_reckoning_prediction(episode) for episode in test])
    dead_mse = mse(dark_y, dead, cp)
    causal_pass = dark_mse < dead_mse and dark_mse < shuffled_mse
    return {
        "seed": seed,
        "held_out_multimodal_position_r2": r2(multi_y, multi_p, cp),
        "dark_audio_position_mse": dark_mse,
        "integration_only_dark_mse": mse(dark_y, dark_integration, cp),
        "dead_reckoning_mse": dead_mse,
        "shuffled_audio_mse": shuffled_mse,
        "silent_audio_mse": silent_mse,
        "audio_vs_dead_reckoning_improvement": dead_mse / max(dark_mse, EPS),
        "real_vs_shuffled_audio_improvement": shuffled_mse / max(dark_mse, EPS),
        "causal_audio_pass": causal_pass,
        "strict_smoke_pass": r2(multi_y, multi_p, cp) > 0.80 and causal_pass,
        "training_seconds": train_seconds,
        "training_frames_per_second": len(train) * args.steps / train_seconds,
        "visual_candidate_mse": visual_candidate_mse,
        "target_candidate_mse": target_candidate_mse,
        "integration_candidate_mse": integration_candidate_mse,
        "visual_ridge": visual_alpha,
        "target_ridge": target_alpha,
        "integration_ridge": integration_alpha,
        "auditory_cross_timescale_relations": _cross_timescale_audio_count(
            auditory_relations, 2, cp
        ),
        "body_class_bytes": args.max_body_bytes,
        "grown_body_bytes": grown_body_bytes,
        "unused_carrying_capacity_bytes": args.max_body_bytes - grown_body_bytes,
        "organ_bytes": organ_bytes,
        "organ_relation_counts": {
            "vision": int(len(visual_relations.score)),
            "audition": int(len(auditory_relations.score)),
            "integration": int(len(integration_relations.score)),
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
    result: dict[str, Any] = {}
    for name in names:
        values = np.asarray([report[name] for report in reports])
        result[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    result["strict_smoke_pass_count"] = sum(r["strict_smoke_pass"] for r in reports)
    result["causal_audio_pass_count"] = sum(r["causal_audio_pass"] for r in reports)
    result["all_seeds_pass"] = all(r["strict_smoke_pass"] for r in reports)
    body = np.asarray([report["grown_body_bytes"] for report in reports])
    result["grown_body_bytes"] = {
        "mean": float(body.mean()),
        "min": int(body.min()),
        "max": int(body.max()),
    }
    return result


def main() -> None:
    args = parse_args()
    cp = _cupy()
    reports = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(f"three-organ echolocation seed {seed}", file=sys.stderr, flush=True)
        reports.append(run_seed(args, seed, cp))
    report = {
        "experiment": "alm_64mb_1c_three_organ_native_vector_echolocation",
        "alm_name": "ALM-64MB-1C",
        "body_class_bytes": args.max_body_bytes,
        "developmental_columns": args.columns,
        "execution": "cupy_cuda_no_autograd",
        "organs": ["vision", "audition", "integration"],
        "organ_topology": (
            "one developmental column containing differentiated visual, auditory, "
            "and integration organs; vision has 4x auditory relation capacity"
        ),
        "input_contract": (
            "five simultaneous float32 sensory channels; no tokenization, learned "
            "embedding, distance, velocity, or cadence scalar"
        ),
        "body_contract": (
            "64MB is a hard organism-wide ceiling; grown anatomy is measured "
            "separately and may remain smaller at smoke-test exposure"
        ),
        "reliability_rule": (
            "vision dominates while supported; integration owns darkness"
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
