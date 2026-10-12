"""Phase 7 (design doc section 54: "Dense continuous vectors -- Test
arbitrary continuous streams. No sparse categorical assumptions."): does
relation discovery, run entirely on dense real-valued vectors (no
one-hot, no discrete vocabulary anywhere), improve next-value prediction
on a continuous oscillator stream over a naive persistence baseline?

Direction (+1 or -1 angular step per event) is fixed for an episode but
unresolvable from the instantaneous (sin, cos) position alone -- the same
structural ambiguity as Phase 5's ring task, just continuous. A
persistence baseline ("next value equals current value") is already
fairly strong for a slowly-varying oscillator, so this is a harder bar
to clear than Phase 5's unigram baseline was.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.continuous_relation_discovery import (
    build_continuous_diary,
    discover_and_rank_continuous_relations,
)
from manifoldflow.dendritic_basin.tasks.continuous_oscillator import (
    N_CHANNELS,
    RESET_CHANNEL,
    SIN_CHANNEL,
    generate_oscillator_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-episode", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--angular-step", type=float, default=0.35)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    timescales = (1, 2, 4, 8)
    current_channel = SIN_CHANNEL
    target_channel = SIN_CHANNEL

    train_stream = generate_oscillator_stream(
        args.train_episodes,
        args.steps_per_episode,
        args.angular_step,
        args.noise_std,
        rng,
    )
    train_diary = build_continuous_diary(
        train_stream, timescales, boundary_channel=RESET_CHANNEL
    )

    ranked = discover_and_rank_continuous_relations(
        train_diary,
        current_channel=current_channel,
        target_channel=target_channel,
        n_channels=N_CHANNELS,
        n_timescales=len(timescales),
        seed=args.seed + 1,
        min_gain=args.min_gain,
        baseline_channel=current_channel,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=5,
    )

    # Continuous analogue of Phase 5's empirically-learned vote targets:
    # each relation's prediction is the mean *residual* (actual next value
    # minus the persistence baseline's own prediction) among training
    # positions where it fires -- consistent with baseline_channel above,
    # so a relation only contributes what persistence doesn't already
    # capture.
    relation_residual_means: dict[tuple[int, int], float] = {}
    for key, unit, _gain in ranked:
        trace_channel, timescale_index = key
        firing_residuals = []
        for x, q, x_next in train_diary:
            a = float(x[current_channel])
            b = float(q[timescale_index, trace_channel])
            if unit.respond(a, b) > 0.5:
                firing_residuals.append(
                    float(x_next[target_channel]) - float(x[current_channel])
                )
        relation_residual_means[key] = (
            float(np.mean(firing_residuals)) if firing_residuals else 0.0
        )

    eval_stream = generate_oscillator_stream(
        args.eval_episodes,
        args.steps_per_episode,
        args.angular_step,
        args.noise_std,
        rng,
    )
    eval_diary = build_continuous_diary(
        eval_stream, timescales, boundary_channel=RESET_CHANNEL
    )

    persistence_errors = []
    relation_errors = []
    for x, q, x_next in eval_diary:
        actual = float(x_next[target_channel])
        persistence_prediction = float(x[current_channel])
        persistence_errors.append((actual - persistence_prediction) ** 2)

        weighted_residual_sum, weight_total = 0.0, 0.0
        for key, unit, gain in ranked:
            trace_channel, timescale_index = key
            a = float(x[current_channel])
            b = float(q[timescale_index, trace_channel])
            if unit.respond(a, b) > 0.5:
                weighted_residual_sum += gain * relation_residual_means[key]
                weight_total += gain
        predicted_residual = (
            weighted_residual_sum / weight_total if weight_total > 0 else 0.0
        )
        relation_prediction = persistence_prediction + predicted_residual
        relation_errors.append((actual - relation_prediction) ** 2)

    report = {
        "discovered_relations": [
            {"trace_channel": ch, "timescale_index": ti, "gain": g}
            for (ch, ti), _u, g in ranked
        ],
        "persistence_baseline_mse": float(np.mean(persistence_errors)),
        "relation_blended_mse": float(np.mean(relation_errors)),
        "n_eval_positions": len(eval_diary),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
