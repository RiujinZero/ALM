"""Phase 10 (design doc section 54: "Audio -- Continuous frequency
vectors."): does Phase 7's continuous relation-discovery machinery
(`continuous_relation_discovery.py`), *completely unmodified*, resolve a
Phase-2-style ambiguity on multi-bin "frequency vector" data instead of
one-hot tokens or a 2-3 channel toy vector -- section 57's "only the
sensory adapter may differ" applied to a higher-dimensional continuous
stream?

The ambiguous tone's own frequency vector carries zero phrase
information by construction; only the earlier `A`-vs-`C` tone's trace
resolves whether `E` or `F` comes next. `current_channel` is fixed to
the ambiguous tone's own bin (which is why a persistence baseline isn't
needed here the way Phase 7's oscillator needed one: the current value
carries no autocorrelation with the target by construction, so there is
no trivial signal for a relation to redundantly rediscover).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.continuous_relation_discovery import (
    build_continuous_diary,
    discover_and_rank_continuous_relations,
)
from manifoldflow.dendritic_basin.tasks.audio_motif import (
    AMBIGUOUS_BIN,
    BOUNDARY_BIN,
    E_BIN,
    N_CHANNELS,
    generate_motif_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--min-gain", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    timescales = (1, 2, 4, 8)
    current_channel = AMBIGUOUS_BIN
    target_channel = E_BIN  # near 1.0 if E follows (phrase 1), near 0.0 if F follows

    train_stream = generate_motif_stream(args.train_episodes, args.noise_std, rng)
    train_diary = build_continuous_diary(
        train_stream, timescales, boundary_channel=BOUNDARY_BIN
    )

    ranked = discover_and_rank_continuous_relations(
        train_diary,
        current_channel=current_channel,
        target_channel=target_channel,
        n_channels=N_CHANNELS,
        n_timescales=len(timescales),
        seed=args.seed + 1,
        min_gain=args.min_gain,
        anchor_active_threshold=0.5,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=5,
    )

    relation_means: dict[tuple[int, int], float] = {}
    for key, unit, _gain in ranked:
        trace_channel, timescale_index = key
        firing_values = []
        for x, q, x_next in train_diary:
            if x[current_channel] <= 0.5:  # same anchor-active gate as discovery
                continue
            a = float(x[current_channel])
            b = float(q[timescale_index, trace_channel])
            if unit.respond(a, b) > 0.5:
                firing_values.append(float(x_next[target_channel]))
        relation_means[key] = float(np.mean(firing_values)) if firing_values else 0.5

    eval_stream = generate_motif_stream(args.eval_episodes, args.noise_std, rng)
    eval_diary = build_continuous_diary(
        eval_stream, timescales, boundary_channel=BOUNDARY_BIN
    )

    baseline_errors, relation_errors = [], []
    correct_baseline, correct_relation, total = 0, 0, 0
    for x, q, x_next in eval_diary:
        if x[current_channel] <= 0.5:  # only score at the ambiguous-tone position
            continue
        total += 1
        actual = float(x_next[target_channel])
        baseline_prediction = 0.5  # unconditional midpoint: no phrase info at all
        baseline_errors.append((actual - baseline_prediction) ** 2)
        correct_baseline += int(round(baseline_prediction) == round(actual))

        weighted_sum, weight_total = 0.0, 0.0
        for key, unit, gain in ranked:
            trace_channel, timescale_index = key
            b = float(q[timescale_index, trace_channel])
            if unit.respond(float(x[current_channel]), b) > 0.5:
                weighted_sum += gain * relation_means[key]
                weight_total += gain
        relation_prediction = (
            weighted_sum / weight_total if weight_total > 0 else baseline_prediction
        )
        relation_errors.append((actual - relation_prediction) ** 2)
        correct_relation += int(round(relation_prediction) == round(actual))

    report = {
        "discovered_relations": [
            {"trace_channel": ch, "timescale_index": ti, "gain": g}
            for (ch, ti), _u, g in ranked
        ],
        "baseline_mse": float(np.mean(baseline_errors)) if baseline_errors else None,
        "relation_mse": float(np.mean(relation_errors)) if relation_errors else None,
        "baseline_accuracy": correct_baseline / total if total else None,
        "relation_accuracy": correct_relation / total if total else None,
        "n_eval_positions": total,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
