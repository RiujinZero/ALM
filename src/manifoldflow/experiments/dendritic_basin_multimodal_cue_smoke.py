"""Phase 11 (design doc section 54: "Multimodal stream -- Vision + audio
+ language using the same relational core."; section 58: cross-modal
assemblies should emerge). Does `continuous_relation_discovery.py`,
completely unmodified since Phase 10, discover a genuinely *cross-modal*
relation -- the visual cue's trace predicting the audio target -- when
an irrelevant, same-vector "language-like" decoy channel is also
present?

No new discovery mechanism is introduced this phase: the entire point of
sections 54/57/58 is that the *same* relational core should handle this
without modification, only the task (sensory composition) differs. The
check that matters here is not just "does accuracy improve" (Phase 10
already showed that) but "is the *specific* relation discovered actually
cross-modal" -- i.e. does its trace_channel come from the vision channel,
not from the audio modality itself or the irrelevant decoy.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.continuous_relation_discovery import (
    build_continuous_diary,
    discover_and_rank_continuous_relations,
)
from manifoldflow.dendritic_basin.tasks.multimodal_cue import (
    AMBIGUOUS_TONE,
    BOUNDARY,
    DECOY,
    E_BIN,
    N_CHANNELS,
    VISUAL_CUE,
    generate_multimodal_stream,
)

MODALITY_BY_CHANNEL = {
    BOUNDARY: "language",
    VISUAL_CUE: "vision",
    DECOY: "language",
    AMBIGUOUS_TONE: "audio",
    E_BIN: "audio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--min-gain", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    timescales = (1, 2, 4, 8)
    current_channel = AMBIGUOUS_TONE
    target_channel = E_BIN

    train_stream = generate_multimodal_stream(args.train_episodes, args.noise_std, rng)
    train_diary = build_continuous_diary(
        train_stream, timescales, boundary_channel=BOUNDARY
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
            if x[current_channel] <= 0.5:
                continue
            b = float(q[timescale_index, trace_channel])
            if unit.respond(float(x[current_channel]), b) > 0.5:
                firing_values.append(float(x_next[target_channel]))
        relation_means[key] = float(np.mean(firing_values)) if firing_values else 0.5

    eval_stream = generate_multimodal_stream(args.eval_episodes, args.noise_std, rng)
    eval_diary = build_continuous_diary(
        eval_stream, timescales, boundary_channel=BOUNDARY
    )

    baseline_errors, relation_errors = [], []
    correct_baseline, correct_relation, total = 0, 0, 0
    for x, q, x_next in eval_diary:
        if x[current_channel] <= 0.5:
            continue
        total += 1
        actual = float(x_next[target_channel])
        baseline_prediction = 0.5
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

    cross_modal_relations = [
        {
            "trace_channel": ch,
            "trace_modality": MODALITY_BY_CHANNEL.get(ch, "unknown"),
            "timescale_index": ti,
            "gain": g,
        }
        for (ch, ti), _u, g in ranked
    ]
    top_relation_modality = (
        cross_modal_relations[0]["trace_modality"] if cross_modal_relations else None
    )

    report = {
        "discovered_relations": cross_modal_relations,
        "top_relation_is_cross_modal": top_relation_modality == "vision",
        "any_relation_used_the_decoy_channel": any(
            r["trace_channel"] == DECOY for r in cross_modal_relations
        ),
        "baseline_mse": float(np.mean(baseline_errors)) if baseline_errors else None,
        "relation_mse": float(np.mean(relation_errors)) if relation_errors else None,
        "baseline_accuracy": correct_baseline / total if total else None,
        "relation_accuracy": correct_relation / total if total else None,
        "n_eval_positions": total,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
