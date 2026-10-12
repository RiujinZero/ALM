"""Phase 5 (design doc section 54: "Generic vector relation learner" /
section 31's "data-agnostic vector smoke"): does the *exact same*
discovery mechanism validated on language (Phases 2-4, unmodified code)
also improve prediction on a structurally unrelated, non-language event
stream -- a point moving around a fixed-size ring, direction unknown from
the current position alone?

No language-specific assumption anywhere in `relation_discovery.py` is
touched for this task; only the sensory adapter (`tasks/moving_point.py`)
differs, per the design's own "only the sensory adapter may differ"
data-agnostic success criterion (section 57).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.prediction_table import (
    PredictionTable,
    bits_per_token,
    mix_distributions,
)
from manifoldflow.dendritic_basin.relation_discovery import (
    build_stream_diary,
    discover_and_rank_relations,
)
from manifoldflow.dendritic_basin.tasks.moving_point import (
    generate_position_stream,
    vocab_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-size", type=int, default=8)
    parser.add_argument("--steps-per-episode", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--min-gain", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    v = vocab_size(args.ring_size)
    timescales = (1, 2, 4, 8)
    anchor = 1 + 0  # channel for ring position 0 (position channels are offset by 1)

    train_stream = generate_position_stream(
        args.train_episodes, args.ring_size, args.steps_per_episode, rng
    )
    train_diary = build_stream_diary(train_stream, v, timescales)

    ranked = discover_and_rank_relations(
        train_diary,
        anchor=anchor,
        vocab_size=v,
        n_timescales=len(timescales),
        seed=args.seed + 1,
        min_gain=args.min_gain,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=5,
    )
    # Position channels only (exclude the episode-boundary marker channel 0,
    # which is a distinct structural signal, not a ring position).
    ranked = [item for item in ranked if item[0][0] != 0]

    # What a relation predicts when active must be *learned* from data, not
    # assumed: a generic dendrite can converge to either the "fires when
    # co-active" or "fires when anti-co-active" family (section 8 of the
    # v0.2 interaction-dendrite addendum), so "activation > 0.5" does not
    # reliably mean the same thing across different discovered relations.
    # The correct vote is whichever next-position was actually most common
    # among training positions where this specific relation fired.
    vote_targets: dict[tuple[int, int], int] = {}
    for key, unit, _gain in ranked:
        trace_channel, timescale_index = key
        counts = np.zeros(v)
        for token_id, q, next_token in train_diary:
            if token_id != anchor:
                continue
            activation = unit.respond(1.0, float(q[timescale_index, trace_channel]))
            if activation > 0.5:
                counts[next_token] += 1
        vote_targets[key] = int(np.argmax(counts))

    unigram = PredictionTable(v, decay=1.0, lr=1.0)
    for token_id in train_stream:
        unigram.update(1.0, token_id)
    unigram_dist = unigram.distribution()

    eval_stream = generate_position_stream(
        args.eval_episodes, args.ring_size, args.steps_per_episode, rng
    )
    eval_diary = build_stream_diary(eval_stream, v, timescales)

    unigram_probs, relation_probs = [], []
    correct = 0
    total = 0
    for token_id, q, next_token in eval_diary:
        p_unigram = unigram_dist[next_token]
        unigram_probs.append(p_unigram)
        if token_id != anchor:
            relation_probs.append(p_unigram)
            continue
        total += 1
        weighted = []
        for key, unit, gain in ranked:
            trace_channel, timescale_index = key
            activation = unit.respond(1.0, float(q[timescale_index, trace_channel]))
            if activation <= 0.5:
                continue
            target_dist = np.zeros(v)
            target_dist[vote_targets[key]] = 1.0
            weighted.append((gain, target_dist))
        mixed = mix_distributions(unigram_dist, weighted)
        relation_probs.append(mixed[next_token])
        correct += int(np.argmax(mixed) == next_token)

    report = {
        "discovered_relations": [
            {"position": ch - 1, "timescale_index": ti, "gain_bits": g}
            for (ch, ti), _u, g in ranked
        ],
        "unigram_bits_per_token": bits_per_token(unigram_probs),
        "relation_bits_per_token": bits_per_token(relation_probs),
        "direction_accuracy_at_anchor": correct / total if total else 0.0,
        "n_anchor_positions": total,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
