"""Phase 6 (design doc section 54: "Adaptive bell dendrites on generic
vectors"): does the bounded self-organizing population
(`self_organizing_population.py`) match the exhaustive enumeration
(`relation_discovery.discover_and_rank_relations`) on prediction quality
while evaluating far fewer candidates, on a ring large enough that
exhaustive scanning actually costs something?

Reuses the moving-point task from Phase 5 unmodified, just with a larger
ring -- the exhaustive search's cost grows with `ring_size`, the
population's does not (it is bounded by
`population_size * max_candidates_per_slot`).
"""

from __future__ import annotations

import argparse
import json
import time

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
from manifoldflow.dendritic_basin.self_organizing_population import (
    discover_relations_by_self_organizing_population,
)
from manifoldflow.dendritic_basin.tasks.moving_point import (
    generate_position_stream,
    vocab_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-size", type=int, default=32)
    parser.add_argument("--steps-per-episode", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--min-gain", type=float, default=0.2)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--max-candidates-per-slot", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _score(ranked_items, train_diary, eval_diary, anchor: int, v: int) -> dict:
    """Shared scoring: empirically-learned vote targets (Phase 5's fix),
    then bits/token and direction accuracy at the anchor -- identical
    scoring logic for both the exhaustive and population candidate lists
    so the comparison is only about search strategy, not evaluation.
    """
    vote_targets: dict[tuple[int, int], int] = {}
    for key, unit, _gain in ranked_items:
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
    for token_id, _q, _next in train_diary:
        unigram.update(1.0, token_id)
    unigram_dist = unigram.distribution()

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
        for key, unit, gain in ranked_items:
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

    return {
        "n_discovered_relations": len(ranked_items),
        "unigram_bits_per_token": bits_per_token(unigram_probs),
        "relation_bits_per_token": bits_per_token(relation_probs),
        "direction_accuracy_at_anchor": correct / total if total else 0.0,
        "n_anchor_positions": total,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    v = vocab_size(args.ring_size)
    timescales = (1, 2, 4, 8)
    anchor = 1 + 0

    train_stream = generate_position_stream(
        args.train_episodes, args.ring_size, args.steps_per_episode, rng
    )
    train_diary = build_stream_diary(train_stream, v, timescales)
    eval_stream = generate_position_stream(
        args.eval_episodes, args.ring_size, args.steps_per_episode, rng
    )
    eval_diary = build_stream_diary(eval_stream, v, timescales)

    exhaustive_candidate_count = (v - 1) * len(timescales)

    start = time.perf_counter()
    exhaustive_ranked = discover_and_rank_relations(
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
    exhaustive_ranked = [item for item in exhaustive_ranked if item[0][0] != 0]
    exhaustive_seconds = time.perf_counter() - start

    start = time.perf_counter()
    population_results, stats = discover_relations_by_self_organizing_population(
        train_diary,
        anchor=anchor,
        vocab_size=v,
        n_timescales=len(timescales),
        seed=args.seed + 1,
        population_size=args.population_size,
        max_candidates_per_slot=args.max_candidates_per_slot,
        min_gain=args.min_gain,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=5,
    )
    population_ranked = [
        (r.key, r.unit, r.gain) for r in population_results if r.key[0] != 0
    ]
    population_seconds = time.perf_counter() - start

    report = {
        "ring_size": args.ring_size,
        "exhaustive": {
            "candidate_space_size": exhaustive_candidate_count,
            "seconds": exhaustive_seconds,
            **_score(exhaustive_ranked, train_diary, eval_diary, anchor, v),
        },
        "self_organizing_population": {
            "candidate_evaluations": stats.total_candidate_evaluations,
            "candidate_budget": stats.population_size * stats.max_candidates_per_slot,
            "seconds": population_seconds,
            **_score(population_ranked, train_diary, eval_diary, anchor, v),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
