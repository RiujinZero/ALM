"""Phase 8 (design doc section 54: "Larger raw language stream -- Move
to TinyStories and then unrelated text such as Wikipedia"; section 55:
"Start with raw UTF-8 TinyStories... Feed 1-10 MB continuously for the
first run... Measure bits/byte"): the first run of this project's
relation-discovery machinery on real downloaded text, not a synthetic
template.

Per the active `/goal` instruction, this stays CPU-only and at a scale
the sandbox rig can actually run in minutes, not section 55's GPU-scale
1-10 MB (that line explicitly says "create a GPU implementation" for
that scale, which this project is deliberately not doing while a
separate GPU training run is in progress elsewhere). Real vocab size
(256 raw bytes) makes Phase 6's bounded self-organizing population a
hard requirement here, not an optimization: an exhaustive
(256 channels x n_timescales) scan per anchor, times several anchors,
times storing a full (n_timescales, 256) trace snapshot per diary
position, is exactly the dense-scan / dense-memory cost section 50 says
to avoid, and was previously only a theoretical concern -- this is the
first phase where it actually bites (see the implementation journal).

Anchors are the most frequent byte values in the training slice --
a generic, data-agnostic prioritization rule (not a hand-picked list of
specific characters), so this generalizes to Phase 8's later
cross-dataset test (section 56, Wikipedia/code/conversational text)
without modification.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from manifoldflow.dendritic_basin.prediction_table import (
    PredictionTable,
    bits_per_token,
    mix_distributions,
)
from manifoldflow.dendritic_basin.relation_discovery import build_stream_diary
from manifoldflow.dendritic_basin.self_organizing_population import (
    discover_relations_by_self_organizing_population,
)
from manifoldflow.dendritic_basin.tasks.tinystories_bytes import (
    BYTE_VOCAB_SIZE,
    load_byte_stream,
)

DEFAULT_TRAIN_PATH = r"C:\path\to\gdrive\training_data\TinyStories-train.txt"
DEFAULT_VALID_PATH = r"C:\path\to\gdrive\training_data\TinyStories-valid.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=50_000)
    parser.add_argument("--eval-bytes", type=int, default=10_000)
    parser.add_argument("--n-anchors", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--max-candidates-per-slot", type=int, default=12)
    parser.add_argument("--min-gain", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timescales = (1, 2, 4, 8)

    train_stream = load_byte_stream(args.train_path, args.train_bytes)
    eval_stream = load_byte_stream(args.valid_path, args.eval_bytes)

    byte_counts = Counter(train_stream)
    anchors = [byte for byte, _count in byte_counts.most_common(args.n_anchors)]

    train_diary = build_stream_diary(train_stream, BYTE_VOCAB_SIZE, timescales)
    eval_diary = build_stream_diary(eval_stream, BYTE_VOCAB_SIZE, timescales)

    unigram = PredictionTable(BYTE_VOCAB_SIZE, decay=1.0, lr=1.0)
    for token_id in train_stream:
        unigram.update(1.0, token_id)
    unigram_dist = unigram.distribution()

    per_anchor_reports = []
    total_evaluations = 0
    start = time.perf_counter()

    for anchor in anchors:
        ranked, stats = discover_relations_by_self_organizing_population(
            train_diary,
            anchor=anchor,
            vocab_size=BYTE_VOCAB_SIZE,
            n_timescales=len(timescales),
            seed=args.seed + anchor,
            population_size=args.population_size,
            max_candidates_per_slot=args.max_candidates_per_slot,
            min_gain=args.min_gain,
            n_candidates=3,
            rounds=40,
            perturbations_per_round=12,
            episodes_per_candidate=3,
        )
        total_evaluations += stats.total_candidate_evaluations
        ranked_items = [(r.key, r.unit, r.gain) for r in ranked]

        vote_targets: dict[tuple[int, int], int] = {}
        for key, unit, _gain in ranked_items:
            trace_channel, timescale_index = key
            counts = np.zeros(BYTE_VOCAB_SIZE)
            for token_id, q, next_token in train_diary:
                if token_id != anchor:
                    continue
                activation = unit.respond(1.0, float(q[timescale_index, trace_channel]))
                if activation > 0.5:
                    counts[next_token] += 1
            vote_targets[key] = int(np.argmax(counts))

        unigram_probs, relation_probs = [], []
        correct = 0
        total = 0
        for token_id, q, next_token in eval_diary:
            if token_id != anchor:
                continue
            total += 1
            p_unigram = unigram_dist[next_token]
            unigram_probs.append(p_unigram)
            weighted = []
            for key, unit, gain in ranked_items:
                trace_channel, timescale_index = key
                activation = unit.respond(1.0, float(q[timescale_index, trace_channel]))
                if activation > 0.5:
                    target_dist = np.zeros(BYTE_VOCAB_SIZE)
                    target_dist[vote_targets[key]] = 1.0
                    weighted.append((gain, target_dist))
            mixed = mix_distributions(unigram_dist, weighted)
            relation_probs.append(mixed[next_token])
            correct += int(np.argmax(mixed) == next_token)

        anchor_char = chr(anchor) if 32 <= anchor < 127 else repr(bytes([anchor]))
        per_anchor_reports.append(
            {
                "anchor_byte": anchor,
                "anchor_char": anchor_char,
                "n_discovered_relations": len(ranked_items),
                "candidate_evaluations": stats.total_candidate_evaluations,
                "n_eval_positions": total,
                "unigram_bits_per_position": (
                    bits_per_token(unigram_probs) if unigram_probs else None
                ),
                "relation_bits_per_position": (
                    bits_per_token(relation_probs) if relation_probs else None
                ),
                "accuracy": correct / total if total else None,
            }
        )

    elapsed = time.perf_counter() - start
    report = {
        "train_bytes": len(train_stream),
        "eval_bytes": len(eval_stream),
        "anchors": per_anchor_reports,
        "total_candidate_evaluations": total_evaluations,
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
