"""Phase 2 (design doc section 54: "Limited-vocabulary stream" /
Smoke 1 of `Dendritic_Basin_Learning_Design_v0.3_Language_Smoke_Findings.md`),
for real: does automatic temporal-relation discovery (real
InteractionDendrites over real temporal traces, ranked by predictive
information gain -- not the sandbox's coincidence-detector shortcut)
reach near-oracle bits/token on the ambiguous-object task?

  <BOS> alice sees cat .
  <BOS> bob sees dog .
  <BOS> carol sees bird .
  <BOS> dave sees fish .

Compares: unigram baseline, discovered-relation predictor, and the
exact-trigram oracle (perfect knowledge of which name preceded "sees").
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
from manifoldflow.dendritic_basin.tasks.ambiguous_object import (
    NAME_TO_OBJECT,
    TOKEN_TO_ID,
    VOCAB_SIZE,
    generate_token_stream,
)

CHANNEL_TO_OBJECT_TOKEN = {
    TOKEN_TO_ID[name]: TOKEN_TO_ID[obj] for name, obj in NAME_TO_OBJECT.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sentences", type=int, default=500)
    parser.add_argument("--eval-sentences", type=int, default=500)
    parser.add_argument("--timescales", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--min-gain", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def relation_prediction(
    ranked: list, q: np.ndarray, unigram_dist: np.ndarray
) -> np.ndarray:
    """Mixture of the unigram baseline with every currently-active
    discovered relation's (one-hot object) prediction, weighted by that
    relation's information gain.
    """
    weighted = []
    for (trace_channel, timescale_index), unit, gain in ranked:
        object_token = CHANNEL_TO_OBJECT_TOKEN.get(trace_channel)
        if object_token is None:
            continue
        activation = unit.respond(1.0, float(q[timescale_index, trace_channel]))
        if activation > 0.5:
            target_dist = np.zeros(VOCAB_SIZE)
            target_dist[object_token] = 1.0
            weighted.append((gain, target_dist))
    return mix_distributions(unigram_dist, weighted)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    timescales = tuple(args.timescales)
    anchor = TOKEN_TO_ID["sees"]

    train_stream = generate_token_stream(args.train_sentences, rng)
    train_diary = build_stream_diary(train_stream, VOCAB_SIZE, timescales)

    ranked = discover_and_rank_relations(
        train_diary,
        anchor=anchor,
        vocab_size=VOCAB_SIZE,
        n_timescales=len(timescales),
        seed=args.seed + 1,
        min_gain=args.min_gain,
        n_candidates=3,
        rounds=60,
        perturbations_per_round=12,
        episodes_per_candidate=5,
    )
    discovered = [
        {"trace_channel": key[0], "timescale_index": key[1], "gain_bits": gain}
        for key, _unit, gain in ranked
    ]

    unigram = PredictionTable(VOCAB_SIZE, decay=1.0, lr=1.0)
    for token_id in train_stream:
        unigram.update(1.0, token_id)
    unigram_dist = unigram.distribution()

    eval_stream = generate_token_stream(args.eval_sentences, rng)
    eval_diary = build_stream_diary(eval_stream, VOCAB_SIZE, timescales)

    unigram_probs = []
    relation_probs = []
    object_position_unigram_probs = []
    object_position_relation_probs = []
    object_position_oracle_probs = []
    for token_id, q, next_token in eval_diary:
        p_unigram = unigram_dist[next_token]
        unigram_probs.append(p_unigram)

        if token_id == anchor:
            mixed = relation_prediction(ranked, q, unigram_dist)
            relation_probs.append(mixed[next_token])
            object_position_unigram_probs.append(p_unigram)
            object_position_relation_probs.append(mixed[next_token])
            object_position_oracle_probs.append(1.0)  # oracle knows the true name
        else:
            relation_probs.append(p_unigram)

    report = {
        "discovered_relations": discovered,
        "overall": {
            "unigram_bits_per_token": bits_per_token(unigram_probs),
            "relation_bits_per_token": bits_per_token(relation_probs),
        },
        "object_position_only": {
            "unigram_bits": bits_per_token(object_position_unigram_probs),
            "relation_bits": bits_per_token(object_position_relation_probs),
            "oracle_bits": bits_per_token(object_position_oracle_probs),
            "n_positions": len(object_position_unigram_probs),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
