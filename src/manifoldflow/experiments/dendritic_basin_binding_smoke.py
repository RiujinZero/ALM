"""Phase 3 (design doc section 54: distractor/binding), for real: does
persistent-assembly discovery (real InteractionDendrites over
`PersistentBigramAssemblies`, anchor-conditioned) solve the first-entity
binding task, where decaying temporal traces alone are shown (in
`docs/ManifoldFlow_ALM_Journal.md`) to collapse under variable delay?

  alice meets bob sees cat .   (subject=alice -> object=cat, bob is a distractor)
  bob meets alice sees dog .   (subject=bob -> object=dog, alice is a distractor)

with 0-2 random filler words both after the subject and after the
distractor, so fixed-position decay cannot trivially solve it either.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.assembly_discovery import (
    build_assembly_diary,
    discover_assembly_relations,
)
from manifoldflow.dendritic_basin.prediction_table import (
    PredictionTable,
    bits_per_token,
    mix_distributions,
)
from manifoldflow.dendritic_basin.tasks.binding import NAME_TO_OBJECT, NAMES

FILLERS = ["now", "softly", "then"]


def _build_vocab() -> tuple[list[str], dict[str, int]]:
    base = ["<bos>", *NAMES, "meets", "sees", *NAME_TO_OBJECT.values(), ".", *FILLERS]
    vocab = list(dict.fromkeys(base))  # de-dup while preserving order
    return vocab, {t: i for i, t in enumerate(vocab)}


VOCAB, TOK2ID = _build_vocab()
VOCAB_SIZE = len(VOCAB)
BOS = TOK2ID["<bos>"]
SEES = TOK2ID["sees"]
CHANNEL_TO_OBJECT_TOKEN = {
    TOK2ID[name]: TOK2ID[obj] for name, obj in NAME_TO_OBJECT.items()
}


def _random_fillers(rng: np.random.Generator, max_fillers: int) -> list[str]:
    count = int(rng.integers(0, max_fillers + 1))
    return [FILLERS[int(rng.integers(0, len(FILLERS)))] for _ in range(count)]


def generate_stream(
    n_sentences: int, rng: np.random.Generator, max_fillers: int
) -> list[int]:
    tokens: list[str] = []
    for _ in range(n_sentences):
        subject = NAMES[int(rng.integers(0, len(NAMES)))]
        remaining = [n for n in NAMES if n != subject]
        distractor = remaining[int(rng.integers(0, len(remaining)))]
        obj = NAME_TO_OBJECT[subject]
        fillers_a = _random_fillers(rng, max_fillers)
        fillers_b = _random_fillers(rng, max_fillers)
        sentence = [
            "<bos>", subject, *fillers_a, "meets", distractor,
            *fillers_b, "sees", obj, ".",
        ]
        tokens.extend(sentence)
    return [TOK2ID[t] for t in tokens]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sentences", type=int, default=800)
    parser.add_argument("--eval-sentences", type=int, default=500)
    parser.add_argument("--max-fillers", type=int, default=2)
    parser.add_argument("--min-gain", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    train_stream = generate_stream(args.train_sentences, rng, args.max_fillers)
    train_diary = build_assembly_diary(train_stream, VOCAB_SIZE, reset_channel=BOS)

    name_channels = [TOK2ID[n] for n in NAMES]
    discovered = discover_assembly_relations(
        train_diary,
        anchor=SEES,
        first_channels=[BOS],
        second_channels=name_channels,
        seed=args.seed + 1,
        rounds=60,
        perturbations_per_round=12,
        min_gain=args.min_gain,
    )

    unigram = PredictionTable(VOCAB_SIZE, decay=1.0, lr=1.0)
    for token_id in train_stream:
        unigram.update(1.0, token_id)
    unigram_dist = unigram.distribution()

    eval_stream = generate_stream(args.eval_sentences, rng, args.max_fillers)
    eval_diary = build_assembly_diary(eval_stream, VOCAB_SIZE, reset_channel=BOS)

    correct = 0
    total = 0
    object_probs = []
    for token_id, matrix, next_token in eval_diary:
        if token_id != SEES:
            continue
        total += 1
        weighted = []
        for (first_channel, second_channel), unit, gain in discovered:
            object_token = CHANNEL_TO_OBJECT_TOKEN.get(second_channel)
            if object_token is None:
                continue
            activation = unit.respond(1.0, float(matrix[first_channel, second_channel]))
            if activation > 0.5:
                target_dist = np.zeros(VOCAB_SIZE)
                target_dist[object_token] = 1.0
                weighted.append((gain, target_dist))
        mixed = mix_distributions(unigram_dist, weighted)
        predicted = int(np.argmax(mixed))
        correct += int(predicted == next_token)
        object_probs.append(mixed[next_token])

    report = {
        "discovered_assemblies": [
            {"first": VOCAB[f], "second": VOCAB[s], "gain_bits": gain}
            for (f, s), _unit, gain in discovered
        ],
        "object_accuracy": correct / total if total else 0.0,
        "object_bits_per_position": bits_per_token(object_probs),
        "n_positions": total,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
