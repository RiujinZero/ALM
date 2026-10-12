"""Phase 4 (design doc section 54: "Raw character stream"), for real: does
the same discovery mechanism validated at word granularity (Phases 2-3)
still work when the input is a raw character stream (`a l i c e _ s e e s
_ c a t _ .`) instead of pre-segmented word tokens?

No tokenizer, no word IDs as input -- only a candidate list of character
sequences to check for exact completion (see word_completion.py's module
docstring for what this does and does not claim: full unsupervised
discovery of *which* substrings matter is not attempted here, only
whether persistence + anchor-conditioned discovery still works once
"which entity" has to be assembled from individual characters rather
than handed over as one already-formed token).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.persistent_assembly import PersistentEventFlags
from manifoldflow.dendritic_basin.prediction_table import bits_per_token
from manifoldflow.dendritic_basin.relation_unit import RelationUnit
from manifoldflow.dendritic_basin.tasks.character_stream import (
    NAME_TO_OBJECT,
    NAMES,
    TOKEN_TO_ID,
    VOCAB,
    generate_char_stream,
)
from manifoldflow.dendritic_basin.word_completion import WordCompletionDetector

BOS = TOKEN_TO_ID["<bos>"]
CANDIDATE_WORDS = {word: [TOKEN_TO_ID[c] for c in word] for word in [*NAMES, "sees"]}
NAME_EVENTS = [name for name in CANDIDATE_WORDS if name != "sees"]


def build_event_diary(token_stream: list[int]) -> list[tuple[dict[str, float], int]]:
    """One pass over the character stream: at each position, the sticky
    "has this candidate word completed since the last <bos>" flags, and
    the actual next character. Both word-completion detection and
    persistence are instantaneous, local, per-position operations -- no
    future information leaks in.
    """
    detector = WordCompletionDetector(CANDIDATE_WORDS)
    flags = PersistentEventFlags(list(CANDIDATE_WORDS))
    diary = []
    for t in range(len(token_stream) - 1):
        token_id = token_stream[t]
        completions = detector.update(token_id)
        flags.update(completions, reset=(token_id == BOS))
        diary.append((flags.snapshot(), token_stream[t + 1]))
    return diary


def _sees_just_completed_positions(
    diary: list[tuple[dict[str, float], int]],
) -> np.ndarray:
    """The exact instant "sees" finishes being spelled (its sticky flag
    turns on this step, having been off the step before) -- the moment
    the object becomes predictable, not every later position too (the
    flag stays sticky-true for the rest of the sentence). Note this
    position's *own* next-token target is always the mandatory space
    character (`sees` is always followed by `_`, trivially predictable);
    the actual object ambiguity is resolved one further step later, at
    the space -> first-object-character transition -- callers should read
    the target two steps ahead of this position, not `diary[i][1]`.
    """
    positions = []
    for i, (flags, _next) in enumerate(diary):
        if flags["sees"] <= 0.5:
            continue
        if i == 0 or diary[i - 1][0]["sees"] <= 0.5:
            positions.append(i)
    return np.array(positions)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _object_first_char_target(
    diary: list[tuple[dict[str, float], int]], anchor: int
) -> int:
    """The object's first character, two steps after `anchor` ("sees" just
    completed -> mandatory space -> first object character). See
    `_sees_just_completed_positions`'s docstring.
    """
    return diary[anchor + 1][1]


def _name_information_gain(
    unit: RelationUnit,
    name: str,
    diary: list[tuple[dict[str, float], int]],
    anchor_positions: np.ndarray,
    vocab_size: int,
) -> float:
    with_c = np.zeros(vocab_size)
    without_c = np.zeros(vocab_size)
    for i in anchor_positions:
        flags_i, _ = diary[int(i)]
        target = _object_first_char_target(diary, int(i))
        activation = unit.respond(1.0, flags_i[name])
        if activation > 0.5:
            with_c[target] += 1
        else:
            without_c[target] += 1
    if with_c.sum() == 0:
        return 0.0
    return _entropy(with_c + without_c) - _entropy(with_c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sentences", type=int, default=500)
    parser.add_argument("--eval-sentences", type=int, default=500)
    parser.add_argument("--min-gain", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    vocab_size = len(VOCAB)

    train_stream = generate_char_stream(args.train_sentences, rng)
    train_diary = build_event_diary(train_stream)
    train_anchor_positions = _sees_just_completed_positions(train_diary)

    discovered: dict[str, tuple[RelationUnit, float]] = {}
    for name in NAME_EVENTS:

        def draw_batch(
            n: int, rng: np.random.Generator, name: str = name
        ) -> list[tuple[float, float]]:
            idx = rng.choice(train_anchor_positions, size=n, replace=True)
            return [(1.0, train_diary[int(i)][0][name]) for i in idx]

        best_unit, best_gain = None, -np.inf
        for episode in range(5):
            unit = RelationUnit(
                n_candidates=3, seed=args.seed * 100 + episode, acceptance_threshold=0.0
            )
            rng2 = np.random.default_rng(args.seed * 1000 + episode)
            unit.adapt(
                draw_batch,
                rounds=60,
                perturbations_per_round=12,
                step_size=0.3,
                rng=rng2,
            )
            gain = _name_information_gain(
                unit, name, train_diary, train_anchor_positions, vocab_size
            )
            if gain > best_gain:
                best_unit, best_gain = unit, gain
        if best_gain >= args.min_gain:
            discovered[name] = (best_unit, best_gain)

    eval_stream = generate_char_stream(args.eval_sentences, rng)
    eval_diary = build_event_diary(eval_stream)
    eval_anchor_positions = _sees_just_completed_positions(eval_diary)

    correct = 0
    object_probs = []
    for i in eval_anchor_positions:
        flags_i, _ = eval_diary[int(i)]
        target = _object_first_char_target(eval_diary, int(i))
        votes = []
        for name, (unit, gain) in discovered.items():
            activation = unit.respond(1.0, flags_i[name])
            if activation > 0.5:
                votes.append((gain, TOKEN_TO_ID[NAME_TO_OBJECT[name][0]]))
        if votes:
            votes.sort(key=lambda v: v[0], reverse=True)
            predicted_first_char = votes[0][1]
        else:
            predicted_first_char = None
        correct += int(predicted_first_char == target)
        object_probs.append(1.0 if predicted_first_char == target else 1e-6)

    report = {
        "discovered_names": {name: gain for name, (_u, gain) in discovered.items()},
        "object_first_char_accuracy": correct / len(eval_anchor_positions),
        "n_positions": len(eval_anchor_positions),
        "bits_per_position": bits_per_token(object_probs),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
