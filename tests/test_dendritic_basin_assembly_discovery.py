"""Tests for persistent-assembly-based relation discovery (design doc
sections 21-22 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.assembly_discovery import (
    assembly_information_gain,
    build_assembly_diary,
    discover_assembly_relations,
)

# 0=<bos>, 1=alice, 2=bob, 3=meets, 4=sees, 5=cat, 6=dog
VOCAB_SIZE = 7
BOS, ALICE, BOB, MEETS, SEES, CAT, DOG = 0, 1, 2, 3, 4, 5, 6
NAME_TO_OBJECT = {ALICE: CAT, BOB: DOG}
NAMES = list(NAME_TO_OBJECT)


def _generate_binding_stream(n_sentences: int, rng: np.random.Generator) -> list[int]:
    """<bos> SUBJECT meets DISTRACTOR sees OBJECT, object determined by
    the first name -- the minimal binding task at this tiny vocab size.
    """
    tokens = []
    for _ in range(n_sentences):
        subject = NAMES[int(rng.integers(0, 2))]
        distractor = BOB if subject == ALICE else ALICE
        obj = NAME_TO_OBJECT[subject]
        tokens += [BOS, subject, MEETS, distractor, SEES, obj]
    return tokens


def test_build_assembly_diary_length_and_alignment() -> None:
    stream = [BOS, ALICE, MEETS, BOB, SEES, CAT]
    diary = build_assembly_diary(stream, VOCAB_SIZE, reset_channel=BOS)
    assert len(diary) == len(stream) - 1
    for i, (token_id, matrix, next_token) in enumerate(diary):
        assert token_id == stream[i]
        assert next_token == stream[i + 1]
        assert matrix.shape == (VOCAB_SIZE, VOCAB_SIZE)


def test_discover_assembly_relations_solves_first_entity_binding() -> None:
    """The actual binding-problem fix: (BOS, subject_name) assemblies,
    trained anchor-conditioned on "sees", must be discovered and must
    correctly separate subject from distractor -- unlike a decaying
    trace, which the journal shows conflates the two once delay varies.
    """
    rng = np.random.default_rng(0)
    stream = _generate_binding_stream(400, rng)
    diary = build_assembly_diary(stream, VOCAB_SIZE, reset_channel=BOS)

    discovered = discover_assembly_relations(
        diary,
        anchor=SEES,
        first_channels=[BOS],
        second_channels=[ALICE, BOB],
        seed=1,
        rounds=60,
        perturbations_per_round=12,
        min_gain=0.5,
    )

    discovered_seconds = {key[1] for key, _unit, _gain in discovered}
    assert discovered_seconds == {ALICE, BOB}
    for _key, _unit, gain in discovered:
        assert gain > 0.8  # near the ~1-bit ceiling for this 1-of-2 object choice


def test_assembly_information_gain_is_zero_for_a_relation_that_never_fires() -> None:
    rng = np.random.default_rng(1)
    stream = _generate_binding_stream(100, rng)
    diary = build_assembly_diary(stream, VOCAB_SIZE, reset_channel=BOS)

    class _NeverFires:
        def respond(self, a: float, b: float) -> float:
            return 0.0

    gain = assembly_information_gain(_NeverFires(), (BOS, ALICE), diary, anchor=SEES)
    assert gain == 0.0
