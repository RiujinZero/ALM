"""Tests for persistent (non-decaying) bigram assemblies (design doc
sections 21-22 of `Manifold_Dendritic_Stream_Learner_Design_v0.2.md`)."""

from __future__ import annotations

from manifoldflow.dendritic_basin.persistent_assembly import (
    PersistentBigramAssemblies,
    PersistentEventFlags,
)

BOS, ALICE, MEETS, BOB, SEES, CAT = 0, 1, 2, 3, 4, 5
VOCAB_SIZE = 6


def test_assembly_becomes_active_immediately_after_the_bigram() -> None:
    assemblies = PersistentBigramAssemblies(VOCAB_SIZE, reset_channel=BOS)
    assemblies.update(BOS)
    assert not assemblies.is_active(BOS, ALICE)
    assemblies.update(ALICE)
    assert assemblies.is_active(BOS, ALICE)


def test_assembly_persists_through_intervening_tokens() -> None:
    assemblies = PersistentBigramAssemblies(VOCAB_SIZE, reset_channel=BOS)
    for token in [BOS, ALICE, MEETS, BOB, SEES]:
        assemblies.update(token)
    assert assemblies.is_active(BOS, ALICE)  # still true, despite MEETS/BOB/SEES since


def test_reset_channel_clears_all_assemblies() -> None:
    assemblies = PersistentBigramAssemblies(VOCAB_SIZE, reset_channel=BOS)
    for token in [BOS, ALICE, MEETS, BOB]:
        assemblies.update(token)
    assert assemblies.is_active(BOS, ALICE)
    assemblies.update(BOS)  # new episode
    assert not assemblies.is_active(BOS, ALICE)


def test_distractor_bigram_is_never_confused_with_subject_bigram() -> None:
    """(BOS, SUBJECT) and (BOS, DISTRACTOR) are different assemblies --
    only the one matching the token that actually followed BOS fires.
    """
    assemblies = PersistentBigramAssemblies(VOCAB_SIZE, reset_channel=BOS)
    for token in [BOS, ALICE, MEETS, BOB, SEES, CAT]:
        assemblies.update(token)
    assert assemblies.is_active(BOS, ALICE)
    assert not assemblies.is_active(BOS, BOB)  # BOB followed MEETS, not BOS


def test_event_flags_go_sticky_true_and_stay_true_until_reset() -> None:
    flags = PersistentEventFlags(["alice_done", "bob_done"])
    flags.update({"alice_done": 1.0, "bob_done": 0.0}, reset=False)
    assert flags.is_active("alice_done")
    assert not flags.is_active("bob_done")

    flags.update({"alice_done": 0.0, "bob_done": 0.0}, reset=False)
    assert flags.is_active("alice_done")  # still true, no reset happened


def test_event_flags_reset_clears_everything() -> None:
    flags = PersistentEventFlags(["alice_done"])
    flags.update({"alice_done": 1.0}, reset=False)
    flags.update({"alice_done": 0.0}, reset=True)
    assert not flags.is_active("alice_done")


def test_event_flags_snapshot_matches_is_active() -> None:
    flags = PersistentEventFlags(["a", "b"])
    flags.update({"a": 1.0, "b": 0.0}, reset=False)
    assert flags.snapshot() == {"a": 1.0, "b": 0.0}


def test_reset_method_clears_state_and_prev_token() -> None:
    assemblies = PersistentBigramAssemblies(VOCAB_SIZE, reset_channel=BOS)
    assemblies.update(BOS)
    assemblies.update(ALICE)
    assemblies.reset()
    assert not assemblies.is_active(BOS, ALICE)
    assemblies.update(MEETS)
    assert not assemblies.is_active(ALICE, MEETS)  # prev_token was cleared too
