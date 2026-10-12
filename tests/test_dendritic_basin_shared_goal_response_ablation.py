from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    Decision,
    decisions_from_conversation,
    fit_codebook,
    fit_ridge,
    hard_negative_candidates,
    hashed_text_vector,
    parse_turns,
    split_name,
    stack_decisions,
)


def test_turn_parser_preserves_paragraphs_inside_roles() -> None:
    text = (
        "USER: first question\n\nwith detail\n\n"
        "ASSISTANT: first answer\n\nUSER: follow up\n\n"
        "ASSISTANT: later answer"
    )

    turns = parse_turns(text)

    assert [turn.role for turn in turns] == [
        "USER",
        "ASSISTANT",
        "USER",
        "ASSISTANT",
    ]
    assert turns[0].text == "first question\n\nwith detail"


def test_hashed_vectors_and_split_are_deterministic() -> None:
    left = hashed_text_vector("Carry the blue plan forward", 32)
    right = hashed_text_vector("Carry the blue plan forward", 32)

    np.testing.assert_array_equal(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)
    assert split_name("one complete conversation") == split_name(
        "one complete conversation"
    )


def test_decisions_use_past_state_and_later_targets() -> None:
    turns = parse_turns(
        "USER: establish blue garden\n\n"
        "ASSISTANT: we will plant blue flowers\n\n"
        "USER: what comes next\n\n"
        "ASSISTANT: prepare the soil\n\n"
        "USER: and after that\n\n"
        "ASSISTANT: add blue seeds\n\n"
        "USER: final step\n\n"
        "ASSISTANT: water the blue garden"
    )

    rows = decisions_from_conversation(7, turns, dimensions=32)

    assert len(rows) == 2
    assert rows[0].conversation_id == 7
    assert rows[0].persistent.shape == (32 * 3,)
    assert rows[0].successor_target.shape == (32 * 3,)
    np.testing.assert_allclose(
        rows[0].interlocutor_target,
        hashed_text_vector("and after that", 32),
    )
    np.testing.assert_allclose(
        rows[0].far_future_target,
        hashed_text_vector("water the blue garden", 32),
    )


def _decision(conversation: int, local: np.ndarray) -> Decision:
    unit = local / np.linalg.norm(local)
    return Decision(
        conversation_id=conversation,
        decision_index=0,
        local=unit,
        persistent=np.tile(unit, 3),
        response=np.roll(unit, 1),
        theme_target=np.tile(unit, 3),
        far_future_target=unit,
        interlocutor_target=unit,
        successor_target=np.tile(unit, 3),
        response_chars=20,
    )


def test_hard_negatives_never_come_from_the_same_conversation() -> None:
    rows = [_decision(index, np.eye(4)[index % 4] + 0.01) for index in range(8)]
    batch = stack_decisions(rows)

    choices = hard_negative_candidates(batch, 4)

    np.testing.assert_array_equal(choices[:, 0], np.arange(len(batch)))
    for row, negatives in enumerate(choices[:, 1:]):
        assert np.all(batch.conversation_ids[negatives] != batch.conversation_ids[row])


def test_ridge_map_and_codebook_recover_simple_structure() -> None:
    rng = np.random.default_rng(0)
    evidence = rng.standard_normal((100, 6))
    transform = rng.standard_normal((6, 4))
    target = evidence @ transform

    model = fit_ridge(evidence, target, alpha=1e-6)
    prediction = model.predict(evidence)
    codebook = fit_codebook(target, prototypes=4, iterations=10, seed=0)

    assert np.mean((prediction - target) ** 2) < 1e-10
    labels = codebook.assign(target)
    assert len(np.unique(labels)) >= 2
