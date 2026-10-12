from __future__ import annotations

import hashlib

import numpy as np
import pytest

from manifoldflow.experiments.dendritic_basin_future_continuous_pair_audit import (
    SpectralRidge,
    TargetSpec,
    _target_specs,
    _validate_target_metadata,
    causal_evidence,
    causal_turns_from_prefix,
    choose_calibration_scale,
    pair_margins,
    salted_text_vector,
    sattolo_permutation,
    successor_signature,
    target_vector,
)


def test_salted_hash_is_deterministic_normalized_and_salt_sensitive() -> None:
    text = "preserve the durable objective while changing the local wording"
    first = salted_text_vector(text, 32, 3)
    repeated = salted_text_vector(text, 32, 3)
    other_salt = salted_text_vector(text, 32, 11)

    assert np.array_equal(first, repeated)
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert not np.array_equal(first, other_salt)


def test_successor_signature_has_exact_multiscale_shape_without_role_ids() -> None:
    first = b"\n\nUSER: revisit the objective\n\nASSISTANT: carry out the durable plan"
    roles_swapped = (
        b"\n\nASSISTANT: revisit the objective\n\nUSER: carry out the durable plan"
    )
    vector = successor_signature(
        first, base_dimensions=16, gammas=(0.5, 0.85, 0.97), salt=3
    )
    swapped = successor_signature(
        roles_swapped, base_dimensions=16, gammas=(0.5, 0.85, 0.97), salt=3
    )

    assert vector.shape == (48,)
    assert all(np.isclose(np.linalg.norm(chunk), 1.0) for chunk in np.split(vector, 3))
    # Role markers are used to find event boundaries, but no role/modality ID is
    # appended to the learned target vector.
    assert np.array_equal(vector, swapped)


def test_target_specs_and_dimensions_match_goal_kind() -> None:
    successor = _target_specs("successor", (16, 32), (0.5, 0.85, 0.97))
    interlocutor = _target_specs("interlocutor", (16, 32), (0.5, 0.85, 0.97))

    assert [(row.name, row.output_dimensions) for row in successor] == [
        ("whole_payload_32", 32),
        ("successor_turns_16", 48),
        ("successor_turns_32", 96),
    ]
    assert [(row.name, row.output_dimensions) for row in interlocutor] == [
        ("whole_payload_32", 32)
    ]
    payload = b"literal next user content"
    assert target_vector(
        payload, TargetSpec("whole", "whole_payload", 32), 3
    ).shape == (32,)


def test_causal_prefix_excludes_bridge_and_pending_control_is_exact_first_block() -> (
    None
):
    prefix = (
        b"USER: establish a durable objective\n\n"
        b"ASSISTANT: acknowledge the objective\n\n"
        b"USER: choose the next action"
    )
    metadata = {
        "target_user_content_end": len(prefix),
        "unanswered_prefix_bytes": len(prefix),
        "unanswered_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
    }
    first = causal_turns_from_prefix(prefix + b"\n\nASSISTANT: bridge one", metadata)
    second = causal_turns_from_prefix(
        prefix + b"\n\nUSER: unrelated bridge two", metadata
    )
    evidence = causal_evidence(first, dimensions=32, decays=(0.5, 0.85), salt=11)

    assert first == second
    assert evidence.shape == (96,)
    assert np.array_equal(
        evidence[:32], salted_text_vector("choose the next action", 32, 11)
    )


def test_sattolo_shuffle_is_reproducible_and_has_no_fixed_points() -> None:
    first = sattolo_permutation(64, seed=104_732)
    second = sattolo_permutation(64, seed=104_732)

    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(64))
    assert not np.any(first == np.arange(64))


def test_pair_margin_reverses_under_crossed_predictions() -> None:
    targets = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    matched = pair_margins(targets, targets)
    crossed = pair_margins(targets[:, ::-1], targets)

    assert np.all(matched > 0.0)
    assert np.all(crossed < 0.0)
    assert np.allclose(matched, -crossed)
    assert choose_calibration_scale(matched, (0.0, 1.0, 4.0)) == 4.0
    assert choose_calibration_scale(crossed, (0.0, 1.0, 4.0)) == 0.0


def test_spectral_ridge_matches_centered_closed_form() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.asarray([[0.0], [1.0], [4.0], [9.0]])
    evaluate = np.asarray([[0.5], [2.5]])
    alpha = 1.0
    prediction = SpectralRidge.from_evidence(train_x, evaluate).predict(train_y, alpha)

    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    centered_x = train_x - x_mean
    weight = np.linalg.solve(
        centered_x.T @ centered_x + alpha * np.eye(1),
        centered_x.T @ (train_y - y_mean),
    )
    expected = (evaluate - x_mean) @ weight + y_mean
    assert np.allclose(prediction, expected)


def test_target_metadata_rejects_response_overlap() -> None:
    valid = {
        "goal_target_kind": "successor",
        "target_content_end": 100,
        "goal_target_local_start": 100,
        "goal_target_turn_start": 6,
        "goal_target_turn_end": 8,
        "target_turn_index": 5,
    }
    assert _validate_target_metadata(valid, "successor") == 0
    with pytest.raises(ValueError, match="overlaps"):
        _validate_target_metadata({**valid, "goal_target_local_start": 99}, "successor")
    with pytest.raises(ValueError, match="fewer than two"):
        _validate_target_metadata({**valid, "goal_target_turn_end": 6}, "successor")
