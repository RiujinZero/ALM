from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.tasks.modular_arithmetic import (
    build_modular_ladder,
    modular_expression,
    modular_trajectory_prefix,
    unfolded_modular_lesson,
)
from manifoldflow.experiments.dendritic_basin_echolocation_smoke import (
    GrownRelations,
    relation_activations,
)
from manifoldflow.experiments.dendritic_basin_modular_arithmetic_fill import (
    _activation_block,
)
from manifoldflow.experiments.dendritic_basin_modular_arithmetic_smoke import (
    _fit_ridge_adaptive,
)
from manifoldflow.experiments.dendritic_basin_three_organ_modular_composition_smoke import (  # noqa: E501
    _decode_cues,
    _process_state,
    ascii_trace_features,
    role_gated_ascii_features,
)


def _keys(split: object) -> set[tuple[int, int, int]]:
    return {
        (int(a), int(b), int(n))
        for (a, b), n in zip(split.operands, split.moduli, strict=True)
    }


def test_modular_ladder_has_no_combination_leakage() -> None:
    ladder = build_modular_ladder(seed=7)
    train = _keys(ladder.train)
    candidate = _keys(ladder.candidate)
    test = _keys(ladder.held_out_pairs)
    unseen = _keys(ladder.unseen_modulus)
    assert train.isdisjoint(candidate | test | unseen)
    assert candidate.isdisjoint(test | unseen)
    assert test.isdisjoint(unseen)


def test_unseen_modulus_is_completely_absent_from_selection_data() -> None:
    ladder = build_modular_ladder(unseen_modulus=11)
    assert 11 not in set(ladder.train.moduli)
    assert 11 not in set(ladder.candidate.moduli)
    assert 11 not in set(ladder.held_out_pairs.moduli)
    assert set(ladder.unseen_modulus.moduli) == {11}


def test_operand_pair_partition_holds_pairs_out_across_every_modulus() -> None:
    ladder = build_modular_ladder(seed=17, partition_by_operand_pair=True)

    def pairs(split: object) -> set[tuple[int, int]]:
        return {(int(a), int(b)) for a, b in split.operands}

    train = pairs(ladder.train)
    candidate = pairs(ladder.candidate)
    held_out = pairs(ladder.held_out_pairs)
    assert train.isdisjoint(candidate | held_out)
    assert candidate.isdisjoint(held_out)


def test_raw_inputs_are_only_scaled_a_b_n_coordinates() -> None:
    ladder = build_modular_ladder(operand_limit=16, unseen_modulus=11)
    split = ladder.unseen_modulus
    assert split.inputs.shape[1] == 3
    np.testing.assert_allclose(split.inputs[:, 0], split.operands[:, 0] / 15)
    np.testing.assert_allclose(split.inputs[:, 1], split.operands[:, 1] / 15)
    np.testing.assert_allclose(split.inputs[:, 2], 1.0)


def test_residue_is_computed_after_partitioning() -> None:
    ladder = build_modular_ladder(seed=13)
    for split in (
        ladder.train,
        ladder.candidate,
        ladder.held_out_pairs,
        ladder.unseen_modulus,
    ):
        expected = np.asarray(
            [
                (int(a) + int(b)) % int(n)
                for (a, b), n in zip(split.operands, split.moduli, strict=True)
            ]
        )
        np.testing.assert_array_equal(split.residues, expected)


def test_dual_ridge_matches_primal_solution_for_wide_anatomy() -> None:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(12, 24))
    y = rng.normal(size=(12, 3))
    alpha = 0.7
    actual = _fit_ridge_adaptive(x, y, alpha, np)
    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean(axis=0)
    expected_weights = np.linalg.solve(
        centered_x.T @ centered_x + alpha * np.eye(x.shape[1]),
        centered_x.T @ centered_y,
    )
    expected_bias = y.mean(axis=0) - x.mean(axis=0) @ expected_weights
    expected = np.concatenate((expected_weights, expected_bias[None, :]), axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_blocked_fill_activation_matches_full_relation_evaluation() -> None:
    rng = np.random.default_rng(53)
    features = rng.normal(size=(17, 9)).astype(np.float32)
    relations = GrownRelations(
        parent_a=np.asarray([0, 2, 4, 6, 8]),
        parent_b=np.asarray([1, 3, 5, 7, 0]),
        w_a=rng.normal(size=5).astype(np.float32),
        w_b=rng.normal(size=5).astype(np.float32),
        mu=rng.normal(size=5).astype(np.float32),
        log_sigma=rng.uniform(-2, 0, size=5).astype(np.float32),
        score=np.zeros(5, dtype=np.float32),
    )
    expected = relation_activations(features, relations, np)[:, 1:4]
    actual = _activation_block(features, relations, 1, 4, np)
    np.testing.assert_allclose(actual, expected)


def test_conventional_expression_places_answer_after_equals_boundary() -> None:
    assert modular_expression(17, 8, 5) == b"17 + 8 mod 5 = "
    assert modular_expression(17, 8, 5, 0) == b"17 + 8 mod 5 = 0\n"


def test_expression_field_masks_preserve_decimal_width_and_answer() -> None:
    assert modular_expression(
        17, 8, 5, 0, mask_fields=frozenset(("a",))
    ) == b"?? + 8 mod 5 = 0\n"
    assert modular_expression(
        17, 8, 5, 0, mask_fields=frozenset(("b", "n"))
    ) == b"17 + ? mod ? = 0\n"


def test_unfolded_lesson_exposes_every_unit_step_and_wrap() -> None:
    assert unfolded_modular_lesson(3, 4, 5, 2) == (
        b"ring 5; start 3 at 3; add 4: 3 -> 4 -> 0 -> 1 -> 2; "
        b"3 + 4 mod 5 = 2\n"
    )
    assert modular_trajectory_prefix(3, 4, 5, 1) == (
        b"ring 5; start 3 at 3; add 4: 3 -> 4 -> "
    )


def test_ascii_trace_features_preserve_stream_order() -> None:
    forward = ascii_trace_features([b"12 = "], (2, 4), np)
    reverse = ascii_trace_features([b"21 = "], (2, 4), np)
    assert forward.shape == (1, 128 * 3)
    assert not np.allclose(forward, reverse)


def test_role_gated_ascii_features_preserve_field_and_digit_roles() -> None:
    features = role_gated_ascii_features(
        [b"17 + 8 mod 5 = ", b"8 + 17 mod 5 = "], np
    ).reshape(2, 3, 2, 10)
    assert features[0, 0, 0, 1] == 1.0
    assert features[0, 0, 1, 7] == 1.0
    assert features[0, 1, 1, 8] == 1.0
    assert features[0, 2, 1, 5] == 1.0
    assert not np.array_equal(features[0], features[1])


def test_process_cue_decoder_recovers_exact_prototypes() -> None:
    relations = GrownRelations(
        parent_a=np.asarray([0]),
        parent_b=np.asarray([1]),
        w_a=np.asarray([1.0]),
        w_b=np.asarray([1.0]),
        mu=np.asarray([0.5]),
        log_sigma=np.asarray([0.0]),
        score=np.asarray([1.0]),
    )
    raw = np.asarray([[2 / 11, 5 / 11], [6 / 11, 7 / 11]], dtype=np.float32)
    state = _process_state(raw, relations, np)
    prediction = np.concatenate((state, np.asarray([[4 / 31], [9 / 31]])), axis=1)
    position, modulus, moves = _decode_cues(prediction, relations, 11, 32, np)
    np.testing.assert_array_equal(position, [2, 6])
    np.testing.assert_array_equal(modulus, [5, 7])
    np.testing.assert_array_equal(moves, [4, 9])
