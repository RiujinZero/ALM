from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    hashed_text_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_trajectory_reranker_ablation import (  # noqa: E501
    _action_score_matrix,
    _evaluate_split,
    _fit_partitioned_reranker,
    _matched_choice_outcomes,
    _pair_goal_predictions,
    _pair_vectors,
    _sattolo_permutation,
    _seed_gate,
    _select_gain,
    _source_stratified_disjoint_halves,
    action_features,
    fit_reranker,
    fit_target_encoder,
    symmetric_margin,
    validate_dataset_semantics,
)


def _synthetic_mechanism() -> tuple[list[TrainingExample], CrossoverPair, object]:
    dimensions = 8
    response_a = b"alpha deliberate trajectory"
    response_b = b"beta deliberate trajectory"
    vector_a = hashed_text_vector(response_a.decode(), dimensions)
    vector_b = hashed_text_vector(response_b.decode(), dimensions)
    pending = np.zeros(dimensions)
    pending[0] = 1.0
    future_a = np.asarray([1.0, 0.0, 0.0, 0.0])
    future_b = np.asarray([-1.0, 0.0, 0.0, 0.0])
    examples = []
    for index in range(80):
        side_a = index % 2 == 0
        persistent = np.zeros(dimensions)
        persistent[1] = 1.0 if side_a else -1.0
        examples.append(
            TrainingExample(
                episode_id=f"episode-{index}",
                response=response_a if side_a else response_b,
                prefix=b"ASSISTANT: ",
                evidence=np.concatenate((pending, persistent)),
                response_vector=vector_a if side_a else vector_b,
                goal_target_vector=future_a if side_a else future_b,
            )
        )
    pair = CrossoverPair(
        pair_id="synthetic-pair",
        context_a=b"context a; common suffix",
        context_b=b"context b; common suffix",
        response_a=response_a,
        response_b=response_b,
        evidence_a=np.concatenate((pending, np.eye(1, dimensions, 1)[0])),
        evidence_b=np.concatenate((pending, -np.eye(1, dimensions, 1)[0])),
        source_label="synthetic",
        goal_target_a=b"later-a",
        goal_target_b=b"later-b",
        goal_target_kind="successor",
        goal_target_vector_a=future_a,
        goal_target_vector_b=future_b,
    )
    model = fit_reranker(
        examples,
        feature_dimensions=dimensions,
        latent_dimensions=2,
        goal_ridge_alpha=1e-4,
        action_ridge_alpha=1e-4,
        plausibility_ridge_alpha=1e-4,
        seed=7,
    )
    return examples, pair, model


def test_bounded_interaction_is_candidate_specific() -> None:
    pending = np.asarray([[1.0, -2.0, 0.0], [1.0, -2.0, 0.0]])
    candidate = np.asarray([[2.0, 1.0, 0.0], [-2.0, -1.0, 0.0]])
    features = action_features(pending, candidate)

    assert features.shape == (2, 10)
    assert np.max(np.abs(features)) <= 1.0
    assert not np.array_equal(features[0], features[1])


def test_continuous_goal_and_action_maps_recover_delayed_latent() -> None:
    _, raw_pair, model = _synthetic_mechanism()
    pair = _pair_vectors([raw_pair], model)[0]
    prediction = _pair_goal_predictions([raw_pair], model)["causal_full_history"]
    pending_only = _pair_goal_predictions([raw_pair], model)["causal_pending_user_only"]
    full_scores = _action_score_matrix(pair, model, prediction[0], prediction[1])
    local_scores = _action_score_matrix(pair, model, pending_only[0], pending_only[1])
    swapped_scores = _action_score_matrix(pair, model, prediction[1], prediction[0])
    blind_scores = _action_score_matrix(
        pair,
        model,
        prediction[0],
        prediction[1],
        candidate_blind=True,
    )

    assert symmetric_margin(full_scores) > 3.9
    assert symmetric_margin(swapped_scores) < -3.9
    assert symmetric_margin(local_scores) == pytest.approx(0.0, abs=1e-10)
    assert symmetric_margin(blind_scores) == 0.0


def test_gain_selection_uses_causal_dev_candidate_choices() -> None:
    _, pair, model = _synthetic_mechanism()
    selected, curve = _select_gain(
        [pair], model, gains=(0.0, 0.25, 1.0), seed=5, interaction=True
    )

    assert selected == 0.25
    assert curve["0.0"]["causal_matched_candidate_accuracy_ties_half"] == 0.5
    assert curve["0.25"]["causal_matched_candidate_accuracy_ties_half"] == 1.0


def test_matched_choice_outcomes_cluster_the_two_contexts_by_pair() -> None:
    outcomes = _matched_choice_outcomes(
        [
            {"aa": 2.0, "ab": 1.0, "ba": 2.0, "bb": 1.0},
            {"aa": 1.0, "ab": 1.0, "ba": 1.0, "bb": 1.0},
            {"aa": 2.0, "ab": 1.0, "ba": 1.0, "bb": 2.0},
        ]
    )

    assert np.array_equal(outcomes, np.asarray([0.5, 0.5, 1.0]))


def test_seed_gate_requires_pair_clustered_choice_improvement() -> None:
    _, pair, model = _synthetic_mechanism()
    split = _evaluate_split(
        [pair],
        model,
        gain=0.25,
        native_gain=None,
        seed=5,
        resamples=100,
        interaction=True,
        cortex=None,
        include_audit_rows=False,
    )
    seed_report = {"test": split}

    summary = split["primary_semantic_base"]["conditions"]["causal_full_history"][
        "matched_candidate_accuracy_delta_vs_goal_off"
    ]
    assert summary["observations"] == 1
    assert summary["mean"] > 0.0
    assert _seed_gate(seed_report)[
        "causal_matched_candidate_accuracy_delta_ci_above_zero"
    ]
    assert _seed_gate(seed_report)["causal_action_full_history_increment_ci_above_zero"]

    summary["mean_bootstrap_ci95_low"] = -1e-9
    assert not _seed_gate(seed_report)[
        "causal_matched_candidate_accuracy_delta_ci_above_zero"
    ]

    history_increment = split["primary_semantic_base"][
        "paired_causal_control_contrasts"
    ]["causal_full_history_minus_causal_pending_user_only"]
    history_increment["mean_bootstrap_ci95_low"] = -1e-9
    assert not _seed_gate(seed_report)[
        "causal_action_full_history_increment_ci_above_zero"
    ]


def test_sattolo_control_has_no_fixed_training_targets() -> None:
    first = _sattolo_permutation(50, np.random.default_rng(19))
    second = _sattolo_permutation(50, np.random.default_rng(19))
    assert np.array_equal(first, second)
    assert np.all(first != np.arange(50))
    assert sorted(first.tolist()) == list(range(50))


def test_source_stratified_halves_are_stable_balanced_and_disjoint() -> None:
    examples, _, _ = _synthetic_mechanism()
    sources = tuple("source-a" if index < 51 else "source-b" for index in range(80))

    fold_a, fold_b = _source_stratified_disjoint_halves(examples, sources)
    repeat_a, repeat_b = _source_stratified_disjoint_halves(examples, sources)

    assert np.array_equal(fold_a, repeat_a)
    assert np.array_equal(fold_b, repeat_b)
    assert not np.intersect1d(fold_a, fold_b).size
    assert sorted(np.concatenate((fold_a, fold_b)).tolist()) == list(range(80))
    for source in set(sources):
        count_a = sum(sources[int(index)] == source for index in fold_a)
        count_b = sum(sources[int(index)] == source for index in fold_b)
        assert abs(count_a - count_b) <= 1


def test_partitioned_fit_keeps_goal_and_action_supervision_disjoint() -> None:
    examples, _, _ = _synthetic_mechanism()
    sources = tuple("synthetic" for _ in examples)
    fold_a, fold_b = _source_stratified_disjoint_halves(examples, sources)
    encoder = fit_target_encoder(
        np.asarray([row.goal_target_vector for row in examples]),
        latent_dimensions=2,
    )

    model = _fit_partitioned_reranker(
        examples,
        encoder=encoder,
        goal_indices=fold_a,
        action_indices=fold_b,
        feature_dimensions=8,
        goal_ridge_alpha=1e-4,
        action_ridge_alpha=1e-4,
        plausibility_ridge_alpha=1e-4,
        seed=17,
        interaction=False,
    )

    assert len(model.predictor_target_permutation) == len(fold_a)
    assert len(model.action_target_permutation) == len(fold_b)
    assert len(model.candidate_action_permutation) == len(fold_b)
    assert np.all(
        model.predictor_target_permutation
        != np.arange(len(model.predictor_target_permutation))
    )
    assert np.all(
        model.action_target_permutation
        != np.arange(len(model.action_target_permutation))
    )


def _strict_sidecars(directory: Path, *, kind: str = "successor") -> None:
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"config": {"goal_target_kind": kind}}), encoding="utf-8"
    )
    train = {
        "goal_target_kind": kind,
        "goal_target_local_start": 120,
        "goal_target_turn_start": 4,
        "target_content_end": 120,
        "target_turn_index": 3,
        "target_offsets": {"content_end": 120},
        "goal_target": {"kind": kind, "start": 120},
    }
    pair_episode = {
        "goal_target_kind": kind,
        "goal_target_local_start": 120,
        "goal_target_turn_start": 4,
        "target_content_end": 120,
        "target_turn_index": 3,
    }
    pair = {
        "goal_target_kind": kind,
        "blob_ranges": {"goal_target_a": {}, "goal_target_b": {}},
        "episodes": {"a": pair_episode, "b": pair_episode},
    }
    (directory / "train.index.jsonl").write_text(
        json.dumps(train) + "\n", encoding="utf-8"
    )
    for split in ("dev", "test"):
        (directory / f"{split}_pairs.jsonl").write_text(
            json.dumps(pair) + "\n", encoding="utf-8"
        )


def test_dataset_guard_accepts_only_explicit_strictly_later_targets(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid"
    _strict_sidecars(valid)
    assert validate_dataset_semantics(valid) == "successor"

    response = tmp_path / "response"
    _strict_sidecars(response, kind="response")
    with pytest.raises(ValueError, match="never response"):
        validate_dataset_semantics(response)

    overlap = tmp_path / "overlap"
    _strict_sidecars(overlap)
    rows = json.loads((overlap / "train.index.jsonl").read_text())
    rows["goal_target_local_start"] = 119
    (overlap / "train.index.jsonl").write_text(
        json.dumps(rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not strictly after"):
        validate_dataset_semantics(overlap)


def test_symmetric_margin_uses_all_four_exact_assignments() -> None:
    scores = {"aa": 0.9, "ab": 0.2, "ba": 0.4, "bb": 0.8}
    assert symmetric_margin(scores) == pytest.approx(1.1)
