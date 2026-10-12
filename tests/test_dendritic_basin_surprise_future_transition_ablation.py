"""Focused tests for the latent surprise-grown future-transition assay."""

from __future__ import annotations

import ast
import inspect
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from manifoldflow.dendritic_basin import surprise_grown_future_transition
from manifoldflow.dendritic_basin.future_turn_trajectory import (
    EncodedTurn,
    FutureTurnTrajectory,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as assay,
)


def _turn(vector: tuple[float, float], role: str) -> EncodedTurn:
    return EncodedTurn(
        np.asarray(vector, dtype=np.float64),
        source_id=assay.TEXT_MODALITY_ID,
        role_id=role,
    )


def _sequences() -> tuple[tuple[EncodedTurn, ...], ...]:
    return (
        (
            _turn((1.0, 0.0), codec_assay.USER_ROLE),
            _turn((0.5, 1.0), codec_assay.ASSISTANT_ROLE),
            _turn((-0.5, 0.5), codec_assay.USER_ROLE),
        ),
        (
            _turn((0.0, 1.0), codec_assay.USER_ROLE),
            _turn((1.0, -0.5), codec_assay.ASSISTANT_ROLE),
            _turn((0.5, -1.0), codec_assay.USER_ROLE),
        ),
        (
            _turn((-1.0, 0.0), codec_assay.USER_ROLE),
            _turn((-0.5, -1.0), codec_assay.ASSISTANT_ROLE),
            _turn((0.75, 0.25), codec_assay.USER_ROLE),
        ),
        (
            _turn((0.0, -1.0), codec_assay.USER_ROLE),
            _turn((-1.0, 0.5), codec_assay.ASSISTANT_ROLE),
            _turn((-0.75, 0.75), codec_assay.USER_ROLE),
        ),
    )


def _model(sequences: tuple[tuple[EncodedTurn, ...], ...]) -> FutureTurnTrajectory:
    return FutureTurnTrajectory(
        2,
        source_ids=(assay.TEXT_MODALITY_ID,),
        role_ids=codec_assay.ROLE_LABELS,
        horizon_widths=(2,),
        memory_decay=0.8,
        ridge=0.1,
    ).fit(sequences)


def _conversation(index: int) -> codec_assay.TrainingConversation:
    return codec_assay.TrainingConversation(
        episode_id=f"episode-{index}",
        source_label="synthetic",
        payload=b"unused",
        assistant_turns=(),
    )


def _args(**changes: Any) -> Namespace:
    values = vars(
        assay._parser().parse_args(
            ["--dataset-dir", "unused", "--report", "unused.json"]
        )
    )
    values.update(changes)
    return Namespace(**values)


def test_context_projection_is_deterministic_finite_and_train_fit() -> None:
    rows = np.asarray(
        [
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 1.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, -1.0, -1.0, 0.0],
        ]
    )
    first = assay.ContextProjector.fit(
        rows, dimensions=3, scale_floor=1e-3, clip=2.0
    )
    second = assay.ContextProjector.fit(
        rows, dimensions=3, scale_floor=1e-3, clip=2.0
    )

    assert first.fingerprint() == second.fingerprint()
    np.testing.assert_array_equal(first.transform(rows), second.transform(rows))
    assert first.transform(rows).shape == (4, 3)
    assert np.max(np.abs(first.transform(np.full(4, 1e9)))) <= 2.0
    assert not first.mean.flags.writeable
    assert not first.basis.flags.writeable
    assert not first.scales.flags.writeable


def test_each_causal_basal_forecast_precedes_its_target_observation() -> None:
    sequences = _sequences()
    model = _model(sequences[:3])
    original = assay.causal_transition_examples(
        model,
        sequences[3],
        conversation_index=3,
        episode_id="original",
    )
    changed_sequence = (
        sequences[3][0],
        _turn((9.0, 9.0), codec_assay.ASSISTANT_ROLE),
        sequences[3][2],
    )
    changed = assay.causal_transition_examples(
        model,
        changed_sequence,
        conversation_index=3,
        episode_id="changed",
    )

    # The first target may change only after its context and basal prediction
    # have already been captured from the identical one-event prefix.
    np.testing.assert_array_equal(original[0].context_state, changed[0].context_state)
    np.testing.assert_array_equal(
        original[0].basal_prediction, changed[0].basal_prediction
    )
    assert not np.array_equal(original[0].target, changed[0].target)
    # It is then legitimate for that observed target to alter the next context.
    assert not np.array_equal(original[1].context_state, changed[1].context_state)
    assert model.observed_count == 0


def test_whole_conversation_crossfit_never_fits_a_held_out_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequences = _sequences()
    conversations = tuple(_conversation(index) for index in range(len(sequences)))
    fit_memberships: dict[int, set[int]] = {}
    sequence_indexes = {id(sequence): index for index, sequence in enumerate(sequences)}

    class FakeModel:
        def __init__(self, membership: set[int]):
            self.membership = membership

        def fingerprint(self, *, include_state: bool) -> str:
            assert not include_state
            return "-".join(map(str, sorted(self.membership)))

    def fake_fit(
        fit_sequences: tuple[tuple[EncodedTurn, ...], ...], **_: Any
    ) -> FakeModel:
        membership = {sequence_indexes[id(sequence)] for sequence in fit_sequences}
        model = FakeModel(membership)
        fit_memberships[id(model)] = membership
        return model

    def fake_examples(
        model: FakeModel,
        sequence: tuple[EncodedTurn, ...],
        *,
        conversation_index: int,
        episode_id: str,
    ) -> tuple[assay.TransitionExample, ...]:
        assert sequence is sequences[conversation_index]
        assert conversation_index not in fit_memberships[id(model)]
        target = sequence[1]
        return (
            assay.TransitionExample(
                conversation_index,
                0,
                episode_id,
                np.asarray([conversation_index, 1.0]),
                np.zeros(2),
                target.vector,
                target,
            ),
        )

    monkeypatch.setattr(
        assay.recurrent_assay, "fit_recurrent_transition", fake_fit
    )
    monkeypatch.setattr(assay, "causal_transition_examples", fake_examples)
    result = assay.cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=2,
        folds=2,
        fold_seed=7,
        memory_decay=0.8,
        ridge=0.1,
    )

    assert len(result.examples) == len(conversations)
    assert result.audit["unit"] == "whole_conversation"
    assert sorted(result.audit["fold_counts"].values()) == [2, 2]
    assert all(len(membership) == 2 for membership in fit_memberships.values())


def test_crossfit_rejects_duplicate_training_episode_ids() -> None:
    sequences = _sequences()
    conversations = tuple(_conversation(index) for index in range(4))
    conversations = (
        conversations[0],
        conversations[1],
        conversations[2],
        codec_assay.TrainingConversation(
            episode_id=conversations[0].episode_id,
            source_label="synthetic",
            payload=b"unused",
            assistant_turns=(),
        ),
    )

    with pytest.raises(ValueError, match="globally unique"):
        assay.cross_fitted_basal_transitions(
            conversations,
            sequences,
            content_dimensions=2,
            folds=2,
            fold_seed=7,
            memory_decay=0.8,
            ridge=0.1,
        )


def test_successor_control_is_deterministic_role_stratified_derangement() -> None:
    targets = (
        _turn((1.0, 0.0), codec_assay.USER_ROLE),
        _turn((2.0, 0.0), codec_assay.USER_ROLE),
        _turn((0.0, 1.0), codec_assay.ASSISTANT_ROLE),
        _turn((0.0, 2.0), codec_assay.ASSISTANT_ROLE),
    )
    examples = tuple(
        assay.TransitionExample(
            conversation_index=index,
            transition_index=0,
            episode_id=str(index),
            context_state=np.asarray([index, 1.0]),
            basal_prediction=np.zeros(2),
            target=target.vector,
            target_turn=target,
        )
        for index, target in enumerate(targets)
    )
    first, first_audit = assay.successor_target_permutation(examples, seed=11)
    second, second_audit = assay.successor_target_permutation(examples, seed=11)

    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert not np.any(first == np.arange(len(first)))
    for index, donor in enumerate(first):
        assert targets[index].role_id == targets[int(donor)].role_id
        assert targets[index].source_id == targets[int(donor)].source_id
        assert not np.array_equal(targets[index].vector, targets[int(donor)].vector)


def test_gaussian_mixture_score_is_proper_surface_and_best_mode_is_nonproper() -> None:
    target = np.asarray([1.0, -1.0])
    variance = np.ones(2)
    correct = assay.gaussian_bits_per_coordinate(
        target, (target,), (1.0,), variance
    )
    wrong = assay.gaussian_bits_per_coordinate(
        target, (np.zeros(2),), (1.0,), variance
    )
    mixture = assay.gaussian_bits_per_coordinate(
        target,
        (target, np.zeros(2)),
        (0.75, 0.25),
        variance,
    )

    assert correct < mixture < wrong
    with pytest.raises(ValueError, match="positive"):
        assay.gaussian_bits_per_coordinate(
            target, (target,), (0.0,), variance
        )


def test_consolidated_only_surface_keeps_basal_and_renormalizes_read_only() -> None:
    basal = np.zeros(2)
    juvenile = surprise_grown_future_transition.FutureModeCandidate(
        0, 0, np.asarray([2.0, 0.0]), 0.25, "juvenile"
    )
    mature = surprise_grown_future_transition.FutureModeCandidate(
        1, 0, np.asarray([4.0, 0.0]), 0.25, "consolidated"
    )
    forecast = surprise_grown_future_transition.FutureTransitionForecast(
        prediction=np.asarray([1.5, 0.0]),
        basal_prediction=basal,
        basal_weight=0.5,
        candidates=(juvenile, mature),
    )

    filtered = assay._consolidated_only_forecast(forecast)

    assert len(filtered.candidates) == 1
    assert filtered.candidates[0].mode_id == 1
    assert filtered.basal_weight == pytest.approx(2.0 / 3.0)
    assert filtered.candidates[0].weight == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(filtered.prediction, [4.0 / 3.0, 0.0])
    assert forecast.candidates == (juvenile, mature)


def test_surprise_arms_are_deterministic_and_evaluation_is_read_only() -> None:
    sequences = _sequences()
    basal = _model(sequences)
    examples = tuple(
        row
        for conversation_index, sequence in enumerate(sequences)
        for row in assay.causal_transition_examples(
            basal,
            sequence,
            conversation_index=conversation_index,
            episode_id=str(conversation_index),
        )
    )
    projector = assay.ContextProjector.fit(
        np.stack([row.context_state for row in examples]),
        dimensions=2,
        scale_floor=1e-3,
        clip=5.0,
    )
    args = _args(
        max_context_assemblies=4,
        max_residual_modes=8,
        max_modes_per_context=2,
        max_nursery_candidates=8,
        context_bandwidth=2.0,
        min_context_affinity=0.01,
        residual_floor=0.05,
        surprise_birth_z=0.1,
        poor_representation_z=0.1,
        nursery_context_z=4.0,
        nursery_residual_z=20.0,
        nursery_confirmations=2,
        nursery_target_consistency=0.0,
        consolidation_min_age=2,
        consolidation_min_fires=1,
        overlay_training_passes=1,
    )
    first_ordered, first_shuffled, first_audit = assay.train_surprise_arms(
        examples, projector, conversations=4, args=args
    )
    second_ordered, second_shuffled, second_audit = assay.train_surprise_arms(
        examples, projector, conversations=4, args=args
    )

    assert assay._learner_fingerprint(first_ordered) == assay._learner_fingerprint(
        second_ordered
    )
    assert assay._learner_fingerprint(first_shuffled) == assay._learner_fingerprint(
        second_shuffled
    )
    assert first_audit == second_audit
    before = assay._learner_fingerprint(first_ordered)
    context = projector.transform(examples[0].context_state)
    for _ in range(3):
        forecast = first_ordered.forecast(
            context, examples[0].basal_prediction, remember=False
        )
    means, weights = assay._mixture_surface(forecast)
    assert len(means) == len(forecast.candidates) + 1
    assert weights[0] == pytest.approx(forecast.basal_weight)
    assert sum(weights) == pytest.approx(1.0)
    assert assay._learner_fingerprint(first_ordered) == before
    assert not first_ordered.pending


def test_runner_surface_has_no_native_split_or_backpropagation_route() -> None:
    source = inspect.getsource(assay)
    parser = assay._parser()
    destinations = {action.dest for action in parser._actions}

    assert "checkpoint" not in destinations
    assert "split" not in destinations
    assert "native" not in destinations
    assert "torch" not in source
    assert "autograd" in source  # Explicit negative claim is documented.
    assert ".backward(" not in source
    assert "test_pairs" not in source
    with pytest.raises(ValueError, match="only --latent-only"):
        args = _args(latent_only=False)
        # Validation must reject this before attempting any filesystem access.
        assay.run_ablation(args)


def test_mechanism_and_assay_import_no_autograd_framework() -> None:
    forbidden = {"torch", "jax", "tensorflow", "autograd"}
    for module in (assay, surprise_grown_future_transition):
        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden


def test_dataset_loader_requests_only_declared_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    args = _args(
        dataset_dir=Path("dataset"),
        max_train_conversations=7,
        max_dev_pairs=3,
        exclude_dev_pair_ids=("known-smoke",),
    )
    monkeypatch.setattr(
        assay.codec_assay,
        "_schema4_manifest",
        lambda path: calls.append(("manifest", path)) or {"schema_version": 4},
    )
    monkeypatch.setattr(
        assay.codec_assay.receiver,
        "_verify_manifest_payloads",
        lambda directory, manifest, split: calls.append(
            ("verify", (directory, manifest, split))
        )
        or {"train.bin": "hash", "dev_pairs.bin": "hash"},
    )
    monkeypatch.setattr(
        assay.codec_assay,
        "load_training_conversations",
        lambda directory, maximum: calls.append(
            ("train", (directory, maximum))
        )
        or ["training"],
    )
    monkeypatch.setattr(
        assay.codec_assay,
        "load_development_pairs",
        lambda directory, maximum, excluded_pair_ids: calls.append(
            ("development", (directory, maximum, excluded_pair_ids))
        )
        or ["development"],
    )

    manifest, verified, conversations, pairs = (
        assay.load_declared_training_and_development(args)
    )

    assert manifest == {"schema_version": 4}
    assert verified["train.bin"] == "hash"
    assert conversations == ["training"]
    assert pairs == ["development"]
    assert calls == [
        ("manifest", Path("dataset")),
        (
            "verify",
            (Path("dataset"), Path("dataset/manifest.json"), "dev"),
        ),
        ("train", (Path("dataset"), 7)),
        ("development", (Path("dataset"), 3, ("known-smoke",))),
    ]


def test_config_records_are_plain_finite_json_values() -> None:
    args = _args()
    learner = assay.make_surprise_transition(
        context_dimensions=3,
        target_dimensions=2,
        args=args,
    )
    record = assay._config_record(learner.config)

    assert record["context_dim"] == 3
    assert record["target_dim"] == 2
    assert all(
        not isinstance(value, float) or np.isfinite(value)
        for value in record.values()
    )


def test_transition_example_owns_read_only_copies() -> None:
    source = np.asarray([1.0, 2.0])
    target_turn = _turn((3.0, 4.0), codec_assay.ASSISTANT_ROLE)
    row = assay.TransitionExample(
        0,
        0,
        "episode",
        source,
        source,
        target_turn.vector,
        target_turn,
    )
    source[:] = 9.0

    np.testing.assert_array_equal(row.context_state, [1.0, 2.0])
    np.testing.assert_array_equal(row.basal_prediction, [1.0, 2.0])
    assert not row.context_state.flags.writeable
    assert not row.basal_prediction.flags.writeable
    assert not row.target.flags.writeable


def test_parser_defaults_to_latent_only_and_smoke_slicing_is_explicit() -> None:
    parsed = assay._parser().parse_args(
        [
            "--dataset-dir",
            str(Path("dataset")),
            "--report",
            str(Path("report.json")),
            "--smoke",
            "--max-train-conversations",
            "10",
            "--max-dev-pairs",
            "2",
        ]
    )

    assert parsed.latent_only
    assert parsed.smoke
    assert parsed.max_train_conversations == 10
    assert parsed.max_dev_pairs == 2
