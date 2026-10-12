from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from manifoldflow.experiments.dendritic_basin_shared_goal_action_memory_ablation import (  # noqa: E501
    PACK_HEADER,
    PACK_MAGIC,
    ActionAdapters,
    HeldOutReferences,
    MemoryBundle,
    PreparedSplit,
    TrainingAction,
    _hashed_cue_ngram_vector,
    _output_value,
    _pack_memory,
    condition_outcomes,
    load_full_training_actions,
    load_heldout_references,
    source_stratified_three_way_folds,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _hashed_byte_ngram_vector,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    RidgeMap,
)


def _ridge(weights: np.ndarray) -> RidgeMap:
    return RidgeMap(
        weights=np.asarray(weights, dtype=np.float64),
        input_mean=np.zeros(weights.shape[0]),
        output_mean=np.zeros(weights.shape[1]),
    )


def _range(blob: bytes, start: int, end: int) -> dict[str, int | str]:
    value = blob[start:end]
    return {
        "start": start,
        "end": end,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def test_packed_memory_reports_the_actual_canonical_serialization() -> None:
    keys = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    cues = np.asarray([[0.0, 1.0], [0.0, -1.0]])
    actions = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    memory = _pack_memory(
        keys, cues, actions, [b"alpha", b"longer beta"], key_kind="successor"
    )

    accounting = memory.byte_accounting()
    expected = (
        PACK_HEADER.size
        + memory.keys.nbytes
        + memory.cue_vectors.nbytes
        + memory.action_vectors.nbytes
        + memory.offsets.nbytes
        + len(memory.value_blob)
    )
    assert memory.pack()[:8] == PACK_MAGIC
    assert len(memory.pack()) == expected
    assert accounting["packed_serialization_bytes"] == expected
    assert accounting["packed_serialization_sha256"] == hashlib.sha256(
        memory.pack()
    ).hexdigest()
    assert memory.value(0) == b"alpha"
    assert memory.value(1) == b"longer beta"


def test_full_output_zero_keeps_the_whole_train_action() -> None:
    action = TrainingAction("episode", b"complete response", b"question", b"history")
    assert _output_value(action, 0) == b"complete response"
    assert _output_value(action, 8) == b"complete"


def test_full_train_loader_uses_verified_assistant_content_not_score_window(
    tmp_path: Path,
) -> None:
    blob = b"USER: cue\nASSISTANT: this is the complete response\nUSER: later"
    cue_start = blob.index(b"cue")
    cue_end = cue_start + len(b"cue")
    response_start = blob.index(b"this")
    response_end = response_start + len(b"this is the complete response")
    row = {
        "episode_id": "episode-1",
        "start": 0,
        "end": len(blob),
        "target_turn_index": 1,
        "target_user_turn_index": 0,
        "target_user_content_start": cue_start,
        "target_user_content_end": cue_end,
        "target_offsets": {
            "content_start": response_start,
            "content_end": response_end,
        },
        "turns": [
            {
                "index": 0,
                "role": "USER",
                "content_start": cue_start,
                "end": cue_end,
            },
            {
                "index": 1,
                "role": "ASSISTANT",
                "content_start": response_start,
                "end": response_end,
            },
        ],
        "goal_target": {"start": response_end},
        "target_full_response_bytes": response_end - response_start,
        "target_full_response_sha256": hashlib.sha256(
            blob[response_start:response_end]
        ).hexdigest(),
    }
    (tmp_path / "train.bin").write_bytes(blob)
    (tmp_path / "train.index.jsonl").write_text(json.dumps(row) + "\n")
    example = TrainingExample(
        episode_id="episode-1",
        response=b"this is",
        prefix=b"ASSISTANT: ",
        evidence=np.zeros(4),
        response_vector=np.zeros(2),
        goal_target_vector=np.zeros(2),
    )

    loaded = load_full_training_actions(tmp_path, [example])

    assert loaded[0].data == b"this is the complete response"
    assert loaded[0].pending_cue == b"cue"
    assert loaded[0].history_cue.endswith(b"cue")
    assert loaded[0].data != example.response


def test_reference_loader_prefers_prospectively_packed_full_turns(
    tmp_path: Path,
) -> None:
    score_a = b"alpha score"
    score_b = b"beta score"
    full_a = score_a + b" and complete continuation"
    full_b = score_b + b" and complete continuation"
    pieces = [score_a, score_b, full_a, full_b]
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    blob = b"".join(pieces)
    ranges = {
        "response_a": _range(blob, offsets[0], offsets[1]),
        "response_b": _range(blob, offsets[1], offsets[2]),
        "full_response_a": _range(blob, offsets[2], offsets[3]),
        "full_response_b": _range(blob, offsets[3], offsets[4]),
    }
    row = {
        "pair_id": "pair-1",
        "blob_ranges": ranges,
        "episodes": {
            "a": {
                "target_user_content_start": 0,
                "target_user_content_end": 4,
                "unanswered_prefix_bytes": 4,
            },
            "b": {
                "target_user_content_start": 0,
                "target_user_content_end": 4,
                "unanswered_prefix_bytes": 4,
            },
        },
    }
    (tmp_path / "dev_pairs.bin").write_bytes(blob)
    (tmp_path / "dev_pairs.jsonl").write_text(json.dumps(row) + "\n")
    pair = CrossoverPair(
        pair_id="pair-1",
        context_a=b"cue A",
        context_b=b"cue B",
        response_a=score_a,
        response_b=score_b,
        evidence_a=np.zeros(4),
        evidence_b=np.zeros(4),
        source_label="synthetic",
    )

    loaded = load_heldout_references(tmp_path, "dev", [pair])[0]

    assert loaded.preferred_kind == "prospective_full_assistant_turn"
    assert loaded.preferred_a == full_a
    assert loaded.score_window_a == score_a


def test_three_way_folds_are_source_balanced_stable_and_disjoint() -> None:
    examples = [
        TrainingExample(
            episode_id=f"episode-{index}",
            response=b"response",
            prefix=b"prefix",
            evidence=np.zeros(4),
            response_vector=np.zeros(2),
            goal_target_vector=np.zeros(2),
        )
        for index in range(30)
    ]
    sources = tuple("a" if index < 18 else "b" for index in range(30))

    folds = source_stratified_three_way_folds(examples, sources)
    repeated = source_stratified_three_way_folds(examples, sources)

    assert all(
        np.array_equal(left, right)
        for left, right in zip(folds, repeated, strict=True)
    )
    assert sorted(np.concatenate(folds).tolist()) == list(range(30))
    assert sum(
        np.intersect1d(folds[left], folds[right]).size
        for left in range(3)
        for right in range(left + 1, 3)
    ) == 0
    for source in set(sources):
        counts = [
            sum(sources[int(index)] == source for index in fold) for fold in folds
        ]
        assert max(counts) - min(counts) <= 1


def test_causal_successor_retrieval_pairs_train_actions_and_nulls_blind_control() -> (
    None
):
    value_a = b"alpha topic answer"
    value_b = b"beta topic answer"
    keys = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    cue_row = _hashed_cue_ngram_vector(b"cue", 16)
    cue = np.asarray([cue_row, cue_row])
    actions = keys.copy()
    successor = _pack_memory(keys, cue, actions, [value_a, value_b], key_kind="future")
    shuffled = _pack_memory(
        keys[::-1], cue, actions, [value_a, value_b], key_kind="shuffled"
    )
    random = _pack_memory(
        keys[::-1], cue, actions, [value_a, value_b], key_kind="random"
    )
    bundle = MemoryBundle(
        successor=successor,
        pending_nn=successor,
        history_nn=successor,
        shuffled_binding=shuffled,
        random_keys=random,
        source_indices=np.asarray([2, 3]),
        binding_permutation=np.asarray([1, 0]),
        requested_capacity=None,
        output_bytes=0,
    )
    pair = CrossoverPair(
        pair_id="pair",
        context_a=b"a",
        context_b=b"b",
        response_a=value_a,
        response_b=value_b,
        evidence_a=np.asarray([0.0, 0.0]),
        evidence_b=np.asarray([0.0, 0.0]),
        source_label="synthetic",
        goal_target_kind="successor",
    )
    reference = HeldOutReferences(
        "pair", value_a, value_b, value_a, value_b, b"cue", b"cue", b"a", b"b", "full"
    )
    reference_vectors = np.asarray(
        [_hashed_byte_ngram_vector(value, 64) for value in (value_a, value_b)]
    )
    prepared = PreparedSplit(
        raw_pairs=(pair,),
        references=(reference,),
        pending_events=np.zeros((2, 2)),
        full_evidence=np.zeros((2, 2)),
        oracle_targets=keys,
        causal_goals=keys,
        pending_goals=np.zeros((2, 2)),
        shuffled_goals=keys[::-1],
        pending_nn_queries=np.zeros((2, 2)),
        history_nn_queries=np.zeros((2, 2)),
        cue_queries=np.asarray(
            [_hashed_cue_ngram_vector(b"cue", 16)] * 2
        ),
        preferred_reference_vectors=reference_vectors,
        score_window_reference_vectors=reference_vectors,
        shuffled_goal_permutation=np.asarray([1, 0]),
    )
    # Cue/base scores tie. Candidate rows alone map to their future in E.
    evaluator_weights = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    adapters = ActionAdapters(
        output_bytes=0,
        cue_dimensions=16,
        cue_kind="pending",
        base_mode="cue",
        learned_base_weight=0.0,
        plausibility_predictor=_ridge(np.zeros((2, 2))),
        evaluator_action_successor=_ridge(evaluator_weights),
        training_cue_vectors=np.zeros((4, 2)),
        training_action_vectors=np.zeros((4, 2)),
    )

    def outcomes(name: str) -> dict[str, np.ndarray]:
        return condition_outcomes(
            prepared,
            bundle,
            adapters,
            name,
            causal_gain=1.0,
            pending_nn_gain=1.0,
            history_nn_gain=1.0,
            lexical_dimensions=64,
            include_pair_audit=False,
        )

    causal = outcomes("causal_full_history")
    goal_off = outcomes("goal_off")
    blind = outcomes("candidate_blind")
    shuffled_binding = outcomes("shuffled_key_response_binding")

    assert causal["preferred_reference_margin"][0] > 0.0
    assert causal["evaluator_action_successor_margin"][0] > 0.0
    assert goal_off["preferred_reference_margin"][0] == 0.0
    assert np.array_equal(
        blind["preferred_reference_margin"], goal_off["preferred_reference_margin"]
    )
    assert shuffled_binding["preferred_reference_margin"][0] < 0.0
    successor_base, shuffled_base = (
        memory.cue_vectors @ prepared.cue_queries[0]
        for memory in (successor, shuffled)
    )
    assert np.array_equal(successor_base, shuffled_base)
    assert len({
        successor.byte_accounting()["packed_serialization_bytes"],
        shuffled.byte_accounting()["packed_serialization_bytes"],
        random.byte_accounting()["packed_serialization_bytes"],
    }) == 1
