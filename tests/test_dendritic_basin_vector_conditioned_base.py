from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.vector_conditioned_base import (
    StatefulVectorConditionedAdapter,
    StatefulVectorConditionedConfig,
    StatefulVectorConditionedTurn,
    fit_vector_conditioned_turn,
    generate_from_vector,
    generate_from_vector_after_context,
    generate_stateful_after_context,
    generate_stateful_multi_turn,
    score_vector_conditioned_turn,
    vector_input_frame,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)


class FakeCortex:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, bool]] = []
        self.reset_count = 0
        self.next_symbol = ord("O")

    def reset_episode(self) -> None:
        self.reset_count += 1

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        payload = np.asarray(byte_ids, dtype=np.uint8).tobytes()
        self.calls.append((payload, learn))
        if learn:
            return np.full(len(payload), 0.5, dtype=np.float64)
        return np.full(len(payload), np.nan, dtype=np.float64)

    def predict_next_distribution(self) -> np.ndarray:
        result = np.zeros(256, dtype=np.float64)
        result[self.next_symbol] = 1.0
        return result

    def advance_observation(self, byte_id: int, *, learn: bool = False) -> float:
        self.calls.append((bytes([byte_id]), learn))
        return 1.0


def test_vector_frame_is_canonical_dense_and_shape_bound() -> None:
    vector = np.array([1.25, -2.5, 0.0], dtype=np.float64)
    first = vector_input_frame(vector)
    second = vector_input_frame(vector.astype(np.float32))
    assert first == second
    assert first.endswith(vector.astype("<f4").tobytes())
    assert vector_input_frame(np.array([1.25, -2.5])) != first
    with pytest.raises(ValueError, match="finite"):
        vector_input_frame(np.array([np.nan]))


def test_fit_uses_vector_only_as_context_and_target_for_surprise_learning() -> None:
    cortex = FakeCortex()
    diagnostics = fit_vector_conditioned_turn(
        cortex, np.array([0.25, -0.5]), b"actual example"
    )
    assert cortex.calls[0][1] is False
    assert cortex.calls[0][0] == vector_input_frame(np.array([0.25, -0.5]))
    assert cortex.calls[1] == (b"actual example", True)
    assert diagnostics.target_bytes == len(b"actual example")
    assert diagnostics.mean_surprise_bits == pytest.approx(1.0)
    assert cortex.reset_count == 2


def test_score_and_generation_never_train_or_apply_a_receiver() -> None:
    cortex = FakeCortex()
    score = score_vector_conditioned_turn(cortex, [1.0, 2.0], b"OO")
    assert score.mean_surprise_bits == 0.0
    assert all(not learn for _, learn in cortex.calls)


def test_stateful_adapter_retains_associative_memory_across_context_resets() -> None:
    cortex = FakeCortex()
    config = StatefulVectorConditionedConfig(
        state_dimensions=8,
        learning_rate=1.0,
        state_decay=1.0,
        weight_decay=0.0,
        modulation_strength=8.0,
        seed=17,
    )
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    vector = np.array([0.2, -0.4, 0.8, 0.1])
    target = ord("Z")

    adapter.prime(vector, clear_memory=True)
    adapter.observe_chunk_probabilities(np.array([target], dtype=np.uint8), learn=True)
    assert np.linalg.norm(adapter.weights) > 0.0

    adapter.reset_context()
    adapter.prime(vector)
    retained_probability = float(adapter.predict_next_distribution()[target])

    fresh = StatefulVectorConditionedAdapter(FakeCortex(), config)
    fresh.prime(vector, clear_memory=True)
    fresh_probability = float(fresh.predict_next_distribution()[target])
    assert retained_probability > fresh_probability

    adapter.clear_memory()
    cleared_probability = float(adapter.predict_next_distribution()[target])
    assert cleared_probability == pytest.approx(fresh_probability)


def test_stateful_scoring_is_read_only_when_generation_updates_are_enabled() -> None:
    cortex = FakeCortex()
    config = StatefulVectorConditionedConfig(
        state_dimensions=4,
        learning_rate=1.0,
        modulation_strength=2.0,
        update_on_generated=True,
        seed=3,
    )
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    vector = np.array([0.2, -0.4, 0.8])
    adapter.prime(vector, clear_memory=True)
    before_state = adapter.state.copy()
    before_weights = adapter.weights.copy()

    adapter.observe_chunk_probabilities(
        np.array([ord("Z")], dtype=np.uint8), learn=False
    )

    np.testing.assert_array_equal(adapter.state, before_state)
    np.testing.assert_array_equal(adapter.weights, before_weights)
    assert all(not learn for _, learn in cortex.calls)


def test_stateful_generation_can_use_local_output_updates() -> None:
    cortex = FakeCortex()
    config = StatefulVectorConditionedConfig(
        state_dimensions=4,
        learning_rate=1.0,
        weight_decay=0.0,
        modulation_strength=2.0,
        seed=5,
    )
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    adapter.prime([0.1, -0.2, 0.3], clear_memory=True)
    adapter.set_generation_updates(True)
    adapter.advance_observation(ord("Z"), learn=False)
    assert np.linalg.norm(adapter.weights) > 0.0


def test_stateful_context_generation_separates_memory_from_context_exposure() -> None:
    config = StatefulVectorConditionedConfig(
        state_dimensions=4,
        learning_rate=1.0,
        weight_decay=0.0,
        modulation_strength=4.0,
        seed=11,
    )
    vector = np.array([0.1, -0.2, 0.3])
    context = b"prior turn"

    retained = generate_stateful_after_context(
        FakeCortex(), vector, context, horizon=2, config=config
    )
    cleared = generate_stateful_after_context(
        FakeCortex(),
        vector,
        context,
        horizon=2,
        config=config,
        clear_memory_after_context=True,
    )
    no_update = generate_stateful_after_context(
        FakeCortex(),
        vector,
        context,
        horizon=2,
        config=config,
        adapt_context=False,
    )

    assert retained.probability_sha256 != cleared.probability_sha256
    assert cleared.probability_sha256 == no_update.probability_sha256


def test_stateful_session_retains_persistent_state_and_weights_across_turns() -> None:
    cortex = FakeCortex()
    config = StatefulVectorConditionedConfig(
        state_dimensions=8,
        learning_rate=1.0,
        state_decay=1.0,
        weight_decay=0.0,
        modulation_strength=8.0,
        seed=17,
    )
    adapter = StatefulVectorConditionedAdapter(cortex, config)
    vector = np.array([0.2, -0.4, 0.8, 0.1])

    adapter.prime(vector, clear_memory=True)
    adapter.observe_context(b"Z", adapt=True)
    adapter.end_turn()
    adapter.prime(vector)

    assert np.linalg.norm(adapter.weights) > 0.0
    assert np.linalg.norm(adapter.retained_state) > 0.0

    adapter.prime(vector, clear_memory=True)
    assert np.linalg.norm(adapter.weights) == 0.0
    assert np.linalg.norm(adapter.retained_state) == 0.0


def test_stateful_generation_controls_separate_context_and_output_updates() -> None:
    config = StatefulVectorConditionedConfig(
        state_dimensions=4,
        learning_rate=1.0,
        weight_decay=0.0,
        modulation_strength=4.0,
        seed=11,
    )
    vector = np.array([0.1, -0.2, 0.3])
    context = b"prior turn"

    retained = generate_stateful_after_context(
        FakeCortex(), vector, context, horizon=2, config=config
    )
    cleared = generate_stateful_after_context(
        FakeCortex(),
        vector,
        context,
        horizon=2,
        config=config,
        clear_memory_after_context=True,
    )
    no_context_adapt = generate_stateful_after_context(
        FakeCortex(),
        vector,
        context,
        horizon=2,
        config=config,
        adapt_context=False,
    )
    no_generated_adapt = generate_stateful_after_context(
        FakeCortex(),
        vector,
        context,
        horizon=2,
        config=config,
        update_generated=False,
    )

    assert retained.probability_sha256 != cleared.probability_sha256
    assert cleared.probability_sha256 == no_context_adapt.probability_sha256
    assert no_generated_adapt.probability_sha256 != retained.probability_sha256


def test_stateful_multi_turn_api_preserves_memory_between_turn_boundaries() -> None:
    config = StatefulVectorConditionedConfig(
        state_dimensions=4,
        learning_rate=1.0,
        weight_decay=0.0,
        modulation_strength=4.0,
        seed=19,
    )
    vector = np.array([0.1, -0.2, 0.3])
    outputs = generate_stateful_multi_turn(
        FakeCortex(),
        (
            StatefulVectorConditionedTurn(
                vector=vector,
                context=b"first context",
                observed=b"first output",
                horizon=1,
                clear_memory=True,
            ),
            StatefulVectorConditionedTurn(
                vector=vector,
                context=b"second context",
                horizon=1,
            ),
        ),
        config=config,
    )
    assert len(outputs) == 2
    assert all(len(output.data) == 1 for output in outputs)


def test_context_matched_base_generation_observes_context_before_vector() -> None:
    cortex = FakeCortex()
    vector = np.array([0.1, -0.2, 0.3])
    context = b"prior"
    generated = generate_from_vector_after_context(
        cortex, vector, context, horizon=1
    )

    assert generated.data == b"O"
    assert cortex.calls[0] == (context, False)
    assert cortex.calls[1][0] == vector_input_frame(vector)
    assert cortex.calls[1][1] is False


def test_vector_frame_rejects_complex_values() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        vector_input_frame([1.0 + 2.0j])


def test_codec_experiment_trains_dense_oracle_vectors_directly_into_base() -> None:
    cortex = FakeCortex()
    turn = codec_assay.AssistantTurn(
        episode_id="episode",
        source_label="source",
        turn_index=1,
        prompt=b"",
        current_user_turn=b"question",
        content=b"answer",
        content_start=0,
        content_end=6,
    )
    conversation = codec_assay.TrainingConversation(
        episode_id="episode",
        source_label="source",
        payload=b"answer",
        assistant_turns=(turn,),
    )
    codec = codec_assay.FutureTurnCodec.fit(
        (
            turn,
            codec_assay.AssistantTurn(
                episode_id="other",
                source_label="source",
                turn_index=1,
                prompt=b"",
                current_user_turn=b"question",
                content=b"different",
                content_start=0,
                content_end=9,
            ),
        ),
        raw_dimensions=16,
        content_dimensions=2,
        active_content_features=1,
    )
    report = codec_assay.fit_vector_conditioned_base(
        cortex,
        (conversation,),
        codec,
        response_cap=32,
        training_seed=7,
    )
    assert report["architecture"] == "dense_predictor_vector_to_base_to_bytes"
    assert report["receiver_present"] is False
    assert report["symbols"] == len(b"answer")
    assert cortex.calls[-1] == (b"answer", True)

    cortex.calls.clear()
    generated = generate_from_vector(cortex, [1.0, 2.0], horizon=3)
    assert generated.data == b"OOO"
    assert len(generated.probability_sha256) == 64
    assert all(not learn for _, learn in cortex.calls)
