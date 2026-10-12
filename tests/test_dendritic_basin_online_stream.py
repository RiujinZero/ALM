"""Phase 12 tests for the unified bounded online stream learner."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)


def _context_stream(repeats: int) -> list[int]:
    # Token 0 has an ambiguous successor under a bigram model.  Its previous
    # context resolves the transition: 1,0 -> 2 and 2,0 -> 1.
    return [1, 0, 2, 0] * repeats


def _small_config(**overrides) -> OnlineDendriticConfig:
    values = {
        "vocab_size": 4,
        "timescales": (1, 2, 4, 8),
        "max_relations": 16,
        "max_active_relations": 4,
        "surprise_birth_bits": 1.0,
        "birth_cooldown": 1,
        "min_prune_age": 32,
        "max_staleness": 128,
        "persistence_decay": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return OnlineDendriticConfig(**values)


def test_predictions_are_prequential_and_future_blind() -> None:
    left = OnlineDendriticStreamLearner(_small_config())
    right = OnlineDendriticStreamLearner(_small_config())
    prefix = _context_stream(40)

    for token_id in prefix:
        left_step = left.observe(token_id)
        right_step = right.observe(token_id)
        np.testing.assert_array_equal(left_step.prediction, right_step.prediction)

    prediction_before_different_futures = left_step.prediction.copy()
    left.observe(1)
    right.observe(3)
    np.testing.assert_array_equal(
        prediction_before_different_futures, right_step.prediction
    )


def test_context_relations_beat_the_online_unigram_baseline() -> None:
    learner = OnlineDendriticStreamLearner(_small_config(max_relations=32))
    model_probabilities: list[float] = []
    base_probabilities: list[float] = []

    for token_id in _context_stream(750):
        if learner._pending is not None:
            base_probabilities.append(
                float(learner._pending.base_distribution[token_id])
            )
        step = learner.observe(token_id)
        if step.observed_probability is not None:
            model_probabilities.append(step.observed_probability)

    model_bits = float(-np.log2(model_probabilities[-1000:]).mean())
    base_bits = float(-np.log2(base_probabilities[-1000:]).mean())
    assert learner.stats.births > 0
    assert learner.stats.relation_firings > 0
    assert model_bits < 0.75 * base_bits


def test_population_and_every_branch_cache_remain_bounded() -> None:
    config = _small_config(
        vocab_size=16,
        max_relations=7,
        max_active_relations=3,
        local_buffer_size=11,
        surprise_birth_bits=0.5,
    )
    learner = OnlineDendriticStreamLearner(config)
    rng = np.random.default_rng(9)
    for token_id in rng.integers(0, config.vocab_size, size=5000):
        learner.observe(int(token_id))

    assert learner.live_relation_count <= config.max_relations
    assert learner.live_nursery_count <= config.max_nursery_candidates
    assert learner.stats.max_live_relations <= config.max_relations
    assert all(
        len(slot.local_samples) <= config.local_buffer_size
        for slot in learner.slots
        if slot is not None
    )
    assert np.all(learner._local_sample_counts <= config.local_buffer_size)
    assert learner._local_samples.shape == (
        config.max_relations,
        config.local_buffer_size,
        2,
    )


def test_relation_graph_is_birth_ordered_and_depth_bounded() -> None:
    learner = OnlineDendriticStreamLearner(
        _small_config(
            max_relations=24,
            max_depth=3,
            recursive_birth_probability=1.0,
        )
    )
    for token_id in _context_stream(1000):
        learner.observe(token_id)

    for slot_id, slot in enumerate(learner.slots):
        if slot is None:
            continue
        assert slot.depth <= learner.config.max_depth
        for channel in (slot.current_channel, slot.trace_channel):
            if channel >= learner.config.vocab_size:
                assert channel - learner.config.vocab_size < slot_id


def test_nursery_requires_repeated_consistent_evidence() -> None:
    learner = OnlineDendriticStreamLearner(
        _small_config(nursery_confirmations=3, nursery_target_consistency=0.75)
    )
    parents = (0, 1, 0, 1.0, 0.4)

    candidate = learner._nursery_candidate(parents, token_id=2)
    assert not learner._candidate_is_ready(candidate)
    candidate = learner._nursery_candidate(parents, token_id=2)
    assert not learner._candidate_is_ready(candidate)
    candidate = learner._nursery_candidate(parents, token_id=2)
    assert learner._candidate_is_ready(candidate)

    inconsistent = (0, 2, 0, 1.0, 0.4)
    learner._nursery_candidate(inconsistent, token_id=1)
    learner._nursery_candidate(inconsistent, token_id=2)
    candidate = learner._nursery_candidate(inconsistent, token_id=3)
    assert not learner._candidate_is_ready(candidate)


def test_checkpoint_resume_is_exact(tmp_path) -> None:
    learner = OnlineDendriticStreamLearner(
        _small_config(max_relations=20, contextual_residual=True)
    )
    stream = _context_stream(300)
    split = 731
    for token_id in stream[:split]:
        learner.observe(token_id)

    checkpoint = tmp_path / "online.pkl"
    learner.save_checkpoint(checkpoint)
    resumed = OnlineDendriticStreamLearner.load_checkpoint(checkpoint)

    for token_id in stream[split:]:
        expected = learner.observe(token_id)
        actual = resumed.observe(token_id)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    np.testing.assert_array_equal(learner.traces, resumed.traces)
    np.testing.assert_array_equal(learner.base_counts, resumed.base_counts)
    np.testing.assert_array_equal(learner.context_counts, resumed.context_counts)
    assert learner.previous_token == resumed.previous_token
    assert learner.stats == resumed.stats


def test_phase13_state_loads_without_contextual_tissue() -> None:
    config = _small_config(max_relations=20)
    learner = OnlineDendriticStreamLearner(config)
    for token_id in _context_stream(50):
        learner.observe(token_id)
    state = learner.state_dict()
    for key in ("context_decay", "context_prior", "contextual_residual"):
        state["config"].pop(key)
    state.pop("context_counts")
    state.pop("previous_token")

    resumed = OnlineDendriticStreamLearner(config)
    resumed.load_state_dict(state)

    assert resumed.context_counts.shape == (0, 0)
    assert resumed.previous_token == -1
    np.testing.assert_array_equal(resumed.base_counts, learner.base_counts)


def test_episode_reset_clears_activity_but_keeps_anatomy() -> None:
    learner = OnlineDendriticStreamLearner(_small_config(persistence_decay=0.99))
    for token_id in _context_stream(100):
        learner.observe(token_id)
    relation_count = learner.live_relation_count

    learner.reset_episode()

    assert learner.live_relation_count == relation_count
    assert learner._pending is None
    assert not np.any(learner.traces)
    assert all(
        slot.persistence == 0.0 for slot in learner.slots if slot is not None
    )


def test_frozen_evaluation_changes_activity_not_learned_structure() -> None:
    learner = OnlineDendriticStreamLearner(_small_config(max_relations=20))
    for token_id in _context_stream(200):
        learner.observe(token_id)
    learner.reset_episode()
    base_counts = learner.base_counts.copy()
    nursery = learner.live_nursery_count
    population_stats = learner.stats.__dict__.copy()
    slot_snapshots = [
        None
        if slot is None
        else (
            slot.predictor_counts.copy(),
            slot.state_predictor_counts.copy(),
            slot.utility_bits,
            slot.state_utility_bits,
            slot.state_observations,
            slot.observations,
            slot.fires,
            list(slot.local_samples),
            (
                slot.dendrite.w_a,
                slot.dendrite.w_b,
                slot.dendrite.mu,
                slot.dendrite.sigma,
            ),
        )
        for slot in learner.slots
    ]

    for token_id in _context_stream(20):
        learner.observe(token_id, learn=False)

    np.testing.assert_array_equal(learner.base_counts, base_counts)
    assert learner.live_nursery_count == nursery
    assert learner.stats.__dict__ == population_stats
    for slot, snapshot in zip(learner.slots, slot_snapshots, strict=True):
        if slot is None:
            assert snapshot is None
            continue
        assert snapshot is not None
        (
            counts,
            state_counts,
            utility,
            state_utility,
            state_observations,
            observations,
            fires,
            samples,
            dendrite,
        ) = snapshot
        np.testing.assert_array_equal(slot.predictor_counts, counts)
        np.testing.assert_array_equal(slot.state_predictor_counts, state_counts)
        assert slot.utility_bits == utility
        assert slot.state_utility_bits == state_utility
        assert slot.state_observations == state_observations
        assert slot.observations == observations
        assert slot.fires == fires
        assert list(slot.local_samples) == samples
        assert (
            slot.dendrite.w_a,
            slot.dendrite.w_b,
            slot.dendrite.mu,
            slot.dendrite.sigma,
        ) == dendrite


def test_batched_backend_matches_scalar_learning_trajectory() -> None:
    scalar = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, response_backend="scalar")
    )
    batched = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, response_backend="batched")
    )

    for token_id in _context_stream(500):
        expected = scalar.observe(token_id)
        actual = batched.observe(token_id)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert scalar.stats == batched.stats
    for expected_slot, actual_slot in zip(
        scalar.slots, batched.slots, strict=True
    ):
        assert (expected_slot is None) == (actual_slot is None)
        if expected_slot is None or actual_slot is None:
            continue
        assert expected_slot.observations == actual_slot.observations
        assert expected_slot.fires == actual_slot.fires
        assert expected_slot.utility_bits == actual_slot.utility_bits
        assert expected_slot.state_utility_bits == actual_slot.state_utility_bits
        assert (
            expected_slot.dendrite.w_a,
            expected_slot.dendrite.w_b,
            expected_slot.dendrite.mu,
            expected_slot.dendrite.sigma,
        ) == (
            actual_slot.dendrite.w_a,
            actual_slot.dendrite.w_b,
            actual_slot.dendrite.mu,
            actual_slot.dendrite.sigma,
        )


def test_batched_competition_matches_scalar_learning_trajectory() -> None:
    scalar = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, competition_backend="scalar")
    )
    batched = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, competition_backend="batched")
    )

    for token_id in _context_stream(500):
        expected = scalar.observe(token_id)
        actual = batched.observe(token_id)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert scalar.stats == batched.stats


def test_compiled_competition_matches_batched_learning_trajectory() -> None:
    batched = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, competition_backend="batched")
    )
    compiled = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, competition_backend="compiled")
    )
    compiled.warm_execution_backend()

    for token_id in _context_stream(500):
        expected = batched.observe(token_id)
        actual = compiled.observe(token_id)
        np.testing.assert_allclose(
            expected.prediction, actual.prediction, rtol=1e-14, atol=1e-14
        )
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert batched.stats == compiled.stats
    np.testing.assert_allclose(
        batched._utility_bits, compiled._utility_bits, rtol=1e-14, atol=1e-14
    )


def test_compiled_responses_match_batched_learning_trajectory() -> None:
    batched = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, response_backend="batched")
    )
    compiled = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, response_backend="compiled")
    )
    compiled.warm_execution_backend()

    for token_id in _context_stream(500):
        expected = batched.observe(token_id)
        actual = compiled.observe(token_id)
        np.testing.assert_allclose(
            expected.prediction, actual.prediction, rtol=1e-14, atol=1e-14
        )
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert batched.stats == compiled.stats


def test_compiled_adaptation_matches_scalar_learning_trajectory() -> None:
    scalar = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, adaptation_backend="scalar")
    )
    compiled = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, adaptation_backend="compiled")
    )
    compiled.warm_execution_backend()

    for token_id in _context_stream(500):
        expected = scalar.observe(token_id)
        actual = compiled.observe(token_id)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert scalar.stats == compiled.stats


def test_compiled_outcomes_match_scalar_learning_trajectory() -> None:
    scalar = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, outcome_backend="scalar")
    )
    compiled = OnlineDendriticStreamLearner(
        _small_config(max_relations=24, outcome_backend="compiled")
    )
    compiled.warm_execution_backend()

    for token_id in _context_stream(500):
        expected = scalar.observe(token_id)
        actual = compiled.observe(token_id)
        np.testing.assert_allclose(
            expected.prediction, actual.prediction, rtol=1e-14, atol=1e-14
        )
        assert expected.active_relations == actual.active_relations
        assert expected.born_relation == actual.born_relation
        assert expected.pruned_relation == actual.pruned_relation

    assert scalar.stats == compiled.stats
    np.testing.assert_allclose(
        scalar._utility_bits, compiled._utility_bits, rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("contextual_residual", [False, True])
def test_compiled_byte_chunk_matches_one_at_a_time_learning(
    contextual_residual: bool,
) -> None:
    config = _small_config(
        max_relations=24,
        adaptation_rng="counter",
        column_size=4,
        max_active_columns=3,
        max_active_per_column=2,
        consolidation_min_age=32,
        consolidation_min_fires=2,
        consolidation_utility_threshold=-1.0,
        dormancy_staleness=64,
        contextual_residual=contextual_residual,
    )
    scalar_loop = OnlineDendriticStreamLearner(config)
    chunk_loop = OnlineDendriticStreamLearner(config)
    prefix = np.asarray(_context_stream(80), dtype=np.int64)
    continuation = np.asarray(_context_stream(250), dtype=np.int64)
    for token_id in prefix:
        scalar_loop.observe(int(token_id))
        chunk_loop.observe(int(token_id))

    expected_probabilities = []
    expected_base_probabilities = []
    for token_id in continuation:
        step = scalar_loop.observe(int(token_id))
        expected_probabilities.append(step.observed_probability)
        expected_base_probabilities.append(step.base_observed_probability)
    actual_probabilities, actual_base_probabilities = (
        chunk_loop.observe_chunk_probabilities(continuation)
    )

    np.testing.assert_allclose(
        actual_probabilities,
        expected_probabilities,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        actual_base_probabilities,
        expected_base_probabilities,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(scalar_loop.traces, chunk_loop.traces)
    np.testing.assert_allclose(scalar_loop.base_counts, chunk_loop.base_counts)
    np.testing.assert_allclose(
        scalar_loop.context_counts, chunk_loop.context_counts
    )
    assert scalar_loop.previous_token == chunk_loop.previous_token
    np.testing.assert_allclose(
        scalar_loop._predictor_counts, chunk_loop._predictor_counts
    )
    assert scalar_loop.stats == chunk_loop.stats
    assert [slot is None for slot in scalar_loop.slots] == [
        slot is None for slot in chunk_loop.slots
    ]


def test_contextual_basal_column_learns_causal_byte_transitions() -> None:
    contextual = OnlineDendriticStreamLearner(
        _small_config(
            contextual_residual=True,
            surprise_birth_bits=100.0,
        )
    )
    unigram = OnlineDendriticStreamLearner(
        _small_config(surprise_birth_bits=100.0)
    )
    contextual_probabilities = []
    unigram_probabilities = []
    stream = [0, 1] * 1000
    for token_id in stream:
        contextual_step = contextual.observe(token_id)
        unigram_step = unigram.observe(token_id)
        if contextual_step.observed_probability is not None:
            contextual_probabilities.append(contextual_step.observed_probability)
            unigram_probabilities.append(unigram_step.observed_probability)

    contextual_bits = float(-np.log2(contextual_probabilities[-500:]).mean())
    unigram_bits = float(-np.log2(unigram_probabilities[-500:]).mean())
    assert contextual_bits < 0.1
    assert contextual_bits < 0.1 * unigram_bits


def test_contextual_dendrites_resolve_higher_order_ambiguity() -> None:
    learner = OnlineDendriticStreamLearner(
        _small_config(
            max_relations=32,
            contextual_residual=True,
        )
    )
    model_probabilities = []
    basal_probabilities = []
    for token_id in _context_stream(1000):
        step = learner.observe(token_id)
        if step.observed_probability is not None:
            model_probabilities.append(step.observed_probability)
            basal_probabilities.append(step.base_observed_probability)

    model_bits = float(-np.log2(model_probabilities[-1000:]).mean())
    basal_bits = float(-np.log2(basal_probabilities[-1000:]).mean())
    assert learner.stats.relation_firings > 0
    assert model_bits < 0.1 * basal_bits


def test_event_routing_matches_dense_frozen_causal_execution() -> None:
    dense_config = _small_config(
        max_relations=32,
        metabolic_backend="dense",
    )
    trained = OnlineDendriticStreamLearner(dense_config)
    for token_id in _context_stream(300):
        trained.observe(token_id)

    state = trained.state_dict()
    dense = OnlineDendriticStreamLearner(dense_config)
    dense.load_state_dict(state)
    event = OnlineDendriticStreamLearner(
        replace(dense_config, metabolic_backend="event")
    )
    event_state = {
        **state,
        "config": {**state["config"], "metabolic_backend": "event"},
    }
    event.load_state_dict(event_state)
    dense.reset_episode()
    event.reset_episode()

    for token_id in _context_stream(100):
        expected = dense.observe(token_id, learn=False)
        actual = event.observe(token_id, learn=False)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)
        assert expected.active_relations == actual.active_relations


def test_byte_class_derives_capacity_and_enforces_checkpoint_ceiling(
    tmp_path,
) -> None:
    config = _small_config(
        vocab_size=16,
        timescales=(1, 2, 4),
        max_alm_bytes=2_000_000,
    )
    learner = OnlineDendriticStreamLearner(config)
    rng = np.random.default_rng(19)
    for token_id in rng.integers(0, config.vocab_size, size=3000):
        learner.observe(int(token_id))

    checkpoint = tmp_path / "alm-2mb.pkl"
    learner.save_checkpoint(checkpoint)

    assert config.max_relations == config.max_nursery_candidates
    assert config.estimated_max_serialized_bytes is not None
    assert config.estimated_max_serialized_bytes <= config.max_alm_bytes
    assert checkpoint.stat().st_size == learner.serialized_size_bytes()
    assert checkpoint.stat().st_size <= config.max_alm_bytes


def test_byte_budget_pressure_sheds_provisional_tissue(tmp_path) -> None:
    config = _small_config(
        vocab_size=16,
        max_relations=4,
        max_nursery_candidates=128,
        surprise_birth_bits=0.5,
    )
    learner = OnlineDendriticStreamLearner(config)
    rng = np.random.default_rng(23)
    for token_id in rng.integers(0, config.vocab_size, size=500):
        learner.observe(int(token_id))

    assert learner.nursery
    unrestricted_size = learner.serialized_size_bytes()
    nursery_before = len(learner.nursery)
    object.__setattr__(config, "max_alm_bytes", unrestricted_size - 1)

    checkpoint = tmp_path / "pressure.pkl"
    learner.save_checkpoint(checkpoint)

    assert checkpoint.stat().st_size <= config.max_alm_bytes
    assert len(learner.nursery) < nursery_before
    assert learner.stats.budget_evictions > 0


def test_columns_distribute_anatomy_and_compete_locally() -> None:
    config = _small_config(
        vocab_size=8,
        timescales=(1, 2, 4),
        max_relations=24,
        max_nursery_candidates=64,
        column_size=8,
        max_active_relations=6,
        max_active_columns=3,
        max_active_per_column=2,
        inter_column_birth_probability=0.0,
        surprise_birth_bits=0.5,
        seed=3,
    )
    learner = OnlineDendriticStreamLearner(config)
    stream = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0] * 300
    maximum_column_winners = 0
    for token_id in stream:
        step = learner.observe(token_id)
        for column_id in range(learner.column_count):
            winners = sum(
                slot_id // config.column_size == column_id
                for slot_id in step.active_relations
            )
            maximum_column_winners = max(maximum_column_winners, winners)
            assert winners <= config.max_active_per_column

    occupancy = [
        sum(
            slot is not None and slot.column_id == column_id
            for slot in learner.slots
        )
        for column_id in range(learner.column_count)
    ]
    assert all(count > 0 for count in occupancy)
    assert maximum_column_winners > 0
    for slot in learner.slots:
        if slot is None:
            continue
        assert slot.column_id == slot.slot_id // config.column_size
        for channel in (slot.current_channel, slot.trace_channel):
            if channel >= config.vocab_size:
                parent_column = (
                    channel - config.vocab_size
                ) // config.column_size
                assert parent_column == slot.column_id


def test_consolidated_branch_can_sleep_and_reactivate() -> None:
    learner = OnlineDendriticStreamLearner(
        _small_config(
            max_relations=24,
            consolidation_min_age=1,
            consolidation_min_fires=1,
            consolidation_utility_threshold=-1.0,
            dormancy_staleness=32,
            dormant_min_residence=1,
            lifecycle_sweep_interval=1,
        )
    )
    for token_id in _context_stream(100):
        learner.observe(token_id)
    slot_id = int(np.flatnonzero(learner._live)[0])
    slot = learner.slots[slot_id]
    assert slot is not None
    learner.config = replace(learner.config, surprise_birth_bits=100.0)
    learner._born_steps[slot_id] = learner.stats.steps - 2
    learner._fires[slot_id] = 2
    learner._utility_bits[slot_id] = 0.1
    learner._last_fire_steps[slot_id] = learner.stats.steps

    learner.observe(3)
    assert learner._lifecycle[slot_id] == 1

    learner._last_fire_steps[slot_id] = learner.stats.steps - 33
    learner.observe(3)
    assert learner._lifecycle[slot_id] == 2

    assert slot.current_channel < learner.config.vocab_size
    slot.dendrite.w_a = 0.0
    slot.dendrite.w_b = 0.0
    slot.dendrite.mu = 0.0
    slot.dendrite.sigma = 1.0
    learner._sync_slot_parameters(slot_id)
    learner.observe(slot.current_channel)
    assert learner._lifecycle[slot_id] == 1
    assert learner.stats.consolidations >= 1
    assert learner.stats.dormancies >= 1
    assert learner.stats.reactivations >= 1

    learner._lifecycle[slot_id] = 2
    learner._dormant_steps[slot_id] = learner.stats.steps
    learner._utility_bits[slot_id] = learner.config.prune_utility - 1.0
    learner.observe(slot.current_channel)
    assert learner._lifecycle[slot_id] == 2
