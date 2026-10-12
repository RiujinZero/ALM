from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from manifoldflow.dendritic_basin.compiled_kernels import NUMBA_AVAILABLE
from manifoldflow.dendritic_basin.cpu_column_stream import (
    CpuColumnFrozenEvaluator,
    observe_developmental_probabilities,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is unavailable")
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.parametrize("learn", (False, True))
def test_resident_cpu_columns_match_event_oracle_across_chunks(
    workers: int, learn: bool
) -> None:
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_relations=512,
        max_nursery_candidates=512,
        contextual_residual=True,
        surprise_birth_bits=2.0,
        birth_cooldown=1,
        nursery_confirmations=2,
        seed=31,
    )
    source = OnlineDendriticStreamLearner(config)
    training = np.resize(
        np.asarray((65, 66, 65, 67, 65, 68), dtype=np.uint8), 5000
    )
    source.observe_chunk_probabilities(training, learn=True)
    state = source.state_dict()
    host = OnlineDendriticStreamLearner(config)
    host.load_state_dict(state)
    host.config = replace(
        host.config,
        metabolic_backend="event",
        utility_gated_persistence=True,
        surprise_birth_bits=1e9,
        adapt_interval=1_000_000_000,
        consolidated_adapt_interval=1_000_000_000,
    )
    host.reset_episode()
    tokens = np.resize(
        np.asarray((65, 66, 65, 68, 65, 67), dtype=np.uint8), 512
    )
    evaluator = CpuColumnFrozenEvaluator(host, workers=workers)
    expected, _ = host.observe_chunk_probabilities(tokens, learn=learn)

    first = evaluator.observe_chunk_probabilities(tokens[:173], learn=learn)
    second = evaluator.observe_chunk_probabilities(tokens[173:], learn=learn)
    actual = np.concatenate((first, second))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-7)
    if learn:
        evaluator._materialize_traces()
        np.testing.assert_allclose(evaluator.base_counts, host.base_counts)
        np.testing.assert_allclose(evaluator.context_counts, host.context_counts)
        np.testing.assert_allclose(
            evaluator.traces, host.traces, rtol=0.0, atol=2e-5
        )
        np.testing.assert_allclose(
            evaluator.trace_means, host.trace_means, rtol=0.0, atol=2e-5
        )
        np.testing.assert_allclose(evaluator.predictor_counts, host._predictor_counts)
        for name in ("w_a", "w_b", "mu", "sigma"):
            np.testing.assert_allclose(
                getattr(evaluator, name),
                getattr(host, f"_{name}"),
                rtol=0.0,
                atol=2e-5,
            )
        np.testing.assert_allclose(
            evaluator.state_predictor_counts, host._state_predictor_counts
        )
        np.testing.assert_array_equal(evaluator.observations, host._observations)
        np.testing.assert_array_equal(evaluator.fires, host._fires)
        np.testing.assert_array_equal(evaluator.lifecycle, host._lifecycle)
        np.testing.assert_array_equal(
            evaluator.consolidated_steps, host._consolidated_steps
        )
        np.testing.assert_array_equal(evaluator.dormant_steps, host._dormant_steps)
        np.testing.assert_array_equal(
            evaluator.local_sample_counts, host._local_sample_counts
        )
        np.testing.assert_array_equal(evaluator.stats_n, host._stats_n)
        np.testing.assert_allclose(
            evaluator.local_samples, host._local_samples, rtol=0.0, atol=2e-5
        )
        for name in (
            "stats_mean_a",
            "stats_mean_b",
            "stats_mean_d",
            "stats_m2_a",
            "stats_m2_b",
            "stats_m2_d",
            "stats_cov_da_acc",
            "stats_cov_db_acc",
        ):
            np.testing.assert_allclose(
                getattr(evaluator, name),
                getattr(host, f"_{name}"),
                rtol=0.0,
                atol=5e-5,
            )
        resumed = OnlineDendriticStreamLearner(config)
        resumed.load_state_dict(state)
        resumed.config = host.config
        resumed.reset_episode()
        evaluator.synchronize_plasticity_to(resumed)
        continuation = np.asarray((65, 66, 65, 67), dtype=np.uint8)
        expected_continuation, _ = host.observe_chunk_probabilities(
            continuation, learn=False
        )
        actual_continuation, _ = resumed.observe_chunk_probabilities(
            continuation, learn=False
        )
        np.testing.assert_allclose(
            actual_continuation, expected_continuation, rtol=0.0, atol=3e-7
        )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is unavailable")
def test_resident_columns_cross_exact_developmental_barriers() -> None:
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_relations=64,
        max_nursery_candidates=64,
        contextual_residual=True,
        surprise_birth_bits=2.0,
        birth_cooldown=1,
        nursery_confirmations=2,
        seed=7,
        metabolic_backend="event",
        adaptation_rng="counter",
        adapt_interval=1_000_000_000,
        consolidated_adapt_interval=1_000_000_000,
    )
    expected_learner = OnlineDendriticStreamLearner(config)
    actual_learner = OnlineDendriticStreamLearner(config)
    tokens = np.resize(
        np.asarray((65, 66, 65, 67, 65, 68), dtype=np.uint8), 1500
    )

    expected, expected_base = expected_learner.observe_chunk_probabilities(
        tokens, learn=True
    )
    actual, actual_base = observe_developmental_probabilities(
        actual_learner, tokens, workers=2
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(actual_base, expected_base, rtol=0.0, atol=3e-6)
    assert actual_learner.stats.births == expected_learner.stats.births
    assert actual_learner.stats.prunes == expected_learner.stats.prunes
    assert actual_learner.live_relation_count == expected_learner.live_relation_count
