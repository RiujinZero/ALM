from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from manifoldflow.dendritic_basin.cpu_column_stream import CpuColumnFrozenEvaluator
from manifoldflow.dendritic_basin.gpu_multicolumn_stream import (
    CudaDevelopmentalRunner,
    CudaMultiColumnFrozenEvaluator,
)
from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticConfig,
    OnlineDendriticStreamLearner,
)

try:
    import cupy as cp

    CUDA_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - depends on test host
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA runtime is unavailable")
@pytest.mark.parametrize("metabolic_backend", ("dense", "event"))
def test_gpu_plasticity_sync_preserves_cpu_causal_continuation(
    metabolic_backend: str,
) -> None:
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_relations=256,
        max_nursery_candidates=256,
        contextual_residual=True,
        surprise_birth_bits=2.0,
        birth_cooldown=1,
        nursery_confirmations=2,
        min_prune_age=64,
        seed=31,
    )
    source = OnlineDendriticStreamLearner(config)
    training = np.resize(np.asarray((65, 66, 65, 67, 65, 68), dtype=np.uint8), 5000)
    source.observe_chunk_probabilities(training, learn=True)
    assert source.live_relation_count > 0

    state = source.state_dict()
    cpu = OnlineDendriticStreamLearner(config)
    cpu.load_state_dict(state)
    gpu_host = OnlineDendriticStreamLearner(config)
    gpu_host.load_state_dict(state)
    frozen_development = {
        "surprise_birth_bits": 1e9,
        "adapt_interval": 1_000_000_000,
        "consolidated_adapt_interval": 1_000_000_000,
        "consolidation_min_age": 1_000_000_000,
        "dormancy_staleness": 1_000_000_000,
        "lifecycle_sweep_interval": 1_000_000_000,
        "metabolic_backend": metabolic_backend,
        "utility_gated_persistence": metabolic_backend == "event",
    }
    cpu.config = replace(cpu.config, **frozen_development)
    gpu_host.config = replace(gpu_host.config, **frozen_development)
    cpu.reset_episode()
    gpu_host.reset_episode()

    stream = np.resize(np.asarray((65, 66, 65, 68, 65, 67), dtype=np.uint8), 1024)
    cpu_probability, _ = cpu.observe_chunk_probabilities(stream, learn=True)
    evaluator = CudaMultiColumnFrozenEvaluator(gpu_host)
    gpu_probability = evaluator.observe_chunk_probabilities(stream, learn=True)
    tolerance = 1e-5 if metabolic_backend == "event" else 3e-6
    np.testing.assert_allclose(
        gpu_probability[1:], cpu_probability[1:], rtol=0.0, atol=tolerance
    )

    evaluator.synchronize_plasticity_to(gpu_host)
    continuation = np.resize(np.asarray((65, 68, 65, 66), dtype=np.uint8), 256)
    cpu_continuation, _ = cpu.observe_chunk_probabilities(continuation, learn=False)
    gpu_continuation, _ = gpu_host.observe_chunk_probabilities(
        continuation, learn=False
    )
    np.testing.assert_allclose(
        gpu_continuation, cpu_continuation, rtol=0.0, atol=tolerance
    )
    np.testing.assert_array_equal(gpu_host._observations, cpu._observations)
    np.testing.assert_array_equal(gpu_host._fires, cpu._fires)
    np.testing.assert_array_equal(gpu_host._state_observations, cpu._state_observations)
    assert gpu_host.stats.steps == cpu.stats.steps
    assert gpu_host.stats.predictions_scored == cpu.stats.predictions_scored
    assert gpu_host.stats.branch_evaluations == cpu.stats.branch_evaluations
    assert gpu_host.stats.relation_firings == cpu.stats.relation_firings
    np.testing.assert_array_equal(
        gpu_host._local_sample_counts, cpu._local_sample_counts
    )
    np.testing.assert_array_equal(
        gpu_host._local_sample_positions, cpu._local_sample_positions
    )
    np.testing.assert_array_equal(gpu_host._stats_n, cpu._stats_n)
    np.testing.assert_allclose(
        gpu_host._local_samples, cpu._local_samples, rtol=0.0, atol=1e-5
    )
    for name in (
        "_stats_mean_a",
        "_stats_mean_b",
        "_stats_mean_d",
        "_stats_m2_a",
        "_stats_m2_b",
        "_stats_m2_d",
        "_stats_cov_da_acc",
        "_stats_cov_db_acc",
    ):
        np.testing.assert_allclose(
            getattr(gpu_host, name), getattr(cpu, name), rtol=0.0, atol=1e-3
        )

    cpu.config = replace(cpu.config, adapt_interval=1, consolidated_adapt_interval=1)
    gpu_host.config = replace(
        gpu_host.config, adapt_interval=1, consolidated_adapt_interval=1
    )
    cpu.observe(65, learn=True)
    gpu_host.observe(65, learn=True)
    np.testing.assert_allclose(gpu_host._w_a, cpu._w_a, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(gpu_host._w_b, cpu._w_b, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(gpu_host._mu, cpu._mu, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(gpu_host._sigma, cpu._sigma, rtol=0.0, atol=2e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA runtime is unavailable")
def test_gpu_developmental_barrier_preserves_birth_prune_trajectory() -> None:
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_relations=256,
        max_nursery_candidates=256,
        contextual_residual=True,
        surprise_birth_bits=2.0,
        birth_cooldown=1,
        nursery_confirmations=2,
        min_prune_age=8,
        prune_utility=1e9,
        metabolic_backend="event",
        utility_gated_persistence=True,
        seed=37,
    )
    stream = np.resize(
        np.asarray((65, 66, 65, 67, 65, 68, 66, 67), dtype=np.uint8),
        768,
    )
    cpu = OnlineDendriticStreamLearner(config)
    gpu_host = OnlineDendriticStreamLearner(config)
    cpu_probability, cpu_base = cpu.observe_chunk_probabilities(stream, learn=True)
    runner = CudaDevelopmentalRunner(gpu_host, structural_wake_bytes=64)
    gpu_probability, gpu_base = runner.observe_chunk_probabilities(stream)
    runner.synchronize()

    np.testing.assert_allclose(
        gpu_probability[1:], cpu_probability[1:], rtol=0.0, atol=2e-5
    )
    np.testing.assert_allclose(gpu_base[1:], cpu_base[1:], rtol=0.0, atol=2e-5)
    assert gpu_host.stats.births == cpu.stats.births
    assert gpu_host.stats.prunes == cpu.stats.prunes
    assert gpu_host.live_relation_count == cpu.live_relation_count
    assert gpu_host.stats.steps == cpu.stats.steps


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA runtime is unavailable")
def test_gpu_resident_adaptation_and_lifecycle_match_cpu() -> None:
    config = OnlineDendriticConfig(
        vocab_size=256,
        max_relations=256,
        max_nursery_candidates=256,
        contextual_residual=True,
        surprise_birth_bits=2.0,
        birth_cooldown=1,
        nursery_confirmations=2,
        min_prune_age=64,
        seed=41,
    )
    source = OnlineDendriticStreamLearner(config)
    developmental = np.resize(
        np.asarray((65, 66, 65, 67, 65, 68), dtype=np.uint8), 5000
    )
    source.observe_chunk_probabilities(developmental, learn=True)
    assert source.live_relation_count > 0

    state = source.state_dict()
    cpu = OnlineDendriticStreamLearner(config)
    cpu.load_state_dict(state)
    oracle = OnlineDendriticStreamLearner(config)
    oracle.load_state_dict(state)
    gpu_host = OnlineDendriticStreamLearner(config)
    gpu_host.load_state_dict(state)
    resident_development = {
        "surprise_birth_bits": 1e9,
        "metabolic_backend": "event",
        "utility_gated_persistence": True,
        "adapt_interval": 4,
        "consolidated_adapt_interval": 8,
        "min_adapt_samples": 4,
        "lifecycle_sweep_interval": 1,
        "consolidation_min_age": 1,
        "consolidation_min_fires": 1,
        "consolidation_utility_threshold": -1e6,
        "dormancy_staleness": 32,
        "prune_utility": -1e6,
    }
    cpu.config = replace(cpu.config, **resident_development)
    oracle.config = replace(oracle.config, **resident_development)
    gpu_host.config = replace(gpu_host.config, **resident_development)
    cpu.reset_episode()
    oracle.reset_episode()
    gpu_host.reset_episode()

    stream = np.resize(np.asarray((65, 68, 65, 66, 65, 67), dtype=np.uint8), 512)
    cpu_evaluator = CpuColumnFrozenEvaluator(cpu, workers=1)
    cpu_probability = cpu_evaluator.observe_chunk_probabilities(stream, learn=True)
    cpu_evaluator.synchronize_plasticity_to(cpu)
    oracle.observe_chunk_probabilities(stream, learn=True)
    evaluator = CudaMultiColumnFrozenEvaluator(gpu_host)
    gpu_probability = evaluator.observe_chunk_probabilities(stream, learn=True)
    evaluator.synchronize_plasticity_to(gpu_host)

    np.testing.assert_allclose(
        gpu_probability[1:], cpu_probability[1:], rtol=0.0, atol=2e-4
    )
    np.testing.assert_allclose(gpu_host._w_a, cpu._w_a, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(gpu_host._w_b, cpu._w_b, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(gpu_host._mu, cpu._mu, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(gpu_host._sigma, cpu._sigma, rtol=0.0, atol=2e-5)
    np.testing.assert_array_equal(gpu_host._lifecycle, oracle._lifecycle)
    np.testing.assert_array_equal(
        gpu_host._consolidated_steps, oracle._consolidated_steps
    )
    np.testing.assert_array_equal(gpu_host._dormant_steps, oracle._dormant_steps)
    np.testing.assert_array_equal(gpu_host._last_fire_steps, oracle._last_fire_steps)
    assert gpu_host.stats.consolidations == oracle.stats.consolidations
    assert gpu_host.stats.dormancies == oracle.stats.dormancies
    assert gpu_host.stats.reactivations == oracle.stats.reactivations
