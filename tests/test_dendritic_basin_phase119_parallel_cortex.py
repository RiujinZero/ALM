import numpy as np
import pytest

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_phase119_parallel_cortex import (
    pool_probabilities,
)


def test_uniform_parallel_cortex_is_arithmetic_mixture() -> None:
    columns = np.asarray(((0.8, 0.2), (0.4, 0.6)), dtype=np.float64)

    pooled, weights = pool_probabilities(columns, mode="uniform")

    np.testing.assert_allclose(pooled, (0.6, 0.4))
    np.testing.assert_allclose(weights, (0.5, 0.5))


def test_competitive_parallel_cortex_uses_only_past_performance() -> None:
    columns = np.asarray(((0.9, 0.9), (0.1, 0.1)), dtype=np.float64)

    pooled, weights = pool_probabilities(
        columns, mode="competitive", learning_rate=1.0
    )

    assert pooled[0] == pytest.approx(0.5)
    assert pooled[1] == pytest.approx(0.82)
    assert weights[0] == pytest.approx(81.0 / 82.0)


def test_parallel_cortex_rejects_unknown_pooling_mode() -> None:
    with pytest.raises(ValueError, match="uniform or competitive"):
        pool_probabilities(np.ones((2, 3)), mode="rank")


def test_sequence_sleep_enforces_physical_byte_budget() -> None:
    learner = SequenceCortexOpponentLearner(
        sequence_confirmations=1,
        max_sequence_nursery_paths=1_000,
    )
    stream = np.frombuffer(
        (b"the red ball rolled away. the blue kite flew away. " * 20),
        dtype=np.uint8,
    )
    learner.observe_chunk_probabilities(stream, learn=True)
    before = learner.embodied_bytes
    target = before - 1_000

    report = learner.consolidate_to_byte_budget(target)

    assert learner.embodied_bytes <= target
    assert report["body_bytes_before_sleep"] == before
    assert report["body_bytes_after_sleep"] == learner.embodied_bytes
    assert report["pruned_mature_paths"] > 0
    assert learner.sequence_nursery_count == 0


def test_sequence_sleep_rejects_nonpositive_budget() -> None:
    learner = SequenceCortexOpponentLearner()
    with pytest.raises(ValueError, match="positive"):
        learner.consolidate_to_byte_budget(0)
