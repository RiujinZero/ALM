from __future__ import annotations

import hashlib

import numpy as np
import pytest

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    EpisodeByteTrace,
    MatchedGoalByteHead,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    _condition_margin,
    _context_evidence,
    _goal_target_vector,
    _slice,
    _summary,
)


def test_successor_goal_adapter_preserves_multiple_future_timescales() -> None:
    payload = (
        b"\n\nUSER: clarify the durable objective"
        b"\n\nASSISTANT: refine the plan around that objective"
        b"\n\nUSER: verify the resulting outcome"
    )
    dimensions = 24
    gammas = (0.25, 0.75, 0.95)
    vector = _goal_target_vector(
        payload,
        "successor",
        dimensions=dimensions,
        successor_gammas=gammas,
    )

    assert vector.shape == (dimensions * len(gammas),)
    assert np.all(np.isfinite(vector))
    assert all(
        np.isclose(np.linalg.norm(chunk), 1.0)
        for chunk in np.split(vector, len(gammas))
    )


def test_successor_goal_adapter_accepts_a_single_observed_future_event() -> None:
    vector = _goal_target_vector(
        b"\n\nASSISTANT: one terminal observed successor",
        "successor",
        dimensions=24,
        successor_gammas=(0.25, 0.75, 0.95),
    )

    chunks = np.split(vector, 3)
    assert all(np.isclose(np.linalg.norm(chunk), 1.0) for chunk in chunks)
    np.testing.assert_allclose(chunks[0], chunks[1])
    np.testing.assert_allclose(chunks[1], chunks[2])
    with pytest.raises(ValueError, match="at least one event"):
        _goal_target_vector(
            b"no role marker",
            "successor",
            dimensions=24,
            successor_gammas=(0.25, 0.75, 0.95),
        )


def _counterbalanced_head() -> MatchedGoalByteHead:
    head = MatchedGoalByteHead(
        2,
        context_order=4,
        context_bucket_count=4093,
        local_prior_mass=1.0,
        goal_backoff_mass=0.5,
    )
    episodes = []
    for _ in range(64):
        episodes.extend(
            (
                EpisodeByteTrace(b"A", (1.0, 0.0), prefix=b"tail"),
                EpisodeByteTrace(b"B", (0.0, 1.0), prefix=b"tail"),
            )
        )
    head.fit_episodes(episodes)
    return head


def _pair() -> CrossoverPair:
    return CrossoverPair(
        pair_id="pair-0",
        context_a=b"first long context::tail",
        context_b=b"other long context::tail",
        response_a=b"A",
        response_b=b"B",
        evidence_a=np.asarray([1.0, 0.0]),
        evidence_b=np.asarray([0.0, 1.0]),
        source_label="synthetic",
    )


def _local_base(
    head: MatchedGoalByteHead, pair: CrossoverPair
) -> dict[str, np.ndarray]:
    prior = np.asarray([0.5, 0.5])
    base = {}
    for context_name, context in (("a", pair.context_a), ("b", pair.context_b)):
        for response_name, response in (("a", pair.response_a), ("b", pair.response_b)):
            local, _ = head.trace_probabilities(context, response, prior)
            base[context_name + response_name] = local
    return base


def _tilt_observed_bytes(
    trace: np.ndarray, candidate: bytes, *, log_weight: float
) -> np.ndarray:
    """Apply a small valid base-only bias to the teacher-forced target bytes."""
    tilted = trace.copy()
    for index, target in enumerate(candidate):
        weights = tilted[index].copy()
        weights[target] *= np.exp(log_weight)
        tilted[index] = weights / weights.sum()
    return tilted


def test_matched_local_q0_is_bit_identical_and_goal_off_cancels_exactly() -> None:
    head = _counterbalanced_head()
    pair = _pair()
    base = _local_base(head, pair)

    for response in (pair.response_a, pair.response_b):
        local_a, _ = head.trace_probabilities(pair.context_a, response, (1.0, 0.0))
        local_b, _ = head.trace_probabilities(pair.context_b, response, (0.0, 1.0))
        assert np.array_equal(local_a, local_b)

    first = _condition_margin(
        pair,
        head,
        base,
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        beta=0.0,
        log_ratio_clip=2.0,
    )
    swapped = _condition_margin(
        pair,
        head,
        base,
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
        beta=0.0,
        log_ratio_clip=2.0,
    )

    # Beta zero is the goal-off control: code identity is irrelevant and the
    # paired subtraction is deliberately ordered to produce an exact zero.
    assert first == 0.0
    assert swapped == 0.0


def test_pairwise_difference_in_differences_removes_base_residual() -> None:
    head = _counterbalanced_head()
    pair = _pair()
    code_a = np.asarray([1.0, 0.0])
    code_b = np.asarray([0.0, 1.0])
    clean_base = _local_base(head, pair)
    biased_base = {name: trace.copy() for name, trace in clean_base.items()}
    biased_base["aa"] = _tilt_observed_bytes(
        biased_base["aa"], pair.response_a, log_weight=1e-3
    )

    clean_goal_off = _condition_margin(
        pair,
        head,
        clean_base,
        code_a,
        code_b,
        beta=0.0,
        log_ratio_clip=2.0,
    )
    clean_goal_on = _condition_margin(
        pair,
        head,
        clean_base,
        code_a,
        code_b,
        beta=1.0,
        log_ratio_clip=2.0,
    )
    raw_goal_off = _condition_margin(
        pair,
        head,
        biased_base,
        code_a,
        code_b,
        beta=0.0,
        log_ratio_clip=2.0,
    )
    raw_goal_on = _condition_margin(
        pair,
        head,
        biased_base,
        code_a,
        code_b,
        beta=1.0,
        log_ratio_clip=2.0,
    )
    adjusted_goal_effect = raw_goal_on - raw_goal_off

    assert clean_goal_off == 0.0
    assert 0.0 < raw_goal_off < 0.01
    assert clean_goal_on > 1.0
    assert adjusted_goal_effect > 1.0
    # The small context-dependent base preference appears in both raw margins.
    # Pairwise subtraction removes it while retaining the goal preference.
    assert np.isclose(adjusted_goal_effect, clean_goal_on, rtol=2e-4, atol=1e-8)


def test_true_and_swapped_codes_reverse_the_adjusted_pairing_effect() -> None:
    head = _counterbalanced_head()
    pair = _pair()
    base = _local_base(head, pair)
    code_a = np.asarray([1.0, 0.0])
    code_b = np.asarray([0.0, 1.0])

    true_margin = _condition_margin(
        pair,
        head,
        base,
        code_a,
        code_b,
        beta=1.0,
        log_ratio_clip=2.0,
    )
    swapped_margin = _condition_margin(
        pair,
        head,
        base,
        code_b,
        code_a,
        beta=1.0,
        log_ratio_clip=2.0,
    )

    assert true_margin > 0.0
    assert swapped_margin < 0.0
    assert np.isclose(true_margin, -swapped_margin)


def test_context_evidence_excludes_the_synthetic_bridge() -> None:
    original = (
        b"USER: establish the objective\n\n"
        b"ASSISTANT: objective recorded\n\n"
        b"USER: choose the next action"
    )
    metadata = {
        "target_user_content_end": len(original),
        "unanswered_prefix_bytes": len(original),
    }
    first = _context_evidence(
        original + b"\n\nASSISTANT: interruption one",
        metadata,
        dimensions=24,
        decays=(0.5, 0.9),
    )
    second = _context_evidence(
        original + b"\n\nUSER: entirely different bridge bytes",
        metadata,
        dimensions=24,
        decays=(0.5, 0.9),
    )

    assert np.array_equal(first, second)


def test_sidecar_slice_checks_range_and_digest_and_summary_is_seeded() -> None:
    blob = b"prefix|payload|suffix"
    payload = b"payload"
    record = {
        "start": blob.index(payload),
        "end": blob.index(payload) + len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert _slice(blob, record) == payload

    with pytest.raises(ValueError, match="checksum"):
        _slice(blob, {**record, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="outside"):
        _slice(blob, {"start": -1, "end": 2})

    values = np.asarray([-0.5, 0.0, 1.5, 2.0])
    first = _summary(values, seed=17, resamples=200)
    second = _summary(values, seed=17, resamples=200)
    assert first == second
    assert first["pairs"] == 4
    assert first["positive_fraction"] == 0.5
    assert first["zero_fraction"] == 0.25
