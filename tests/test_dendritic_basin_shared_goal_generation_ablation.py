from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    MatchedGoalByteHead,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    GeneratedBranch,
    _generate_branch,
    _generation_record,
    _paired_reference_discrimination,
    _sample_with_uniform,
    _sattolo_indices,
)


class _FakeFrozenCortex:
    story_boundary = b"<END>"

    def __init__(self) -> None:
        self.reset_calls = 0
        self.learn_flags: list[bool] = []
        self.state = bytearray()

    def reset_episode(self) -> None:
        self.reset_calls += 1
        self.state.clear()

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.learn_flags.append(learn)
        self.state.extend(byte_ids.tolist())
        return np.zeros(len(byte_ids), dtype=np.float64)

    def predict_next_distribution(self) -> np.ndarray:
        probability = np.arange(1, 257, dtype=np.float64)
        if self.state and self.state[-1] % 2:
            probability = probability[::-1].copy()
        return probability / probability.sum()

    def advance_observation(self, byte_id: int, *, learn: bool) -> float:
        self.learn_flags.append(learn)
        self.state.append(byte_id)
        return 0.0


def _head() -> MatchedGoalByteHead:
    head = MatchedGoalByteHead(2, context_order=2, context_bucket_count=127)
    for _ in range(32):
        head.fit_episode(b"A", (1.0, 0.0), prefix=b"p>")
        head.fit_episode(b"B", (0.0, 1.0), prefix=b"p>")
    return head


def _branch(
    cortex: _FakeFrozenCortex,
    head: MatchedGoalByteHead | None,
    code: np.ndarray,
    *,
    beta: float,
    draws: np.ndarray,
    top_k: int = 0,
) -> GeneratedBranch:
    return _generate_branch(
        cortex,
        b"p>",
        draws,
        head=head,
        goal_code=code,
        beta=beta,
        log_ratio_clip=4.0,
        temperature=1.0,
        top_k=top_k,
        top_p=1.0,
        stop_at_story_boundary=False,
    )


def test_beta_zero_exactly_reproduces_unmodified_native_generation() -> None:
    cortex = _FakeFrozenCortex()
    head = _head()
    draws = np.random.default_rng(7).random(12)

    native = _branch(cortex, None, np.array([0.5, 0.5]), beta=0.0, draws=draws)
    goal_off = _branch(cortex, head, np.array([0.5, 0.5]), beta=0.0, draws=draws)

    assert native.data == goal_off.data
    assert goal_off.beta_zero_probabilities_exact is True
    assert cortex.reset_calls == 2
    assert cortex.learn_flags and not any(cortex.learn_flags)


def test_clamped_goal_can_change_generation_under_the_same_draw() -> None:
    cortex = _FakeFrozenCortex()
    head = _head()
    draws = np.array([0.25])

    goal_a = _branch(
        cortex,
        head,
        np.array([1.0, 0.0]),
        beta=8.0,
        draws=draws,
        top_k=2,
    )
    goal_b = _branch(
        cortex,
        head,
        np.array([0.0, 1.0]),
        beta=8.0,
        draws=draws,
        top_k=2,
    )

    assert goal_a.data == b"A"
    assert goal_b.data == b"B"


def test_paired_reference_metric_rewards_correct_and_rejects_swapped_pairing() -> None:
    generated_a = b"red apples grow in this orchard"
    generated_b = b"blue boats cross the open harbor"
    reference_a = b"the orchard grows red apples"
    reference_b = b"blue boats sail through the harbor"

    correct = _paired_reference_discrimination(
        generated_a,
        generated_b,
        reference_a,
        reference_b,
        dimensions=1024,
    )
    swapped = _paired_reference_discrimination(
        generated_b,
        generated_a,
        reference_a,
        reference_b,
        dimensions=1024,
    )

    assert correct["correct_pairing_wins"] is True
    assert correct["correct_minus_crossed_margin"] > 0.0
    assert np.isclose(
        correct["correct_minus_crossed_margin"],
        -swapped["correct_minus_crossed_margin"],
    )


def test_generation_record_reports_text_utf8_repetition_and_diversity() -> None:
    result = GeneratedBranch(b"aaaa\xff", None)

    record = _generation_record(
        b"prompt>", result, boundary=b"<END>", prompt_tail_bytes=4
    )

    assert record["generated_text"] == "aaaa\\xff"
    assert record["prompt_tail_text"] == "mpt>"
    assert record["diagnostics"]["valid_utf8"] is False
    assert record["diagnostics"]["longest_identical_byte_run"] == 4
    assert "4" in record["diagnostics"]["distinct_ngram_fraction"]


def test_sampler_and_shuffled_control_are_deterministic() -> None:
    distribution = np.zeros(256)
    distribution[3] = 0.25
    distribution[7] = 0.75

    assert (
        _sample_with_uniform(distribution, 0.1, temperature=1.0, top_k=0, top_p=1.0)
        == 3
    )
    assert (
        _sample_with_uniform(distribution, 0.9, temperature=1.0, top_k=0, top_p=1.0)
        == 7
    )
    first = _sattolo_indices(12, seed=9)
    second = _sattolo_indices(12, seed=9)
    assert np.array_equal(first, second)
    assert np.all(first != np.arange(12))
