from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_antecedent_state_ablation as antecedent,
)

AntecedentCrossoverPair = antecedent.AntecedentCrossoverPair
_pair_report = antecedent._pair_report
aggregate_pair_reports = antecedent.aggregate_pair_reports
beta_selection_rank = antecedent.beta_selection_rank
dual_greedy_trace = antecedent.dual_greedy_trace
evaluate_side_condition = antecedent.evaluate_side_condition
geometric_product_distribution = antecedent.geometric_product_distribution
load_antecedent_pairs = antecedent.load_antecedent_pairs
paired_bootstrap_contrasts = antecedent.paired_bootstrap_contrasts
run_ablation = antecedent.run_ablation


class FakeCortex:
    """Deterministic recurrent cortex exposing which state was primed."""

    def __init__(self) -> None:
        self.initial_payload = b""
        self.history = bytearray()
        self.advance_log: list[int] = []
        self.prime_log: list[bytes] = []
        self.predict_calls = 0

    def reset_episode(self) -> None:
        self.initial_payload = b""
        self.history = bytearray()
        self.advance_log = []

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        assert learn is False
        payload = bytes(np.asarray(byte_ids, dtype=np.uint8))
        self.initial_payload = payload
        self.history.extend(payload)
        self.prime_log.append(payload)
        return np.ones(len(payload), dtype=np.float64)

    def predict_next_distribution(self) -> np.ndarray:
        self.predict_calls += 1
        if self.initial_payload.endswith(b"::COMMON"):
            sequence = b"xx"
        elif b"ORIGINAL-A" in self.initial_payload:
            sequence = b"ab"
        elif b"ORIGINAL-B" in self.initial_payload:
            sequence = b"cd"
        else:
            sequence = b"??"
        position = min(len(self.advance_log), len(sequence) - 1)
        winner = sequence[position]
        distribution = np.full(256, 0.1 / 255.0, dtype=np.float64)
        distribution[winner] = 0.9
        return distribution

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        assert learn is False
        self.advance_log.append(int(byte_id))
        self.history.append(int(byte_id))


def _range(blob: bytes, start: int, end: int) -> dict[str, int | str]:
    payload = blob[start:end]
    return {
        "start": start,
        "end": end,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_schema_v4_pair(dataset_dir: Path, split: str = "dev") -> None:
    original_a = b"ORIGINAL-A"
    original_b = b"ORIGINAL-B"
    common = b"::COMMON"
    response_a = b"ab"
    response_b = b"cd"
    full_a = b"ab full"
    full_b = b"cd full"
    future_a = b"FUTURE-A-SENTINEL"
    future_b = b"FUTURE-B-SENTINEL"
    pieces = [
        original_a + common,
        original_b + common,
        response_a,
        response_b,
        full_a,
        full_b,
        future_a,
        future_b,
    ]
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    blob = b"".join(pieces)
    names = (
        "context_a",
        "context_b",
        "response_a",
        "response_b",
        "full_response_a",
        "full_response_b",
        "goal_target_a",
        "goal_target_b",
    )
    ranges = {
        name: _range(blob, offsets[index], offsets[index + 1])
        for index, name in enumerate(names)
    }
    audit = {
        "original_prefix_a": _range(blob, offsets[0], offsets[0] + len(original_a)),
        "original_prefix_b": _range(blob, offsets[1], offsets[1] + len(original_b)),
    }
    row = {
        "schema_version": 4,
        "pair_id": "pair-1",
        "invariants": {"valid": True},
        "blob_ranges": ranges,
        "audit_subranges": audit,
        "episodes": {
            "a": {
                "unanswered_prefix_bytes": len(original_a),
                "target_user_content_end": len(original_a),
            },
            "b": {
                "unanswered_prefix_bytes": len(original_b),
                "target_user_content_end": len(original_b),
            },
        },
        "common_suffix": {
            "bytes": len(common),
            "context_a_start": offsets[0] + len(original_a),
            "context_b_start": offsets[1] + len(original_b),
        },
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 4}), encoding="utf-8"
    )
    (dataset_dir / f"{split}_pairs.bin").write_bytes(blob)
    (dataset_dir / f"{split}_pairs.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def _pair() -> AntecedentCrossoverPair:
    return AntecedentCrossoverPair(
        pair_id="pair",
        context_a=b"ORIGINAL-A::COMMON",
        context_b=b"ORIGINAL-B::COMMON",
        original_prefix_a=b"ORIGINAL-A",
        original_prefix_b=b"ORIGINAL-B",
        response_a=b"ab",
        response_b=b"cd",
        common_suffix_bytes=len(b"::COMMON"),
    )


def _contrast_pair_report(
    *,
    pair_id: str,
    native: tuple[float, float],
    correct: tuple[float, float],
    swapped: tuple[float, float],
    static: tuple[float, float],
) -> dict[str, object]:
    def endpoint(values: tuple[float, float]) -> dict[str, object]:
        bpb, margin = values
        return {
            "mean_teacher_forced_bpb": bpb,
            "paired_raw_byte_ngram": {
                "correct_minus_crossed_margin": margin,
            },
        }

    return {
        "pair_id": pair_id,
        "condition_endpoints": {
            "native_base": endpoint(native),
            "correct_antecedent": endpoint(correct),
            "swapped_antecedent": endpoint(swapped),
            "static_query_no_feedback": endpoint(static),
        },
    }


def test_schema_v4_loader_uses_only_audited_original_prefixes(tmp_path: Path) -> None:
    _write_schema_v4_pair(tmp_path)

    pair = load_antecedent_pairs(tmp_path, "dev")[0]

    assert pair.original_prefix_a == b"ORIGINAL-A"
    assert pair.original_prefix_b == b"ORIGINAL-B"
    assert pair.context_a == b"ORIGINAL-A::COMMON"
    assert pair.response_a == b"ab"
    assert not hasattr(pair, "goal_target_a")
    assert b"FUTURE-A-SENTINEL" not in pair.original_prefix_a
    assert b"ab full" not in pair.original_prefix_a


def test_geometric_product_beta_zero_and_duplicate_state_are_exact() -> None:
    base = np.arange(1, 257, dtype=np.float64)
    base /= base.sum()
    route = base[::-1].copy()

    disconnected = geometric_product_distribution(base, route, 0.0)
    duplicate = geometric_product_distribution(base, base, 0.63)

    assert np.array_equal(disconnected, base)
    assert np.array_equal(duplicate, base)


def test_geometric_exact_paths_do_not_renormalize_near_unit_input() -> None:
    # A native predictor can be normalized within tolerance while its floating
    # sum is one ULP away from 1.  The disabled and duplicate routes must still
    # preserve the predictor bit-for-bit.
    base = np.full(256, 1.0 / 256.0, dtype=np.float64)
    base[0] += 1e-15
    route = np.roll(base, 1)

    assert float(base.sum()) != 1.0
    assert np.array_equal(geometric_product_distribution(base, route, 0.0), base)
    assert np.array_equal(geometric_product_distribution(base, base, 0.7), base)


def test_correct_swapped_disconnected_and_sham_route_expected_states() -> None:
    pair = _pair()

    correct = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="correct_antecedent",
        beta=1.0,
        max_new_bytes=2,
    )
    swapped = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="swapped_antecedent",
        beta=1.0,
        max_new_bytes=2,
    )
    native = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="native_base",
        beta=1.0,
        max_new_bytes=2,
    )
    disconnected = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="disconnected_correct_antecedent",
        beta=1.0,
        max_new_bytes=2,
    )
    sham = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="duplicate_current_state_sham",
        beta=1.0,
        max_new_bytes=2,
    )
    static = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="static_query_no_feedback",
        beta=1.0,
        max_new_bytes=2,
    )

    assert correct.greedy.observed == b"ab"
    assert swapped.greedy.observed == b"cd"
    assert native.greedy.observed == b"xx"
    assert disconnected.greedy.observed == native.greedy.observed
    assert np.array_equal(
        disconnected.greedy.combined_distributions,
        native.greedy.combined_distributions,
    )
    assert sham.greedy.observed == native.greedy.observed
    assert static.greedy.observed == b"aa"
    assert np.array_equal(
        sham.greedy.combined_distributions,
        native.greedy.combined_distributions,
    )
    assert correct.teacher_forced.target_bpb is not None
    assert native.teacher_forced.target_bpb is not None
    assert correct.teacher_forced.target_bpb < native.teacher_forced.target_bpb


def test_free_generation_advances_the_same_emitted_byte_into_both_states() -> None:
    base = FakeCortex()
    route = FakeCortex()

    trace = dual_greedy_trace(
        base,
        route,
        base_context=b"ORIGINAL-A::COMMON",
        route_prefix=b"ORIGINAL-A",
        beta=1.0,
        max_new_bytes=2,
    )

    assert trace.observed == b"ab"
    assert base.advance_log == list(trace.observed)
    assert route.advance_log == list(trace.observed)


def test_free_generation_horizon_does_not_reveal_short_reference_length() -> None:
    pair = _pair()
    result = evaluate_side_condition(
        FakeCortex(),
        FakeCortex(),
        pair,
        side="a",
        condition="correct_antecedent",
        beta=1.0,
        max_new_bytes=5,
    )

    assert result.teacher_forced.steps == len(pair.response_a)
    assert result.greedy.steps == 5
    assert len(result.greedy.observed) > len(pair.response_a)


def test_correct_and_swapped_route_prime_only_the_selected_original_prefix() -> None:
    pair = _pair()
    correct_base = FakeCortex()
    correct_route = FakeCortex()
    evaluate_side_condition(
        correct_base,
        correct_route,
        pair,
        side="a",
        condition="correct_antecedent",
        beta=0.5,
        max_new_bytes=2,
    )
    swapped_base = FakeCortex()
    swapped_route = FakeCortex()
    evaluate_side_condition(
        swapped_base,
        swapped_route,
        pair,
        side="a",
        condition="swapped_antecedent",
        beta=0.5,
        max_new_bytes=2,
    )

    assert correct_base.prime_log == [pair.context_a, pair.context_a]
    assert correct_route.prime_log == [
        pair.original_prefix_a,
        pair.original_prefix_a,
    ]
    assert swapped_base.prime_log == [pair.context_a, pair.context_a]
    assert swapped_route.prime_log == [
        pair.original_prefix_b,
        pair.original_prefix_b,
    ]
    assert pair.response_a not in correct_route.prime_log
    assert pair.response_b not in swapped_route.prime_log


def test_pair_report_enforces_disconnect_and_compute_matched_sham() -> None:
    report = _pair_report(
        FakeCortex(),
        FakeCortex(),
        _pair(),
        beta=1.0,
        max_new_bytes=2,
        minimum_probability=1e-300,
    )

    assert all(report["protocol_invariants"].values())
    assert report["provenance"]["response_or_future_bytes_used_to_prime_route"] is False
    assert (
        report["conditions"]["correct_antecedent"]["a"]["greedy"]["emitted_text"]
        == "ab"
    )
    assert (
        report["conditions"]["swapped_antecedent"]["a"]["greedy"]["emitted_text"]
        == "cd"
    )


def test_aggregate_endpoints_and_beta_rank_are_free_output_first() -> None:
    low = _pair_report(
        FakeCortex(),
        FakeCortex(),
        _pair(),
        beta=0.0,
        max_new_bytes=2,
        minimum_probability=1e-300,
        lexical_dimensions=64,
    )
    high = _pair_report(
        FakeCortex(),
        FakeCortex(),
        _pair(),
        beta=1.0,
        max_new_bytes=2,
        minimum_probability=1e-300,
        lexical_dimensions=64,
    )

    low_aggregate = aggregate_pair_reports([low])
    high_aggregate = aggregate_pair_reports([high])

    assert (
        high_aggregate["correct_antecedent"][
            "mean_teacher_forced_bpb_improvement_vs_native"
        ]
        > 0.0
    )
    assert (
        high_aggregate["correct_antecedent"]["exact_reference_prefix_rates"]["horizon"][
            "exact_fraction"
        ]
        == 1.0
    )
    assert (
        high_aggregate["correct_antecedent"][
            "paired_raw_byte_ngram_correct_minus_crossed"
        ]["mean_margin"]
        > 0.0
    )
    assert beta_selection_rank(high_aggregate, 1.0) > beta_selection_rank(
        low_aggregate, 0.0
    )


def test_paired_bootstrap_contrasts_are_deterministic_and_positive_favors_correct() -> (
    None
):
    reports = [
        _contrast_pair_report(
            pair_id="pair-b",
            native=(4.0, 0.1),
            correct=(2.0, 0.7),
            swapped=(3.0, 0.2),
            static=(2.5, 0.4),
        ),
        _contrast_pair_report(
            pair_id="pair-a",
            native=(3.0, 0.0),
            correct=(2.0, 0.4),
            swapped=(2.5, 0.1),
            static=(2.2, 0.25),
        ),
        _contrast_pair_report(
            pair_id="pair-c",
            native=(3.5, 0.0),
            correct=(2.0, 0.5),
            swapped=(2.75, 0.1),
            static=(2.35, 0.275),
        ),
    ]

    first = paired_bootstrap_contrasts(reports, seed=17, resamples=200)
    second = paired_bootstrap_contrasts(reports, seed=17, resamples=200)
    reversed_order = paired_bootstrap_contrasts(
        list(reversed(reports)), seed=17, resamples=200
    )

    assert first == second
    assert first == reversed_order
    assert (
        first["resample_indices_sha256"]
        == "197968e7b380783f2e78711fe76470d3b84702c23438274b564bbdc4bef58d56"
    )
    contrasts = first["contrasts"]
    assert contrasts["correct_vs_native"]["teacher_forced_bpb_advantage"][
        "mean"
    ] == pytest.approx(1.5)
    assert contrasts["correct_vs_native"][
        "free_paired_raw_byte_ngram_margin_advantage"
    ]["mean"] == pytest.approx(0.5)
    assert contrasts["correct_vs_swapped"]["teacher_forced_bpb_advantage"][
        "mean"
    ] == pytest.approx(0.75)
    assert contrasts["correct_vs_swapped"][
        "free_paired_raw_byte_ngram_margin_advantage"
    ]["mean"] == pytest.approx(0.4)
    assert contrasts["recurrent_correct_vs_static"]["teacher_forced_bpb_advantage"][
        "mean"
    ] == pytest.approx(0.35)
    assert contrasts["recurrent_correct_vs_static"][
        "free_paired_raw_byte_ngram_margin_advantage"
    ]["mean"] == pytest.approx(0.225)
    assert all(first["preregistered_gates"].values())


def test_paired_bootstrap_zero_effects_fail_strict_ci_gates() -> None:
    reports = [
        _contrast_pair_report(
            pair_id=f"pair-{index}",
            native=(2.0, 0.25),
            correct=(2.0, 0.25),
            swapped=(2.0, 0.25),
            static=(2.0, 0.25),
        )
        for index in range(3)
    ]

    summary = paired_bootstrap_contrasts(reports, seed=31, resamples=100)

    for contrast in summary["contrasts"].values():
        for metric in (
            "teacher_forced_bpb_advantage",
            "free_paired_raw_byte_ngram_margin_advantage",
        ):
            assert contrast[metric]["mean_bootstrap_ci95_low"] == 0.0
            assert contrast[metric]["mean_bootstrap_ci95_high"] == 0.0
            assert contrast[metric]["mean_bootstrap_ci95_above_zero"] is False
        assert contrast["both_ci95_above_zero"] is False
    assert summary["preregistered_gates"]["all_six_ci95_above_zero"] is False


def test_bootstrap_cli_defaults_are_preregistered_values() -> None:
    args = antecedent._parser().parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--checkpoint",
            "checkpoint.bin",
            "--report",
            "report.json",
        ]
    )

    assert args.bootstrap_seed == 260_916
    assert args.bootstrap_resamples == 4_000


def test_test_split_requires_explicit_authorization_before_any_file_is_opened(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        split="test",
        allow_test=False,
        beta=0.5,
        max_new_bytes=2,
        dataset_dir=tmp_path / "does-not-exist",
    )

    with pytest.raises(ValueError, match="refusing to open test split"):
        run_ablation(args)


def test_test_beta_must_be_frozen_and_is_never_selected() -> None:
    args = argparse.Namespace(split="test", beta=None, beta_values=None)
    with pytest.raises(ValueError, match="requires one frozen --beta"):
        antecedent._resolve_beta_values(args)

    args.beta = 0.5
    assert antecedent._resolve_beta_values(args) == ((0.5,), False)

    args.beta_values = [0.25, 0.5]
    with pytest.raises(ValueError, match="cannot sweep or select beta"):
        antecedent._resolve_beta_values(args)
