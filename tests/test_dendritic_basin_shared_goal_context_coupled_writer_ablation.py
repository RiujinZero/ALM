from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from manifoldflow.experiments import (
    dendritic_basin_shared_goal_context_coupled_writer_ablation as assay,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    CrossoverPair,
    TrainingExample,
)


class FakeCortex:
    """Small deterministic cortex whose preferred byte follows its last input."""

    embodied_bytes = 500_000_000

    def __init__(self) -> None:
        self.history = bytearray()
        self.advance_calls: list[tuple[int, bool]] = []
        self.chunk_learn_flags: list[bool] = []

    def reset_episode(self) -> None:
        self.history.clear()

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool
    ) -> np.ndarray:
        self.chunk_learn_flags.append(learn)
        self.history.extend(int(value) for value in byte_ids)
        return np.empty((len(byte_ids), 256), dtype=np.float64)

    def predict_next_distribution(self) -> np.ndarray:
        preferred = (self.history[-1] + 1) % 256 if self.history else 0
        result = np.full(256, 0.5 / 255.0, dtype=np.float64)
        result[preferred] = 0.5
        return result

    def advance_observation(self, byte_id: int, *, learn: bool) -> None:
        self.advance_calls.append((int(byte_id), learn))
        self.history.append(int(byte_id))


def _example(index: int) -> TrainingExample:
    evidence = np.linspace(-1.0, 1.0, 16, dtype=np.float64) + index * 0.1
    target = np.asarray(
        [
            (-1.0) ** index * (index + 1.0),
            0.25 * index,
            3.0 - index,
            -0.5 - index,
        ],
        dtype=np.float64,
    )
    response = bytes((65 + index, 97 + index, 10))
    return TrainingExample(
        episode_id=f"train-{index}",
        response=response,
        prefix=b"ignored-context-prefix",
        evidence=evidence,
        response_vector=np.zeros(4, dtype=np.float64),
        goal_target_vector=target,
    )


def _mechanism() -> assay.ContextCoupledMechanism:
    return assay.fit_context_coupled_mechanism(
        [_example(index) for index in range(4)],
        feature_dimensions=8,
        ridge_alpha=1.0,
        state_feature_count=2,
        table_bits=5,
        learning_rate=0.55,
        local_orders=(1, 2),
        training_epochs=1,
        control_seed=17,
    )


def _pair(index: int = 0) -> CrossoverPair:
    return CrossoverPair(
        pair_id=f"pair-{index}",
        context_a=(
            b"USER: alpha\n\nASSISTANT: hold\n\nUSER: common resume\n\nASSISTANT:"
        ),
        context_b=(
            b"USER: bravo\n\nASSISTANT: hold\n\nUSER: common resume\n\nASSISTANT:"
        ),
        response_a=b"Alpha\n",
        response_b=b"Bravo\n",
        evidence_a=np.linspace(-1.0, 1.0, 16, dtype=np.float64),
        evidence_b=np.linspace(1.0, -1.0, 16, dtype=np.float64),
        source_label="fixture",
        goal_target_a=b"unused",
        goal_target_b=b"unused",
        goal_target_kind="successor",
        goal_target_vector_a=np.asarray([3.0, 1.0, -2.0, 0.5]),
        goal_target_vector_b=np.asarray([-3.0, 0.5, 2.0, -1.0]),
    )


def _locked_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "split": "dev",
        "feature_dimensions": assay.LOCKED_FEATURE_DIMENSIONS,
        "state_decays": assay.LOCKED_STATE_DECAYS,
        "successor_gammas": assay.LOCKED_SUCCESSOR_GAMMAS,
        "ridge_alpha": assay.LOCKED_RIDGE_ALPHA,
        "state_feature_count": assay.LOCKED_STATE_FEATURE_COUNT,
        "table_bits": assay.LOCKED_TABLE_BITS,
        "learning_rate": assay.LOCKED_LEARNING_RATE,
        "local_orders": assay.LOCKED_LOCAL_ORDERS,
        "training_epochs": assay.LOCKED_TRAINING_EPOCHS,
        "log_ratio_clip": assay.LOCKED_LOG_RATIO_CLIP,
        "writer_temperature": assay.LOCKED_WRITER_TEMPERATURE,
        "generation_horizon": assay.LOCKED_GENERATION_HORIZON,
        "reference_dimensions": assay.LOCKED_REFERENCE_DIMENSIONS,
        "control_seed": assay.LOCKED_CONTROL_SEED,
        "bootstrap_seed": assay.LOCKED_BOOTSTRAP_SEED,
        "bootstrap_resamples": assay.LOCKED_BOOTSTRAP_RESAMPLES,
        "max_pairs": 0,
        "claim_limit_bytes": assay.DEFAULT_CLAIM_LIMIT_BYTES,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_top_signed_ids_are_stable_under_ties_and_encode_sign() -> None:
    vector = np.asarray([-4.0, 4.0, -2.0, 2.0, -0.0])

    assert assay.top_signed_feature_ids(vector, 4) == (1, 2, 5, 6)
    assert assay.top_signed_feature_ids(vector, 5)[-1] == 8
    with pytest.raises(ValueError, match="finite"):
        assay.top_signed_feature_ids(np.asarray([1.0, np.nan]), 1)
    with pytest.raises(ValueError, match="vector width"):
        assay.top_signed_feature_ids(vector, 6)


def test_sattolo_has_no_self_assignments_and_is_seed_deterministic() -> None:
    first = assay.sattolo_indices(17, 92)
    second = assay.sattolo_indices(17, 92)

    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(17))
    assert not np.any(first == np.arange(17))
    with pytest.raises(ValueError, match="at least two"):
        assay.sattolo_indices(1, 0)


def test_fit_uses_matched_allocation_counts_and_no_self_training_control() -> None:
    mechanism = _mechanism()
    writers = (
        *mechanism.writers.values(),
        mechanism.shuffled_training_writer,
        *mechanism.antecedent_writers.values(),
        mechanism.antecedent_shuffled_training_writer,
    )

    assert {writer.allocated_bytes for writer in writers} == {32 * 256 * 4}
    assert {writer.episodes_seen for writer in writers} == {4}
    assert {writer.symbols_seen for writer in writers} == {12}
    assert not np.any(mechanism.shuffled_training_permutation == np.arange(4))
    assert not np.any(
        mechanism.antecedent_shuffled_training_permutation == np.arange(4)
    )
    assert mechanism.writers["additive_wide"].raw_active_feature_count(
        2
    ) == mechanism.writers["coupled"].raw_active_feature_count(2)
    # The supplied TrainingExample.prefix is intentionally not consumed.
    expected = assay._responses_digest([_example(i) for i in range(4)], 1)
    assert mechanism.training_response_sha256 == expected


def test_pair_state_prediction_has_strict_no_self_episode_shuffle() -> None:
    mechanism = _mechanism()
    rows, permutation, arrays = assay.predict_pair_states(
        [_pair()],
        mechanism,
        feature_dimensions=8,
        state_decays=(0.5,),
        control_seed=33,
    )

    assert len(rows) == 1
    np.testing.assert_array_equal(permutation, np.asarray([1, 0]))
    np.testing.assert_array_equal(rows[0].shuffled_causal_a, rows[0].causal_full_b)
    np.testing.assert_array_equal(rows[0].shuffled_causal_b, rows[0].causal_full_a)
    np.testing.assert_array_equal(
        rows[0].antecedent_online_pending_a,
        rows[0].antecedent_online_pending_b,
    )
    assert not np.array_equal(
        rows[0].antecedent_reinstated_a, rows[0].antecedent_online_a
    )
    assert arrays["target"].shape == (2, 4)


def test_condition_map_includes_oracle_causal_and_all_lesions() -> None:
    mechanism = _mechanism()
    states, _, _ = assay.predict_pair_states(
        [_pair()],
        mechanism,
        feature_dimensions=8,
        state_decays=(0.5,),
        control_seed=33,
    )
    assignments = assay.condition_assignments(mechanism, states[0], beta=0.75)

    assert tuple(assignments) == assay.CONDITION_NAMES
    assert assignments["native"].beta == 0.0
    assert assignments["disconnected_beta_zero"].beta == 0.0
    assert assignments["coupled_causal_full"].beta == 0.75
    assert assignments["coupled_frozen_joint_causal_full"].freeze_joint
    assert (
        assignments["coupled_paired_side_swapped"].state_a
        == assignments["coupled_causal_full"].state_b
    )
    assert (
        assignments["coupled_pooled_prior"].state_a
        == assignments["coupled_pooled_prior"].state_b
    )
    assert (
        assignments["shuffled_training_coupled_oracle"].writer
        is mechanism.shuffled_training_writer
    )
    assert (
        assignments["antecedent_reinstated_additive_wide"].writer
        is mechanism.antecedent_writers["additive_wide"]
    )
    assert (
        assignments["antecedent_reinstated_paired_side_swapped"].state_a
        == assignments["antecedent_reinstated_coupled"].state_b
    )
    assert (
        assignments["antecedent_online_paired_side_swapped"].state_a
        == assignments["antecedent_online_causal"].state_b
    )
    assert (
        assignments["antecedent_online_additive_wide"].writer
        is mechanism.antecedent_writers["additive_wide"]
    )
    assert (
        assignments["antecedent_online_shuffled_training"].writer
        is mechanism.antecedent_shuffled_training_writer
    )
    assert assignments["antecedent_reinstated_frozen_joint"].freeze_joint
    assert assignments["antecedent_online_frozen_joint"].freeze_joint


def test_beta_zero_bypasses_invalid_route_and_greedy_feedback_is_frozen() -> None:
    base = np.tile(np.arange(1, 257, dtype=np.float64), (2, 1))
    base /= base.sum(axis=1, keepdims=True)
    bypass = assay.corrected_probability_trace(
        base,
        np.full((1, 3), np.nan),
        np.full((7, 9), np.nan),
        beta=0.0,
        log_ratio_clip=2.0,
    )
    np.testing.assert_array_equal(bypass, base)

    mechanism = _mechanism()
    cortex = FakeCortex()
    before = {name: writer.fingerprint() for name, writer in mechanism.writers.items()}
    trace = assay.greedy_route_trace(
        cortex,
        b"\x05",
        mechanism.writers["local"],
        mechanism.writers["coupled"],
        (1, 2),
        beta=0.0,
        horizon=4,
        log_ratio_clip=2.0,
    )

    assert trace.observed == bytes((6, 7, 8, 9))
    assert cortex.advance_calls[-4:] == [(6, False), (7, False), (8, False), (9, False)]
    assert cortex.chunk_learn_flags == [False]
    assert before == {
        name: writer.fingerprint() for name, writer in mechanism.writers.items()
    }

    routed = assay.greedy_route_trace(
        FakeCortex(),
        b"\x05",
        mechanism.writers["local"],
        mechanism.writers["coupled"],
        (1, 2),
        beta=0.5,
        horizon=4,
        log_ratio_clip=2.0,
    )
    assert len(routed.observed) == 4
    assert routed.distributions.shape == (4, 256)
    np.testing.assert_allclose(routed.distributions.sum(axis=1), 1.0)
    assert before == {
        name: writer.fingerprint() for name, writer in mechanism.writers.items()
    }


def test_full_pair_beta_zero_is_exact_for_every_condition() -> None:
    mechanism = _mechanism()
    states, _, _ = assay.predict_pair_states(
        [_pair()],
        mechanism,
        feature_dimensions=8,
        state_decays=(0.5,),
        control_seed=33,
    )

    report = assay.evaluate_pair(
        _pair(),
        states[0],
        mechanism,
        FakeCortex(),
        beta=0.0,
        horizon=5,
        log_ratio_clip=2.0,
        writer_temperature=1.0,
        reference_dimensions=16,
    )

    assert tuple(report["condition_endpoints"]) == assay.CONDITION_NAMES
    assert report["invariants"] == {
        "disconnected_teacher_trace_exactly_native": True,
        "disconnected_greedy_trace_exactly_native": True,
        "global_beta_zero_all_conditions_exactly_native": True,
    }
    native = report["condition_endpoints"]["native"]
    assert native["exact_reference_prefix"]["8"] == [None, None]
    assert isinstance(
        native["paired_raw_byte_ngram"]["correct_minus_crossed_margin"], float
    )


def test_trace_prefixes_are_null_when_reference_or_generation_is_short() -> None:
    probabilities = np.full((20, 256), 1.0 / 256.0)
    trace = assay.ProductionTrace(b"a" * 20, probabilities, None, "greedy")

    report = assay.trace_report(trace, b"a" * 10)

    assert report["exact_reference_prefix"] == {
        "8": True,
        "16": None,
        "32": None,
        "96": None,
        "horizon": None,
    }
    assert report["common_prefix_bytes"] == 10
    assert report["common_prefix_fraction"] is None


def test_memory_report_discloses_diagnostic_and_candidate_budget_breach() -> None:
    mechanism = _mechanism()

    report = assay.memory_report(mechanism, FakeCortex(), claim_limit_bytes=500_000_000)

    assert report["native_base_embodied_bytes"] == 500_000_000
    assert report["successor_candidate_exceeds_claim_limit"] is True
    assert report["antecedent_candidate_exceeds_claim_limit"] is True
    assert report["diagnostic_exceeds_claim_limit"] is True
    assert report["all_nine_writer_tables_bytes_exact"] == 9 * 32 * 256 * 4
    assert report["diagnostic_permutation_arrays_bytes_exact"] == 8 * 8


def test_test_firewall_rejects_before_any_loader_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("a corpus loader ran before the test firewall")

    monkeypatch.setattr(assay, "load_training_examples", forbidden)
    monkeypatch.setattr(assay, "load_crossover_pairs", forbidden)
    args = SimpleNamespace(
        split="test",
        allow_test=False,
        beta_values=(0.5,),
    )
    with pytest.raises(ValueError, match="without --allow-test"):
        assay.run_ablation(args)

    args.allow_test = True
    args.beta_values = (0.25, 0.5)
    with pytest.raises(ValueError, match="exactly one fixed beta"):
        assay.run_ablation(args)

    args.beta_values = (0.5,)
    args.max_pairs = 1
    with pytest.raises(ValueError, match="forbids --max-pairs"):
        assay.run_ablation(args)

    args.max_pairs = 0
    with pytest.raises(ValueError, match="test access is disabled"):
        assay.run_ablation(args)


def test_test_freeze_report_never_reads_test_gates_or_selects() -> None:
    class ExplodingRuns(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            raise AssertionError(f"test result was inspected for selection: {key}")

    decision = assay.beta_freeze_decision("test", (0.5,), ExplodingRuns())

    assert decision == {
        "mode": "fixed_beta_test_no_within_run_selection",
        "supplied_beta_preselected": 0.5,
        "selection_performed": False,
        "fixed_beta_candidate_for_separate_test_run": None,
        "test_outcomes_not_used_for_beta_selection": True,
        "successor_family_excluded_from_decision": True,
        "external_preregistered_beta_lock_required_for_confirmation": True,
    }

    mechanism = _mechanism()
    pair = _pair()
    states, _, _ = assay.predict_pair_states(
        [pair],
        mechanism,
        feature_dimensions=8,
        state_decays=(0.5,),
        control_seed=33,
    )
    fixed_beta_report = assay.evaluate_beta(
        [pair],
        states,
        mechanism,
        FakeCortex(),
        beta=0.5,
        horizon=1,
        log_ratio_clip=2.0,
        writer_temperature=1.0,
        reference_dimensions=16,
        bootstrap_seed=5,
        bootstrap_resamples=100,
        include_development_gate=False,
    )
    assert "antecedent_preoutput_gate" not in fixed_beta_report


def test_ineligible_development_run_cannot_inspect_gates_or_select() -> None:
    class ExplodingRuns(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            raise AssertionError(f"ineligible result was inspected: {key}")

    decision = assay.beta_freeze_decision(
        "dev",
        (0.5,),
        ExplodingRuns(),
        selection_eligible=False,
        ineligible_reasons=("smoke pair slice",),
    )

    assert decision["selection_performed"] is False
    assert decision["beta_outcomes_not_inspected_for_selection"] is True
    assert decision["test_run_permitted_by_this_development_result"] is False
    assert decision["ineligible_reasons"] == ["smoke pair slice"]


def test_protocol_eligibility_requires_full_locked_config_and_provenance() -> None:
    args = _locked_args()
    eligible = assay.protocol_eligibility_report(
        args,
        assay.DEFAULT_BETA_VALUES,
        loaded_pair_count=assay.LOCKED_DEV_PAIRS,
        evaluated_pair_count=assay.LOCKED_DEV_PAIRS,
        dataset_manifest_sha256=assay.LOCKED_DATASET_MANIFEST_SHA256,
        checkpoint_sha256=assay.LOCKED_CHECKPOINT_SHA256,
    )
    assert eligible["eligible"] is True
    assert eligible["deviations"] == []

    smoke = assay.protocol_eligibility_report(
        _locked_args(max_pairs=4),
        (0.5,),
        loaded_pair_count=assay.LOCKED_DEV_PAIRS,
        evaluated_pair_count=4,
        dataset_manifest_sha256=assay.LOCKED_DATASET_MANIFEST_SHA256,
        checkpoint_sha256=assay.LOCKED_CHECKPOINT_SHA256,
    )
    assert smoke["eligible"] is False
    assert any("max_pairs" in reason for reason in smoke["deviations"])
    assert any("beta_values" in reason for reason in smoke["deviations"])
    assert any("evaluated_pair_count" in reason for reason in smoke["deviations"])


def test_external_test_lock_is_bound_to_source_locked_development_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assay, "LOCKED_DEVELOPMENT_REPORT_SHA256", "a" * 64)
    lock = tmp_path / "lock.json"
    lock.write_text(
        "{\n"
        '  "status": "locked_before_test_access",\n'
        '  "selected_beta": 0.5,\n'
        f'  "dataset_manifest_sha256": "{assay.LOCKED_DATASET_MANIFEST_SHA256}",\n'
        f'  "checkpoint_sha256": "{assay.LOCKED_CHECKPOINT_SHA256}",\n'
        f'  "development_report_sha256": "{"a" * 64}"\n'
        "}\n",
        encoding="utf-8",
    )

    report = assay._validate_external_test_lock(
        lock,
        beta=0.5,
        dataset_manifest_sha256=assay.LOCKED_DATASET_MANIFEST_SHA256,
        checkpoint_sha256=assay.LOCKED_CHECKPOINT_SHA256,
    )

    assert report["validated_before_test_pair_access"] is True
    assert report["development_report_sha256"] == "a" * 64


def test_requested_split_payloads_are_verified_against_manifest(
    tmp_path: Path,
) -> None:
    records = []
    for name, payload in (
        ("train.bin", b"train"),
        ("train.index.jsonl", b"{}\n"),
        ("dev_pairs.bin", b"dev"),
        ("dev_pairs.jsonl", b"{}\n"),
        ("test_pairs.bin", b"untouched-test"),
        ("test_pairs.jsonl", b"{}\n"),
    ):
        (tmp_path / name).write_bytes(payload)
        records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"outputs": records}), encoding="utf-8")

    verified = assay._verify_manifest_payloads(tmp_path, manifest, "dev")

    assert set(verified) == {
        "train.bin",
        "train.index.jsonl",
        "dev_pairs.bin",
        "dev_pairs.jsonl",
    }
    (tmp_path / "dev_pairs.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count differs"):
        assay._verify_manifest_payloads(tmp_path, manifest, "dev")


def test_antecedent_gate_requires_every_capacity_and_causal_control() -> None:
    names = (
        "antecedent_reinstated_coupled_vs_additive_wide",
        "antecedent_reinstated_coupled_vs_paired_side_swap",
        "antecedent_reinstated_coupled_vs_frozen_joint",
        "antecedent_online_causal_vs_additive_wide",
        "antecedent_online_causal_vs_native",
        "antecedent_online_causal_vs_pending_only",
        "antecedent_online_causal_vs_shuffled_episode",
        "antecedent_online_causal_vs_shuffled_training_mapping",
        "antecedent_online_causal_vs_paired_side_swap",
        "antecedent_online_causal_vs_frozen_joint",
    )

    def row(passes: bool = True) -> dict[str, object]:
        endpoint = {"mean_bootstrap_ci95_above_zero": passes}
        return {
            "teacher_forced_bpb_advantage": dict(endpoint),
            "free_ngram_margin_advantage": dict(endpoint),
        }

    bootstrap = {"contrasts": {name: row() for name in names}}
    positive = assay.antecedent_preoutput_gate(bootstrap)
    assert positive["full_gate_passed"] is True

    bootstrap["contrasts"]["antecedent_online_causal_vs_additive_wide"] = row(False)
    negative = assay.antecedent_preoutput_gate(bootstrap)
    assert negative["full_gate_passed"] is False
    assert negative["reinstated_ceiling_passed"] is True
    assert negative["online_causal_transfer_passed"] is False


def test_pair_bootstrap_keeps_pair_as_unit_and_is_deterministic() -> None:
    mechanism = _mechanism()
    pair = _pair()
    states, _, _ = assay.predict_pair_states(
        [pair],
        mechanism,
        feature_dimensions=8,
        state_decays=(0.5,),
        control_seed=33,
    )
    report = assay.evaluate_pair(
        pair,
        states[0],
        mechanism,
        FakeCortex(),
        beta=0.0,
        horizon=3,
        log_ratio_clip=2.0,
        writer_temperature=1.0,
        reference_dimensions=16,
    )

    first = assay.paired_bootstrap_contrasts([report], seed=7, resamples=100)
    second = assay.paired_bootstrap_contrasts([report], seed=7, resamples=100)

    assert first == second
    assert first["experimental_unit"].startswith("one crossover pair")
    assert all(
        contrast["teacher_forced_bpb_advantage"]["pairs"] == 1
        for contrast in first["contrasts"].values()
    )
