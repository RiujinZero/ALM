from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "audit_future_turn_successor_shuffle_native500.py"
_SPEC = importlib.util.spec_from_file_location("successor_shuffle_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


def _side(byte: int, *, greedy_only: bool = False) -> dict:
    payload = bytes((byte, byte + 1))
    row = {
        "sampled_hex": payload.hex(),
        "sampled_probability_sha256": f"{byte:064x}"[-64:],
    }
    if greedy_only:
        row.update(
            {
                "greedy_hex": "ff",
                "greedy_probability_sha256": "f" * 64,
                "sampled_text": "ignored by replay",
            }
        )
    return row


def _replay_reports() -> tuple[dict, dict]:
    current_modes = {}
    reference_modes = {}
    for mode_index, mode_name in enumerate(audit.CONTEXT_MODES):
        current_rows = {}
        reference_rows = {}
        for condition_index, condition in enumerate(audit.CONDITIONS):
            rows = []
            for pair_index in range(2):
                byte = 32 + mode_index * 16 + condition_index * 4 + pair_index
                rows.append(
                    {
                        "pair_id": f"pair-{pair_index}",
                        "partition": (
                            "selection" if pair_index == 0 else "evaluation"
                        ),
                        "sampled_margin": 0.1 * (condition_index + pair_index),
                        "mean_teacher_forced_bpb": 2.0 + condition_index,
                        "sides": {
                            "a": _side(byte),
                            "b": _side(byte + 2),
                        },
                    }
                )
            current_rows[condition] = rows
            if condition in audit.REPLAY_CONDITIONS:
                prior = copy.deepcopy(rows)
                for pair in prior:
                    for side in pair["sides"].values():
                        side.update(
                            {
                                "greedy_hex": "ff",
                                "greedy_probability_sha256": "f" * 64,
                                "sampled_text": "reference-only decoration",
                            }
                        )
                reference_rows[condition] = prior
        current_modes[mode_name] = {"available": True, "pair_rows": current_rows}
        reference_modes[mode_name] = {
            "available": True,
            "pair_rows": reference_rows,
        }
    return (
        {"evaluation": {"context_modes": current_modes}},
        {"evaluation": {"context_modes": reference_modes}},
    )


def _gate_fixture(*, effect: float = 0.2, recurrent_payload: bytes = b"ABCD"):
    native = []
    recurrent = []
    shuffled = []
    for index in range(29):
        common = {
            "pair_id": f"pair-{index:02d}",
            "partition": "evaluation",
            "mean_teacher_forced_bpb": 2.0,
        }
        native.append(
            {
                **common,
                "sampled_margin": 0.0,
                "sides": {
                    "a": {"sampled_hex": b"ABCD".hex()},
                    "b": {"sampled_hex": b"EFGH".hex()},
                },
            }
        )
        recurrent.append(
            {
                **common,
                "sampled_margin": effect,
                "sides": {
                    "a": {"sampled_hex": recurrent_payload.hex()},
                    "b": {"sampled_hex": recurrent_payload.hex()},
                },
            }
        )
        shuffled.append(
            {
                **common,
                "sampled_margin": 0.0,
                "sides": {
                    "a": {"sampled_hex": b"IJKL".hex()},
                    "b": {"sampled_hex": b"MNOP".hex()},
                },
            }
        )
    preregistration = {
        "locked_inputs": {
            "expected_fixed_partition_counts": {"evaluation": 29},
            "expected_development_pairs_after_exclusion": 29,
        },
        "locked_config": {"bootstrap_seed": 17, "bootstrap_resamples": 200},
    }
    interval = audit._mean_ci(
        [effect] * 29,
        seed=17 + audit.CONTROL_CONTRAST_BOOTSTRAP_OFFSET,
        resamples=200,
    )
    mode = {
        "intact_vs_successor_shuffled_quantized_code_equal_sides": 0,
        "intact_vs_successor_shuffled_quantized_code_distinct_fraction": 1.0,
        "pair_rows": {
            "native": native,
            "recurrent_code_coupled": recurrent,
            "successor_shuffled_code_coupled": shuffled,
        },
        "summaries": {
            "evaluation": {
                "controlled_contrasts": {
                    "recurrent_coupled_minus_successor_shuffled": {
                        "sampled_free_generation_margin_difference": interval
                    }
                },
                "arms": {
                    "native": {
                        "sampled_degeneration": audit._summarize_health(native)
                    },
                    "recurrent_code_coupled": {
                        "sampled_degeneration": audit._summarize_health(recurrent)
                    },
                },
            }
        },
    }
    result = {
        "evaluation": {
            "context_modes": {"original_unanswered_prefix": mode}
        }
    }
    return result, preregistration


@pytest.fixture(scope="module")
def real_reports() -> tuple[dict, dict, dict, dict]:
    paths = (
        "runs/shared_goal/future_turn_successor_shuffle_native500_v1_preregistration.json",
        "runs/shared_goal/future_turn_successor_shuffle_native500_v1_dev.json",
        "runs/shared_goal/future_turn_recurrent_codec_native500_v1_dev.json",
        "runs/shared_goal/future_turn_successor_shuffle_v1_latent_wsl_posthoc.json",
    )
    return tuple(
        json.loads((_ROOT / relative).read_text(encoding="utf-8"))
        for relative in paths
    )


def test_exact_replay_uses_only_locked_sampled_fields() -> None:
    result, reference = _replay_reports()

    replay = audit.compare_exact_replay(result, reference)

    assert replay["passed"]
    assert replay["compared_pair_condition_rows"] == 8
    assert replay["compared_side_condition_rows"] == 16
    assert replay["compared_exact_fields"] == 56

    reference["evaluation"]["context_modes"]["original_unanswered_prefix"][
        "pair_rows"
    ]["native"][0]["sides"]["a"]["greedy_hex"] = "00"
    assert audit.compare_exact_replay(result, reference)["passed"]


def test_exact_replay_detects_probability_and_pair_order_changes() -> None:
    result, reference = _replay_reports()
    rows = result["evaluation"]["context_modes"]["long_interruption_crossover"][
        "pair_rows"
    ]["recurrent_code_coupled"]
    rows[0]["sides"]["b"]["sampled_probability_sha256"] = "0" * 64

    changed = audit.compare_exact_replay(result, reference)

    assert not changed["passed"]
    assert changed["mismatch_count"] == 1
    assert changed["first_mismatches"][0].endswith(
        "/b/sampled_probability_sha256"
    )

    reordered, reference = _replay_reports()
    target = reordered["evaluation"]["context_modes"]["original_unanswered_prefix"][
        "pair_rows"
    ]["native"]
    target.reverse()
    mismatch = audit.compare_exact_replay(reordered, reference)
    assert not mismatch["passed"]
    assert "pair ID/order mismatch" in mismatch["first_mismatches"][0]


def test_primary_gate_is_independently_recomputed_from_pair_rows() -> None:
    result, preregistration = _gate_fixture()

    gate = audit.compute_locked_primary_gate(result, preregistration)

    assert gate["effect"]["strictly_positive_ci_passed"]
    assert gate["code_separation"]["exact_evaluation_distinct_fraction"] == 1.0
    assert gate["output_health"]["passed"]
    assert gate["reported_summaries_exactly_reproduced"]
    assert gate["passed"]


def test_primary_interval_is_strict_and_summary_tampering_is_visible() -> None:
    result, preregistration = _gate_fixture(effect=0.0)

    zero = audit.compute_locked_primary_gate(result, preregistration)

    assert zero["effect"]["independently_recomputed_interval"]["ci95_low"] == 0.0
    assert not zero["effect"]["strictly_positive_ci_passed"]
    assert not zero["passed"]

    result, preregistration = _gate_fixture()
    reported = result["evaluation"]["context_modes"][
        "original_unanswered_prefix"
    ]["summaries"]["evaluation"]["controlled_contrasts"][
        "recurrent_coupled_minus_successor_shuffled"
    ]["sampled_free_generation_margin_difference"]
    reported["mean"] += 1.0
    tampered = audit.compute_locked_primary_gate(result, preregistration)
    assert tampered["passed"]
    assert not tampered["reported_summaries_exactly_reproduced"]


def test_output_health_is_recomputed_from_generated_bytes() -> None:
    result, preregistration = _gate_fixture(recurrent_payload=b"\x00\x01\x02\x03")

    gate = audit.compute_locked_primary_gate(result, preregistration)

    assert gate["effect"]["strictly_positive_ci_passed"]
    assert gate["output_health"]["printable_fraction_loss"] == 1.0
    assert not gate["output_health"]["passed"]
    assert not gate["passed"]


def test_evaluation_code_separation_uses_conservative_partition_bounds() -> None:
    def mode(equal: int) -> dict:
        return {
            "intact_vs_successor_shuffled_quantized_code_equal_sides": equal,
            "intact_vs_successor_shuffled_quantized_code_distinct_fraction": (
                1.0 - equal / 124
            ),
        }

    exact = audit.assess_evaluation_code_separation(
        mode(0), total_sides=124, evaluation_sides=58
    )
    boundary = audit.assess_evaluation_code_separation(
        mode(2), total_sides=124, evaluation_sides=58
    )
    unresolved = audit.assess_evaluation_code_separation(
        mode(3), total_sides=124, evaluation_sides=58
    )
    failed = audit.assess_evaluation_code_separation(
        mode(70), total_sides=124, evaluation_sides=58
    )

    assert exact["exact_evaluation_distinct_fraction"] == 1.0
    assert exact["passed"]
    assert boundary["passed"]
    assert unresolved["status"] == "unresolved_from_all_side_aggregate"
    assert not unresolved["passed"]
    assert failed["status"] == "failed"


def test_completed_result_passes_integrity_with_explicit_platform_deviation(
    real_reports: tuple[dict, dict, dict, dict],
) -> None:
    preregistration, result, reference, reproduction = real_reports

    report = audit.audit_loaded_reports(
        preregistration,
        result,
        reference,
        platform_reproduction=reproduction,
        repo_root=_ROOT,
    )

    assert report["integrity_passed"]
    assert report["integrity_failures"] == []
    assert not report["strict_preregistered_shuffle_hashes_match"]
    assert len(report["explained_cross_platform_deviations"]) == 3
    for row in report["explained_cross_platform_deviations"].values():
        assert row["passed"]
        assert not row["strict_preregistered_match"]
        assert row["explained_cross_platform_deviation"]
    replay = report["exact_recurrent_reference_replay"]
    assert replay["passed"]
    assert replay["compared_exact_fields"] == 1736
    gate = report["locked_primary_result_gate"]
    assert gate["effect"]["independently_recomputed_interval"] == {
        "pairs": 29,
        "mean": 0.027962103039890517,
        "ci95_low": -0.008283921639554942,
        "ci95_high": 0.06267423081969811,
        "positive_fraction": 0.6551724137931034,
    }
    assert gate["code_separation"]["passed"]
    assert gate["output_health"]["passed"]
    assert not gate["effect"]["strictly_positive_ci_passed"]
    assert not gate["passed"]


def test_gain_zero_and_frozen_state_mutations_fail_integrity(
    real_reports: tuple[dict, dict, dict, dict],
) -> None:
    preregistration, original, reference, reproduction = real_reports
    result = copy.deepcopy(original)
    mode = result["evaluation"]["context_modes"]["original_unanswered_prefix"]
    mode["pair_rows"]["recurrent_code_gain0"][0]["sampled_margin"] += 0.01
    result["evaluation"]["writer_fingerprints_unchanged_by_evaluation"] = False

    report = audit.audit_loaded_reports(
        preregistration,
        result,
        reference,
        platform_reproduction=reproduction,
        repo_root=_ROOT,
    )

    assert not report["integrity_passed"]
    assert "evaluation/writer_fingerprints_unchanged_by_evaluation" in report[
        "integrity_failures"
    ]
    assert (
        "evaluation/original_unanswered_prefix/gain0_rows_exactly_native"
        in report["integrity_failures"]
    )


def test_unexplained_shuffle_hash_change_fails_integrity(
    real_reports: tuple[dict, dict, dict, dict],
) -> None:
    preregistration, original, reference, reproduction = real_reports
    result = copy.deepcopy(original)
    result["transitions"]["successor_shuffle_audit"][
        "permutation_sha256"
    ] = "0" * 64

    report = audit.audit_loaded_reports(
        preregistration,
        result,
        reference,
        platform_reproduction=reproduction,
        repo_root=_ROOT,
    )

    assert not report["integrity_passed"]
    assert "shuffle/permutation_sha256" in report["integrity_failures"]


def test_sidecar_and_audit_artifact_hash_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"

    audit.write_audit_artifact(artifact, {"status": "valid", "note": "a\nb"})

    assert audit.verify_sidecar(artifact)["matches"]
    assert b"\r\n" not in artifact.read_bytes()
    (_ROOT / "runs").exists()
    sidecar = artifact.with_name(artifact.name + ".sha256")
    sidecar.write_text(
        f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  wrong.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not name"):
        audit.verify_sidecar(artifact)


def test_protected_dataset_paths_are_rejected_without_opening() -> None:
    protected_payload = _ROOT / audit.PROTECTED_DATASET_DIRECTORY / "dev_pairs.bin"

    with pytest.raises(ValueError, match="protected dataset directory"):
        audit._assert_outside_protected(protected_payload, repo_root=_ROOT)


def test_locked_posix_native_path_is_absolute_cross_platform() -> None:
    assert audit._locked_path_is_absolute(audit.LOCKED_NATIVE_PATH)
    assert not audit._locked_path_is_absolute("src/module.py")
