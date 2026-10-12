from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_future_turn_recurrent_codec_native500.py"
)
_SPEC = importlib.util.spec_from_file_location("recurrent_codec_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


def _interval(low: float) -> dict[str, float | int]:
    return {
        "pairs": 29,
        "mean": low + 0.1,
        "ci95_low": low,
        "ci95_high": low + 0.2,
        "positive_fraction": 0.8,
    }


def _gate_inputs() -> tuple[dict, dict]:
    native_health = {
        "mean_printable_fraction": 0.95,
        "valid_utf8_fraction": 0.98,
        "maximum_identical_byte_run": 8,
        "outputs_with_run_at_least_16": 0,
    }
    recurrent_health = {
        "mean_printable_fraction": 0.92,
        "valid_utf8_fraction": 0.96,
        "maximum_identical_byte_run": 10,
        "outputs_with_run_at_least_16": 1,
    }
    result = {
        "evaluation": {
            "context_modes": {
                "original_unanswered_prefix": {
                    "code_control_collision_audit": {
                        "recurrent_full_history": {
                            "paired_swap_pairs": 62,
                            "paired_swap_distinct_code_fraction": 1.0,
                        }
                    },
                    "summaries": {
                        "evaluation": {
                            "controlled_contrasts": {
                                "recurrent_coupled_minus_paired_swap": {
                                    "sampled_free_generation_margin_difference": (
                                        _interval(0.04)
                                    )
                                },
                                "recurrent_coupled_minus_current_user_only": {
                                    "sampled_free_generation_margin_difference": (
                                        _interval(0.03)
                                    )
                                },
                                "recurrent_coupled_minus_transition_shuffle": {
                                    "sampled_free_generation_margin_difference": (
                                        _interval(0.02)
                                    )
                                },
                            },
                            "arms": {
                                "native": {
                                    "sampled_degeneration": native_health,
                                },
                                "recurrent_code_coupled": {
                                    "sampled_degeneration": recurrent_health,
                                },
                            },
                        }
                    },
                }
            }
        }
    }
    preregistration = {
        "locked_inputs": {
            "expected_fixed_partition_counts": {"evaluation": 29},
            "expected_development_pairs_after_exclusion": 62,
        },
        "primary_causal_code_gate": {
            "mode": "original_unanswered_prefix",
            "partition": "evaluation",
        },
        "recurrent_history_gate": {},
    }
    return result, preregistration


def _replay_report() -> dict:
    modes = {}
    for mode_name in audit.CONTEXT_MODES:
        rows = {}
        for condition_index, condition in enumerate(audit.REPLAY_CONDITIONS):
            rows[condition] = [
                {
                    "pair_id": "pair-1",
                    "partition": "evaluation",
                    "greedy_margin": 0.1 + condition_index,
                    "sampled_margin": 0.2 + condition_index,
                    "mean_teacher_forced_bpb": 2.5 + condition_index,
                    "sides": {
                        side: {
                            "greedy_hex": f"{condition_index:02x}",
                            "sampled_hex": f"{condition_index + 1:02x}",
                            "greedy_probability_sha256": side * 64,
                            "sampled_probability_sha256": side.upper() * 64,
                        }
                        for side in ("a", "b")
                    },
                }
            ]
        modes[mode_name] = {"available": True, "pair_rows": rows}
    return {"evaluation": {"context_modes": modes}}


def test_locked_gate_calculation_requires_effect_code_and_health() -> None:
    result, preregistration = _gate_inputs()

    gates = audit.compute_locked_gates(result, preregistration)

    assert gates["primary_causal_code"]["positive_interval_passed"]
    assert gates["primary_causal_code"]["code_control_passed"]
    assert gates["primary_causal_code"]["health"]["passed"]
    assert gates["recurrent_history"]["historical_context_contribution"]["passed"]
    assert gates["recurrent_history"]["ordered_successor_learning"]["passed"]
    assert gates["recurrent_history"]["passed"]
    assert gates["mechanism_advancement_passed"]

    failed = copy.deepcopy(result)
    failed["evaluation"]["context_modes"]["original_unanswered_prefix"]["summaries"][
        "evaluation"
    ]["controlled_contrasts"]["recurrent_coupled_minus_transition_shuffle"][
        "sampled_free_generation_margin_difference"
    ]["ci95_low"] = 0.0
    failed_gates = audit.compute_locked_gates(failed, preregistration)
    assert failed_gates["primary_causal_code"]["passed"]
    assert not failed_gates["recurrent_history"]["passed"]
    assert not failed_gates["mechanism_advancement_passed"]


def test_semantic_deviation_disqualifies_only_ordered_successor_gate() -> None:
    result, preregistration = _gate_inputs()

    gates = audit.compute_locked_gates(
        result,
        preregistration,
        ordered_successor_control_eligible=False,
    )

    assert gates["primary_causal_code"]["passed"]
    assert gates["recurrent_history"]["historical_context_contribution"]["passed"]
    ordered = gates["recurrent_history"]["ordered_successor_learning"]
    assert ordered["raw_positive_interval_passed"]
    assert not ordered["control_semantically_eligible"]
    assert not ordered["passed"]
    assert not gates["recurrent_history"]["full_preregistered_gate_passed"]
    assert not gates["mechanism_advancement_passed"]


def test_transition_protocol_audit_records_source_and_role_collapse() -> None:
    preregistration = {
        "context_protocol": {
            "transition_shuffle_control": (
                "Fits the same analytic transition after within-conversation "
                "successor shuffling with a separately locked seed."
            )
        }
    }
    result = {
        "evaluation": {
            "context_modes": {
                mode: {
                    "forecast_diagnostics": {
                        "recurrent_full_history": {
                            "predicted_assistant_role_fraction": 1.0
                        },
                        "recurrent_transition_shuffled": {
                            "predicted_assistant_role_fraction": shuffled
                        },
                    }
                }
                for mode, shuffled in (
                    ("original_unanswered_prefix", 0.47580645161290325),
                    ("long_interruption_crossover", 0.3790322580645161),
                )
            }
        }
    }

    record = audit.audit_transition_shuffle_protocol(
        preregistration,
        result,
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert record["semantic_deviation"]
    assert record["detected_output_coordinate_permutation"]
    assert not record["ordered_successor_gate_eligible"]
    assert all(item["present"] for item in record["implementation_evidence"])
    original = record["role_binding_evidence"]["original_unanswered_prefix"]
    assert original["absolute_drop"] == 1.0 - 0.47580645161290325


def test_health_thresholds_are_relative_to_native() -> None:
    result, preregistration = _gate_inputs()
    recurrent_health = result["evaluation"]["context_modes"][
        "original_unanswered_prefix"
    ]["summaries"]["evaluation"]["arms"]["recurrent_code_coupled"][
        "sampled_degeneration"
    ]
    recurrent_health["mean_printable_fraction"] = 0.899

    gates = audit.compute_locked_gates(result, preregistration)

    assert gates["primary_causal_code"]["positive_interval_passed"]
    assert not gates["primary_causal_code"]["health"]["passed"]
    assert not gates["primary_causal_code"]["passed"]


def test_exact_replay_detects_one_changed_probability_path() -> None:
    reference = _replay_report()
    result = copy.deepcopy(reference)

    exact = audit.compare_exact_replay(result, reference)

    assert exact["passed"]
    assert exact["compared_pair_condition_rows"] == 8
    assert exact["compared_side_condition_rows"] == 16
    assert exact["compared_exact_fields"] == 96

    result["evaluation"]["context_modes"]["original_unanswered_prefix"]["pair_rows"][
        "native"
    ][0]["sides"]["a"]["sampled_probability_sha256"] = "0" * 64
    changed = audit.compare_exact_replay(result, reference)
    assert not changed["passed"]
    assert changed["mismatch_count"] == 1
    assert changed["first_mismatches"][0].endswith("/a/sampled_probability_sha256")


def test_sidecar_verification_checks_digest_and_filename(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    payload = b'{"status":"complete"}\n'
    result.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "result.json.sha256").write_text(
        f"{digest}  result.json\n", encoding="utf-8"
    )

    record = audit.verify_sidecar(result)

    assert record["matches"]
    (tmp_path / "result.json.sha256").write_text(
        f"{'0' * 64}  result.json\n", encoding="utf-8"
    )
    assert not audit.verify_sidecar(result)["matches"]


def test_audit_writer_hashes_exact_bytes_cross_platform(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"

    audit.write_audit_artifact(path, {"status": "valid", "note": "line\nbreak"})

    assert audit.verify_sidecar(path)["matches"]
    assert b"\r\n" not in path.read_bytes()


def test_locked_posix_runtime_path_is_absolute_cross_platform() -> None:
    assert audit._locked_path_is_absolute("/home/runtime/libnative.so")
    assert not audit._locked_path_is_absolute("src/module.py")
