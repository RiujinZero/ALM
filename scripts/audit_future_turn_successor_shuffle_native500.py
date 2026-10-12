#!/usr/bin/env python3
"""Audit the locked native successor-shuffle development result.

This verifier is deliberately independent of the experiment runner.  It reads
only the locked preregistration, the completed result, the preceding recurrent
reference result, optional captured stdout, checksum sidecars, and explicitly
locked implementation files.  It never opens a dataset payload.

A negative scientific result is valid.  The command exits nonzero only when an
integrity/reproduction check fails; the preregistered primary gate is reported
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOCKED_PREREGISTRATION_SHA256 = (
    "397cd4d549945d0f40a603a5e3b97b657ae227a5a3124055278971f21008dc2b"
)
LOCKED_REFERENCE_SHA256 = (
    "e3b5d5403e2b632b2dcd197d8ee0fceff791aa32a5344ca7a3f459e35d0752ae"
)
PLATFORM_REPRODUCTION_PATH = Path(
    "runs/shared_goal/future_turn_successor_shuffle_v1_latent_wsl_posthoc.json"
)
PLATFORM_REPRODUCTION_STDOUT_PATH = Path(
    "runs/shared_goal/"
    "future_turn_successor_shuffle_v1_latent_wsl_posthoc.stdout.json"
)
LOCKED_PLATFORM_REPRODUCTION_SHA256 = (
    "1c5107f3b1d84f600a392ec0cb5581d59e2d249c73d1d5b8ff2bb8475d466fb4"
)
PROTECTED_DATASET_DIRECTORY = Path(
    "runs/shared_goal/"
    "confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3"
)
LOCKED_NATIVE_PATH = "/home/user/sonic-manifold-flow/cpp/build/bin/libalm_native.so"

REPORTED_IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_successor_shuffle_ablation.py",
    "tests/test_dendritic_basin_future_turn_successor_shuffle_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_future_turn_recurrent_codec_ablation.py",
    "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py",
    "src/manifoldflow/dendritic_basin/future_turn_trajectory.py",
    "src/manifoldflow/dendritic_basin/context_coupled_byte_writer.py",
    "src/manifoldflow/dendritic_basin/native_cortex.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_context_coupled_writer_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_crossover_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_generation_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_shared_goal_response_ablation.py",
    "src/manifoldflow/experiments/"
    "dendritic_basin_phase126_three_epoch_generation.py",
)
LOCAL_ONLY_IMPLEMENTATION_PATHS = (
    "cpp/include/alm/c_api.h",
    "cpp/include/alm/cortex.hpp",
    "cpp/src/c_api.cpp",
)
LOCKED_IMPLEMENTATION_PATHS = (
    *REPORTED_IMPLEMENTATION_PATHS,
    *LOCAL_ONLY_IMPLEMENTATION_PATHS,
    LOCKED_NATIVE_PATH,
)
PLATFORM_NUMERIC_SHUFFLE_HASHES = {
    "predictor_state_sha256": (
        "e270982ad146eb60664bdc7d797214acf0c0c25e68b39a495d6ba1a982bb145e"
    ),
    "original_target_token_sha256": (
        "e26cc903b3ce348e7f5a32da78cbf92eab2eaa2538fab1b082fb77867c8c3421"
    ),
    "shuffled_target_token_sha256": (
        "2cca8cafa036d9e9ef5a7c4d06043f7889fe47d13a97553e08206ed7c811d7e7"
    ),
}

CONTEXT_MODES = (
    "original_unanswered_prefix",
    "long_interruption_crossover",
)
CONDITIONS = (
    "native",
    "recurrent_code_gain0",
    "recurrent_code_coupled",
    "successor_shuffled_code_coupled",
)
REPLAY_CONDITIONS = (
    "native",
    "recurrent_code_coupled",
)
PAIR_REPLAY_FIELDS = (
    "partition",
    "sampled_margin",
    "mean_teacher_forced_bpb",
)
SIDE_REPLAY_FIELDS = (
    "sampled_hex",
    "sampled_probability_sha256",
)
INPUT_HASH_NAMES = (
    "train.bin",
    "train.index.jsonl",
    "dev_pairs.bin",
    "dev_pairs.jsonl",
)
EXPECTED_MODE_INVARIANTS = (
    "gain0_recurrent_exactly_native",
    "actual_bytes_generated_not_candidate_reranked",
    "common_random_draws_shared_across_arms",
    "observed_prefix_predictor_states_identical_across_transitions",
    "control_is_target_shuffle_not_output_coordinate_shuffle",
)

# _summarize_rows assigns the second contrast the locked offset
# 10_000 + 1 * 37.  The original-prefix mode adds no mode offset.
CONTROL_CONTRAST_BOOTSTRAP_OFFSET = 10_037
CODE_DISTINCT_THRESHOLD = 0.95
MAX_PRINTABLE_LOSS = 0.05
MAX_VALID_UTF8_LOSS = 0.05
MAX_ADDED_LONG_RUN_OUTPUTS = 2

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_HEX = re.compile(r"^(?:[0-9a-f]{2})*$")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _locked_path_is_absolute(value: str) -> bool:
    """Recognize locked POSIX paths when the audit is run on Windows."""

    return value.startswith("/") or Path(value).is_absolute()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _assert_outside_protected(path: Path, *, repo_root: Path) -> None:
    protected = repo_root / PROTECTED_DATASET_DIRECTORY
    if _is_within(path, protected):
        raise ValueError(
            f"audit input is inside the protected dataset directory: {path}"
        )


def verify_sidecar(path: Path) -> dict[str, Any]:
    """Verify ``path`` against its adjacent GNU-style SHA-256 sidecar."""

    sidecar = _sidecar_path(path)
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not _is_sha256(fields[0].lower()):
        raise ValueError(f"invalid SHA-256 sidecar: {sidecar}")
    if len(fields) != 2 or fields[1].lstrip("*") != path.name:
        raise ValueError(f"sidecar filename does not name {path.name}")
    declared = fields[0].lower()
    actual = _sha256_file(path)
    return {
        "path": str(path),
        "sidecar": str(sidecar),
        "declared_sha256": declared,
        "actual_sha256": actual,
        "matches": declared == actual,
    }


@dataclass
class CheckBook:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        label: str,
        passed: bool,
        *,
        expected: Any = None,
        actual: Any = None,
        detail: str | None = None,
    ) -> None:
        if label in self.records:
            raise ValueError(f"duplicate audit check: {label}")
        row: dict[str, Any] = {"passed": bool(passed)}
        if expected is not None:
            row["expected"] = expected
        if actual is not None:
            row["actual"] = actual
        if detail is not None:
            row["detail"] = detail
        self.records[label] = row

    def equal(self, label: str, actual: Any, expected: Any) -> None:
        self.add(label, actual == expected, expected=expected, actual=actual)

    def true(self, label: str, actual: Any) -> None:
        self.add(label, actual is True, expected=True, actual=actual)

    @property
    def passed(self) -> bool:
        return all(row["passed"] for row in self.records.values())

    @property
    def failures(self) -> list[str]:
        return [label for label, row in self.records.items() if not row["passed"]]


def _rows_by_id(mode: dict[str, Any], condition: str) -> dict[str, dict[str, Any]]:
    rows = mode["pair_rows"][condition]
    if not isinstance(rows, list):
        raise ValueError(f"pair rows for {condition} are not a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate pair row for {condition}: {pair_id}")
        result[pair_id] = row
    return result


def compare_exact_replay(
    result: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Compare the two unchanged sampled-generation arms exactly."""

    mismatches: list[str] = []
    compared_rows = 0
    compared_sides = 0
    compared_fields = 0
    result_modes = result["evaluation"]["context_modes"]
    reference_modes = reference["evaluation"]["context_modes"]
    for mode_name in CONTEXT_MODES:
        current_mode = result_modes[mode_name]
        prior_mode = reference_modes[mode_name]
        if current_mode.get("available") is not True:
            mismatches.append(f"{mode_name}: current mode unavailable")
            continue
        if prior_mode.get("available") is not True:
            mismatches.append(f"{mode_name}: reference mode unavailable")
            continue
        for condition in REPLAY_CONDITIONS:
            current = _rows_by_id(current_mode, condition)
            prior = _rows_by_id(prior_mode, condition)
            if list(current) != list(prior):
                mismatches.append(f"{mode_name}/{condition}: pair ID/order mismatch")
            for pair_id in sorted(set(current) | set(prior)):
                if pair_id not in current or pair_id not in prior:
                    mismatches.append(
                        f"{mode_name}/{condition}/{pair_id}: missing pair row"
                    )
                    continue
                compared_rows += 1
                current_row = current[pair_id]
                prior_row = prior[pair_id]
                for field_name in PAIR_REPLAY_FIELDS:
                    compared_fields += 1
                    if current_row.get(field_name) != prior_row.get(field_name):
                        mismatches.append(
                            f"{mode_name}/{condition}/{pair_id}/{field_name}"
                        )
                for side_name in ("a", "b"):
                    compared_sides += 1
                    current_side = current_row["sides"][side_name]
                    prior_side = prior_row["sides"][side_name]
                    for field_name in SIDE_REPLAY_FIELDS:
                        compared_fields += 1
                        if current_side.get(field_name) != prior_side.get(field_name):
                            mismatches.append(
                                f"{mode_name}/{condition}/{pair_id}/"
                                f"{side_name}/{field_name}"
                            )
    return {
        "passed": not mismatches,
        "context_modes": list(CONTEXT_MODES),
        "conditions": list(REPLAY_CONDITIONS),
        "compared_pair_condition_rows": compared_rows,
        "compared_side_condition_rows": compared_sides,
        "compared_exact_fields": compared_fields,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:50],
    }


def _mean_ci(
    values: Sequence[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    """Independently reproduce the runner's fixed pair-bootstrap interval."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("bootstrap resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[indexes], axis=1)
    return {
        "pairs": len(array),
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _decode_sampled_hex(value: Any, *, horizon: int | None = None) -> bytes:
    if not isinstance(value, str) or _LOWER_HEX.fullmatch(value) is None:
        raise ValueError("sampled_hex is not canonical lowercase byte hex")
    payload = bytes.fromhex(value)
    if horizon is not None and len(payload) != horizon:
        raise ValueError(
            f"sampled output has {len(payload)} bytes; expected {horizon}"
        )
    return payload


def _generation_diagnostics(payload: bytes) -> dict[str, Any]:
    printable = sum(
        byte in (9, 10, 13) or 32 <= byte <= 126 for byte in payload
    ) / max(len(payload), 1)
    maximum_run = 0
    current_run = 0
    previous: int | None = None
    for byte in payload:
        current_run = current_run + 1 if byte == previous else 1
        maximum_run = max(maximum_run, current_run)
        previous = byte
    try:
        payload.decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    return {
        "printable_fraction": printable,
        "valid_utf8": valid_utf8,
        "maximum_identical_byte_run": maximum_run,
        "contains_newline": b"\n" in payload,
        "unique_byte_fraction": len(set(payload)) / max(len(payload), 1),
    }


def _summarize_health(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [
        _generation_diagnostics(_decode_sampled_hex(row["sides"][side]["sampled_hex"]))
        for row in rows
        for side in ("a", "b")
    ]
    if not diagnostics:
        raise ValueError("cannot summarize output health without pair rows")
    return {
        "mean_printable_fraction": float(
            np.mean([row["printable_fraction"] for row in diagnostics])
        ),
        "valid_utf8_fraction": float(
            np.mean([row["valid_utf8"] for row in diagnostics])
        ),
        "maximum_identical_byte_run": int(
            max(row["maximum_identical_byte_run"] for row in diagnostics)
        ),
        "outputs_with_run_at_least_16": int(
            sum(row["maximum_identical_byte_run"] >= 16 for row in diagnostics)
        ),
    }


def assess_evaluation_code_separation(
    mode: dict[str, Any],
    *,
    total_sides: int,
    evaluation_sides: int,
) -> dict[str, Any]:
    """Bound the evaluation-only rate from the runner's all-side aggregate.

    The locked runner emits no per-side collision bit.  Therefore a nonzero
    all-development collision count generally determines only an interval for
    the evaluation partition.  The lower bound is sufficient to certify the
    locked threshold when at most two of 58 evaluation sides could collide.
    """

    equal = mode.get("intact_vs_successor_shuffled_quantized_code_equal_sides")
    fraction = mode.get(
        "intact_vs_successor_shuffled_quantized_code_distinct_fraction"
    )
    valid_count = (
        isinstance(equal, int)
        and not isinstance(equal, bool)
        and 0 <= equal <= total_sides
    )
    valid_fraction = (
        isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and math.isfinite(float(fraction))
        and 0.0 <= float(fraction) <= 1.0
    )
    expected_fraction = 1.0 - equal / total_sides if valid_count else None
    aggregate_consistent = (
        valid_count and valid_fraction and float(fraction) == expected_fraction
    )
    if (
        not aggregate_consistent
        or evaluation_sides <= 0
        or total_sides < evaluation_sides
    ):
        return {
            "status": "invalid_aggregate",
            "passed": False,
            "threshold": CODE_DISTINCT_THRESHOLD,
            "reported_all_side_equal_count": equal,
            "reported_all_side_distinct_fraction": fraction,
            "aggregate_consistent": False,
        }

    selection_sides = total_sides - evaluation_sides
    minimum_evaluation_equal = max(0, equal - selection_sides)
    maximum_evaluation_equal = min(equal, evaluation_sides)
    lower = 1.0 - maximum_evaluation_equal / evaluation_sides
    upper = 1.0 - minimum_evaluation_equal / evaluation_sides
    exact = lower == upper
    if lower >= CODE_DISTINCT_THRESHOLD:
        status = "passed"
    elif upper < CODE_DISTINCT_THRESHOLD:
        status = "failed"
    else:
        status = "unresolved_from_all_side_aggregate"
    return {
        "status": status,
        "passed": status == "passed",
        "threshold": CODE_DISTINCT_THRESHOLD,
        "reported_scope": "all_development_sides",
        "required_scope": "evaluation_sides",
        "reported_all_side_equal_count": equal,
        "reported_all_side_distinct_fraction": float(fraction),
        "aggregate_consistent": True,
        "evaluation_sides": evaluation_sides,
        "possible_evaluation_equal_count": {
            "minimum": minimum_evaluation_equal,
            "maximum": maximum_evaluation_equal,
        },
        "possible_evaluation_distinct_fraction": {
            "lower": lower,
            "upper": upper,
        },
        "exact_evaluation_distinct_fraction": lower if exact else None,
        "partition_allocation_reported": False,
    }


def _positive_interval(record: dict[str, Any], *, expected_pairs: int) -> bool:
    values = (record.get("mean"), record.get("ci95_low"), record.get("ci95_high"))
    return (
        record.get("pairs") == expected_pairs
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        )
        and all(math.isfinite(float(value)) for value in values)
        and float(record["ci95_low"]) > 0.0
    )


def compute_locked_primary_gate(
    result: dict[str, Any], preregistration: dict[str, Any]
) -> dict[str, Any]:
    """Recompute the locked original/evaluation primary gate from pair rows."""

    mode_name = "original_unanswered_prefix"
    partition = "evaluation"
    mode = result["evaluation"]["context_modes"][mode_name]
    expected_pairs = int(
        preregistration["locked_inputs"]["expected_fixed_partition_counts"][
            partition
        ]
    )
    expected_total_pairs = int(
        preregistration["locked_inputs"][
            "expected_development_pairs_after_exclusion"
        ]
    )
    recurrent = [
        row
        for row in mode["pair_rows"]["recurrent_code_coupled"]
        if row["partition"] == partition
    ]
    shuffled = [
        row
        for row in mode["pair_rows"]["successor_shuffled_code_coupled"]
        if row["partition"] == partition
    ]
    native = [
        row
        for row in mode["pair_rows"]["native"]
        if row["partition"] == partition
    ]
    if [row["pair_id"] for row in recurrent] != [
        row["pair_id"] for row in shuffled
    ]:
        raise ValueError("primary contrast pair IDs/order do not align")
    if len(recurrent) != expected_pairs or len(native) != expected_pairs:
        raise ValueError("primary contrast does not contain the locked pair count")

    differences = [
        float(left["sampled_margin"]) - float(right["sampled_margin"])
        for left, right in zip(recurrent, shuffled, strict=True)
    ]
    bootstrap_seed = int(preregistration["locked_config"]["bootstrap_seed"])
    bootstrap_resamples = int(
        preregistration["locked_config"]["bootstrap_resamples"]
    )
    interval = _mean_ci(
        differences,
        seed=bootstrap_seed + CONTROL_CONTRAST_BOOTSTRAP_OFFSET,
        resamples=bootstrap_resamples,
    )
    summary = mode["summaries"][partition]
    reported_interval = summary["controlled_contrasts"][
        "recurrent_coupled_minus_successor_shuffled"
    ]["sampled_free_generation_margin_difference"]
    effect_passed = _positive_interval(interval, expected_pairs=expected_pairs)

    native_health = _summarize_health(native)
    recurrent_health = _summarize_health(recurrent)
    reported_native_health = summary["arms"]["native"]["sampled_degeneration"]
    reported_recurrent_health = summary["arms"]["recurrent_code_coupled"][
        "sampled_degeneration"
    ]
    printable_loss = float(native_health["mean_printable_fraction"]) - float(
        recurrent_health["mean_printable_fraction"]
    )
    utf8_loss = float(native_health["valid_utf8_fraction"]) - float(
        recurrent_health["valid_utf8_fraction"]
    )
    added_long_runs = int(
        recurrent_health["outputs_with_run_at_least_16"]
    ) - int(native_health["outputs_with_run_at_least_16"])
    health_passed = (
        printable_loss <= MAX_PRINTABLE_LOSS
        and utf8_loss <= MAX_VALID_UTF8_LOSS
        and added_long_runs <= MAX_ADDED_LONG_RUN_OUTPUTS
    )

    code = assess_evaluation_code_separation(
        mode,
        total_sides=2 * expected_total_pairs,
        evaluation_sides=2 * expected_pairs,
    )
    recomputation_passed = (
        interval == reported_interval
        and native_health == reported_native_health
        and recurrent_health == reported_recurrent_health
        and code["aggregate_consistent"]
    )
    passed = effect_passed and code["passed"] and health_passed
    return {
        "endpoint": {
            "mode": mode_name,
            "partition": partition,
            "pairs": expected_pairs,
            "contrast": (
                "recurrent_code_coupled minus "
                "successor_shuffled_code_coupled"
            ),
            "sampled_free_generation_primary": True,
        },
        "effect": {
            "independently_recomputed_interval": interval,
            "reported_interval": reported_interval,
            "reported_interval_exactly_reproduced": interval == reported_interval,
            "strictly_positive_ci_passed": effect_passed,
        },
        "code_separation": code,
        "output_health": {
            "native": native_health,
            "recurrent_code_coupled": recurrent_health,
            "reported_native_exactly_reproduced": (
                native_health == reported_native_health
            ),
            "reported_recurrent_exactly_reproduced": (
                recurrent_health == reported_recurrent_health
            ),
            "printable_fraction_loss": printable_loss,
            "valid_utf8_fraction_loss": utf8_loss,
            "added_outputs_with_run_at_least_16": added_long_runs,
            "thresholds": {
                "maximum_printable_fraction_loss": MAX_PRINTABLE_LOSS,
                "maximum_valid_utf8_fraction_loss": MAX_VALID_UTF8_LOSS,
                "maximum_added_outputs_with_run_at_least_16": (
                    MAX_ADDED_LONG_RUN_OUTPUTS
                ),
            },
            "passed": health_passed,
        },
        "reported_summaries_exactly_reproduced": recomputation_passed,
        "success_requires_all_rules": True,
        "passed": passed,
    }


def _validate_generated_rows(
    result: dict[str, Any], *, horizon: int
) -> dict[str, Any]:
    mismatches: list[str] = []
    checked_outputs = 0
    checked_rows = 0
    for mode_name in CONTEXT_MODES:
        mode = result["evaluation"]["context_modes"][mode_name]
        for condition in CONDITIONS:
            for row in mode["pair_rows"][condition]:
                checked_rows += 1
                for numeric_name in ("sampled_margin", "mean_teacher_forced_bpb"):
                    value = row.get(numeric_name)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    ):
                        mismatches.append(
                            f"{mode_name}/{condition}/{row.get('pair_id')}/{numeric_name}"
                        )
                for side_name in ("a", "b"):
                    checked_outputs += 1
                    side = row.get("sides", {}).get(side_name, {})
                    prefix = f"{mode_name}/{condition}/{row.get('pair_id')}/{side_name}"
                    try:
                        payload = _decode_sampled_hex(
                            side.get("sampled_hex"), horizon=horizon
                        )
                    except ValueError as error:
                        mismatches.append(f"{prefix}/sampled_hex: {error}")
                        continue
                    if not _is_sha256(side.get("sampled_probability_sha256")):
                        mismatches.append(f"{prefix}/sampled_probability_sha256")
                    expected = _generation_diagnostics(payload)
                    if side.get("sampled_diagnostics") != expected:
                        mismatches.append(f"{prefix}/sampled_diagnostics")
    return {
        "passed": not mismatches,
        "checked_pair_condition_rows": checked_rows,
        "checked_outputs": checked_outputs,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:50],
    }


def _validate_expected_structure(
    checks: CheckBook,
    preregistration: dict[str, Any],
    result: dict[str, Any],
    reference: dict[str, Any],
    platform_reproduction: dict[str, Any] | None,
) -> None:
    locked_inputs = preregistration["locked_inputs"]
    checks.equal(
        "preregistration/dataset_directory",
        locked_inputs.get("dataset_directory"),
        str(PROTECTED_DATASET_DIRECTORY).replace("\\", "/"),
    )
    checks.equal(
        "preregistration/implementation_keyset",
        sorted(preregistration["locked_implementation"]),
        sorted(LOCKED_IMPLEMENTATION_PATHS),
    )
    checks.equal(
        "preregistration/reference_path",
        preregistration["integrity_gate"].get("reference_recurrent_result_path"),
        "runs/shared_goal/future_turn_recurrent_codec_native500_v1_dev.json",
    )
    checks.equal(
        "preregistration/reference_sha256",
        preregistration["integrity_gate"].get(
            "reference_recurrent_result_sha256"
        ),
        LOCKED_REFERENCE_SHA256,
    )
    primary = preregistration["primary_result"]
    checks.equal(
        "primary_spec/mode",
        preregistration["context_protocol"].get("primary_mode"),
        "original_unanswered_prefix",
    )
    checks.equal("primary_spec/partition", primary.get("partition"), "evaluation")
    checks.equal(
        "primary_spec/contrast",
        primary.get("contrast"),
        "recurrent_code_coupled minus successor_shuffled_code_coupled",
    )
    checks.true(
        "primary_spec/all_rules_required",
        primary.get("success_requires_all_rules"),
    )

    checks.equal("result/schema_version", result.get("schema_version"), 1)
    checks.equal(
        "result/experiment",
        result.get("experiment"),
        "future_turn_role_stratified_successor_shuffle_development",
    )
    checks.equal("result/status", result.get("status"), "complete_development")
    checks.equal("result/not_smoke", result["config"].get("smoke"), False)
    checks.equal("result/not_latent_only", result["config"].get("latent_only"), False)

    expected_inputs = {name: locked_inputs[name] for name in INPUT_HASH_NAMES}
    checks.equal(
        "inputs/verified_hashes", result.get("verified_inputs"), expected_inputs
    )
    checks.equal(
        "checkpoint/path",
        result["checkpoint"].get("path"),
        locked_inputs["checkpoint_path"],
    )
    checks.equal(
        "checkpoint/sha256",
        result["checkpoint"].get("sha256"),
        locked_inputs["checkpoint_sha256"],
    )
    checks.equal(
        "checkpoint/frozen_embodied_bytes",
        result["checkpoint"].get("transient_embodied_bytes_at_report"),
        result["checkpoint"].get("embodied_bytes_at_load"),
    )
    checks.equal(
        "checkpoint/reproduces_reference_embodied_bytes",
        result["checkpoint"].get("embodied_bytes_at_load"),
        reference["checkpoint"].get("embodied_bytes_at_load"),
    )

    count_map = {
        "training_conversations": "expected_training_conversations",
        "completed_training_assistant_turns": "expected_completed_assistant_turns",
        "training_role_tagged_events": "expected_role_tagged_training_events",
        "adjacent_training_event_transitions": (
            "expected_adjacent_training_event_transitions"
        ),
        "development_pairs": "expected_development_pairs_after_exclusion",
    }
    for report_name, locked_name in count_map.items():
        checks.equal(
            f"counts/{report_name}",
            result["counts"].get(report_name),
            locked_inputs[locked_name],
        )
    checks.equal(
        "counts/development_sides",
        result["counts"].get("development_sides"),
        2 * int(locked_inputs["expected_development_pairs_after_exclusion"]),
    )
    checks.equal(
        "counts/partitions",
        result["evaluation"].get("development_partition_counts"),
        locked_inputs["expected_fixed_partition_counts"],
    )

    config_map = {
        "raw_dimensions": "raw_dimensions",
        "content_dimensions": "content_dimensions",
        "active_content_features": "active_content_features",
        "memory_decay": "memory_decay",
        "transition_ridge": "transition_ridge",
        "successor_shuffle_seed": "successor_shuffle_seed",
        "table_bits": "writer_table_bits",
        "learning_rate": "writer_learning_rate",
        "local_orders": "writer_local_orders",
        "decoder_training_passes": "decoder_training_passes",
        "response_cap": "response_cap",
        "generation_horizon": "generation_horizon",
        "gain": "gain",
        "residual_clip": "residual_clip",
        "training_seed": "training_seed",
        "sample_seed": "sample_seed",
        "bootstrap_seed": "bootstrap_seed",
        "bootstrap_resamples": "bootstrap_resamples",
    }
    for report_name, locked_name in config_map.items():
        checks.equal(
            f"config/{report_name}",
            result["config"].get(report_name),
            preregistration["locked_config"][locked_name],
        )
    checks.equal(
        "config/excluded_engineering_pairs",
        result["config"].get("excluded_development_pair_ids"),
        locked_inputs["excluded_engineering_pair_ids"],
    )
    checks.equal(
        "codec/fit_inputs",
        result["codec"].get("fit_inputs"),
        "completed training assistant turns only",
    )
    checks.true(
        "codec/anchored_during_both_transition_fits",
        result["codec"].get("anchored_during_both_transition_fits"),
    )
    checks.equal(
        "codec/content_dimensions",
        result["codec"].get("dense_content_dimensions"),
        preregistration["locked_config"]["content_dimensions"],
    )
    checks.equal(
        "codec/persistent_features",
        result["codec"].get("sparse_persistent_features"),
        preregistration["locked_config"][
            "persistent_features_including_role_and_source"
        ],
    )

    wall = result.get("wall_seconds")
    wall_limit = float(
        preregistration["locked_config"]["wall_limit_seconds_report_check"]
    )
    checks.add(
        "config/wall_limit",
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(float(wall))
        and 0.0 < float(wall) <= wall_limit,
        expected=f"(0, {wall_limit}]",
        actual=wall,
    )

    expected_claims = {
        "isolated_follow_up_does_not_modify_locked_recurrent_runner": True,
        "control_preserves_exact_teacher_forced_predictor_states": True,
        "control_deranges_successor_content_within_target_role_source": True,
        "target_role_source_constant_coordinates_preserved": True,
        "deterministic_one_step_conditional_mean_only": True,
        "particle_or_multihorizon_claim": False,
        "goal_or_preference_claim": False,
        "free_generation_primary_when_native_enabled": True,
        "teacher_forced_bpb_secondary_only": True,
    }
    checks.equal("claim_scope/exact", result.get("claim_scope"), expected_claims)

    transitions = result["transitions"]
    shuffle = transitions["successor_shuffle_audit"]
    checks.equal(
        "shuffle/control",
        shuffle.get("control"),
        "role_source_stratified_successor_content_derangement",
    )
    checks.equal(
        "shuffle/seed",
        shuffle.get("seed"),
        preregistration["locked_config"]["successor_shuffle_seed"],
    )
    platform_shuffle = (
        platform_reproduction.get("transitions", {}).get(
            "successor_shuffle_audit", {}
        )
        if platform_reproduction is not None
        else {}
    )
    shuffle_field_map = {
        "expected_rows": "rows",
        "expected_strata": "strata",
    }
    for locked_name, expected in preregistration[
        "locked_shuffle_invariants"
    ].items():
        name = shuffle_field_map.get(locked_name, locked_name)
        actual = shuffle.get(name)
        if name in PLATFORM_NUMERIC_SHUFFLE_HASHES:
            strict_match = actual == expected
            platform_match = (
                actual == PLATFORM_NUMERIC_SHUFFLE_HASHES[name]
                and platform_shuffle.get(name) == actual
            )
            checks.add(
                f"shuffle/{name}",
                strict_match or platform_match,
                expected=expected,
                actual=actual,
                detail=(
                    "strict preregistered Windows numeric hash"
                    if strict_match
                    else "explained WSL numeric hash reproduced by the pinned "
                    "posthoc WSL latent run"
                    if platform_match
                    else "neither the preregistered Windows hash nor the pinned "
                    "WSL reproduction hash matched"
                ),
            )
            row = checks.records[f"shuffle/{name}"]
            row["strict_preregistered_match"] = strict_match
            row["explained_cross_platform_deviation"] = (
                platform_match and not strict_match
            )
            if platform_match and not strict_match:
                row["platform_reproduction_sha256"] = (
                    LOCKED_PLATFORM_REPRODUCTION_SHA256
                )
        else:
            checks.equal(f"shuffle/{name}", actual, expected)
    checks.add(
        "transitions/intact_fingerprint_format",
        _is_sha256(transitions.get("intact_fingerprint")),
        expected="64 lowercase hexadecimal characters",
        actual=transitions.get("intact_fingerprint"),
    )
    checks.add(
        "transitions/shuffled_fingerprint_format",
        _is_sha256(transitions.get("successor_shuffled_fingerprint")),
        expected="64 lowercase hexadecimal characters",
        actual=transitions.get("successor_shuffled_fingerprint"),
    )
    checks.equal(
        "transitions/intact_reproduces_reference",
        transitions.get("intact_fingerprint"),
        reference["recurrent_transition"].get("fitted_fingerprint"),
    )
    checks.add(
        "transitions/fingerprints_distinct",
        transitions.get("intact_fingerprint")
        != transitions.get("successor_shuffled_fingerprint"),
        actual={
            "intact": transitions.get("intact_fingerprint"),
            "shuffled": transitions.get("successor_shuffled_fingerprint"),
        },
    )
    checks.equal(
        "transitions/state_dimensions",
        transitions.get("state_dimensions"),
        reference["recurrent_transition"].get("state_dimensions"),
    )
    checks.equal(
        "transitions/token_dimensions",
        transitions.get("token_dimensions"),
        reference["recurrent_transition"].get("token_dimensions"),
    )

    checks.equal("decoder_training/passes", result["decoder_training"].get("passes"), 1)
    checks.true(
        "decoder_training/matched_updates",
        result["decoder_training"].get("matched_episode_and_symbol_updates"),
    )
    for name in ("conversation_order_sha256", "observed_response_sequence_sha256"):
        checks.equal(
            f"decoder_training/reproduces_reference/{name}",
            result["decoder_training"].get(name),
            reference["decoder_training"].get(name),
        )
    checks.equal(
        "decoder_training/writer_keyset",
        sorted(result["decoder_training"]["writers"]),
        ["additive", "coupled", "local"],
    )
    expected_symbols = reference["counts"]["decoder_target_bytes_one_pass"]
    for name, writer in result["decoder_training"]["writers"].items():
        checks.equal(
            f"decoder_training/{name}/episodes",
            writer.get("episodes"),
            locked_inputs["expected_completed_assistant_turns"],
        )
        checks.equal(
            f"decoder_training/{name}/symbols",
            writer.get("symbols"),
            expected_symbols,
        )
    checks.equal(
        "decoder_allocation/reproduces_reference",
        result.get("decoder_allocation"),
        reference.get("decoder_allocation"),
    )
    checks.true(
        "decoder_allocation/matched_tables",
        result["decoder_allocation"].get("all_weight_tables_allocation_matched"),
    )
    checks.true(
        "decoder_allocation/matched_active_count",
        result["decoder_allocation"].get(
            "additive_and_coupled_active_count_matched"
        ),
    )

    evaluation = result["evaluation"]
    for name in (
        "writer_fingerprints_unchanged_by_evaluation",
        "transition_fingerprints_unchanged_by_evaluation",
        "transient_recurrent_states_reset_after_evaluation",
    ):
        checks.true(f"evaluation/{name}", evaluation.get(name))
    checks.equal(
        "latent/native_cortex_not_loaded",
        result["latent_diagnostic"].get("native_cortex_loaded"),
        False,
    )

    expected_pairs = int(locked_inputs["expected_development_pairs_after_exclusion"])
    expected_partition_counts = locked_inputs["expected_fixed_partition_counts"]
    mode_ids: dict[str, list[str]] = {}
    for mode_name in CONTEXT_MODES:
        mode = evaluation["context_modes"][mode_name]
        checks.true(f"evaluation/{mode_name}/available", mode.get("available"))
        checks.true(
            f"evaluation/{mode_name}/predictor_states_matched",
            mode.get("predictor_states_exactly_matched"),
        )
        checks.equal(
            f"evaluation/{mode_name}/condition_keyset",
            tuple(mode["pair_rows"]),
            CONDITIONS,
        )
        native_ids = [str(row["pair_id"]) for row in mode["pair_rows"]["native"]]
        mode_ids[mode_name] = native_ids
        checks.equal(
            f"evaluation/{mode_name}/unique_native_pair_ids",
            len(set(native_ids)),
            expected_pairs,
        )
        for condition in CONDITIONS:
            rows = mode["pair_rows"][condition]
            checks.equal(
                f"evaluation/{mode_name}/{condition}/row_count",
                len(rows),
                expected_pairs,
            )
            checks.equal(
                f"evaluation/{mode_name}/{condition}/pair_id_order",
                [str(row["pair_id"]) for row in rows],
                native_ids,
            )
            partition_counts = {
                name: sum(row.get("partition") == name for row in rows)
                for name in ("selection", "evaluation")
            }
            checks.equal(
                f"evaluation/{mode_name}/{condition}/partitions",
                partition_counts,
                expected_partition_counts,
            )
        checks.equal(
            f"evaluation/{mode_name}/gain0_rows_exactly_native",
            mode["pair_rows"]["recurrent_code_gain0"],
            mode["pair_rows"]["native"],
        )
        checks.equal(
            f"evaluation/{mode_name}/invariant_keyset",
            sorted(mode["invariants"]),
            sorted(EXPECTED_MODE_INVARIANTS),
        )
        for name in EXPECTED_MODE_INVARIANTS:
            checks.true(
                f"evaluation/{mode_name}/invariant/{name}",
                mode["invariants"].get(name),
            )
        for partition, count in (
            ("selection", expected_partition_counts["selection"]),
            ("evaluation", expected_partition_counts["evaluation"]),
            ("all", expected_pairs),
        ):
            checks.equal(
                f"evaluation/{mode_name}/summary/{partition}/pairs",
                mode["summaries"][partition].get("pairs"),
                count,
            )

        latent = result["latent_diagnostic"]["context_modes"][mode_name]
        checks.true(f"latent/{mode_name}/available", latent.get("available"))
        checks.true(
            f"latent/{mode_name}/predictor_states_matched",
            latent.get("predictor_states_exactly_matched"),
        )
        checks.add(
            f"latent/{mode_name}/predictor_state_sequence_hash",
            _is_sha256(latent.get("predictor_state_sequence_sha256")),
            expected="64 lowercase hexadecimal characters",
            actual=latent.get("predictor_state_sequence_sha256"),
        )
        checks.equal(
            f"latent/{mode_name}/code_equal_sides_reproduced",
            latent.get("quantized_code_equal_sides"),
            mode.get("intact_vs_successor_shuffled_quantized_code_equal_sides"),
        )
        checks.equal(
            f"latent/{mode_name}/code_distinct_fraction_reproduced",
            latent.get("quantized_code_distinct_fraction"),
            mode.get("intact_vs_successor_shuffled_quantized_code_distinct_fraction"),
        )

    checks.equal(
        "evaluation/pair_ids_match_across_modes",
        mode_ids["long_interruption_crossover"],
        mode_ids["original_unanswered_prefix"],
    )
    excluded = set(locked_inputs["excluded_engineering_pair_ids"])
    checks.equal(
        "evaluation/excluded_engineering_pairs_absent",
        sorted(excluded.intersection(mode_ids["original_unanswered_prefix"])),
        [],
    )

    checks.add(
        "platform_reproduction/provided",
        platform_reproduction is not None,
        expected=True,
        actual=platform_reproduction is not None,
    )
    if platform_reproduction is not None:
        checks.equal(
            "platform_reproduction/status",
            platform_reproduction.get("status"),
            "complete_latent_development",
        )
        checks.equal(
            "platform_reproduction/not_smoke",
            platform_reproduction["config"].get("smoke"),
            False,
        )
        checks.true(
            "platform_reproduction/latent_only",
            platform_reproduction["config"].get("latent_only"),
        )
        checks.equal(
            "platform_reproduction/verified_inputs",
            platform_reproduction.get("verified_inputs"),
            result.get("verified_inputs"),
        )
        checks.equal(
            "platform_reproduction/counts",
            platform_reproduction.get("counts"),
            result.get("counts"),
        )
        reproduction_config_names = (
            "raw_dimensions",
            "content_dimensions",
            "active_content_features",
            "memory_decay",
            "transition_ridge",
            "successor_shuffle_seed",
            "excluded_development_pair_ids",
        )
        checks.equal(
            "platform_reproduction/config",
            {
                name: platform_reproduction["config"].get(name)
                for name in reproduction_config_names
            },
            {name: result["config"].get(name) for name in reproduction_config_names},
        )
        checks.equal(
            "platform_reproduction/implementation_files",
            platform_reproduction.get("implementation_files"),
            result.get("implementation_files"),
        )
        for name in (
            "intact_fingerprint",
            "successor_shuffled_fingerprint",
            "state_dimensions",
            "token_dimensions",
            "successor_shuffle_audit",
        ):
            checks.equal(
                f"platform_reproduction/transitions/{name}",
                platform_reproduction["transitions"].get(name),
                result["transitions"].get(name),
            )
        checks.equal(
            "platform_reproduction/latent_diagnostic",
            platform_reproduction.get("latent_diagnostic"),
            result.get("latent_diagnostic"),
        )


def _validate_implementation_and_runtime(
    checks: CheckBook,
    preregistration: dict[str, Any],
    result: dict[str, Any],
    reference: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    locked = preregistration["locked_implementation"]
    reported = result["implementation_files"]
    checks.equal(
        "implementation/reported_keyset",
        tuple(reported),
        REPORTED_IMPLEMENTATION_PATHS,
    )
    local_records: dict[str, Any] = {}
    for name in REPORTED_IMPLEMENTATION_PATHS:
        row = reported.get(name, {})
        checks.equal(
            f"implementation/report/{name}/sha256",
            row.get("sha256"),
            locked.get(name),
        )
        checks.add(
            f"implementation/report/{name}/bytes",
            isinstance(row.get("bytes"), int)
            and not isinstance(row.get("bytes"), bool)
            and int(row["bytes"]) > 0,
            expected="positive integer",
            actual=row.get("bytes"),
        )

    for name in (*REPORTED_IMPLEMENTATION_PATHS, *LOCAL_ONLY_IMPLEMENTATION_PATHS):
        path = repo_root / name
        _assert_outside_protected(path, repo_root=repo_root)
        if path.is_file():
            actual_hash = _sha256_file(path)
            actual_bytes = path.stat().st_size
            checks.equal(
                f"implementation/local/{name}/sha256", actual_hash, locked[name]
            )
            if name in reported:
                checks.equal(
                    f"implementation/local/{name}/bytes",
                    reported[name].get("bytes"),
                    actual_bytes,
                )
            local_records[name] = {
                "path": str(path),
                "bytes": actual_bytes,
                "sha256": actual_hash,
                "independently_rehashed": True,
            }
        else:
            checks.add(
                f"implementation/local/{name}/exists",
                False,
                expected=True,
                actual=False,
            )
            local_records[name] = {
                "path": str(path),
                "independently_rehashed": False,
            }

    absolute = [name for name in locked if _locked_path_is_absolute(name)]
    checks.equal(
        "native_runtime/one_locked_absolute_binary",
        absolute,
        [LOCKED_NATIVE_PATH],
    )
    native = result["native_runtime"]
    prior_native = reference["native_runtime"]
    checks.equal(
        "native_runtime/resolved_path", native.get("resolved_path"), LOCKED_NATIVE_PATH
    )
    checks.equal(
        "native_runtime/sha256", native.get("sha256"), locked[LOCKED_NATIVE_PATH]
    )
    checks.add(
        "native_runtime/bytes_reported",
        isinstance(native.get("bytes"), int)
        and not isinstance(native.get("bytes"), bool)
        and int(native["bytes"]) > 0,
        expected="positive integer",
        actual=native.get("bytes"),
    )
    checks.add(
        "native_runtime/version_reported",
        isinstance(native.get("version"), str) and bool(native["version"].strip()),
        expected="nonempty string",
        actual=native.get("version"),
    )
    for name in (
        "resolved_path",
        "bytes",
        "sha256",
        "version",
        "alm_native_dll_override_set",
        "loaded_via_override",
    ):
        checks.equal(
            f"native_runtime/reproduces_reference/{name}",
            native.get(name),
            prior_native.get(name),
        )
    checks.add(
        "native_runtime/override_flags_are_boolean",
        isinstance(native.get("alm_native_dll_override_set"), bool)
        and isinstance(native.get("loaded_via_override"), bool),
        actual={
            "alm_native_dll_override_set": native.get("alm_native_dll_override_set"),
            "loaded_via_override": native.get("loaded_via_override"),
        },
    )
    local_records[LOCKED_NATIVE_PATH] = {
        "path": LOCKED_NATIVE_PATH,
        "bytes": native.get("bytes"),
        "sha256": native.get("sha256"),
        "version": native.get("version"),
        "independently_rehashed": False,
        "verified_from_locked_runtime_provenance": True,
    }
    return local_records


def audit_loaded_reports(
    preregistration: dict[str, Any],
    result: dict[str, Any],
    reference: dict[str, Any],
    *,
    repo_root: Path,
    platform_reproduction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit already-loaded reports without opening any dataset payload."""

    checks = CheckBook()
    _validate_expected_structure(
        checks,
        preregistration,
        result,
        reference,
        platform_reproduction,
    )
    implementation = _validate_implementation_and_runtime(
        checks,
        preregistration,
        result,
        reference,
        repo_root=repo_root,
    )
    generated = _validate_generated_rows(
        result, horizon=int(preregistration["locked_config"]["generation_horizon"])
    )
    checks.true("generated_rows/schema_and_diagnostics", generated["passed"])
    replay = compare_exact_replay(result, reference)
    checks.true("exact_reference_replay", replay["passed"])
    gate = compute_locked_primary_gate(result, preregistration)
    checks.true(
        "primary_gate/reported_summaries_recomputed",
        gate["reported_summaries_exactly_reproduced"],
    )
    explained_deviations = {
        name: row
        for name, row in checks.records.items()
        if row.get("explained_cross_platform_deviation") is True
    }
    integrity_passed = checks.passed
    deviation_suffix = "_with_explained_platform_numeric_hash_deviation" if (
        explained_deviations
    ) else ""
    status = (
        "invalid_integrity"
        if not integrity_passed
        else "valid_primary_gate_passed" + deviation_suffix
        if gate["passed"]
        else "valid_primary_gate_not_demonstrated" + deviation_suffix
    )
    return {
        "schema_version": 1,
        "task": "future_turn_successor_shuffle_native500_independent_audit",
        "status": status,
        "integrity_passed": integrity_passed,
        "integrity_failures": checks.failures,
        "checks": checks.records,
        "strict_preregistered_shuffle_hashes_match": not explained_deviations,
        "explained_cross_platform_deviations": explained_deviations,
        "local_implementation_hashes": implementation,
        "generated_row_audit": generated,
        "exact_recurrent_reference_replay": replay,
        "locked_primary_result_gate": gate,
        "scope": {
            "dataset_payloads_opened": False,
            "protected_dataset_directory": str(PROTECTED_DATASET_DIRECTORY).replace(
                "\\", "/"
            ),
            "sampled_free_generation_is_primary": True,
            "teacher_forced_bpb_is_secondary_only": True,
            "negative_scientific_result_is_not_an_integrity_failure": True,
            "no_farther_horizon_particle_goal_or_preference_claim": True,
        },
    }


def _sidecar_record(path: Path) -> dict[str, Any]:
    try:
        return verify_sidecar(path)
    except (OSError, ValueError) as error:
        return {"path": str(path), "matches": False, "error": str(error)}


def audit_files(
    preregistration_path: Path,
    result_path: Path,
    *,
    repo_root: Path,
    reference_path: Path | None = None,
    stdout_path: Path | None = None,
    platform_reproduction_path: Path | None = None,
    platform_reproduction_stdout_path: Path | None = None,
) -> dict[str, Any]:
    """Verify artifact hashes, then perform the independent report audit."""

    repo_root = repo_root.resolve()
    if platform_reproduction_path is None:
        platform_reproduction_path = repo_root / PLATFORM_REPRODUCTION_PATH
    if platform_reproduction_stdout_path is None:
        platform_reproduction_stdout_path = (
            repo_root / PLATFORM_REPRODUCTION_STDOUT_PATH
        )
    for path in (
        preregistration_path,
        result_path,
        stdout_path,
        platform_reproduction_path,
        platform_reproduction_stdout_path,
    ):
        if path is not None:
            _assert_outside_protected(path, repo_root=repo_root)

    preregistration_record = _sidecar_record(preregistration_path)
    preregistration = _read_json(preregistration_path)
    if reference_path is None:
        reference_path = repo_root / preregistration["integrity_gate"][
            "reference_recurrent_result_path"
        ]
    _assert_outside_protected(reference_path, repo_root=repo_root)

    result_record = _sidecar_record(result_path)
    reference_record = _sidecar_record(reference_path)
    platform_record = _sidecar_record(platform_reproduction_path)
    platform_stdout_record = _sidecar_record(platform_reproduction_stdout_path)
    result = _read_json(result_path)
    reference = _read_json(reference_path)
    platform_reproduction = _read_json(platform_reproduction_path)
    audit = audit_loaded_reports(
        preregistration,
        result,
        reference,
        repo_root=repo_root,
        platform_reproduction=platform_reproduction,
    )

    file_checks = CheckBook()
    sidecars = {
        "preregistration": preregistration_record,
        "result": result_record,
        "reference": reference_record,
        "platform_reproduction": platform_record,
        "platform_reproduction_stdout": platform_stdout_record,
    }
    for label, record in sidecars.items():
        file_checks.true(f"sidecar/{label}", record.get("matches"))
    file_checks.equal(
        "preregistration/pinned_sha256",
        preregistration_record.get("actual_sha256"),
        LOCKED_PREREGISTRATION_SHA256,
    )
    file_checks.equal(
        "reference/pinned_sha256",
        reference_record.get("actual_sha256"),
        LOCKED_REFERENCE_SHA256,
    )
    file_checks.equal(
        "reference/preregistered_sha256",
        reference_record.get("actual_sha256"),
        preregistration["integrity_gate"]["reference_recurrent_result_sha256"],
    )
    file_checks.equal(
        "platform_reproduction/pinned_sha256",
        platform_record.get("actual_sha256"),
        LOCKED_PLATFORM_REPRODUCTION_SHA256,
    )
    file_checks.true(
        "platform_reproduction/stdout_exactly_matches_result",
        platform_reproduction_stdout_path.read_bytes()
        == platform_reproduction_path.read_bytes(),
    )

    if stdout_path is not None:
        stdout_record = _sidecar_record(stdout_path)
        sidecars["stdout"] = stdout_record
        file_checks.true("sidecar/stdout", stdout_record.get("matches"))
        file_checks.true(
            "stdout/exactly_matches_result",
            stdout_path.read_bytes() == result_path.read_bytes(),
        )
    else:
        sidecars["stdout"] = {"provided": False, "required": False}

    combined_passed = audit["integrity_passed"] and file_checks.passed
    audit["artifact_hashes"] = sidecars
    audit["file_checks"] = file_checks.records
    audit["integrity_failures"] = [
        *audit["integrity_failures"],
        *file_checks.failures,
    ]
    audit["integrity_passed"] = combined_passed
    deviation_suffix = (
        "_with_explained_platform_numeric_hash_deviation"
        if audit["explained_cross_platform_deviations"]
        else ""
    )
    audit["status"] = (
        "invalid_integrity"
        if not combined_passed
        else "valid_primary_gate_passed" + deviation_suffix
        if audit["locked_primary_result_gate"]["passed"]
        else "valid_primary_gate_not_demonstrated" + deviation_suffix
    )
    audit["opened_files"] = [
        str(preregistration_path),
        str(_sidecar_path(preregistration_path)),
        str(result_path),
        str(_sidecar_path(result_path)),
        str(reference_path),
        str(_sidecar_path(reference_path)),
        str(platform_reproduction_path),
        str(_sidecar_path(platform_reproduction_path)),
        str(platform_reproduction_stdout_path),
        str(_sidecar_path(platform_reproduction_stdout_path)),
        *(
            [str(stdout_path), str(_sidecar_path(stdout_path))]
            if stdout_path is not None
            else []
        ),
        *[
            row["path"]
            for row in audit["local_implementation_hashes"].values()
            if row["independently_rehashed"]
        ],
    ]
    return audit


def write_audit_artifact(path: Path, audit: dict[str, Any]) -> str:
    """Write canonical UTF-8 audit bytes and a checksum of those bytes."""

    payload = json.dumps(audit, indent=2) + "\n"
    encoded = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    _sidecar_path(path).write_bytes(
        f"{_sha256(encoded)}  {path.name}\n".encode()
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--stdout-result", type=Path)
    parser.add_argument("--platform-reproduction", type=Path)
    parser.add_argument("--platform-reproduction-stdout", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = audit_files(
        args.preregistration,
        args.result,
        repo_root=args.repo_root,
        reference_path=args.reference_result,
        stdout_path=args.stdout_result,
        platform_reproduction_path=args.platform_reproduction,
        platform_reproduction_stdout_path=args.platform_reproduction_stdout,
    )
    payload = write_audit_artifact(args.audit, audit)
    print(payload, end="")
    if not audit["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
