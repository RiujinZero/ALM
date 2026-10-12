"""Qualitative free-generation read of the core/edge future representation.

The core/edge assays so far are latent-only: the cheap per-episode mean core
passed its cross-pair specificity gate and every persistent-core rescue
failed, but no bytes were ever decoded. Per the standing doctrine that free
generation is primary and BPB is only a diagnostic, this probe drives the
frozen native 500 MB cortex's existing context-coupled receiver with the
representation's own vectors and reads the actual generated text.

Both ends reuse locked machinery unchanged. The latent side is exactly the
core/edge chain: ledger shortlist narrowed by flat top-k, mass-weighted mean
core, probability-weighted edge sum. The actuator side is exactly the
codec native500 receiver: same ContextCoupledByteWriter fit (one matched
pass, oracle turn codes, identical seeds and parameters), which has already
demonstrated causal code control of free generation. The probe adds no new
training beyond that one matched decoder pass.

Arms per side: native (no decoder), local decoder (no code), oracle code
(the held-out answer itself -- a leakage ceiling, never deployable), the
mixture point (core plus this side's weighted edge sum -- the flat top-k
point estimate the latent chain scored), the core alone, a Sattolo-deranged
foreign pair's core (causal specificity control), and this side's core plus
the paired other side's weighted edge sum (the disputed-content control).

This is an exploratory qualitative probe, not a promotion gate. Its primary
output is the full generated text of every arm, written both into the report
and into a standalone human-readable digest; paired sampled-generation
reference margins and output-health diagnostics are recorded as support.
It opens no test payload and claims no goal-directedness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_selective_ledger_ablation as ledger_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_future_core_edge_native_generation.py",
    "tests/test_dendritic_basin_future_core_edge_native_generation.py",
    *sweep_assay.IMPLEMENTATION_PATHS,
    *codec_assay.IMPLEMENTATION_PATHS,
)

ARM_NAMES = (
    "native",
    "local_decoder",
    "oracle_code_coupled",
    "mixture_point_coupled",
    "core_coupled",
    "core_foreign_coupled",
    "edge_swapped_coupled",
)

CONTRAST_NAMES = (
    "core_minus_mixture",
    "core_minus_foreign",
    "core_minus_native",
    "mixture_minus_native",
    "edge_swapped_minus_core",
    "oracle_minus_native",
    "oracle_minus_core",
)

CONTRAST_ARMS = {
    "core_minus_mixture": ("core_coupled", "mixture_point_coupled"),
    "core_minus_foreign": ("core_coupled", "core_foreign_coupled"),
    "core_minus_native": ("core_coupled", "native"),
    "mixture_minus_native": ("mixture_point_coupled", "native"),
    "edge_swapped_minus_core": ("edge_swapped_coupled", "core_coupled"),
    "oracle_minus_native": ("oracle_code_coupled", "native"),
    "oracle_minus_core": ("oracle_code_coupled", "core_coupled"),
}

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")

# Codec native500's locked margin-CI seed. The latent chain keeps the
# inherited surprise-chain bootstrap_seed (261_004) so its fitted objects
# stay identical to every locked run; only the generation-side summaries
# use this seed.
GENERATION_BOOTSTRAP_SEED = 260_933


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(IMPLEMENTATION_PATHS):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


def probe_assignments(
    bank: Any,
    *,
    oracle: tuple[int, ...],
    mixture: tuple[int, ...],
    core: tuple[int, ...],
    core_foreign: tuple[int, ...],
    edge_swapped: tuple[int, ...],
    gain: float,
) -> dict[str, codec_assay.RouteAssignment]:
    result = {
        "native": codec_assay.RouteAssignment(None, (), 0.0),
        "local_decoder": codec_assay.RouteAssignment(bank.local, (), gain),
        "oracle_code_coupled": codec_assay.RouteAssignment(bank.coupled, oracle, gain),
        "mixture_point_coupled": codec_assay.RouteAssignment(
            bank.coupled, mixture, gain
        ),
        "core_coupled": codec_assay.RouteAssignment(bank.coupled, core, gain),
        "core_foreign_coupled": codec_assay.RouteAssignment(
            bank.coupled, core_foreign, gain
        ),
        "edge_swapped_coupled": codec_assay.RouteAssignment(
            bank.coupled, edge_swapped, gain
        ),
    }
    if tuple(result) != ARM_NAMES:
        raise AssertionError("probe arm coverage changed")
    return result


def paired_contrasts(
    rows: Mapping[str, list[Mapping[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if tuple(rows) != ARM_NAMES:
        raise AssertionError("contrast arm coverage changed")
    pair_ids = [row["pair_id"] for row in rows["native"]]
    for name, arm_rows in rows.items():
        if [row["pair_id"] for row in arm_rows] != pair_ids:
            raise AssertionError(f"arm {name} rows do not align with native rows")
    values = {
        name: [float(row[metric]) for row in arm_rows]
        for name, arm_rows in rows.items()
    }
    summaries = {
        name: codec_assay._mean_ci(values[name], seed=seed, resamples=resamples)
        for name in ARM_NAMES
    }
    contrasts: dict[str, Any] = {}
    for contrast_index, name in enumerate(CONTRAST_NAMES):
        positive, negative = CONTRAST_ARMS[name]
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    return {"arms": summaries, "contrasts": contrasts}


def output_health(
    rows: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    if tuple(rows) != ARM_NAMES:
        raise AssertionError("health arm coverage changed")

    def arm_health(arm_rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, int]:
        printable = float(np.mean([row["sampled_printable"] for row in arm_rows]))
        utf8 = float(np.mean([row["sampled_utf8"] for row in arm_rows]))
        long_runs = sum(1 for row in arm_rows if row["sampled_maximum_run"] >= 16)
        return printable, utf8, long_runs

    native_printable, native_utf8, native_long_runs = arm_health(rows["native"])
    health: dict[str, Any] = {}
    for name in ARM_NAMES:
        printable, utf8, long_runs = arm_health(rows[name])
        health[name] = {
            "mean_sampled_printable_fraction": printable,
            "mean_sampled_utf8_fraction": utf8,
            "sampled_outputs_with_run_ge_16": long_runs,
            "printable_loss_vs_native": native_printable - printable,
            "utf8_loss_vs_native": native_utf8 - utf8,
            "extra_long_run_outputs_vs_native": long_runs - native_long_runs,
            "meets_nondegeneration_rule": (
                native_printable - printable <= 0.05
                and native_utf8 - utf8 <= 0.05
                and long_runs - native_long_runs <= 2
            ),
            "sides": len(rows[name]),
        }
    return health


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def write_digest(path: Path, modes: Mapping[str, Sequence[Mapping[str, Any]]]) -> int:
    """Human-readable dump of every arm for every pair; none omitted."""

    lines: list[str] = []
    for mode, pair_rows in modes.items():
        for row in pair_rows:
            lines.append(
                "=== mode="
                f"{mode} pair={row['pair_id']} partition={row['partition']} ==="
            )
            for side_name in ("a", "b"):
                side = row["sides"][side_name]
                lines.append(f"--- side {side_name} reference ---")
                lines.append(side["reference_text"])
                for arm in ARM_NAMES:
                    lines.append(f"--- side {side_name} | {arm} | sampled ---")
                    lines.append(side[arm]["sampled_text"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def _summarize_mode(
    rows_by_arm: Mapping[str, list[dict[str, Any]]],
    *,
    bootstrap_seed: int,
    resamples: int,
) -> dict[str, Any]:
    partitions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    pair_ids = [row["pair_id"] for row in rows_by_arm["native"]]
    partitions_set = {
        row["partition"] for row in rows_by_arm["native"]
    }
    for partition in (*sorted(partitions_set), "all"):
        partitions[partition] = {
            arm: [
                row
                for row in arm_rows
                if partition == "all" or row["partition"] == partition
            ]
            for arm, arm_rows in rows_by_arm.items()
        }
    summaries: dict[str, Any] = {}
    for partition, arm_rows_by_partition in partitions.items():
        if not arm_rows_by_partition["native"]:
            continue
        summaries[partition] = {
            "pairs": len(arm_rows_by_partition["native"]),
            "sampled": paired_contrasts(
                arm_rows_by_partition,
                metric="sampled_margin",
                seed=bootstrap_seed,
                resamples=resamples,
            ),
            "greedy": paired_contrasts(
                arm_rows_by_partition,
                metric="greedy_margin",
                seed=bootstrap_seed + 5_000,
                resamples=resamples,
            ),
            "health": output_health(arm_rows_by_partition),
        }
    return {
        "pair_ids": pair_ids,
        "partitions": summaries,
        "collision_audit": None,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )

    # -- latent chain: identical to the locked core/edge sweep setup ----------
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    crossfit = surprise_assay.cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=codec.content_dimensions,
        folds=args.crossfit_folds,
        fold_seed=args.crossfit_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    fast_matrix = np.stack([item.context_state for item in crossfit.examples])
    fast_projector = surprise_assay.ContextProjector.fit(
        fast_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    token_dimensions = crossfit.examples[0].context_state.size // 2
    token_matrix = np.stack(
        [item.context_state[:token_dimensions] for item in crossfit.examples]
    )
    event_projector = surprise_assay.ContextProjector.fit(
        token_matrix,
        dimensions=args.ledger_event_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    latent_bank, persistent, current, bank_audit = (
        ledger_assay.fit_prototype_bank_and_traces(
            crossfit.examples,
            event_projector,
            conversations=len(conversations),
            token_dimensions=token_dimensions,
            args=args,
        )
    )
    ordered, _unused, overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    fast_states = fast_projector.transform(fast_matrix)
    base, readouts, _global_counts, retained, ledger_training = (
        ledger_assay.train_selective_ledger(
            crossfit.examples,
            fast_states,
            persistent,
            current,
            ordered,
            conversations=len(conversations),
            args=args,
        )
    )
    full_basal = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    assistant_examples = [
        item
        for item in crossfit.examples
        if item.target_turn.role_id == codec_assay.ASSISTANT_ROLE
    ]
    residual_vectors = np.stack(
        [item.target - item.basal_prediction for item in assistant_examples]
    )
    proper_variance = np.maximum(
        np.mean(np.square(residual_vectors), axis=0), args.proper_variance_floor
    )

    frozen_latent = {
        "basal": full_basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    latent_rows: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for mode in CONTEXT_MODES:
        rows = sweep_assay.build_selection_rows(
            full_basal,
            codec,
            pairs,
            fast_projector,
            event_projector,
            latent_bank,
            ordered,
            base,
            readouts["persistent"],
            retained,
            proper_variance,
            mode=mode,
            args=args,
        )
        if len(rows) != 2 * len(pairs):
            raise AssertionError(f"{mode} latent rows do not cover every side")
        latent_rows[mode] = {
            (row["pair_id"], row["side"]): row for row in rows
        }

    # -- actuator: identical to the codec native500 receiver ------------------
    cortex = _load_native(args.checkpoint)
    if cortex is None:
        raise ValueError("a frozen native checkpoint is required")
    decoder_bank = codec_assay.make_decoder_bank(
        persistent_feature_count=codec.persistent_feature_count,
        table_bits=args.table_bits,
        learning_rate=args.learning_rate,
        local_orders=args.local_orders,
    )
    decoder_training = codec_assay.fit_decoder_bank(
        cortex,
        conversations,
        codec,
        decoder_bank,
        response_cap=args.response_cap,
        training_seed=args.training_seed,
    )

    flat_sides = [
        (pair, side_name, pair.side_a if side_name == "a" else pair.side_b)
        for pair in pairs
        for side_name in ("a", "b")
    ]
    shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), args.control_seed)

    def quantized_content(content: np.ndarray, source_label: str) -> tuple[int, ...]:
        return codec.quantize(
            codec.code_from_content(
                content, role=codec_assay.ASSISTANT_ROLE, source=source_label
            )
        )

    writer_fingerprints_before = {
        name: writer.fingerprint() for name, writer in decoder_bank.items()
    }
    mode_outputs: dict[str, dict[str, Any]] = {}
    for mode in CONTEXT_MODES:
        available = all(
            side.original_context is not None for _, _, side in flat_sides
        )
        if mode == "original_unanswered_prefix" and not available:
            mode_outputs[mode] = {
                "available": False,
                "reason": "original unanswered prefix metadata unavailable",
            }
            continue
        contexts = [
            (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            for _, _, side in flat_sides
        ]
        if any(context is None for context in contexts):
            raise AssertionError("available context mode contains a missing prefix")
        oracle_states = [
            codec.quantize(
                codec.encode(
                    side.full_response,
                    role=codec_assay.ASSISTANT_ROLE,
                    source=pair.source_label,
                )
            )
            for pair, _side_name, side in flat_sides
        ]
        mixture_states = [
            quantized_content(
                latent_rows[mode][(pair.pair_id, side_name)]["core"]
                + latent_rows[mode][(pair.pair_id, side_name)]["weighted_edge_sum"],
                pair.source_label,
            )
            for pair, side_name, _side in flat_sides
        ]
        core_states = [
            quantized_content(
                latent_rows[mode][(pair.pair_id, side_name)]["core"],
                pair.source_label,
            )
            for pair, side_name, _side in flat_sides
        ]
        collision_audit = {
            "oracle_sparse_quantized_codes": codec_assay.code_control_collision_audit(
                oracle_states, shuffle
            ),
            "core_sparse_quantized_codes": codec_assay.code_control_collision_audit(
                core_states, shuffle
            ),
            "mixture_sparse_quantized_codes": codec_assay.code_control_collision_audit(
                mixture_states, shuffle
            ),
        }
        rows_by_arm: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ARM_NAMES
        }
        pair_texts: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(pairs):
            payloads: dict[str, dict[str, dict[str, Any]]] = {}
            for side_offset, side_name in enumerate(("a", "b")):
                flat_index = pair_index * 2 + side_offset
                context = contexts[flat_index]
                row = latent_rows[mode][(pair.pair_id, side_name)]
                foreign_pair, foreign_side_name, _foreign_side = flat_sides[
                    int(shuffle[flat_index])
                ]
                foreign_row = latent_rows[mode][
                    (foreign_pair.pair_id, foreign_side_name)
                ]
                swapped_pair, swapped_side_name, _swapped_side = flat_sides[
                    flat_index ^ 1
                ]
                swapped_row = latent_rows[mode][
                    (swapped_pair.pair_id, swapped_side_name)
                ]
                assignments = probe_assignments(
                    decoder_bank,
                    oracle=oracle_states[flat_index],
                    mixture=mixture_states[flat_index],
                    core=core_states[flat_index],
                    core_foreign=quantized_content(
                        foreign_row["core"], foreign_pair.source_label
                    ),
                    edge_swapped=quantized_content(
                        row["core"] + swapped_row["weighted_edge_sum"],
                        pair.source_label,
                    ),
                    gain=args.gain,
                )
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=args.sample_seed,
                    horizon=args.generation_horizon,
                )
                reference = (
                    pair.side_a.response_window
                    if side_name == "a"
                    else pair.side_b.response_window
                )
                side_payloads: dict[str, dict[str, Any]] = {}
                for arm, assignment in assignments.items():
                    greedy = codec_assay.generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=args.generation_horizon,
                        residual_clip=args.residual_clip,
                        uniform_draws=None,
                    )
                    sampled = codec_assay.generate_branch(
                        cortex,
                        context,
                        assignment,
                        horizon=args.generation_horizon,
                        residual_clip=args.residual_clip,
                        uniform_draws=draws,
                    )
                    side_payloads[arm] = {"greedy": greedy, "sampled": sampled}
                payloads[side_name] = side_payloads | {
                    "reference": reference,
                    "core_state": assignments["core_coupled"].state,
                    "edge_swapped_state": assignments[
                        "edge_swapped_coupled"
                    ].state,
                }
            pair_texts.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "sides": {
                        side_name: {
                            "reference_text": _text(payloads[side_name]["reference"]),
                            "reference_hex": payloads[side_name]["reference"].hex(),
                            **{
                                arm: {
                                    "greedy_text": _text(
                                        payloads[side_name][arm]["greedy"].data
                                    ),
                                    "sampled_text": _text(
                                        payloads[side_name][arm]["sampled"].data
                                    ),
                                    "sampled_hex": payloads[side_name][arm][
                                        "sampled"
                                    ].data.hex(),
                                    "sampled_probability_sha256": payloads[
                                        side_name
                                    ][arm]["sampled"].probability_sha256,
                                }
                                for arm in ARM_NAMES
                            },
                        }
                        for side_name in ("a", "b")
                    },
                }
            )
            for arm in ARM_NAMES:
                sampled_pair = _paired_reference_discrimination(
                    payloads["a"][arm]["sampled"].data,
                    payloads["b"][arm]["sampled"].data,
                    payloads["a"]["reference"],
                    payloads["b"]["reference"],
                    dimensions=1024,
                )
                greedy_pair = _paired_reference_discrimination(
                    payloads["a"][arm]["greedy"].data,
                    payloads["b"][arm]["greedy"].data,
                    payloads["a"]["reference"],
                    payloads["b"]["reference"],
                    dimensions=1024,
                )
                sampled_diagnostics = {
                    side_name: codec_assay._generation_diagnostics(
                        payloads[side_name][arm]["sampled"].data
                    )
                    for side_name in ("a", "b")
                }
                rows_by_arm[arm].append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "sampled_margin": sampled_pair["correct_minus_crossed_margin"],
                        "greedy_margin": greedy_pair["correct_minus_crossed_margin"],
                        "sampled_prefix_accuracy_a": codec_assay._prefix_accuracy(
                            payloads["a"][arm]["sampled"].data,
                            payloads["a"]["reference"],
                        ),
                        "sampled_prefix_accuracy_b": codec_assay._prefix_accuracy(
                            payloads["b"][arm]["sampled"].data,
                            payloads["b"]["reference"],
                        ),
                        "sampled_printable": float(
                            np.mean(
                                [
                                    sampled_diagnostics[side]["printable_fraction"]
                                    for side in ("a", "b")
                                ]
                            )
                        ),
                        "sampled_utf8": float(
                            np.mean(
                                [
                                    1.0
                                    if sampled_diagnostics[side]["valid_utf8"]
                                    else 0.0
                                    for side in ("a", "b")
                                ]
                            )
                        ),
                        "sampled_maximum_run": max(
                            sampled_diagnostics[side]["maximum_identical_byte_run"]
                            for side in ("a", "b")
                        ),
                    }
                )
        summary = _summarize_mode(
            rows_by_arm,
            bootstrap_seed=GENERATION_BOOTSTRAP_SEED
            + (0 if mode == "original_unanswered_prefix" else 100_000),
            resamples=args.bootstrap_resamples,
        )
        summary["collision_audit"] = collision_audit
        mode_outputs[mode] = {
            "available": True,
            "summary": summary,
            "pair_texts": pair_texts,
        }

    writer_fingerprints_after = {
        name: writer.fingerprint() for name, writer in decoder_bank.items()
    }
    if writer_fingerprints_before != writer_fingerprints_after:
        raise AssertionError("probe generation modified a decoder writer")
    after_latent = {
        "basal": full_basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    if frozen_latent != after_latent:
        raise AssertionError("probe modified frozen latent predictors")
    digest_lines = write_digest(
        args.digest,
        {
            mode: data["pair_texts"]
            for mode, data in mode_outputs.items()
            if data.get("available")
        },
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached after completed probe")
    return {
        "schema_version": 1,
        "experiment": "future_core_edge_native_generation_probe",
        "status": "complete_development_probe",
        "claim_scope": {
            "exploratory_qualitative_probe_not_a_promotion_gate": True,
            "native_checkpoint_frozen": True,
            "receiver_identical_to_codec_native500_fit": True,
            "receiver_trained_on_oracle_turn_codes_not_cores": True,
            "actual_free_bytes_generated": True,
            "free_generation_primary": True,
            "full_text_dump_written": True,
            "no_goal_or_consequence_learning": True,
            "protected_test_payload_accessed": False,
            "backpropagation_used": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(_manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "config": {
            "raw_dimensions": args.raw_dimensions,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "table_bits": args.table_bits,
            "learning_rate": args.learning_rate,
            "local_orders": list(args.local_orders),
            "response_cap": args.response_cap,
            "generation_horizon": args.generation_horizon,
            "gain": args.gain,
            "residual_clip": args.residual_clip,
            "training_seed": args.training_seed,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "prototype_bank": bank_audit,
        "surprise_population_training": overlay,
        "ledger_training": ledger_training,
        "decoder_training": decoder_training,
        "evaluation": {
            "context_modes": mode_outputs,
            "writer_fingerprints_unchanged_by_evaluation": True,
            "latent_fingerprints_unchanged": True,
            "digest_lines": digest_lines,
            "digest_path": str(args.digest),
        },
        "wall_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digest", type=Path, required=True)
    parser.add_argument("--table-bits", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--local-orders", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=2.0)
    parser.add_argument("--training-seed", type=int, default=260_930)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    # --bootstrap-seed / --bootstrap-resamples / --wall-limit-seconds are
    # inherited from the chain; only the wall limit is re-defaulted here.
    parser.set_defaults(wall_limit_seconds=7_200.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_probe(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
