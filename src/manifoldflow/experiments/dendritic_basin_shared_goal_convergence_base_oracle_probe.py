"""Base transducer probe: can the native base articulate the held-out answer
when primed with the answer's own content code?

The convergence campaign assigns long-horizon coherence to the predictor
organ and the base's role to translating the primed vector into language.
This probe tests the translation alone on a fresh native base: no organ,
no convergence passes.

Two phases per checkpoint:
  - untaught: the base as loaded, never exposed to a primed frame.
  - taught: after the campaign's stage-A interface teaching, one pass over
    every training assistant turn (oracle content code -> response bytes),
    exactly as the veccond arms receive it before pass 0.

Conditions per development pair/side/context mode, each in sampled (seeded
draws) and greedy (argmax) decoding:
  - native:   unprimed free-run
  - oracle:   primed with the held-out answer's own content code
              (leakage ceiling)
  - shuffled: primed with the cross-pair shuffled oracle code (control)

Scored by the paired byte-4gram reference discrimination margin with
bootstrap CIs; full text of every condition dumped for qualitative
reading. Local learning only; no backpropagation; no protected test payload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.vector_conditioned_base import (
    fit_vector_conditioned_turn,
    generate_from_vector,
)
from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond import (
    generate_unprimed,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)
from manifoldflow.experiments.dendritic_basin_surprise_future_transition_ablation import (
    load_declared_training_and_development,
)

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")
PHASES = ("untaught", "taught")
CONDITIONS = ("native", "oracle", "shuffled")
DECODINGS = ("sampled", "greedy")
CONTRAST_CONDITIONS = {
    "oracle_minus_native": ("oracle", "native"),
    "oracle_minus_shuffled": ("oracle", "shuffled"),
}


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/"
            "dendritic_basin_shared_goal_convergence_base_oracle_probe.py",
            "src/manifoldflow/dendritic_basin/vector_conditioned_base.py",
        )
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    for relative, record in sweep_assay.implementation_hashes().items():
        result.setdefault(relative, record)
    return result


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def contrast_series(
    rows_by_condition: dict[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in rows_by_condition["native"]]
    values = {}
    for condition, rows in rows_by_condition.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"condition {condition} rows misaligned")
        values[condition] = [float(row[metric]) for row in rows]
    conditions = {
        condition: codec_assay._mean_ci(
            values[condition], seed=seed, resamples=resamples
        )
        for condition in rows_by_condition
    }
    contrasts = {}
    for contrast_index, (name, (positive, negative)) in enumerate(
        CONTRAST_CONDITIONS.items()
    ):
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    return {"conditions": conditions, "contrasts": contrasts}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    _manifest, verified, conversations, pairs = (
        load_declared_training_and_development(args)
    )

    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )

    base = _load_native(args.checkpoint)
    if base is None:
        raise ValueError("a frozen native checkpoint is required")

    # -- development rows per context mode ----------------------------------
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in CONTEXT_MODES:
        rows = []
        for pair in pairs:
            for side_name in ("a", "b"):
                side = pair.side_a if side_name == "a" else pair.side_b
                context = (
                    side.original_context
                    if mode == "original_unanswered_prefix"
                    else side.interrupted_context
                )
                if context is None:
                    raise AssertionError("available context mode has a missing prefix")
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                        "side": side_name,
                        "reference": side.response_window,
                        "full_response": side.full_response,
                        "source_label": pair.source_label,
                    }
                )
        rows_by_mode[mode] = rows

    # -- oracle content codes (held-out answer's own code) ------------------
    shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), args.control_seed)
    oracle_by_mode: dict[str, list[np.ndarray]] = {}
    shuffled_by_mode: dict[str, list[np.ndarray]] = {}
    for mode in CONTEXT_MODES:
        rows = rows_by_mode[mode]
        oracle = [
            codec.encode(
                row["full_response"],
                role=codec_assay.ASSISTANT_ROLE,
                source=row["source_label"],
            ).content
            for row in rows
        ]
        oracle_by_mode[mode] = oracle
        shuffled_by_mode[mode] = [oracle[int(other)] for other in shuffle]

    index_by_key_by_mode = {
        mode: {
            (row["pair_id"], row["side"]): index
            for index, row in enumerate(rows_by_mode[mode])
        }
        for mode in CONTEXT_MODES
    }

    phase_outputs: dict[str, dict[str, Any]] = {}
    for phase_index, phase in enumerate(PHASES):
        if phase == "taught":
            # Stage-A interface teaching, exactly as the veccond arms receive
            # it before pass 0: oracle content code -> response bytes.
            for conversation in conversations:
                for turn in conversation.assistant_turns:
                    target = turn.content[: args.response_cap]
                    if not target:
                        continue
                    oracle_content = codec.encode(
                        turn.content,
                        role=codec_assay.ASSISTANT_ROLE,
                        source=turn.source_label,
                    ).content
                    fit_vector_conditioned_turn(base, oracle_content, target)
        mode_outputs = {}
        for mode_index, mode in enumerate(CONTEXT_MODES):
            rows = rows_by_mode[mode]
            oracle = oracle_by_mode[mode]
            shuffled_oracle = shuffled_by_mode[mode]
            condition_rows: dict[str, list[dict[str, Any]]] = {
                condition: [] for condition in CONDITIONS
            }
            pair_texts = []
            for pair in pairs:
                side_payloads: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
                for side_name in ("a", "b"):
                    index = index_by_key_by_mode[mode][(pair.pair_id, side_name)]
                    draws = codec_assay._seeded_draws(
                        f"{phase}|{mode}|{pair.pair_id}|{side_name}",
                        seed=args.sample_seed,
                        horizon=args.generation_horizon,
                    )
                    per_condition: dict[str, dict[str, dict[str, Any]]] = {}
                    for condition in CONDITIONS:
                        if condition == "native":
                            sampled_data, sampled_digest = generate_unprimed(
                                base,
                                horizon=args.generation_horizon,
                                uniform_draws=draws,
                            )
                            greedy_data, greedy_digest = generate_unprimed(
                                base,
                                horizon=args.generation_horizon,
                                uniform_draws=None,
                            )
                        else:
                            vector = (
                                oracle[index]
                                if condition == "oracle"
                                else shuffled_oracle[index]
                            )
                            sampled_gen = generate_from_vector(
                                base,
                                vector,
                                horizon=args.generation_horizon,
                                uniform_draws=draws,
                            )
                            greedy_gen = generate_from_vector(
                                base,
                                vector,
                                horizon=args.generation_horizon,
                                uniform_draws=None,
                            )
                            sampled_data, sampled_digest = (
                                sampled_gen.data,
                                sampled_gen.probability_sha256,
                            )
                            greedy_data, greedy_digest = (
                                greedy_gen.data,
                                greedy_gen.probability_sha256,
                            )
                        per_condition[condition] = {
                            "sampled": {
                                "data": sampled_data,
                                "digest": sampled_digest,
                            },
                            "greedy": {
                                "data": greedy_data,
                                "digest": greedy_digest,
                            },
                        }
                    side_payloads[side_name] = per_condition
                texts = {
                    side_name: {
                        "reference_text": _text(
                            rows[index_by_key_by_mode[mode][(pair.pair_id, side_name)]]
                            ["reference"]
                        ),
                        **{
                            f"{condition}.{decoding}": _text(
                                side_payloads[side_name][condition][decoding]["data"]
                            )
                            for condition in CONDITIONS
                            for decoding in DECODINGS
                        },
                    }
                    for side_name in ("a", "b")
                }
                pair_texts.append(
                    {"pair_id": pair.pair_id, "partition": pair.partition, "sides": texts}
                )
                for condition in CONDITIONS:
                    row = {
                        "pair_id": pair.pair_id,
                        "partition": pair.partition,
                    }
                    for decoding in DECODINGS:
                        index_a = index_by_key_by_mode[mode][(pair.pair_id, "a")]
                        index_b = index_by_key_by_mode[mode][(pair.pair_id, "b")]
                        pair_margin = _paired_reference_discrimination(
                            side_payloads["a"][condition][decoding]["data"],
                            side_payloads["b"][condition][decoding]["data"],
                            rows[index_a]["reference"],
                            rows[index_b]["reference"],
                            dimensions=1024,
                        )
                        row[f"{decoding}_margin"] = pair_margin[
                            "correct_minus_crossed_margin"
                        ]
                        if decoding == "sampled":
                            diagnostics = {
                                side: codec_assay._generation_diagnostics(
                                    side_payloads[side][condition]["sampled"]["data"]
                                )
                                for side in ("a", "b")
                            }
                            row["sampled_printable"] = float(
                                np.mean(
                                    [
                                        diagnostics[side]["printable_fraction"]
                                        for side in ("a", "b")
                                    ]
                                )
                            )
                            row["sampled_utf8"] = float(
                                np.mean(
                                    [
                                        1.0
                                        if diagnostics[side]["valid_utf8"]
                                        else 0.0
                                        for side in ("a", "b")
                                    ]
                                )
                            )
                            row["sampled_maximum_run"] = max(
                                diagnostics[side]["maximum_identical_byte_run"]
                                for side in ("a", "b")
                            )
                    condition_rows[condition].append(row)
            partition_summaries = {}
            for partition in ("evaluation", "selection", "all"):
                partition_condition_rows = {
                    condition: [
                        row
                        for row in condition_rows[condition]
                        if partition == "all" or row["partition"] == partition
                    ]
                    for condition in CONDITIONS
                }
                if not partition_condition_rows["native"]:
                    continue
                seed_base = (
                    args.bootstrap_seed
                    + phase_index * 1_000_000
                    + mode_index * 10_000
                )
                summaries = {}
                for decoding_index, decoding in enumerate(DECODINGS):
                    summaries[decoding] = contrast_series(
                        partition_condition_rows,
                        metric=f"{decoding}_margin",
                        seed=seed_base + decoding_index * 100,
                        resamples=args.bootstrap_resamples,
                    )
                summaries["health"] = {
                    condition: {
                        "mean_sampled_printable_fraction": float(
                            np.mean(
                                [row["sampled_printable"] for row in rows_list]
                            )
                        ),
                        "mean_sampled_utf8_fraction": float(
                            np.mean([row["sampled_utf8"] for row in rows_list])
                        ),
                        "outputs_with_run_ge_16": sum(
                            1
                            for row in rows_list
                            if row["sampled_maximum_run"] >= 16
                        ),
                    }
                    for condition, rows_list in partition_condition_rows.items()
                }
                partition_summaries[partition] = summaries
            mode_outputs[mode] = {
                "summaries": partition_summaries,
                "pair_texts": pair_texts,
            }
        phase_outputs[phase] = mode_outputs

    write_digest(args.digest, phase_outputs)
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    return {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_base_oracle_probe",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "inverted_architecture": True,
            "no_organ": True,
            "free_generation_primary": True,
            "teacher_forced_bpb_computed": False,
            "protected_test_payload_accessed": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "generation_horizon": args.generation_horizon,
            "response_cap": args.response_cap,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "phases": phase_outputs,
        "evaluation": {
            "digest_path": str(args.digest),
        },
        "wall_seconds": elapsed,
    }


def write_digest(
    path: Path, phases: dict[str, dict[str, dict[str, Any]]]
) -> None:
    lines: list[str] = []
    for phase in PHASES:
        for mode in CONTEXT_MODES:
            for row in phases[phase][mode]["pair_texts"]:
                lines.append(
                    f"=== phase={phase} mode={mode} "
                    f"pair={row['pair_id']} partition={row['partition']} ==="
                )
                for side_name in ("a", "b"):
                    side = row["sides"][side_name]
                    lines.append(f"--- side {side_name} reference ---")
                    lines.append(side["reference_text"])
                    for key, payload in side.items():
                        if key == "reference_text":
                            continue
                        lines.append(f"--- side {side_name} | {key} ---")
                        lines.append(payload)
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digest", type=Path, required=True)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.set_defaults(wall_limit_seconds=14_400.0)
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
