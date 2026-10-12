"""Prime-encoding probe: which prime frame format actually carries a turn's
content into the local byte cortex?

The veccond interface primes the base with a 2061-byte dense float32
serialization of the 512-dim content code. The base is a local,
surprise-grown byte cortex that can only condition on a bounded recent
byte window, so most of the frame is invisible at generation time and the
frame carries little learnable cross-turn structure. This probe tests
alternative prime frames with teacher-forced NLL on held-out turns:

  dense:          the current full float frame (interface baseline)
  dense_tail_L:   the current frame truncated to its last L float bytes
  compact_float:  header + float32 of the top-k content features
  sparse_id:      header + 3 bytes per active feature (index, sign+bin)

For each frame format a fresh copy of the native checkpoint receives
stage-A interface teaching (oracle prime -> response bytes, exactly as the
veccond arms receive it) for P passes, then held-out assistant turns are
teacher-forced under the oracle prime (the turn's own content code) and the
cross-pair shuffled prime (control). The oracle-minus-shuffled NLL gap is
the information signal: a prime format that carries the answer must make
the held-out response substantially less surprising than a shuffled prime.

Local learning only; no backpropagation; no protected test payload accessed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.vector_conditioned_base import (
    vector_input_frame,
)
from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_crossover_ablation import (
    _load_native,
)
from manifoldflow.experiments.dendritic_basin_surprise_future_transition_ablation import (
    load_declared_training_and_development,
)

MAGIC = b"\x00ALM-VEC\x01"
SID_MAGIC = b"\x00ALM-SID\x01"
TOP_K = 64
TARGET_CAP = 512
TAIL_LENGTHS = (32, 64, 128, 256, 1024)
TRAIN_PASSES = (1, 100)
FORMATS = ("dense", "compact_float", "sparse_id", "sparse_tail", "sparse_id2")
MEMORIZE_SAMPLE = 48
MEMORIZE_SHIFT = 137


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def dense_frame(values: np.ndarray) -> bytes:
    return vector_input_frame(values)


def dense_tail_frame(values: np.ndarray, tail_floats: int) -> bytes:
    dense = np.asarray(values, dtype="<f4")
    payload = dense.tobytes(order="C")
    keep = tail_floats * 4
    if keep >= len(payload):
        return dense_frame(values)
    return MAGIC + (b"\x00" * 4) + payload[-keep:]


def compact_float_frame(values: np.ndarray, k: int = TOP_K) -> bytes:
    import struct

    mags = np.abs(values)
    order = np.lexsort((np.arange(len(mags)), -mags))
    top = order[:k]
    return (
        MAGIC
        + struct.pack("<II", 1, int(k))
        + np.asarray(top, dtype="<u4").tobytes(order="C")
        + np.asarray(values[top], dtype="<f4").tobytes(order="C")
    )


def sparse_id_frame(values: np.ndarray, k: int = TOP_K) -> bytes:
    """Top-k features, strongest first, 3 bytes each (index, sign+bin)."""
    mags = np.abs(values)
    order = np.lexsort((np.arange(len(mags)), -mags))
    top = order[:k]
    out = bytearray(SID_MAGIC)
    out += (b"\x02" * 4)
    out += len(top).to_bytes(4, "little")
    for index in top:
        value = float(values[index])
        magnitude = abs(value)
        bin_index = 0 if magnitude < 0.5 else (1 if magnitude < 1.5 else 2)
        flags = ((1 if value < 0.0 else 0) << 7) | bin_index
        out += bytes(((int(index) >> 8) & 0xFF, int(index) & 0xFF, flags))
    return bytes(out)


def sparse_tail_frame(values: np.ndarray, k: int = TOP_K) -> bytes:
    """Top-k features, weakest first so the strongest land in the frame tail
    (the only part a local byte cortex can condition on)."""
    mags = np.abs(values)
    order = np.lexsort((np.arange(len(mags)), mags))
    top = order[-k:]  # ascending: strongest is last
    out = bytearray(SID_MAGIC)
    out += (b"\x03" * 4)
    out += len(top).to_bytes(4, "little")
    for index in top:
        value = float(values[index])
        magnitude = abs(value)
        bin_index = 0 if magnitude < 0.5 else (1 if magnitude < 1.5 else 2)
        flags = ((1 if value < 0.0 else 0) << 7) | bin_index
        out += bytes(((int(index) >> 8) & 0xFF, int(index) & 0xFF, flags))
    return bytes(out)


def sparse_id2_frame(values: np.ndarray, k: int = TOP_K) -> bytes:
    """Top-k features, weakest first, 2 bytes each (index only). Compact
    enough to sit wholly inside a local context window."""
    mags = np.abs(values)
    order = np.lexsort((np.arange(len(mags)), mags))
    top = order[-k:]
    out = bytearray(SID_MAGIC)
    out += (b"\x04" * 4)
    out += len(top).to_bytes(4, "little")
    for index in top:
        out += bytes(((int(index) >> 8) & 0xFF, int(index) & 0xFF))
    return bytes(out)


def frame_builder(format_name: str):
    if format_name == "dense":
        return lambda values: dense_frame(values)
    if format_name == "compact_float":
        return lambda values: compact_float_frame(values)
    if format_name == "sparse_id":
        return lambda values: sparse_id_frame(values)
    if format_name == "sparse_tail":
        return lambda values: sparse_tail_frame(values)
    if format_name == "sparse_id2":
        return lambda values: sparse_id2_frame(values)
    raise ValueError(f"unknown frame format {format_name}")


def prime_and_score(
    cortex: Any, frame: bytes, target: bytes, *, curve: bool = False
) -> tuple[float, list[float]]:
    cortex.reset_episode()
    cortex.observe_chunk_probabilities(np.frombuffer(frame, dtype=np.uint8), learn=False)
    total = 0.0
    per_byte: list[float] = []
    tiny = np.finfo(np.float64).tiny
    for symbol in target:
        law = np.asarray(cortex.predict_next_distribution(), dtype=np.float64)
        total_mass = float(law.sum())
        if total_mass <= 0.0:
            probability = 1.0 / 256.0
        else:
            probability = max(float(law[symbol]) / total_mass, tiny)
        surprise = -math.log2(probability)
        total += surprise
        per_byte.append(surprise)
        cortex.advance_observation(int(symbol), learn=False)
    cortex.reset_episode()
    return total, (per_byte if curve else [])


def _mean_ci(values: list[float], *, seed: int, resamples: int) -> dict[str, Any]:
    return codec_assay._mean_ci(values, seed=seed, resamples=resamples)


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

    all_codes = [
        codec.encode(
            turn.content,
            role=codec_assay.ASSISTANT_ROLE,
            source=turn.source_label,
        ).content
        for turn in assistant_turns
    ]

    # Stage-A teaching subsample (codec is still fit on the full training set
    # so its statistics match the campaign).
    if args.max_train_turns and args.max_train_turns < len(assistant_turns):
        stride = max(1, len(assistant_turns) // args.max_train_turns)
        pick = list(range(0, len(assistant_turns), stride))[: args.max_train_turns]
        train_turns = [assistant_turns[i] for i in pick]
        train_codes = [all_codes[i] for i in pick]
    else:
        train_turns = list(assistant_turns)
        train_codes = list(all_codes)

    # -- held-out rows: evaluation partition, both sides ---------------------
    eval_rows: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.partition != "evaluation":
            continue
        for side_name in ("a", "b"):
            side = pair.side_a if side_name == "a" else pair.side_b
            target = side.full_response[: args.target_cap]
            if not target:
                continue
            # the held-out turn's own content code
            own_code = codec.encode(
                side.full_response,
                role=codec_assay.ASSISTANT_ROLE,
                source=pair.source_label,
            ).content
            eval_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "side": side_name,
                    "target": target,
                    "own_code": own_code,
                    "other_code_index": None,
                }
            )
    if not eval_rows:
        raise ValueError("no evaluation rows available")

    # cross-pair shuffle at the row level (never the same pair). eval_rows is
    # already laid out (pair, side a, side b) in pair order, matching the
    # 2*pair_count row convention of cross_pair_shuffle_indices.
    n_rows = len(eval_rows)
    pair_count = n_rows // 2
    shuffle = codec_assay.cross_pair_shuffle_indices(pair_count, args.control_seed)
    for index in range(n_rows):
        other_index = int(shuffle[index])
        assert 0 <= other_index < n_rows
        assert (
            other_index // 2 != index // 2
        ), "cross-pair shuffle must never map a row onto its own pair"
        eval_rows[index]["other_code_index"] = eval_rows[other_index]["own_code"]

    # -- frame builders ------------------------------------------------------
    builders = {name: frame_builder(name) for name in args.formats}

    # -- untrained NLL (no stage A) ------------------------------------------
    base = _load_native(args.checkpoint)
    if base is None:
        raise ValueError("a frozen native checkpoint is required")
    untrained: dict[str, Any] = {}
    for format_name in args.formats:
        oracle_bits = []
        shuffled_bits = []
        for row in eval_rows:
            target = row["target"]
            bits_oracle, _ = prime_and_score(
                base, builders[format_name](row["own_code"]), target
            )
            bits_shuffled, _ = prime_and_score(
                base,
                builders[format_name](row["other_code_index"]),
                target,
            )
            oracle_bits.append(bits_oracle / len(target))
            shuffled_bits.append(bits_shuffled / len(target))
        gap = [a - b for a, b in zip(oracle_bits, shuffled_bits, strict=True)]
        untrained[format_name] = {
            "oracle_bpb": _mean_ci(oracle_bits, seed=args.bootstrap_seed, resamples=args.bootstrap_resamples),
            "shuffled_bpb": _mean_ci(shuffled_bits, seed=args.bootstrap_seed + 1, resamples=args.bootstrap_resamples),
            "oracle_minus_shuffled_bpb": _mean_ci(gap, seed=args.bootstrap_seed + 2, resamples=args.bootstrap_resamples),
            "n_turns": len(eval_rows),
        }

    # dense-tail sweep (untrained): how much of the frame is visible?
    tail_sweep: dict[str, Any] = {}
    tail_rows = eval_rows[: max(8, len(eval_rows) // 4)]
    for tail_floats in TAIL_LENGTHS:
        oracle_bits = []
        shuffled_bits = []
        for row in tail_rows:
            target = row["target"]
            bits_oracle, _ = prime_and_score(
                base,
                dense_tail_frame(row["own_code"], tail_floats),
                target,
            )
            bits_shuffled, _ = prime_and_score(
                base,
                dense_tail_frame(row["other_code_index"], tail_floats),
                target,
            )
            oracle_bits.append(bits_oracle / len(target))
            shuffled_bits.append(bits_shuffled / len(target))
        gap = [a - b for a, b in zip(oracle_bits, shuffled_bits, strict=True)]
        tail_sweep[str(tail_floats)] = {
            "oracle_bpb": _mean_ci(oracle_bits, seed=args.bootstrap_seed + 10, resamples=args.bootstrap_resamples),
            "shuffled_bpb": _mean_ci(shuffled_bits, seed=args.bootstrap_seed + 11, resamples=args.bootstrap_resamples),
            "oracle_minus_shuffled_bpb": _mean_ci(gap, seed=args.bootstrap_seed + 12, resamples=args.bootstrap_resamples),
            "n_turns": len(tail_rows),
        }

    # -- trained NLL per format / pass count ---------------------------------
    trained: dict[str, Any] = {}
    for format_name in args.formats:
        trained[format_name] = {}
        for passes in args.train_passes:
            stage_base = _load_native(args.checkpoint)
            total_training_surprise = 0.0
            training_bytes = 0
            for pass_index in range(passes):
                for code, turn in zip(train_codes, train_turns, strict=True):
                    target = turn.content[: args.response_cap]
                    if not target:
                        continue
                    frame = builders[format_name](code)
                    stage_base.reset_episode()
                    stage_base.observe_chunk_probabilities(
                        np.frombuffer(frame, dtype=np.uint8), learn=False
                    )
                    observed = np.asarray(
                        stage_base.observe_chunk_probabilities(
                            np.frombuffer(target, dtype=np.uint8), learn=True
                        ),
                        dtype=np.float64,
                    )
                    probabilities = np.where(
                        np.isfinite(observed), observed, 1.0 / 256.0
                    )
                    probabilities = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
                    total_training_surprise += float(
                        (-np.log2(probabilities)).sum()
                    )
                    training_bytes += len(target)
            oracle_bits = []
            shuffled_bits = []
            for row in eval_rows:
                target = row["target"]
                bits_oracle, _ = prime_and_score(
                    stage_base, builders[format_name](row["own_code"]), target
                )
                bits_shuffled, _ = prime_and_score(
                    stage_base,
                    builders[format_name](row["other_code_index"]),
                    target,
                )
                oracle_bits.append(bits_oracle / len(target))
                shuffled_bits.append(bits_shuffled / len(target))
            gap = [a - b for a, b in zip(oracle_bits, shuffled_bits, strict=True)]
            # -- memorization diagnostic: training turns, own frame vs a
            # shifted other turn's frame, same target. A large gap here with
            # a near-zero held-out gap means the cortex memorized frame->text
            # lookups instead of learning a generalizable prime->text rule.
            n_train = len(train_codes)
            mem_own: list[float] = []
            mem_shuf: list[float] = []
            for index in range(0, n_train, max(1, n_train // MEMORIZE_SAMPLE)):
                target = train_turns[index].content[: args.response_cap]
                if not target:
                    continue
                bits_own, _ = prime_and_score(
                    stage_base, builders[format_name](train_codes[index]), target
                )
                other = (index + MEMORIZE_SHIFT) % n_train
                bits_other, _ = prime_and_score(
                    stage_base,
                    builders[format_name](train_codes[other]),
                    target,
                )
                mem_own.append(bits_own / len(target))
                mem_shuf.append(bits_other / len(target))
            mem_gap = [a - b for a, b in zip(mem_own, mem_shuf, strict=True)]
            trained[format_name][str(passes)] = {
                "training_surprise_bpb": total_training_surprise / max(training_bytes, 1),
                "oracle_bpb": _mean_ci(oracle_bits, seed=args.bootstrap_seed + 100 + passes, resamples=args.bootstrap_resamples),
                "shuffled_bpb": _mean_ci(shuffled_bits, seed=args.bootstrap_seed + 101 + passes, resamples=args.bootstrap_resamples),
                "oracle_minus_shuffled_bpb": _mean_ci(gap, seed=args.bootstrap_seed + 102 + passes, resamples=args.bootstrap_resamples),
                "memorized_own_bpb": _mean_ci(mem_own, seed=args.bootstrap_seed + 200 + passes, resamples=args.bootstrap_resamples),
                "memorized_other_bpb": _mean_ci(mem_shuf, seed=args.bootstrap_seed + 201 + passes, resamples=args.bootstrap_resamples),
                "memorized_own_minus_other_bpb": _mean_ci(mem_gap, seed=args.bootstrap_seed + 202 + passes, resamples=args.bootstrap_resamples),
                "n_turns": len(eval_rows),
            }

    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    return {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_prime_encoding_probe",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "teacher_forced_nll": True,
            "free_generation_computed": False,
            "protected_test_payload_accessed": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "top_k": TOP_K,
            "target_cap": args.target_cap,
            "response_cap": args.response_cap,
            "tail_lengths": list(TAIL_LENGTHS),
            "formats": list(args.formats),
            "train_passes": list(args.train_passes),
            "bootstrap_seed": args.bootstrap_seed,
            "control_seed": args.control_seed,
            "memorize_sample": MEMORIZE_SAMPLE,
            "memorize_shift": MEMORIZE_SHIFT,
        },
        "counts": {
            "assistant_turns": len(assistant_turns),
            "train_turns": len(train_turns),
            "evaluation_turns": len(eval_rows),
        },
        "untrained": untrained,
        "dense_tail_sweep": tail_sweep,
        "trained": trained,
        "wall_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-cap", type=int, default=TARGET_CAP)
    parser.add_argument("--response-cap", type=int, default=128)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.add_argument(
        "--train-passes",
        type=int,
        nargs="+",
        default=list(TRAIN_PASSES),
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=list(FORMATS),
    )
    parser.add_argument(
        "--max-train-turns",
        type=int,
        default=0,
        help="subsample of training turns for stage-A teaching (0 = all)",
    )
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
