"""Orchestrator v1: cross-organ guidance built from the two existing
substrates (cortex.hpp dendritic, vision.hpp synaptic) pointed at a fused
evidence stream instead of raw sensory data. See
docs/ALM_V1_Architecture.md, "Sketch: an executive orchestrator for
cross-organ guidance (v1)" and its "v1 empirical results" subsection.

Two organs predict the same next-byte target from the same corpus:
  organ_A: dendritic byte cortex (`NativeGrowableCortex`), narrow exact-
    match context.
  organ_B: synaptic vision predictor (`NativeGrowingVision`), dense byte-
    embedding evidence -- full synapse coverage (lesson from the byte-
    synapse probe: partial coverage on a sparse representation leaves
    most assemblies blind).

The orchestrator watches both organs plus a trace of the real input:
  orch_dendritic: a second byte cortex, LONGER context, over the actual
    raw byte stream -- a genuinely broader view of the same ground
    truth, not the organs' interpretation of it.
  orch_synaptic: a vision instance whose evidence is organ_A's full vote
    distribution + organ_B's full vote distribution + a small dense
    embedding of the last couple of raw bytes. Predicts the same target.

Combining the two orchestrator halves by a flat sum measurably hurt --
diluting organ_A's sharp, well-calibrated distribution with
orch_synaptic's much more diffuse one, regardless of how much temporal
smoothing was applied to the inputs (both a ~7-step and ~33-step EMA
were tried and made no measurable difference: this is a fusion-rule
problem, not a timing problem). Confidence-weighting the combine step
(each half scaled by its own peak probability before summing) fixed it:
validated to make the orchestrator's own prediction beat organ_A
outright (not just on accuracy) and to make soft-bias blending help both
organs monotonically, not just the weak one, across the weights tested.

Reports (JSON): per-organ and orchestrator bpb/accuracy overall and
conditioned on organ_A/organ_B disagreement, plus a soft-bias sweep
(blending the orchestrator's vote into each organ's own vote, additive,
same shape as episodic recall's boost) across --bias-weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.native_cortex import NativeGrowableCortex
from manifoldflow.dendritic_basin.native_vision import NativeGrowingVision

CLASSES = 256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help=(
            "Byte offset at which to start the online run. Use this to select "
            "an episode-aligned source block instead of always measuring the "
            "first bytes of an interleaved corpus."
        ),
    )
    parser.add_argument("--total-bytes", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--rng-seed", type=int, default=0)

    parser.add_argument("--organ-a-context-bytes", type=int, default=4)

    parser.add_argument("--organ-b-context-bytes", type=int, default=4)
    parser.add_argument("--organ-b-embed-dim", type=int, default=24)
    parser.add_argument("--organ-b-max-assemblies", type=int, default=40_000)

    parser.add_argument(
        "--orch-dendritic-context-bytes",
        type=int,
        default=8,
        help="cortex.hpp's max_context_bytes hard ceiling is 8.",
    )
    parser.add_argument("--orch-trace-bytes", type=int, default=2)
    parser.add_argument("--orch-trace-embed-dim", type=int, default=16)
    parser.add_argument("--orch-max-assemblies", type=int, default=40_000)

    parser.add_argument(
        "--bias-weights",
        type=str,
        default="0.0,0.1,0.25,0.5,0.65,0.8,0.9,1.0",
        help="Comma-separated soft-bias weights to sweep.",
    )
    parser.add_argument(
        "--laplace-eps",
        type=float,
        default=0.5,
        help=(
            "Smoothing for raw vision.hpp vote arrays only -- these are "
            "NOT already a probability distribution (many classes can be "
            "exactly zero). Do not apply this to cortex.hpp's "
            "predict_next_distribution() output, which already is one; "
            "doing so was a real bug in an earlier iteration (collapsed "
            "a 3.0 bpb distribution to ~7.4, near-uniform)."
        ),
    )
    return parser


def make_byte_embeddings(rng: np.random.Generator, classes: int, dim: int) -> np.ndarray:
    table = rng.standard_normal((classes, dim)).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    return table


def embed_context(history_tail: bytes, table: np.ndarray) -> np.ndarray:
    return np.concatenate([table[b] for b in history_tail]).astype(np.float32)


def to_probs(vec: np.ndarray, eps: float) -> np.ndarray:
    """Raw vote/count arrays from vision.hpp: real Laplace smoothing needed
    to avoid catastrophic zero-probability misses on unseen classes."""
    v = np.clip(np.asarray(vec, dtype=np.float64), 0.0, None)
    total = v.sum()
    return (v + eps) / (total + eps * len(v))


def dendritic_probs(vec: np.ndarray) -> np.ndarray:
    """cortex.hpp's predict_next_distribution(): already a normalized,
    calibrated probability distribution. Only a numerical floor, no mass
    redistribution."""
    v = np.clip(np.asarray(vec, dtype=np.float64), 1e-12, None)
    return v / v.sum()


def bpb_of(prob_true: float) -> float:
    return -np.log2(max(prob_true, 1e-300))


def make_vision(dims_needed: int, max_assemblies: int) -> tuple[NativeGrowingVision, int]:
    image_size = int(np.ceil(np.sqrt(dims_needed / 10.0)))
    dims = 10 * image_size * image_size
    learner = NativeGrowingVision(
        classes=CLASSES,
        image_size=image_size,
        columns=4,
        maximum_synapses=dims,  # full coverage -- see module docstring.
        maximum_visual_assemblies=max_assemblies,
    )
    return learner, dims


def run(args: argparse.Namespace) -> dict:
    if args.start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if args.total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    bias_weights = [float(w) for w in args.bias_weights.split(",") if w.strip()]

    rng = np.random.default_rng(args.rng_seed)
    organ_b_table = make_byte_embeddings(rng, CLASSES, args.organ_b_embed_dim)
    trace_table = make_byte_embeddings(rng, CLASSES, args.orch_trace_embed_dim)

    organ_a = NativeGrowableCortex(
        max_context_bytes=args.organ_a_context_bytes, max_sequence_nursery_paths=50_000
    )
    organ_b, organ_b_dims = make_vision(
        args.organ_b_context_bytes * args.organ_b_embed_dim, args.organ_b_max_assemblies
    )
    orch_dendritic = NativeGrowableCortex(
        max_context_bytes=args.orch_dendritic_context_bytes, max_sequence_nursery_paths=50_000
    )
    orch_synaptic_dims_needed = CLASSES + CLASSES + args.orch_trace_bytes * args.orch_trace_embed_dim
    orch_synaptic, orch_synaptic_dims = make_vision(orch_synaptic_dims_needed, args.orch_max_assemblies)

    min_context = max(args.organ_b_context_bytes, args.orch_trace_bytes)
    with open(args.corpus, "rb") as f:
        f.seek(args.start_offset)
        data = f.read(args.total_bytes)
    if len(data) <= min_context:
        raise ValueError(
            "corpus slice must contain more bytes than the required context "
            f"({min_context}); read {len(data)} bytes at offset {args.start_offset}"
        )

    history = bytearray()

    bits_a = bits_b = bits_orch = 0.0
    correct_a = correct_b = correct_orch = 0
    bits_a_disagree = bits_b_disagree = bits_orch_disagree = 0.0
    n_disagree = 0
    bits_blend_a = {w: 0.0 for w in bias_weights}
    bits_blend_b = {w: 0.0 for w in bias_weights}
    scored = 0

    started = time.perf_counter()
    for i, target in enumerate(data):
        if len(history) >= min_context:
            probs_a_raw = organ_a.predict_next_distribution()
            probs_a = dendritic_probs(probs_a_raw)

            ctx_b = bytes(history[-args.organ_b_context_bytes :])
            vec_b = embed_context(ctx_b, organ_b_table)
            if vec_b.shape[0] < organ_b_dims:
                vec_b = np.pad(vec_b, (0, organ_b_dims - vec_b.shape[0]))
            _, votes_b_raw = organ_b.predict_single_column(vec_b)
            probs_b = to_probs(votes_b_raw, args.laplace_eps)

            probs_od_raw = orch_dendritic.predict_next_distribution()
            probs_od = dendritic_probs(probs_od_raw)

            trace_ctx = bytes(history[-args.orch_trace_bytes :])
            trace_vec = embed_context(trace_ctx, trace_table)
            fused = np.concatenate(
                [probs_a.astype(np.float32), probs_b.astype(np.float32), trace_vec]
            )
            if fused.shape[0] < orch_synaptic_dims:
                fused = np.pad(fused, (0, orch_synaptic_dims - fused.shape[0]))
            _, votes_os_raw = orch_synaptic.predict_single_column(fused)
            probs_os = to_probs(votes_os_raw, args.laplace_eps)

            conf_od = float(probs_od.max())
            conf_os = float(probs_os.max())
            probs_orch = conf_od * probs_od + conf_os * probs_os
            probs_orch = probs_orch / probs_orch.sum()

            scored += 1
            bits_a += bpb_of(probs_a[target])
            bits_b += bpb_of(probs_b[target])
            bits_orch += bpb_of(probs_orch[target])
            correct_a += int(np.argmax(probs_a) == target)
            correct_b += int(np.argmax(probs_b) == target)
            correct_orch += int(np.argmax(probs_orch) == target)

            if int(np.argmax(probs_a)) != int(np.argmax(probs_b)):
                n_disagree += 1
                bits_a_disagree += bpb_of(probs_a[target])
                bits_b_disagree += bpb_of(probs_b[target])
                bits_orch_disagree += bpb_of(probs_orch[target])

            for w in bias_weights:
                blended_a = to_probs((1 - w) * probs_a + w * probs_orch, eps=0.0)
                blended_b = to_probs((1 - w) * probs_b + w * probs_orch, eps=0.0)
                bits_blend_a[w] += bpb_of(blended_a[target])
                bits_blend_b[w] += bpb_of(blended_b[target])

            organ_b.observe_external(vec_b, target)
            orch_synaptic.observe_external(fused, target)

        organ_a.observe_chunk_probabilities(np.array([target], dtype=np.uint8), learn=True)
        orch_dendritic.observe_chunk_probabilities(np.array([target], dtype=np.uint8), learn=True)
        history.append(target)

        if (i + 1) % args.checkpoint_every == 0 and scored > 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {i + 1:,} bytes | bpb A={bits_a / scored:.3f} B={bits_b / scored:.3f} "
                f"orch={bits_orch / scored:.3f} | acc A={correct_a / scored:.3f} "
                f"B={correct_b / scored:.3f} orch={correct_orch / scored:.3f} | "
                f"{(i + 1) / elapsed:.1f} bytes/s",
                flush=True,
            )

    result = {
        "config": vars(args) | {"corpus": str(args.corpus), "report": str(args.report)},
        "scored": scored,
        "disagreements": n_disagree,
        "organ_a": {"bpb": bits_a / scored, "accuracy": correct_a / scored},
        "organ_b": {"bpb": bits_b / scored, "accuracy": correct_b / scored},
        "orchestrator": {"bpb": bits_orch / scored, "accuracy": correct_orch / scored},
        "conditioned_on_disagreement": {
            "n": n_disagree,
            "organ_a_bpb": bits_a_disagree / n_disagree if n_disagree else None,
            "organ_b_bpb": bits_b_disagree / n_disagree if n_disagree else None,
            "orchestrator_bpb": bits_orch_disagree / n_disagree if n_disagree else None,
        },
        "soft_bias_sweep": {
            str(w): {"organ_a_bpb": bits_blend_a[w] / scored, "organ_b_bpb": bits_blend_b[w] / scored}
            for w in bias_weights
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
