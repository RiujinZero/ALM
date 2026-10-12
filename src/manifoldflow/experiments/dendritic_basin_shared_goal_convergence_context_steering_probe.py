"""Context-steering probe: does the native base generate on-topic continuations
when primed with the real conversation context, and does that beat (or lose to)
the abstract content-vector prime the veccond interface currently uses?

The veccond interface primes the base with only the answer's content vector
and discards the conversation. The base, however, was trained on conversational
byte streams, so it is a sequence predictor that can continue from a text
prefix. This probe compares, on held-out development pairs (both sides,
original unanswered prefix), the paired byte-4gram reference-discrimination
margin under:

  native:          no prime (free-run baseline)
  context:         primed with the side's original_context (user query +
                   history + assistant prompt) -- the multi-turn setting
  context_tail256: primed with only the last 256 bytes of that context
                   (the recent user query; tests the local context window)
  context_cross:   primed with a different pair's original_context (control)
  vector_oracle:   primed with the answer's own content vector (current
                   veccond interface, leakage ceiling)
  vector_shuffled: primed with the cross-pair shuffled content vector

Each condition runs in sampled (seeded draws) and greedy (argmax) decoding.
A strongly positive context margin with a ~0 vector margin would mean the
base can be steered by text context (its trained capability) but not by the
abstract vector -- i.e. the missing piece for multi-turn generation is to
condition on the conversation, not on a vector.

Local learning only; no backpropagation; no protected test payload.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.vector_conditioned_base import (
    _normalized_distribution,
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

CONDITIONS = (
    "native",
    "context",
    "context_tail256",
    "context_cross",
    "vector_oracle",
    "vector_shuffled",
)
DECODINGS = ("sampled", "greedy")
ASSISTANT_PROMPT = codec_assay.ASSISTANT_PROMPT
CONTRASTS = {
    "context_minus_native": ("context", "native"),
    "context_tail256_minus_native": ("context_tail256", "native"),
    "context_minus_context_cross": ("context", "context_cross"),
    "vector_oracle_minus_native": ("vector_oracle", "native"),
    "vector_oracle_minus_vector_shuffled": ("vector_oracle", "vector_shuffled"),
    "context_minus_vector_oracle": ("context", "vector_oracle"),
}


def _condition_summary(
    condition_rows: dict[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in condition_rows["native"]]
    values = {}
    for condition, rows in condition_rows.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"condition {condition} rows misaligned")
        values[condition] = [float(row[metric]) for row in rows]
    conditions = {
        condition: codec_assay._mean_ci(values[condition], seed=seed, resamples=resamples)
        for condition in condition_rows
    }
    contrasts = {}
    for contrast_index, (name, (positive, negative)) in enumerate(CONTRASTS.items()):
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    return {"conditions": conditions, "contrasts": contrasts}


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def generate_after_prime_bytes(
    cortex: Any,
    prime_bytes: bytes,
    *,
    horizon: int,
    uniform_draws: np.ndarray | None = None,
) -> tuple[bytes, str]:
    """Prime with arbitrary bytes (learn=False) then autoregressively generate,
    mirroring generate_from_vector's sampling loop."""
    import hashlib

    if horizon < 1:
        raise ValueError("horizon must be positive")
    draws = None
    if uniform_draws is not None:
        draws = np.asarray(uniform_draws, dtype=np.float64)
        if draws.shape != (horizon,):
            raise ValueError("uniform_draws must have shape (horizon,)")
    cortex.reset_episode()
    if prime_bytes:
        cortex.observe_chunk_probabilities(
            np.frombuffer(prime_bytes, dtype=np.uint8), learn=False
        )
    generated = bytearray()
    digest = hashlib.sha256()
    for position in range(horizon):
        law = _normalized_distribution(cortex.predict_next_distribution())
        digest.update(law.tobytes())
        if draws is None:
            symbol = int(np.argmax(law))
        else:
            symbol = min(
                int(np.searchsorted(np.cumsum(law), draws[position], side="right")),
                255,
            )
        generated.append(symbol)
        cortex.advance_observation(symbol, learn=False)
    cortex.reset_episode()
    return bytes(generated), digest.hexdigest()


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

    # -- development rows: evaluation partition, original prefix -------------
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.partition != "evaluation":
            continue
        for side_name in ("a", "b"):
            side = pair.side_a if side_name == "a" else pair.side_b
            context = side.original_context
            if context is None:
                continue
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "side": side_name,
                    "context": context,
                    "reference": side.response_window,
                    "full_response": side.full_response,
                    "source_label": pair.source_label,
                    "turn_count": context.count(ASSISTANT_PROMPT),
                }
            )
    if not rows:
        raise ValueError("no evaluation rows with an original context")

    # -- oracle / shuffled content codes, aligned to rows --------------------
    pair_order = list(dict.fromkeys(row["pair_id"] for row in rows))
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_order)}
    shuffle = codec_assay.cross_pair_shuffle_indices(len(pair_order), args.control_seed)
    # row r's shuffled code = code of the row at position shuffle[2*p + side]
    row_of_pair_position: dict[tuple[int, str], int] = {}
    for position, row in enumerate(rows):
        p = pair_index[row["pair_id"]]
        row_of_pair_position[(p, row["side"])] = position
    oracle_codes: list[np.ndarray] = []
    for row in rows:
        oracle_codes.append(
            codec.encode(
                row["full_response"],
                role=codec_assay.ASSISTANT_ROLE,
                source=row["source_label"],
            ).content
        )
    shuffled_codes: list[np.ndarray] = []
    for row in rows:
        p = pair_index[row["pair_id"]]
        side_slot = 0 if row["side"] == "a" else 1
        shuffled_position = int(shuffle[2 * p + side_slot])
        shuffled_codes.append(oracle_codes[shuffled_position])

    # -- generation per condition --------------------------------------------
    condition_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    pair_texts = []
    for pair_id in pair_order:
        p = pair_index[pair_id]
        side_payloads: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for side_name in ("a", "b"):
            side_slot = 0 if side_name == "a" else 1
            row_position = row_of_pair_position[(p, side_name)]
            row = rows[row_position]
            draws = codec_assay._seeded_draws(
                f"context-steer|{pair_id}|{side_name}",
                seed=args.sample_seed,
                horizon=args.generation_horizon,
            )
            per_condition: dict[str, dict[str, dict[str, Any]]] = {}
            for condition in CONDITIONS:
                if condition == "native":
                    sampled = generate_unprimed(
                        base, horizon=args.generation_horizon, uniform_draws=draws
                    )
                    greedy = generate_unprimed(
                        base, horizon=args.generation_horizon, uniform_draws=None
                    )
                    sampled_data, sampled_digest = sampled
                    greedy_data, greedy_digest = greedy
                elif condition == "context":
                    sampled_data, sampled_digest = generate_after_prime_bytes(
                        base, row["context"],
                        horizon=args.generation_horizon, uniform_draws=draws,
                    )
                    greedy_data, greedy_digest = generate_after_prime_bytes(
                        base, row["context"],
                        horizon=args.generation_horizon, uniform_draws=None,
                    )
                elif condition == "context_tail256":
                    prime = row["context"][-256:]
                    sampled_data, sampled_digest = generate_after_prime_bytes(
                        base, prime,
                        horizon=args.generation_horizon, uniform_draws=draws,
                    )
                    greedy_data, greedy_digest = generate_after_prime_bytes(
                        base, prime,
                        horizon=args.generation_horizon, uniform_draws=None,
                    )
                elif condition == "context_cross":
                    other_position = int(shuffle[2 * p + side_slot])
                    other_context = rows[other_position]["context"]
                    sampled_data, sampled_digest = generate_after_prime_bytes(
                        base, other_context,
                        horizon=args.generation_horizon, uniform_draws=draws,
                    )
                    greedy_data, greedy_digest = generate_after_prime_bytes(
                        base, other_context,
                        horizon=args.generation_horizon, uniform_draws=None,
                    )
                elif condition == "vector_oracle":
                    sampled = generate_from_vector(
                        base, oracle_codes[row_position],
                        horizon=args.generation_horizon, uniform_draws=draws,
                    )
                    greedy = generate_from_vector(
                        base, oracle_codes[row_position],
                        horizon=args.generation_horizon, uniform_draws=None,
                    )
                    sampled_data, sampled_digest = sampled.data, sampled.probability_sha256
                    greedy_data, greedy_digest = greedy.data, greedy.probability_sha256
                elif condition == "vector_shuffled":
                    sampled = generate_from_vector(
                        base, shuffled_codes[row_position],
                        horizon=args.generation_horizon, uniform_draws=draws,
                    )
                    greedy = generate_from_vector(
                        base, shuffled_codes[row_position],
                        horizon=args.generation_horizon, uniform_draws=None,
                    )
                    sampled_data, sampled_digest = sampled.data, sampled.probability_sha256
                    greedy_data, greedy_digest = greedy.data, greedy.probability_sha256
                else:
                    raise ValueError(f"unknown condition {condition}")
                per_condition[condition] = {
                    "sampled": {"data": sampled_data, "digest": sampled_digest},
                    "greedy": {"data": greedy_data, "digest": greedy_digest},
                }
            side_payloads[side_name] = per_condition

        texts = {
            side_name: {
                "reference_text": _text(
                    rows[row_of_pair_position[(pair_index[pair_id], side_name)]]["reference"]
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
        pair_texts.append({"pair_id": pair_id, "sides": texts})

        for condition in CONDITIONS:
            row = {"pair_id": pair_id}
            for decoding in DECODINGS:
                idx_a = row_of_pair_position[(pair_index[pair_id], "a")]
                idx_b = row_of_pair_position[(pair_index[pair_id], "b")]
                pair_margin = _paired_reference_discrimination(
                    side_payloads["a"][condition][decoding]["data"],
                    side_payloads["b"][condition][decoding]["data"],
                    rows[idx_a]["reference"],
                    rows[idx_b]["reference"],
                    dimensions=1024,
                )
                row[f"{decoding}_margin"] = pair_margin["correct_minus_crossed_margin"]
                if decoding == "sampled":
                    diagnostics = {
                        side: codec_assay._generation_diagnostics(
                            side_payloads[side][condition]["sampled"]["data"]
                        )
                        for side in ("a", "b")
                    }
                    row["sampled_printable"] = float(
                        np.mean([diagnostics[s]["printable_fraction"] for s in ("a", "b")])
                    )
                    row["sampled_utf8"] = float(
                        np.mean([1.0 if diagnostics[s]["valid_utf8"] else 0.0 for s in ("a", "b")])
                    )
                    row["sampled_maximum_run"] = max(
                        diagnostics[s]["maximum_identical_byte_run"] for s in ("a", "b")
                    )
            row["turn_count"] = max(
                rows[row_of_pair_position[(pair_index[pair_id], s)]]["turn_count"] for s in ("a", "b")
            )
            row["context_len"] = len(
                rows[row_of_pair_position[(pair_index[pair_id], "a")]]["context"]
            )
            condition_rows[condition].append(row)

    # -- summaries -----------------------------------------------------------
    summaries: dict[str, Any] = {}
    for decoding_index, decoding in enumerate(DECODINGS):
        summaries[decoding] = _condition_summary(
            condition_rows,
            metric=f"{decoding}_margin",
            seed=args.bootstrap_seed + decoding_index * 100,
            resamples=args.bootstrap_resamples,
        )
    summaries["health"] = {
        condition: {
            "mean_sampled_printable_fraction": float(
                np.mean([row["sampled_printable"] for row in condition_rows[condition]])
            ),
            "mean_sampled_utf8": float(
                np.mean([row["sampled_utf8"] for row in condition_rows[condition]])
            ),
            "mean_sampled_maximum_run": float(
                np.mean([row["sampled_maximum_run"] for row in condition_rows[condition]])
            ),
        }
        for condition in CONDITIONS
    }
    # multi-turn depth: margin vs number of prior assistant turns
    depth: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    depth_counts: dict[str, list[int]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for row in condition_rows[condition]:
            depth[condition].append(row["greedy_margin"])
            depth_counts[condition].append(row["turn_count"])
    summaries["greedy_margin_by_turn_count"] = {
        condition: {
            str(tc): float(np.mean(vals))
            for tc, vals in (
                (tc, [m for m, c in zip(depth[condition], depth_counts[condition]) if c == tc])
                for tc in sorted(set(depth_counts[condition]))
            )
            if vals
        }
        for condition in CONDITIONS
    }

    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    return {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_context_steering_probe",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "free_generation_computed": True,
            "protected_test_payload_accessed": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "generation_horizon": args.generation_horizon,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "content_dimensions": args.content_dimensions,
            "active_content_features": args.active_content_features,
        },
        "counts": {
            "pairs": len(pair_order),
            "sides": len(rows),
        },
        "summaries": summaries,
        "pair_texts": pair_texts,
        "wall_seconds": elapsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-horizon", type=int, default=128)
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
