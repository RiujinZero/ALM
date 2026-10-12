"""Generate raw bytes from a frozen dendritic opponent-cortex checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
import zlib
from math import log2
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    OpponentAttractorAssemblyLearner,
)

GOAL_ROUTE_COUNTERS = (
    "goal_route_context_calls",
    "goal_route_proposals",
    "goal_route_denials",
    "goal_route_switches",
    "goal_route_flow_uses",
)


def _goal_route_telemetry(
    learner: OpponentAttractorAssemblyLearner,
) -> dict[str, int]:
    return {name: int(getattr(learner, name, 0)) for name in GOAL_ROUTE_COUNTERS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", default="Once upon a time")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-bytes", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--goal-apical-weight",
        type=float,
        default=None,
        help="override a goal-conditioned checkpoint's gain in memory",
    )
    parser.add_argument("--stop-at-story-boundary", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sample_byte(
    distribution: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    probability = np.asarray(distribution, dtype=np.float64).reshape(-1)
    if probability.shape != (256,):
        raise ValueError("the output basin must contain 256 byte probabilities")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0 <= top_k <= 256:
        raise ValueError("top_k must be in [0, 256]")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0.0 or top_k == 1:
        return int(np.argmax(probability))
    logits = np.log(np.maximum(probability, 1e-300)) / temperature
    if 0 < top_k < 256:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full(256, -np.inf, dtype=np.float64)
        masked[keep] = logits[keep]
        logits = masked
    logits -= np.max(logits)
    probability = np.exp(logits)
    probability /= probability.sum()
    if top_p < 1.0:
        order = np.argsort(probability)[::-1]
        cumulative = np.cumsum(probability[order])
        keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        keep = order[:keep_count]
        nucleus = np.zeros(256, dtype=np.float64)
        nucleus[keep] = probability[keep]
        probability = nucleus / nucleus.sum()
    return int(rng.choice(256, p=probability))


def generate_bytes(
    learner: OpponentAttractorAssemblyLearner,
    prompt: bytes,
    *,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
    seed: int,
    stop_at_story_boundary: bool = False,
) -> bytes:
    if max_new_bytes < 0:
        raise ValueError("max_new_bytes must be non-negative")
    learner.reset_episode()
    if prompt:
        learner.observe_chunk_probabilities(
            np.frombuffer(prompt, dtype=np.uint8), learn=False
        )
    rng = np.random.default_rng(seed)
    generated = bytearray()
    boundary = getattr(learner, "story_boundary", b"<|endoftext|>\n")
    for _ in range(max_new_bytes):
        byte_id = _sample_byte(
            learner.predict_next_distribution(),
            rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(byte_id)
        learner.advance_observation(byte_id, learn=False)
        if stop_at_story_boundary and generated.endswith(boundary):
            break
    return bytes(generated)


def _generation_diagnostics(generated: bytes, boundary: bytes) -> dict[str, object]:
    if not generated:
        return {
            "valid_utf8": True,
            "utf8_replacement_characters": 0,
            "printable_ascii_fraction": 1.0,
            "empirical_byte_entropy": 0.0,
            "zlib_compression_ratio": 1.0,
            "longest_identical_byte_run": 0,
            "distinct_ngram_fraction": {},
            "longest_repeated_block": {
                "block_bytes": 0,
                "repetitions": 0,
                "repeated_bytes": 0,
            },
            "story_boundary_found": False,
        }
    try:
        generated.decode("utf-8", errors="strict")
        valid_utf8 = True
    except UnicodeDecodeError:
        valid_utf8 = False
    decoded = generated.decode("utf-8", errors="replace")
    printable = sum(
        byte in (9, 10, 13) or 32 <= byte <= 126 for byte in generated
    )
    counts = np.bincount(np.frombuffer(generated, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(generated)
    entropy = -sum(float(p) * log2(float(p)) for p in probabilities)
    longest_run = 1
    current_run = 1
    for previous, current in zip(generated, generated[1:], strict=False):
        if current == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    distinct = {}
    for order in (2, 3, 4, 8, 16):
        count = max(0, len(generated) - order + 1)
        distinct[str(order)] = (
            len({generated[i : i + order] for i in range(count)}) / count
            if count
            else 1.0
        )
    repeated_block = {"block_bytes": 0, "repetitions": 0, "repeated_bytes": 0}
    for width in range(1, min(64, len(generated) // 2) + 1):
        for start in range(0, len(generated) - 2 * width + 1):
            block = generated[start : start + width]
            repetitions = 1
            position = start + width
            while generated[position : position + width] == block:
                repetitions += 1
                position += width
            repeated_bytes = repetitions * width
            if repetitions >= 2 and repeated_bytes > repeated_block["repeated_bytes"]:
                repeated_block = {
                    "block_bytes": width,
                    "repetitions": repetitions,
                    "repeated_bytes": repeated_bytes,
                }
    return {
        "valid_utf8": valid_utf8,
        "utf8_replacement_characters": decoded.count("\ufffd"),
        "printable_ascii_fraction": printable / len(generated),
        "empirical_byte_entropy": entropy,
        "zlib_compression_ratio": len(zlib.compress(generated)) / len(generated),
        "longest_identical_byte_run": longest_run,
        "distinct_ngram_fraction": distinct,
        "longest_repeated_block": repeated_block,
        "story_boundary_found": boundary in generated,
    }


def _summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "boundary_completion_fraction": 0.0,
            "valid_utf8_fraction": 0.0,
            "mean_generated_bytes": 0.0,
            "mean_printable_ascii_fraction": 0.0,
            "mean_zlib_compression_ratio": 0.0,
            "mean_distinct_4gram_fraction": 0.0,
            "total_goal_route_switches": 0,
        }
    diagnostics = [sample["diagnostics"] for sample in samples]
    return {
        "sample_count": len(samples),
        "boundary_completion_fraction": float(
            np.mean([item["story_boundary_found"] for item in diagnostics])
        ),
        "valid_utf8_fraction": float(
            np.mean([item["valid_utf8"] for item in diagnostics])
        ),
        "mean_generated_bytes": float(
            np.mean([sample["generated_bytes"] for sample in samples])
        ),
        "mean_printable_ascii_fraction": float(
            np.mean([item["printable_ascii_fraction"] for item in diagnostics])
        ),
        "mean_zlib_compression_ratio": float(
            np.mean([item["zlib_compression_ratio"] for item in diagnostics])
        ),
        "mean_distinct_4gram_fraction": float(
            np.mean(
                [item["distinct_ngram_fraction"]["4"] for item in diagnostics]
            )
        ),
        "total_goal_route_switches": sum(
            int(sample.get("goal_route_switches", 0)) for sample in samples
        ),
    }


def _read_checkpoint(path: Path) -> OpponentAttractorAssemblyLearner:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    learner = payload.get("learner") if isinstance(payload, dict) else payload
    if not isinstance(learner, OpponentAttractorAssemblyLearner):
        raise TypeError("checkpoint does not contain an opponent byte cortex")
    # Older ALM checkpoints predate the optional transient binding organ.  Its
    # absence means "disabled", but unpickling does not run the newer
    # constructor that supplies the attribute.
    if not hasattr(learner, "binding_index"):
        learner.binding_index = None
    return learner


def main() -> None:
    args = parse_args()
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_bytes()
    else:
        prompt = args.prompt.encode("utf-8")
    learner = _read_checkpoint(args.checkpoint)
    checkpoint_goal_apical_weight = getattr(learner, "goal_apical_weight", None)
    if args.goal_apical_weight is not None:
        if checkpoint_goal_apical_weight is None:
            raise ValueError("checkpoint does not contain a goal-conditioned cortex")
        if args.goal_apical_weight < 0.0:
            raise ValueError("goal apical weight must be non-negative")
        learner.goal_apical_weight = args.goal_apical_weight
    checkpoint_embodied_bytes = learner.embodied_bytes
    samples = []
    boundary = getattr(learner, "story_boundary", b"<|endoftext|>\n")
    for index in range(args.samples):
        telemetry_before = _goal_route_telemetry(learner)
        generated = generate_bytes(
            learner,
            prompt,
            max_new_bytes=args.max_new_bytes,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + index,
            stop_at_story_boundary=args.stop_at_story_boundary,
        )
        samples.append(
            {
                "seed": args.seed + index,
                "generated_bytes": len(generated),
                "generated_hex": generated.hex(),
                "text": (prompt + generated).decode("utf-8", errors="backslashreplace"),
                "diagnostics": _generation_diagnostics(generated, boundary),
                "goal_route_telemetry": {
                    name: value - telemetry_before[name]
                    for name, value in _goal_route_telemetry(learner).items()
                },
                "goal_route_switches": (
                    getattr(learner, "goal_route_switches", 0)
                    - telemetry_before["goal_route_switches"]
                ),
            }
        )
    learner.reset_episode()
    report = {
        "checkpoint": str(args.checkpoint),
        "learner_type": type(learner).__name__,
        "checkpoint_embodied_bytes": checkpoint_embodied_bytes,
        "reset_embodied_bytes": learner.embodied_bytes,
        "checkpoint_goal_apical_weight": checkpoint_goal_apical_weight,
        "effective_goal_apical_weight": getattr(
            learner, "goal_apical_weight", None
        ),
        "prompt_hex": prompt.hex(),
        "prompt_text": prompt.decode("utf-8", errors="backslashreplace"),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "samples": samples,
        "summary": _summarize_samples(samples),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
