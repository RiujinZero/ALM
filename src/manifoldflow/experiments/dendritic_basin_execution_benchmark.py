"""Benchmark causally equivalent ALM execution backends from one checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.online_stream import (
    OnlineDendriticStreamLearner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("stream", type=Path)
    parser.add_argument("--bytes", type=int, default=20_000)
    parser.add_argument(
        "--competition-backends",
        nargs="+",
        choices=("scalar", "batched", "compiled"),
        default=("batched", "compiled"),
    )
    parser.add_argument(
        "--response-backends",
        nargs="+",
        choices=("scalar", "batched", "compiled"),
        default=None,
    )
    parser.add_argument(
        "--adaptation-backends",
        nargs="+",
        choices=("scalar", "compiled"),
        default=None,
    )
    parser.add_argument(
        "--adaptation-rng",
        choices=("global", "counter"),
        default=None,
    )
    parser.add_argument(
        "--loop-backends",
        nargs="+",
        choices=("step", "chunk"),
        default=("step",),
    )
    parser.add_argument("--learn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bytes <= 0:
        raise ValueError("--bytes must be positive")
    with args.stream.open("rb") as source:
        tokens = source.read(args.bytes)
    if len(tokens) != args.bytes:
        raise ValueError("stream is shorter than requested benchmark")

    source_learner = OnlineDendriticStreamLearner.load_checkpoint(args.checkpoint)
    rows = []
    response_backends = args.response_backends or (
        source_learner.config.response_backend,
    )
    adaptation_backends = args.adaptation_backends or (
        source_learner.config.adaptation_backend,
    )
    for (
        competition_backend,
        response_backend,
        adaptation_backend,
        loop_backend,
    ) in product(
        args.competition_backends,
        response_backends,
        adaptation_backends,
        args.loop_backends,
    ):
        config = replace(
            source_learner.config,
            competition_backend=competition_backend,
            response_backend=response_backend,
            adaptation_backend=adaptation_backend,
            adaptation_rng=(
                args.adaptation_rng or source_learner.config.adaptation_rng
            ),
        )
        state = source_learner.state_dict()
        state["config"] = asdict(config)
        learner = OnlineDendriticStreamLearner(config)
        learner.load_state_dict(state)
        learner.reset_episode()
        learner.warm_execution_backend()

        bit_sum = 0.0
        scored = 0
        started = time.perf_counter()
        if loop_backend == "chunk":
            observed, _base = learner.observe_chunk_probabilities(
                np.frombuffer(tokens, dtype=np.uint8), learn=args.learn
            )
            valid = observed[np.isfinite(observed)]
            bit_sum = float(-np.log2(np.maximum(valid, 1e-12)).sum())
            scored = len(valid)
        else:
            for token_id in tokens:
                step = learner.observe(token_id, learn=args.learn)
                if step.observed_probability is not None:
                    bit_sum -= math.log2(max(step.observed_probability, 1e-12))
                    scored += 1
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "competition_backend": competition_backend,
                "response_backend": response_backend,
                "adaptation_backend": adaptation_backend,
                "adaptation_rng": config.adaptation_rng,
                "loop_backend": loop_backend,
                "effective_adaptation_backend": (
                    learner.effective_adaptation_backend
                ),
                "effective_response_backend": learner.effective_response_backend,
                "effective_competition_backend": (
                    learner.effective_competition_backend
                ),
                "bytes": len(tokens),
                "learn": args.learn,
                "live_relations_at_start": source_learner.live_relation_count,
                "seconds": elapsed,
                "bytes_per_second": len(tokens) / elapsed,
                "bits_per_byte": bit_sum / max(scored, 1),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
