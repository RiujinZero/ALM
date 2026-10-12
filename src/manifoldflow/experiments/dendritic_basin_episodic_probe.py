"""Compare local and content-addressable episodic raw-byte organs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
    EpisodicOpponentAttractorLearner,
    SequenceCortexOpponentLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=1_000_000)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _bpb(probability: np.ndarray) -> float:
    finite = probability[np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))


def main() -> None:
    args = parse_args()
    with open(args.train_path, "rb") as handle:
        train = np.frombuffer(handle.read(args.train_bytes), dtype=np.uint8)
    with open(args.valid_path, "rb") as handle:
        valid = np.frombuffer(handle.read(args.eval_bytes), dtype=np.uint8)
    rows: dict[str, dict[str, float | int]] = {}
    for name, learner in (
        ("local", ContextualOpponentAttractorLearner()),
        ("sequence", SequenceCortexOpponentLearner()),
        ("episodic", EpisodicOpponentAttractorLearner()),
    ):
        started = time.perf_counter()
        train_probability = learner.observe_chunk_probabilities(train)
        train_seconds = time.perf_counter() - started
        learner.reset_episode()
        started = time.perf_counter()
        eval_probability = learner.observe_chunk_probabilities(valid, learn=False)
        eval_seconds = time.perf_counter() - started
        rows[name] = {
            "train_bits_per_byte": _bpb(train_probability),
            "eval_bits_per_byte": _bpb(eval_probability),
            "train_bytes_per_second": len(train) / max(train_seconds, 1e-12),
            "eval_bytes_per_second": len(valid) / max(eval_seconds, 1e-12),
            "embodied_bytes": learner.embodied_bytes,
        }
        if isinstance(learner, EpisodicOpponentAttractorLearner):
            rows[name].update(
                {
                    "content_assemblies": learner.live_content_count,
                    "episodic_branches": learner.live_episode_count,
                    "episodic_synapses": learner.live_episode_synapse_count,
                    "utility_gated_branches": sum(
                        utility > learner.minimum_utility_bits
                        for utility in learner.episodic_utility.values()
                    ),
                }
            )
    report = {
        "train_bytes": len(train),
        "eval_bytes": len(valid),
        "local": rows["local"],
        "sequence": rows["sequence"],
        "episodic": rows["episodic"],
        "sequence_gain_bits_per_byte": (
            rows["local"]["eval_bits_per_byte"]
            - rows["sequence"]["eval_bits_per_byte"]
        ),
        "episodic_gain_bits_per_byte": (
            rows["sequence"]["eval_bits_per_byte"]
            - rows["episodic"]["eval_bits_per_byte"]
        ),
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
