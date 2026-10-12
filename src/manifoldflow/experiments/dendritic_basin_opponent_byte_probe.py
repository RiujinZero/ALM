"""Measure a lossless opponent-receptor encoding on raw TinyStories bytes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
    DevelopmentalOpponentAttractorLearner,
    OpponentAssemblyTransitionLearner,
    OpponentAttractorAssemblyLearner,
    OpponentPairDendriteLearner,
    OpponentTransitionLearner,
)
from manifoldflow.experiments.dendritic_basin_online_stream_smoke import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    StreamingNGramControls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--train-bytes", type=int, default=20_000)
    parser.add_argument("--eval-bytes", type=int, default=5_000)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _read_prefix(path: str, size: int) -> np.ndarray:
    with open(path, "rb") as handle:
        return np.frombuffer(handle.read(size), dtype=np.uint8).copy()


def _bits_per_byte(probability: np.ndarray) -> float:
    valid = probability[np.isfinite(probability)]
    return float(np.mean(-np.log2(np.maximum(valid, 1e-300))))


def main() -> None:
    args = parse_args()
    train = _read_prefix(args.train_path, args.train_bytes)
    valid = _read_prefix(args.valid_path, args.eval_bytes)
    learner = OpponentTransitionLearner()
    pair_learner = OpponentPairDendriteLearner()
    assembly_learner = OpponentAssemblyTransitionLearner()
    attractor_learner = OpponentAttractorAssemblyLearner()
    developmental_learner = DevelopmentalOpponentAttractorLearner()
    contextual_learner = ContextualOpponentAttractorLearner()

    start = time.perf_counter()
    train_probability = learner.observe_chunk_probabilities(train, learn=True)
    receptor_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pair_train_probability = pair_learner.observe_chunk_probabilities(train, learn=True)
    pair_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    assembly_train_probability = assembly_learner.observe_chunk_probabilities(
        train, learn=True
    )
    assembly_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    attractor_train_probability = attractor_learner.observe_chunk_probabilities(
        train, learn=True
    )
    attractor_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    developmental_train_probability = developmental_learner.observe_chunk_probabilities(
        train, learn=True
    )
    developmental_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    contextual_train_probability = contextual_learner.observe_chunk_probabilities(
        train, learn=True
    )
    contextual_train_seconds = time.perf_counter() - start
    learner.reset_episode()
    pair_learner.reset_episode()
    assembly_learner.reset_episode()
    attractor_learner.reset_episode()
    developmental_learner.reset_episode()
    contextual_learner.reset_episode()
    start = time.perf_counter()
    eval_probability = learner.observe_chunk_probabilities(valid, learn=False)
    receptor_eval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pair_eval_probability = pair_learner.observe_chunk_probabilities(valid, learn=False)
    pair_eval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    assembly_eval_probability = assembly_learner.observe_chunk_probabilities(
        valid, learn=False
    )
    assembly_eval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    attractor_eval_probability = attractor_learner.observe_chunk_probabilities(
        valid, learn=False
    )
    attractor_eval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    developmental_eval_probability = developmental_learner.observe_chunk_probabilities(
        valid, learn=False
    )
    developmental_eval_seconds = time.perf_counter() - start
    start = time.perf_counter()
    contextual_eval_probability = contextual_learner.observe_chunk_probabilities(
        valid, learn=False
    )
    contextual_eval_seconds = time.perf_counter() - start

    controls = StreamingNGramControls(max_order=3)
    for byte_id in train:
        controls.observe_train(int(byte_id))
    controls.begin_evaluation()
    for byte_id in valid:
        controls.observe_eval(int(byte_id))

    report = {
        "mode": "raw_byte_opponent_receptors",
        "encoding": "eight simultaneous ON/OFF receptor pairs",
        "sensory_dimensions": 16,
        "active_receptors_per_byte": 8,
        "train_bytes": len(train),
        "eval_bytes": len(valid),
        "train_bits_per_byte": _bits_per_byte(train_probability),
        "eval_bits_per_byte": _bits_per_byte(eval_probability),
        "pair_dendrite_train_bits_per_byte": _bits_per_byte(pair_train_probability),
        "pair_dendrite_eval_bits_per_byte": _bits_per_byte(pair_eval_probability),
        "recursive_assembly_train_bits_per_byte": _bits_per_byte(
            assembly_train_probability
        ),
        "recursive_assembly_eval_bits_per_byte": _bits_per_byte(
            assembly_eval_probability
        ),
        "attractor_assembly_train_bits_per_byte": _bits_per_byte(
            attractor_train_probability
        ),
        "attractor_assembly_eval_bits_per_byte": _bits_per_byte(
            attractor_eval_probability
        ),
        "developmental_attractor_train_bits_per_byte": _bits_per_byte(
            developmental_train_probability
        ),
        "developmental_attractor_eval_bits_per_byte": _bits_per_byte(
            developmental_eval_probability
        ),
        "contextual_residual_train_bits_per_byte": _bits_per_byte(
            contextual_train_probability
        ),
        "contextual_residual_eval_bits_per_byte": _bits_per_byte(
            contextual_eval_probability
        ),
        "eval_unigram_bits_per_byte": float(
            controls.eval_bit_sums[0] / controls.eval_count
        ),
        "eval_bigram_bits_per_byte": float(
            controls.eval_bit_sums[1] / controls.eval_count
        ),
        "eval_trigram_bits_per_byte": float(
            controls.eval_bit_sums[2] / controls.eval_count
        ),
        "receptor_train_bytes_per_second": len(train)
        / max(receptor_train_seconds, 1e-12),
        "pair_train_bytes_per_second": len(train) / max(pair_train_seconds, 1e-12),
        "recursive_train_bytes_per_second": len(train)
        / max(assembly_train_seconds, 1e-12),
        "attractor_train_bytes_per_second": len(train)
        / max(attractor_train_seconds, 1e-12),
        "developmental_train_bytes_per_second": len(train)
        / max(developmental_train_seconds, 1e-12),
        "contextual_residual_train_bytes_per_second": len(train)
        / max(contextual_train_seconds, 1e-12),
        "receptor_eval_bytes_per_second": len(valid)
        / max(receptor_eval_seconds, 1e-12),
        "pair_eval_bytes_per_second": len(valid) / max(pair_eval_seconds, 1e-12),
        "recursive_eval_bytes_per_second": len(valid)
        / max(assembly_eval_seconds, 1e-12),
        "attractor_eval_bytes_per_second": len(valid)
        / max(attractor_eval_seconds, 1e-12),
        "developmental_eval_bytes_per_second": len(valid)
        / max(developmental_eval_seconds, 1e-12),
        "contextual_residual_eval_bytes_per_second": len(valid)
        / max(contextual_eval_seconds, 1e-12),
        "embodied_bytes": learner.embodied_bytes,
        "pair_dendrite_embodied_bytes": pair_learner.embodied_bytes,
        "pair_dendrite_count": len(pair_learner.pairs),
        "recursive_assembly_embodied_bytes": assembly_learner.embodied_bytes,
        "live_recursive_assemblies": assembly_learner.live_assembly_count,
        "attractor_assembly_embodied_bytes": attractor_learner.embodied_bytes,
        "live_attractor_assemblies": attractor_learner.live_assembly_count,
        "live_attractor_transitions": attractor_learner.live_transition_count,
        "developmental_embodied_bytes": developmental_learner.embodied_bytes,
        "developmental_assembly_births": developmental_learner.assembly_births,
        "developmental_nursery": developmental_learner.nursery_count,
        "developmental_transitions": developmental_learner.live_transition_count,
        "contextual_residual_embodied_bytes": contextual_learner.embodied_bytes,
        "contextual_residual_births": contextual_learner.context_births,
        "contextual_residual_nursery": len(contextual_learner.context_nursery),
        "contextual_residual_branches": contextual_learner.live_context_count,
        "contextual_residual_synapses": contextual_learner.live_context_synapse_count,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
