from __future__ import annotations

import pickle

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    OpponentAttractorAssemblyLearner,
)
from manifoldflow.experiments.dendritic_basin_generate import (
    _generation_diagnostics,
    _read_checkpoint,
    _summarize_samples,
    generate_bytes,
)
from manifoldflow.experiments.dendritic_basin_goal_generation_ablation import (
    _first_divergence,
)


def _trained_cycle() -> OpponentAttractorAssemblyLearner:
    learner = OpponentAttractorAssemblyLearner()
    stream = np.resize(np.frombuffer(b"abc", dtype=np.uint8), 3_000)
    learner.observe_chunk_probabilities(stream, learn=True)
    return learner


def test_next_distribution_is_frozen_and_normalized() -> None:
    learner = _trained_cycle()
    learner.reset_episode()
    learner.advance_observation(ord("a"), learn=False)
    before = pickle.dumps(learner, protocol=5)
    distribution = learner.predict_next_distribution()

    assert distribution.shape == (256,)
    assert np.isclose(distribution.sum(), 1.0)
    assert pickle.dumps(learner, protocol=5) == before


def test_greedy_generation_closes_a_learned_byte_cycle() -> None:
    learner = _trained_cycle()
    generated = generate_bytes(
        learner,
        b"a",
        max_new_bytes=8,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        seed=0,
    )

    assert generated == b"bcabcabc"


def test_generation_checkpoint_loader_accepts_training_payload(tmp_path) -> None:
    learner = _trained_cycle()
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(pickle.dumps({"learner": learner, "report": {}}))

    loaded = _read_checkpoint(checkpoint)

    assert isinstance(loaded, OpponentAttractorAssemblyLearner)
    assert loaded.embodied_bytes == learner.embodied_bytes


def test_generation_diagnostics_expose_repetition_and_invalid_utf8() -> None:
    diagnostics = _generation_diagnostics(b"aaaa\xff", b"<end>")

    assert diagnostics["valid_utf8"] is False
    assert diagnostics["utf8_replacement_characters"] == 1
    assert diagnostics["longest_identical_byte_run"] == 4
    assert diagnostics["zlib_compression_ratio"] > 0.0
    assert diagnostics["story_boundary_found"] is False

    repeated = _generation_diagnostics(b"abcabcabcx", b"<end>")
    assert repeated["longest_repeated_block"] == {
        "block_bytes": 3,
        "repetitions": 3,
        "repeated_bytes": 9,
    }


def test_sample_summary_reports_completion_and_repetition_means() -> None:
    samples = [
        {
            "generated_bytes": 10,
            "diagnostics": {
                "story_boundary_found": True,
                "valid_utf8": True,
                "printable_ascii_fraction": 1.0,
                "zlib_compression_ratio": 0.5,
                "distinct_ngram_fraction": {"4": 0.8},
            },
        },
        {
            "generated_bytes": 20,
            "diagnostics": {
                "story_boundary_found": False,
                "valid_utf8": True,
                "printable_ascii_fraction": 0.9,
                "zlib_compression_ratio": 0.7,
                "distinct_ngram_fraction": {"4": 0.6},
            },
        },
    ]

    summary = _summarize_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["boundary_completion_fraction"] == 0.5
    assert summary["valid_utf8_fraction"] == 1.0
    assert summary["mean_generated_bytes"] == 15.0
    assert summary["mean_printable_ascii_fraction"] == 0.95
    assert summary["mean_zlib_compression_ratio"] == 0.6
    assert summary["mean_distinct_4gram_fraction"] == 0.7


def test_first_generation_divergence_handles_equal_and_prefix_outputs() -> None:
    assert _first_divergence(b"same", b"same") is None
    assert _first_divergence(b"abcd", b"abXd") == 2
    assert _first_divergence(b"abc", b"abcde") == 3
