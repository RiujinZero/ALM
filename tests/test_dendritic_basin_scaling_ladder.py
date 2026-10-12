"""Tests for resumable ALM scaling-ladder aggregation."""

from __future__ import annotations

import argparse
import json

import pytest

from manifoldflow.experiments.dendritic_basin_scaling_ladder import (
    _run_prefix,
    _valid_report,
    summarize_reports,
)


def _report(seed: int, bpb: float) -> dict:
    return {
        "alm_class": "ALM-16MB",
        "train_bytes": 10_000_000,
        "eval_bytes": 1_000_000,
        "config": {"seed": seed},
        "eval_model_bits_per_byte": bpb,
        "train_bytes_per_second": 1500.0 + seed,
        "serialized_alm_bytes": 15_000_000,
        "births_per_kilobyte": 0.2,
        "prunes_per_kilobyte": 0.1,
        "corpus_bytes_per_alm_byte": 2.0 / 3.0,
        "live_relations": 1000,
        "population": {"births": 2000},
        "anatomy": {
            "mean_branch_age_bytes": 5_000_000.0,
            "depth_histogram": {"1": 500, "2": 500},
            "lifecycle_histogram": {
                "juvenile": 100,
                "consolidated": 700,
                "dormant": 200,
            },
        },
    }


def test_scaling_ladder_validates_and_aggregates_reports(tmp_path) -> None:
    paths = []
    for seed, bpb in enumerate((3.7, 3.8, 3.6)):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(_report(seed, bpb)), encoding="utf-8")
        paths.append(path)
        assert _valid_report(
            path, train_bytes=10_000_000, eval_bytes=1_000_000, seed=seed
        )

    summary = summarize_reports(paths)
    aggregate = summary["classes"]["ALM-16MB"]
    assert aggregate["seed_count"] == 3
    assert aggregate["eval_bpb_mean"] == pytest.approx(3.7)
    assert aggregate["eval_bpb_sample_sd"] == pytest.approx(0.1)
    assert aggregate["live_birth_survival_fraction_mean"] == 0.5
    assert summary["runs"][0]["contextual_residual"] is False


def test_run_prefix_is_one_path_component() -> None:
    assert _run_prefix("alm_phase13_compiled") == "alm_phase13_compiled"
    with pytest.raises(argparse.ArgumentTypeError):
        _run_prefix("legacy/run")
