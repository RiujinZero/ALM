"""Focused tests for the core/edge native-generation probe."""

from __future__ import annotations

import pytest

from manifoldflow.experiments import (
    dendritic_basin_future_core_edge_native_generation as probe,
)


class _FakeWriter:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBank:
    def __init__(self) -> None:
        self.local = _FakeWriter("local")
        self.coupled = _FakeWriter("coupled")


def _pair_row(pair_id: str, partition: str, margin: float) -> dict:
    return {
        "pair_id": pair_id,
        "partition": partition,
        "sampled_margin": margin,
        "greedy_margin": margin + 0.1,
        "sampled_printable": 0.99,
        "sampled_utf8": 1.0,
        "sampled_maximum_run": 3,
    }


def _rows_by_arm(
    margins: dict[str, list[float]], partition: str = "evaluation"
) -> dict:
    pair_count = len(next(iter(margins.values())))
    pair_ids = [f"p{index}" for index in range(pair_count)]
    return {
        arm: [_pair_row(pair_id, partition, margin)
              for pair_id, margin in zip(pair_ids, arm_margins, strict=True)]
        for arm, arm_margins in margins.items()
    }


def test_probe_assignments_covers_every_arm_with_expected_routes() -> None:
    bank = _FakeBank()
    assignments = probe.probe_assignments(
        bank,
        oracle=(1,),
        mixture=(2,),
        core=(3,),
        core_foreign=(4,),
        edge_swapped=(5,),
        gain=1.0,
    )
    assert tuple(assignments) == probe.ARM_NAMES
    assert assignments["native"].writer is None
    assert assignments["local_decoder"].writer is bank.local
    assert assignments["local_decoder"].state == ()
    for arm in ("oracle_code_coupled", "mixture_point_coupled", "core_coupled",
                "core_foreign_coupled", "edge_swapped_coupled"):
        assert assignments[arm].writer is bank.coupled
    assert assignments["oracle_code_coupled"].state == (1,)
    assert assignments["core_coupled"].state == (3,)
    assert assignments["core_foreign_coupled"].state == (4,)
    assert assignments["edge_swapped_coupled"].state == (5,)


def test_paired_contrasts_compute_paired_differences() -> None:
    rows = _rows_by_arm(
        {
            "native": [0.0, 0.0],
            "local_decoder": [0.0, 0.0],
            "oracle_code_coupled": [1.0, 1.0],
            "mixture_point_coupled": [0.5, 0.5],
            "core_coupled": [0.25, 0.25],
            "core_foreign_coupled": [-0.25, -0.25],
            "edge_swapped_coupled": [0.25, 0.25],
        }
    )
    result = probe.paired_contrasts(
        rows, metric="sampled_margin", seed=1, resamples=200
    )
    assert result["arms"]["core_coupled"]["mean"] == pytest.approx(0.25)
    assert result["contrasts"]["core_minus_mixture"]["mean"] == pytest.approx(-0.25)
    assert result["contrasts"]["core_minus_foreign"]["mean"] == pytest.approx(0.5)
    assert result["contrasts"]["core_minus_native"]["mean"] == pytest.approx(0.25)
    assert result["contrasts"]["oracle_minus_native"]["mean"] == pytest.approx(1.0)


def test_paired_contrasts_reject_misaligned_rows() -> None:
    rows = _rows_by_arm(
        {
            "native": [0.0],
            "local_decoder": [0.0],
            "oracle_code_coupled": [1.0],
            "mixture_point_coupled": [0.5],
            "core_coupled": [0.25],
            "core_foreign_coupled": [-0.25],
            "edge_swapped_coupled": [0.25],
        }
    )
    rows["core_coupled"][0]["pair_id"] = "different"
    with pytest.raises(AssertionError):
        probe.paired_contrasts(rows, metric="sampled_margin", seed=1, resamples=200)


def test_output_health_flags_degeneration_relative_to_native() -> None:
    healthy = _rows_by_arm({arm: [0.0] for arm in probe.ARM_NAMES})
    health = probe.output_health(healthy)
    assert all(entry["meets_nondegeneration_rule"] for entry in health.values())

    degraded = _rows_by_arm({arm: [0.0] for arm in probe.ARM_NAMES})
    degraded["core_coupled"][0]["sampled_printable"] = 0.50
    degraded["core_coupled"][0]["sampled_maximum_run"] = 32
    health = probe.output_health(degraded)
    assert not health["core_coupled"]["meets_nondegeneration_rule"]
    assert health["native"]["meets_nondegeneration_rule"]


def test_manifest_includes_probe_locked_chain_files() -> None:
    paths = probe.implementation_hashes()
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_core_edge_native_generation.py"
        in paths
    )
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_turn_codec_ablation.py"
        in paths
    )
    assert (
        "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_sweep_ablation.py"
        in paths
    )
