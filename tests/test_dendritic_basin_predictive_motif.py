from __future__ import annotations

from manifoldflow.experiments.dendritic_basin_predictive_motif_probe import (
    run_probe,
)


def test_one_predictive_motif_organ_generalizes_across_domains() -> None:
    results = run_probe()

    assert [row["task"] for row in results] == [
        "delayed_symbols",
        "occluded_ring",
        "time_warped_audio",
        "spider_cross_modal",
    ]
    assert all(float(row["motif_gain_bits"]) > 0.15 for row in results)
    assert all(int(row["pruned_event_branches"]) > 0 for row in results)
    assert all(int(row["motif_assemblies"]) > 0 for row in results)
    assert all(int(row["learned_boundaries"]) > 0 for row in results)
