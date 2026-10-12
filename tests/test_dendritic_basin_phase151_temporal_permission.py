import numpy as np

from manifoldflow.experiments import (
    dendritic_basin_phase151_temporal_geometric_permission as phase151,
)


def test_delayed_external_utility_reinforces_fading_pathway_trace() -> None:
    tissue = phase151.TemporalGeometricPermission(4, half_lives=(8.0, 32.0))
    query = np.asarray([1.0, 0.5, 0.0, 0.0])

    tissue.participate(query, receptor=3, amount=1.0)
    for _ in range(8):
        tissue.advance()
    tissue.confirm(0.5)

    assert tissue.value(query, 3) > 0.0
    assert tissue.permission(query, 3) > 0.0
    assert tissue.value(np.asarray([0.0, 0.0, 1.0, 1.0]), 3) == 0.0


def test_delayed_harm_can_veto_a_previously_helpful_geometry() -> None:
    tissue = phase151.TemporalGeometricPermission(
        3, half_lives=(8.0,), risk_weight=4.0
    )
    query = np.asarray([1.0, 0.0, 0.0])

    tissue.participate(query, receptor=2, amount=1.0)
    tissue.confirm(0.1)
    tissue.confirm(-0.1)

    assert tissue.value(query, 2) < 0.0
    assert tissue.permission(query, 2) == 0.0
