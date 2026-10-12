from __future__ import annotations

from manifoldflow.dendritic_basin.transient_binding import FastSparseBindingIndex


def test_partial_address_reinstates_an_existing_cortical_filler() -> None:
    index = FastSparseBindingIndex(max_cells=16, cells_per_binding=2)
    index.bind(3, 7, 41)

    recalled = index.recall(3, 7)

    assert recalled.filler == 41
    assert recalled.active_cells == 2
    assert index.address(41) == (3, 7)


def test_binding_swap_changes_content_while_preserving_address() -> None:
    left = FastSparseBindingIndex(max_cells=8)
    right = FastSparseBindingIndex(max_cells=8)
    left.bind(2, 5, 11)
    right.bind(2, 5, 19)

    assert left.recall(2, 5).filler == 11
    assert right.recall(2, 5).filler == 19


def test_basin_swap_changes_structure_address_without_changing_fillers() -> None:
    index = FastSparseBindingIndex(max_cells=8)
    index.bind(1, 4, 23)
    index.bind(2, 4, 29)

    assert index.recall(1, 4).filler == 23
    assert index.recall(2, 4).filler == 29


def test_weak_old_cells_are_reused_under_bounded_competition() -> None:
    index = FastSparseBindingIndex(
        max_cells=2,
        cells_per_binding=1,
        activity_decay=0.5,
        minimum_activity=0.0,
    )
    index.bind(0, 0, 1)
    index.bind(0, 1, 2)
    index.bind(0, 2, 3)

    assert index.live_cells == 2
    assert index.replacements == 1
    assert index.recall(0, 0).filler is None
    assert index.recall(0, 2).filler == 3
