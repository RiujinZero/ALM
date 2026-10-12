from manifoldflow.experiments.dendritic_basin_phase122_full_corpus_4c import (
    _partition,
)


def test_full_corpus_partition_is_disjoint_and_exact() -> None:
    partitions = _partition(17, 4)

    assert partitions == [(0, 5), (5, 4), (9, 4), (13, 4)]
    assert sum(length for _, length in partitions) == 17

