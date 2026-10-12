from __future__ import annotations

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
)


def test_boundary_capsule_opens_on_change_and_closes_on_stability() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=2.0, close_threshold=0.5, stable_steps=2
    )
    assert not memory.observe_boundary(
        3.0, query=11, active_items=((1, 7, 2.0),), basin=4
    )
    assert not memory.observe_boundary(
        0.2, query=12, active_items=((2, 9, 1.0),), basin=4
    )
    assert memory.observe_boundary(0.1, query=13, active_items=((1, 7, 3.0),), basin=4)
    assert memory.captures == 1
    assert memory.capsules[0].items[0] == (7, 1)
    assert memory.capsules[0].created_at == memory.clock


def test_signed_read_credit_calibrates_permission_and_retention() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=1.0, close_threshold=0.5, stable_steps=1
    )
    memory.observe_boundary(2.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.observe_boundary(0.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.clock += memory.minimum_read_age
    read = memory.candidate(21, None)
    assert read is not None

    memory.credit(read, 0.5)
    memory.clock += memory.minimum_read_age
    improved = memory.candidate(21, None)
    assert improved is not None and improved.permission > 0.0
    strength = memory.capsules[0].strength
    memory.credit(improved, -1.0)

    assert memory.capsules[0].strength < strength


def test_only_live_reinstatement_refreshes_recall_recency() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=1.0, close_threshold=0.5, stable_steps=1
    )
    memory.observe_boundary(2.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.observe_boundary(0.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.clock += memory.minimum_read_age
    read = memory.candidate(21, None)
    assert read is not None

    memory.touch(read)
    assert memory.capsules[0].last_recalled is None
    memory.reinstate(read)
    assert memory.capsules[0].last_recalled == memory.clock


def test_exact_memory_branch_learns_its_own_surprise_reduction() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=1.0, close_threshold=0.5, stable_steps=1
    )
    memory.observe_boundary(2.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.observe_boundary(0.0, query=21, active_items=((4, 8, 1.0),), basin=None)
    memory.clock += memory.minimum_read_age
    read = memory.candidate(21, None)
    assert read is not None

    memory.credit_branch(read, 0.75)

    assert memory.branch_evidence(read) == 1
    assert memory.branch_value(read) == 0.375


def test_partial_cortical_trace_selects_matching_occurrence() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=1.0, close_threshold=0.5, stable_steps=1, minimum_read_age=1
    )
    for assembly in (8, 9):
        memory.observe_boundary(
            2.0, query=21, active_items=((4, assembly, 1.0),), basin=None
        )
        memory.observe_boundary(
            0.0, query=21, active_items=((4, assembly, 1.0),), basin=None
        )
    memory.clock += 1

    reads = memory.candidates(21, None, context_items=((9, 4),))

    assert reads[0].items == ((9, 4),)
    assert reads[0].trace_surprise == 0.0


def test_sparse_predictive_surprise_selects_matching_event() -> None:
    memory = BoundaryEventCapsuleMemory(
        open_threshold=1.0, close_threshold=0.5, stable_steps=1, minimum_read_age=1
    )
    for assembly, fingerprint in (
        (8, ((65, 0.8), (66, 0.1))),
        (9, ((90, 0.8), (91, 0.1))),
    ):
        memory.observe_boundary(
            2.0,
            query=21,
            active_items=((4, assembly, 1.0),),
            basin=None,
            prediction_fingerprint=fingerprint,
        )
        memory.observe_boundary(
            0.0,
            query=21,
            active_items=((4, assembly, 1.0),),
            basin=None,
            prediction_fingerprint=fingerprint,
        )
    memory.clock += 1

    reads = memory.candidates(
        21,
        None,
        context_fingerprint=((90, 0.7), (91, 0.2)),
    )

    assert reads[0].items == ((9, 4),)
    assert reads[0].predictive_surprise < reads[1].predictive_surprise


def test_diverse_compaction_preserves_a_distinct_middle_assembly() -> None:
    memory = BoundaryEventCapsuleMemory(
        frames_per_capsule=3,
        frame_compaction="diverse",
        open_threshold=1.0,
        close_threshold=0.5,
        stable_steps=1,
    )
    frames = (
        (1, ((1, 1),)),
        (2, ((1, 1),)),
        (3, ((7, 2),)),
        (4, ((1, 1),)),
        (5, ((1, 1),)),
    )

    compacted = memory._diverse_frames(frames)

    assert len(compacted) == 3
    assert frames[2] in compacted
