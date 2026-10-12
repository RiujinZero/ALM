from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.event_capsule import (
    BoundaryEventCapsuleMemory,
    EventCapsule,
)
from manifoldflow.dendritic_basin.resonant_capsule import (
    SparseResonantCapsuleIndex,
)


def test_partial_distributed_cue_completes_the_matching_capsule() -> None:
    memory = BoundaryEventCapsuleMemory()
    memory.capsules = [
        EventCapsule((), ((8, 4), (9, 5)), None, created_at=10),
        EventCapsule((), ((80, 4), (90, 5)), None, created_at=10),
    ]
    memory.clock = 20
    index = SparseResonantCapsuleIndex(
        inverse_temperature=16.0, completion_threshold=0.01
    )

    completion = index.resonate(memory, ((8, 4),))

    assert completion is not None
    assert completion.read.capsule == 0
    assert (8, 4) in completion.read.items
    assert completion.maximum_overlap > 0.0


def test_recent_reinstatement_is_an_additive_excitability_bias() -> None:
    memory = BoundaryEventCapsuleMemory()
    memory.capsules = [
        EventCapsule((), ((8, 4),), None, created_at=10),
        EventCapsule((), ((8, 4),), None, created_at=10, last_recalled=99),
    ]
    memory.clock = 100
    index = SparseResonantCapsuleIndex(
        inverse_temperature=8.0,
        completion_threshold=0.0,
        recall_recency_gain=2.0,
    )

    completion = index.resonate(memory, ((8, 4),))

    assert completion is not None
    assert completion.read.capsule == 1
    assert completion.posterior_peak > 0.5


def test_event_frame_addresses_content_that_is_not_itself_in_the_cue() -> None:
    memory = BoundaryEventCapsuleMemory()
    memory.capsules = [
        EventCapsule(
            (),
            ((80, 4),),
            None,
            frames=((1, ((8, 4), (9, 5))),),
        ),
        EventCapsule((), ((90, 4),), None, frames=((2, ((18, 4),)),)),
    ]
    index = SparseResonantCapsuleIndex(
        inverse_temperature=16.0, completion_threshold=0.01
    )

    completion = index.resonate(memory, ((8, 4),))

    assert completion is not None
    assert completion.read.capsule == 0
    assert (80, 4) in completion.read.items


def test_basal_compatibility_gates_content_before_completion_threshold() -> None:
    memory = BoundaryEventCapsuleMemory()
    memory.capsules = [
        EventCapsule((), ((80, 4),), None, frames=((1, ((8, 4),)),)),
        EventCapsule((), ((90, 4),), None, frames=((2, ((8, 4),)),)),
    ]
    index = SparseResonantCapsuleIndex(completion_threshold=0.0)

    completion = index.resonate(
        memory,
        ((8, 4),),
        content_gain={(80, 4): 0.0, (90, 4): 1.0},
    )

    assert completion is not None
    assert completion.read.items == ((90, 4),)


def test_recurrent_field_projection_can_leave_conjunctions_to_dendrites() -> None:
    index = SparseResonantCapsuleIndex(include_pair_hashes=False)

    code = index.sparse_code(((8, 4), (9, 5)))

    assert np.count_nonzero(code) <= 2 * index.hashes_per_item
