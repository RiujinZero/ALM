from __future__ import annotations

import numpy as np

from manifoldflow.dendritic_basin.recurrent_episodic_field import (
    SparseRecurrentEpisodicField,
)
from manifoldflow.dendritic_basin.relational_access import RelationalAccessField


def _code(*active: int) -> np.ndarray:
    result = np.zeros(32, dtype=np.float32)
    result[list(active)] = 1.0
    return result


def test_external_confirmation_teaches_delayed_relational_access() -> None:
    access = RelationalAccessField(32, trace_decay=0.9, activation_threshold=0.5)
    access.propose(_code(1, 3, 7, 9))
    access.advance()

    assert (
        access.confirm(
            (80, 4), (2,), external=True, source_items=((70, 2),)
        )
        == 1
    )
    signal = access.activate(_code(1, 3, 7, 9), 4)

    assert signal.engram_support[2] > 0.5
    assert signal.content_support[2][(80, 4)] > 0.5
    assert signal.source_support[2][(70, 2)] > 0.5


def test_signed_access_error_depresses_an_incorrect_query_memory() -> None:
    access = RelationalAccessField(
        32,
        trace_decay=0.9,
        activation_threshold=0.3,
        learning_mode="prediction_error",
    )
    query = _code(1, 3, 7, 9)
    access.propose(query)
    access.confirm((80, 4), (0,), external=True)
    access.reset_episode()
    access.propose(query)
    access.confirm((90, 4), (1,), external=True)
    signal = access.activate(query, 2)

    assert signal.engram_support[1] > 0.5
    assert signal.engram_support[0] < access.activation_threshold
    assert (90, 4) in signal.content_support[1]
    assert 0 not in signal.content_support


def test_signed_access_preserves_simultaneous_external_assemblies() -> None:
    access = RelationalAccessField(
        32,
        activation_threshold=0.3,
        learning_mode="prediction_error",
    )
    query = _code(1, 3, 7, 9)
    access.propose(query)
    access.confirm_many(
        (
            ((80, 4), (0,), ()),
            ((90, 4), (1,), ()),
        ),
        external=True,
    )
    signal = access.activate(query, 2)

    assert signal.engram_support[0] > 0.5
    assert signal.engram_support[1] > 0.5
    assert (80, 4) in signal.content_support[0]
    assert (90, 4) in signal.content_support[1]


def test_access_trace_can_be_consumed_by_its_first_external_consequence() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="prediction_error",
        consume_on_confirmation=True,
    )
    access.propose(_code(1, 3, 7, 9))
    access.confirm((80, 4), (0,), external=True)

    assert not np.any(access.eligibility)
    assert not np.any(access.current_query)


def test_latched_access_trace_ignores_intervening_cortical_states() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="prediction_error",
        latch_until_confirmation=True,
    )
    original = _code(1, 3, 7, 9)
    access.propose(original)
    eligibility = access.eligibility.copy()
    access.propose(_code(2, 4, 6, 8))

    np.testing.assert_array_equal(access.eligibility, eligibility)


def test_additive_access_trace_retains_graded_previous_activity() -> None:
    access = RelationalAccessField(
        32,
        trace_decay=0.5,
        learning_mode="prediction_error",
        eligibility_mode="additive",
    )
    access.propose(_code(1, 3))
    access.advance()
    access.propose(_code(5, 7))

    assert access.eligibility[1] > 0.0
    assert access.eligibility[5] > access.eligibility[1]


def test_parallel_access_traces_are_consumed_by_the_next_consequence() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="prediction_error",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
    )
    access.propose(_code(1, 3))
    access.advance()
    access.propose(_code(5, 7))

    assert len(access.eligibility_bank) == 2
    assert access.eligibility_bank[0][1] < access.eligibility_bank[1][5]

    access.confirm((80, 4), (0,), external=True)

    assert access.eligibility_bank == []
    assert not np.any(access.eligibility)


def test_parallel_access_trace_matures_before_claiming_a_consequence() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="branches",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
        minimum_confirmation_age=2,
        activation_threshold=0.3,
    )
    query = _code(1, 3, 5)
    access.propose(query)

    assert access.confirm((70, 4), (0,), external=True) == 0
    access.advance()
    assert access.confirm((75, 4), (0,), external=True) == 0
    access.advance()
    assert access.confirm((80, 4), (0,), external=True) == 1

    signal = access.activate(query, 1)
    assert (80, 4) in signal.content_support[0]
    assert (70, 4) not in signal.content_support[0]
    assert (75, 4) not in signal.content_support[0]
    assert access.eligibility_bank == []
    assert access.eligibility_births == []


def test_occurrence_branches_preserve_distinct_queries_for_one_memory() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="branches",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
        activation_threshold=0.3,
    )
    item = (80, 4)
    for query in (_code(1, 3, 5), _code(20, 22, 24)):
        access.propose(query)
        access.confirm(item, (0,), external=True)

    assert len(access.branches[(0, item)]) == 2
    assert item in access.activate(_code(1, 3, 5), 1).content_support[0]
    assert item in access.activate(_code(20, 22, 24), 1).content_support[0]


def test_occurrence_branch_false_responses_compete_for_survival() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="branches",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
        activation_threshold=0.3,
        branch_false_response_decay=0.5,
        branch_survival_threshold=0.2,
    )
    query = _code(1, 3, 5)
    access.propose(query)
    access.confirm((80, 4), (0,), external=True)
    for _ in range(3):
        access.propose(query)
        access.confirm((90, 4), (1,), external=True)

    assert (80, 4) not in access.activate(query, 2).content_support.get(0, {})
    assert access.branches[(0, (80, 4))][0].false_responses >= 2


def test_recently_confirmed_branch_is_protected_from_immediate_successors() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="branches",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
        activation_threshold=0.3,
        branch_false_response_decay=0.5,
        branch_false_response_grace=2,
    )
    query = _code(1, 3, 5)
    access.propose(query)
    access.confirm((80, 4), (0,), external=True)
    branch = access.branches[(0, (80, 4))][0]

    access.propose(query)
    access.confirm((90, 4), (1,), external=True)
    assert branch.strength == 1.0
    assert branch.false_responses == 0

    access.advance()
    access.advance()
    access.propose(query)
    access.confirm((90, 4), (1,), external=True)
    assert branch.strength == 0.5
    assert branch.false_responses == 1


def test_more_surprising_cue_is_required_to_weaken_a_branch() -> None:
    access = RelationalAccessField(
        32,
        learning_mode="branches",
        eligibility_mode="parallel",
        consume_on_confirmation=True,
        activation_threshold=0.3,
        branch_false_response_decay=0.5,
        branch_survival_mode="surprise",
    )
    query = _code(1, 3, 5)
    access.propose(query, salience=3.0)
    access.confirm((80, 4), (0,), external=True)
    branch = access.branches[(0, (80, 4))][0]

    access.propose(query, salience=1.0)
    access.confirm((90, 4), (1,), external=True)
    assert branch.strength == 1.0
    assert branch.false_responses == 0

    access.propose(query, salience=4.0)
    access.confirm((90, 4), (1,), external=True)
    assert branch.strength == 0.5
    assert branch.false_responses == 1


def test_recalled_activity_cannot_teach_access() -> None:
    access = RelationalAccessField(32)
    access.propose(_code(1, 2, 3))

    assert access.confirm((80, 4), (0,), external=False) == 0
    assert not np.any(access.activate(_code(1, 2, 3), 1).engram_support)


def test_learned_access_seeds_safe_completion_and_bound_return() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=32,
        memory_units=64,
        assembly_size=4,
        branch_input_mode="running_residual",
        residual_seed_gain=1.0,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((80, 4),))
    engram = field._engram_by_units[tuple(sorted(units))]
    access = RelationalAccessField(32, activation_threshold=0.5)
    query = _code(5, 6, 7, 8)
    access.propose(query)
    access.confirm((80, 4), field.engrams_containing((80, 4)), external=True)
    signal = access.activate(query, len(field.engram_units))

    completion = field.retrieve(
        ((90, 1),),
        external_apical_support=signal.engram_support,
        access_content=signal.content_support,
    )

    assert completion is not None
    assert completion.items == ((80, 4),)
    assert signal.engram_support[engram] > 0.0


def test_learned_access_can_recruit_existing_successor_instead_of_filler() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=32,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=1,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((70, 2),), reset_sequence=True)
    field.encode(cue, ((80, 4),), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]
    support = np.ones(1, dtype=np.float64)

    completion = field.retrieve(
        cue,
        sequence_cue_items=((70, 2),),
        external_apical_support=support,
        access_content={engram: {(99, 8): 1.0}},
        access_return_mode="successor",
    )

    assert completion is not None
    assert completion.items == ((80, 4),)
    assert completion.sequence_used


def test_learned_access_can_align_a_new_query_to_an_old_predecessor_phase() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=32,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=1,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    old_context = ((1, 1), (2, 2), (3, 3))
    units = field.encode(old_context, ((70, 2),), reset_sequence=True)
    field.encode(old_context, ((80, 4),), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]
    access = RelationalAccessField(32, activation_threshold=0.5)
    new_query = _code(20, 21, 22)
    access.propose(new_query)
    access.confirm(
        (80, 4),
        (engram,),
        external=True,
        source_items=((70, 2),),
    )
    signal = access.activate(new_query, len(field.engram_units))

    completion = field.retrieve(
        ((20, 1),),
        external_apical_support=signal.engram_support,
        access_content=signal.content_support,
        access_return_mode="aligned_successor",
        access_sequence_seed=signal.source_support,
    )

    assert completion is not None
    assert completion.items == ((80, 4),)
    assert completion.sequence_used


def test_learned_context_item_can_seed_its_existing_successor() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=32,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=1,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    cue = ((1, 1), (2, 2), (3, 3))
    units = field.encode(cue, ((70, 2),), reset_sequence=True)
    field.encode(cue, ((80, 4),), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]

    completion = field.retrieve(
        cue,
        external_apical_support=np.ones(1, dtype=np.float64),
        access_content={engram: {(70, 2): 1.0}},
        access_return_mode="content_successor",
    )

    assert completion is not None
    assert completion.items == ((80, 4),)
    assert completion.sequence_used


def test_contextual_successor_branches_separate_reused_source_occurrences() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=128,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        contextual_successor_binding=True,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        seed=901,
    )
    first_context = ((1, 1), (2, 2), (3, 3), (4, 4))
    second_context = ((11, 1), (12, 2), (13, 3), (14, 4))
    source = (70, 2)
    units = field.encode(first_context, (source,), reset_sequence=True)
    field.encode(first_context, ((80, 4),), units=units)
    field.encode(second_context, (source,), units=units, reset_sequence=True)
    field.encode(second_context, ((90, 4),), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]

    first = field.retrieve(
        first_context,
        external_apical_support=np.ones(1, dtype=np.float64),
        access_content={engram: {source: 1.0}},
        access_return_mode="contextual_successor",
    )
    second = field.retrieve(
        second_context,
        external_apical_support=np.ones(1, dtype=np.float64),
        access_content={engram: {source: 1.0}},
        access_return_mode="contextual_successor",
    )

    assert first is not None and first.items == ((80, 4),)
    assert second is not None and second.items == ((90, 4),)


def test_contextual_successor_uses_residual_context_for_birth_and_recall() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=128,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        contextual_successor_binding=True,
        branch_input_mode="running_residual",
        background_learning_rate=0.001,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        seed=903,
    )
    context = ((21, 1), (22, 2), (23, 3), (24, 4))
    source = (70, 2)
    units = field.encode(context, (source,), reset_sequence=True)
    field.encode(context, ((80, 4),), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]

    completion = field.retrieve(
        context,
        external_apical_support=np.ones(1, dtype=np.float64),
        access_content={engram: {source: 1.0}},
        access_return_mode="contextual_successor",
    )

    assert field.engram_contextual_successors[engram]
    assert completion is not None and completion.items == ((80, 4),)


def test_contextual_successor_competition_sharpens_learned_transition() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=128,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        contextual_successor_binding=True,
        contextual_successor_competition_gain=12.0,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        seed=904,
    )
    context = ((31, 1), (32, 2), (33, 3), (34, 4))
    source = (70, 2)
    units = field.encode(context, (source,), reset_sequence=True)
    field.encode(context, ((80, 4), (90, 4)), units=units)
    engram = field._engram_by_units[tuple(sorted(units))]
    for branch in field.engram_contextual_successors[engram]:
        branch.target_strengths = (1.0, 0.25)

    completion = field.retrieve(
        context,
        external_apical_support=np.ones(1, dtype=np.float64),
        access_content={engram: {source: 1.0}},
        access_return_mode="contextual_successor",
    )

    assert completion is not None
    assert completion.items == ((80, 4),)


def test_contextual_successor_nursery_stays_bounded_under_turnover() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=256,
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        contextual_successor_binding=True,
        contextual_successor_branches_per_source=2,
        contextual_successor_capacity=12,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        seed=902,
    )
    units = None
    for occurrence in range(20):
        context = tuple(
            (occurrence * 10 + offset, offset)
            for offset in range(1, 6)
        )
        source = (70 + occurrence % 4, 2)
        target = (100 + occurrence, 4)
        units = field.encode(
            context,
            (source,),
            units=units,
            reset_sequence=True,
        )
        field.encode(context, (target,), units=units)

    branches = field.engram_contextual_successors[0]
    source_counts: dict[tuple[int, int], int] = {}
    for branch in branches:
        source_counts[branch.source] = source_counts.get(branch.source, 0) + 1

    assert len(branches) <= 12
    assert max(source_counts.values(), default=0) <= 2
    assert field.contextual_successor_births > len(branches)
    assert field.contextual_successor_deaths > 0


def test_access_content_cannot_route_an_item_not_bound_to_the_engram() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=32,
        memory_units=64,
        assembly_size=4,
        branch_input_mode="running_residual",
        residual_seed_gain=1.0,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    field.encode(((1, 1), (2, 2), (3, 3)), ((80, 4),))
    support = np.ones(1, dtype=np.float64)

    completion = field.retrieve(
        ((90, 1),),
        external_apical_support=support,
        access_content={0: {(99, 4): 1.0}},
    )

    assert completion is None


def test_learned_access_inhibits_an_incompatible_raw_branch_before_commitment() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=64,
        memory_units=64,
        assembly_size=4,
        branch_input_mode="running_residual",
        residual_seed_gain=1.0,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    field.encode(((1, 1), (2, 2), (3, 3)), ((80, 4),))
    field.encode(((11, 1), (12, 2), (13, 3)), ((90, 4),))
    support = np.asarray((1.0, 0.0), dtype=np.float64)

    completion = field.retrieve(
        ((11, 1), (12, 2), (13, 3)),
        external_apical_support=support,
        access_content={0: {(80, 4): 1.0}},
    )

    assert completion is not None
    assert completion.items == ((80, 4),)
    assert completion.committed_assemblies == 1


def test_cortical_support_distinguishes_occurrences_with_the_same_content() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=128,
        memory_units=64,
        assembly_size=4,
    )
    first = ((1, 1), (2, 2), (3, 3))
    second = ((11, 1), (12, 2), (13, 3))
    field.encode(first, ((80, 4),))
    field.encode(second, ((80, 4),))
    engrams = field.engrams_containing((80, 4))

    support = field.engram_cortical_support(first, engrams)

    assert len(engrams) == 2
    assert support[0] > support[1]
