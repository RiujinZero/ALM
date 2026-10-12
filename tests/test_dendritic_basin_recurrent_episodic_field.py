from __future__ import annotations

from time import monotonic_ns
from types import SimpleNamespace

import numpy as np

from manifoldflow.dendritic_basin.recurrent_episodic_field import (
    SparseRecurrentEpisodicField,
)
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _receptor_route_distribution,
)


def test_partial_cue_recurrently_completes_receptor_bound_content() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    field.encode(((1, 1), (2, 2), (3, 4)), ((90, 8),))

    completion = field.retrieve(((1, 1), (2, 2)))

    assert completion is not None
    assert completion.items == ((90, 8),)
    assert len(completion.units) == 4
    assert completion.coherence > 0.0


def test_incompatible_memory_is_suppressed_by_sparse_completion() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    first = ((1, 1), (2, 2), (3, 3), (4, 4))
    second = ((11, 1), (12, 2), (13, 3), (14, 4))
    field.encode(first, ((80, 4),))
    field.encode(second, ((90, 8),))

    completion = field.retrieve(first[:3])

    assert completion is not None
    assert (80, 4) in completion.items
    assert (90, 8) not in completion.items


def test_two_branch_supported_memories_veto_commitment() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    first = ((1, 1), (2, 2), (3, 3), (4, 4))
    second = ((11, 1), (12, 2), (13, 3), (14, 4))
    field.encode(first, ((80, 4),))
    field.encode(second, ((90, 8),))

    completion = field.retrieve((first[0], first[1], second[0], second[1]))

    assert completion is None


def test_only_opened_return_path_refreshes_recall_excitability() -> None:
    field = SparseRecurrentEpisodicField(memory_units=32, assembly_size=4)
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))
    completion = field.retrieve(cue[:3])
    assert completion is not None
    assert all(field.recall_clock[unit] < 0 for unit in completion.units)

    field.reinstate(completion)

    assert all(field.recall_clock[unit] == field.clock for unit in completion.units)


def test_episode_reset_clears_fast_dendritic_state_only() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=32,
        assembly_size=4,
        dendritic_buffer_half_life=4.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))
    completion = field.retrieve(cue[:3])
    assert completion is not None
    field.reinstate(completion)
    learned_units = field.engram_units

    field.reset_episode()

    assert field.engram_units == learned_units
    assert field.engram_facilitation == [0.0]
    assert field.return_buffer == {}
    assert field.return_buffer_time == {}
    assert np.all(field.recall_clock == -1)


def test_reinstated_return_persists_then_decays_without_becoming_observation() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=32,
        assembly_size=4,
        return_buffer_half_life=2.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))
    completion = field.retrieve(cue[:3])
    assert completion is not None

    field.reinstate(completion)
    initial = field.return_signal()
    field.advance()
    decayed = field.return_signal()

    assert initial is not None
    assert decayed is not None
    assert initial.items == decayed.items == ((80, 4),)
    assert decayed.commitment < initial.commitment


def test_millisecond_return_buffer_expires_by_elapsed_time() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=32,
        assembly_size=4,
        return_buffer_mode="milliseconds",
        return_buffer_half_life_ms=10.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))
    completion = field.retrieve(cue[:3])
    assert completion is not None
    field.reinstate(completion)
    field.return_buffer_time[(80, 4)] = (
        completion.commitment,
        monotonic_ns() - 100_000_000,
    )

    assert field.return_signal() is None


def test_layered_return_buffer_survives_while_either_trace_is_live() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=32,
        assembly_size=4,
        return_buffer_mode="either",
        return_buffer_half_life=16.0,
        return_buffer_half_life_ms=10.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))
    completion = field.retrieve(cue[:3])
    assert completion is not None
    field.reinstate(completion)
    field.return_buffer_time[(80, 4)] = (
        completion.commitment,
        monotonic_ns() - 100_000_000,
    )

    signal = field.return_signal()

    assert signal is not None
    assert signal.items == ((80, 4),)


def test_recency_bias_cannot_complete_without_matching_cortical_input() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
    )
    field.encode(((1, 1), (2, 2)), ((80, 4),))

    completion = field.retrieve(((90, 9),))

    assert completion is None


def test_frames_from_one_event_can_bind_the_same_episodic_assembly() -> None:
    field = SparseRecurrentEpisodicField(memory_units=64, assembly_size=4)
    units = field.encode(((1, 1),), ((80, 4),))

    repeated = field.encode(((2, 2),), ((80, 4),), units=units)

    assert repeated == units
    assert len(set(units)) == 4
    assert all(field.use_count[unit] == 2 for unit in units)
    assert len(field.engram_codes) == 1
    assert len(field.engram_codes[0]) == 2
    assert len(field.engram_branches[0]) <= field.branch_count


def test_local_eligibility_binds_earlier_activity_to_later_activity() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        order_sensitive_binding=True,
        sequence_trace_decay=0.5,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((80, 4),), reset_sequence=True)
    field.encode(cue, ((90, 8),), units=units)

    completion = field.retrieve(cue, sequence_cue_items=((80, 4),))

    assert completion is not None
    assert completion.items == ((90, 8),)
    assert completion.sequence_used
    assert field.sequence_updates == 1


def test_local_eligibility_does_not_infer_the_reverse_transition() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
        order_sensitive_binding=True,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((80, 4),), reset_sequence=True)
    field.encode(cue, ((90, 8),), units=units)

    completion = field.retrieve(cue, sequence_cue_items=((90, 8),))

    assert completion is None


def test_local_eligibility_decays_with_order_and_resets_at_event_boundary() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_trace_decay=0.5,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((80, 4),), reset_sequence=True)
    field.encode(cue, ((90, 8),), units=units)
    field.encode(cue, ((100, 16),), units=units)
    transitions = field.engram_sequence_weights[0]

    assert transitions[(80, 4)][(90, 8)] == 0.5
    assert transitions[(80, 4)][(100, 16)] == 0.25
    assert transitions[(90, 8)][(100, 16)] == 0.5

    field.encode(cue, ((110, 32),), units=units, reset_sequence=True)

    assert (110, 32) not in transitions.get((100, 16), {})


def test_sequence_field_propagates_after_the_external_cue_is_released() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=4,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    units = field.encode(cue, ((80, 4),), reset_sequence=True)
    field.encode(cue, ((90, 8),), units=units)
    field.encode(cue, ((100, 16),), units=units)

    completion = field.retrieve(cue, sequence_cue_items=((80, 4),))

    assert completion is not None
    assert completion.items == ((100, 16),)
    assert completion.sequence_used
    assert completion.iterations > field.settling_steps


def test_cortical_compatibility_stops_the_matching_sequence_branch() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=4,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),), reset_sequence=True)
    field.engram_sequence_weights[0] = {
        (80, 4): {(90, 8): 1.0, (91, 8): 1.0},
        (90, 8): {(100, 16): 1.0},
        (91, 8): {(101, 16): 1.0},
    }

    completion = field.retrieve(
        cue,
        sequence_cue_items=((80, 4),),
        content_gain={(100, 16): 1.0, (101, 16): 0.0},
    )

    assert completion is not None
    assert completion.items == ((100, 16),)
    assert (101, 16) not in completion.items


def test_sequence_settling_preserves_graded_predecessor_strength() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        order_sensitive_binding=True,
        sequence_settling_steps=1,
    )
    cue = ((1, 1), (2, 2), (3, 3))
    field.encode(cue, ((70, 2),), reset_sequence=True)
    field.engram_sequence_weights[0] = {
        (70, 2): {(80, 4): 1.0},
        (71, 2): {(81, 4): 1.0},
    }

    settled, _ = field._settle_sequence(
        0,
        ((70, 2), (71, 2)),
        None,
        cue_strength={(70, 2): 1.0, (71, 2): 0.5},
    )

    assert settled[(80, 4)] == 1.0
    assert settled[(81, 4)] == 0.5


def test_running_residual_suppresses_expected_cortical_activity() -> None:
    field = SparseRecurrentEpisodicField(
        branch_input_mode="running_residual",
        background_learning_rate=0.5,
    )
    code = field._code(((1, 1), (2, 2), (3, 3)))
    novel = field._branch_input(code, None)
    for _ in range(6):
        field._observe_background(code, None)

    familiar = field._branch_input(code, None)

    assert float(np.sum(familiar)) < float(np.sum(novel))


def test_context_residual_preserves_context_specific_surprise() -> None:
    field = SparseRecurrentEpisodicField(
        branch_input_mode="context_residual",
        background_learning_rate=0.5,
    )
    first = field._code(((1, 1), (2, 2), (3, 3)))
    second = field._code(((8, 8), (9, 9), (10, 10)))
    for _ in range(6):
        field._observe_background(first, 1)
        field._observe_background(second, 2)

    expected_here = field._branch_input(first, 1)
    surprising_there = field._branch_input(first, 2)

    assert float(np.sum(expected_here)) < float(np.sum(surprising_there))


def test_contextual_recruitment_reads_existing_basal_and_apical_support() -> None:
    field = SparseRecurrentEpisodicField(
        cortical_dimensions=128,
        memory_units=64,
        assembly_size=4,
        branch_input_mode="running_residual",
    )
    first = ((1, 1), (2, 2), (3, 3))
    second = ((11, 1), (12, 2), (13, 3))
    field.encode(first, ((80, 4),))
    field.encode(second, ((90, 4),))
    background = field.background_activity.copy()

    basal, apical, branch_peak = field.contextual_recruitment(first)

    assert basal.shape == apical.shape == branch_peak.shape == (2,)
    assert basal[0] > basal[1]
    assert apical[0] > apical[1]
    assert branch_peak[0] > branch_peak[1]
    np.testing.assert_array_equal(field.background_activity, background)


def test_attractor_probe_releases_seed_before_measuring_completion() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        capture_recency_gain=0.0,
        recall_recency_gain=0.0,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))

    probe = field.probe_unique_attractor(cue, seed_fractions=(0.25, 0.5))

    assert probe is not None
    assert probe["seed"]["0.25"]["target"] >= 0.0
    assert probe["persistent"]["target"] >= 0.8
    assert probe["persistent"]["competitor"] == 0.0


def test_dendritic_buffer_facilitates_only_repeated_selective_evidence() -> None:
    field = SparseRecurrentEpisodicField(dendritic_buffer_half_life=4.0)
    field.engram_facilitation = [0.0, 0.0]
    support = np.asarray((0.2, 0.1), dtype=np.float64)

    first = field._facilitate_apical_support(support)
    repeated = field._facilitate_apical_support(support)
    silent = field._facilitate_apical_support(np.zeros_like(support))

    assert first[0] == 0.2
    assert repeated[0] > first[0]
    assert first[1] == 0.0
    assert repeated[1] == 0.0
    assert np.array_equal(silent, np.zeros_like(support))
    assert 0.0 < field.engram_facilitation[0] < repeated[0]


def test_disabled_dendritic_buffer_is_bitwise_transparent() -> None:
    field = SparseRecurrentEpisodicField(dendritic_buffer_half_life=0.0)
    field.engram_facilitation = [0.0, 0.0]
    support = np.asarray((0.2, 0.1), dtype=np.float64)

    facilitated = field._facilitate_apical_support(support)

    assert np.array_equal(facilitated, support)
    assert field.engram_facilitation == [0.0, 0.0]


def test_other_event_locally_prunes_nondiscriminative_branches() -> None:
    field = SparseRecurrentEpisodicField(
        memory_units=64,
        assembly_size=4,
        heterosynaptic_decay=0.5,
        branch_survival_threshold=0.6,
    )
    cue = ((1, 1), (2, 2), (3, 3), (4, 4))
    field.encode(cue, ((80, 4),))

    field.encode(cue, ((90, 8),))

    assert field.branch_depressions > 0
    assert field.branch_deaths > 0
    assert any(not branch.synapses for branch in field.engram_branches[0])


def test_receptor_return_projects_only_predictive_routes_through_prefix() -> None:
    learner = SimpleNamespace(
        current_word=bytearray(b"ca"),
        route_counts={(4, 7): {0: 3, 1: 1}, (8, 9): {2: 5}},
        route_totals={(4, 7): 4, (8, 9): 5},
        route_utility={(4, 7): 0.2, (8, 9): -0.1},
        minimum_route_utility_bits=0.0,
        word_forms=[b"cat", b"car", b"dog"],
    )

    distribution = _receptor_route_distribution(
        learner, ((7, 4), (9, 8)), predictive_only=True
    )

    assert distribution is not None
    assert np.isclose(distribution[ord("t")], 0.75)
    assert np.isclose(distribution[ord("r")], 0.25)
    assert distribution[ord("g")] == 0.0


def test_receptor_return_preserves_graded_sequence_strength() -> None:
    learner = SimpleNamespace(
        current_word=bytearray(),
        route_counts={(4, 7): {0: 1}, (8, 9): {1: 1}},
        route_totals={(4, 7): 1, (8, 9): 1},
        route_utility={(4, 7): 1.0, (8, 9): 1.0},
        minimum_route_utility_bits=0.0,
        word_forms=[b"a", b"b"],
    )

    distribution = _receptor_route_distribution(
        learner,
        ((7, 4), (9, 8)),
        predictive_only=False,
        item_strengths=(0.25, 1.0),
    )

    assert distribution is not None
    assert np.isclose(distribution[ord("a")], 0.2)
    assert np.isclose(distribution[ord("b")], 0.8)
