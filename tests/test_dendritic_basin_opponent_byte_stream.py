from __future__ import annotations

import pickle

import numpy as np

from manifoldflow.dendritic_basin.opponent_byte_stream import (
    ContextualOpponentAttractorLearner,
    DelayedOpponentAttractorControl,
    DevelopmentalOpponentAttractorLearner,
    EpisodicOpponentAttractorLearner,
    OpponentAssemblyTransitionLearner,
    OpponentAttractorAssemblyLearner,
    OpponentByteCodec,
    OpponentPairDendriteLearner,
    OpponentTransitionLearner,
    PhraseEventOpponentLearner,
    PredictiveMotifOpponentLearner,
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.routed_event_memory import (
    GoalConditionedTemporalBasisMotifOpponentLearner,
    RoutedMotifOpponentLearner,
    TemporalBasisMotifOpponentLearner,
)


class _LegacyPrefixScanMotifLearner(PredictiveMotifOpponentLearner):
    def _apical_byte_counts(self) -> tuple[dict[int, int], int] | None:
        if len(self.current_word) < self.minimum_prefix_bytes:
            return None
        prefix = bytes(self.current_word)
        if self.active_word_candidates is None:
            row = self._active_event_row()
            if row is None:
                return None
            self.active_word_candidates = dict(row)
        candidates = {
            word_id: observations
            for word_id, observations in self.active_word_candidates.items()
            if len(self.word_forms[word_id]) > len(prefix)
            and self.word_forms[word_id].startswith(prefix)
        }
        self.active_word_candidates = candidates
        counts: dict[int, int] = {}
        for word_id, observations in candidates.items():
            word = self.word_forms[word_id]
            next_byte = word[len(prefix)]
            counts[next_byte] = counts.get(next_byte, 0) + observations
        if not counts:
            return None
        return counts, sum(counts.values())


def test_opponent_codec_round_trips_every_byte() -> None:
    codec = OpponentByteCodec()
    byte_ids = np.arange(256, dtype=np.uint8)
    encoded = codec.encode(byte_ids)

    assert encoded.shape == (256, 16)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.full(256, 8.0))
    np.testing.assert_array_equal(codec.decode(encoded), byte_ids)


def test_opponent_stream_is_chunk_causal() -> None:
    stream = np.resize(np.asarray((65, 66, 67, 65), dtype=np.uint8), 1000)
    whole = OpponentTransitionLearner()
    split = OpponentTransitionLearner()

    whole_probability = whole.observe_chunk_probabilities(stream)
    split_probability = np.concatenate(
        (
            split.observe_chunk_probabilities(stream[:317]),
            split.observe_chunk_probabilities(stream[317:]),
        )
    )

    np.testing.assert_allclose(split_probability, whole_probability)
    np.testing.assert_array_equal(split.unigram_counts, whole.unigram_counts)
    np.testing.assert_array_equal(split.receptor_counts, whole.receptor_counts)


def test_opponent_transition_learns_a_repeating_vector_sequence() -> None:
    learner = OpponentTransitionLearner()
    stream = np.resize(np.asarray((65, 66, 67), dtype=np.uint8), 3000)
    probability = learner.observe_chunk_probabilities(stream)
    early = -np.log2(probability[1:101]).mean()
    late = -np.log2(probability[-100:]).mean()

    assert late < early
    # Individual receptors overlap across bytes, so the basal circuit cannot
    # bind the full eight-receptor identity without dendritic conjunctions.
    assert late < 2.0
    assert learner.embodied_bytes == 18_560


def test_pair_dendrites_bind_a_repeating_population_sequence() -> None:
    learner = OpponentPairDendriteLearner()
    receptor_learner = OpponentTransitionLearner()
    stream = np.resize(np.asarray((65, 66, 67), dtype=np.uint8), 3000)
    probability = learner.observe_chunk_probabilities(stream)
    receptor_probability = receptor_learner.observe_chunk_probabilities(stream)
    late = -np.log2(probability[-100:]).mean()
    receptor_late = -np.log2(receptor_probability[-100:]).mean()

    assert learner.pairs.shape == (112, 2)
    np.testing.assert_array_equal(learner.branch_codebook.sum(axis=1), np.full(256, 28))
    assert late < receptor_late


def test_recursive_opponent_assemblies_recover_byte_identity() -> None:
    learner = OpponentAssemblyTransitionLearner()
    stream = np.resize(np.asarray((65, 66, 67), dtype=np.uint8), 3000)
    probability = learner.observe_chunk_probabilities(stream)
    late = -np.log2(probability[-100:]).mean()

    assert learner.assembly_path(65)[2] == 65
    assert learner.live_assembly_count == 3
    assert late < 0.1


def test_output_attractor_preserves_assembly_correlations() -> None:
    learner = OpponentAttractorAssemblyLearner()
    stream = np.resize(np.asarray((65, 66, 67), dtype=np.uint8), 3000)
    probability = learner.observe_chunk_probabilities(stream)

    assert learner.live_assembly_count == 3
    assert -np.log2(probability[-100:]).mean() < 0.1
    assert learner.embodied_bytes < 25_000


def test_developmental_receptor_tree_requires_repeated_coactivity() -> None:
    learner = DevelopmentalOpponentAttractorLearner(confirmations=2)

    learner.observe(65)
    assert learner.live_assembly_count == 0
    assert learner.nursery_count > 0
    for _ in range(5):
        learner.observe(65)

    assert learner.live_assembly_count == 1
    assert learner.assembly_births == 1
    assert learner.previous_assembly == 65


def test_sparse_streaming_matches_full_basin_broadcast() -> None:
    stream = np.resize(np.asarray((65, 66, 67, 65), dtype=np.uint8), 1000)
    for learner_type in (
        OpponentAttractorAssemblyLearner,
        DevelopmentalOpponentAttractorLearner,
        ContextualOpponentAttractorLearner,
        SequenceCortexOpponentLearner,
        PhraseEventOpponentLearner,
        PredictiveMotifOpponentLearner,
    ):
        streamed = learner_type()
        broadcast = learner_type()
        fast_probability = streamed.observe_chunk_probabilities(stream)
        full_probability = np.asarray(
            [broadcast.observe(int(byte)).observed_probability for byte in stream],
            dtype=np.float64,
        )
        np.testing.assert_allclose(fast_probability, full_probability, equal_nan=True)
        assert streamed.embodied_bytes == broadcast.embodied_bytes


def test_context_dendrite_resolves_ambiguous_basal_transition() -> None:
    stream = np.resize(np.asarray((65, 66, 67, 68, 66, 69), dtype=np.uint8), 6000)
    basal = DevelopmentalOpponentAttractorLearner(confirmations=1)
    contextual = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )

    basal_probability = basal.observe_chunk_probabilities(stream)
    context_probability = contextual.observe_chunk_probabilities(stream)

    assert contextual.live_context_count > 0
    assert -np.log2(context_probability[-500:]).mean() < 0.1
    assert (
        -np.log2(context_probability[-500:]).mean()
        < -np.log2(basal_probability[-500:]).mean()
    )


def test_sequence_cortex_resolves_deeper_temporal_ambiguity() -> None:
    stream = np.resize(
        np.asarray((65, 66, 67, 68, 69, 70, 66, 67, 68, 71), dtype=np.uint8),
        10_000,
    )
    local = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )
    sequence = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        max_context_bytes=4,
        sequence_confirmations=2,
    )

    local_probability = local.observe_chunk_probabilities(stream)
    sequence_probability = sequence.observe_chunk_probabilities(stream)
    ambiguous = ((stream == 69) | (stream == 71)) & (np.roll(stream, 1) == 68)
    ambiguous[:1000] = False

    assert sequence.live_sequence_count > 0
    assert (
        -np.log2(sequence_probability[ambiguous]).mean()
        < -np.log2(local_probability[ambiguous]).mean()
    )


def test_packed_sequence_paths_match_tuple_oracle_exactly() -> None:
    stream = np.resize(
        np.asarray((65, 66, 67, 68, 69, 70, 66, 67, 68, 71), dtype=np.uint8),
        10_000,
    )
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "max_context_bytes": 6,
        "sequence_confirmations": 2,
    }
    packed = SequenceCortexOpponentLearner(**settings, packed_paths=True)
    oracle = SequenceCortexOpponentLearner(**settings, packed_paths=False)

    packed_probability = packed.observe_chunk_probabilities(stream)
    oracle_probability = oracle.observe_chunk_probabilities(stream)

    np.testing.assert_array_equal(packed_probability, oracle_probability)
    assert packed.live_sequence_count == oracle.live_sequence_count
    assert packed.live_sequence_synapse_count == oracle.live_sequence_synapse_count


def test_fused_exact_sequence_chunk_matches_subclass_oracle() -> None:
    """The optimized exact-class loop must preserve every learned bit."""
    stream = np.random.default_rng(3601).integers(
        0, 256, size=20_000, dtype=np.uint8
    )
    settings = dict(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        max_context_bytes=6,
        packed_paths=True,
    )

    class SequenceOracle(SequenceCortexOpponentLearner):
        pass

    fused = SequenceCortexOpponentLearner(**settings)
    oracle = SequenceOracle(**settings)
    fused_probability = fused.observe_chunk_probabilities(stream, learn=True)
    oracle_probability = oracle.observe_chunk_probabilities(stream, learn=True)

    np.testing.assert_array_equal(fused_probability, oracle_probability)
    for name in (
        "unigram_counts",
        "transition_counts",
        "context_totals",
        "residual_counts",
        "residual_totals",
        "sequence_counts",
        "sequence_totals",
        "sequence_nursery",
        "sequence_history",
        "steps",
    ):
        assert pickle.dumps(getattr(fused, name), protocol=5) == pickle.dumps(
            getattr(oracle, name), protocol=5
        )


def test_phrase_event_cortex_resolves_actor_conditioned_word_ambiguity() -> None:
    stream = np.frombuffer((b"red cue apple. blue cue apricot. " * 300), dtype=np.uint8)
    local = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
    )
    event = PhraseEventOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        minimum_prefix_bytes=2,
    )

    local_probability = local.observe_chunk_probabilities(stream)
    event_probability = event.observe_chunk_probabilities(stream)
    ambiguous = (
        ((stream == ord("p")) | (stream == ord("r")))
        & (np.roll(stream, 1) == ord("p"))
        & (np.roll(stream, 2) == ord("a"))
    )
    ambiguous[:1000] = False

    assert event.live_event_count > 0
    assert -np.log2(event_probability[ambiguous]).mean() < 0.5
    assert (
        -np.log2(event_probability[ambiguous]).mean()
        < -np.log2(local_probability[ambiguous]).mean()
    )


def test_predictive_motifs_resolve_nonlanguage_delayed_ambiguity() -> None:
    rng = np.random.default_rng(3)
    values = []
    for _ in range(2000):
        positive = bool(rng.integers(0, 2))
        values.extend([1 if positive else 2] + [9] * 8 + [50, 60 if positive else 70])
    stream = np.asarray(values, dtype=np.uint8)
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
    }
    local = SequenceCortexOpponentLearner(**settings)
    motif = PredictiveMotifOpponentLearner(
        **settings,
        event_confirmations=2,
        boundary_surprise_bits=2.0,
        minimum_motif_events=2,
        maximum_motif_events=16,
    )

    local_probability = local.observe_chunk_probabilities(stream)
    motif_probability = motif.observe_chunk_probabilities(stream)
    ambiguous = ((stream == 60) | (stream == 70)) & (np.roll(stream, 1) == 50)
    ambiguous[:5000] = False

    assert motif.learned_boundaries > 0
    assert motif.live_motif_count > 0
    assert (
        -np.log2(motif_probability[ambiguous]).mean()
        < -np.log2(local_probability[ambiguous]).mean()
    )


def test_predictive_motifs_are_invariant_to_receptor_relabeling() -> None:
    rng = np.random.default_rng(3)
    values = []
    for _ in range(1000):
        positive = bool(rng.integers(0, 2))
        values.extend([1 if positive else 2] + [9] * 8 + [50, 60 if positive else 70])
    original = np.asarray(values, dtype=np.uint8)
    relabeling = {1: 201, 2: 17, 9: 88, 50: 3, 60: 145, 70: 254}
    relabeled = np.asarray(
        [relabeling[int(value)] for value in original], dtype=np.uint8
    )
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
        "event_confirmations": 2,
    }
    losses = []
    for stream, targets, anchor in (
        (original, (60, 70), 50),
        (relabeled, (145, 254), 3),
    ):
        learner = PredictiveMotifOpponentLearner(**settings)
        probability = learner.observe_chunk_probabilities(stream)
        ambiguous = ((stream == targets[0]) | (stream == targets[1])) & (
            np.roll(stream, 1) == anchor
        )
        ambiguous[:2500] = False
        losses.append(float(-np.log2(probability[ambiguous]).mean()))

    assert losses[0] == losses[1]


def test_routed_motif_roles_resolve_ambiguous_recent_context() -> None:
    rng = np.random.default_rng(46)
    motifs = {
        "left": (1, 2, 3, 4),
        "right": (5, 6, 7, 8),
        "filler_a": (20, 21, 22, 23),
        "filler_b": (24, 25, 26, 27),
        "filler_c": (28, 29, 30, 31),
        "target_left": (50, 60, 80, 90),
        "target_right": (50, 70, 80, 90),
    }
    values: list[int] = []
    targets: list[bool] = []
    for _ in range(3000):
        left = bool(rng.integers(0, 2))
        episode = (
            motifs["left" if left else "right"]
            + motifs["filler_a"]
            + motifs["filler_b"]
            + motifs["filler_c"]
            + motifs["target_left" if left else "target_right"]
        )
        values.extend(episode)
        targets.extend([False] * 17 + [True] + [False] * 2)
    stream = np.asarray(values, dtype=np.uint8)
    target_mask = np.asarray(targets, dtype=np.bool_)
    split = 40_000
    settings = dict(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        boundary_surprise_bits=100.0,
        minimum_motif_events=4,
        maximum_motif_events=4,
    )
    local = PredictiveMotifOpponentLearner(**settings)
    routed = RoutedMotifOpponentLearner(
        **settings,
        route_lags=(1, 2, 4),
        route_confirmations=2,
        route_utility_decay=0.9,
        minimum_route_utility_bits=0.01,
    )
    local.observe_chunk_probabilities(stream[:split], learn=True)
    routed.observe_chunk_probabilities(stream[:split], learn=True)
    local.reset_episode()
    routed.reset_episode()
    local_probability = local.observe_chunk_probabilities(stream[split:], learn=False)
    routed_probability = routed.observe_chunk_probabilities(stream[split:], learn=False)
    selected = target_mask[split:]
    local_bits = float(-np.log2(local_probability[selected]).mean())
    routed_bits = float(-np.log2(routed_probability[selected]).mean())

    assert routed.live_route_count > 0
    assert any(lag == 4 for lag, _ in routed.route_counts)
    # A perfect route is mixed with the ambiguous basal prediction because
    # apical authority is deliberately capped at 0.5.
    assert routed_bits < 0.45
    assert routed_bits < local_bits


def test_routed_byte_class_prunes_cross_tissue_anatomy_to_exact_ceiling() -> None:
    stream = np.random.default_rng(47).integers(0, 256, size=5000, dtype=np.uint8)
    learner = RoutedMotifOpponentLearner(
        max_alm_bytes=350_000,
        byte_budget_check_interval=1_000_000,
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        boundary_surprise_bits=100.0,
        minimum_motif_events=4,
        maximum_motif_events=4,
    )
    learner.observe_chunk_probabilities(stream, learn=True)
    unconsolidated = learner.serialized_size_bytes()

    evictions = learner.enforce_byte_budget(target_fraction=1.0)

    assert unconsolidated > learner.max_alm_bytes
    assert learner.serialized_size_bytes() <= learner.max_alm_bytes
    assert evictions > 0
    assert learner.budget_evictions_by_tissue
    assert learner.growth_frozen is False


def test_strong_route_nursery_can_displace_weak_mature_route() -> None:
    learner = RoutedMotifOpponentLearner(max_alm_bytes=None, motif_confirmations=1)
    left = learner._word_id(b"left", create=True)
    right = learner._word_id(b"right", create=True)
    assert left is not None and right is not None
    weak = (1, left)
    challenger = (1, right)
    learner.route_counts[weak] = {left: 1, right: 1}
    learner.route_totals[weak] = 2
    learner.route_utility[weak] = 0.0
    learner.route_nursery[challenger] = {left: 20}
    learner.route_target_counts = {left: 10, right: 10}
    learner.route_target_total = 20
    learner.max_alm_bytes = 1_000_000
    learner.max_alm_bytes = learner.serialized_size_bytes() - 20

    learner.enforce_byte_budget(target_fraction=1.0)

    assert weak not in learner.route_counts
    assert challenger in learner.route_nursery
    assert learner.budget_evictions_by_tissue == {"route": 1}
    assert learner.budget_mature_replacements == 1
    assert learner.budget_nursery_losses == 0


def test_motif_nursery_recycles_stale_receptor_and_its_branches() -> None:
    learner = RoutedMotifOpponentLearner(
        max_alm_bytes=None,
        max_motif_assemblies=2,
        motif_confirmations=1,
        motif_activity_decay=0.5,
    )
    stale = learner._word_id(b"stale", create=True)
    resident = learner._word_id(b"resident", create=True)
    assert stale == 0 and resident == 1
    learner.route_counts[(1, stale)] = {resident: 3}
    learner.route_totals[(1, stale)] = 3
    learner.route_utility[(1, stale)] = 0.2
    learner.steps = 100

    newcomer = learner._word_id(b"newcomer", create=True)
    learner.enforce_byte_budget()

    assert newcomer is None
    assert b"stale" not in learner.word_ids
    assert learner.word_ids[b"newcomer"] == stale
    assert len(learner.word_forms) == 2
    assert (1, stale) not in learner.route_counts
    assert learner.motif_recycles == 1


def test_temporal_basis_preserves_event_identity_across_age_bands() -> None:
    learner = TemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        temporal_bins=(1, 2, 4, 8),
        max_active_temporal_traces=2,
    )
    first = learner._word_id(b"first", create=True)
    second = learner._word_id(b"second", create=True)
    third = learner._word_id(b"third", create=True)
    assert first is not None and second is not None and third is not None

    learner._advance_temporal_trace(first)
    assert learner._active_route_keys() == ((1, first),)
    learner._advance_temporal_trace(second)
    assert set(learner._active_route_keys()) == {(2, first), (1, second)}
    learner._advance_temporal_trace(third)

    assert first not in learner.temporal_last_seen
    assert set(learner._active_route_keys()) == {(2, second), (1, third)}
    assert learner.temporal_trace_evictions == 1


def test_zero_goal_gain_is_exact_temporal_basis_ablation() -> None:
    stream = np.frombuffer(
        (b"alpha beta gamma delta. alpha beta theta delta. ") * 80,
        dtype=np.uint8,
    )
    settings = dict(
        max_alm_bytes=None,
        motif_confirmations=1,
        confirmations=1,
        context_confirmations=1,
        surprise_bits=0.0,
        sequence_confirmations=1,
        event_confirmations=1,
        boundary_surprise_bits=100.0,
        minimum_motif_events=2,
        maximum_motif_events=8,
        temporal_bins=(1, 2, 4, 8),
    )
    control = TemporalBasisMotifOpponentLearner(**settings)
    frosty = GoalConditionedTemporalBasisMotifOpponentLearner(
        **settings, goal_apical_weight=0.0
    )

    control_probability = control.observe_chunk_probabilities(stream)
    frosty_probability = frosty.observe_chunk_probabilities(stream)

    np.testing.assert_array_equal(control_probability, frosty_probability)
    assert frosty.goal_organ.restoration_sum
    assert frosty.goal_organ.embodied_bytes > 0
    assert frosty.goal_transient_bytes > 0


def test_goal_projects_reachable_restorative_basin_not_one_motif() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_apical_weight=2.0,
        goal_basin_width=0.1,
        minimum_motif_events=3,
    )
    north = learner._word_id(b"north", create=True)
    nest = learner._word_id(b"nest", create=True)
    south = learner._word_id(b"south", create=True)
    assert north is not None and nest is not None and south is not None
    learner.goal_organ.restoration_sum.update({north: 8.0, nest: 7.9, south: 3.0})
    learner.goal_organ.restoration_evidence.update({north: 9.0, nest: 9.0, south: 9.0})
    row = {north: 4, nest: 2, south: 100}

    learner.goal_organ.current_goal = north
    north_distribution = learner._goal_byte_distribution(row)
    learner.goal_organ.current_goal = nest
    nest_distribution = learner._goal_byte_distribution(row)
    learner.goal_organ.current_goal = south
    south_distribution = learner._goal_byte_distribution(row)

    assert north_distribution == nest_distribution
    assert north_distribution == {ord("n"): 1.0}
    assert south_distribution == {ord("s"): 1.0}

    learner.current_word.clear()
    learner.current_word.extend(b"n")
    learner.goal_organ.current_goal = north
    continuation = learner._goal_byte_distribution(row)
    assert set(continuation) == {ord("o"), ord("e")}
    assert continuation[ord("o")] > continuation[ord("e")]


def test_equivalent_restorative_event_does_not_prematurely_end_goal() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_basin_width=0.1,
        minimum_motif_events=3,
    )
    north = learner._word_id(b"north", create=True)
    nest = learner._word_id(b"nest", create=True)
    assert north is not None and nest is not None
    learner.goal_organ.restoration_sum.update({north: 8.0, nest: 7.9})
    learner.goal_organ.restoration_evidence.update({north: 9.0, nest: 9.0})
    learner.goal_organ.observe_deviation(1.0, learn=False)
    learner.goal_organ.current_goal = north
    learner.goal_organ.goal_reference_deviation = 1.0
    learner.goal_surprise_sum = 8.0
    learner.goal_surprise_count = 1
    learner.current_word.extend(b"nest")

    learner._complete_word(learn=False)

    assert learner.goal_organ.current_goal == north


def test_goal_basin_conditions_local_route_recall() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_apical_weight=1.0,
        goal_basin_width=0.1,
        minimum_motif_events=3,
    )
    first_source = learner._word_id(b"first", create=True)
    second_source = learner._word_id(b"second", create=True)
    north = learner._word_id(b"north", create=True)
    south = learner._word_id(b"south", create=True)
    assert None not in (first_source, second_source, north, south)
    learner.temporal_clock = 1
    learner.temporal_last_seen[first_source] = 1
    learner.temporal_last_seen[second_source] = 1
    north_route = (1, first_source)
    south_route = (1, second_source)
    learner.route_counts[north_route] = {north: 8}
    learner.route_counts[south_route] = {south: 8}
    learner.route_totals.update({north_route: 8, south_route: 8})
    learner.route_utility.update({north_route: 1.0, south_route: 1.0})
    learner.goal_route_restoration_sum[north_route] = [8.0, 0.0, 0.0]
    learner.goal_route_restoration_sum[south_route] = [0.0, 8.0, 0.0]
    learner.goal_route_restoration_evidence[north_route] = [9.0, 0.0, 0.0]
    learner.goal_route_restoration_evidence[south_route] = [0.0, 9.0, 0.0]
    permission = learner._goal_selection_key(1, north_route, south_route)
    learner.goal_selection_utility[permission] = 0.1

    learner.goal_organ.current_mode = 0
    cold_row = learner._active_event_row()
    learner.goal_organ.current_mode = 1
    hot_row = learner._active_event_row()

    assert cold_row == {north: 8}
    assert hot_row == {south: 8}
    assert learner.goal_route_context_calls == 2
    assert learner.goal_route_proposals == 1
    assert learner.goal_route_switches == 1

    learner.goal_selection_utility[permission] = -0.1
    assert learner._active_event_row() == {north: 8}
    assert learner.goal_route_denials == 1


def test_goal_columns_consolidate_restoration_credit_by_event_form() -> None:
    settings = dict(max_alm_bytes=None, motif_confirmations=1)
    left = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    right = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    left_shared = left._word_id(b"shared", create=True)
    left._word_id(b"left", create=True)
    right_shared = right._word_id(b"shared", create=True)
    right_target = right._word_id(b"right", create=True)
    assert None not in (left_shared, right_shared, right_target)
    incoming_key = (1, right_shared)
    right.route_counts[incoming_key] = {right_target: 5}
    right.route_totals[incoming_key] = 5
    right.route_utility[incoming_key] = 0.5
    right.goal_route_restoration_sum[incoming_key] = [3.0, 1.0, -1.0]
    right.goal_route_restoration_evidence[incoming_key] = [4.0, 2.0, 2.0]
    incoming_permission = right._goal_selection_key(1, incoming_key, (2, right_target))
    right.goal_selection_utility[incoming_permission] = 0.25
    right.goal_selection_evidence[incoming_permission] = 4
    right.goal_permission_restoration_sum[1] = 2.5
    right.goal_permission_restoration_evidence[1] = 3.5
    right.goal_permission_trials[1] = 2
    right.goal_organ.restoration_sum[right_target] = 2.0
    right.goal_organ.restoration_evidence[right_target] = 3.0
    right.goal_organ.component_restoration_sum[right_target] = np.array([2.0, 0.0, 0.0])
    right.goal_organ.component_restoration_evidence[right_target] = np.array(
        [3.0, 0.0, 0.0]
    )

    left.consolidate(right)

    local_key = (1, left.word_ids[b"shared"])
    local_target = left.word_ids[b"right"]
    assert left.route_counts[local_key][local_target] == 5
    assert left.goal_route_restoration_sum[local_key] == [3.0, 1.0, -1.0]
    assert left.goal_selection_utility[incoming_permission] == 0.25
    assert left.goal_selection_evidence[incoming_permission] == 4
    assert left.goal_permission_restoration_sum[1] == 2.5
    assert left.goal_permission_restoration_evidence[1] == 3.5
    assert left.goal_permission_trials[1] == 2
    assert left.goal_organ.value(local_target) == 0.5
    np.testing.assert_array_equal(
        left.goal_organ.component_restoration_sum[local_target],
        np.array([2.0, 0.0, 0.0]),
    )


def test_harmful_goal_permission_trials_exhaust_and_are_denied() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        goal_apical_weight=8.0,
        goal_permission_trial_limit=2,
        motif_confirmations=1,
    )
    baseline_source = learner._word_id(b"near", create=True)
    goal_source = learner._word_id(b"far", create=True)
    target = learner._word_id(b"target", create=True)
    assert None not in (baseline_source, goal_source, target)
    baseline = (1, baseline_source)
    conditioned = (16, goal_source)
    for key in (baseline, conditioned):
        learner.route_counts[key] = {target: 8}
        learner.route_totals[key] = 8
        learner.route_utility[key] = 1.0
    learner.goal_route_restoration_sum[baseline] = [0.0, 0.0, 0.0]
    learner.goal_route_restoration_evidence[baseline] = [0.0, 0.0, 1.0]
    learner.goal_route_restoration_sum[conditioned] = [0.0, 0.0, 8.0]
    learner.goal_route_restoration_evidence[conditioned] = [0.0, 0.0, 9.0]
    learner.temporal_clock = 16
    learner.temporal_last_seen[baseline_source] = 16
    learner.temporal_last_seen[goal_source] = 1
    learner.goal_organ.current_mode = 2
    learner._goal_learning_active = True

    assert learner._selected_route_key() == conditioned
    learner._advance_goal_route_credit((), (0.0, 0.0, -1.0), learn=True)
    assert learner._selected_route_key() == conditioned
    learner._advance_goal_route_credit((), (0.0, 0.0, -1.0), learn=True)
    assert learner._selected_route_key() == baseline

    permission = learner._goal_selection_key(2, baseline, conditioned)
    temporal_direction = permission[1]
    assert learner.goal_permission_trials[temporal_direction] == 2
    assert learner.goal_permission_restoration_sum[temporal_direction] < 0.0
    assert learner.goal_route_denials == 1


def test_temporal_disagreement_measures_cross_horizon_consensus() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_third_component="temporal_disagreement",
    )
    sources = [
        learner._word_id(f"source-{index}".encode(), create=True) for index in range(3)
    ]
    targets = [
        learner._word_id(f"target-{index}".encode(), create=True) for index in range(2)
    ]
    assert all(source is not None for source in sources)
    assert all(target is not None for target in targets)
    routes = [(1, sources[0]), (4, sources[1]), (16, sources[2])]
    for key, winner in zip(routes, (targets[0], targets[0], targets[1]), strict=True):
        learner.route_counts[key] = {winner: 8}
        learner.route_totals[key] = 8
        learner.route_utility[key] = 1.0

    assert learner._temporal_disagreement(tuple(routes[:2])) == 0.0
    assert np.isclose(learner._temporal_disagreement(tuple(routes)), 1.0 / 3.0)


def test_restorative_goal_basin_conditions_local_route_competition() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_apical_weight=8.0,
        goal_route_context="restorative_basin",
        goal_reserve_activation=True,
        maximum_goal_route_byte_weight=0.1,
    )
    near_source = learner._word_id(b"near-source", create=True)
    far_source = learner._word_id(b"far-source", create=True)
    goal = learner._word_id(b"goal", create=True)
    distractor = learner._word_id(b"distractor", create=True)
    aligned = learner._word_id(bytes((0xF1, 0x02, 0x7F)), create=True)
    assert None not in (near_source, far_source, goal, distractor, aligned)
    near_route = (1, near_source)
    far_route = (16, far_source)
    learner.route_counts[near_route] = {distractor: 10}
    learner.route_counts[far_route] = {aligned: 10}
    learner.route_totals.update({near_route: 10, far_route: 10})
    learner.route_utility.update({near_route: 0.2, far_route: 0.01})
    learner.goal_organ.component_restoration_sum[goal] = np.array([8.0, 0.0, 0.0])
    learner.goal_organ.component_restoration_sum[aligned] = np.array([8.0, 0.0, 0.0])
    learner.goal_organ.component_restoration_sum[distractor] = np.array([0.0, 8.0, 0.0])
    for assembly in (goal, aligned, distractor):
        learner.goal_organ.component_restoration_evidence[assembly] = np.array(
            [9.0, 9.0, 9.0]
        )
    learner.temporal_clock = 16
    learner.temporal_last_seen[near_source] = 16
    learner.temporal_last_seen[far_source] = 1
    learner.goal_organ.current_goal = goal
    learner.goal_organ.current_mode = 2
    permission = learner._goal_selection_key(2, near_route, far_route)
    learner.goal_selection_utility[permission] = 0.1

    assert learner._route_goal_basin_mass(near_route, goal) == 0.0
    assert learner._route_goal_basin_mass(far_route, goal) == 1.0
    assert learner._selected_route_key() == far_route
    learner.current_word.extend(bytes((0x03, 0x81, 0x00)))
    assert learner._goal_route_next_motif_distribution() == {0xF1: 1.0}
    calls = learner.goal_route_context_calls
    assert learner._goal_route_next_motif_distribution() == {0xF1: 1.0}
    assert learner.goal_route_context_calls == calls
    assert learner._goal_route_next_motif_weight() > 0.0


def test_restorative_goal_can_recruit_one_positive_reserve_route() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_apical_weight=8.0,
        goal_route_context="restorative_basin",
        goal_reserve_activation=True,
        goal_permission_trial_limit=1,
    )
    source = learner._word_id(b"source", create=True)
    goal = learner._word_id(b"goal", create=True)
    aligned = learner._word_id(b"aligned", create=True)
    assert None not in (source, goal, aligned)
    route = (1, source)
    learner.route_counts[route] = {aligned: 10}
    learner.route_totals[route] = 10
    learner.route_utility[route] = 0.01
    for assembly in (goal, aligned):
        learner.goal_organ.component_restoration_sum[assembly] = np.array(
            [8.0, 0.0, 0.0]
        )
        learner.goal_organ.component_restoration_evidence[assembly] = np.array(
            [9.0, 9.0, 9.0]
        )
    learner.temporal_clock = 1
    learner.temporal_last_seen[source] = 1
    learner.goal_organ.current_goal = goal
    learner.goal_organ.current_mode = 0
    learner._goal_learning_active = True

    assert learner._selected_route_key() == route
    assert learner.goal_route_proposals == 1
    assert learner.goal_route_switches == 1


def test_route_restoration_signature_supplies_modality_neutral_goal_alignment() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_route_alignment="restoration_signature",
    )
    goal = learner._word_id(bytes((0xE1, 0x02, 0x03)), create=True)
    source = learner._word_id(bytes((0x91, 0x04, 0x05)), create=True)
    target = learner._word_id(bytes((0x81, 0x06, 0x07)), create=True)
    assert None not in (goal, source, target)
    aligned = (1, source)
    orthogonal = (4, source)
    for key in (aligned, orthogonal):
        learner.route_counts[key] = {target: 8}
        learner.route_totals[key] = 8
        learner.route_utility[key] = 1.0
        learner.goal_route_restoration_evidence[key] = [9.0, 9.0, 9.0]
    learner.goal_organ.component_restoration_sum[goal] = np.array([8.0, 0.0, 0.0])
    learner.goal_organ.component_restoration_evidence[goal] = np.array([9.0, 9.0, 9.0])
    learner.goal_route_restoration_sum[aligned] = [8.0, 0.0, 0.0]
    learner.goal_route_restoration_sum[orthogonal] = [0.0, 8.0, 0.0]

    assert np.isclose(learner._route_goal_alignment(aligned, goal), 1.0)
    assert np.isclose(learner._route_goal_alignment(orthogonal, goal), 0.0)


def test_joint_regulatory_goal_flow_requires_need_and_target_agreement() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_flow_polarity="regulatory_goal",
        goal_route_alignment="restoration_signature",
    )
    goal = learner._word_id(b"goal", create=True)
    source = learner._word_id(b"source", create=True)
    target = learner._word_id(b"target", create=True)
    assert None not in (goal, source, target)
    aligned = (1, source)
    wrong_target = (4, source)
    learner.goal_organ.current_mode = 0
    learner.goal_organ.current_goal = goal
    learner.goal_organ.component_restoration_sum[goal] = np.array([8.0, 0.0, 0.0])
    learner.goal_organ.component_restoration_evidence[goal] = np.array([9.0, 9.0, 9.0])
    learner.goal_route_restoration_sum[aligned] = [8.0, 0.0, 0.0]
    learner.goal_route_restoration_sum[wrong_target] = [0.0, 8.0, 0.0]
    for key in (aligned, wrong_target):
        learner.goal_route_restoration_evidence[key] = [9.0, 9.0, 9.0]

    assert np.isclose(learner._route_goal_alignment(aligned, goal), 1.0)
    assert np.isclose(learner._route_goal_alignment(wrong_target, goal), 0.0)
    learner.goal_organ.current_goal = None
    assert learner._route_goal_alignment(aligned, None) == 0.0


def test_predictive_basin_softly_coordinates_structurally_similar_routes() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        goal_flow_polarity="regulatory_basin",
        predictive_goal_basins=4,
        predictive_goal_weight=0.25,
    )
    source = learner._word_id(bytes((0x01,)), create=True)
    aligned_target = learner._word_id(bytes((0x00,)), create=True)
    opponent_target = learner._word_id(bytes((0xFF,)), create=True)
    assert None not in (source, aligned_target, opponent_target)
    aligned = (1, source)
    opponent = (4, source)
    learner.route_counts[aligned] = {aligned_target: 8}
    learner.route_counts[opponent] = {opponent_target: 8}
    learner.route_totals[aligned] = 8
    learner.route_totals[opponent] = 8
    for key in (aligned, opponent):
        learner.goal_route_restoration_sum[key] = [8.0, 0.0, 0.0]
        learner.goal_route_restoration_evidence[key] = [9.0, 9.0, 9.0]
    learner.goal_organ.current_mode = 0
    assert learner.predictive_goal_organ is not None
    learner.predictive_goal_organ.observe(
        learner._predictive_successor_signature({aligned_target: 8})
    )

    assert np.isclose(learner._route_goal_alignment(aligned, None), 1.0)
    assert np.isclose(learner._route_goal_alignment(opponent, None), 0.75)


def test_transient_binder_reinstates_only_existing_motif_assemblies() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
        route_confirmations=1,
        minimum_route_utility_bits=0.0,
        predictive_goal_basins=4,
        predictive_goal_hysteresis=0.0,
        binding_index_cells=32,
        binding_role_assemblies=8,
        binding_surprise_bits=0.0,
    )
    stream = np.frombuffer(
        b"red fox jumped. blue fox slept. green fox ran. " * 20,
        dtype=np.uint8,
    )

    learner.observe_chunk_probabilities(stream, learn=True)

    assert learner.binding_role_organ is not None
    assert learner.binding_index is not None
    assert learner.binding_role_organ.live_basins > 0
    assert learner.binding_index.live_cells > 0
    assert learner.binding_formation_events > 0
    assert learner.binding_recall_attempts > 0
    assert not hasattr(learner.binding_index, "byte_counts")
    assert all(
        filler < len(learner.word_forms)
        for filler in learner.binding_index.filler
        if filler >= 0
    )


def test_bounded_motif_columns_skip_unadmitted_forms_during_consolidation() -> None:
    settings = dict(
        max_alm_bytes=None,
        max_motif_assemblies=2,
        motif_confirmations=1,
    )
    left = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    right = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    assert left._word_id(b"left-a", create=True) is not None
    assert left._word_id(b"left-b", create=True) is not None
    assert right._word_id(b"right-a", create=True) is not None
    assert right._word_id(b"right-b", create=True) is not None

    left.consolidate(right)

    assert len(left.word_forms) == 2


def test_internal_recall_can_predict_but_cannot_create_route_evidence() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        max_alm_bytes=None,
        motif_confirmations=1,
    )
    observed = learner._word_id(b"observed", create=True)
    recalled = learner._word_id(b"recalled", create=True)
    assert observed is not None and recalled is not None
    learner.temporal_clock = 8
    learner.temporal_last_seen = {observed: 8, recalled: 7}
    learner.temporal_recalled_sources.add(recalled)

    active = learner._active_route_keys()
    learning = learner._learning_route_keys()

    assert any(source == recalled for _, source in active)
    assert all(source != recalled for _, source in learning)
    assert any(source == observed for _, source in learning)


def test_goal_surprise_scope_can_cover_every_raw_byte() -> None:
    alphabetic = GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_surprise_scope="alphabetic"
    )
    modality_neutral = GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_surprise_scope="all_bytes"
    )

    assert alphabetic._goal_scores_byte(ord("A"))
    assert not alphabetic._goal_scores_byte(0xE3)
    assert modality_neutral._goal_scores_byte(ord("A"))
    assert modality_neutral._goal_scores_byte(0xE3)


def test_regulatory_flow_is_invariant_to_opaque_opponent_relabeling() -> None:
    original = np.frombuffer(
        b"red cue apple. blue cue apricot. green cue berry. " * 300,
        dtype=np.uint8,
    ).copy()
    opaque = np.bitwise_xor(original, np.uint8(0xA5))
    settings = dict(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        boundary_surprise_bits=3.0,
        minimum_motif_events=3,
        maximum_motif_events=16,
        temporal_bins=(1, 2, 4, 8, 16),
        route_confirmations=2,
        goal_apical_weight=8.0,
        goal_third_component="temporal_disagreement",
        goal_route_context="restorative_basin",
        maximum_goal_route_byte_weight=0.02,
        goal_route_projection_policy="aligned_flow",
        goal_route_timing="learned_hazard",
        goal_route_alignment="restoration_signature",
        goal_flow_polarity="regulatory_mode",
        goal_surprise_scope="all_bytes",
    )
    plain = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    relabeled = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)

    plain_probability = plain.observe_chunk_probabilities(original)
    opaque_probability = relabeled.observe_chunk_probabilities(opaque)

    np.testing.assert_array_equal(plain_probability, opaque_probability)
    assert plain.goal_route_flow_uses > 0
    assert relabeled.goal_route_flow_uses == plain.goal_route_flow_uses
    assert relabeled.goal_organ.mode_births == plain.goal_organ.mode_births


def test_learned_regulatory_flow_gate_uses_prior_outcomes_causally() -> None:
    learner = GoalConditionedTemporalBasisMotifOpponentLearner(
        goal_flow_mode_gate="learned_utility",
        goal_flow_mode_trials=1,
    )
    learner._goal_learning_active = True

    # One developmental trial is permitted before its consequence is known.
    assert learner._goal_flow_mode_is_permitted(1)
    learner._record_goal_flow_mode_outcome(1, 0.1, 0.05, applied=True)
    assert not learner._goal_flow_mode_is_permitted(1)

    # The exact counterfactual remains observable while the output gate is shut,
    # so a changing stream can reopen the mode without privileged supervision.
    learner._record_goal_flow_mode_outcome(1, 0.1, 0.4, applied=False)
    assert learner._goal_flow_mode_is_permitted(1)
    assert learner.goal_flow_mode_evidence.tolist() == [0, 2, 0]
    assert learner.goal_flow_mode_candidates.tolist() == [0, 2, 0]
    assert learner.goal_flow_mode_applications.tolist() == [0, 1, 0]


def test_regulatory_flow_permission_synapses_consolidate_across_columns() -> None:
    settings = {
        "goal_flow_mode_gate": "learned_utility",
        "goal_flow_mode_trials": 8,
    }
    left = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    right = GoalConditionedTemporalBasisMotifOpponentLearner(**settings)
    left._goal_learning_active = True
    right._goal_learning_active = True
    left._record_goal_flow_mode_outcome(0, 0.1, 0.2, applied=True)
    right._record_goal_flow_mode_outcome(0, 0.2, 0.1, applied=True)
    right._record_goal_flow_mode_outcome(2, 0.1, 0.4, applied=False)

    left.consolidate(right)

    np.testing.assert_allclose(left.goal_flow_mode_utility, [0.0, 0.0, 2.0])
    assert left.goal_flow_mode_evidence.tolist() == [2, 0, 1]
    assert left.goal_flow_mode_candidates.tolist() == [2, 0, 1]
    assert left.goal_flow_mode_applications.tolist() == [2, 0, 0]


def test_incremental_motif_filter_matches_full_prefix_scan_exactly() -> None:
    stream = np.random.default_rng(40).integers(0, 32, size=30_000, dtype=np.uint8)
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
        "event_confirmations": 2,
    }
    incremental = PredictiveMotifOpponentLearner(**settings)
    legacy = _LegacyPrefixScanMotifLearner(**settings)

    incremental_probability = incremental.observe_chunk_probabilities(stream)
    legacy_probability = legacy.observe_chunk_probabilities(stream)

    np.testing.assert_array_equal(incremental_probability, legacy_probability)
    incremental.reset_episode()
    legacy.reset_episode()
    incremental_frozen = incremental.observe_chunk_probabilities(stream, learn=False)
    legacy_frozen = legacy.observe_chunk_probabilities(stream, learn=False)
    np.testing.assert_array_equal(incremental_frozen, legacy_frozen)
    assert incremental.event_receptor_cache
    restored = pickle.loads(pickle.dumps(incremental, protocol=5))
    assert not restored.event_receptor_cache


def test_weak_event_dendrites_lose_sleep_competition() -> None:
    rng = np.random.default_rng(41)
    motifs = np.asarray(
        [
            [index, index * 17 % 251, index * 31 % 251, index * 47 % 251]
            for index in range(1, 101)
        ],
        dtype=np.uint8,
    )
    train = motifs[rng.integers(0, len(motifs), size=2500)].reshape(-1)
    learner = PredictiveMotifOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        sequence_confirmations=2,
        event_confirmations=2,
        boundary_surprise_bits=100.0,
        minimum_motif_events=4,
        maximum_motif_events=4,
    )
    learner.observe_chunk_probabilities(train, learn=True)
    learner.shed_sequence_nursery()
    learner.shed_event_nursery()
    before = learner.embodied_bytes
    branches_before = learner.live_event_count
    synapses_before = learner.live_event_synapse_count

    removed_branches, removed_synapses = learner.consolidate_event_anatomy(
        minimum_observations=3
    )
    after = learner.embodied_bytes

    assert removed_branches > 0
    assert removed_synapses > 0
    assert after < before
    assert learner.live_event_count == branches_before - removed_branches
    assert learner.live_event_synapse_count == synapses_before - removed_synapses


def test_awake_nursery_competition_bounds_immature_anatomy() -> None:
    stream = np.random.default_rng(44).integers(0, 256, size=10_000, dtype=np.uint8)
    settings = dict(
        confirmations=1,
        context_confirmations=1,
        surprise_bits=0.0,
        sequence_confirmations=100,
        max_sequence_nursery_paths=8,
        event_confirmations=100,
        max_event_nursery_paths=8,
        boundary_surprise_bits=100.0,
        minimum_motif_events=4,
        maximum_motif_events=4,
    )
    learner = PredictiveMotifOpponentLearner(**settings)

    learner.observe_chunk_probabilities(stream, learn=True)

    assert all(len(bank) <= 8 for bank in learner.sequence_nursery)
    assert all(len(bank) <= 8 for bank in learner.event_nursery)
    assert learner.sequence_nursery_evictions > 0
    assert learner.event_nursery_evictions > 0

    other = PredictiveMotifOpponentLearner(**settings)
    other.observe_chunk_probabilities(stream[::-1], learn=True)
    learner.consolidate(other)

    assert all(len(bank) <= 8 for bank in learner.sequence_nursery)
    assert all(len(bank) <= 8 for bank in learner.event_nursery)


def test_contextual_organ_resumes_with_temporal_state_intact() -> None:
    stream = np.resize(np.asarray((65, 66, 67, 68, 66, 69), dtype=np.uint8), 2000)
    uninterrupted = ContextualOpponentAttractorLearner()
    resumed = ContextualOpponentAttractorLearner()

    expected = uninterrupted.observe_chunk_probabilities(stream)
    first = resumed.observe_chunk_probabilities(stream[:731])
    resumed = pickle.loads(pickle.dumps(resumed, protocol=5))
    actual = np.concatenate((first, resumed.observe_chunk_probabilities(stream[731:])))

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert resumed.embodied_bytes == uninterrupted.embodied_bytes


def test_phrase_event_organ_resumes_midword_with_active_hypotheses() -> None:
    stream = np.frombuffer(b"red cue apple. blue cue apricot. " * 200, dtype=np.uint8)
    uninterrupted = PhraseEventOpponentLearner(event_confirmations=2)
    resumed = PhraseEventOpponentLearner(event_confirmations=2)

    expected = uninterrupted.observe_chunk_probabilities(stream)
    first = resumed.observe_chunk_probabilities(stream[:2003])
    assert resumed.current_word
    resumed = pickle.loads(pickle.dumps(resumed, protocol=5))
    actual = np.concatenate((first, resumed.observe_chunk_probabilities(stream[2003:])))

    np.testing.assert_array_equal(actual, expected)
    assert resumed.embodied_bytes == uninterrupted.embodied_bytes


def test_contextual_columns_consolidate_sparse_anatomy() -> None:
    left = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )
    right = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )
    left.observe_chunk_probabilities(
        np.resize(np.asarray((65, 66, 67), dtype=np.uint8), 600)
    )
    right.observe_chunk_probabilities(
        np.resize(np.asarray((68, 69, 70), dtype=np.uint8), 600)
    )
    left_steps = left.steps
    right_steps = right.steps

    left.consolidate(right)

    assert left.steps == left_steps + right_steps
    assert left.previous_assembly is None
    assert left.older_assembly is None
    assert left.live_assembly_count == 6
    assert left.live_context_count >= 6
    assert left.transition_counts[65][66] > 0
    assert left.transition_counts[68][69] > 0


def test_sequence_columns_consolidate_recursive_temporal_anatomy() -> None:
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "max_context_bytes": 4,
        "sequence_confirmations": 2,
    }
    left = SequenceCortexOpponentLearner(**settings)
    right = SequenceCortexOpponentLearner(**settings)
    left.observe_chunk_probabilities(
        np.resize(np.asarray((65, 66, 67, 68), dtype=np.uint8), 800)
    )
    right.observe_chunk_probabilities(
        np.resize(np.asarray((87, 88, 89, 90), dtype=np.uint8), 800)
    )

    left.consolidate(right)

    assert left.sequence_history == []
    assert left.live_sequence_count > 0
    assert any(68 in row for row in left.sequence_counts[0].values())
    assert any(90 in row for row in left.sequence_counts[0].values())
    assert any(65 in row for row in left.sequence_counts[1].values())
    assert any(87 in row for row in left.sequence_counts[1].values())
    assert all(
        totals[context] == sum(row.values())
        for bank, totals in zip(left.sequence_counts, left.sequence_totals, strict=True)
        for context, row in bank.items()
    )


def test_sequence_delta_consolidation_does_not_duplicate_shared_ancestry() -> None:
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "max_context_bytes": 5,
        "sequence_confirmations": 2,
    }
    ancestor = SequenceCortexOpponentLearner(**settings)
    ancestor.observe_chunk_probabilities(
        np.frombuffer(b"shared cortical ancestry. " * 20, dtype=np.uint8)
    )
    ancestor.reset_episode()
    left = pickle.loads(pickle.dumps(ancestor, protocol=5))
    right = pickle.loads(pickle.dumps(ancestor, protocol=5))
    left.observe_chunk_probabilities(
        np.frombuffer(b"Mia carried the red ball. " * 12, dtype=np.uint8)
    )
    right.observe_chunk_probabilities(
        np.frombuffer(b"Leo carried the blue kite. " * 12, dtype=np.uint8)
    )
    merged = pickle.loads(pickle.dumps(ancestor, protocol=5))

    merged.consolidate_delta(left, ancestor)
    merged.consolidate_delta(right, ancestor)

    expected_unigrams = (
        left.unigram_counts + right.unigram_counts - ancestor.unigram_counts
    )
    np.testing.assert_array_equal(merged.unigram_counts, expected_unigrams)
    for source in range(256):
        outputs = (
            set(ancestor.transition_counts[source])
            | set(left.transition_counts[source])
            | set(right.transition_counts[source])
        )
        for target in outputs:
            expected = (
                left.transition_counts[source].get(target, 0)
                + right.transition_counts[source].get(target, 0)
                - ancestor.transition_counts[source].get(target, 0)
            )
            assert merged.transition_counts[source].get(target, 0) == expected
    assert merged.steps == left.steps + right.steps - ancestor.steps
    assert merged.live_sequence_count >= max(
        left.live_sequence_count, right.live_sequence_count
    )
    assert merged.sequence_history == []


def test_sequence_delta_consolidation_reproduces_one_descendant() -> None:
    ancestor = SequenceCortexOpponentLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        max_context_bytes=4,
        sequence_confirmations=2,
    )
    ancestor.observe_chunk_probabilities(
        np.frombuffer(b"the common body. " * 20, dtype=np.uint8)
    )
    ancestor.reset_episode()
    descendant = pickle.loads(pickle.dumps(ancestor, protocol=5))
    descendant.begin_lineage_tracking()
    descendant.observe_chunk_probabilities(
        np.frombuffer(b"the novel experience. " * 20, dtype=np.uint8)
    )
    merged = pickle.loads(pickle.dumps(ancestor, protocol=5))

    merged.consolidate_delta(descendant, ancestor)

    np.testing.assert_array_equal(merged.unigram_counts, descendant.unigram_counts)
    assert merged.transition_counts == descendant.transition_counts
    assert merged.residual_counts == descendant.residual_counts
    assert merged.context_nursery == descendant.context_nursery
    assert merged.sequence_counts == descendant.sequence_counts
    assert merged.sequence_nursery == descendant.sequence_nursery
    assert merged.steps == descendant.steps


def test_phrase_columns_remap_word_assemblies_by_receptor_form() -> None:
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
        "event_confirmations": 2,
    }
    left = PhraseEventOpponentLearner(**settings)
    right = PhraseEventOpponentLearner(**settings)
    left.observe_chunk_probabilities(
        np.frombuffer(b"red cue apple. red cue apple. ", dtype=np.uint8)
    )
    right.observe_chunk_probabilities(
        np.frombuffer(b"blue cue apricot. blue cue apricot. ", dtype=np.uint8)
    )

    left.consolidate(right)

    assert b"cue" in left.word_ids
    assert left.word_forms.count(b"cue") == 1
    assert b"red" in left.word_ids
    assert b"blue" in left.word_ids
    assert all(
        target_id < left.live_word_count
        for bank in left.event_counts
        for row in bank.values()
        for target_id in row
    )
    assert all(
        totals[context] == sum(row.values())
        for bank, totals in zip(left.event_counts, left.event_totals, strict=True)
        for context, row in bank.items()
    )


def test_phrase_consolidation_decodes_context_with_source_vocabulary() -> None:
    settings = {
        "confirmations": 1,
        "context_confirmations": 2,
        "surprise_bits": 0.0,
        "sequence_confirmations": 2,
        "event_confirmations": 1,
    }
    left = PhraseEventOpponentLearner(**settings)
    right = PhraseEventOpponentLearner(**settings)
    left.observe_chunk_probabilities(
        np.frombuffer(b"one two. one two. ", dtype=np.uint8)
    )
    right.observe_chunk_probabilities(
        np.frombuffer(
            b"alpha beta gamma delta. alpha beta gamma delta. ", dtype=np.uint8
        )
    )

    left.consolidate(right)

    assert all(
        word in left.word_ids for word in (b"alpha", b"beta", b"gamma", b"delta")
    )
    assert all(
        target < left.live_word_count
        for bank in left.event_counts
        for row in bank.values()
        for target in row
    )


def test_developmental_sleep_sheds_only_silent_nursery_paths() -> None:
    train = np.frombuffer(
        b"red cue apple. blue cue apricot. green cue aprium. " * 20
        + b"solitary nebula quasar.",
        dtype=np.uint8,
    )
    valid = np.frombuffer(b"red cue apple. blue cue apricot. ", dtype=np.uint8)
    awake = PhraseEventOpponentLearner(event_confirmations=3)
    awake.observe_chunk_probabilities(train)
    sleeping = pickle.loads(pickle.dumps(awake, protocol=5))

    before = sleeping.embodied_bytes
    shed_sequence = sleeping.shed_sequence_nursery()
    shed_events = sleeping.shed_event_nursery()
    sleeping_probability = sleeping.observe_chunk_probabilities(valid, learn=False)
    awake_probability = awake.observe_chunk_probabilities(valid, learn=False)

    assert shed_sequence + shed_events > 0
    assert sleeping.embodied_bytes < before
    np.testing.assert_array_equal(sleeping_probability, awake_probability)


def test_fixed_delay_control_resolves_shared_local_context() -> None:
    stream = np.resize(
        np.asarray((65, 88, 66, 67, 68, 69, 88, 66, 67, 70), dtype=np.uint8),
        10_000,
    )
    local = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )
    coherent = DelayedOpponentAttractorControl(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        fiber_lags=(4,),
        fiber_confirmations=2,
        utility_decay=0.9,
    )

    local_probability = local.observe_chunk_probabilities(stream)
    coherent_probability = coherent.observe_chunk_probabilities(stream)

    assert coherent.live_fiber_count > 0
    assert (
        -np.log2(coherent_probability[-1000:]).mean()
        < -np.log2(local_probability[-1000:]).mean()
    )


def test_content_index_resolves_story_specific_ambiguity() -> None:
    marker = b"<|endoftext|>\n"
    lily = b"lily lily lily qxA qxA " + marker
    beep = b"beep beep beep qxB qxB " + marker
    stream = np.frombuffer((lily + beep) * 300, dtype=np.uint8)
    local = ContextualOpponentAttractorLearner(
        confirmations=1, context_confirmations=2, surprise_bits=0.0
    )
    episodic = EpisodicOpponentAttractorLearner(
        confirmations=1,
        context_confirmations=2,
        surprise_bits=0.0,
        content_confirmations=2,
        episodic_confirmations=2,
        utility_decay=0.9,
    )

    local_probability = local.observe_chunk_probabilities(stream)
    episodic_probability = episodic.observe_chunk_probabilities(stream)

    assert episodic.live_content_count > 0
    assert episodic.live_episode_count > 0
    ambiguous = ((stream == ord("A")) | (stream == ord("B"))) & (
        np.roll(stream, 1) == ord("x")
    )
    ambiguous[:1000] = False
    assert -np.log2(episodic_probability[ambiguous]).mean() < 0.1
    assert (
        -np.log2(episodic_probability[ambiguous]).mean()
        < -np.log2(local_probability[ambiguous]).mean()
    )


def test_episodic_consolidation_removes_only_silent_mature_branches() -> None:
    learner = EpisodicOpponentAttractorLearner(minimum_utility_bits=0.05)
    learner.episodic_counts = {(0, 65): {66: 10}, (1, 65): {67: 10}}
    learner.episodic_totals = {(0, 65): 10, (1, 65): 10}
    learner.episodic_utility = {(0, 65): -0.1, (1, 65): 0.2}

    pruned = learner.consolidate_episodes()

    assert pruned == 1
    assert (0, 65) not in learner.episodic_counts
    assert learner.episodic_counts[(1, 65)] == {67: 10}


def test_episodic_columns_align_content_by_receptor_path() -> None:
    left = EpisodicOpponentAttractorLearner(
        confirmations=1, content_confirmations=1, episodic_confirmations=1
    )
    right = EpisodicOpponentAttractorLearner(
        confirmations=1, content_confirmations=1, episodic_confirmations=1
    )
    left.observe_chunk_probabilities(
        np.frombuffer(b"lily lily was kind. ", dtype=np.uint8)
    )
    right.observe_chunk_probabilities(
        np.frombuffer(b"beep met lily. beep smiled. ", dtype=np.uint8)
    )

    left.consolidate(right)

    assert b"lily" in left.content_ids
    assert b"beep" in left.content_ids
    assert left.content_forms.count(b"lily") == 1
    assert left.content_counts[left.content_ids[b"lily"]] >= 3
    assert all(
        content_id < left.live_content_count for content_id, _ in left.episodic_counts
    )
