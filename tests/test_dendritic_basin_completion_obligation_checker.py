from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import pytest

from manifoldflow.experiments import (
    dendritic_basin_completion_obligation_checker as assay,
)


def _canonical_claims(
    world: assay.CompletionObligationWorld,
) -> dict[str, bytes]:
    return {
        role: assay.render_claim(world.entity, role, value)
        for role, value in world.required_role_to_value
    }


def test_pair_is_deterministic_lexically_balanced_and_long_range() -> None:
    pair = assay.build_completion_obligation_pair(7, interruption_bytes=640)
    repeated = assay.build_completion_obligation_pair(7, interruption_bytes=640)
    assert pair == repeated
    assert pair != assay.build_completion_obligation_pair(8, interruption_bytes=640)

    a, b = pair.worlds
    assert pair.pair_id == "completion-obligation-pair-000007"
    assert a.world_id == f"{pair.pair_id}-a"
    assert b.world_id == f"{pair.pair_id}-b"
    assert pair.counterpart(a) == b
    assert pair.counterpart(b.world_id) == a
    assert a.counterpart_world_id == b.world_id
    assert b.counterpart_world_id == a.world_id

    assert a.prompt != b.prompt
    assert len(a.prompt) == len(b.prompt)
    assert Counter(a.prompt) == Counter(b.prompt)
    assert assay.lexical_multiset(a.prompt) == assay.lexical_multiset(b.prompt)
    assert a.common_suffix == b.common_suffix
    assert a.interruption == b.interruption
    assert len(a.interruption) == 640
    assert a.prompt.endswith(a.common_suffix)
    assert b.prompt.endswith(b.common_suffix)
    assert a.prompt[-512:] == b.prompt[-512:]
    assert a.final_resume_request == b.final_resume_request
    assert a.common_suffix.endswith(a.final_resume_request)

    prefix_a = a.prompt[: -len(a.common_suffix)]
    prefix_b = b.prompt[: -len(b.common_suffix)]
    assert prefix_a.count(b"ANCHOR ENTITY") == 3
    assert prefix_a.count(b"CORRECTION ENTITY") == 3
    assert prefix_b.count(b"ANCHOR ENTITY") == 3
    assert prefix_b.count(b"CORRECTION ENTITY") == 3
    assert a.entity.encode("ascii") in prefix_a
    assert b.entity.encode("ascii") in prefix_b

    mapping_a = a.required_mapping
    mapping_b = b.required_mapping
    assert tuple(mapping_a) == assay.ROLES
    assert set(mapping_a.values()) == set(assay.VALUES)
    assert all(mapping_a[role] != mapping_b[role] for role in assay.ROLES)
    assay.validate_completion_obligation_pair(pair)


def test_all_six_full_clause_orders_are_exactly_accepted() -> None:
    world = assay.build_completion_obligation_pair().world_a
    accepted = world.accepted_full_completions
    assert len(accepted) == 6
    assert len(set(accepted)) == 6
    assert accepted == assay.render_full_completions(
        world.entity, world.required_role_to_value
    )
    for generated in accepted:
        assert generated.endswith(assay.TERMINATOR)
        result = assay.check_completion(world, generated)
        assert result.completed_obligations == assay.ROLES
        assert result.missing_obligations == ()
        assert result.wrong_claims == ()
        assert result.unsupported_claims == ()
        assert result.conflicting_repeats == ()
        assert result.unparsed_content == ()
        assert result.explicit_termination
        assert result.termination_count == 1
        assert not result.terminated_early
        assert not result.trailing_content
        assert result.trailing_bytes == b""
        assert result.exact_accepted


def test_consistency_only_control_contains_every_ordered_true_subset() -> None:
    world = assay.build_completion_obligation_pair().world_a
    sequences = world.consistency_accepted_sequences
    assert len(sequences) == 16
    assert len(set(sequences)) == 16
    assert assay.TERMINATOR in sequences
    assert sum(
        assay.check_completion(world, generated).exact_accepted
        for generated in sequences
    ) == 6
    observed_completed_sets = set()
    for generated in sequences:
        result = assay.check_completion(world, generated)
        observed_completed_sets.add(frozenset(result.completed_obligations))
        assert result.wrong_claims == ()
        assert result.unsupported_claims == ()
        assert result.conflicting_repeats == ()
        assert result.unparsed_content == ()
        assert result.explicit_termination
        assert not result.trailing_content
    assert len(observed_completed_sets) == 2 ** len(assay.ROLES)


def test_marginal_unbound_control_enumerates_every_binding_and_order() -> None:
    pair = assay.build_completion_obligation_pair()
    a, b = pair.worlds
    sequences = a.marginal_unbound_sequences
    assert sequences == b.marginal_unbound_sequences
    assert sequences == assay.render_marginal_unbound_sequences(a.entity)
    assert len(sequences) == 36
    assert len(set(sequences)) == 36
    accepted_a = sum(
        assay.check_completion(a, value).exact_accepted for value in sequences
    )
    accepted_b = sum(
        assay.check_completion(b, value).exact_accepted for value in sequences
    )
    assert accepted_a == 6
    assert accepted_b == 6
    for generated in sequences:
        result = assay.check_completion(a, generated)
        assert len(result.claims) == 3
        assert result.unsupported_claims == ()
        assert result.conflicting_repeats == ()
        assert result.explicit_termination
        assert not result.trailing_content


def test_missing_clause_reports_an_unfinished_obligation_and_early_done() -> None:
    world = assay.build_completion_obligation_pair().world_a
    claims = _canonical_claims(world)
    generated = claims["alpha"] + claims["beta"] + assay.TERMINATOR
    result = assay.check_completion(world, generated)
    assert result.completed_obligations == ("alpha", "beta")
    assert result.missing_obligations == ("gamma",)
    assert result.terminated_early
    assert not result.exact_accepted


def test_swapped_binding_reports_wrong_and_missing_obligations() -> None:
    world = assay.build_completion_obligation_pair().world_a
    mapping = world.required_mapping
    swapped = {
        "alpha": mapping["beta"],
        "beta": mapping["alpha"],
        "gamma": mapping["gamma"],
    }
    generated = assay.render_full_completions(world.entity, swapped)[0]
    result = assay.check_completion(world, generated)
    assert tuple(claim.role for claim in result.wrong_claims) == ("alpha", "beta")
    assert result.completed_obligations == ("gamma",)
    assert result.missing_obligations == ("alpha", "beta")
    assert result.unsupported_claims == ()
    assert not result.exact_accepted


@pytest.mark.parametrize(
    ("claim", "reason"),
    (
        (b"ENTITY Other ROLE alpha VALUE red\n", "unknown_entity"),
        (b"ENTITY Nera000000 ROLE delta VALUE red\n", "unknown_role"),
        (b"ENTITY Nera000000 ROLE alpha VALUE amber\n", "unknown_value"),
    ),
)
def test_unknown_entity_role_and_value_are_unsupported(
    claim: bytes, reason: str
) -> None:
    world = assay.build_completion_obligation_pair().world_a
    generated = claim + world.accepted_full_completions[0]
    result = assay.check_completion(world, generated)
    assert len(result.unsupported_claims) == 1
    assert result.unsupported_claims[0].reasons == (reason,)
    assert result.completed_obligations == assay.ROLES
    assert not result.exact_accepted


def test_conflicting_repeat_is_distinct_from_a_missing_obligation() -> None:
    pair = assay.build_completion_obligation_pair()
    world = pair.world_a
    correct = _canonical_claims(world)
    wrong_alpha = assay.render_claim(
        world.entity, "alpha", pair.world_b.required_mapping["alpha"]
    )
    generated = (
        correct["alpha"]
        + wrong_alpha
        + correct["beta"]
        + correct["gamma"]
        + assay.TERMINATOR
    )
    result = assay.check_completion(world, generated)
    assert result.completed_obligations == assay.ROLES
    assert result.missing_obligations == ()
    assert len(result.wrong_claims) == 1
    assert len(result.conflicting_repeats) == 1
    assert result.conflicting_repeats[0].role == "alpha"
    assert set(result.conflicting_repeats[0].values) == {
        world.required_mapping["alpha"],
        pair.world_b.required_mapping["alpha"],
    }
    assert not result.exact_accepted


def test_immediate_done_is_explicit_but_early() -> None:
    world = assay.build_completion_obligation_pair().world_a
    result = assay.check_completion(world, assay.TERMINATOR)
    assert result.completed_obligations == ()
    assert result.missing_obligations == assay.ROLES
    assert result.explicit_termination
    assert result.terminated_early
    assert not result.trailing_content
    assert not result.exact_accepted


def test_missing_done_is_not_explicitly_terminated() -> None:
    world = assay.build_completion_obligation_pair().world_a
    generated = world.accepted_full_completions[0][: -len(assay.TERMINATOR)]
    result = assay.check_completion(world, generated)
    assert result.completed_obligations == assay.ROLES
    assert result.missing_obligations == ()
    assert not result.explicit_termination
    assert result.termination_count == 0
    assert not result.terminated_early
    assert not result.exact_accepted


def test_content_after_done_is_reported_as_trailing_bytes() -> None:
    world = assay.build_completion_obligation_pair().world_a
    generated = world.accepted_full_completions[0] + b"EXTRA\n"
    result = assay.check_completion(world, generated)
    assert result.explicit_termination
    assert result.trailing_content
    assert result.trailing_bytes == b"EXTRA\n"
    assert not result.exact_accepted


def test_malformed_content_is_never_silently_accepted() -> None:
    world = assay.build_completion_obligation_pair().world_a
    generated = b"perhaps\n" + world.accepted_full_completions[0]
    result = assay.check_completion(world, generated)
    assert result.unparsed_content == (b"perhaps\n",)
    assert not result.exact_accepted


def test_crossed_evaluation_requires_the_pair_specific_binding() -> None:
    pair = assay.build_completion_obligation_pair()
    generated_a = pair.world_a.accepted_full_completions[5]
    generated_b = pair.world_b.accepted_full_completions[2]
    crossed = assay.check_crossed_pair(pair, generated_a, generated_b)
    assert crossed == assay.evaluate_crossed_pair(pair, generated_a, generated_b)
    assert crossed.a_on_a.exact_accepted
    assert crossed.b_on_b.exact_accepted
    assert not crossed.a_on_b.exact_accepted
    assert not crossed.b_on_a.exact_accepted
    assert crossed.matched_exact_count == 2
    assert crossed.crossed_exact_count == 0
    assert crossed.matched_pair_accepted
    assert not crossed.crossed_pair_accepted
    assert crossed.binding_sensitive_pair_accepted


def test_checker_surface_accepts_only_world_record_and_generated_bytes() -> None:
    signature = inspect.signature(assay.check_completion)
    assert tuple(signature.parameters) == ("world", "generated")


def test_checker_does_not_reuse_the_precompiled_accepted_set() -> None:
    pair = assay.build_completion_obligation_pair()
    world = pair.world_a
    valid = world.accepted_full_completions[0]
    corrupt = replace(
        world,
        accepted_full_completions=(b"unsupported but precompiled\nDONE\n",),
    )
    assert assay.check_completion(corrupt, valid).exact_accepted
    assert not assay.check_completion(
        corrupt, corrupt.accepted_full_completions[0]
    ).exact_accepted
    corrupt_pair = replace(pair, world_a=corrupt)
    crossed = assay.check_crossed_pair(
        corrupt_pair, valid, pair.world_b.accepted_full_completions[0]
    )
    assert crossed.binding_sensitive_pair_accepted
    with pytest.raises(ValueError, match="accepted full-completion"):
        assay.validate_completion_obligation_pair(corrupt_pair)


@pytest.mark.parametrize(
    ("call", "exception"),
    (
        (lambda: assay.build_completion_obligation_pair(-1), ValueError),
        (
            lambda: assay.build_completion_obligation_pair(interruption_bytes=511),
            ValueError,
        ),
        (lambda: assay.build_completion_obligation_pairs(-1), ValueError),
        (lambda: assay.render_claim("bad entity", "alpha", "red"), ValueError),
        (
            lambda: assay.render_full_completions(
                "Nera", {"alpha": "red", "beta": "blue"}
            ),
            ValueError,
        ),
    ),
)
def test_invalid_generator_inputs_fail_closed(
    call: Callable[[], object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        call()
