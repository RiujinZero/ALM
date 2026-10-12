"""Test delayed sensory teaching of a query-to-engram access pathway.

This is a controlled acquisition diagnostic, not a held-out ALM score.  A
query leaves local dendritic eligibility.  Later externally observed identity
activity strengthens only access paths into engrams that had already bound
that identity.  The same query is then replayed without its answer.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from manifoldflow.dendritic_basin.relational_access import RelationalAccessField
from manifoldflow.experiments.dendritic_basin_phase76_recurrent_field import (
    _sensory_context,
    _temporal_cortical_context,
)
from manifoldflow.experiments.dendritic_basin_phase77_generation_quality import (
    CASES,
    GenerationCase,
)
from manifoldflow.experiments.dendritic_basin_phase80_expression_ceiling import (
    _common_prefix,
    _first_word,
)
from manifoldflow.experiments.dendritic_basin_phase81_cortical_expression import (
    _calibrated_path,
    _evaluate,
    _natural_signal,
)
from manifoldflow.experiments.dendritic_basin_phase82_multiscale_capture import (
    _build_multiscale,
    _prime,
)


def _query_context(arm: object) -> tuple[tuple[int, int], ...]:
    return _temporal_cortical_context(
        tuple(arm.route_history),
        include_identity=getattr(arm, "identity_context", False),
        identity_receptor_limit=getattr(arm, "identity_receptor_limit", 8),
    ) + _sensory_context(
        int.from_bytes(arm.raw_window, "big") if arm.raw_window else None
    )


def _teach(
    arm: object,
    access: RelationalAccessField,
    multiscale: object,
    query: tuple[tuple[int, int], ...],
    observed_word: bytes,
    item: tuple[int, int],
) -> dict[str, object]:
    access.propose(arm.field.cortical_code(query))
    externally_reactivated = False
    for byte_id in observed_word:
        access.advance()
        activations = multiscale.observe(byte_id)
        if any(activation.assembly == item[0] for activation in activations):
            externally_reactivated = True
    engrams = arm.field.engrams_containing(item) if externally_reactivated else ()
    compatibility = arm.field.engram_cortical_support(query, engrams)
    # Simultaneously reactivated occurrences inhibit one another while the
    # still-live query trace supplies the differentiating basal current.
    winners = (
        (
            max(
                engrams,
                key=lambda engram: (compatibility[engram], -engram),
            ),
        )
        if engrams
        else ()
    )
    learned = access.confirm(item, winners, external=externally_reactivated)
    return {
        "externally_reactivated": externally_reactivated,
        "eligible_engrams": len(engrams),
        "engram_compatibility": {
            str(engram): compatibility[engram] for engram in engrams
        },
        "competitive_winners": list(winners),
        "learned_paths": learned,
    }


def _retrieve_with_access(
    arm: object,
    access: RelationalAccessField,
    query: tuple[tuple[int, int], ...],
) -> tuple[object | None, object, dict[str, float]]:
    signal = access.activate(
        arm.field.cortical_code(query), len(arm.field.engram_units)
    )
    organ = getattr(arm.learner, "predictive_goal_organ", None)
    basin = None if organ is None else organ.current_basin
    completion = arm.field.retrieve(
        query,
        context_id=basin,
        external_apical_support=signal.engram_support,
        access_content=signal.content_support,
    )
    if completion is not None:
        arm.field.reinstate(completion)
    return arm.field.return_signal(completion), signal, dict(arm.field.last_diagnostics)


def _run_case(
    payload: dict[str, object],
    case: GenerationCase,
    *,
    frames_per_capsule: int,
    items_per_frame: int,
    expression_gain: float,
) -> dict[str, object]:
    expected = case.expected.encode()
    contrast = case.contrast.encode()
    shared = _common_prefix(expected, contrast)
    target_word = _first_word(expected[len(shared) :])
    contrast_word = _first_word(contrast[len(shared) :])
    learner = payload["learner"]
    target_id = learner.word_ids.get(target_word)
    contrast_id = learner.word_ids.get(contrast_word)
    if target_id is None or contrast_id is None:
        return {"case": asdict(case), "available": False}
    multiscale = _build_multiscale(learner, trace_half_life_bytes=32.0)
    arm, _ = _prime(
        payload,
        multiscale,
        case.prompt.encode() + shared,
        frames_per_capsule=frames_per_capsule,
        items_per_frame=items_per_frame,
    )
    target = (int(target_id), 0)
    swapped = (int(contrast_id), 0)
    query = _query_context(arm)
    path = _calibrated_path(learner, (target, swapped), repetitions=24)
    conditions = {}
    for name, taught_word, taught_item in (
        ("correct_identity", target_word, target),
        ("identity_swapped", contrast_word, swapped),
    ):
        test_arm = pickle.loads(pickle.dumps(arm, protocol=pickle.HIGHEST_PROTOCOL))
        access = RelationalAccessField(
            test_arm.field.cortical_dimensions,
            trace_decay=0.95,
            activation_threshold=0.3,
        )
        teacher_multiscale = _build_multiscale(learner, trace_half_life_bytes=0.0)
        teacher_multiscale.reset_episode()
        teaching = _teach(
            test_arm,
            access,
            teacher_multiscale,
            query,
            taught_word,
            taught_item,
        )
        returned, signal, diagnostics = _retrieve_with_access(test_arm, access, query)
        returned_items = () if returned is None else returned.items
        selected = returned_items[0] if returned_items else None
        expression = _evaluate(
            test_arm,
            path,
            target=target_word,
            contrast=contrast_word,
            mode="access",
            item=selected,
            gain=expression_gain,
            generate=True,
        )
        conditions[name] = {
            "teaching": teaching,
            "access_body_bytes": access.embodied_bytes,
            "active_engrams": int(np.count_nonzero(signal.engram_support)),
            "maximum_access_support": float(
                np.max(signal.engram_support, initial=0.0)
            ),
            "returned_items": [
                {
                    "assembly": item[0],
                    "receptor": item[1],
                    "form": learner.word_forms[item[0]].decode(
                        "utf-8", errors="backslashreplace"
                    ),
                }
                for item in returned_items
            ],
            "returns_target": target in returned_items,
            "returns_swapped": swapped in returned_items,
            "field_diagnostics": diagnostics,
            "expression": expression,
        }
    natural_arm = pickle.loads(pickle.dumps(arm, protocol=pickle.HIGHEST_PROTOCOL))
    _, natural = _natural_signal(natural_arm)
    natural_items = () if natural is None else natural.items
    return {
        "case": asdict(case),
        "available": True,
        "target": target_word.decode(),
        "contrast": contrast_word.decode(),
        "target_bound_engrams": len(arm.field.engrams_containing(target)),
        "contrast_bound_engrams": len(arm.field.engrams_containing(swapped)),
        "natural_returns_target": target in natural_items,
        "conditions": conditions,
    }


def run_probe(
    state: Path,
    *,
    frames_per_capsule: int = 24,
    items_per_frame: int = 16,
    expression_gain: float = 0.5,
    cases: tuple[GenerationCase, ...] = CASES,
) -> dict[str, object]:
    with state.open("rb") as handle:
        payload = pickle.load(handle)
    payload["field"].sequence_settling_steps = 4
    payload["identity_context"] = True
    payload["identity_receptor_limit"] = -1
    results = [
        _run_case(
            payload,
            case,
            frames_per_capsule=frames_per_capsule,
            items_per_frame=items_per_frame,
            expression_gain=expression_gain,
        )
        for case in cases
    ]
    available = [result for result in results if result["available"]]
    correct = [result["conditions"]["correct_identity"] for result in available]
    swapped = [result["conditions"]["identity_swapped"] for result in available]
    return {
        "task": "phase83_delayed_relational_access",
        "diagnostic_only": True,
        "state": str(state),
        "external_answer_used_only_for_access_acquisition": True,
        "recalled_activity_can_teach": False,
        "summary": {
            "available_cases": len(available),
            "natural_target_return_fraction": float(
                np.mean([result["natural_returns_target"] for result in available])
            )
            if available
            else 0.0,
            "learned_target_return_fraction": float(
                np.mean([result["returns_target"] for result in correct])
            )
            if correct
            else 0.0,
            "learned_exact_identity_fraction": float(
                np.mean([result["expression"]["exact"] for result in correct])
            )
            if correct
            else 0.0,
            "swapped_returns_target_fraction": float(
                np.mean([result["returns_target"] for result in swapped])
            )
            if swapped
            else 0.0,
            "swapped_returns_swapped_fraction": float(
                np.mean([result["returns_swapped"] for result in swapped])
            )
            if swapped
            else 0.0,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (CASES[args.case_index],) if args.case_index is not None else CASES
    report = run_probe(args.state, cases=cases)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
