"""Family-first shortlisting of frozen future-vector candidates.

The retained selective ledger supplies a causal probability distribution over
the frozen surprise-grown futures.  This assay clusters candidate vectors into
four Ward families in training-scaled latent geometry, selects at most two
high-mass shared cores, and sends only the highest-probability residual edges
inside those cores to a matched eight-candidate shortlist.  Flat top-k and
random size-matched family controls use the identical probabilities.

This is development-only latent routing.  It does not run a native checkpoint,
decode bytes, or claim that geometric agreement is semantic agreement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_vector_families import (
    agglomerative_families,
    random_size_matched_families,
    shortlist_families,
)
from manifoldflow.experiments import (
    dendritic_basin_future_mode_selector_ablation as selector_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_codec_ablation as codec_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_turn_recurrent_codec_ablation as recurrent_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_selective_ledger_ablation as ledger_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_future_family_ablation.py",
    "tests/test_dendritic_basin_future_family_ablation.py",
    "src/manifoldflow/dendritic_basin/future_vector_families.py",
    "tests/test_dendritic_basin_future_vector_families.py",
    *ledger_assay.IMPLEMENTATION_PATHS,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(IMPLEMENTATION_PATHS):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    return result


def _competition_pairs(
    chosen: tuple[int, ...],
    families: tuple[tuple[int, ...], ...] | None,
) -> float:
    if families is None:
        return float(len(chosen) * (len(chosen) - 1) // 2)
    chosen_set = set(chosen)
    populated = [
        len(chosen_set.intersection(family))
        for family in families
        if chosen_set.intersection(family)
    ]
    return float(
        len(populated) * (len(populated) - 1) // 2
        + sum(size * (size - 1) // 2 for size in populated)
    )


def shortlist_metrics(
    target: np.ndarray,
    surface: selector_assay.ResponsibilitySurface,
    probabilities: np.ndarray,
    chosen: tuple[int, ...],
    *,
    proper_variance: np.ndarray,
    families: tuple[tuple[int, ...], ...] | None,
    selected_family_count: int,
    eligible_candidates: int,
    shared_core_fraction: float,
) -> dict[str, float]:
    if not chosen:
        raise ValueError("shortlist cannot be empty")
    indexes = np.asarray(chosen, dtype=np.int64)
    weights = np.asarray(probabilities, dtype=np.float64)[indexes]
    selected_probability_mass = float(weights.sum())
    weights /= weights.sum()
    point = sum(
        (
            weight * surface.means[index]
            for weight, index in zip(weights, chosen, strict=True)
        ),
        np.zeros_like(target),
    )
    oracle_class = int(np.argmax(surface.target_probabilities))
    return {
        "shortlist_size": float(len(chosen)),
        "shortlist_fraction": float(len(chosen) / len(probabilities)),
        "posterior_mass_captured": float(surface.target_probabilities[indexes].sum()),
        "posterior_argmax_recalled": float(oracle_class in chosen),
        "selected_probability_mass": selected_probability_mass,
        "mse": float(np.mean(np.square(target - point))),
        "cosine": selector_assay._cosine(target, point),
        "proper_bits_per_coordinate": surprise_assay.gaussian_bits_per_coordinate(
            target,
            tuple(surface.means[index] for index in chosen),
            tuple(float(item) for item in weights),
            proper_variance,
        ),
        "selected_family_count": float(selected_family_count),
        "eligible_candidates": float(eligible_candidates),
        "competition_pairs": _competition_pairs(chosen, families),
        "shared_core_fraction": float(shared_core_fraction),
    }


def _random_seed(base: int, mode: str, pair_id: str, side: str) -> int:
    payload = f"{base}\0{mode}\0{pair_id}\0{side}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _summarize(
    rows: Sequence[Mapping[str, Any]], *, partition: str, seed: int, resamples: int
) -> dict[str, Any]:
    selected = [row for row in rows if row["partition"] == partition]
    pair_ids = tuple(dict.fromkeys(row["pair_id"] for row in selected))
    arms = tuple(selected[0]["arms"])
    metrics = tuple(selected[0]["arms"][arms[0]])

    def pair_means(arm: str, metric: str) -> list[float]:
        return [
            float(
                np.mean(
                    [
                        row["arms"][arm][metric]
                        for row in selected
                        if row["pair_id"] == pair_id
                    ]
                )
            )
            for pair_id in pair_ids
        ]

    arm_summary = {
        arm: {
            metric: surprise_assay._mean_ci(
                pair_means(arm, metric),
                seed=seed + arm_index * 100 + metric_index,
                resamples=resamples,
            )
            for metric_index, metric in enumerate(metrics)
        }
        for arm_index, arm in enumerate(arms)
    }
    contrasts = {}
    contrast_metrics = (
        "posterior_mass_captured",
        "posterior_argmax_recalled",
        "mse",
        "proper_bits_per_coordinate",
        "competition_pairs",
    )
    counter = 0
    for control in ("flat_top_k", "random_size_matched_families"):
        for metric in contrast_metrics:
            geometry = pair_means("geometric_families", metric)
            other = pair_means(control, metric)
            contrasts[f"geometric_families_minus_{control}_{metric}"] = (
                surprise_assay._mean_ci(
                    [left - right for left, right in zip(geometry, other, strict=True)],
                    seed=seed + 10_000 + counter,
                    resamples=resamples,
                )
            )
            counter += 1
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def evaluate_families(
    basal: Any,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    fast_projector: surprise_assay.ContextProjector,
    event_projector: surprise_assay.ContextProjector,
    bank: Any,
    learner: Any,
    base: Any,
    readout: Any,
    retained: tuple[int, ...],
    proper_variance: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frozen = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        "readout": readout.fingerprint(),
    }
    retained_set = set(retained)
    context_results = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        examples = ledger_assay._development_examples_with_ledger(
            basal, codec, pairs, event_projector, bank, mode=mode
        )
        scorer = copy.deepcopy(learner)
        rows = []
        for pair, side_name, example, active, _current in examples:
            fast = fast_projector.transform(example.context_state)
            surface = selector_assay.responsibility_surface(
                scorer, fast, example.basal_prediction, example.target
            )
            prior = base.predict(fast, surface.mode_ids).probabilities
            active_retained = tuple(item for item in active if item in retained_set)
            probabilities = readout.predict(active_retained, surface.mode_ids, prior)
            flat = selector_assay.shortlist_indices(
                probabilities,
                mass=args.shortlist_mass,
                maximum=args.shortlist_maximum,
            )
            families = agglomerative_families(
                surface.means,
                proper_variance,
                maximum_families=args.family_count,
            )
            geometric = shortlist_families(
                surface.means,
                probabilities,
                proper_variance,
                families,
                family_mass=args.family_mass,
                maximum_selected_families=args.maximum_selected_families,
                candidate_mass=args.shortlist_mass,
                maximum_candidates=args.shortlist_maximum,
            )
            random_families = random_size_matched_families(
                families,
                seed=_random_seed(
                    args.family_random_seed, mode, pair.pair_id, side_name
                ),
            )
            random = shortlist_families(
                surface.means,
                probabilities,
                proper_variance,
                random_families,
                family_mass=args.family_mass,
                maximum_selected_families=args.maximum_selected_families,
                candidate_mass=args.shortlist_mass,
                maximum_candidates=args.shortlist_maximum,
            )
            arms = {
                "flat_top_k": shortlist_metrics(
                    example.target,
                    surface,
                    probabilities,
                    flat,
                    proper_variance=proper_variance,
                    families=None,
                    selected_family_count=len(flat),
                    eligible_candidates=len(flat),
                    shared_core_fraction=0.0,
                ),
                "geometric_families": shortlist_metrics(
                    example.target,
                    surface,
                    probabilities,
                    geometric.selected_candidates,
                    proper_variance=proper_variance,
                    families=families,
                    selected_family_count=len(geometric.selected_families),
                    eligible_candidates=len(geometric.eligible_candidates),
                    shared_core_fraction=geometric.shared_core_fraction,
                ),
                "random_size_matched_families": shortlist_metrics(
                    example.target,
                    surface,
                    probabilities,
                    random.selected_candidates,
                    proper_variance=proper_variance,
                    families=random_families,
                    selected_family_count=len(random.selected_families),
                    eligible_candidates=len(random.eligible_candidates),
                    shared_core_fraction=random.shared_core_fraction,
                ),
            }
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "candidate_count": len(probabilities),
                    "family_sizes": [len(item) for item in families],
                    "arms": arms,
                }
            )
        context_results[mode] = {
            "summaries": {
                partition: _summarize(
                    rows,
                    partition=partition,
                    seed=args.bootstrap_seed
                    + mode_index * 100_000
                    + (0 if partition == "selection" else 50_000),
                    resamples=args.bootstrap_resamples,
                )
                for partition in ("selection", "evaluation")
            },
            "pair_rows": rows,
        }
    after = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        "readout": readout.fingerprint(),
    }
    if frozen != after or basal.observed_count != 0:
        raise AssertionError("family evaluation modified frozen predictors")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": frozen,
        "fitted_fingerprints_after": after,
        "all_predictors_frozen_and_unchanged": True,
    }


def development_gate(
    evaluation: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    summary = evaluation["context_modes"]["long_interruption_crossover"]["summaries"][
        "evaluation"
    ]
    arms = summary["arms"]
    contrasts = summary["paired_contrasts"]
    geometry = arms["geometric_families"]
    mass_flat = contrasts["geometric_families_minus_flat_top_k_posterior_mass_captured"]
    recall_flat = contrasts[
        "geometric_families_minus_flat_top_k_posterior_argmax_recalled"
    ]
    pair_flat = contrasts["geometric_families_minus_flat_top_k_competition_pairs"]
    mass_random = contrasts[
        "geometric_families_minus_random_size_matched_families_posterior_mass_captured"
    ]
    proper_random = contrasts[
        "geometric_families_minus_random_size_matched_families_"
        "proper_bits_per_coordinate"
    ]
    components = {
        "at_most_locked_family_count": geometry["selected_family_count"]["mean"]
        <= args.maximum_selected_families,
        "at_most_locked_candidate_count": geometry["shortlist_size"]["mean"]
        <= args.shortlist_maximum,
        "posterior_mass_at_least_locked_minimum": geometry["posterior_mass_captured"][
            "mean"
        ]
        >= args.minimum_posterior_mass,
        "argmax_recall_at_least_locked_minimum": geometry["posterior_argmax_recalled"][
            "mean"
        ]
        >= args.minimum_argmax_recall,
        "mass_noninferior_to_flat": mass_flat["ci95_low"]
        >= -args.family_mass_noninferiority,
        "recall_noninferior_to_flat": recall_flat["ci95_low"]
        >= -args.family_recall_noninferiority,
        "strictly_fewer_pairwise_competitions_than_flat": pair_flat["ci95_high"] < 0,
        "geometry_beats_random_grouping": mass_random["ci95_low"] > 0
        or proper_random["ci95_high"] < 0,
    }
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "geometric_family_means": {
            metric: geometry[metric]["mean"]
            for metric in (
                "selected_family_count",
                "shortlist_size",
                "posterior_mass_captured",
                "posterior_argmax_recalled",
                "competition_pairs",
                "shared_core_fraction",
                "mse",
                "proper_bits_per_coordinate",
            )
        },
        "paired_contrasts": {
            "mass_minus_flat": mass_flat,
            "recall_minus_flat": recall_flat,
            "competition_pairs_minus_flat": pair_flat,
            "mass_minus_random": mass_random,
            "proper_bits_minus_random": proper_random,
        },
        "components": components,
        "advances_to_edge_competition_sandbox": all(components.values()),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.latent_only:
        raise ValueError("this assay is latent-only")
    manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )
    assistant_turns = [
        turn for conversation in conversations for turn in conversation.assistant_turns
    ]
    codec = codec_assay.FutureTurnCodec.fit(
        assistant_turns,
        raw_dimensions=args.raw_dimensions,
        content_dimensions=args.content_dimensions,
        active_content_features=args.active_content_features,
    )
    sequences = recurrent_assay.encode_training_sequences(conversations, codec)
    crossfit = surprise_assay.cross_fitted_basal_transitions(
        conversations,
        sequences,
        content_dimensions=codec.content_dimensions,
        folds=args.crossfit_folds,
        fold_seed=args.crossfit_seed,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    fast_matrix = np.stack([item.context_state for item in crossfit.examples])
    fast_projector = surprise_assay.ContextProjector.fit(
        fast_matrix,
        dimensions=args.context_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    fast_states = fast_projector.transform(fast_matrix)
    token_dimensions = crossfit.examples[0].context_state.size // 2
    token_matrix = np.stack(
        [item.context_state[:token_dimensions] for item in crossfit.examples]
    )
    event_projector = surprise_assay.ContextProjector.fit(
        token_matrix,
        dimensions=args.ledger_event_dimensions,
        scale_floor=args.context_scale_floor,
        clip=args.context_clip,
    )
    bank, persistent, current, bank_audit = ledger_assay.fit_prototype_bank_and_traces(
        crossfit.examples,
        event_projector,
        conversations=len(conversations),
        token_dimensions=token_dimensions,
        args=args,
    )
    ordered, _unused, overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    base, readouts, _global_counts, retained, ledger_training = (
        ledger_assay.train_selective_ledger(
            crossfit.examples,
            fast_states,
            persistent,
            current,
            ordered,
            conversations=len(conversations),
            args=args,
        )
    )
    full_basal = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )
    assistant_examples = [
        item
        for item in crossfit.examples
        if item.target_turn.role_id == codec_assay.ASSISTANT_ROLE
    ]
    residual_vectors = np.stack(
        [item.target - item.basal_prediction for item in assistant_examples]
    )
    proper_variance = np.maximum(
        np.mean(np.square(residual_vectors), axis=0), args.proper_variance_floor
    )
    evaluation = evaluate_families(
        full_basal,
        codec,
        pairs,
        fast_projector,
        event_projector,
        bank,
        ordered,
        base,
        readouts["persistent"],
        retained,
        proper_variance,
        args=args,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    result = {
        "schema_version": 1,
        "experiment": "future_vector_shared_core_family_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "expensive_particle_competition_run": False,
            "candidate_probabilities_identical_across_family_controls": True,
            "families_use_candidate_geometry_without_target": True,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "geometric_similarity_proves_semantic_agreement": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "config": {
            "family_count": args.family_count,
            "family_mass": args.family_mass,
            "maximum_selected_families": args.maximum_selected_families,
            "candidate_mass": args.shortlist_mass,
            "maximum_candidates": args.shortlist_maximum,
            "family_random_seed": args.family_random_seed,
            "mass_noninferiority_margin": args.family_mass_noninferiority,
            "recall_noninferiority_margin": args.family_recall_noninferiority,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(crossfit.examples),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "prototype_bank": bank_audit,
        "surprise_population_training": overlay,
        "ledger_training": ledger_training,
        "evaluation": evaluation,
        "development_gate": development_gate(evaluation, args),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = ledger_assay._parser()
    parser.description = __doc__
    parser.add_argument("--family-count", type=int, default=4)
    parser.add_argument("--family-mass", type=float, default=0.80)
    parser.add_argument("--maximum-selected-families", type=int, default=2)
    parser.add_argument("--family-random-seed", type=int, default=261_501)
    parser.add_argument("--family-mass-noninferiority", type=float, default=0.05)
    parser.add_argument("--family-recall-noninferiority", type=float, default=0.10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_ablation(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
