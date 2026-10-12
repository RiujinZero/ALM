"""Does a persistent, cross-episode core generalize, or does it regress?

The per-episode mass-weighted mean core (future_vector_core_edge.py) passed
its cross-pair context-specificity gate, but is refit from scratch every
episode -- it cannot generalize across mutually compatible futures. The
native500 decoder-exposure curve then showed that more receiver exposure does
not reliably close a train/predict distribution gap; the actual fix for that
kind of gap is making predicted-time input look more like training-time
input, which a from-scratch-per-episode estimator structurally cannot do.

This assay fits a `PersistentCoreEstimator` (future_vector_persistent_core.py)
on the "selection" partition's own per-pair mean cores -- each already
computed, frozen, read-only -- processed in a fixed deterministic order, then
evaluates it on the "evaluation" partition, which the estimator never sees
during fitting. This is the honest checkpoint: does context-based retrieval
of a core built from *other* episodes predict a *held-out* episode's true
target about as well as recomputing a fresh mean from that episode's own
candidates? Noninferiority, not necessarily outright superiority, is the bar
-- generalizing across episodes is a strictly harder task than fitting the
one at hand.

This is development-only latent routing. It does not run a native checkpoint,
decode bytes, or claim that clearing this bar makes the persistent core
receiver-ready -- only that it is not a regression from the cheap baseline
before a receiver is trained on it.
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

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.dendritic_basin.future_vector_persistent_core import (
    PersistentCoreEstimator,
    PersistentCoreEstimatorConfig,
)
from manifoldflow.experiments import (
    dendritic_basin_future_core_edge_ablation as core_edge_assay,
)
from manifoldflow.experiments import (
    dendritic_basin_future_family_ablation as family_assay,
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
    "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_ablation.py",
    "tests/test_dendritic_basin_future_persistent_core_ablation.py",
    "src/manifoldflow/dendritic_basin/future_vector_persistent_core.py",
    "tests/test_dendritic_basin_future_vector_persistent_core.py",
    *core_edge_assay.IMPLEMENTATION_PATHS,
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


def _summarize_single_partition(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> dict[str, Any]:
    pair_ids = tuple(dict.fromkeys(row["pair_id"] for row in rows))
    arms = tuple(rows[0]["arms"])
    metrics = tuple(rows[0]["arms"][arms[0]])

    def pair_means(arm: str, metric: str) -> list[float]:
        return [
            float(
                np.mean(
                    [row["arms"][arm][metric] for row in rows if row["pair_id"] == pair_id]
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
    own_mse = pair_means("own_core", "mse")
    persistent_mse = pair_means("persistent_core", "mse")
    own_bits = pair_means("own_core", "proper_bits_per_coordinate")
    persistent_bits = pair_means("persistent_core", "proper_bits_per_coordinate")
    contrasts = {
        "persistent_minus_own_mse": surprise_assay._mean_ci(
            [p - o for p, o in zip(persistent_mse, own_mse, strict=True)],
            seed=seed + 10_000,
            resamples=resamples,
        ),
        "persistent_minus_own_proper_bits_per_coordinate": surprise_assay._mean_ci(
            [p - o for p, o in zip(persistent_bits, own_bits, strict=True)],
            seed=seed + 10_001,
            resamples=resamples,
        ),
    }
    return {
        "pairs": len(pair_ids),
        "sides": len(rows),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def persistent_core_gate(
    evaluation: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    summary = evaluation["context_modes"]["long_interruption_crossover"]["evaluation_summary"]
    contrasts = summary["paired_contrasts"]
    mse_gap = contrasts["persistent_minus_own_mse"]
    bits_gap = contrasts["persistent_minus_own_proper_bits_per_coordinate"]
    components = {
        "persistent_noninferior_to_own_on_mse": mse_gap["ci95_high"]
        <= args.mse_noninferiority_margin,
        "persistent_noninferior_to_own_on_proper_score": bits_gap["ci95_high"]
        <= args.proper_bits_noninferiority_margin,
    }
    passed = all(components.values())
    return {
        "scope": "long_interruption_crossover/evaluation_partition "
        "(estimator fit on selection_partition only, never evaluation)",
        "paired_contrasts": {
            "mse_persistent_minus_own": mse_gap,
            "proper_bits_persistent_minus_own": bits_gap,
        },
        "components": components,
        "beats_own_outright_on_mse": mse_gap["ci95_high"] < 0.0,
        "beats_own_outright_on_proper_score": bits_gap["ci95_high"] < 0.0,
        "persistent_core_generalizes_without_regressing": passed,
        "advances_to_receiver_causal_controls_and_training": passed,
        "if_this_fails": (
            "the per-episode mean core does not generalize across episodes "
            "under this affinity-slot estimator -- widen slot capacity or "
            "affinity bandwidth before concluding persistence itself is "
            "wrong, since a too-narrow or too-few-slot configuration would "
            "produce exactly this failure mode without saying anything "
            "about whether the underlying context is generalizable."
        ),
    }


def evaluate_persistent_core(
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
        built = []
        for pair, side_name, example, active, _current in examples:
            fast = fast_projector.transform(example.context_state)
            surface = selector_assay.responsibility_surface(
                scorer, fast, example.basal_prediction, example.target
            )
            prior = base.predict(fast, surface.mode_ids).probabilities
            active_retained = tuple(item for item in active if item in retained_set)
            probabilities = readout.predict(active_retained, surface.mode_ids, prior)
            flat = selector_assay.shortlist_indices(
                probabilities, mass=args.shortlist_mass, maximum=args.shortlist_maximum
            )
            split = mass_weighted_core_edge_split(surface.means, probabilities, flat)
            weighted_edge_sum = core_edge_assay._weighted_edge_sum(probabilities, flat, split)
            own_metrics = core_edge_assay.core_edge_point_metrics(
                example.target, split.core, weighted_edge_sum, proper_variance=proper_variance
            )
            flat_metrics = family_assay.shortlist_metrics(
                example.target,
                surface,
                probabilities,
                flat,
                proper_variance=proper_variance,
                families=None,
                selected_family_count=len(flat),
                eligible_candidates=len(flat),
                shared_core_fraction=0.0,
            )
            if abs(own_metrics["mse"] - flat_metrics["mse"]) > 1e-9:
                raise AssertionError(
                    "core+edge point diverged from the flat top-k point estimate"
                )
            built.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "context": fast,
                    "core": split.core,
                    "weighted_edge_sum": weighted_edge_sum,
                    "target": example.target,
                    "own_metrics": own_metrics,
                }
            )

        selection_rows = sorted(
            (row for row in built if row["partition"] == "selection"),
            key=lambda row: (row["pair_id"], row["side"]),
        )
        evaluation_rows = [row for row in built if row["partition"] == "evaluation"]

        estimator = PersistentCoreEstimator(
            PersistentCoreEstimatorConfig(
                context_dim=int(args.context_dimensions),
                core_dim=int(codec.content_dimensions),
                max_slots=args.persistent_max_slots,
                context_bandwidth=args.persistent_context_bandwidth,
                min_context_affinity=args.persistent_min_context_affinity,
                slot_learning_rate=args.persistent_slot_learning_rate,
                global_learning_rate=args.persistent_global_learning_rate,
            )
        )
        for row in selection_rows:
            estimator.update(row["context"], row["core"])

        evaluation_output_rows = []
        for row in evaluation_rows:
            persistent_core = estimator.predict(row["context"])
            persistent_metrics = core_edge_assay.core_edge_point_metrics(
                row["target"],
                persistent_core,
                row["weighted_edge_sum"],
                proper_variance=proper_variance,
            )
            evaluation_output_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "side": row["side"],
                    "arms": {
                        "own_core": row["own_metrics"],
                        "persistent_core": persistent_metrics,
                    },
                }
            )

        context_results[mode] = {
            "selection_fit_pairs": len(dict.fromkeys(row["pair_id"] for row in selection_rows)),
            "selection_fit_sides": len(selection_rows),
            "evaluation_pairs": len(dict.fromkeys(row["pair_id"] for row in evaluation_rows)),
            "estimator_slot_count": estimator.slot_count,
            "estimator_births": estimator.births,
            "estimator_recycles": estimator.recycles,
            "evaluation_summary": _summarize_single_partition(
                evaluation_output_rows,
                seed=args.bootstrap_seed + mode_index * 100_000,
                resamples=args.bootstrap_resamples,
            ),
            "evaluation_rows": evaluation_output_rows,
        }
    after = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        "readout": readout.fingerprint(),
    }
    if frozen != after or basal.observed_count != 0:
        raise AssertionError("persistent-core evaluation modified frozen predictors")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": frozen,
        "fitted_fingerprints_after": after,
        "all_predictors_frozen_and_unchanged": True,
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
    evaluation = evaluate_persistent_core(
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
        "experiment": "future_vector_persistent_core_generalization_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "receiver_trained_or_decoded": False,
            "narrowing_method": "flat_top_k",
            "estimator_fit_partition": "selection",
            "estimator_evaluated_on": "evaluation",
            "estimator_never_sees_evaluation_partition_during_fitting": True,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "config": {
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "persistent_max_slots": args.persistent_max_slots,
            "persistent_context_bandwidth": args.persistent_context_bandwidth,
            "persistent_min_context_affinity": args.persistent_min_context_affinity,
            "persistent_slot_learning_rate": args.persistent_slot_learning_rate,
            "persistent_global_learning_rate": args.persistent_global_learning_rate,
            "mse_noninferiority_margin": args.mse_noninferiority_margin,
            "proper_bits_noninferiority_margin": args.proper_bits_noninferiority_margin,
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
        "development_gate": persistent_core_gate(evaluation, args),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = ledger_assay._parser()
    parser.description = __doc__
    parser.add_argument("--persistent-max-slots", type=int, default=16)
    parser.add_argument("--persistent-context-bandwidth", type=float, default=1.0)
    parser.add_argument("--persistent-min-context-affinity", type=float, default=0.50)
    parser.add_argument("--persistent-slot-learning-rate", type=float, default=0.5)
    parser.add_argument("--persistent-global-learning-rate", type=float, default=0.5)
    parser.add_argument("--mse-noninferiority-margin", type=float, default=0.10)
    parser.add_argument("--proper-bits-noninferiority-margin", type=float, default=0.05)
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
