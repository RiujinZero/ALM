"""Does a narrower affinity radius rescue the persistent core?

The first persistent-core configuration (bandwidth 1.0, min affinity 0.50)
collapsed to exactly one slot on the long-interruption selection partition:
its eligibility radius (bandwidth * sqrt(2 ln(1/affinity)) RMS, about 1.177)
covered every selection context, so every episode was absorbed into one slot
and `predict` returned a near-global mean. That failed the proper-score
noninferiority gate while leaving all 16 slots unused. A too-wide radius
produces exactly that failure mode without saying anything about whether the
underlying context is generalizable, so the prescribed rescue is narrowing,
not widening; slot capacity was never the binding constraint.

This sweep crosses bandwidths {1.0, 0.5, 0.25} with affinity floors
{0.50, 0.75, 0.90} -- effective radii from about 1.177 down to about 0.115
RMS -- with every other estimator knob frozen at the v1 values and the v1
anchor as config 0. The evaluation partition is never touched during
selection: configs are ranked by a deterministic pair-parity cross-validation
inside the selection partition only, and the chosen configuration is then run
once through the unchanged locked assay
(dendritic_basin_future_persistent_core_ablation), whose gate scores the
held-out evaluation partition with the same margins and seed scheme as v1.

Development-only latent routing. It runs no native checkpoint, decodes no
bytes, and trains no receiver.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
    dendritic_basin_future_persistent_core_ablation as persistent_core_assay,
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
    "src/manifoldflow/experiments/dendritic_basin_future_persistent_core_sweep_ablation.py",
    "tests/test_dendritic_basin_future_persistent_core_sweep_ablation.py",
    *persistent_core_assay.IMPLEMENTATION_PATHS,
)

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")
PRIMARY_MODE = "long_interruption_crossover"


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


def pair_parity_split(
    selection_pairs: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
    """Deterministic CV split: even-indexed pairs fit, odd-indexed eval.

    The split is over pair ids, not rows, so both sides of a held-out pair
    stay held out together -- mirroring the fit-on-selection /
    evaluate-on-evaluation structure of the locked assay, where a whole
    episode is never seen during fitting.
    """

    ordered = sorted(selection_pairs, key=lambda pair: pair.pair_id)
    return list(ordered[0::2]), list(ordered[1::2])


def context_is_eligible(
    context: np.ndarray, estimator: PersistentCoreEstimator
) -> bool:
    """Mirror of PersistentCoreEstimator's affinity gate for diagnostics.

    True iff at least one slot would receive the context at predict time;
    when False, predict falls back to the global running core.
    """

    for slot in estimator.slot_states:
        z = (
            math.sqrt(float(np.mean(np.square(context - slot.context_prototype))))
            / estimator.config.context_bandwidth
        )
        if math.exp(-0.5 * z * z) >= estimator.config.min_context_affinity:
            return True
    return False


def choose_config(
    entries: Sequence[Mapping[str, Any]], tie_tolerance: float
) -> Mapping[str, Any]:
    """Lowest mean CV proper-bits contrast; ties broken by MSE, then grid order."""

    if not entries:
        raise ValueError("cannot choose from an empty config list")
    bits = _selection_bits_mean
    mse = _selection_mse_mean
    best_bits = min(bits(entry) for entry in entries)
    bits_tied = [entry for entry in entries if bits(entry) <= best_bits + tie_tolerance]
    best_mse = min(mse(entry) for entry in bits_tied)
    final = [entry for entry in bits_tied if mse(entry) <= best_mse + tie_tolerance]
    return min(final, key=lambda entry: int(entry["config_index"]))


def _selection_bits_mean(entry: Mapping[str, Any]) -> float:
    return float(
        entry[PRIMARY_MODE]["evaluation_summary"]["paired_contrasts"][
            "persistent_minus_own_proper_bits_per_coordinate"
        ]["mean"]
    )


def _selection_mse_mean(entry: Mapping[str, Any]) -> float:
    return float(
        entry[PRIMARY_MODE]["evaluation_summary"]["paired_contrasts"][
            "persistent_minus_own_mse"
        ]["mean"]
    )


def build_selection_rows(
    basal: Any,
    codec: codec_assay.FutureTurnCodec,
    selection_pairs: Sequence[codec_assay.DevelopmentPair],
    fast_projector: surprise_assay.ContextProjector,
    event_projector: surprise_assay.ContextProjector,
    bank: Any,
    learner: Any,
    base: Any,
    readout: Any,
    retained: tuple[int, ...],
    proper_variance: np.ndarray,
    *,
    mode: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Row construction, verbatim in substance from the locked evaluation loop.

    Rows carry the context vector, the episode's own mass-weighted mean core,
    its weighted edge sum, the true target, and the own-core metrics. The
    own-core point estimate is asserted equal (1e-9) to the flat top-k point
    estimate, exactly as in the locked assay.
    """

    retained_set = set(retained)
    examples = ledger_assay._development_examples_with_ledger(
        basal, codec, selection_pairs, event_projector, bank, mode=mode
    )
    scorer = copy.deepcopy(learner)
    built: list[dict[str, Any]] = []
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
        weighted_edge_sum = core_edge_assay._weighted_edge_sum(
            probabilities, flat, split
        )
        own_metrics = core_edge_assay.core_edge_point_metrics(
            example.target,
            split.core,
            weighted_edge_sum,
            proper_variance=proper_variance,
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
    return built


def cross_validate_config(
    fit_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    bandwidth: float,
    min_affinity: float,
    content_dimensions: int,
    proper_variance: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    estimator = PersistentCoreEstimator(
        PersistentCoreEstimatorConfig(
            context_dim=int(args.context_dimensions),
            core_dim=int(content_dimensions),
            max_slots=args.persistent_max_slots,
            context_bandwidth=bandwidth,
            min_context_affinity=min_affinity,
            slot_learning_rate=args.persistent_slot_learning_rate,
            global_learning_rate=args.persistent_global_learning_rate,
        )
    )
    for row in fit_rows:
        estimator.update(row["context"], row["core"])

    output_rows = []
    fallbacks = 0
    for row in evaluation_rows:
        if not context_is_eligible(row["context"], estimator):
            fallbacks += 1
        persistent_core = estimator.predict(row["context"])
        persistent_metrics = core_edge_assay.core_edge_point_metrics(
            row["target"],
            persistent_core,
            row["weighted_edge_sum"],
            proper_variance=proper_variance,
        )
        output_rows.append(
            {
                "pair_id": row["pair_id"],
                "side": row["side"],
                "arms": {
                    "own_core": row["own_metrics"],
                    "persistent_core": persistent_metrics,
                },
            }
        )
    return {
        "estimator_slot_count": estimator.slot_count,
        "estimator_births": estimator.births,
        "estimator_recycles": estimator.recycles,
        "global_fallback_fraction": float(fallbacks / len(evaluation_rows)),
        "evaluation_summary": persistent_core_assay._summarize_single_partition(
            output_rows, seed=seed, resamples=args.bootstrap_resamples
        ),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
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

    selection_pairs = [pair for pair in pairs if pair.partition == "selection"]
    if not selection_pairs:
        raise ValueError("no selection-partition pairs available for cross-validation")
    cv_fit_pairs, cv_eval_pairs = pair_parity_split(selection_pairs)
    cv_fit_ids = {pair.pair_id for pair in cv_fit_pairs}
    cv_eval_ids = {pair.pair_id for pair in cv_eval_pairs}

    frozen = {
        "basal": full_basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    rows_by_mode = {
        mode: build_selection_rows(
            full_basal,
            codec,
            selection_pairs,
            fast_projector,
            event_projector,
            bank,
            ordered,
            base,
            readouts["persistent"],
            retained,
            proper_variance,
            mode=mode,
            args=args,
        )
        for mode in CONTEXT_MODES
    }
    fit_rows_by_mode = {
        mode: sorted(
            (row for row in rows if row["pair_id"] in cv_fit_ids),
            key=lambda row: (row["pair_id"], row["side"]),
        )
        for mode, rows in rows_by_mode.items()
    }
    eval_rows_by_mode = {
        mode: sorted(
            (row for row in rows if row["pair_id"] in cv_eval_ids),
            key=lambda row: (row["pair_id"], row["side"]),
        )
        for mode, rows in rows_by_mode.items()
    }

    grid = [
        (bandwidth, min_affinity)
        for bandwidth in args.sweep_bandwidths
        for min_affinity in args.sweep_min_context_affinities
    ]
    if not grid:
        raise ValueError("sweep grid is empty")

    config_entries = []
    for config_index, (bandwidth, min_affinity) in enumerate(grid):
        entry: dict[str, Any] = {
            "config_index": config_index,
            "bandwidth": float(bandwidth),
            "min_context_affinity": float(min_affinity),
        }
        for mode in CONTEXT_MODES:
            entry[mode] = cross_validate_config(
                fit_rows_by_mode[mode],
                eval_rows_by_mode[mode],
                bandwidth=bandwidth,
                min_affinity=min_affinity,
                content_dimensions=codec.content_dimensions,
                proper_variance=proper_variance,
                args=args,
                seed=args.bootstrap_seed
                + (200_000 if mode == PRIMARY_MODE else 300_000)
                + config_index,
            )
        config_entries.append(entry)

    chosen = choose_config(config_entries, tie_tolerance=args.sweep_tie_tolerance)
    after = {
        "basal": full_basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    if frozen != after or full_basal.observed_count != 0:
        raise AssertionError("persistent-core sweep modified frozen predictors")

    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    chosen_bits = _selection_bits_mean(chosen)
    chosen_mse = _selection_mse_mean(chosen)
    result = {
        "schema_version": 1,
        "experiment": "future_vector_persistent_core_sweep_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "receiver_trained_or_decoded": False,
            "narrowing_method": "flat_top_k",
            "evaluation_partition_used_during_selection": False,
            "cv_fit_partition": "selection even-indexed pairs",
            "cv_evaluation_partition": "selection odd-indexed pairs",
            "final_gate_run_separately_by_locked_assay": True,
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
            "sweep_bandwidths": list(args.sweep_bandwidths),
            "sweep_min_context_affinities": list(args.sweep_min_context_affinities),
            "sweep_grid_order": "bandwidth outer in listed order, affinity inner",
            "sweep_tie_tolerance": args.sweep_tie_tolerance,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "persistent_max_slots": args.persistent_max_slots,
            "persistent_slot_learning_rate": args.persistent_slot_learning_rate,
            "persistent_global_learning_rate": args.persistent_global_learning_rate,
            "frozen_at_v1_values": [
                "persistent_max_slots",
                "persistent_slot_learning_rate",
                "persistent_global_learning_rate",
                "shortlist_mass",
                "shortlist_maximum",
            ],
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(crossfit.examples),
            "development_pairs": len(pairs),
            "selection_pairs": len(selection_pairs),
            "cv_fit_pairs": len(cv_fit_pairs),
            "cv_eval_pairs": len(cv_eval_pairs),
            "cv_fit_rows": {mode: len(rows) for mode, rows in fit_rows_by_mode.items()},
            "cv_eval_rows": {
                mode: len(rows) for mode, rows in eval_rows_by_mode.items()
            },
        },
        "prototype_bank": bank_audit,
        "surprise_population_training": overlay,
        "ledger_training": ledger_training,
        "cross_validation": {
            "selection_metric": (
                "minimum mean paired proper-bits contrast "
                "(persistent_minus_own) on long_interruption_crossover CV; "
                "ties within sweep_tie_tolerance broken by mean MSE "
                "contrast, then by grid order"
            ),
            "bootstrap_seeds": {
                "primary_mode": "bootstrap_seed + 200000 + config_index",
                "secondary_mode": "bootstrap_seed + 300000 + config_index",
            },
            "configs": config_entries,
            "chosen_config": {
                "config_index": chosen["config_index"],
                "bandwidth": chosen["bandwidth"],
                "min_context_affinity": chosen["min_context_affinity"],
                "selection_bits_mean": chosen_bits,
                "selection_mse_mean": chosen_mse,
                "primary_mode_global_fallback_fraction": chosen[PRIMARY_MODE][
                    "global_fallback_fraction"
                ],
                "primary_mode_estimator_slot_count": chosen[PRIMARY_MODE][
                    "estimator_slot_count"
                ],
            },
        },
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = persistent_core_assay._parser()
    parser.description = __doc__
    parser.add_argument(
        "--sweep-bandwidths", type=float, nargs="+", default=[1.0, 0.5, 0.25]
    )
    parser.add_argument(
        "--sweep-min-context-affinities",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 0.90],
    )
    parser.add_argument("--sweep-tie-tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
