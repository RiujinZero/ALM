"""Cheapest-first core/edge causal test: is the mass-weighted core specific?

Narrowing is fixed to flat top-k (the family-clustering assay found every
tested alternative -- categorical selector, two-timescale selector, nested
slow selector, selective ledger, geometric families -- statistically
indistinguishable from flat top-k on posterior mass, recall, and competition
count, so there is no basis yet for paying clustering/selector/ledger cost at
this layer). This assay narrows with flat top-k exactly as that assay did,
then asks the next question: does splitting the retained shortlist into
``z_m = core + edge_m`` (core = mass-weighted mean of the shortlist) actually
buy anything, or does competition just move inside the split the way it moved
inside the family?

Within-shortlist reconstruction (``core + edge_m == z_m``) is exact by
construction for a mean-subtraction split -- not evidence of anything, and
not tested as a claim here (see future_vector_core_edge.py's docstring). The
one genuinely falsifiable question at this cheap operationalization: is the
resulting core context/episode-specific? For each development pair/side,
this assay substitutes a *different*, Sattolo-deranged pair/side's core
(holding this row's own probability-weighted edge sum fixed) and compares
reconstruction of the TRUE target -- not the candidate's own vector, which
would again be tautological -- between the row's own core and the foreign
one. If own does not reliably beat foreign, the mean carries no reusable,
context-specific information, and the next investment is a better core
estimator (persistent EMA or a local assembly), not a trained receiver for
this one.

This is development-only latent routing. It does not run a native checkpoint,
decode bytes, or claim that a passing gate proves the core/edge factorization
is production-useful -- only that it clears the cheapest bar before a
receiver is trained on it at all.
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
    CoreEdgeSplit,
    mass_weighted_core_edge_split,
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
    "src/manifoldflow/experiments/dendritic_basin_future_core_edge_ablation.py",
    "tests/test_dendritic_basin_future_core_edge_ablation.py",
    "src/manifoldflow/dendritic_basin/future_vector_core_edge.py",
    "tests/test_dendritic_basin_future_vector_core_edge.py",
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


def sattolo_derangement(count: int, seed: int) -> np.ndarray:
    """A permutation with no fixed points (Sattolo's algorithm).

    Every row is guaranteed to be paired with a genuinely different row --
    the same discipline the project's other shuffled-code controls use
    (see goal_conditioned_byte_head.py), so a foreign-core substitution can
    never silently degenerate into an own-core comparison.
    """

    if count <= 0:
        raise ValueError("derangement requires at least one row")
    permutation = np.arange(count, dtype=np.int64)
    if count > 1:
        rng = np.random.default_rng(seed)
        for index in range(count - 1, 0, -1):
            other = int(rng.integers(0, index))
            permutation[index], permutation[other] = (
                permutation[other],
                permutation[index],
            )
    return permutation


def _foreign_core_seed(base: int, mode: str, partition: str) -> int:
    payload = f"{base}\0{mode}\0{partition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def core_edge_point_metrics(
    target: np.ndarray,
    core: np.ndarray,
    weighted_edge_sum: np.ndarray,
    *,
    proper_variance: np.ndarray,
) -> dict[str, float]:
    """Score `core + weighted_edge_sum` as a point prediction of `target`."""

    point = np.asarray(core, dtype=np.float64) + np.asarray(
        weighted_edge_sum, dtype=np.float64
    )
    return {
        "mse": float(np.mean(np.square(np.asarray(target, dtype=np.float64) - point))),
        "cosine": selector_assay._cosine(target, point),
        "proper_bits_per_coordinate": surprise_assay.gaussian_bits_per_coordinate(
            target, (point,), (1.0,), proper_variance
        ),
    }


def _weighted_edge_sum(
    probabilities: np.ndarray, chosen: tuple[int, ...], split: CoreEdgeSplit
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)[np.asarray(chosen)]
    weights = values / values.sum()
    return sum(
        (weight * edge for weight, edge in zip(weights, split.edges, strict=True)),
        np.zeros_like(split.core),
    )


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
    own = pair_means("own_core", "mse")
    own_bits = pair_means("own_core", "proper_bits_per_coordinate")
    foreign = pair_means("foreign_core", "mse")
    foreign_bits = pair_means("foreign_core", "proper_bits_per_coordinate")
    contrasts = {
        "foreign_core_minus_own_core_mse": surprise_assay._mean_ci(
            [f - o for f, o in zip(foreign, own, strict=True)],
            seed=seed + 10_000,
            resamples=resamples,
        ),
        "own_core_minus_foreign_core_proper_bits_per_coordinate": surprise_assay._mean_ci(
            [o - f for o, f in zip(own_bits, foreign_bits, strict=True)],
            seed=seed + 10_001,
            resamples=resamples,
        ),
    }
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def evaluate_core_edge(
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
            weighted_edge_sum = _weighted_edge_sum(probabilities, flat, split)
            own_metrics = core_edge_point_metrics(
                example.target,
                split.core,
                weighted_edge_sum,
                proper_variance=proper_variance,
            )
            # Cross-check: core + weighted-edge-sum must equal flat top-k's
            # own point estimate exactly -- both are the same probability-
            # weighted mean of the same shortlist, computed two ways.
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
                    "candidate_count": len(probabilities),
                    "shortlist_size": len(flat),
                    "core": split.core,
                    "weighted_edge_sum": weighted_edge_sum,
                    "target": example.target,
                    "own_metrics": own_metrics,
                }
            )
        rows = []
        for partition in ("selection", "evaluation"):
            group = [row for row in built if row["partition"] == partition]
            order = sorted(
                range(len(group)),
                key=lambda index: (group[index]["pair_id"], group[index]["side"]),
            )
            seed = _foreign_core_seed(args.core_edge_random_seed, mode, partition)
            derangement = sattolo_derangement(len(order), seed)
            for position, row_index in enumerate(order):
                partner_index = order[int(derangement[position])]
                row = group[row_index]
                partner = group[partner_index]
                if partner["pair_id"] == row["pair_id"] and partner["side"] == row["side"]:
                    raise AssertionError("Sattolo derangement produced a fixed point")
                foreign_metrics = core_edge_point_metrics(
                    row["target"],
                    partner["core"],
                    row["weighted_edge_sum"],
                    proper_variance=proper_variance,
                )
                rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "partition": partition,
                        "side": row["side"],
                        "candidate_count": row["candidate_count"],
                        "shortlist_size": row["shortlist_size"],
                        "foreign_pair_id": partner["pair_id"],
                        "foreign_side": partner["side"],
                        "arms": {
                            "own_core": row["own_metrics"],
                            "foreign_core": foreign_metrics,
                        },
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
        raise AssertionError("core/edge evaluation modified frozen predictors")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": frozen,
        "fitted_fingerprints_after": after,
        "all_predictors_frozen_and_unchanged": True,
    }


def core_edge_gate(
    evaluation: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    summary = evaluation["context_modes"]["long_interruption_crossover"]["summaries"][
        "evaluation"
    ]
    arms = summary["arms"]
    contrasts = summary["paired_contrasts"]
    own = arms["own_core"]
    mse_gain = contrasts["foreign_core_minus_own_core_mse"]
    bits_gain = contrasts["own_core_minus_foreign_core_proper_bits_per_coordinate"]
    components = {
        "own_core_beats_foreign_core_on_mse": mse_gain["ci95_low"] > 0.0,
        "own_core_beats_foreign_core_on_proper_score": bits_gain["ci95_high"] < 0.0,
    }
    passed = all(components.values())
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "own_core_means": {
            metric: own[metric]["mean"]
            for metric in ("mse", "cosine", "proper_bits_per_coordinate")
        },
        "paired_contrasts": {
            "mse_foreign_minus_own": mse_gain,
            "proper_bits_own_minus_foreign": bits_gain,
        },
        "components": components,
        "core_is_context_specific": passed,
        "advances_to_learned_core_estimator_and_receiver_training": passed,
        "if_this_fails": (
            "invest in a persistent/learned core estimator (EMA across "
            "episodes, or a local assembly) before training any receiver "
            "on this mean-based core -- do not conclude core/edge itself "
            "is wrong from this cheap version failing."
        ),
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
    evaluation = evaluate_core_edge(
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
        "experiment": "future_vector_core_edge_context_specificity_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "receiver_trained_or_decoded": False,
            "narrowing_method": "flat_top_k",
            "core_estimator": "mass_weighted_mean_of_shortlist",
            "within_shortlist_reconstruction_is_exact_by_construction": True,
            "tested_claim_is_cross_pair_core_context_specificity_only": True,
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
            "core_edge_random_seed": args.core_edge_random_seed,
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
        "development_gate": core_edge_gate(evaluation, args),
        "wall_seconds": elapsed,
    }
    json.dumps(result, allow_nan=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = ledger_assay._parser()
    parser.description = __doc__
    parser.add_argument("--core-edge-random-seed", type=int, default=305_177)
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
