"""Selective persistent-ledger ablation for future-mode shortlisting.

This development-only assay asks whether surprise-created event assemblies can
retain useful evidence across a conversation and narrow the frozen future-mode
population beyond a fast recurrent-state selector.  Event prototypes are born
without successor labels.  Active prototypes persist until the conversation
ends; later successor responsibilities train per-assembly residual logits and
pre-update leave-one-out log gain decides which assemblies are retained.

The candidate population and fast selector are frozen before ledger learning.
Matched current-turn-only and successor-shuffled readouts separate persistence
from extra capacity and target-frequency effects.  There is no native ALM
checkpoint, byte generation, autograd, backpropagation, or BPTT.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.categorical_future_mode_selector import (
    CategoricalFutureModeSelector,
    CategoricalFutureModeSelectorConfig,
)
from manifoldflow.dendritic_basin.future_turn_trajectory import FutureTurnTrajectory
from manifoldflow.dendritic_basin.selective_predictive_ledger import (
    PredictiveLedgerReadout,
    PredictiveLedgerReadoutConfig,
    SurprisePrototypeBank,
    SurprisePrototypeBankConfig,
)
from manifoldflow.dendritic_basin.surprise_grown_future_transition import (
    SurpriseGrownFutureTransition,
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
    dendritic_basin_surprise_future_transition_ablation as surprise_assay,
)

IMPLEMENTATION_PATHS = (
    "src/manifoldflow/experiments/dendritic_basin_selective_ledger_ablation.py",
    "tests/test_dendritic_basin_selective_ledger_ablation.py",
    "src/manifoldflow/dendritic_basin/selective_predictive_ledger.py",
    "tests/test_dendritic_basin_selective_predictive_ledger.py",
    *selector_assay.IMPLEMENTATION_PATHS,
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


def _fast_selector(args: argparse.Namespace) -> CategoricalFutureModeSelector:
    return CategoricalFutureModeSelector(
        CategoricalFutureModeSelectorConfig(
            context_dim=args.context_dimensions,
            mode_capacity=args.max_residual_modes,
            random_features=args.selector_random_features,
            feature_scale=args.selector_feature_scale,
            learning_rate=args.selector_learning_rate,
            weight_decay=args.selector_weight_decay,
            initial_readout_scale=args.selector_initial_readout_scale,
            probability_floor=args.selector_probability_floor,
            seed=args.selector_seed,
        )
    )


def _ledger_readout(args: argparse.Namespace) -> PredictiveLedgerReadout:
    return PredictiveLedgerReadout(
        PredictiveLedgerReadoutConfig(
            assembly_capacity=args.ledger_capacity,
            mode_capacity=args.max_residual_modes,
            learning_rate=args.ledger_learning_rate,
            utility_decay=args.ledger_utility_decay,
            weight_decay=args.ledger_weight_decay,
            probability_floor=args.selector_probability_floor,
        )
    )


def fit_prototype_bank_and_traces(
    examples: Sequence[surprise_assay.TransitionExample],
    event_projector: surprise_assay.ContextProjector,
    *,
    conversations: int,
    token_dimensions: int,
    args: argparse.Namespace,
) -> tuple[
    SurprisePrototypeBank,
    tuple[tuple[int, ...], ...],
    tuple[int | None, ...],
    dict[str, Any],
]:
    """Grow target-free prototypes, freeze them, then replay ledger traces."""

    tokens = np.stack(
        [example.context_state[:token_dimensions] for example in examples]
    )
    events = event_projector.transform(tokens)
    bank = SurprisePrototypeBank(
        SurprisePrototypeBankConfig(
            input_dim=events.shape[1],
            capacity=args.ledger_capacity,
            birth_distance=args.ledger_birth_distance,
            match_distance=args.ledger_match_distance,
            prototype_learning_rate=args.ledger_prototype_learning_rate,
        )
    )
    growth_order = surprise_assay._training_order(
        examples, conversations=conversations, seed=args.ledger_growth_order_seed
    )
    for raw_index in growth_order:
        bank.observe(events[int(raw_index)], learn=True)
    frozen = bank.fingerprint()

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.conversation_index].append(index)
    persistent: list[tuple[int, ...] | None] = [None] * len(examples)
    current: list[int | None] = [None] * len(examples)
    unmatched = 0
    for conversation in range(conversations):
        active: list[int] = []
        rows = grouped[conversation]
        if rows != sorted(rows, key=lambda item: examples[item].transition_index):
            raise AssertionError("training ledger chronology changed")
        for index in rows:
            assembly = bank.observe(events[index], learn=False)
            current[index] = assembly
            if assembly is None:
                unmatched += 1
            elif assembly not in active:
                active.append(assembly)
            persistent[index] = tuple(active)
    if bank.fingerprint() != frozen or any(item is None for item in persistent):
        raise AssertionError("frozen replay changed the prototype bank")
    return (
        bank,
        tuple(item for item in persistent if item is not None),
        tuple(current),
        {
            "growth_order_sha256": _sha256(
                np.asarray(growth_order, dtype="<i8").tobytes()
            ),
            "event_matrix_sha256": surprise_assay._array_sha256(events),
            "births": bank.births,
            "prototypes": bank.count,
            "unmatched_during_growth_at_capacity": bank.unmatched_at_capacity,
            "unmatched_frozen_training_events": unmatched,
            "frozen_fingerprint": frozen,
            "successor_targets_used_for_birth_or_matching": 0,
        },
    )


def train_selective_ledger(
    examples: Sequence[surprise_assay.TransitionExample],
    fast_states: np.ndarray,
    persistent: Sequence[tuple[int, ...]],
    current: Sequence[int | None],
    learner: SurpriseGrownFutureTransition,
    *,
    conversations: int,
    args: argparse.Namespace,
) -> tuple[
    CategoricalFutureModeSelector,
    dict[str, PredictiveLedgerReadout],
    np.ndarray,
    tuple[int, ...],
    dict[str, Any],
]:
    """Freeze a fast base, then train persistent and matched ledger residuals."""

    base = _fast_selector(args)
    order = surprise_assay._training_order(
        examples, conversations=conversations, seed=args.selector_order_seed
    )
    permutation, shuffle_audit = surprise_assay.successor_target_permutation(
        examples, seed=args.selector_shuffle_seed
    )
    global_counts = np.full(args.max_residual_modes + 1, args.global_prior)
    scorer = copy.deepcopy(learner)
    base_loss = 0.0
    for _pass in range(args.selector_training_passes):
        for raw_index in order:
            index = int(raw_index)
            example = examples[index]
            surface = selector_assay.responsibility_surface(
                scorer,
                fast_states[index],
                example.basal_prediction,
                example.target,
            )
            score = base.update(
                fast_states[index], surface.mode_ids, surface.target_probabilities
            )
            base_loss += score["log_loss_bits"]
            classes = np.asarray(
                (0, *(mode_id + 1 for mode_id in surface.mode_ids)), dtype=np.int64
            )
            global_counts[classes] += surface.target_probabilities
    base_fingerprint = base.fingerprint()

    readouts = {
        "persistent": _ledger_readout(args),
        "current_only": _ledger_readout(args),
        "successor_shuffled": _ledger_readout(args),
    }
    actual_scorer = copy.deepcopy(learner)
    shuffled_scorer = copy.deepcopy(learner)
    totals = {name: 0.0 for name in ("fast_base", *readouts)}
    events = 0
    for _pass in range(args.ledger_training_passes):
        for raw_index in order:
            index = int(raw_index)
            example = examples[index]
            actual = selector_assay.responsibility_surface(
                actual_scorer,
                fast_states[index],
                example.basal_prediction,
                example.target,
            )
            donor = examples[int(permutation[index])]
            shuffled = selector_assay.responsibility_surface(
                shuffled_scorer,
                fast_states[index],
                example.basal_prediction,
                donor.target,
            )
            if shuffled.mode_ids != actual.mode_ids:
                raise AssertionError("target shuffle changed causal candidates")
            prior = base.predict(fast_states[index], actual.mode_ids).probabilities
            current_active = () if current[index] is None else (int(current[index]),)
            predictions = {
                "persistent": readouts["persistent"].predict(
                    persistent[index], actual.mode_ids, prior
                ),
                "current_only": readouts["current_only"].predict(
                    current_active, actual.mode_ids, prior
                ),
                "successor_shuffled": readouts["successor_shuffled"].predict(
                    persistent[index], actual.mode_ids, prior
                ),
            }
            totals["fast_base"] += selector_assay._cross_entropy_bits(
                actual.target_probabilities, prior
            )
            for name, prediction in predictions.items():
                target = (
                    shuffled.target_probabilities
                    if name == "successor_shuffled"
                    else actual.target_probabilities
                )
                totals[name] += selector_assay._cross_entropy_bits(target, prediction)
            readouts["persistent"].update(
                persistent[index], actual.mode_ids, prior, actual.target_probabilities
            )
            readouts["current_only"].update(
                current_active, actual.mode_ids, prior, actual.target_probabilities
            )
            readouts["successor_shuffled"].update(
                persistent[index],
                shuffled.mode_ids,
                prior,
                shuffled.target_probabilities,
            )
            events += 1
    if base.fingerprint() != base_fingerprint:
        raise AssertionError("ledger training modified the frozen fast selector")
    allocations = {name: readout.trainable_bytes for name, readout in readouts.items()}
    if len(set(allocations.values())) != 1:
        raise AssertionError("ledger readout allocations differ")
    retained = readouts["persistent"].retained(
        minimum_fires=args.ledger_minimum_fires,
        minimum_utility_bits=args.ledger_minimum_utility_bits,
    )
    return (
        base,
        readouts,
        global_counts,
        retained,
        {
            "base_stage": {
                "events": len(order) * args.selector_training_passes,
                "passes": args.selector_training_passes,
                "prequential_log_loss_bits": base_loss
                / (len(order) * args.selector_training_passes),
                "frozen_fingerprint": base_fingerprint,
            },
            "ledger_stage": {
                "events": events,
                "passes": args.ledger_training_passes,
                "prequential_log_loss_bits": {
                    name: value / events for name, value in totals.items()
                },
                "trainable_bytes": allocations,
                "trainable_allocation_matched": True,
                "retention_source": "persistent ordered training utility only",
                "retained_assembly_ids": list(retained),
                "retained_assemblies": len(retained),
                "utility_bits": readouts["persistent"].utility.tolist(),
                "fires": readouts["persistent"].fires.tolist(),
                "fingerprints": {
                    name: readout.fingerprint() for name, readout in readouts.items()
                },
            },
            "order_sha256": _sha256(np.asarray(order, dtype="<i8").tobytes()),
            "successor_shuffle": shuffle_audit,
            "global_counts_sha256": surprise_assay._array_sha256(global_counts),
        },
    )


def _development_examples_with_ledger(
    model: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    event_projector: surprise_assay.ContextProjector,
    bank: SurprisePrototypeBank,
    *,
    mode: str,
) -> list[
    tuple[
        codec_assay.DevelopmentPair,
        str,
        surprise_assay.TransitionExample,
        tuple[int, ...],
        int | None,
    ]
]:
    rows = []
    frozen = bank.fingerprint()
    for pair_index, pair in enumerate(pairs):
        for side_index, (side_name, side) in enumerate(
            (("a", pair.side_a), ("b", pair.side_b))
        ):
            context = (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            if context is None:
                raise ValueError("requested development context is unavailable")
            observed = recurrent_assay.observed_turns_from_transcript(
                context, generation_context=True
            )
            turns = tuple(
                recurrent_assay._encoded_event(
                    codec, turn, source_label=pair.source_label
                )
                for turn in observed
            )
            target_turn = recurrent_assay._encoded_event(
                codec,
                recurrent_assay.ObservedTurn(
                    codec_assay.ASSISTANT_ROLE, side.full_response
                ),
                source_label=pair.source_label,
            )
            active: list[int] = []
            current = None
            for turn in turns:
                event = event_projector.transform(model._token(turn))
                current = bank.observe(event, learn=False)
                if current is not None and current not in active:
                    active.append(current)
            model.reset()
            try:
                for turn in turns:
                    model.observe(turn)
                snapshot = model.snapshot()
                basal = model.predicted_tape().slots[0].vector
            finally:
                model.reset()
            rows.append(
                (
                    pair,
                    side_name,
                    surprise_assay.TransitionExample(
                        conversation_index=pair_index,
                        transition_index=side_index,
                        episode_id=pair.pair_id,
                        context_state=snapshot.state,
                        basal_prediction=basal,
                        target=target_turn.vector,
                        target_turn=target_turn,
                    ),
                    tuple(active),
                    current,
                )
            )
    if bank.fingerprint() != frozen:
        raise AssertionError("development matching modified prototype bank")
    return rows


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
    for index, control in enumerate(
        (
            "fast_base",
            "current_only_retained",
            "global_frequency",
            "successor_shuffled_retained",
        )
    ):
        ledger = pair_means("persistent_retained", "categorical_log_loss_bits")
        other = pair_means(control, "categorical_log_loss_bits")
        contrasts[f"persistent_retained_minus_{control}_categorical_log_loss_bits"] = (
            surprise_assay._mean_ci(
                [left - right for left, right in zip(ledger, other, strict=True)],
                seed=seed + 10_000 + index,
                resamples=resamples,
            )
        )
    for index, metric in enumerate(("mse", "proper_bits_per_coordinate")):
        ledger = pair_means("persistent_retained", metric)
        basal = pair_means("basal", metric)
        contrasts[f"persistent_retained_minus_basal_{metric}"] = (
            surprise_assay._mean_ci(
                [left - right for left, right in zip(ledger, basal, strict=True)],
                seed=seed + 20_000 + index,
                resamples=resamples,
            )
        )
    return {
        "pairs": len(pair_ids),
        "sides": len(selected),
        "arms": arm_summary,
        "paired_contrasts": contrasts,
    }


def evaluate(
    basal: FutureTurnTrajectory,
    codec: codec_assay.FutureTurnCodec,
    pairs: Sequence[codec_assay.DevelopmentPair],
    fast_projector: surprise_assay.ContextProjector,
    event_projector: surprise_assay.ContextProjector,
    bank: SurprisePrototypeBank,
    learner: SurpriseGrownFutureTransition,
    base: CategoricalFutureModeSelector,
    readouts: Mapping[str, PredictiveLedgerReadout],
    global_counts: np.ndarray,
    retained: tuple[int, ...],
    proper_variance: np.ndarray,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before = {
        "basal": basal.fingerprint(include_state=False),
        "surprise": surprise_assay._learner_fingerprint(learner),
        "base": base.fingerprint(),
        "bank": bank.fingerprint(),
        **{name: item.fingerprint() for name, item in readouts.items()},
    }
    retained_set = set(retained)
    context_results = {}
    for mode_index, mode in enumerate(
        ("original_unanswered_prefix", "long_interruption_crossover")
    ):
        examples = _development_examples_with_ledger(
            basal, codec, pairs, event_projector, bank, mode=mode
        )
        scorer = copy.deepcopy(learner)
        rows = []
        for pair, side_name, example, active, current in examples:
            fast = fast_projector.transform(example.context_state)
            surface = selector_assay.responsibility_surface(
                scorer, fast, example.basal_prediction, example.target
            )
            prior = base.predict(fast, surface.mode_ids).probabilities
            active_retained = tuple(item for item in active if item in retained_set)
            current_retained = (
                () if current is None or current not in retained_set else (current,)
            )
            probability_arms = {
                "fast_base": prior,
                "persistent_all": readouts["persistent"].predict(
                    active, surface.mode_ids, prior
                ),
                "persistent_retained": readouts["persistent"].predict(
                    active_retained, surface.mode_ids, prior
                ),
                "current_only_retained": readouts["current_only"].predict(
                    current_retained, surface.mode_ids, prior
                ),
                "successor_shuffled_retained": readouts["successor_shuffled"].predict(
                    active_retained, surface.mode_ids, prior
                ),
            }
            distributions = {
                name: selector_assay._selection_from_probabilities(
                    surface.mode_ids, probabilities
                )
                for name, probabilities in probability_arms.items()
            }
            distributions["global_frequency"] = selector_assay._global_selection(
                global_counts, surface.mode_ids
            )
            distributions["affinity"] = selector_assay._selection_from_probabilities(
                surface.mode_ids, surface.affinity_probabilities
            )
            basal_metrics = {
                "categorical_log_loss_bits": -math.log2(
                    max(float(surface.target_probabilities[0]), 1e-12)
                ),
                "shortlist_size": 1.0,
                "shortlist_fraction": 1.0 / len(surface.target_probabilities),
                "posterior_mass_captured": float(surface.target_probabilities[0]),
                "posterior_argmax_recalled": float(
                    int(np.argmax(surface.target_probabilities)) == 0
                ),
                "mse": float(
                    np.mean(np.square(example.target - example.basal_prediction))
                ),
                "cosine": selector_assay._cosine(
                    example.target, example.basal_prediction
                ),
                "proper_bits_per_coordinate": (
                    surprise_assay.gaussian_bits_per_coordinate(
                        example.target,
                        (example.basal_prediction,),
                        (1.0,),
                        proper_variance,
                    )
                ),
            }
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "candidate_modes": len(surface.mode_ids),
                    "active_assemblies": len(active),
                    "retained_active_assemblies": len(active_retained),
                    "arms": {
                        "basal": basal_metrics,
                        **{
                            name: selector_assay.selection_metrics(
                                example.target,
                                surface,
                                distribution,
                                shortlist_mass=args.shortlist_mass,
                                shortlist_maximum=args.shortlist_maximum,
                                proper_variance=proper_variance,
                            )
                            for name, distribution in distributions.items()
                        },
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
        **{name: item.fingerprint() for name, item in readouts.items()},
    }
    if before != after or basal.observed_count != 0:
        raise AssertionError("evaluation modified frozen state")
    return {
        "context_modes": context_results,
        "fitted_fingerprints_before": before,
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
    comparison_names = {
        "fast_base": "persistent_retained_minus_fast_base_categorical_log_loss_bits",
        "current_only": (
            "persistent_retained_minus_current_only_retained_categorical_log_loss_bits"
        ),
        "global_frequency": (
            "persistent_retained_minus_global_frequency_categorical_log_loss_bits"
        ),
        "successor_shuffled": (
            "persistent_retained_minus_successor_shuffled_retained_"
            "categorical_log_loss_bits"
        ),
    }
    comparisons = {name: contrasts[key] for name, key in comparison_names.items()}
    mse = contrasts["persistent_retained_minus_basal_mse"]
    proper = contrasts["persistent_retained_minus_basal_proper_bits_per_coordinate"]
    ledger = arms["persistent_retained"]
    components = {
        "persistent_beats_fast_base": comparisons["fast_base"]["ci95_high"] < 0,
        "persistent_beats_current_only": comparisons["current_only"]["ci95_high"] < 0,
        "persistent_beats_global_frequency": comparisons["global_frequency"][
            "ci95_high"
        ]
        < 0,
        "persistent_beats_successor_shuffled": comparisons["successor_shuffled"][
            "ci95_high"
        ]
        < 0,
        "shortlist_size_at_most_locked_maximum": ledger["shortlist_size"]["mean"]
        <= args.shortlist_maximum,
        "posterior_mass_at_least_locked_minimum": ledger["posterior_mass_captured"][
            "mean"
        ]
        >= args.minimum_posterior_mass,
        "argmax_recall_at_least_locked_minimum": ledger["posterior_argmax_recalled"][
            "mean"
        ]
        >= args.minimum_argmax_recall,
        "shortlisted_prediction_beats_basal": mse["ci95_high"] < 0
        or proper["ci95_high"] < 0,
    }
    return {
        "scope": "long_interruption_crossover/evaluation_partition",
        "categorical_comparisons": comparisons,
        "mean_shortlist_size": ledger["shortlist_size"]["mean"],
        "mean_posterior_mass_captured": ledger["posterior_mass_captured"]["mean"],
        "posterior_argmax_recall": ledger["posterior_argmax_recalled"]["mean"],
        "persistent_retained_minus_basal_mse": mse,
        "persistent_retained_minus_basal_proper_bits_per_coordinate": proper,
        "selective_persistence_demonstrated": all(
            components[name]
            for name in (
                "persistent_beats_fast_base",
                "persistent_beats_current_only",
                "persistent_beats_global_frequency",
                "persistent_beats_successor_shuffled",
            )
        ),
        "components": components,
        "advances_to_expensive_competition": all(components.values()),
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
    bank, persistent, current, bank_audit = fit_prototype_bank_and_traces(
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
    base, readouts, global_counts, retained, ledger_training = train_selective_ledger(
        crossfit.examples,
        fast_states,
        persistent,
        current,
        ordered,
        conversations=len(conversations),
        args=args,
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
    evaluation = evaluate(
        full_basal,
        codec,
        pairs,
        fast_projector,
        event_projector,
        bank,
        ordered,
        base,
        readouts,
        global_counts,
        retained,
        proper_variance,
        args=args,
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit exceeded")
    result = {
        "schema_version": 1,
        "experiment": "selective_predictive_ledger_future_mode_development",
        "status": "complete_development",
        "claim_scope": {
            "schema4_train_and_development_only": True,
            "native_alm_loaded": False,
            "free_byte_generation_run": False,
            "expensive_particle_competition_run": False,
            "candidate_population_frozen_before_ledger_training": True,
            "fast_selector_frozen_before_ledger_training": True,
            "prototype_birth_and_matching_are_target_free": True,
            "retention_uses_preupdate_predictive_utility": True,
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "goal_or_preference_learning": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "dataset": {
            "path": str(args.dataset_dir),
            "manifest_schema_version": int(manifest["schema_version"]),
            "addressed_splits": ["train", "dev"],
        },
        "config": {
            "fast_dimensions": args.context_dimensions,
            "event_dimensions": args.ledger_event_dimensions,
            "ledger_capacity": args.ledger_capacity,
            "birth_distance": args.ledger_birth_distance,
            "match_distance": args.ledger_match_distance,
            "prototype_learning_rate": args.ledger_prototype_learning_rate,
            "ledger_learning_rate": args.ledger_learning_rate,
            "ledger_utility_decay": args.ledger_utility_decay,
            "ledger_weight_decay": args.ledger_weight_decay,
            "minimum_fires": args.ledger_minimum_fires,
            "minimum_utility_bits": args.ledger_minimum_utility_bits,
            "growth_order_seed": args.ledger_growth_order_seed,
            "shortlist_mass": args.shortlist_mass,
            "shortlist_maximum": args.shortlist_maximum,
            "minimum_posterior_mass": args.minimum_posterior_mass,
            "minimum_argmax_recall": args.minimum_argmax_recall,
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "training_conversations": len(conversations),
            "training_transitions": len(crossfit.examples),
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "cross_fitted_basal": crossfit.audit,
        "fast_projector": {
            "fingerprint": fast_projector.fingerprint(),
            "development_rows_used": 0,
        },
        "event_projector": {
            "fingerprint": event_projector.fingerprint(),
            "development_rows_used": 0,
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
    parser = selector_assay._parser()
    parser.description = __doc__
    parser.add_argument("--ledger-event-dimensions", type=int, default=24)
    parser.add_argument("--ledger-capacity", type=int, default=64)
    parser.add_argument("--ledger-birth-distance", type=float, default=1.0)
    parser.add_argument("--ledger-match-distance", type=float, default=2.5)
    parser.add_argument("--ledger-prototype-learning-rate", type=float, default=0.05)
    parser.add_argument("--ledger-training-passes", type=int, default=1)
    parser.add_argument("--ledger-learning-rate", type=float, default=0.08)
    parser.add_argument("--ledger-utility-decay", type=float, default=0.99)
    parser.add_argument("--ledger-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ledger-minimum-fires", type=int, default=32)
    parser.add_argument("--ledger-minimum-utility-bits", type=float, default=0.0)
    parser.add_argument("--ledger-growth-order-seed", type=int, default=261_401)
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
