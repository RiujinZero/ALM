"""Snapshot evaluation for the dendritic-predictor convergence 2x2.

Loads one mark's snapshots (the shared live dendritic organ plus each
arm's own trained base) and scores the 2x2 by independently checked free
generation on the development pairs, both context modes.

Per arm, conditions: the arm (prime with the organ-mixture construction --
K4 arms far-agreement reweighted, ledger arms add the bounded ledger
core), native (unprimed free-run of the arm's own base), the oracle
content code (the held-out answer -- leakage ceiling), reset-ledger
(ledger arms: core removed from the prime), and a cross-pair shuffled
control. Sampled (seeded draws) and greedy (argmax) branches, scored by
the paired byte-4gram reference discrimination margins, with output health
and the full text of every condition dumped for qualitative reading.
Teacher-forced BPB is not computed anywhere; local learning only; no
protected test payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.future_vector_core_edge import (
    mass_weighted_core_edge_split,
)
from manifoldflow.dendritic_basin.vector_conditioned_base import (
    StatefulVectorConditionedAdapter,
    StatefulVectorConditionedConfig,
    _normalized_distribution,
    generate_from_vector,
    generate_from_vector_after_context,
    generate_stateful_after_context,
)
from manifoldflow.experiments import (
    dendritic_basin_future_persistent_core_sweep_ablation as sweep_assay,
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
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_evaluate import (
    far_reweighted_candidate_mean,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond import (
    generate_unprimed,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_convergence_veccond2 import (
    ARM_NAMES,
    CROSS_ARM_CONDITIONS,
    CROSS_ARM_NAMES,
    FAR_ARMS,
    LEDGER_ARMS,
    mixture_point,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_generation_ablation import (
    _paired_reference_discrimination,
)

CONTEXT_MODES = ("original_unanswered_prefix", "long_interruption_crossover")

EVALUATION_BOOTSTRAP_SEED = 260_933

CONTRAST_CONDITIONS_BY_CONDITION = {
    "arm": "native",
    "arm_shuffled": "arm",
    "oracle": "native",
    "arm_reset": "arm",
}

STATEFUL_CONTRASTS = {
    "arm_stateful_minus_arm_context": ("arm_stateful", "arm_context"),
    "arm_stateful_clear_minus_arm_stateful": (
        "arm_stateful_clear",
        "arm_stateful",
    ),
    "arm_stateful_no_context_adapt_minus_arm_stateful": (
        "arm_stateful_no_context_adapt",
        "arm_stateful",
    ),
    "arm_stateful_no_generated_adapt_minus_arm_stateful": (
        "arm_stateful_no_generated_adapt",
        "arm_stateful",
    ),
    "arm_stateful_no_updates_minus_arm_stateful": (
        "arm_stateful_no_updates",
        "arm_stateful",
    ),
    "arm_shuffled_stateful_minus_arm_stateful": (
        "arm_shuffled_stateful",
        "arm_stateful",
    ),
    "oracle_stateful_minus_oracle_context": ("oracle_stateful", "oracle_context"),
}


def conditions_for_arm(arm: str, *, stateful: bool) -> tuple[str, ...]:
    conditions = ["native", "oracle", "arm"]
    if arm in LEDGER_ARMS:
        conditions.append("arm_reset")
    conditions.append("arm_shuffled")
    if not stateful:
        return tuple(conditions)
    conditions.extend(
        (
            "arm_context",
            "oracle_context",
            "oracle_stateful",
            "arm_stateful",
            "arm_stateful_clear",
            "arm_stateful_no_context_adapt",
            "arm_stateful_no_generated_adapt",
            "arm_stateful_no_updates",
            "arm_stateful_multi_turn",
        )
    )
    if arm in LEDGER_ARMS:
        conditions.append("arm_reset_stateful")
    conditions.append("arm_shuffled_stateful")
    return tuple(conditions)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    result: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(
        (
            "src/manifoldflow/experiments/"
            "dendritic_basin_shared_goal_convergence_veccond2_evaluate.py",
            "tests/"
            "test_dendritic_basin_shared_goal_convergence_veccond2_evaluate.py",
            "src/manifoldflow/dendritic_basin/vector_conditioned_base.py",
        )
    ):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        payload = path.read_bytes()
        result[relative] = {"bytes": len(payload), "sha256": _sha256(payload)}
    for relative, record in sweep_assay.implementation_hashes().items():
        result.setdefault(relative, record)
    return result


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="backslashreplace")


def arm_contrast_series(
    rows_by_condition: Mapping[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in rows_by_condition["native"]]
    values = {}
    for condition, rows in rows_by_condition.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"condition {condition} rows misaligned")
        values[condition] = [float(row[metric]) for row in rows]
    arms = {
        condition: codec_assay._mean_ci(
            values[condition], seed=seed, resamples=resamples
        )
        for condition in rows_by_condition
    }
    contrasts = {}
    fixed_contrasts = {
        "arm_minus_native": ("arm", "native"),
        "arm_minus_shuffled": ("arm", "arm_shuffled"),
        "oracle_minus_native": ("oracle", "native"),
        "arm_reset_minus_arm": ("arm_reset", "arm"),
    }
    for contrast_index, (name, (positive, negative)) in enumerate(
        fixed_contrasts.items()
    ):
        if positive not in values or negative not in values:
            continue
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 10_000 + contrast_index, resamples=resamples
        )
    for contrast_index, (name, (positive, negative)) in enumerate(
        STATEFUL_CONTRASTS.items()
    ):
        if positive not in values or negative not in values:
            continue
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired,
            seed=seed + 40_000 + contrast_index,
            resamples=resamples,
        )
    return {"conditions": arms, "contrasts": contrasts}


def cross_arm_series(
    arm_rows: Mapping[str, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in arm_rows["A_k1_nooledger"]]
    values = {}
    for arm, rows in arm_rows.items():
        if [row["pair_id"] for row in rows] != pair_ids:
            raise AssertionError(f"arm {arm} rows misaligned")
        values[arm] = [float(row[metric]) for row in rows]
    arms = {
        arm: codec_assay._mean_ci(values[arm], seed=seed, resamples=resamples)
        for arm in ARM_NAMES
    }
    contrasts = {}
    for contrast_index, name in enumerate(CROSS_ARM_NAMES):
        positive, negative = CROSS_ARM_CONDITIONS[name]
        paired = [
            a - b for a, b in zip(values[positive], values[negative], strict=True)
        ]
        contrasts[name] = codec_assay._mean_ci(
            paired, seed=seed + 20_000 + contrast_index, resamples=resamples
        )
    return {"arms": arms, "contrasts": contrasts}


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    organ_path = Path(args.snapshot_dir) / f"organ_mark_{int(args.mark):08d}.pkl"
    with organ_path.open("rb") as handle:
        organ_snapshot = pickle.load(handle)
    organ = organ_snapshot["organ"]
    far_w = (
        np.asarray(organ_snapshot["far_w"], dtype=np.float64)
        if "far_w" in organ_snapshot
        else None
    )
    arm_bases: dict[str, Any] = {}
    for arm in ARM_NAMES:
        arm_path = Path(args.snapshot_dir) / f"{arm}_mark_{int(args.mark):08d}.pkl"
        with arm_path.open("rb") as handle:
            arm_snap = pickle.load(handle)
        arm_bases[arm] = arm_snap["base"]
    if args.stateful:
        episodic_layer_configured = []
        inference_adaptation_frozen = []
        for cortex in arm_bases.values():
            if hasattr(cortex, "configure_episodic_layer"):
                cortex.configure_episodic_layer(False, 6, 5)
                episodic_layer_configured.append(True)
            if hasattr(cortex, "set_debug_freeze_inference_adaptation"):
                cortex.set_debug_freeze_inference_adaptation(True)
                inference_adaptation_frozen.append(True)
    else:
        episodic_layer_configured = []
        inference_adaptation_frozen = []
    updates_done = int(organ_snapshot["updates_done"])

    _manifest, verified, conversations, pairs = (
        surprise_assay.load_declared_training_and_development(args)
    )
    if getattr(args, "max_dev_pairs", 0):
        if args.max_dev_pairs < 1:
            raise ValueError("max_dev_pairs must be positive")
        pairs = pairs[: args.max_dev_pairs]
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
    latent_bank, _persistent, _current, bank_audit = (
        ledger_assay.fit_prototype_bank_and_traces(
            crossfit.examples,
            event_projector,
            conversations=len(conversations),
            token_dimensions=token_dimensions,
            args=args,
        )
    )
    ordered, _shuffled_overlay, _overlay = surprise_assay.train_surprise_arms(
        crossfit.examples,
        fast_projector,
        conversations=len(conversations),
        args=args,
    )
    base, readouts, _global_counts, _retained, _ledger_training = (
        ledger_assay.train_selective_ledger(
            crossfit.examples,
            fast_projector.transform(fast_matrix),
            _persistent,
            _current,
            ordered,
            conversations=len(conversations),
            args=args,
        )
    )
    scorer = copy.deepcopy(ordered)
    code_dim = int(codec.content_dimensions)

    frozen = {
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }

    transition = recurrent_assay.fit_recurrent_transition(
        sequences,
        content_dimensions=codec.content_dimensions,
        memory_decay=args.memory_decay,
        ridge=args.transition_ridge,
    )

    rollout_x: list[np.ndarray] = []
    rollout_y: list[np.ndarray] = []
    for sequence in sequences:
        assistant_events = [
            turn.vector
            for turn in sequence
            if turn.role_id == codec_assay.ASSISTANT_ROLE
        ]
        for current, following in zip(
            assistant_events[:-1], assistant_events[1:], strict=False
        ):
            rollout_x.append(current)
            rollout_y.append(following)
    rollout_mx = np.stack(rollout_x)
    rollout_my = np.stack(rollout_y)
    rollout_gram = rollout_mx.T @ rollout_mx
    rollout_gram.flat[:: rollout_gram.shape[0] + 1] += args.ridge_alpha
    rollout_map = np.linalg.solve(rollout_gram, rollout_mx.T @ rollout_my).T

    def eval_rows(mode: str) -> list[dict[str, Any]]:
        dev_examples = ledger_assay._development_examples_with_ledger(
            transition, codec, pairs, event_projector, latent_bank, mode=mode
        )
        scorer_copy = copy.deepcopy(ordered)
        rows = []
        for pair, side_name, example, _active, _current in dev_examples:
            fast = fast_projector.transform(example.context_state)
            surface = selector_assay_responsibility(
                scorer_copy, fast, example.basal_prediction, example.target
            )
            side = pair.side_a if side_name == "a" else pair.side_b
            context = (
                side.original_context
                if mode == "original_unanswered_prefix"
                else side.interrupted_context
            )
            if context is None:
                raise AssertionError("available context mode has a missing prefix")
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "side": side_name,
                    "context": context,
                    "fast": fast,
                    "basal": example.basal_prediction,
                    "target": example.target,
                    "source_label": pair.source_label,
                    "reference": side.response_window,
                    "full_response": side.full_response,
                    "surface": surface,
                }
            )
        return rows

    def selector_assay_responsibility(scorer_copy, fast, basal, target):
        return selector_responsibility(scorer_copy, fast, basal, target)

    def selector_responsibility(scorer_copy, fast, basal, target):
        from manifoldflow.experiments import (
            dendritic_basin_future_mode_selector_ablation as selector_module,
        )

        return selector_module.responsibility_surface(scorer_copy, fast, basal, target)

    def readout_core(row: Mapping[str, Any]) -> np.ndarray:
        surface = selector_responsibility(
            scorer, row["fast"], row["basal"], row["target"]
        )
        prior = base.predict(row["fast"], surface.mode_ids).probabilities
        probabilities = readouts["persistent"].predict((), surface.mode_ids, prior)
        flat = selector_shortlist(probabilities)
        split = mass_weighted_core_edge_split(surface.means, probabilities, flat)
        return np.clip(split.core, -6.0, 6.0)

    def selector_shortlist(probabilities: np.ndarray) -> tuple[int, ...]:
        from manifoldflow.experiments import (
            dendritic_basin_future_mode_selector_ablation as selector_module,
        )

        return selector_module.shortlist_indices(
            probabilities,
            mass=args.shortlist_mass,
            maximum=args.shortlist_maximum,
        )

    def organ_prime(
        arm: str, row: Mapping[str, Any], *, reset_ledger: bool
    ) -> np.ndarray:
        forecast = organ.forecast(row["fast"], row["basal"])
        if arm in FAR_ARMS:
            far = far_w @ np.concatenate([row["fast"], np.zeros(code_dim)])
            content = far_reweighted_candidate_mean(
                (
                    forecast.basal_prediction,
                    *(candidate.prediction for candidate in forecast.candidates),
                ),
                np.asarray(
                    (
                        forecast.basal_weight,
                        *(candidate.weight for candidate in forecast.candidates),
                    ),
                    dtype=np.float64,
                ),
                rollout_map,
                far,
                far_tau=args.far_tau,
            )
        else:
            content = mixture_point(
                forecast.prediction, forecast.candidates, use_basal=True
            )
        if arm in LEDGER_ARMS and not reset_ledger:
            content = content + readout_core(row)
        content = np.clip(content, -6.0, 6.0)
        if not np.all(np.isfinite(content)):
            raise ValueError(f"non-finite prime ({arm}) at {row['pair_id']}")
        organ.resolve(row["target"], learn=False)
        return content

    mode_outputs: dict[str, Any] = {}
    for mode in CONTEXT_MODES:
        rows = eval_rows(mode)
        index_by_key = {
            (row["pair_id"], row["side"]): index for index, row in enumerate(rows)
        }
        oracle_vectors = [
            codec.encode(
                row["full_response"],
                role=codec_assay.ASSISTANT_ROLE,
                source=row["source_label"],
            ).content
            for row in rows
        ]
        arm_primes = {
            arm: [organ_prime(arm, row, reset_ledger=False) for row in rows]
            for arm in ARM_NAMES
        }
        arm_reset_primes = {
            arm: [organ_prime(arm, row, reset_ledger=True) for row in rows]
            for arm in LEDGER_ARMS
        }
        shuffle = codec_assay.cross_pair_shuffle_indices(len(pairs), args.control_seed)
        shuffled_primes = {
            arm: [arm_primes[arm][int(other)] for other in shuffle]
            for arm in ARM_NAMES
        }
        stateful_config = StatefulVectorConditionedConfig(
            state_dimensions=args.stateful_state_dimensions,
            learning_rate=args.stateful_learning_rate,
            state_decay=args.stateful_state_decay,
            surprise_gate_threshold=args.stateful_surprise_threshold,
            surprise_gate_scale=args.stateful_surprise_scale,
            weight_decay=args.stateful_weight_decay,
            weight_clip=args.stateful_weight_clip,
            modulation_strength=args.stateful_modulation_strength,
            seed=args.stateful_seed,
        )

        stateful_controls = {
            "oracle_stateful": {
                "base_condition": "oracle",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": True,
            },
            "arm_stateful": {
                "base_condition": "arm",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": True,
            },
            "arm_stateful_clear": {
                "base_condition": "arm",
                "clear_memory": True,
                "adapt_context": True,
                "update_generated": True,
            },
            "arm_stateful_no_context_adapt": {
                "base_condition": "arm",
                "clear_memory": False,
                "adapt_context": False,
                "update_generated": True,
            },
            "arm_stateful_no_generated_adapt": {
                "base_condition": "arm",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": False,
            },
            "arm_stateful_no_updates": {
                "base_condition": "arm",
                "clear_memory": False,
                "adapt_context": False,
                "update_generated": False,
            },
            "arm_reset_stateful": {
                "base_condition": "arm_reset",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": True,
            },
            "arm_stateful_multi_turn": {
                "base_condition": "arm",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": False,
            },
            "arm_shuffled_stateful": {
                "base_condition": "arm_shuffled",
                "clear_memory": False,
                "adapt_context": True,
                "update_generated": True,
            },
        }

        def generate_stateful_condition(
            cortex: Any,
            vector: np.ndarray,
            context: bytes,
            *,
            clear_memory: bool,
            adapt_context: bool,
            update_generated: bool,
            uniform_draws: np.ndarray | None,
            config: StatefulVectorConditionedConfig = stateful_config,
        ) -> tuple[bytes, str]:
            generated = generate_stateful_after_context(
                cortex,
                vector,
                context,
                horizon=args.generation_horizon,
                config=config,
                uniform_draws=uniform_draws,
                adapt_context=adapt_context,
                clear_memory_after_context=clear_memory,
                update_generated=update_generated,
            )
            return generated.data, generated.probability_sha256

        def generate_context_condition(
            cortex: Any,
            vector: np.ndarray,
            context: bytes,
            *,
            uniform_draws: np.ndarray | None,
        ) -> tuple[bytes, str]:
            generated = generate_from_vector_after_context(
                cortex,
                vector,
                context,
                horizon=args.generation_horizon,
                uniform_draws=uniform_draws,
            )
            return generated.data, generated.probability_sha256

        def generate_stateful_multi_turn_condition(
            cortex: Any,
            vector: np.ndarray,
            context: bytes,
            *,
            uniform_draws: np.ndarray | None,
            config: StatefulVectorConditionedConfig = stateful_config,
        ) -> tuple[bytes, str]:
            adapter = StatefulVectorConditionedAdapter(cortex, config)
            adapter.prime(vector, clear_memory=False)
            adapter.observe_context(bytes(context), adapt=True)
            half = max(1, args.generation_horizon // 2)
            digest = hashlib.sha256()
            first_bytes = bytearray()
            adapter.set_generation_updates(False)
            for position in range(half):
                law = _normalized_distribution(adapter.predict_next_distribution())
                digest.update(law.tobytes())
                if uniform_draws is None:
                    symbol = int(np.argmax(law))
                else:
                    symbol = min(
                        int(
                            np.searchsorted(
                                np.cumsum(law), uniform_draws[position], side="right"
                            )
                        ),
                        255,
                    )
                first_bytes.append(symbol)
                adapter.advance_observation(symbol, learn=False)
            adapter.observe_turn(bytes(first_bytes), adapt=True, learn=False)
            adapter.end_turn()
            adapter.prime(vector)
            adapter.observe_context(bytes(first_bytes), adapt=True)
            adapter.set_generation_updates(True)
            second_bytes = bytearray()
            for position in range(args.generation_horizon - half):
                law = _normalized_distribution(adapter.predict_next_distribution())
                digest.update(law.tobytes())
                if uniform_draws is None:
                    symbol = int(np.argmax(law))
                else:
                    symbol = min(
                        int(
                            np.searchsorted(
                                np.cumsum(law),
                                uniform_draws[half + position],
                                side="right",
                            )
                        ),
                        255,
                    )
                second_bytes.append(symbol)
                adapter.advance_observation(symbol, learn=False)
            adapter.reset_episode()
            return bytes(first_bytes) + bytes(second_bytes), digest.hexdigest()

        arm_results: dict[str, dict[str, list[dict[str, Any]]]] = {
            arm: {} for arm in ARM_NAMES
        }
        pair_texts = []
        for pair in pairs:
            side_payloads: dict[str, dict[str, dict[str, Any]]] = {}
            for side_name in ("a", "b"):
                index = index_by_key[(pair.pair_id, side_name)]
                row = rows[index]
                draws = codec_assay._seeded_draws(
                    f"{mode}|{pair.pair_id}|{side_name}",
                    seed=args.sample_seed,
                    horizon=args.generation_horizon,
                )
                per_arm: dict[str, dict[str, dict[str, Any]]] = {}
                for arm in ARM_NAMES:
                    cortex = arm_bases[arm]
                    per_arm[arm] = {}
                    for condition in conditions_for_arm(arm, stateful=args.stateful):
                        if condition == "native":
                            sampled_data, sampled_digest = generate_unprimed(
                                cortex,
                                horizon=args.generation_horizon,
                                uniform_draws=draws,
                            )
                            greedy_data, greedy_digest = generate_unprimed(
                                cortex,
                                horizon=args.generation_horizon,
                                uniform_draws=None,
                            )
                        else:
                            if condition in stateful_controls:
                                control = stateful_controls[condition]
                                base_condition = control["base_condition"]
                                if base_condition == "oracle":
                                    vector = oracle_vectors[index]
                                elif base_condition == "arm":
                                    vector = arm_primes[arm][index]
                                elif base_condition == "arm_reset":
                                    vector = arm_reset_primes[arm][index]
                                elif base_condition == "arm_shuffled":
                                    vector = shuffled_primes[arm][index]
                                else:
                                    raise AssertionError(
                                        f"unknown stateful condition {condition}"
                                    )
                                sampled_data, sampled_digest = (
                                    generate_stateful_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        clear_memory=control["clear_memory"],
                                        adapt_context=control["adapt_context"],
                                        update_generated=control["update_generated"],
                                        uniform_draws=draws,
                                    )
                                )
                                greedy_data, greedy_digest = (
                                    generate_stateful_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        clear_memory=control["clear_memory"],
                                        adapt_context=control["adapt_context"],
                                        update_generated=control["update_generated"],
                                        uniform_draws=None,
                                    )
                                )
                            elif condition == "arm_stateful_multi_turn":
                                vector = arm_primes[arm][index]
                                sampled_data, sampled_digest = (
                                    generate_stateful_multi_turn_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        uniform_draws=draws,
                                    )
                                )
                                greedy_data, greedy_digest = (
                                    generate_stateful_multi_turn_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        uniform_draws=None,
                                    )
                                )
                            elif condition in {"arm_context", "oracle_context"}:
                                if condition == "arm_context":
                                    vector = arm_primes[arm][index]
                                else:
                                    vector = oracle_vectors[index]
                                sampled_data, sampled_digest = (
                                    generate_context_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        uniform_draws=draws,
                                    )
                                )
                                greedy_data, greedy_digest = (
                                    generate_context_condition(
                                        cortex,
                                        vector,
                                        row["context"],
                                        uniform_draws=None,
                                    )
                                )
                            else:
                                if condition == "oracle":
                                    vector = oracle_vectors[index]
                                elif condition == "arm":
                                    vector = arm_primes[arm][index]
                                elif condition == "arm_reset":
                                    vector = arm_reset_primes[arm][index]
                                elif condition == "arm_shuffled":
                                    vector = shuffled_primes[arm][index]
                                else:
                                    raise AssertionError(
                                        f"unknown condition {condition}"
                                    )
                                sampled_gen = generate_from_vector(
                                    cortex,
                                    vector,
                                    horizon=args.generation_horizon,
                                    uniform_draws=draws,
                                )
                                greedy_gen = generate_from_vector(
                                    cortex,
                                    vector,
                                    horizon=args.generation_horizon,
                                    uniform_draws=None,
                                )
                                sampled_data, sampled_digest = (
                                    sampled_gen.data,
                                    sampled_gen.probability_sha256,
                                )
                                greedy_data, greedy_digest = (
                                    greedy_gen.data,
                                    greedy_gen.probability_sha256,
                                )
                        per_arm[arm][condition] = {
                            "sampled_data": sampled_data,
                            "sampled_digest": sampled_digest,
                            "greedy_data": greedy_data,
                            "greedy_digest": greedy_digest,
                        }
                side_payloads[side_name] = per_arm | {"reference": row["reference"]}
            texts = {
                side_name: {
                    "reference_text": _text(side_payloads[side_name]["reference"]),
                    **{
                        f"{arm}.{condition}": {
                            "sampled_text": _text(
                                side_payloads[side_name][arm][condition]["sampled_data"]
                            ),
                            "sampled_probability_sha256": side_payloads[side_name][
                                arm
                            ][condition]["sampled_digest"],
                            "greedy_text": _text(
                                side_payloads[side_name][arm][condition]["greedy_data"]
                            ),
                            "greedy_probability_sha256": side_payloads[side_name][
                                arm
                            ][condition]["greedy_digest"],
                        }
                        for arm in ARM_NAMES
                        for condition in conditions_for_arm(arm, stateful=args.stateful)
                    },
                }
                for side_name in ("a", "b")
            }
            pair_texts.append(
                {
                    "pair_id": pair.pair_id,
                    "partition": pair.partition,
                    "sides": texts,
                }
            )
            for arm in ARM_NAMES:
                for condition in conditions_for_arm(arm, stateful=args.stateful):
                    sampled_pair = _paired_reference_discrimination(
                        side_payloads["a"][arm][condition]["sampled_data"],
                        side_payloads["b"][arm][condition]["sampled_data"],
                        side_payloads["a"]["reference"],
                        side_payloads["b"]["reference"],
                        dimensions=1024,
                    )
                    greedy_pair = _paired_reference_discrimination(
                        side_payloads["a"][arm][condition]["greedy_data"],
                        side_payloads["b"][arm][condition]["greedy_data"],
                        side_payloads["a"]["reference"],
                        side_payloads["b"]["reference"],
                        dimensions=1024,
                    )
                    diagnostics = {
                        side: codec_assay._generation_diagnostics(
                            side_payloads[side][arm][condition]["sampled_data"]
                        )
                        for side in ("a", "b")
                    }
                    arm_results[arm].setdefault(condition, []).append(
                        {
                            "pair_id": pair.pair_id,
                            "partition": pair.partition,
                            "sampled_margin": sampled_pair[
                                "correct_minus_crossed_margin"
                            ],
                            "greedy_margin": greedy_pair[
                                "correct_minus_crossed_margin"
                            ],
                            "sampled_printable": float(
                                np.mean(
                                    [
                                        diagnostics[side]["printable_fraction"]
                                        for side in ("a", "b")
                                    ]
                                )
                            ),
                            "sampled_utf8": float(
                                np.mean(
                                    [
                                        1.0
                                        if diagnostics[side]["valid_utf8"]
                                        else 0.0
                                        for side in ("a", "b")
                                    ]
                                )
                            ),
                            "sampled_maximum_run": max(
                                diagnostics[side]["maximum_identical_byte_run"]
                                for side in ("a", "b")
                            ),
                        }
                    )
        arm_summaries: dict[str, Any] = {}
        arm_metric_rows: dict[str, list[dict[str, Any]]] = {}
        for arm in ARM_NAMES:
            partition_summaries = {}
            for partition in ("evaluation", "selection", "all"):
                partition_rows = {
                    condition: [
                        row
                        for row in condition_rows
                        if partition == "all" or row["partition"] == partition
                    ]
                    for condition, condition_rows in arm_results[arm].items()
                }
                if not partition_rows["native"]:
                    continue
                partition_summaries[partition] = {
                    "pairs": len(partition_rows["native"]),
                    "sampled": arm_contrast_series(
                        partition_rows,
                        metric="sampled_margin",
                        seed=EVALUATION_BOOTSTRAP_SEED + updates_done,
                        resamples=args.bootstrap_resamples,
                    ),
                    "greedy": arm_contrast_series(
                        partition_rows,
                        metric="greedy_margin",
                        seed=EVALUATION_BOOTSTRAP_SEED + updates_done + 5_000,
                        resamples=args.bootstrap_resamples,
                    ),
                    "health": {
                        condition: {
                            "mean_sampled_printable_fraction": float(
                                np.mean([row["sampled_printable"] for row in rows])
                            ),
                            "mean_sampled_utf8_fraction": float(
                                np.mean([row["sampled_utf8"] for row in rows])
                            ),
                            "outputs_with_run_ge_16": sum(
                                1
                                for row in rows
                                if row["sampled_maximum_run"] >= 16
                            ),
                        }
                        for condition, rows in partition_rows.items()
                    },
                }
            arm_summaries[arm] = partition_summaries
            evaluation_rows = arm_results[arm].get("arm", [])
            arm_metric_rows[arm] = [
                row for row in evaluation_rows if row["partition"] == "evaluation"
            ]
        cross_arm = cross_arm_series(
            arm_metric_rows,
            metric="sampled_margin",
            seed=EVALUATION_BOOTSTRAP_SEED + updates_done + 30_000,
            resamples=args.bootstrap_resamples,
        )
        mode_outputs[mode] = {
            "arm_summaries": arm_summaries,
            "cross_arm": cross_arm,
            "pair_texts": pair_texts,
        }

    after = {
        "surprise": surprise_assay._learner_fingerprint(ordered),
        "base": base.fingerprint(),
        "bank": latent_bank.fingerprint(),
        "readout": readouts["persistent"].fingerprint(),
    }
    if frozen != after:
        raise AssertionError("evaluation modified frozen latent predictors")
    write_digest(args.digest, mode_outputs)
    elapsed = time.perf_counter() - started
    if elapsed > args.wall_limit_seconds:
        raise TimeoutError("wall limit reached")
    return {
        "schema_version": 1,
        "experiment": "shared_goal_convergence_veccond2_evaluate",
        "status": "complete_development",
        "claim_scope": {
            "development_only": True,
            "inverted_architecture": True,
            "dendritic_predictor": True,
            "stateful_transducer": bool(args.stateful),
            "free_generation_primary": True,
            "teacher_forced_bpb_computed": False,
            "protected_test_payload_accessed": False,
        },
        "implementation_files": implementation_hashes(),
        "verified_inputs": verified,
        "snapshot": {
            "snapshot_dir": str(args.snapshot_dir),
            "mark": int(args.mark),
            "updates_done": updates_done,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": codec_assay._sha256_file(args.checkpoint),
        },
        "config": {
            "generation_horizon": args.generation_horizon,
            "max_dev_pairs": getattr(args, "max_dev_pairs", 0),
            "far_tau": args.far_tau,
            "ridge_alpha": args.ridge_alpha,
            "sample_seed": args.sample_seed,
            "control_seed": args.control_seed,
            "stateful": (
                {
                    "state_dimensions": args.stateful_state_dimensions,
                    "learning_rate": args.stateful_learning_rate,
                    "state_decay": args.stateful_state_decay,
                    "surprise_gate_threshold": args.stateful_surprise_threshold,
                    "surprise_gate_scale": args.stateful_surprise_scale,
                    "weight_decay": args.stateful_weight_decay,
                    "weight_clip": args.stateful_weight_clip,
                    "modulation_strength": args.stateful_modulation_strength,
                    "seed": args.stateful_seed,
                    "update_on_generated_default": True,
                    "base_episodic_layer_configured": bool(episodic_layer_configured),
                    "base_inference_adaptation_frozen": bool(
                        inference_adaptation_frozen
                    ),
                }
                if args.stateful
                else None
            ),
            "excluded_development_pair_ids": list(args.exclude_dev_pair_ids),
        },
        "counts": {
            "development_pairs": len(pairs),
            "development_sides": 2 * len(pairs),
        },
        "prototype_bank": bank_audit,
        "evaluation": {
            "context_modes": mode_outputs,
            "latent_fingerprints_unchanged": True,
            "digest_path": str(args.digest),
        },
        "wall_seconds": elapsed,
    }


def write_digest(path: Path, modes: Mapping[str, Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for mode, data in modes.items():
        for row in data["pair_texts"]:
            lines.append(
                "=== mode="
                f"{mode} pair={row['pair_id']} partition={row['partition']} ==="
            )
            for side_name in ("a", "b"):
                side = row["sides"][side_name]
                lines.append(f"--- side {side_name} reference ---")
                lines.append(side["reference_text"])
                for key, payload in side.items():
                    if key == "reference_text":
                        continue
                    lines.append(f"--- side {side_name} | {key} | sampled ---")
                    lines.append(payload["sampled_text"])
                    lines.append(f"--- side {side_name} | {key} | greedy ---")
                    lines.append(payload["greedy_text"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = sweep_assay._parser()
    parser.description = __doc__
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--mark", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digest", type=Path, required=True)
    parser.add_argument("--generation-horizon", type=int, default=128)
    parser.add_argument("--far-tau", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--sample-seed", type=int, default=260_931)
    parser.add_argument("--control-seed", type=int, default=260_932)
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--stateful-state-dimensions", type=int, default=32)
    parser.add_argument("--stateful-learning-rate", type=float, default=0.03)
    parser.add_argument("--stateful-state-decay", type=float, default=0.98)
    parser.add_argument("--stateful-surprise-threshold", type=float, default=2.0)
    parser.add_argument("--stateful-surprise-scale", type=float, default=1.0)
    parser.add_argument("--stateful-weight-decay", type=float, default=0.001)
    parser.add_argument("--stateful-weight-clip", type=float, default=4.0)
    parser.add_argument("--stateful-modulation-strength", type=float, default=1.0)
    parser.add_argument("--stateful-seed", type=int, default=260_934)
    parser.set_defaults(wall_limit_seconds=14_400.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_evaluation(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
