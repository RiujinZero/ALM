"""Evaluate a persistent future code on an exact local-cancellation assay.

The dataset is produced by ``scripts/build_goal_trajectory_ablation.py``.
Each pair contains two different conversations followed by the same long
interruption and resume cue.  Both held-out responses are scored after both
contexts.  A bounded local model must therefore emit bit-identical traces for
the same candidate under the two contexts.  A native base may legitimately
retain earlier context and leave a nonzero raw pairing residual, so the primary
effect is a per-pair difference-in-differences: each condition's raw symmetric
margin minus that pair's goal-off raw margin.

This experiment separates three questions:

* can a future-consequence code describe a response (oracle code);
* can the pending event infer that code before the interruption (causal code);
* can a matched goal byte head turn the code into an action preference.

The optional frozen cortex supplies the base distribution.  Its state is reset
and re-primed for every branch, learning and episodic writes are disabled, and
the goal code is clamped while a candidate is teacher-forced.  Guidance is a
likelihood-ratio correction, so beta zero is the exact unmodified cortex.
The raw goal-off residual and exact local-q0 trace invariant are both reported.

Natural dialogue can establish long-context compatibility, not goal identity:
topic, entity, style, or request memory may explain a positive result.  The
counterbalanced synthetic trajectory assay is the goal-specific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.goal_conditioned_byte_head import (
    EpisodeByteTrace,
    MatchedGoalByteHead,
    shuffled_goal_control,
)
from manifoldflow.experiments.dendritic_basin_shared_goal_response_ablation import (
    DEFAULT_SUCCESSOR_GAMMAS,
    PrototypeCodebook,
    RidgeMap,
    _successor_signature,
    classifier_probabilities,
    fit_codebook,
    fit_ridge,
    hashed_text_vector,
    normalize,
    parse_turns,
)


@dataclass(frozen=True)
class TrainingExample:
    episode_id: str
    response: bytes
    prefix: bytes
    evidence: np.ndarray
    response_vector: np.ndarray
    goal_target_vector: np.ndarray


@dataclass(frozen=True)
class CrossoverPair:
    pair_id: str
    context_a: bytes
    context_b: bytes
    response_a: bytes
    response_b: bytes
    evidence_a: np.ndarray
    evidence_b: np.ndarray
    source_label: str
    goal_target_a: bytes | None = None
    goal_target_b: bytes | None = None
    goal_target_kind: str = "response"
    goal_target_vector_a: np.ndarray | None = None
    goal_target_vector_b: np.ndarray | None = None


def _slice(blob: bytes, record: dict[str, Any]) -> bytes:
    start = int(record["start"])
    end = int(record["end"])
    if start < 0 or end < start or end > len(blob):
        raise ValueError("sidecar range lies outside its blob")
    payload = blob[start:end]
    expected = record.get("sha256")
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("sidecar range checksum does not match its blob")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"sidecar row {line_number} is not an object")
            rows.append(row)
    return rows


def _event_evidence(
    conversation: bytes,
    turns: list[dict[str, Any]],
    target_user_index: int,
    *,
    dimensions: int,
    decays: tuple[float, ...],
    offsets_are_absolute: bool,
    episode_start: int = 0,
) -> np.ndarray:
    """Encode a pending event plus causal multi-timescale conversation state."""

    vectors: list[np.ndarray] = []
    for turn in turns:
        if int(turn["index"]) > target_user_index:
            break
        start = int(turn["content_start"])
        end = int(turn["end"])
        if offsets_are_absolute:
            start -= episode_start
            end -= episode_start
        if start < 0 or end < start or end > len(conversation):
            raise ValueError("turn offsets lie outside the episode")
        text = conversation[start:end].decode("utf-8", errors="replace")
        vectors.append(hashed_text_vector(text, dimensions))
    if not vectors:
        raise ValueError("episode has no causal events")
    states = [np.zeros(dimensions, dtype=np.float64) for _ in decays]
    for vector in vectors:
        for index, decay in enumerate(decays):
            states[index] = decay * states[index] + (1.0 - decay) * vector
    return np.concatenate((vectors[-1], *(normalize(state) for state in states)))


def _goal_target_vector(
    payload: bytes,
    kind: str,
    *,
    dimensions: int,
    successor_gammas: tuple[float, ...],
) -> np.ndarray:
    """Adapt a self-supervised future target into a modality-neutral vector."""

    text = payload.decode("utf-8", errors="replace")
    if kind in {"response", "interlocutor"}:
        return hashed_text_vector(text, dimensions)
    if kind == "successor":
        turns = parse_turns(text)
        if not turns:
            raise ValueError("successor goal target must contain at least one event")
        future = [hashed_text_vector(turn.text, dimensions) for turn in turns]
        return _successor_signature(future, successor_gammas)
    raise ValueError(f"unknown goal target kind: {kind}")


def load_training_examples(
    dataset_dir: Path,
    *,
    dimensions: int,
    decays: tuple[float, ...],
    successor_gammas: tuple[float, ...] = DEFAULT_SUCCESSOR_GAMMAS,
) -> list[TrainingExample]:
    blob = (dataset_dir / "train.bin").read_bytes()
    rows = _read_jsonl(dataset_dir / "train.index.jsonl")
    examples: list[TrainingExample] = []
    for row in rows:
        start = int(row["start"])
        episode = blob[start : int(row["end"])]
        target = row["target_offsets"]
        response = blob[
            int(target["score_window_start"]) : int(target["score_window_end"])
        ]
        role_start = int(target["content_start"])
        prefix_start = max(start, role_start - 32)
        prefix = blob[prefix_start:role_start]
        evidence = _event_evidence(
            episode,
            list(row["turns"]),
            int(row["target_user_turn_index"]),
            dimensions=dimensions,
            decays=decays,
            offsets_are_absolute=True,
            episode_start=start,
        )
        goal_record = row.get("goal_target")
        if goal_record is None:
            goal_kind = "response"
            goal_payload = response
        else:
            goal_kind = str(goal_record["kind"])
            goal_payload = _slice(blob, goal_record)
        examples.append(
            TrainingExample(
                episode_id=str(row["episode_id"]),
                response=response,
                prefix=prefix,
                evidence=evidence,
                response_vector=hashed_text_vector(
                    response.decode("utf-8", errors="replace"), dimensions
                ),
                goal_target_vector=_goal_target_vector(
                    goal_payload,
                    goal_kind,
                    dimensions=dimensions,
                    successor_gammas=successor_gammas,
                ),
            )
        )
    if len(examples) < 4:
        raise ValueError("at least four indexed training episodes are required")
    return examples


def _context_evidence(
    context: bytes,
    metadata: dict[str, Any],
    *,
    dimensions: int,
    decays: tuple[float, ...],
) -> np.ndarray:
    end = int(metadata["target_user_content_end"])
    if end != int(metadata["unanswered_prefix_bytes"]) or end > len(context):
        raise ValueError("context does not end its original prefix at the target user")
    prefix = context[:end]
    # Reparse only the original prefix: the synthetic bridge must not become
    # evidence for the pending future code.
    parsed = parse_turns(prefix.decode("utf-8", errors="replace"))
    if not parsed or parsed[-1].role != "USER":
        raise ValueError("original crossover prefix does not end in a USER event")
    vectors = [hashed_text_vector(turn.text, dimensions) for turn in parsed]
    states = [np.zeros(dimensions, dtype=np.float64) for _ in decays]
    for vector in vectors:
        for index, decay in enumerate(decays):
            states[index] = decay * states[index] + (1.0 - decay) * vector
    return np.concatenate((vectors[-1], *(normalize(state) for state in states)))


def load_crossover_pairs(
    dataset_dir: Path,
    split: str,
    *,
    dimensions: int,
    decays: tuple[float, ...],
    successor_gammas: tuple[float, ...] = DEFAULT_SUCCESSOR_GAMMAS,
) -> list[CrossoverPair]:
    rows = _read_jsonl(dataset_dir / f"{split}_pairs.jsonl")
    blob = (dataset_dir / f"{split}_pairs.bin").read_bytes()
    pairs: list[CrossoverPair] = []
    for row in rows:
        if not all(bool(value) for value in row["invariants"].values()):
            raise ValueError(
                f"pair {row['pair_id']} has a false construction invariant"
            )
        ranges = row["blob_ranges"]
        context_a = _slice(blob, ranges["context_a"])
        context_b = _slice(blob, ranges["context_b"])
        response_a = _slice(blob, ranges["response_a"])
        response_b = _slice(blob, ranges["response_b"])
        goal_kind = str(row.get("goal_target_kind", "response"))
        goal_target_a = (
            response_a
            if "goal_target_a" not in ranges
            else _slice(blob, ranges["goal_target_a"])
        )
        goal_target_b = (
            response_b
            if "goal_target_b" not in ranges
            else _slice(blob, ranges["goal_target_b"])
        )
        common = int(row["common_suffix"]["bytes"])
        if context_a[-common:] != context_b[-common:]:
            raise ValueError("declared common crossover suffix is not identical")
        pairs.append(
            CrossoverPair(
                pair_id=str(row["pair_id"]),
                context_a=context_a,
                context_b=context_b,
                response_a=response_a,
                response_b=response_b,
                evidence_a=_context_evidence(
                    context_a,
                    row["episodes"]["a"],
                    dimensions=dimensions,
                    decays=decays,
                ),
                evidence_b=_context_evidence(
                    context_b,
                    row["episodes"]["b"],
                    dimensions=dimensions,
                    decays=decays,
                ),
                source_label=str(row["source_label"]),
                goal_target_a=goal_target_a,
                goal_target_b=goal_target_b,
                goal_target_kind=goal_kind,
                goal_target_vector_a=_goal_target_vector(
                    goal_target_a,
                    goal_kind,
                    dimensions=dimensions,
                    successor_gammas=successor_gammas,
                ),
                goal_target_vector_b=_goal_target_vector(
                    goal_target_b,
                    goal_kind,
                    dimensions=dimensions,
                    successor_gammas=successor_gammas,
                ),
            )
        )
    if not pairs:
        raise ValueError(f"{split} sidecar contains no pairs")
    return pairs


def _one_hot(labels: np.ndarray, count: int) -> np.ndarray:
    return np.eye(count, dtype=np.float64)[labels]


def _fit_mechanism(
    examples: list[TrainingExample],
    *,
    prototypes: int,
    prototype_iterations: int,
    ridge_alpha: float,
    context_order: int,
    context_buckets: int,
    seed: int,
) -> tuple[
    PrototypeCodebook, RidgeMap, MatchedGoalByteHead, MatchedGoalByteHead, np.ndarray
]:
    goal_targets = np.asarray([row.goal_target_vector for row in examples])
    codebook = fit_codebook(
        goal_targets,
        prototypes=prototypes,
        iterations=prototype_iterations,
        seed=seed,
    )
    labels = codebook.assign(goal_targets)
    codes = _one_hot(labels, prototypes)
    predictor = fit_ridge(
        np.asarray([row.evidence for row in examples]), codes, ridge_alpha
    )
    head = MatchedGoalByteHead(
        prototypes,
        context_order=context_order,
        context_bucket_count=context_buckets,
    )
    episodes = [
        EpisodeByteTrace(row.response, code, prefix=row.prefix)
        for row, code in zip(examples, codes, strict=True)
    ]
    head.fit_episodes(episodes)
    shuffled_episodes, _ = shuffled_goal_control(episodes, seed=seed + 97)
    shuffled_head = MatchedGoalByteHead(
        prototypes,
        context_order=context_order,
        context_bucket_count=context_buckets,
    )
    shuffled_head.fit_episodes(shuffled_episodes)
    prior = np.bincount(labels, minlength=prototypes).astype(np.float64)
    prior /= prior.sum()
    return codebook, predictor, head, shuffled_head, prior


def _base_trace_from_head(
    head: MatchedGoalByteHead, prefix: bytes, candidate: bytes, prior: np.ndarray
) -> np.ndarray:
    local, _ = head.trace_probabilities(prefix, candidate, prior)
    return local


def _native_trace(cortex: object, prefix: bytes, candidate: bytes) -> np.ndarray:
    cortex.reset_episode()
    if prefix:
        cortex.observe_chunk_probabilities(
            np.frombuffer(prefix, dtype=np.uint8), learn=False
        )
    trace = np.empty((len(candidate), 256), dtype=np.float64)
    for index, target in enumerate(candidate):
        trace[index] = cortex.predict_next_distribution()
        cortex.advance_observation(target, learn=False)
    return trace


def _corrected_log_score(
    base_trace: np.ndarray,
    local_trace: np.ndarray,
    goal_trace: np.ndarray,
    candidate: bytes,
    *,
    beta: float,
    log_ratio_clip: float,
) -> float:
    if beta == 0.0:
        probability = base_trace
    else:
        ratio = np.clip(
            np.log(goal_trace) - np.log(local_trace),
            -log_ratio_clip,
            log_ratio_clip,
        )
        logits = np.log(np.maximum(base_trace, 1e-300)) + beta * ratio
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
    targets = np.frombuffer(candidate, dtype=np.uint8)
    return float(
        np.log2(np.maximum(probability[np.arange(len(targets)), targets], 1e-300)).sum()
    )


def _condition_margin(
    pair: CrossoverPair,
    head: MatchedGoalByteHead,
    base: dict[str, np.ndarray],
    code_a: np.ndarray,
    code_b: np.ndarray,
    *,
    beta: float,
    log_ratio_clip: float,
) -> float:
    scores: dict[str, float] = {}
    for context_name, context in (("a", pair.context_a), ("b", pair.context_b)):
        for response_name, response in (("a", pair.response_a), ("b", pair.response_b)):
            code = code_a if context_name == "a" else code_b
            local, goal = head.trace_probabilities(context, response, code)
            scores[context_name + response_name] = _corrected_log_score(
                base[context_name + response_name],
                local,
                goal,
                response,
                beta=beta,
                log_ratio_clip=log_ratio_clip,
            )
    # Pair equal terms before addition.  When local/base traces are exactly
    # identical this produces a bit-for-bit zero, rather than roundoff from a
    # mathematically equivalent four-term reassociation.
    return (scores["aa"] - scores["ba"]) + (scores["bb"] - scores["ab"])


def _summary(
    values: np.ndarray, *, seed: int, resamples: int
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("cannot summarize an empty margin vector")
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        values[rng.integers(0, len(values), size=(resamples, len(values)))], axis=1
    )
    observed = float(np.mean(values))
    sign_means = np.mean(
        values[None, :] * rng.choice((-1.0, 1.0), size=(resamples, len(values))),
        axis=1,
    )
    return {
        "pairs": int(len(values)),
        "mean_bits": observed,
        "median_bits": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "zero_fraction": float(np.mean(values == 0.0)),
        "mean_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "mean_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "one_sided_sign_flip_p": float(
            (1 + np.sum(sign_means >= observed)) / (resamples + 1)
        ),
    }


def _load_native(checkpoint: Path | None) -> object | None:
    if checkpoint is None:
        return None
    from manifoldflow.experiments import (
        dendritic_basin_phase126_three_epoch_generation as phase126,
    )

    cortex = phase126._load_any_cortex(checkpoint)
    # The native episodic store can write even with learn=False.  It must be
    # disabled so branch order cannot alter a supposedly frozen evaluator.
    cortex.configure_episodic_layer(False, 6, 5)
    # ``learn=False`` normally still adapts the burst EMA and residual-context
    # trust. Counterfactual scoring reuses one cortex across candidate
    # branches, so freeze those runtime updates as well. The flag is an
    # ablation-only no-op for ordinary training/deployment unless enabled.
    cortex.set_debug_freeze_inference_adaptation(True)
    return cortex


def _evaluate_seed(
    examples: list[TrainingExample],
    dev_pairs: list[CrossoverPair],
    test_pairs: list[CrossoverPair],
    *,
    prototypes: int,
    prototype_iterations: int,
    ridge_alpha: float,
    context_order: int,
    context_buckets: int,
    beta_values: tuple[float, ...],
    log_ratio_clip: float,
    seed: int,
    checkpoint: Path | None,
    resamples: int,
    require_exact_native: bool,
) -> dict[str, Any]:
    codebook, predictor, head, shuffled_head, prior = _fit_mechanism(
        examples,
        prototypes=prototypes,
        prototype_iterations=prototype_iterations,
        ridge_alpha=ridge_alpha,
        context_order=context_order,
        context_buckets=context_buckets,
        seed=seed,
    )
    cortex = _load_native(checkpoint)

    def evaluate(
        pairs: list[CrossoverPair], beta: float, *, collect_invariants: bool
    ) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float | int | bool]
    ]:
        margins = {
            name: []
            for name in (
                "goal_off",
                "oracle_future",
                "causal_pending_event",
                "swapped_oracle",
                "swapped_causal",
                "prior_only",
                "shuffled_action_mapping",
                "shuffled_episode_code",
            )
        }
        rng = np.random.default_rng(seed + 31_337 + len(pairs))
        shuffled_codes: list[np.ndarray] = []
        causal_codes: list[tuple[np.ndarray, np.ndarray]] = []
        oracle_codes: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in pairs:
            goal_vectors = np.asarray(
                [
                    pair.goal_target_vector_a
                    if pair.goal_target_vector_a is not None
                    else hashed_text_vector(
                        pair.response_a.decode("utf-8", errors="replace"),
                        codebook.prototypes.shape[1],
                    ),
                    pair.goal_target_vector_b
                    if pair.goal_target_vector_b is not None
                    else hashed_text_vector(
                        pair.response_b.decode("utf-8", errors="replace"),
                        codebook.prototypes.shape[1],
                    ),
                ]
            )
            labels = codebook.assign(goal_vectors)
            oracle_codes.append(
                (
                    _one_hot(labels[:1], prototypes)[0],
                    _one_hot(labels[1:], prototypes)[0],
                )
            )
            probability = classifier_probabilities(
                predictor, np.asarray([pair.evidence_a, pair.evidence_b])
            )
            causal_codes.append((probability[0], probability[1]))
            shuffled_codes.extend((probability[0], probability[1]))
        permutation = rng.permutation(len(shuffled_codes))
        shuffled_codes = [shuffled_codes[int(index)] for index in permutation]

        max_local_difference = 0.0
        max_base_difference = 0.0
        distinct_oracle = 0
        for pair_index, pair in enumerate(pairs):
            oracle_a, oracle_b = oracle_codes[pair_index]
            causal_a, causal_b = causal_codes[pair_index]
            distinct_oracle += int(np.argmax(oracle_a) != np.argmax(oracle_b))
            base: dict[str, np.ndarray] = {}
            order = [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
            rng.shuffle(order)
            for context_name, response_name in order:
                context = pair.context_a if context_name == "a" else pair.context_b
                response = pair.response_a if response_name == "a" else pair.response_b
                base[context_name + response_name] = (
                    _base_trace_from_head(head, context, response, prior)
                    if cortex is None
                    else _native_trace(cortex, context, response)
                )
            for response_name, response in (
                ("a", pair.response_a),
                ("b", pair.response_b),
            ):
                local_a, _ = head.trace_probabilities(pair.context_a, response, prior)
                local_b, _ = head.trace_probabilities(pair.context_b, response, prior)
                max_local_difference = max(
                    max_local_difference, float(np.max(np.abs(local_a - local_b)))
                )
                max_base_difference = max(
                    max_base_difference,
                    float(
                        np.max(
                            np.abs(
                                base["a" + response_name] - base["b" + response_name]
                            )
                        )
                    ),
                )
            assignments = {
                "goal_off": (prior, prior, 0.0, head),
                "oracle_future": (oracle_a, oracle_b, beta, head),
                "causal_pending_event": (causal_a, causal_b, beta, head),
                "swapped_oracle": (oracle_b, oracle_a, beta, head),
                "swapped_causal": (causal_b, causal_a, beta, head),
                "prior_only": (prior, prior, beta, head),
                "shuffled_action_mapping": (
                    oracle_a,
                    oracle_b,
                    beta,
                    shuffled_head,
                ),
                "shuffled_episode_code": (
                    shuffled_codes[2 * pair_index],
                    shuffled_codes[2 * pair_index + 1],
                    beta,
                    head,
                ),
            }
            for name, (
                code_a,
                code_b,
                condition_beta,
                action_head,
            ) in assignments.items():
                margins[name].append(
                    _condition_margin(
                        pair,
                        action_head,
                        base,
                        code_a,
                        code_b,
                        beta=condition_beta,
                        log_ratio_clip=log_ratio_clip,
                    )
                )
        raw_margins = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in margins.items()
        }
        goal_off_residual = raw_margins["goal_off"]
        adjusted_effects = {
            name: values - goal_off_residual for name, values in raw_margins.items()
        }
        invariants: dict[str, float | int | bool] = {
            "max_abs_local_q0_trace_difference": max_local_difference,
            "max_abs_base_trace_difference": max_base_difference,
            "exact_local_q0_trace_identity": max_local_difference == 0.0,
            "exact_base_trace_identity": max_base_difference == 0.0,
            "base_trace_identity_required": cortex is None or require_exact_native,
            "distinct_oracle_code_pair_fraction": distinct_oracle / len(pairs),
        }
        if collect_invariants and max_local_difference != 0.0:
            raise AssertionError("matched local q0 traces did not cancel exactly")
        if collect_invariants and cortex is None and max_base_difference != 0.0:
            raise AssertionError("matched local base traces did not cancel exactly")
        if (
            collect_invariants
            and cortex is not None
            and require_exact_native
            and max_base_difference != 0.0
        ):
            raise AssertionError("native base traces were not bit-identical")
        return raw_margins, adjusted_effects, invariants

    dev_curves: dict[str, dict[str, float]] = {}
    best: tuple[float, float] | None = None
    for beta in beta_values:
        raw_values, adjusted_effects, _ = evaluate(
            dev_pairs, beta, collect_invariants=False
        )
        oracle_mean = float(np.mean(adjusted_effects["oracle_future"]))
        causal_mean = float(np.mean(adjusted_effects["causal_pending_event"]))
        dev_curves[str(beta)] = {
            "goal_off_raw_mean_residual_bits": float(np.mean(raw_values["goal_off"])),
            "oracle_adjusted_mean_effect_bits": oracle_mean,
            "causal_adjusted_mean_effect_bits": causal_mean,
            "oracle_raw_mean_margin_bits": float(np.mean(raw_values["oracle_future"])),
            "causal_raw_mean_margin_bits": float(
                np.mean(raw_values["causal_pending_event"])
            ),
        }
        key = (oracle_mean, -beta)
        if best is None or key > (best[0], -best[1]):
            best = (oracle_mean, beta)
    assert best is not None
    selected_beta = best[1]
    raw_test_values, adjusted_test_effects, invariants = evaluate(
        test_pairs, selected_beta, collect_invariants=True
    )

    train_vectors = np.asarray([row.goal_target_vector for row in examples])
    train_labels = codebook.assign(train_vectors)
    counts = np.bincount(train_labels, minlength=prototypes)
    test_evidence = np.asarray(
        [
            evidence
            for pair in test_pairs
            for evidence in (pair.evidence_a, pair.evidence_b)
        ]
    )
    test_vectors = np.asarray(
        [
            vector
            if vector is not None
            else hashed_text_vector(
                response.decode("utf-8", errors="replace"),
                codebook.prototypes.shape[1],
            )
            for pair in test_pairs
            for vector, response in (
                (pair.goal_target_vector_a, pair.response_a),
                (pair.goal_target_vector_b, pair.response_b),
            )
        ]
    )
    test_labels = codebook.assign(test_vectors)
    predicted = classifier_probabilities(predictor, test_evidence)
    prior_label = int(np.argmax(prior))
    return {
        "seed": seed,
        "selected_beta": selected_beta,
        "dev_beta_curve": dev_curves,
        "representation": {
            "occupied_train_codes": int(np.count_nonzero(counts)),
            "largest_train_code_fraction": float(np.max(counts) / len(train_labels)),
            "causal_code_accuracy": float(
                np.mean(np.argmax(predicted, axis=1) == test_labels)
            ),
            "prior_code_accuracy": float(np.mean(test_labels == prior_label)),
            "causal_code_nll_nats": float(
                np.mean(
                    -np.log(
                        np.maximum(
                            predicted[np.arange(len(test_labels)), test_labels], 1e-300
                        )
                    )
                )
            ),
        },
        "invariants": invariants,
        "effect_definition": (
            "per-pair raw symmetric pairing margin for the condition minus "
            "that same pair's goal_off raw margin"
        ),
        "raw_goal_off_residual": _summary(
            raw_test_values["goal_off"], seed=seed + 70_001, resamples=resamples
        ),
        "conditions": {
            name: {
                "adjusted_effect_vs_goal_off": _summary(
                    adjusted_test_effects[name],
                    seed=seed + index * 101,
                    resamples=resamples,
                ),
                "raw_pairing_margin": _summary(
                    raw_test_values[name],
                    seed=seed + index * 101 + 53,
                    resamples=resamples,
                ),
            }
            for index, name in enumerate(raw_test_values)
        },
        "gate": {
            "local_q0_traces_bit_identical": bool(
                invariants["exact_local_q0_trace_identity"]
            ),
            "goal_off_adjusted_effect_exactly_zero": bool(
                np.array_equal(
                    adjusted_test_effects["goal_off"], np.zeros(len(test_pairs))
                )
            ),
            "oracle_adjusted_positive": bool(
                np.mean(adjusted_test_effects["oracle_future"]) > 0.0
                and np.mean(adjusted_test_effects["oracle_future"] > 0.0) > 0.6
            ),
            "swapped_oracle_adjusted_reverses": bool(
                np.mean(adjusted_test_effects["swapped_oracle"]) < 0.0
            ),
            "causal_adjusted_positive": bool(
                np.mean(adjusted_test_effects["causal_pending_event"]) > 0.0
            ),
        },
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    decays = tuple(float(value) for value in args.state_decays)
    examples = load_training_examples(
        args.dataset_dir,
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=tuple(args.successor_gammas),
    )
    dev = load_crossover_pairs(
        args.dataset_dir,
        "dev",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=tuple(args.successor_gammas),
    )
    test = load_crossover_pairs(
        args.dataset_dir,
        "test",
        dimensions=args.feature_dimensions,
        decays=decays,
        successor_gammas=tuple(args.successor_gammas),
    )
    seeds = [
        _evaluate_seed(
            examples,
            dev,
            test,
            prototypes=args.prototypes,
            prototype_iterations=args.prototype_iterations,
            ridge_alpha=args.ridge_alpha,
            context_order=args.context_order,
            context_buckets=args.context_buckets,
            beta_values=tuple(args.beta_values),
            log_ratio_clip=args.log_ratio_clip,
            seed=seed,
            checkpoint=args.checkpoint,
            resamples=args.resamples,
            require_exact_native=args.require_exact_native,
        )
        for seed in args.seeds
    ]
    return {
        "experiment": "shared_goal_exact_crossover_ablation",
        "status": "complete",
        "claim_scope": (
            "natural-dialogue long-context compatibility; goal identity requires "
            "the counterbalanced trajectory gate"
        ),
        "base": "matched_local_byte_head"
        if args.checkpoint is None
        else "frozen_native_cortex",
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "feature_dimensions": args.feature_dimensions,
            "state_decays": list(decays),
            "successor_gammas": args.successor_gammas,
            "prototypes": args.prototypes,
            "prototype_iterations": args.prototype_iterations,
            "ridge_alpha": args.ridge_alpha,
            "context_order": args.context_order,
            "context_buckets": args.context_buckets,
            "beta_values": args.beta_values,
            "log_ratio_clip": args.log_ratio_clip,
            "seeds": args.seeds,
            "resamples": args.resamples,
            "require_exact_native": args.require_exact_native,
        },
        "counts": {
            "train_episodes": len(examples),
            "dev_pairs": len(dev),
            "test_pairs": len(test),
        },
        "seeds": seeds,
        "all_seed_gates": {
            key: all(bool(row["gate"][key]) for row in seeds)
            for key in seeds[0]["gate"]
        },
        "wall_seconds": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--feature-dimensions", type=int, default=96)
    parser.add_argument(
        "--state-decays", type=float, nargs="+", default=(0.5, 0.85, 0.97)
    )
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--prototype-iterations", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--context-order", type=int, default=4)
    parser.add_argument("--context-buckets", type=int, default=2048)
    parser.add_argument(
        "--beta-values", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--log-ratio-clip", type=float, default=2.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3, 11, 29))
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument(
        "--require-exact-native",
        action="store_true",
        help=(
            "fail when a frozen native base retains context-dependent state across "
            "the common suffix; normally this residual is reported and removed "
            "pairwise by the goal-off control"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.prototypes < 2:
        raise ValueError("prototypes must be at least two")
    if any(not 0.0 < value < 1.0 for value in args.state_decays):
        raise ValueError("state decays must lie in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in args.successor_gammas):
        raise ValueError("successor gammas must lie in (0, 1)")
    if any(value < 0.0 for value in args.beta_values):
        raise ValueError("beta values must be non-negative")
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")
    report = run_ablation(args)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
