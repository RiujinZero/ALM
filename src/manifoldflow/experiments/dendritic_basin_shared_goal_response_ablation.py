"""Screen persistent shared-goal targets on complete conversation episodes.

The core ablation operates on dense event vectors.  This file's chat-specific
adapter turns USER/ASSISTANT text into deterministic hashed lexical vectors;
another modality can supply vision, audio, or organ-state vectors without
changing the target discovery, prediction, or continuation-scoring logic.

All learning targets come from observed events.  Evaluation presents the true
assistant response beside responses from other conversations whose immediately
preceding USER turn is lexically similar.  This controls local plausibility and
asks whether whole-episode state adds information about the actual trajectory.

Target variants:

* ``immediate``: the actual next response (same-target plausibility control),
* ``theme``: the slowly accumulated past state (retrospective control),
* ``far_future``: a later observed turn (fixed-horizon proposal),
* ``interlocutor``: the next USER turn (cross-channel proposal), and
* ``successor``: several discounted summaries of the remaining trajectory.

For each target, spherical prototypes form a small self-supervised bottleneck.
Matched ridge classifiers predict its code from local-only evidence and from
local plus persistent state.  A goal-conditioned response head is compared to
a local response head; only their score difference is allowed to bias the
local score.  Oracle, causal, local-only/reset, shuffled, and swapped-goal
conditions separate representation utility from inference and extra capacity.

This is a mechanism screen, not a claim about autonomous generation quality.
The winning target must still be attached to a frozen production cortex and
pass matched-seed generation tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TURN_PATTERN = re.compile(r"(?m)^(USER|ASSISTANT):[ \t]*")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
DEFAULT_DECAYS = (0.5, 0.85, 0.97)
DEFAULT_SUCCESSOR_GAMMAS = (0.5, 0.8, 0.95)
DEFAULT_BIAS_WEIGHTS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Decision:
    conversation_id: int
    decision_index: int
    local: np.ndarray
    persistent: np.ndarray
    response: np.ndarray
    theme_target: np.ndarray
    far_future_target: np.ndarray
    interlocutor_target: np.ndarray
    successor_target: np.ndarray
    response_chars: int


@dataclass(frozen=True)
class DecisionBatch:
    conversation_ids: np.ndarray
    decision_indexes: np.ndarray
    local: np.ndarray
    persistent: np.ndarray
    response: np.ndarray
    theme_target: np.ndarray
    far_future_target: np.ndarray
    interlocutor_target: np.ndarray
    successor_target: np.ndarray
    response_chars: np.ndarray

    def __len__(self) -> int:
        return len(self.conversation_ids)


@dataclass(frozen=True)
class RidgeMap:
    weights: np.ndarray
    input_mean: np.ndarray
    output_mean: np.ndarray

    def predict(self, evidence: np.ndarray) -> np.ndarray:
        return (evidence - self.input_mean) @ self.weights + self.output_mean


@dataclass(frozen=True)
class PrototypeCodebook:
    mean: np.ndarray
    scale: np.ndarray
    prototypes: np.ndarray

    def transform(self, target: np.ndarray) -> np.ndarray:
        whitened = (target - self.mean) / self.scale
        return normalize_rows(whitened)

    def assign(self, target: np.ndarray) -> np.ndarray:
        return np.argmax(self.transform(target) @ self.prototypes.T, axis=1)


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value.copy() if norm == 0.0 else value / norm


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def parse_turns(text: str) -> list[Turn]:
    """Parse the role markers already present in one JSONL text record."""
    matches = list(TURN_PATTERN.finditer(text))
    turns: list[Turn] = []
    for index, match in enumerate(matches):
        end = len(text) if index + 1 == len(matches) else matches[index + 1].start()
        content = text[match.end() : end].strip()
        if content:
            turns.append(Turn(match.group(1), content))
    return turns


def hashed_text_vector(text: str, dimensions: int) -> np.ndarray:
    """Return a deterministic signed-hash unigram/bigram event vector."""
    if dimensions < 8:
        raise ValueError("feature dimensions must be at least eight")
    tokens = TOKEN_PATTERN.findall(text.casefold())
    features = [f"u:{token}" for token in tokens]
    features.extend(
        f"b:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    )
    if not features:
        features = ["empty"]
    vector = np.zeros(dimensions, dtype=np.float64)
    for feature in features:
        digest = hashlib.blake2s(
            feature.encode("utf-8"), digest_size=8, person=b"alm-goal"
        ).digest()
        code = int.from_bytes(digest, "little")
        index = code % dimensions
        vector[index] += 1.0 if (code >> 63) else -1.0
    vector = np.sign(vector) * np.log1p(np.abs(vector))
    return normalize(vector)


def split_name(text: str) -> str:
    """Conversation-level deterministic 70/15/15 split."""
    digest = hashlib.blake2s(
        text.encode("utf-8"), digest_size=8, person=b"alm-splt"
    ).digest()
    bucket = int.from_bytes(digest, "little") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _persistent_state(
    events: list[np.ndarray], decays: tuple[float, ...]
) -> np.ndarray:
    states = [np.zeros_like(events[0]) for _ in decays]
    for event in events:
        for index, decay in enumerate(decays):
            states[index] = decay * states[index] + (1.0 - decay) * event
    return np.concatenate([normalize(state) for state in states])


def _successor_signature(
    future: list[np.ndarray], gammas: tuple[float, ...]
) -> np.ndarray:
    summaries = []
    for gamma in gammas:
        weights = (1.0 - gamma) * gamma ** np.arange(len(future))
        summary = np.sum(np.asarray(future) * weights[:, None], axis=0)
        summaries.append(normalize(summary))
    return np.concatenate(summaries)


def decisions_from_conversation(
    conversation_id: int,
    turns: list[Turn],
    *,
    dimensions: int,
    decays: tuple[float, ...] = DEFAULT_DECAYS,
    successor_gammas: tuple[float, ...] = DEFAULT_SUCCESSOR_GAMMAS,
) -> list[Decision]:
    """Create causal response decisions with later observed targets."""
    vectors = [hashed_text_vector(turn.text, dimensions) for turn in turns]
    decisions: list[Decision] = []
    decision_index = 0
    for index, turn in enumerate(turns):
        if turn.role != "USER" or index < 2 or index + 3 >= len(turns):
            continue
        if turns[index + 1].role != "ASSISTANT":
            continue
        next_user = next(
            (
                later
                for later in range(index + 2, len(turns))
                if turns[later].role == "USER"
            ),
            None,
        )
        if next_user is None:
            continue
        response_index = index + 1
        future = vectors[response_index + 1 :]
        if len(future) < 2:
            continue
        persistent = _persistent_state(vectors[:index], decays)
        decisions.append(
            Decision(
                conversation_id=conversation_id,
                decision_index=decision_index,
                local=vectors[index],
                persistent=persistent,
                response=vectors[response_index],
                theme_target=persistent.copy(),
                far_future_target=vectors[-1],
                interlocutor_target=vectors[next_user],
                successor_target=_successor_signature(future, successor_gammas),
                response_chars=len(turns[response_index].text),
            )
        )
        decision_index += 1
    return decisions


def stack_decisions(rows: list[Decision]) -> DecisionBatch:
    if not rows:
        raise ValueError("cannot stack an empty decision split")
    return DecisionBatch(
        conversation_ids=np.asarray(
            [row.conversation_id for row in rows], dtype=np.int64
        ),
        decision_indexes=np.asarray(
            [row.decision_index for row in rows], dtype=np.int64
        ),
        local=np.asarray([row.local for row in rows]),
        persistent=np.asarray([row.persistent for row in rows]),
        response=np.asarray([row.response for row in rows]),
        theme_target=np.asarray([row.theme_target for row in rows]),
        far_future_target=np.asarray([row.far_future_target for row in rows]),
        interlocutor_target=np.asarray([row.interlocutor_target for row in rows]),
        successor_target=np.asarray([row.successor_target for row in rows]),
        response_chars=np.asarray([row.response_chars for row in rows]),
    )


def load_decision_splits(
    sources: tuple[Path, ...],
    *,
    dimensions: int,
    max_conversations: int,
    decays: tuple[float, ...],
    successor_gammas: tuple[float, ...],
) -> tuple[dict[str, DecisionBatch], dict[str, int]]:
    grouped: dict[str, list[Decision]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    conversations = {name: 0 for name in grouped}
    seen = 0
    for source in sources:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if seen >= max_conversations:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(record.get("text") or "").strip()
                if not text:
                    continue
                turns = parse_turns(text)
                rows = decisions_from_conversation(
                    seen,
                    turns,
                    dimensions=dimensions,
                    decays=decays,
                    successor_gammas=successor_gammas,
                )
                seen += 1
                if not rows:
                    continue
                name = split_name(text)
                grouped[name].extend(rows)
                conversations[name] += 1
            if seen >= max_conversations:
                break
    missing = [name for name, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"decision splits have no examples: {', '.join(missing)}")
    return (
        {name: stack_decisions(rows) for name, rows in grouped.items()},
        conversations,
    )


def fit_ridge(evidence: np.ndarray, target: np.ndarray, alpha: float) -> RidgeMap:
    if alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    gram.flat[:: len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, centered_x.T @ centered_y)
    return RidgeMap(weights, x_mean, y_mean)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_codebook(
    target: np.ndarray,
    *,
    prototypes: int,
    iterations: int,
    seed: int,
) -> PrototypeCodebook:
    if prototypes < 2 or prototypes > len(target):
        raise ValueError("prototype count must be in [2, number of targets]")
    mean = target.mean(axis=0)
    scale = target.std(axis=0)
    scale[scale < 1e-6] = 1.0
    whitened = normalize_rows((target - mean) / scale)
    rng = np.random.default_rng(seed)
    centers = [whitened[int(rng.integers(len(whitened)))]]
    for _ in range(1, prototypes):
        similarity = whitened @ np.asarray(centers).T
        distance = np.maximum(0.0, 1.0 - similarity.max(axis=1))
        total = float(distance.sum())
        index = (
            int(rng.integers(len(whitened)))
            if total == 0.0
            else int(rng.choice(len(whitened), p=distance / total))
        )
        centers.append(whitened[index])
    matrix = np.asarray(centers)
    for _ in range(iterations):
        assignment = np.argmax(whitened @ matrix.T, axis=1)
        updated = np.zeros_like(matrix)
        for code in range(prototypes):
            members = whitened[assignment == code]
            if len(members):
                updated[code] = normalize(members.mean(axis=0))
            else:
                fit = (whitened @ matrix.T).max(axis=1)
                updated[code] = whitened[int(np.argmin(fit))]
        if np.allclose(updated, matrix, atol=1e-7):
            matrix = updated
            break
        matrix = updated
    return PrototypeCodebook(mean, scale, matrix)


def classifier_probabilities(model: RidgeMap, evidence: np.ndarray) -> np.ndarray:
    return softmax(model.predict(evidence))


def hard_negative_candidates(batch: DecisionBatch, candidate_count: int) -> np.ndarray:
    if candidate_count < 2 or candidate_count > len(batch):
        raise ValueError("candidate_count must be in [2, number of examples]")
    similarity = normalize_rows(batch.local) @ normalize_rows(batch.local).T
    length = np.log1p(batch.response_chars.astype(np.float64))
    length_gap = np.abs(length[:, None] - length[None, :])
    hardness = similarity - 0.05 * length_gap
    same_conversation = (
        batch.conversation_ids[:, None] == batch.conversation_ids[None, :]
    )
    hardness[same_conversation] = -np.inf
    choices = np.empty((len(batch), candidate_count), dtype=np.int64)
    for row in range(len(batch)):
        order = np.argsort(hardness[row])[::-1]
        negatives = order[np.isfinite(hardness[row, order])][: candidate_count - 1]
        if len(negatives) != candidate_count - 1:
            raise ValueError("not enough cross-conversation hard negatives")
        choices[row, 0] = row
        choices[row, 1:] = negatives
    return choices


def cosine_score_matrix(prediction: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return normalize_rows(prediction) @ normalize_rows(candidates).T


def selection_metrics(
    scores: np.ndarray,
    choices: np.ndarray,
    *,
    local_scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = np.take_along_axis(scores, choices, axis=1)
    positive = selected[:, 0]
    negative = selected[:, 1:]
    rank = 1 + np.sum(negative > positive[:, None], axis=1)
    top1 = rank == 1
    margins = positive - negative.max(axis=1)
    report: dict[str, float | int] = {
        "examples": int(len(scores)),
        "top1_accuracy": float(np.mean(top1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "mean_positive_margin": float(np.mean(margins)),
        "median_positive_margin": float(np.median(margins)),
    }
    if local_scores is not None:
        local_selected = np.take_along_axis(local_scores, choices, axis=1)
        local_win = local_selected[:, 0] > local_selected[:, 1:].max(axis=1)
        locally_wrong = ~local_win
        report["locally_wrong_examples"] = int(np.sum(locally_wrong))
        report["rescue_fraction"] = (
            float(np.mean(top1[locally_wrong])) if np.any(locally_wrong) else 0.0
        )
        report["regression_fraction"] = (
            float(np.mean(~top1[local_win])) if np.any(local_win) else 0.0
        )
    return report


def _target(batch: DecisionBatch, name: str) -> np.ndarray:
    if name == "immediate":
        return batch.response
    return getattr(batch, f"{name}_target")


def _full_evidence(batch: DecisionBatch) -> np.ndarray:
    return np.concatenate((batch.local, batch.persistent), axis=1)


def _local_evidence(batch: DecisionBatch) -> np.ndarray:
    return np.concatenate((batch.local, np.zeros_like(batch.persistent)), axis=1)


def _code_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    observed = probability[np.arange(len(labels)), labels]
    return {
        "accuracy": float(np.mean(np.argmax(probability, axis=1) == labels)),
        "nll_nats": float(np.mean(-np.log(np.maximum(observed, 1e-12)))),
    }


def _goal_score_matrix(
    response_head: RidgeMap,
    batch: DecisionBatch,
    code_probability: np.ndarray,
) -> np.ndarray:
    prediction = response_head.predict(
        np.concatenate((batch.local, code_probability), axis=1)
    )
    return cosine_score_matrix(prediction, batch.response)


def _choose_weight(
    local_scores: np.ndarray,
    goal_scores: np.ndarray,
    choices: np.ndarray,
    weights: tuple[float, ...],
) -> tuple[float, dict[str, float | int]]:
    best: tuple[float, float, float, dict[str, float | int]] | None = None
    for weight in weights:
        combined = local_scores + weight * (goal_scores - local_scores)
        metrics = selection_metrics(combined, choices, local_scores=local_scores)
        key = (
            float(metrics["top1_accuracy"]),
            float(metrics["mean_reciprocal_rank"]),
            -weight,
        )
        if best is None or key > best[:3]:
            best = (*key, metrics)
    assert best is not None
    return -best[2], best[3]


def _evaluate_condition(
    local_scores: np.ndarray,
    goal_scores: np.ndarray,
    choices: np.ndarray,
    weight: float,
) -> dict[str, float | int]:
    combined = local_scores + weight * (goal_scores - local_scores)
    return selection_metrics(combined, choices, local_scores=local_scores)


def run_target_ablation(
    name: str,
    splits: dict[str, DecisionBatch],
    *,
    prototype_count: int,
    prototype_iterations: int,
    ridge_alpha: float,
    candidates: int,
    bias_weights: tuple[float, ...],
    seed: int,
    local_heads: dict[str, RidgeMap],
) -> dict[str, object]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    codebook = fit_codebook(
        _target(train, name),
        prototypes=prototype_count,
        iterations=prototype_iterations,
        seed=seed,
    )
    labels = {
        split: codebook.assign(_target(batch, name)) for split, batch in splits.items()
    }
    one_hot = np.eye(prototype_count, dtype=np.float64)[labels["train"]]
    full_classifier = fit_ridge(_full_evidence(train), one_hot, ridge_alpha)
    local_classifier = fit_ridge(_local_evidence(train), one_hot, ridge_alpha)
    response_head = fit_ridge(
        np.concatenate((train.local, one_hot), axis=1),
        train.response,
        ridge_alpha,
    )

    probabilities: dict[str, dict[str, np.ndarray]] = {}
    for split, batch in splits.items():
        full = classifier_probabilities(full_classifier, _full_evidence(batch))
        local = classifier_probabilities(local_classifier, _local_evidence(batch))
        oracle = np.eye(prototype_count, dtype=np.float64)[labels[split]]
        probabilities[split] = {"full": full, "local": local, "oracle": oracle}

    validation_choices = hard_negative_candidates(validation, candidates)
    test_choices = hard_negative_candidates(test, candidates)
    local_validation_scores = cosine_score_matrix(
        local_heads["validation"].predict(validation.local), validation.response
    )
    local_test_scores = cosine_score_matrix(
        local_heads["test"].predict(test.local), test.response
    )
    validation_goal_scores = _goal_score_matrix(
        response_head, validation, probabilities["validation"]["full"]
    )
    weight, validation_metrics = _choose_weight(
        local_validation_scores,
        validation_goal_scores,
        validation_choices,
        bias_weights,
    )

    rng = np.random.default_rng(seed + 10_007)
    permutation = rng.permutation(len(test))
    shuffled = probabilities["test"]["full"][permutation]
    swapped = probabilities["test"]["full"][test_choices[:, 1]]
    condition_scores = {
        "causal_full": _goal_score_matrix(
            response_head, test, probabilities["test"]["full"]
        ),
        "oracle_future": _goal_score_matrix(
            response_head, test, probabilities["test"]["oracle"]
        ),
        "local_only_code": _goal_score_matrix(
            response_head, test, probabilities["test"]["local"]
        ),
        "shuffled_episode_code": _goal_score_matrix(response_head, test, shuffled),
        "swapped_hard_negative_code": _goal_score_matrix(response_head, test, swapped),
    }
    counts = np.bincount(labels["test"], minlength=prototype_count)
    representation = {
        "local": _code_metrics(probabilities["test"]["local"], labels["test"]),
        "full": _code_metrics(probabilities["test"]["full"], labels["test"]),
        "occupied_codes": int(np.count_nonzero(counts)),
        "largest_code_fraction": float(np.max(counts) / len(test)),
        "test_code_counts": counts.tolist(),
    }
    conditions = {
        condition: _evaluate_condition(local_test_scores, scores, test_choices, weight)
        for condition, scores in condition_scores.items()
    }
    direct_add = selection_metrics(
        local_test_scores + weight * condition_scores["causal_full"],
        test_choices,
        local_scores=local_test_scores,
    )
    return {
        "selected_bias_weight": weight,
        "validation_at_selected_weight": validation_metrics,
        "representation": representation,
        "conditions": conditions,
        "direct_add_without_ratio": direct_add,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    splits, conversations = load_decision_splits(
        tuple(args.sources),
        dimensions=args.feature_dimensions,
        max_conversations=args.max_conversations,
        decays=tuple(args.state_decays),
        successor_gammas=tuple(args.successor_gammas),
    )
    train = splits["train"]
    local_response_head = fit_ridge(train.local, train.response, args.ridge_alpha)
    local_heads = {split: local_response_head for split in ("validation", "test")}
    baseline: dict[str, object] = {}
    for split in ("validation", "test"):
        batch = splits[split]
        choices = hard_negative_candidates(batch, args.candidates)
        scores = cosine_score_matrix(
            local_response_head.predict(batch.local), batch.response
        )
        baseline[split] = selection_metrics(scores, choices)

    targets = (
        "immediate",
        "theme",
        "far_future",
        "interlocutor",
        "successor",
    )
    results = {
        name: run_target_ablation(
            name,
            splits,
            prototype_count=args.prototypes,
            prototype_iterations=args.prototype_iterations,
            ridge_alpha=args.ridge_alpha,
            candidates=args.candidates,
            bias_weights=tuple(args.bias_weights),
            seed=args.seed + index * 101,
            local_heads=local_heads,
        )
        for index, name in enumerate(targets)
    }
    return {
        "experiment": "shared_goal_response_selection_ablation",
        "status": "complete",
        "mechanism_scope": (
            "dense event vectors; hashed text is the chat adapter, not part of "
            "the shared-goal mechanism"
        ),
        "causal_contract": (
            "context excludes the candidate response and all future targets; "
            "splits are conversation-disjoint; only observed later events train "
            "future codes"
        ),
        "config": {
            "sources": [str(path) for path in args.sources],
            "max_conversations": args.max_conversations,
            "feature_dimensions": args.feature_dimensions,
            "state_decays": args.state_decays,
            "successor_gammas": args.successor_gammas,
            "prototypes": args.prototypes,
            "prototype_iterations": args.prototype_iterations,
            "ridge_alpha": args.ridge_alpha,
            "candidates": args.candidates,
            "bias_weights": args.bias_weights,
            "seed": args.seed,
        },
        "conversation_counts": conversations,
        "decision_counts": {name: len(batch) for name, batch in splits.items()},
        "local_plausibility_baseline": baseline,
        "targets": results,
        "wall_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-conversations", type=int, default=2_260)
    parser.add_argument("--feature-dimensions", type=int, default=128)
    parser.add_argument("--state-decays", type=float, nargs="+", default=DEFAULT_DECAYS)
    parser.add_argument(
        "--successor-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_SUCCESSOR_GAMMAS,
    )
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--prototype-iterations", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--bias-weights", type=float, nargs="+", default=DEFAULT_BIAS_WEIGHTS
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_conversations < 10:
        parser.error("--max-conversations must be at least 10")
    if any(not 0.0 < decay < 1.0 for decay in args.state_decays):
        parser.error("--state-decays must lie in (0, 1)")
    if any(not 0.0 < gamma < 1.0 for gamma in args.successor_gammas):
        parser.error("--successor-gammas must lie in (0, 1)")
    if any(weight < 0.0 for weight in args.bias_weights):
        parser.error("--bias-weights must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = run_ablation(args)
    payload = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
