"""Size-matched causal-transformer control for the moving-point ring smoke.

The current ring position is insufficient to predict the next position because
each episode moves either clockwise or counter-clockwise.  Two visual frames
identify the signed motion state.  This control tests whether a small
backprop-trained transformer uses that temporal evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from manifoldflow.dendritic_basin.tasks.moving_point import (
    RESET_CHANNEL,
    generate_position_stream,
    vocab_size,
)

ALM_GROWN_BODY_BYTES = 99_344


class RingCausalTransformer(nn.Module):
    """Decoder-style transformer over raw one-hot ring observations."""

    def __init__(
        self,
        *,
        ring_size: int = 8,
        max_steps: int = 40,
        width: int = 24,
        heads: int = 4,
        feedforward: int = 44,
        layers: int = 5,
    ) -> None:
        super().__init__()
        self.ring_size = ring_size
        self.input_projection = nn.Linear(vocab_size(ring_size), width)
        self.position = nn.Embedding(max_steps, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(width)
        self.next_position = nn.Linear(width, ring_size)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        steps = observations.shape[1]
        positions = torch.arange(steps, device=observations.device)
        hidden = self.input_projection(observations) + self.position(positions)[None]
        causal_mask = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=observations.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.normalization(hidden)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.next_position(self.encode(observations))


def parameter_bytes(model: nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def episode_arrays(
    n_episodes: int,
    *,
    ring_size: int,
    steps_per_episode: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw one-hot events and un-offset ring positions by episode."""
    stream = generate_position_stream(
        n_episodes,
        ring_size,
        steps_per_episode,
        np.random.default_rng(seed),
    )
    tokens = np.asarray(stream, dtype=np.int64).reshape(
        n_episodes, steps_per_episode + 1
    )
    observations = np.eye(vocab_size(ring_size), dtype=np.float32)[tokens]
    positions = tokens[:, 1:] - 1
    return observations, positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--ring-size", type=int, default=8)
    parser.add_argument("--steps-per-episode", type=int, default=6)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--candidate-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--match-body-bytes", type=int, default=ALM_GROWN_BODY_BYTES)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _loss(model: nn.Module, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # At event index i, predict the next ring position.  Reset -> random start
    # is excluded because it is intentionally unpredictable.
    logits = model(x)[:, 1:-1]
    targets = positions[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def _predict(model: nn.Module, observations: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(observations)).argmax(dim=-1).cpu().numpy()


@torch.no_grad()
def _hidden(model: RingCausalTransformer, observations: np.ndarray) -> np.ndarray:
    model.eval()
    return model.encode(torch.from_numpy(observations)).cpu().numpy()


def _direction_targets(positions: np.ndarray, ring_size: int) -> np.ndarray:
    step = (positions[:, 1:-1] - positions[:, :-2]) % ring_size
    return np.where(step == 1, 1.0, -1.0)


def _short_cues(
    positions: np.ndarray, *, ring_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build single-frame, two-frame, and reversed-motion evaluation cues."""
    current = positions[:, 1:-1].reshape(-1)
    previous = positions[:, :-2].reshape(-1)
    target = positions[:, 2:].reshape(-1)
    step = (current - previous) % ring_size
    direction = np.where(step == 1, 1, -1)

    single_tokens = np.column_stack(
        [np.full_like(current, RESET_CHANNEL), current + 1]
    )
    pair_tokens = np.column_stack(
        [np.full_like(current, RESET_CHANNEL), previous + 1, current + 1]
    )
    opposite_previous = (current + direction) % ring_size
    opposite_target = (current - direction) % ring_size
    opposite_tokens = np.column_stack(
        [
            np.full_like(current, RESET_CHANNEL),
            opposite_previous + 1,
            current + 1,
        ]
    )
    eye = np.eye(vocab_size(ring_size), dtype=np.float32)
    return (
        eye[single_tokens],
        eye[pair_tokens],
        target,
        (eye[opposite_tokens], opposite_target),
    )


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 901)
    base = 1_000_003 * (seed + 1)
    data_kwargs = dict(
        ring_size=args.ring_size, steps_per_episode=args.steps_per_episode
    )
    train_x, train_positions = episode_arrays(
        args.train_episodes, seed=base + 11, **data_kwargs
    )
    candidate_x, candidate_positions = episode_arrays(
        args.candidate_episodes, seed=base + 23, **data_kwargs
    )
    test_x, test_positions = episode_arrays(
        args.test_episodes, seed=base + 37, **data_kwargs
    )

    model = RingCausalTransformer(ring_size=args.ring_size)
    resident_bytes = parameter_bytes(model)
    mismatch = abs(resident_bytes - args.match_body_bytes) / args.match_body_bytes
    if mismatch > 0.01:
        raise ValueError(
            f"transformer body {resident_bytes:,} B is not within 1% of "
            f"matched ALM body {args.match_body_bytes:,} B"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_positions)
    candidate_x_t = torch.from_numpy(candidate_x)
    candidate_y_t = torch.from_numpy(candidate_positions)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(args.train_episodes)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = torch.from_numpy(order[start : start + args.batch_size])
            loss = _loss(model, train_x_t[indices], train_y_t[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            candidate_loss = float(_loss(model, candidate_x_t, candidate_y_t))
        history.append(
            {
                "epoch": epoch,
                "training_cross_entropy": total / args.train_episodes,
                "candidate_cross_entropy": candidate_loss,
            }
        )
        if candidate_loss < best_loss - 1e-8:
            best_loss = candidate_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - started

    predictions = _predict(model, test_x)
    first_frame_accuracy = float(
        np.mean(predictions[:, 1] == test_positions[:, 1])
    )
    temporal_accuracy = float(
        np.mean(predictions[:, 2:-1] == test_positions[:, 2:])
    )
    all_transition_accuracy = float(
        np.mean(predictions[:, 1:-1] == test_positions[:, 1:])
    )

    single, pair, pair_target, opposite = _short_cues(
        test_positions, ring_size=args.ring_size
    )
    opposite_x, opposite_target = opposite
    single_accuracy = float(np.mean(_predict(model, single)[:, 1] == pair_target))
    pair_accuracy = float(np.mean(_predict(model, pair)[:, 2] == pair_target))
    opposite_accuracy = float(
        np.mean(_predict(model, opposite_x)[:, 2] == opposite_target)
    )

    # Frozen diagnostic only: the probe does not participate in transformer
    # training.  It asks whether temporal hidden activity makes signed motion
    # linearly readable on episodes not used to fit the transformer weights.
    candidate_hidden = _hidden(model, candidate_x)[:, 2:-1].reshape(
        -1, model.normalization.normalized_shape[0]
    )
    candidate_direction = _direction_targets(
        candidate_positions, args.ring_size
    ).reshape(-1)
    probe_x = np.column_stack(
        [candidate_hidden, np.ones(len(candidate_hidden), dtype=np.float32)]
    )
    probe = np.linalg.lstsq(probe_x, candidate_direction, rcond=None)[0]
    test_hidden = _hidden(model, test_x)[:, 2:-1].reshape(
        -1, model.normalization.normalized_shape[0]
    )
    test_direction = _direction_targets(test_positions, args.ring_size).reshape(-1)
    decoded_direction = (
        np.column_stack(
            [test_hidden, np.ones(len(test_hidden), dtype=np.float32)]
        )
        @ probe
    )
    direction_probe_accuracy = float(
        np.mean(np.sign(decoded_direction) == test_direction)
    )
    direction_probe_r2 = float(
        1.0
        - np.sum((test_direction - decoded_direction) ** 2)
        / np.sum((test_direction - test_direction.mean()) ** 2)
    )
    candidate_current = candidate_positions[:, 1:-1].reshape(-1)
    test_current = test_positions[:, 1:-1].reshape(-1)
    conditioned_decoded = np.zeros_like(test_direction)
    for position in range(args.ring_size):
        candidate_mask = candidate_current == position
        test_mask = test_current == position
        local_probe = np.linalg.lstsq(
            probe_x[candidate_mask], candidate_direction[candidate_mask], rcond=None
        )[0]
        conditioned_decoded[test_mask] = (
            np.column_stack(
                [
                    test_hidden[test_mask],
                    np.ones(np.sum(test_mask), dtype=np.float32),
                ]
            )
            @ local_probe
        )
    conditioned_probe_accuracy = float(
        np.mean(np.sign(conditioned_decoded) == test_direction)
    )
    return {
        "seed": seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "resident_parameter_bytes": resident_bytes,
        "relative_size_mismatch": mismatch,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_candidate_cross_entropy": best_loss,
        "first_transition_one_frame_accuracy": first_frame_accuracy,
        "later_transition_temporal_accuracy": temporal_accuracy,
        "all_transition_accuracy": all_transition_accuracy,
        "isolated_single_frame_accuracy": single_accuracy,
        "isolated_two_frame_accuracy": pair_accuracy,
        "opposite_direction_counterfactual_accuracy": opposite_accuracy,
        "frozen_hidden_direction_probe_accuracy": direction_probe_accuracy,
        "frozen_hidden_direction_probe_r2": direction_probe_r2,
        "position_conditioned_hidden_direction_probe_accuracy": (
            conditioned_probe_accuracy
        ),
        "training_seconds": train_seconds,
        "training_episodes_per_second": (
            args.train_episodes * len(history) / train_seconds
        ),
        "history": history,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "first_transition_one_frame_accuracy",
        "later_transition_temporal_accuracy",
        "all_transition_accuracy",
        "isolated_single_frame_accuracy",
        "isolated_two_frame_accuracy",
        "opposite_direction_counterfactual_accuracy",
        "frozen_hidden_direction_probe_accuracy",
        "frozen_hidden_direction_probe_r2",
        "position_conditioned_hidden_direction_probe_accuracy",
        "training_seconds",
    )
    result = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows])
        result[metric] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(
            f"matched transformer moving-point seed {seed}",
            file=sys.stderr,
            flush=True,
        )
        rows.append(run_seed(args, seed))
    report = {
        "experiment": "parameter_matched_transformer_moving_point_control",
        "matched_alm_experiment": "alm_64mb_1c_three_organ_native_vector_echolocation",
        "historical_ring_alm_reference": {
            "train_episodes": 400,
            "eval_episodes": 400,
            "direction_accuracy_at_anchor": 0.765,
            "scope": "older minimal ALM relation-discovery ring smoke",
        },
        "matching_contract": (
            "float32 resident learned parameters are matched within 1% to the "
            "99,344-byte grown ALM echolocation anatomy"
        ),
        "task_contract": (
            "8-position ring; each episode selects clockwise or counter-clockwise "
            "motion; the current raw one-hot visual frame is symmetric and two "
            "frames are sufficient to infer direction"
        ),
        "training_contract": (
            "causal transformer trained with AdamW/backprop on next-position loss; "
            "candidate episodes select the checkpoint; test episodes are held out"
        ),
        "architecture": {
            "width": 24,
            "heads": 4,
            "feedforward": 44,
            "layers": 5,
            "dropout": 0.0,
            "parameter_count": 24_828,
            "resident_parameter_bytes": 99_312,
        },
        "args": vars(args) | {"report": str(args.report) if args.report else None},
        "aggregate": aggregate(rows),
        "seeds": rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
