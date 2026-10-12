"""Parameter-matched causal transformer control for ALM echolocation."""

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

from manifoldflow.dendritic_basin.tasks.echolocation import (
    dead_reckoning_prediction,
    generate_episodes,
    shuffled_ranging_episodes,
    stack_episode_vectors,
)

ALM_GROWN_BODY_BYTES = 99_344


class CausalSensoryTransformer(nn.Module):
    """Small decoder-style transformer over simultaneous raw sensory vectors."""

    def __init__(
        self,
        *,
        steps: int = 40,
        channels: int = 5,
        width: int = 24,
        heads: int = 4,
        feedforward: int = 45,
        layers: int = 5,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(channels, width)
        self.position = nn.Embedding(steps, width)
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
        self.position_readout = nn.Linear(width, 1)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        steps = vectors.shape[1]
        positions = torch.arange(steps, device=vectors.device)
        hidden = self.input_projection(vectors) + self.position(positions)[None]
        causal_mask = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=vectors.device),
            diagonal=1,
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.position_readout(self.normalization(hidden)).squeeze(-1)


def parameter_bytes(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--candidate-episodes", type=int, default=120)
    parser.add_argument("--test-episodes", type=int, default=240)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hidden-steps", type=int, default=24)
    parser.add_argument("--velocity-wander", type=float, default=0.03)
    parser.add_argument(
        "--velocity-eval-wanders",
        type=float,
        nargs="*",
        default=[],
        help="optional held-out velocity-wander values evaluated after training",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--match-body-bytes", type=int, default=ALM_GROWN_BODY_BYTES)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((truth - prediction) ** 2))


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.sum((truth - prediction) ** 2)
    total = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - residual / max(float(total), 1e-8))


@torch.no_grad()
def _predict(model: nn.Module, vectors: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(vectors)).cpu().numpy()


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 701)
    base = 1_000_003 * (seed + 1)
    kwargs = dict(
        steps=args.steps,
        hidden_steps=args.hidden_steps,
        velocity_wander=args.velocity_wander,
    )
    train = generate_episodes(args.train_episodes, seed=base + 11, **kwargs)
    candidate = generate_episodes(args.candidate_episodes, seed=base + 23, **kwargs)
    test = generate_episodes(args.test_episodes, seed=base + 37, **kwargs)
    train_x = stack_episode_vectors(train, visible=True)
    train_y = np.stack([episode.positions for episode in train])
    candidate_x = stack_episode_vectors(candidate, visible=True)
    candidate_y = np.stack([episode.positions for episode in candidate])

    model = CausalSensoryTransformer(steps=args.steps)
    resident_bytes = parameter_bytes(model)
    relative_mismatch = (
        abs(resident_bytes - args.match_body_bytes) / args.match_body_bytes
    )
    if relative_mismatch > 0.01:
        raise ValueError(
            f"transformer body {resident_bytes:,} B is not within 1% of "
            f"matched ALM body {args.match_body_bytes:,} B"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_x_tensor = torch.from_numpy(train_x)
    train_y_tensor = torch.from_numpy(train_y)
    best_candidate = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(args.train_episodes)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = torch.from_numpy(order[start : start + args.batch_size])
            prediction = model(train_x_tensor[indices])
            loss = torch.mean((prediction - train_y_tensor[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        candidate_prediction = _predict(model, candidate_x)
        candidate_mse = _mse(candidate_y, candidate_prediction)
        history.append(
            {
                "epoch": epoch,
                "training_mse": total_loss / args.train_episodes,
                "candidate_mse": candidate_mse,
            }
        )
        if candidate_mse < best_candidate - 1e-8:
            best_candidate = candidate_mse
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

    truth = np.asarray([episode.final_position for episode in test])

    def final_prediction(
        episodes: list[Any], *, visible: bool, audio_mode: str = "real"
    ) -> np.ndarray:
        vectors = stack_episode_vectors(
            episodes, visible=visible, audio_mode=audio_mode
        )
        return _predict(model, vectors)[:, -1]

    multimodal = final_prediction(test, visible=True)
    dark = final_prediction(test, visible=False)
    shuffled_episodes = shuffled_ranging_episodes(test, seed=base + 41)
    shuffled = final_prediction(shuffled_episodes, visible=False)
    silent = final_prediction(test, visible=False, audio_mode="silent")
    dead = np.asarray([dead_reckoning_prediction(episode) for episode in test])
    dark_mse = _mse(truth, dark)
    shuffled_mse = _mse(truth, shuffled)
    silent_mse = _mse(truth, silent)
    dead_mse = _mse(truth, dead)
    velocity_sweep = {}
    for wander in args.velocity_eval_wanders:
        shifted = generate_episodes(
            args.test_episodes,
            seed=base + 37,
            steps=args.steps,
            hidden_steps=args.hidden_steps,
            velocity_wander=wander,
        )
        shifted_truth = np.asarray(
            [episode.final_position for episode in shifted]
        )
        shifted_visible = final_prediction(shifted, visible=True)
        shifted_dark = final_prediction(shifted, visible=False)
        velocity_sweep[str(wander)] = {
            "visible_position_r2": _r2(shifted_truth, shifted_visible),
            "visible_position_mse": _mse(shifted_truth, shifted_visible),
            "dark_position_mse": _mse(shifted_truth, shifted_dark),
        }
    return {
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "resident_parameter_bytes": resident_bytes,
        "relative_size_mismatch": relative_mismatch,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_candidate_sequence_mse": best_candidate,
        "held_out_multimodal_position_r2": _r2(truth, multimodal),
        "dark_audio_position_mse": dark_mse,
        "dead_reckoning_mse": dead_mse,
        "shuffled_audio_mse": shuffled_mse,
        "silent_audio_mse": silent_mse,
        "audio_vs_dead_reckoning_improvement": dead_mse / max(dark_mse, 1e-8),
        "real_vs_shuffled_audio_improvement": shuffled_mse / max(dark_mse, 1e-8),
        "causal_audio_pass": dark_mse < dead_mse and dark_mse < shuffled_mse,
        "training_seconds": train_seconds,
        "training_frames_per_second": (
            args.train_episodes * args.steps * len(history) / train_seconds
        ),
        "velocity_wander_sweep": velocity_sweep,
        "history": history,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "held_out_multimodal_position_r2",
        "dark_audio_position_mse",
        "dead_reckoning_mse",
        "shuffled_audio_mse",
        "silent_audio_mse",
        "audio_vs_dead_reckoning_improvement",
        "real_vs_shuffled_audio_improvement",
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
    result["causal_audio_pass_count"] = sum(row["causal_audio_pass"] for row in rows)
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for offset in range(args.seeds):
        seed = args.seed_offset + offset
        print(
            f"matched transformer echolocation seed {seed}",
            file=sys.stderr,
            flush=True,
        )
        rows.append(run_seed(args, seed))
    report = {
        "experiment": "parameter_matched_transformer_echolocation_control",
        "matched_alm_experiment": "alm_64mb_1c_three_organ_native_vector_echolocation",
        "matching_contract": (
            "float32 resident learned parameters are matched within 1% to ALM's "
            "99,344-byte grown anatomy; optimizer state and training transients are "
            "excluded for both systems"
        ),
        "input_contract": (
            "the same five simultaneous raw float32 sensory channels and identical "
            "episode seeds; no distance, velocity, cadence scalar, audio encoder, "
            "or dark-training augmentation"
        ),
        "training_contract": (
            "causal next-state transformer trained by AdamW/backprop on visible "
            "training episodes; visible candidate episodes select the checkpoint"
        ),
        "architecture": {
            "width": 24,
            "heads": 4,
            "feedforward": 45,
            "layers": 5,
            "dropout": 0.0,
            "parameter_count": 24_802,
            "resident_parameter_bytes": 99_208,
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
