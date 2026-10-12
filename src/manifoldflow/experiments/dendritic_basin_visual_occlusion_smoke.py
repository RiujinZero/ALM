"""Phase 9 (design doc section 54: "Visual world -- Moving objects,
persistence, occlusion, identity."): re-validates Phase 3's core finding
(a plain decaying trace cannot survive a long enough gap; only
persistent, non-decaying state can) on dense continuous vision-flavored
data instead of discrete tokens.

At the instant an object reappears from occlusion, compare two ways of
recovering its identity (hue value):

1. A plain `MultiTimescaleTraces` trace of the hue channel -- exactly
   Phase 7's mechanism, with no special handling for occlusion.
2. A `PersistentContinuousRegister` that holds the last-seen hue value
   exactly, unaffected by how long the occlusion lasts.

Prediction error should grow with occlusion length for (1) and stay flat
for (2) -- the continuous-vector version of Phase 3's original binding
result.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.persistent_continuous_register import (
    PersistentContinuousRegister,
)
from manifoldflow.dendritic_basin.tasks.visual_occlusion import (
    HUE_CHANNEL,
    N_CHANNELS,
    VISIBLE_CHANNEL,
    generate_occlusion_stream,
)
from manifoldflow.dendritic_basin.temporal_trace import MultiTimescaleTraces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--visible-steps-before", type=int, default=6)
    parser.add_argument("--occlusion-min", type=int, default=1)
    parser.add_argument("--occlusion-max", type=int, default=40)
    parser.add_argument("--visible-steps-after", type=int, default=6)
    parser.add_argument("--position-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    hue_values = (-1.0, 0.0, 1.0)
    timescales = (1, 2, 4, 8, 16, 32, 64)

    (
        stream,
        episode_start_indices,
        reappearance_indices,
        true_hues,
    ) = generate_occlusion_stream(
        n_episodes=args.n_episodes,
        visible_steps_before=args.visible_steps_before,
        occlusion_min=args.occlusion_min,
        occlusion_max=args.occlusion_max,
        visible_steps_after=args.visible_steps_after,
        position_step=args.position_step,
        hue_values=hue_values,
        rng=rng,
    )

    traces = MultiTimescaleTraces(N_CHANNELS, timescales)
    register = PersistentContinuousRegister(N_CHANNELS)
    episode_start_set = set(int(i) for i in episode_start_indices)

    trace_snapshots = np.zeros((len(stream), len(timescales)))
    register_snapshots = np.zeros(len(stream))
    occlusion_lengths = np.zeros(len(stream), dtype=np.int64)
    current_occlusion_len = 0

    for t in range(len(stream)):
        if t in episode_start_set:
            # A new, unrelated object begins here -- state from the
            # previous episode must not leak into this one, exactly why
            # persistent_assembly.py's `reset` parameter exists (Phase 3).
            traces.reset()
            register.reset()
        x = stream[t]
        visible = bool(x[VISIBLE_CHANNEL] > 0.5)
        traces.update(x)
        register.update(x, visible)
        trace_snapshots[t] = traces.q[:, HUE_CHANNEL]
        register_snapshots[t] = register.snapshot()[HUE_CHANNEL]
        current_occlusion_len = 0 if visible else current_occlusion_len + 1
        occlusion_lengths[t] = current_occlusion_len

    # Snapshot the step *before* reappearance -- state built up only from
    # what happened before the object came back, not from the reappearing
    # row's own (already-revealed) value. Testing at the reappearance row
    # itself would trivially "recover" the hue from the very observation
    # being predicted.
    just_before_reappearance = reappearance_indices - 1
    trace_hue_at_reappearance = trace_snapshots[just_before_reappearance]
    register_hue_at_reappearance = register_snapshots[just_before_reappearance]
    occlusion_len_at_reappearance = occlusion_lengths[just_before_reappearance]

    per_timescale_mse = {
        f"halflife_{halflife}": float(
            np.mean((trace_hue_at_reappearance[:, i] - true_hues) ** 2)
        )
        for i, halflife in enumerate(timescales)
    }
    register_mse = float(np.mean((register_hue_at_reappearance - true_hues) ** 2))

    # Does the best (shortest-halflife) decaying trace's error grow with
    # occlusion length, while the register's does not? Bucket by length.
    buckets = [(1, 5), (6, 15), (16, 40)]
    shortest_halflife_index = 0
    error_by_bucket = {}
    for lo, hi in buckets:
        mask = (occlusion_len_at_reappearance >= lo) & (
            occlusion_len_at_reappearance <= hi
        )
        if not mask.any():
            continue
        trace_masked = trace_hue_at_reappearance[mask, shortest_halflife_index]
        trace_err = float(np.mean((trace_masked - true_hues[mask]) ** 2))
        register_err = float(
            np.mean((register_hue_at_reappearance[mask] - true_hues[mask]) ** 2)
        )
        error_by_bucket[f"occlusion_{lo}_to_{hi}_steps"] = {
            "n_episodes": int(mask.sum()),
            "shortest_halflife_trace_mse": trace_err,
            "persistent_register_mse": register_err,
        }

    report = {
        "n_episodes": args.n_episodes,
        "decaying_trace_mse_by_halflife": per_timescale_mse,
        "persistent_register_mse": register_mse,
        "mse_by_occlusion_length_bucket": error_by_bucket,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
