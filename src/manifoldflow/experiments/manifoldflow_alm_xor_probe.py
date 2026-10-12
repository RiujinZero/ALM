"""Does MF-ALM-TS's local learning rule solve a non-linearly-separable
task? See `alm/eval/xor_probe.py` for the task itself (temporal XOR) and
the reasoning for why it's the relevant test.

Two mechanisms are compared:
  - `run_variant` (M1): one plastic column, local reconstruction + local
    token-head updates only, no PC-ALM, no eligibility -- the mechanism
    already shown (see docs/ManifoldFlow_ALM_Journal.md) to collapse to a
    constant prediction on this task, because the token head's error
    never reaches the weights that shape the hidden representation.
  - `run_variant_pc_alm` (M2): a small PCALMHierarchy where the top
    column's token-head error is folded into settling as extra activity
    pressure (section 7's `L_token` term), which -- via the same
    neighbor-exchange term that lets adjacent columns negotiate their
    activities -- can reach the *lower* column's weights too. This tests
    directly whether that channel is enough to solve XOR where M1 failed.

Both report label-position accuracy/CE before training (does the random
initialization already solve it) and after/during (did learning help), so
"random features got lucky" and "learning changed something" aren't
conflated into one number.
"""

from __future__ import annotations

import argparse
import json

import torch

from manifoldflow.alm.eval.xor_probe import (
    EOS,
    VOCAB_SIZE,
    build_xor_stream,
    evaluate_at_label_positions,
    evaluate_hierarchy_at_label_positions,
)
from manifoldflow.alm.learning.pc_alm import PCALMConfig, PCALMHierarchy
from manifoldflow.alm.model.plastic_column import PlasticColumn, PlasticColumnConfig
from manifoldflow.alm.train.pc_alm_trainer import train_pc_alm_hierarchy
from manifoldflow.alm.train.stream_trainer import train_single_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=16)
    parser.add_argument("--pc-alm-layers", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=40_000)
    parser.add_argument("--eval-examples", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every-examples", type=int, default=4_000)
    return parser.parse_args()


def run_variant(args: argparse.Namespace, with_separator: bool) -> dict:
    config = PlasticColumnConfig(
        input_dim=args.token_dim,
        state_dim=args.state_dim,
        token_dim=args.token_dim,
        vocab_size=VOCAB_SIZE,
        seed=args.seed,
    )
    column = PlasticColumn(config, generator=torch.Generator().manual_seed(args.seed))

    eval_tokens, eval_positions = build_xor_stream(
        args.eval_examples, seed=args.seed + 1000, with_separator=with_separator
    )
    before = evaluate_at_label_positions(column, eval_tokens, eval_positions)

    train_tokens, _train_positions = build_xor_stream(
        args.train_examples, seed=args.seed, with_separator=with_separator
    )

    # Re-evaluate periodically during training to see the trend, not just
    # a before/after snapshot -- a flat line distinguishes "random features
    # already solved it" from "learning changed something."
    checkpoint_tokens = args.checkpoint_every_examples * (6 if with_separator else 5)
    trend = []
    stats = None
    h_prev = None
    start_index = 0
    while start_index < len(train_tokens) - 1:
        chunk_end = min(start_index + checkpoint_tokens, len(train_tokens) - 1)
        stats, h_prev = train_single_column(
            column,
            train_tokens[: chunk_end + 1],
            eos_id=EOS,
            log_every=max(checkpoint_tokens, 1000),
            start_index=start_index,
            stats=stats,
            h_prev=h_prev,
        )
        snapshot = evaluate_at_label_positions(column, eval_tokens, eval_positions)
        trend.append({"tokens_seen": stats.tokens_seen, **snapshot})
        start_index = chunk_end

    after = evaluate_at_label_positions(column, eval_tokens, eval_positions)
    return {"before_training": before, "after_training": after, "trend": trend}


def run_variant_pc_alm(args: argparse.Namespace, with_separator: bool) -> dict:
    generator = torch.Generator().manual_seed(args.seed)
    columns = []
    for layer in range(args.pc_alm_layers):
        input_dim = args.token_dim if layer == 0 else args.state_dim
        config = PlasticColumnConfig(
            input_dim=input_dim,
            state_dim=args.state_dim,
            token_dim=args.token_dim,
            vocab_size=VOCAB_SIZE,
            seed=args.seed + layer,
        )
        columns.append(PlasticColumn(config, generator=generator))
    hierarchy = PCALMHierarchy(columns, PCALMConfig())

    eval_tokens, eval_positions = build_xor_stream(
        args.eval_examples, seed=args.seed + 1000, with_separator=with_separator
    )
    before = evaluate_hierarchy_at_label_positions(
        hierarchy, eval_tokens, eval_positions
    )

    train_tokens, _train_positions = build_xor_stream(
        args.train_examples, seed=args.seed, with_separator=with_separator
    )

    checkpoint_tokens = args.checkpoint_every_examples * (6 if with_separator else 5)
    trend = []
    stats = None
    h_prev = None
    start_index = 0
    while start_index < len(train_tokens) - 1:
        chunk_end = min(start_index + checkpoint_tokens, len(train_tokens) - 1)
        stats, h_prev = train_pc_alm_hierarchy(
            hierarchy,
            train_tokens[: chunk_end + 1],
            eos_id=EOS,
            log_every=max(checkpoint_tokens, 1000),
            start_index=start_index,
            stats=stats,
            h_prev=h_prev,
        )
        snapshot = evaluate_hierarchy_at_label_positions(
            hierarchy, eval_tokens, eval_positions
        )
        trend.append({"tokens_seen": stats.tokens_seen, **snapshot})
        start_index = chunk_end

    after = evaluate_hierarchy_at_label_positions(
        hierarchy, eval_tokens, eval_positions
    )
    return {"before_training": before, "after_training": after, "trend": trend}


def main() -> None:
    args = parse_args()
    results = {
        "M1_single_column": {
            "no_delay (bit_a, bit_b adjacent)": run_variant(args, with_separator=False),
            "with_delay (bit_a, SEP, bit_b)": run_variant(args, with_separator=True),
        },
        "M2_pc_alm_hierarchy": {
            "no_delay (bit_a, bit_b adjacent)": run_variant_pc_alm(
                args, with_separator=False
            ),
            "with_delay (bit_a, SEP, bit_b)": run_variant_pc_alm(
                args, with_separator=True
            ),
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
