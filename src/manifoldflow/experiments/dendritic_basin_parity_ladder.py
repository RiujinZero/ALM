"""Phase 1 (design doc section 54: "Recursive parity"): does the real
(non-sandbox) interaction-dendrite + relation-tree implementation scale
to the bit widths the design reports (128, 256, 512, 1024), with `N-1`
relation units rather than `2^N` memorized patterns?

Kept out of the pytest suite (`test_dendritic_basin_relation_tree.py`
covers small bit-widths for fast CI) since 1024 bits takes on the order
of a minute or two to train.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from manifoldflow.dendritic_basin.relation_tree import (
    build_relation_tree,
    evaluate_tree_accuracy,
    train_relation_tree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bit-widths", type=int, nargs="+", default=[8, 16, 32, 128, 256, 512, 1024]
    )
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--perturbations-per-round", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.3)
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--max-episodes-per-node", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for n_bits in args.bit_widths:
        t0 = time.time()
        rng = np.random.default_rng(n_bits)
        root, levels = build_relation_tree(leaf_count=n_bits, seed=n_bits * 7)
        unresolved = train_relation_tree(
            levels,
            n_bits=n_bits,
            rounds=args.rounds,
            perturbations_per_round=args.perturbations_per_round,
            step_size=args.step_size,
            rng=rng,
            max_episodes_per_node=args.max_episodes_per_node,
        )
        accuracy = evaluate_tree_accuracy(
            root, n_bits=n_bits, n_eval=args.eval_examples, rng=rng
        )
        elapsed = time.time() - t0
        results[n_bits] = {
            "n_relation_units": n_bits - 1,
            "unresolved_nodes": unresolved,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
        }
        print(json.dumps({n_bits: results[n_bits]}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
