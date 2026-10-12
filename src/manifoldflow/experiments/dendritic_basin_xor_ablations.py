"""Phase 0 (design doc section 39): does Dendritic Basin Learning solve
static XOR, and do the ablations show the expected pattern (section 25)?

  A. Linear local learner       -- expected ~50% (XOR isn't linearly
                                    separable in the literal encoding
                                    either -- it's an affine reparameterization
                                    of the raw two bits).
  B. Dendrites, no structural
     plasticity                -- fixed random sparse branches; measures
                                    whether random conjunctions suffice.
  C. Structural dendrites,
     no recurrent basin        -- readout reads dendritic drive directly.
  D. Full architecture         -- dendrites + structural growth + basin.

Success criterion (section 24): 100% accuracy across >=20 random seeds,
with no hidden representation manually initialized to XOR-specific
conjunctions (grow_branch only ever wires to whatever literals are
currently active in the input it was triggered by -- see dendrites.py).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from manifoldflow.dendritic_basin.model import (
    DendriticBasinConfig,
    DendriticBasinSystem,
)
from manifoldflow.dendritic_basin.readout import LocalReadout, ReadoutConfig
from manifoldflow.dendritic_basin.tasks import xor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-examples", type=int, default=2000)
    parser.add_argument("--n-branches", type=int, default=16)
    parser.add_argument("--synapses-per-branch", type=int, default=2)
    return parser.parse_args()


def run_ablation_a_linear(seed: int, train_examples: int) -> float:
    """Bare local delta-rule readout on the literal encoding -- no dendrites."""
    rng = np.random.default_rng(seed)
    readout = LocalReadout(ReadoutConfig(n_inputs=4, seed=seed))
    for _ in range(train_examples):
        x, y = xor.sample(rng)
        z = np.array([x[0], 1 - x[0], x[1], 1 - x[1]])
        y_hat = readout.predict(z)
        readout.learn(z, modulator=y - y_hat)

    def predict_fn(x: np.ndarray) -> float:
        z = np.array([x[0], 1 - x[0], x[1], 1 - x[1]])
        return readout.predict(z)

    accuracy, _ = xor.evaluate_all_rows(predict_fn)
    return accuracy


def run_dendritic_system(
    seed: int,
    train_examples: int,
    n_branches: int,
    synapses_per_branch: int,
    use_basin: bool,
    structural_plasticity: bool,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    config = DendriticBasinConfig(
        n_inputs=xor.N_INPUTS,
        n_branches=n_branches,
        synapses_per_branch=synapses_per_branch,
        use_basin=use_basin,
        structural_plasticity=structural_plasticity,
        seed=seed,
    )
    system = DendriticBasinSystem(config)
    for _ in range(train_examples):
        x, y = xor.sample(rng)
        system.step(x, target=y)

    accuracy, _ = xor.evaluate_all_rows(system.predict)
    return accuracy, system.dendrites.n_active_branches()


def main() -> None:
    args = parse_args()
    results: dict[str, dict] = {
        "A_linear_local_learner": {"accuracies": []},
        "B_dendrites_no_structural_plasticity": {"accuracies": [], "branches": []},
        "C_structural_dendrites_no_basin": {"accuracies": [], "branches": []},
        "D_full_architecture": {"accuracies": [], "branches": []},
    }

    for seed in range(args.seeds):
        results["A_linear_local_learner"]["accuracies"].append(
            run_ablation_a_linear(seed, args.train_examples)
        )

        acc_b, branches_b = run_dendritic_system(
            seed,
            args.train_examples,
            args.n_branches,
            args.synapses_per_branch,
            use_basin=True,
            structural_plasticity=False,
        )
        results["B_dendrites_no_structural_plasticity"]["accuracies"].append(acc_b)
        results["B_dendrites_no_structural_plasticity"]["branches"].append(branches_b)

        acc_c, branches_c = run_dendritic_system(
            seed,
            args.train_examples,
            args.n_branches,
            args.synapses_per_branch,
            use_basin=False,
            structural_plasticity=True,
        )
        results["C_structural_dendrites_no_basin"]["accuracies"].append(acc_c)
        results["C_structural_dendrites_no_basin"]["branches"].append(branches_c)

        acc_d, branches_d = run_dendritic_system(
            seed,
            args.train_examples,
            args.n_branches,
            args.synapses_per_branch,
            use_basin=True,
            structural_plasticity=True,
        )
        results["D_full_architecture"]["accuracies"].append(acc_d)
        results["D_full_architecture"]["branches"].append(branches_d)

    summary = {}
    for name, data in results.items():
        accuracies = data["accuracies"]
        entry = {
            "mean_accuracy": float(np.mean(accuracies)),
            "n_seeds_at_100pct": int(sum(a == 1.0 for a in accuracies)),
            "n_seeds": len(accuracies),
            "accuracies": accuracies,
        }
        if "branches" in data:
            entry["mean_branches_grown"] = float(np.mean(data["branches"]))
        summary[name] = entry

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
