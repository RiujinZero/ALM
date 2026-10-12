"""DendriticBasinSystem (design doc section 20): wires
encode -> dendrites -> basin -> readout -> modulator -> local learning,
with no backward pass, no autograd, and no shared parameter that any two
components learn jointly through a common gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldflow.dendritic_basin.basin import BasinConfig, RecurrentBasin
from manifoldflow.dendritic_basin.dendrites import DendriticConfig, DendriticPopulation
from manifoldflow.dendritic_basin.readout import LocalReadout, ReadoutConfig


def encode_literals(x: np.ndarray) -> np.ndarray:
    """z = (x_1, 1-x_1, x_2, 1-x_2, ...) (section 4). Not learned."""
    z = np.empty(x.size * 2, dtype=np.float64)
    z[0::2] = x
    z[1::2] = 1.0 - x
    return z


@dataclass
class DendriticBasinConfig:
    n_inputs: int
    n_branches: int = 16
    synapses_per_branch: int = 2
    use_basin: bool = True  # ablation C: False disables the recurrent basin
    structural_plasticity: bool = True  # ablation B: False disables branch growth
    seed: int = 0
    dendrite: DendriticConfig | None = None
    basin: BasinConfig | None = None
    readout: ReadoutConfig | None = None


class DendriticBasinSystem:
    def __init__(self, config: DendriticBasinConfig):
        self.config = config
        literal_dim = config.n_inputs * 2
        dendrite_cfg = config.dendrite or DendriticConfig(
            n_branches=config.n_branches,
            input_dim=literal_dim,
            synapses_per_branch=config.synapses_per_branch,
            structural_plasticity=config.structural_plasticity,
            seed=config.seed,
        )
        self.dendrites = DendriticPopulation(dendrite_cfg)
        self.use_basin = config.use_basin
        if self.use_basin:
            basin_cfg = config.basin or BasinConfig(
                n_units=dendrite_cfg.n_branches, seed=config.seed + 1
            )
            self.basin: RecurrentBasin | None = RecurrentBasin(basin_cfg)
        else:
            self.basin = None
        readout_cfg = config.readout or ReadoutConfig(
            n_inputs=dendrite_cfg.n_branches, seed=config.seed + 2
        )
        self.readout = LocalReadout(readout_cfg)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return encode_literals(x)

    def step(self, x: np.ndarray, target: float | None = None) -> float:
        """Section 19's complete learning cycle, steps 1-9 (step 10,
        prune/split, is not yet implemented -- see the implementation
        journal for Phase 0 scope).
        """
        z = self.encode(x)
        d = self.dendrites.activate(z)
        h = self.basin.settle(d) if self.basin is not None else d
        y_hat = self.readout.predict(h)

        if target is not None:
            modulator = float(target) - y_hat
            self.dendrites.update_eligibility(z, d)
            self.readout.learn(h, modulator)
            if self.basin is not None:
                self.basin.learn(h)
            self.dendrites.learn(modulator)
            self.dendrites.adapt_structure(z, d, modulator)
        return y_hat

    def predict(self, x: np.ndarray) -> float:
        """Inference-only forward pass; never updates any local state."""
        z = self.encode(x)
        d = self.dendrites.activate(z)
        h = self.basin.settle(d) if self.basin is not None else d
        return self.readout.predict(h)
