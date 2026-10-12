"""Cheapest deterministic shared-core/edge split for future-vector candidates.

Operationalizes ``z_m = core + edge_m`` as literally as possible: core is the
posterior-mass-weighted mean of a fixed flat-top-k candidate shortlist, and
``edge_m`` is each retained candidate's residual from that mean.

This makes within-shortlist reconstruction (``core + edge_m == z_m``) and
edge-vs-candidate correspondence exact by construction -- for any set of
vectors and any subtracted constant, the residuals trivially reconstruct the
originals and trivially differ exactly as the originals differ. Neither
property is empirically falsifiable at this operationalization; both are
included below only as implementation sanity checks, not as evidence the
split is useful.

The one claim this cheap version can actually fail is whether the resulting
core is context/episode-specific: does substituting a *different* episode's
core measurably degrade reconstruction relative to the episode's own core? If
it does not, this mean carries no reusable information worth training a
receiver to express separately, and the next investment is a better core
estimator (see docs/ALM_V1_Architecture.md, core/edge section), not a decoder
for this one. See dendritic_basin_future_core_edge_ablation.py for that test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoreEdgeSplit:
    """A mass-weighted core and each retained candidate's residual from it."""

    core: np.ndarray
    edges: tuple[np.ndarray, ...]  # aligned position-for-position to `chosen`
    chosen: tuple[int, ...]
    core_probability_mass: float

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.chosen):
            raise ValueError("edges must align one-for-one with chosen candidates")
        core = np.asarray(self.core, dtype=np.float64).copy()
        core.setflags(write=False)
        object.__setattr__(self, "core", core)
        edges = tuple(np.asarray(edge, dtype=np.float64).copy() for edge in self.edges)
        for edge in edges:
            if edge.shape != core.shape:
                raise ValueError("every edge must share the core's shape")
            edge.setflags(write=False)
        object.__setattr__(self, "edges", edges)


def mass_weighted_core_edge_split(
    means: tuple[np.ndarray, ...],
    probabilities: np.ndarray,
    chosen: tuple[int, ...],
) -> CoreEdgeSplit:
    """The cheapest possible core estimator: a mass-weighted mean of `chosen`.

    No new learned parameters and no training loop -- one weighted average
    per call. Retains original candidate identity: edges are residuals from
    a shared point, never a substitute for the candidates themselves.
    """

    if not chosen:
        raise ValueError("core/edge split requires at least one retained candidate")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(means),) or np.any(values < 0.0):
        raise ValueError("probabilities do not align to candidate means")
    if not np.all(np.isfinite(values)) or float(values.sum()) <= 0.0:
        raise ValueError("probabilities must be finite with positive mass")
    indexes = np.asarray(chosen, dtype=np.int64)
    if indexes.min() < 0 or indexes.max() >= len(means):
        raise ValueError("chosen indexes are outside the candidate set")
    weights = values[indexes]
    mass = float(weights.sum())
    if mass <= 0.0:
        raise ValueError("retained candidates must carry positive probability mass")
    normalized = weights / mass
    stacked = np.stack(
        [np.asarray(means[index], dtype=np.float64) for index in chosen]
    )
    if not np.all(np.isfinite(stacked)):
        raise ValueError("candidate means must be finite")
    core = np.average(stacked, axis=0, weights=normalized)
    edges = tuple(stacked[position] - core for position in range(len(chosen)))
    return CoreEdgeSplit(
        core=core,
        edges=edges,
        chosen=tuple(int(index) for index in chosen),
        core_probability_mass=mass,
    )


def reconstruction_error(
    target: np.ndarray, core: np.ndarray, edge: np.ndarray
) -> float:
    """Mean squared error of `core + edge` against `target`, per coordinate."""

    predicted = np.asarray(core, dtype=np.float64) + np.asarray(edge, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError("core, edge, and target must share one shape")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ValueError("reconstruction inputs must be finite")
    return float(np.mean(np.square(observed - predicted)))
