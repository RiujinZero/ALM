"""Deterministic shared-core families for future-vector candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FamilyShortlist:
    families: tuple[tuple[int, ...], ...]
    selected_families: tuple[int, ...]
    selected_candidates: tuple[int, ...]
    eligible_candidates: tuple[int, ...]
    family_probabilities: np.ndarray
    shared_core_fraction: float

    def __post_init__(self) -> None:
        values = np.asarray(self.family_probabilities, dtype=np.float64).copy()
        values.setflags(write=False)
        object.__setattr__(self, "family_probabilities", values)


def _standardized_rows(
    means: tuple[np.ndarray, ...], variance: np.ndarray
) -> np.ndarray:
    matrix = np.stack([np.asarray(item, dtype=np.float64) for item in means])
    scale = np.asarray(variance, dtype=np.float64)
    if matrix.ndim != 2 or scale.shape != (matrix.shape[1],):
        raise ValueError("means and variance do not align")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(scale)):
        raise ValueError("family geometry must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("family variance must be positive")
    return matrix / np.sqrt(scale)[None, :]


def agglomerative_families(
    means: tuple[np.ndarray, ...],
    variance: np.ndarray,
    *,
    maximum_families: int,
) -> tuple[tuple[int, ...], ...]:
    """Merge closest Ward centroids until only the locked family count remains."""

    rows = _standardized_rows(means, variance)
    if maximum_families <= 0:
        raise ValueError("maximum_families must be positive")
    clusters = [tuple([index]) for index in range(len(rows))]
    while len(clusters) > maximum_families:
        best: tuple[float, int, int] | None = None
        for left in range(len(clusters) - 1):
            left_members = clusters[left]
            left_center = np.mean(rows[np.asarray(left_members)], axis=0)
            for right in range(left + 1, len(clusters)):
                right_members = clusters[right]
                right_center = np.mean(rows[np.asarray(right_members)], axis=0)
                coefficient = (
                    len(left_members)
                    * len(right_members)
                    / (len(left_members) + len(right_members))
                )
                cost = coefficient * float(
                    np.mean(np.square(left_center - right_center))
                )
                candidate = (cost, left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, left, right = best
        merged = tuple(sorted((*clusters[left], *clusters[right])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (left, right)
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0])
    return tuple(clusters)


def random_size_matched_families(
    families: tuple[tuple[int, ...], ...], *, seed: int
) -> tuple[tuple[int, ...], ...]:
    """Destroy geometry while preserving family sizes exactly."""

    sizes = tuple(len(family) for family in families)
    count = sum(sizes)
    permutation = np.random.default_rng(seed).permutation(count)
    shuffled = []
    offset = 0
    for size in sizes:
        shuffled.append(
            tuple(sorted(int(item) for item in permutation[offset : offset + size]))
        )
        offset += size
    return tuple(shuffled)


def shortlist_families(
    means: tuple[np.ndarray, ...],
    probabilities: np.ndarray,
    variance: np.ndarray,
    families: tuple[tuple[int, ...], ...],
    *,
    family_mass: float,
    maximum_selected_families: int,
    candidate_mass: float,
    maximum_candidates: int,
) -> FamilyShortlist:
    """Choose high-mass cores, then keep their highest-probability edges."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(means),) or np.any(values < 0.0):
        raise ValueError("probabilities do not align to candidate means")
    if not np.all(np.isfinite(values)) or float(values.sum()) <= 0.0:
        raise ValueError("probabilities must be finite with positive mass")
    values = values / values.sum()
    flat = tuple(sorted(index for family in families for index in family))
    if flat != tuple(range(len(means))):
        raise ValueError("families must partition every candidate exactly once")
    if not 0.0 < family_mass <= 1.0 or not 0.0 < candidate_mass <= 1.0:
        raise ValueError("mass thresholds must be in (0, 1]")
    if maximum_selected_families <= 0 or maximum_candidates <= 0:
        raise ValueError("selection maxima must be positive")

    family_probabilities = np.asarray(
        [values[np.asarray(family, dtype=np.int64)].sum() for family in families]
    )
    family_order = np.argsort(-family_probabilities, kind="stable")
    selected_families: list[int] = []
    accumulated = 0.0
    for raw_index in family_order[:maximum_selected_families]:
        index = int(raw_index)
        selected_families.append(index)
        accumulated += float(family_probabilities[index])
        if accumulated >= family_mass:
            break
    eligible = tuple(
        sorted(
            member
            for family_index in selected_families
            for member in families[family_index]
        )
    )
    candidate_order = sorted(eligible, key=lambda index: (-values[index], index))
    selected_candidates: list[int] = []
    accumulated = 0.0
    for index in candidate_order[:maximum_candidates]:
        selected_candidates.append(index)
        accumulated += float(values[index])
        if accumulated >= candidate_mass:
            break

    rows = _standardized_rows(means, variance)
    global_center = np.average(rows, axis=0, weights=values)
    total = float(np.sum(values * np.mean(np.square(rows - global_center), axis=1)))
    within = 0.0
    selected_mass = 0.0
    for family_index in selected_families:
        members = np.asarray(families[family_index], dtype=np.int64)
        weights = values[members]
        mass = float(weights.sum())
        if mass <= 0.0:
            continue
        center = np.average(rows[members], axis=0, weights=weights)
        within += float(
            np.sum(weights * np.mean(np.square(rows[members] - center), axis=1))
        )
        selected_mass += mass
    denominator = max(total * selected_mass, 1e-12)
    shared_core_fraction = float(np.clip(1.0 - within / denominator, 0.0, 1.0))
    return FamilyShortlist(
        families=families,
        selected_families=tuple(selected_families),
        selected_candidates=tuple(selected_candidates),
        eligible_candidates=eligible,
        family_probabilities=family_probabilities,
        shared_core_fraction=shared_core_fraction,
    )
