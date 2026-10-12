"""Sparse cortical assemblies jointly driven by basal and apical context."""

from __future__ import annotations

import pickle

import numpy as np


class SharedCorticalAssemblies:
    """Locally learn reusable word assemblies from two provenance compartments."""

    def __init__(
        self,
        cortical_dimensions: int,
        *,
        learning_rate: float = 0.5,
        maximum_synapses: int = 128,
        maximum_competitors: int = 16,
    ) -> None:
        self.cortical_dimensions = int(cortical_dimensions)
        self.dimensions = 2 * self.cortical_dimensions
        self.learning_rate = float(learning_rate)
        self.maximum_synapses = int(maximum_synapses)
        self.maximum_competitors = int(maximum_competitors)
        self.weights: dict[int, dict[int, float]] = {}
        self.observations: dict[int, int] = {}
        self.external_updates = 0

    @property
    def embodied_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def features(
        self, basal: np.ndarray, apical: np.ndarray | None = None
    ) -> np.ndarray:
        basal = np.asarray(basal, dtype=np.float64)
        if basal.shape != (self.cortical_dimensions,):
            raise ValueError("basal cortical shape mismatch")
        if apical is None:
            apical = np.zeros_like(basal)
        else:
            apical = np.asarray(apical, dtype=np.float64)
            if apical.shape != basal.shape:
                raise ValueError("apical cortical shape mismatch")
        values = np.concatenate((basal, apical))
        maximum = float(np.max(np.abs(values), initial=0.0))
        return values / maximum if maximum > 0.0 else values

    @staticmethod
    def _activation(row: dict[int, float], features: np.ndarray) -> float:
        if not row:
            return 0.0
        value = sum(weight * float(features[index]) for index, weight in row.items())
        return float(np.clip(value, 0.0, 1.0))

    def activations(self, features: np.ndarray) -> dict[int, float]:
        return {
            assembly: value
            for assembly, row in self.weights.items()
            if (value := self._activation(row, features)) > 0.0
        }

    def reconstruct(self, activations: dict[int, float]) -> np.ndarray:
        """Return transient cortical evidence without performing learning."""
        reconstructed = np.zeros(self.dimensions, dtype=np.float64)
        total = 0.0
        for assembly, activity in activations.items():
            if activity <= 0.0:
                continue
            row = self.weights.get(assembly, {})
            for index, weight in row.items():
                # Negative afferents suppress an assembly; they are not
                # positive evidence to recreate in the receiving cortex.
                if weight > 0.0:
                    reconstructed[index] += float(activity) * weight
            total += float(activity)
        if total > 0.0:
            reconstructed /= total
        maximum = float(np.max(np.abs(reconstructed), initial=0.0))
        if maximum > 0.0:
            reconstructed /= maximum
        return reconstructed

    def _update_row(
        self, assembly: int, features: np.ndarray, target: float
    ) -> None:
        row = self.weights.setdefault(int(assembly), {})
        active = np.flatnonzero(np.abs(features) > 0.0)
        if not len(active):
            return
        selected = tuple(
            sorted(
                (int(index) for index in active),
                key=lambda index: (-abs(float(features[index])), index),
            )[: self.maximum_synapses]
        )
        prediction = self._activation(row, features)
        error = float(target) - prediction
        energy = sum(float(features[index]) ** 2 for index in selected)
        for index in selected:
            row[index] = float(
                np.clip(
                    row.get(index, 0.0)
                    + self.learning_rate
                    * error
                    * float(features[index])
                    / max(energy, 1e-12),
                    -1.0,
                    1.0,
                )
            )
        for index in tuple(row):
            if abs(row[index]) <= 1e-8:
                del row[index]
        if len(row) > self.maximum_synapses:
            retained = dict(
                sorted(row.items(), key=lambda item: (-abs(item[1]), item[0]))[
                    : self.maximum_synapses
                ]
            )
            self.weights[assembly] = retained

    def observe_external(self, features: np.ndarray, assembly: int) -> None:
        """Prediction-error update; only a sensory outcome may call this."""
        predictions = self.activations(features)
        competitors = sorted(
            (
                (value, candidate)
                for candidate, value in predictions.items()
                if candidate != assembly
            ),
            reverse=True,
        )[: self.maximum_competitors]
        self._update_row(assembly, features, 1.0)
        for _value, candidate in competitors:
            self._update_row(candidate, features, 0.0)
        self.observations[assembly] = self.observations.get(assembly, 0) + 1
        self.external_updates += 1
