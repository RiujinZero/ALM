"""Overlapping cortical assembly recognition from local lag receptors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssemblyActivation:
    assembly: int
    support: float
    matched_synapses: int


class MultiscaleAssemblyField:
    """Let learned constituents remain active inside longer stream motifs.

    Every stored pattern grows one excitatory synapse from each raw byte/lag
    receptor. Current receptors excite all connected assemblies in parallel.
    Normalized membrane support and a local threshold replace exact string
    lookup; assemblies of different sizes may coactivate.
    """

    def __init__(
        self,
        *,
        maximum_lag: int = 16,
        minimum_synapses: int = 3,
        activation_threshold: float = 0.8,
        maximum_active: int = 32,
        trace_half_life_bytes: float = 0.0,
        minimum_trace: float = 0.25,
    ) -> None:
        if maximum_lag < 1 or minimum_synapses < 1 or maximum_active < 1:
            raise ValueError("lag, synapse, and activity limits must be positive")
        if not 0.0 < activation_threshold <= 1.0:
            raise ValueError("activation threshold must be in (0, 1]")
        if trace_half_life_bytes < 0.0 or not 0.0 <= minimum_trace <= 1.0:
            raise ValueError("trace half-life or threshold is invalid")
        self.maximum_lag = maximum_lag
        self.minimum_synapses = minimum_synapses
        self.activation_threshold = activation_threshold
        self.maximum_active = maximum_active
        self.trace_half_life_bytes = trace_half_life_bytes
        self.minimum_trace = minimum_trace
        self.incoming: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        self.synapse_totals: dict[int, float] = {}
        self.synapse_counts: dict[int, int] = {}
        self.history: deque[int] = deque(maxlen=maximum_lag)
        self.trace_strength: dict[int, float] = {}
        self._execution_rows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._assembly_capacity = 0
        self._synapse_total_array = np.zeros(0, dtype=np.float64)
        self._execution_dirty = False

    def __getstate__(self) -> dict[str, object]:
        """Exclude rebuildable current-integration arrays from body bytes."""
        state = self.__dict__.copy()
        state.pop("_execution_rows", None)
        state.pop("_assembly_capacity", None)
        state.pop("_synapse_total_array", None)
        state.pop("_execution_dirty", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._rebuild_execution_rows()

    def _rebuild_execution_rows(self) -> None:
        self._execution_rows = {
            key: (
                np.fromiter(row, dtype=np.intp, count=len(row)),
                np.fromiter(row.values(), dtype=np.float64, count=len(row)),
            )
            for key, row in self.incoming.items()
            if row
        }
        self._assembly_capacity = max(self.synapse_counts, default=-1) + 1
        self._synapse_total_array = np.ones(
            self._assembly_capacity, dtype=np.float64
        )
        for assembly, total in self.synapse_totals.items():
            self._synapse_total_array[assembly] = total
        self._execution_dirty = False

    @property
    def embodied_bytes(self) -> int:
        import pickle

        return len(
            pickle.dumps(
                (
                    self.maximum_lag,
                    self.minimum_synapses,
                    self.activation_threshold,
                    self.maximum_active,
                    self.trace_half_life_bytes,
                    self.minimum_trace,
                    self.incoming,
                    self.synapse_totals,
                    self.synapse_counts,
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    def bind(self, assembly: int, pattern: bytes) -> None:
        """Grow local lag-receptor synapses for one learned sensory pattern."""
        if assembly < 0:
            raise ValueError("assembly id must be non-negative")
        retained = pattern[-self.maximum_lag :]
        if len(retained) < self.minimum_synapses:
            return
        # Rebinding after consolidation replaces this assembly's old anatomy.
        if assembly in self.synapse_counts:
            for row in self.incoming.values():
                row.pop(assembly, None)
        weight = 1.0 / len(retained)
        for lag, byte_id in enumerate(reversed(retained), start=1):
            self.incoming[(lag, int(byte_id))][assembly] = weight
        self.synapse_totals[assembly] = 1.0
        self.synapse_counts[assembly] = len(retained)
        self._execution_dirty = True

    def reset_episode(self) -> None:
        self.history.clear()
        self.trace_strength.clear()

    def observe(self, byte_id: int) -> tuple[AssemblyActivation, ...]:
        """Advance raw receptors and return all locally suprathreshold somata."""
        if not 0 <= byte_id < 256:
            raise ValueError("byte id must be in [0, 255]")
        if self._execution_dirty:
            self._rebuild_execution_rows()
        self.history.append(byte_id)
        drive = np.zeros(self._assembly_capacity, dtype=np.float64)
        hits = np.zeros(self._assembly_capacity, dtype=np.int16)
        for lag, value in enumerate(reversed(self.history), start=1):
            row = self._execution_rows.get((lag, value))
            if row is None:
                continue
            assemblies, weights = row
            drive[assemblies] += weights
            hits[assemblies] += 1
        eligible = np.flatnonzero(
            (hits >= self.minimum_synapses)
            & (
                drive / np.maximum(self._synapse_total_array, 1e-12)
                >= self.activation_threshold
            )
        )
        active = [
            AssemblyActivation(
                int(assembly),
                float(drive[assembly] / max(self.synapse_totals[int(assembly)], 1e-12)),
                int(hits[assembly]),
            )
            for assembly in eligible
        ]
        if self.trace_half_life_bytes > 0.0:
            decay = 2.0 ** (-1.0 / self.trace_half_life_bytes)
            self.trace_strength = {
                assembly: strength * decay
                for assembly, strength in self.trace_strength.items()
                if strength * decay >= self.minimum_trace
            }
            for item in active:
                self.trace_strength[item.assembly] = max(
                    item.support, self.trace_strength.get(item.assembly, 0.0)
                )
            active_by_id = {item.assembly: item for item in active}
            for assembly, strength in self.trace_strength.items():
                if assembly not in active_by_id:
                    active.append(AssemblyActivation(assembly, strength, 0))
        active.sort(
            key=lambda item: (-item.support, -item.matched_synapses, item.assembly)
        )
        return tuple(active[: self.maximum_active])
