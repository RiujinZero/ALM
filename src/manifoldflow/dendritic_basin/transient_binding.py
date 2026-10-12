"""Fast sparse indices between slow context, structural role, and cortical filler."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BindingRecall:
    """A reinstatement request aimed at an existing cortical assembly."""

    filler: int | None
    cortical_role: int | None
    confidence: float
    active_cells: int


class FastSparseBindingIndex:
    """Bounded episode-scale coincidence cells with reciprocal addressing.

    Cells contain no sensory or output model.  They only associate integer
    identities already represented by other organs.  Formation and recall are
    local to an active cell; k-WTA allocation supplies sparse competition.
    """

    def __init__(
        self,
        *,
        max_cells: int = 256,
        cells_per_binding: int = 2,
        activity_decay: float = 0.98,
        minimum_activity: float = 0.05,
    ) -> None:
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        if not 1 <= cells_per_binding <= max_cells:
            raise ValueError("cells_per_binding must fit within max_cells")
        if not 0.0 < activity_decay <= 1.0:
            raise ValueError("activity_decay must be in (0, 1]")
        if not 0.0 <= minimum_activity < 1.0:
            raise ValueError("minimum_activity must be in [0, 1)")
        self.max_cells = int(max_cells)
        self.cells_per_binding = int(cells_per_binding)
        self.activity_decay = float(activity_decay)
        self.minimum_activity = float(minimum_activity)

        # Working synapses.  -1 denotes an uncommitted cell.
        # Addresses may be organ IDs or compact raw sensory receptor codes.
        self.basin = np.full(max_cells, -1, dtype=np.int32)
        self.role = np.full(max_cells, -1, dtype=np.int16)
        self.filler = np.full(max_cells, -1, dtype=np.int32)
        self.cortical_role = np.full(max_cells, -1, dtype=np.int16)
        self.activity = np.zeros(max_cells, dtype=np.float32)
        self.last_used = np.zeros(max_cells, dtype=np.uint64)
        self.clock = 0
        self.bindings = 0
        self.recalls = 0
        self.replacements = 0

    @property
    def live_cells(self) -> int:
        return int(np.count_nonzero(self.filler >= 0))

    @property
    def embodied_bytes(self) -> int:
        # Fast associations are working state; only the organizing constants
        # belong to the durable body.
        return len(
            pickle.dumps(
                (
                    self.max_cells,
                    self.cells_per_binding,
                    self.activity_decay,
                    self.minimum_activity,
                ),
                protocol=5,
            )
        )

    @property
    def transient_bytes(self) -> int:
        return len(
            pickle.dumps(
                (
                    self.basin,
                    self.role,
                    self.filler,
                    self.cortical_role,
                    self.activity,
                    self.last_used,
                    self.clock,
                ),
                protocol=5,
            )
        )

    def _advance(self) -> None:
        self.clock += 1
        self.activity *= self.activity_decay
        expired = (self.filler >= 0) & (self.activity < self.minimum_activity)
        self.basin[expired] = -1
        self.role[expired] = -1
        self.filler[expired] = -1
        self.cortical_role[expired] = -1
        self.activity[expired] = 0.0

    def bind(
        self,
        basin: int,
        role: int,
        filler: int,
        *,
        cortical_role: int | None = None,
        strength: float = 1.0,
    ) -> int:
        """Allocate sparse coincidence cells for one transient association."""
        if min(basin, role, filler) < 0:
            raise ValueError("binding identities must be non-negative")
        if basin > np.iinfo(self.basin.dtype).max:
            # Backward-compatible widening for checkpoints created before raw
            # sensory receptor codes were valid binder addresses.
            self.basin = self.basin.astype(np.int32)
        if not np.isfinite(strength) or strength <= 0.0:
            raise ValueError("binding strength must be finite and positive")
        cortical_role = role if cortical_role is None else int(cortical_role)
        if cortical_role < 0:
            raise ValueError("cortical role must be non-negative")
        self._advance()
        exact = np.flatnonzero(
            (self.basin == basin)
            & (self.role == role)
            & (self.filler == filler)
            & (self.cortical_role == cortical_role)
        )
        winners = list(exact[: self.cells_per_binding])
        if len(winners) < self.cells_per_binding:
            available = np.flatnonzero(self.filler < 0)
            needed = self.cells_per_binding - len(winners)
            winners.extend(int(value) for value in available[:needed])
        while len(winners) < self.cells_per_binding:
            occupied = np.asarray(
                [index for index in range(self.max_cells) if index not in winners],
                dtype=np.int64,
            )
            weakest = int(
                min(
                    occupied,
                    key=lambda index: (
                        float(self.activity[index]),
                        int(self.last_used[index]),
                        index,
                    ),
                )
            )
            winners.append(weakest)
            self.replacements += 1
        for winner in winners:
            self.basin[winner] = basin
            self.role[winner] = role
            self.filler[winner] = filler
            self.cortical_role[winner] = cortical_role
            self.activity[winner] = max(float(self.activity[winner]), strength)
            self.last_used[winner] = self.clock
        self.bindings += 1
        return len(winners)

    def recall(
        self,
        basin: int,
        role: int,
        *,
        exclude_filler: int | None = None,
    ) -> BindingRecall:
        """Complete a partial basin+role cue into one cortical filler."""
        if min(basin, role) < 0:
            return BindingRecall(None, None, 0.0, 0)
        matching = (self.basin == basin) & (self.role == role) & (self.filler >= 0)
        if exclude_filler is not None:
            matching &= self.filler != int(exclude_filler)
        cells = np.flatnonzero(matching)
        if not len(cells):
            return BindingRecall(None, None, 0.0, 0)
        votes: dict[tuple[int, int], float] = {}
        for cell in cells:
            content = int(self.filler[cell]), int(self.cortical_role[cell])
            votes[content] = votes.get(content, 0.0) + float(self.activity[cell])
        winner, confidence = max(
            votes.items(), key=lambda item: (item[1], -item[0][0], -item[0][1])
        )
        winner_filler, winner_role = winner
        winner_cells = cells[
            (self.filler[cells] == winner_filler)
            & (self.cortical_role[cells] == winner_role)
        ]
        self.activity[winner_cells] = np.maximum(self.activity[winner_cells], 1.0)
        self.last_used[winner_cells] = self.clock
        self.recalls += 1
        return BindingRecall(
            winner_filler, winner_role, confidence, int(len(winner_cells))
        )

    def address(self, filler: int) -> tuple[int, int] | None:
        """Follow reciprocal filler synapses back to the strongest address."""
        cells = np.flatnonzero(self.filler == int(filler))
        if not len(cells):
            return None
        winner = int(
            max(cells, key=lambda cell: (float(self.activity[cell]), -int(cell)))
        )
        return int(self.basin[winner]), int(self.role[winner])

    def reset_episode(self) -> None:
        self.basin.fill(-1)
        self.role.fill(-1)
        self.filler.fill(-1)
        self.cortical_role.fill(-1)
        self.activity.fill(0.0)
        self.last_used.fill(0)
        self.clock = 0
