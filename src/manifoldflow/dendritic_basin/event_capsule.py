"""Boundary-triggered episodic capsules with signed retrieval permission."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import exp, log2


@dataclass
class EventCapsule:
    cues: tuple[int, ...]
    items: tuple[tuple[int, int], ...]
    basin: int | None
    strength: float = 1.0
    last_used: int = 0
    created_at: int = 0
    last_recalled: int | None = None
    read_utility_sum: dict[tuple[int, int, int], float] = field(default_factory=dict)
    read_evidence: Counter[tuple[int, int, int]] = field(default_factory=Counter)
    frames: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = ()
    prediction_frames: tuple[tuple[int, tuple[tuple[int, float], ...]], ...] = ()


@dataclass(frozen=True)
class CapsuleRead:
    capsule: int
    query: int
    receptor: int
    items: tuple[tuple[int, int], ...]
    permission: float
    evidence: int
    trace_surprise: float = 0.0
    predictive_surprise: float = 0.0


class BoundaryEventCapsuleMemory:
    """Capture regime transitions first; discover relevance when queried."""

    def __init__(
        self,
        *,
        max_capsules: int = 256,
        items_per_capsule: int = 8,
        cues_per_capsule: int = 8,
        frames_per_capsule: int = 16,
        items_per_frame: int = 4,
        open_threshold: float = 2.0,
        close_threshold: float = 0.5,
        stable_steps: int = 3,
        frame_compaction: str = "edges",
        retention_decay: float = 0.9995,
        permission_decay: float = 0.99,
        minimum_read_age: int = 32,
        permission_context: str = "query_receptor",
    ) -> None:
        self.max_capsules = max_capsules
        self.items_per_capsule = items_per_capsule
        self.cues_per_capsule = cues_per_capsule
        self.frames_per_capsule = frames_per_capsule
        self.items_per_frame = items_per_frame
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.stable_steps = stable_steps
        if frame_compaction not in {"edges", "diverse"}:
            raise ValueError("unknown frame compaction mode")
        self.frame_compaction = frame_compaction
        self.retention_decay = retention_decay
        self.permission_decay = permission_decay
        self.minimum_read_age = minimum_read_age
        if permission_context not in {"query_receptor", "receptor"}:
            raise ValueError("unknown capsule permission context")
        self.permission_context = permission_context
        self.capsules: list[EventCapsule] = []
        self.permission_sum: dict[tuple[int, int], float] = {}
        self.permission_evidence: Counter[tuple[int, int]] = Counter()
        self.clock = 0
        self._open = False
        self._stable = 0
        self._previous_boundary_signal = 0.0
        self._cues: list[int] = []
        self._items: dict[tuple[int, int], float] = {}
        self._frames: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        self._prediction_frames: list[tuple[int, tuple[tuple[int, float], ...]]] = []
        self._basin: int | None = None
        self.captures = 0
        self.replacements = 0
        self.shadow_reads = 0
        self.trusted_reads = 0

    def permission(self, query: int, receptor: int) -> float:
        key = self._permission_key(query, receptor)
        return self.permission_sum.get(key, 0.0) / (self.permission_evidence[key] + 1.0)

    def evidence(self, query: int, receptor: int) -> int:
        return self.permission_evidence[self._permission_key(query, receptor)]

    def _permission_key(self, query: int, receptor: int) -> tuple[int, int]:
        return (
            (query, receptor)
            if self.permission_context == "query_receptor"
            else (0, receptor)
        )

    def _finish_event(self) -> None:
        if not self._items or not self._cues:
            self._clear_open()
            return
        items = tuple(
            key
            for key, _ in sorted(
                self._items.items(), key=lambda item: (-item[1], item[0])
            )[: self.items_per_capsule]
        )
        cues = tuple(dict.fromkeys(self._cues))
        if len(cues) > self.cues_per_capsule:
            edge = self.cues_per_capsule // 2
            cues = cues[:edge] + cues[-(self.cues_per_capsule - edge) :]
        frames = tuple(self._frames)
        if len(frames) > self.frames_per_capsule:
            if self.frame_compaction == "diverse":
                frames = self._diverse_frames(frames)
            else:
                edge = self.frames_per_capsule // 2
                frames = frames[:edge] + frames[-(self.frames_per_capsule - edge) :]
        capsule = EventCapsule(
            cues,
            items,
            self._basin,
            last_used=self.clock,
            created_at=self.clock,
            frames=frames,
            prediction_frames=tuple(self._prediction_frames),
        )
        if len(self.capsules) < self.max_capsules:
            self.capsules.append(capsule)
        else:
            weakest = min(
                range(len(self.capsules)),
                key=lambda index: (
                    self.capsules[index].strength,
                    self.capsules[index].last_used,
                ),
            )
            self.capsules[weakest] = capsule
            self.replacements += 1
        self.captures += 1
        self._clear_open()

    def _diverse_frames(
        self,
        frames: tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
    ) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
        """Keep local states that add distinct cortical assembly activity."""
        selected = {0, len(frames) - 1}
        covered = set(frames[0][1]) | set(frames[-1][1])
        while len(selected) < self.frames_per_capsule:
            candidates = [
                index for index in range(len(frames)) if index not in selected
            ]
            if not candidates:
                break
            winner = max(
                candidates,
                key=lambda index: (
                    len(set(frames[index][1]) - covered),
                    min(index, len(frames) - 1 - index),
                    -index,
                ),
            )
            selected.add(winner)
            covered.update(frames[winner][1])
        return tuple(frames[index] for index in sorted(selected))

    def _clear_open(self) -> None:
        self._open = False
        self._stable = 0
        self._cues.clear()
        self._items.clear()
        self._frames.clear()
        self._prediction_frames.clear()
        self._basin = None

    def observe_boundary(
        self,
        signal: float,
        *,
        query: int | None,
        active_items: tuple[tuple[int, int, float], ...],
        basin: int | None,
        prediction_fingerprint: tuple[tuple[int, float], ...] = (),
    ) -> bool:
        """Open on change, accumulate locally active cortex, close on stability."""
        self.clock += 1
        for capsule in self.capsules:
            capsule.strength *= self.retention_decay
        derivative = signal - self._previous_boundary_signal
        self._previous_boundary_signal = signal
        if not self._open and signal >= self.open_threshold and derivative > 0.0:
            self._open = True
            self._basin = basin
        if not self._open:
            return False
        if query is not None:
            self._cues.append(query)
            frame_items = tuple(
                (assembly, receptor)
                for receptor, assembly, _ in sorted(
                    active_items, key=lambda item: (-item[2], item[0], item[1])
                )[: self.items_per_frame]
            )
            self._frames.append((query, frame_items))
            if prediction_fingerprint:
                self._prediction_frames.append((query, prediction_fingerprint))
        for receptor, assembly, salience in active_items:
            key = assembly, receptor
            self._items[key] = max(self._items.get(key, 0.0), salience)
        self._stable = self._stable + 1 if signal <= self.close_threshold else 0
        if self._stable < self.stable_steps:
            return False
        self._finish_event()
        return True

    def candidate(
        self,
        query: int,
        basin: int | None,
        *,
        permission_query: int | None = None,
        context_items: tuple[tuple[int, int], ...] = (),
        context_fingerprint: tuple[tuple[int, float], ...] = (),
    ) -> CapsuleRead | None:
        candidates = self.candidates(
            query,
            basin,
            permission_query=permission_query,
            context_items=context_items,
            context_fingerprint=context_fingerprint,
            limit=1,
        )
        return candidates[0] if candidates else None

    def candidates(
        self,
        query: int,
        basin: int | None,
        *,
        permission_query: int | None = None,
        context_items: tuple[tuple[int, int], ...] = (),
        context_fingerprint: tuple[tuple[int, float], ...] = (),
        limit: int = 8,
    ) -> tuple[CapsuleRead, ...]:
        matches = [
            (index, capsule)
            for index, capsule in enumerate(self.capsules)
            if query in capsule.cues
            and capsule.items
            and self.clock - capsule.last_used >= self.minimum_read_age
        ]
        if not matches:
            return ()
        permission_query = query if permission_query is None else permission_query

        def trace_surprise(capsule: EventCapsule) -> float:
            observed = set(context_items)
            matches = [set(items) for cue, items in capsule.frames if cue == query]
            if not observed or not matches:
                return 1.0
            return min(
                1.0 - len(observed & stored) / max(1, len(observed | stored))
                for stored in matches
            )

        def predictive_surprise(capsule: EventCapsule) -> float:
            current = dict(context_fingerprint)
            matches = [
                dict(fingerprint)
                for cue, fingerprint in capsule.prediction_frames
                if cue == query
            ]
            if not current or not matches:
                return 40.0
            return min(
                -log2(
                    max(
                        sum(
                            min(probability, current.get(byte_id, 0.0))
                            for byte_id, probability in stored.items()
                        ),
                        1e-12,
                    )
                )
                for stored in matches
            )

        def score(
            pair: tuple[int, EventCapsule],
        ) -> tuple[float, float, float, float, int]:
            index, capsule = pair
            receptor = capsule.items[0][1]
            context = 1.05 if basin is not None and capsule.basin == basin else 1.0
            return (
                -predictive_surprise(capsule),
                -trace_surprise(capsule),
                context * self.permission(permission_query, receptor),
                capsule.strength,
                -index,
            )

        selected = sorted(matches, key=score, reverse=True)[:limit]
        return tuple(
            CapsuleRead(
                index,
                permission_query,
                capsule.items[0][1],
                capsule.items,
                self.permission(permission_query, capsule.items[0][1]),
                self.evidence(permission_query, capsule.items[0][1]),
                trace_surprise(capsule),
                predictive_surprise(capsule),
            )
            for index, capsule in selected
        )

    def credit(self, read: CapsuleRead, utility_bits: float) -> None:
        """Credit only queried memories; signed utility also controls retention."""
        key = self._permission_key(read.query, read.receptor)
        self.permission_sum[key] = (
            self.permission_decay * self.permission_sum.get(key, 0.0) + utility_bits
        )
        self.permission_evidence[key] += 1
        capsule = self.capsules[read.capsule]
        capsule.strength = max(
            0.0,
            self.retention_decay * capsule.strength + utility_bits,
        )
        capsule.last_used = self.clock
        self.shadow_reads += 1

    def branch_value(self, read: CapsuleRead) -> float:
        """Return local surprise-reduction value for this exact memory branch."""
        assembly, receptor = read.items[0]
        capsule = self.capsules[read.capsule]
        key = read.query, assembly, receptor
        return capsule.read_utility_sum.get(key, 0.0) / (
            capsule.read_evidence[key] + 1.0
        )

    def branch_evidence(self, read: CapsuleRead) -> int:
        assembly, receptor = read.items[0]
        capsule = self.capsules[read.capsule]
        return capsule.read_evidence[(read.query, assembly, receptor)]

    def credit_branch(self, read: CapsuleRead, utility_bits: float) -> None:
        """Apply the cortical surprise comparison to one episodic branch."""
        assembly, receptor = read.items[0]
        capsule = self.capsules[read.capsule]
        key = read.query, assembly, receptor
        capsule.read_utility_sum[key] = (
            self.permission_decay * capsule.read_utility_sum.get(key, 0.0)
            + utility_bits
        )
        capsule.read_evidence[key] += 1
        capsule.strength = max(
            0.0,
            self.retention_decay * capsule.strength + utility_bits,
        )
        capsule.last_used = self.clock
        self.shadow_reads += 1

    def touch(self, read: CapsuleRead) -> None:
        """Apply retrieval refractory timing without assigning usefulness."""
        self.capsules[read.capsule].last_used = self.clock

    def reinstate(self, read: CapsuleRead) -> None:
        """Record a recall that was permitted to influence the active cortex."""
        capsule = self.capsules[read.capsule]
        capsule.last_used = self.clock
        capsule.last_recalled = self.clock

    def permitted(self, read: CapsuleRead, *, minimum_evidence: int = 4) -> bool:
        return read.evidence >= minimum_evidence and read.permission > 0.0

    def confidence(self, read: CapsuleRead) -> float:
        return 1.0 / (1.0 + exp(-read.permission))

    def reset_episode(self) -> None:
        self._clear_open()
        self._previous_boundary_signal = 0.0
