"""Leakage-resistant raw-vector modular-arithmetic datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModularSplit:
    inputs: np.ndarray
    residues: np.ndarray
    moduli: np.ndarray
    operands: np.ndarray


@dataclass(frozen=True)
class ModularLadder:
    train: ModularSplit
    candidate: ModularSplit
    held_out_pairs: ModularSplit
    unseen_modulus: ModularSplit
    seen_moduli: tuple[int, ...]
    unseen_modulus_value: int
    operand_limit: int


def modular_expression(
    a: int,
    b: int,
    n: int,
    residue: int | None = None,
    *,
    mask_fields: frozenset[str] = frozenset(),
) -> bytes:
    """Render the conventional byte structure used by the expression control."""
    unknown = mask_fields - {"a", "b", "n"}
    if unknown:
        raise ValueError(f"unknown modular-expression fields: {sorted(unknown)}")

    def rendered(name: str, value: int) -> str:
        text = str(value)
        return "?" * len(text) if name in mask_fields else text

    prefix = (
        f"{rendered('a', a)} + {rendered('b', b)} "
        f"mod {rendered('n', n)} = "
    )
    if residue is None:
        return prefix.encode("ascii")
    return f"{prefix}{residue}\n".encode("ascii")


def unfolded_modular_lesson(a: int, b: int, n: int, residue: int) -> bytes:
    """Expose addition as movement through a finite cyclic world."""
    if n < 2 or min(a, b) < 0:
        raise ValueError("lesson operands must be non-negative and modulus >= 2")
    start = a % n
    positions = [str((start + step) % n) for step in range(b + 1)]
    trajectory = " -> ".join(positions)
    equation = modular_expression(a, b, n, residue).decode("ascii").rstrip("\n")
    return (
        f"ring {n}; start {a} at {start}; add {b}: {trajectory}; {equation}\n"
    ).encode("ascii")


def modular_trajectory_prefix(
    a: int, b: int, n: int, completed_steps: int
) -> bytes:
    """Return an unfolded lesson prefix immediately before its next position."""
    if not 0 <= completed_steps < b:
        raise ValueError("completed_steps must identify a non-final trajectory state")
    start = a % n
    positions = [str((start + step) % n) for step in range(completed_steps + 1)]
    trajectory = " -> ".join(positions)
    return f"ring {n}; start {a} at {start}; add {b}: {trajectory} -> ".encode(
        "ascii"
    )


def _partition_code(a: int, b: int, n: int, seed: int) -> int:
    """Stable combination-level partition; no outcome enters the split rule."""
    value = (
        (a + 1) * 0x9E3779B1
        ^ (b + 1) * 0x85EBCA77
        ^ (n + 1) * 0xC2B2AE3D
        ^ (seed + 1) * 0x27D4EB2F
    )
    value ^= value >> 16
    return int(value % 20)


def _make_split(
    rows: list[tuple[int, int, int]], *, operand_limit: int, max_modulus: int
) -> ModularSplit:
    operands = np.asarray([(a, b) for a, b, _n in rows], dtype=np.int16)
    moduli = np.asarray([n for _a, _b, n in rows], dtype=np.int16)
    residues = np.asarray([(a + b) % n for a, b, n in rows], dtype=np.int16)
    # Fixed range scaling is sensory calibration, not an engineered arithmetic
    # feature: the learner still receives only the three scalar coordinates.
    inputs = np.column_stack(
        (
            operands[:, 0] / max(operand_limit - 1, 1),
            operands[:, 1] / max(operand_limit - 1, 1),
            moduli / max_modulus,
        )
    ).astype(np.float32)
    return ModularSplit(inputs, residues, moduli, operands)


def build_modular_ladder(
    *,
    seen_moduli: tuple[int, ...] = (2, 3, 5, 7),
    unseen_modulus: int = 11,
    operand_limit: int = 32,
    seed: int = 0,
    partition_by_operand_pair: bool = False,
) -> ModularLadder:
    """Build disjoint selection and generalization partitions.

    Every operand value occurs during training, but particular ``(a, b, n)``
    combinations are held out. The unseen-modulus split contains no example of
    its modulus in either training or candidate selection.
    """
    if operand_limit < 2:
        raise ValueError("operand_limit must be at least two")
    if len(set(seen_moduli)) != len(seen_moduli):
        raise ValueError("seen_moduli must be unique")
    if any(n < 2 for n in (*seen_moduli, unseen_modulus)):
        raise ValueError("moduli must be at least two")
    if unseen_modulus in seen_moduli:
        raise ValueError("unseen modulus must not occur among seen moduli")

    train_rows: list[tuple[int, int, int]] = []
    candidate_rows: list[tuple[int, int, int]] = []
    held_out_rows: list[tuple[int, int, int]] = []
    for n in seen_moduli:
        for a in range(operand_limit):
            for b in range(operand_limit):
                row = (a, b, n)
                partition_modulus = 0 if partition_by_operand_pair else n
                code = _partition_code(a, b, partition_modulus, seed)
                if code < 14:
                    train_rows.append(row)
                elif code < 17:
                    candidate_rows.append(row)
                else:
                    held_out_rows.append(row)
    unseen_rows = [
        (a, b, unseen_modulus)
        for a in range(operand_limit)
        for b in range(operand_limit)
    ]
    max_modulus = max(*seen_moduli, unseen_modulus)
    return ModularLadder(
        train=_make_split(
            train_rows, operand_limit=operand_limit, max_modulus=max_modulus
        ),
        candidate=_make_split(
            candidate_rows, operand_limit=operand_limit, max_modulus=max_modulus
        ),
        held_out_pairs=_make_split(
            held_out_rows, operand_limit=operand_limit, max_modulus=max_modulus
        ),
        unseen_modulus=_make_split(
            unseen_rows, operand_limit=operand_limit, max_modulus=max_modulus
        ),
        seen_moduli=seen_moduli,
        unseen_modulus_value=unseen_modulus,
        operand_limit=operand_limit,
    )
