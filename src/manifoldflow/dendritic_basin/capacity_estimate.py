"""Closed-form body-size estimates, replacing empirical floor-hunting.

Ahmad & Hawkins (2016, arXiv:1601.00720) derive a neuron's sparse-coincidence
capacity from population size, sparsity, and dendritic threshold, so HTM's
own parameters (2048 columns, 32 cells, 40 active, threshold 15) are chosen
from that math rather than trial and error. Their formula itself doesn't
transfer directly: it's a false-positive rate for *fuzzy, random-subset*
synapse-overlap matching, and ALM's context/sequence tables use *exact*
packed-integer keys -- there is no meaningful collision/false-positive rate
to estimate there (see the docstring on ``estimate_body_bytes`` for what
does and doesn't carry over).

What *does* transfer is the underlying discipline: derive a number from
config instead of measuring it by trial and error. This session hit
``ValueError: fixed cortical anatomy exceeds the requested byte budget``
repeatedly while picking test/experiment byte budgets, and resolved it each
time by empirically probing (deepcopy-and-clear, or pruning to an absurd
target and reading the resulting error) rather than computing it. This
module replaces that probing for the common case with two closed-form
pieces, each validated against real measurements below:

1. ``fixed_floor_bytes`` -- the serialized size of a cold (zero-data)
   ``NativeGrowableCortex`` at a given ``max_context_bytes``. Confirmed
   linear in ``max_context_bytes`` (re-measured 2026-09-11 after the
   burst_surprise_signal/long_range_bank checkpoint fields were added --
   the constant term shifts whenever the checkpoint format grows, the
   slope does not): 5923, 5971, 6019, 6067 bytes at max_context_bytes
   2, 4, 6, 8 -- exactly +24 bytes per unit of max_context_bytes,
   independent of ``max_sequence_nursery_paths`` (nursery is an empty dict
   until something is actually observed, so its *cap* costs nothing before
   it has entries).
2. ``estimate_body_bytes`` -- adds the nursery's own marginal cost. Measured
   by training a cortex with confirmation thresholds high enough to
   suppress mature promotion, so growth is (almost) purely nursery fill:
   marginal cost converges to ~29.5 bytes/entry as the nursery fills toward
   its cap, from a base of 6019 bytes cold (at max_context_bytes=6). This
   is the exact regime that caused this session's budget-calibration
   failures (a still-growing nursery, not yet pruned) -- so it's also the
   regime this estimate is built and validated for.

**What this does not capture:** once nursery entries promote to mature
sequence/context structure at real scale (not artificially suppressed),
body size grows further in a way this two-term model does not track --
mature-structure growth depends on actual data statistics (how many
distinct contexts/transitions the corpus contains), not just config. Treat
``estimate_body_bytes``'s output as a **lower bound** once training has run
long enough for real promotion to occur, not an exact prediction -- the
same honest framing Ahmad & Hawkins give their own bound (an estimate of
capacity under stated assumptions, not a guarantee for every trained
instance).
"""

from __future__ import annotations

from dataclasses import dataclass

_COLD_BASE_BYTES = 6019
_COLD_BASE_MAX_CONTEXT_BYTES = 6
_BYTES_PER_MAX_CONTEXT_BYTES_UNIT = 24
_NURSERY_BYTES_PER_ENTRY = 29.5


def fixed_floor_bytes(max_context_bytes: int = 6) -> int:
    """The serialized size of a cold (zero-data) native cortex.

    Validated against direct measurement at max_context_bytes in
    {2, 4, 6, 8}; linear in between and beyond is an extrapolation, not
    separately confirmed at every value.
    """
    if max_context_bytes < 2 or max_context_bytes > 8:
        raise ValueError("max_context_bytes must be in [2, 8]")
    return _COLD_BASE_BYTES + _BYTES_PER_MAX_CONTEXT_BYTES_UNIT * (
        max_context_bytes - _COLD_BASE_MAX_CONTEXT_BYTES
    )


def max_nursery_capacity(max_context_bytes: int, max_sequence_nursery_paths: int) -> int:
    """Total nursery slots across all banks (banks = max_context_bytes - 2)."""
    banks = max(0, max_context_bytes - 2)
    return banks * max_sequence_nursery_paths


@dataclass(frozen=True)
class BodyEstimate:
    fixed_floor_bytes: int
    nursery_contribution_bytes: int
    estimated_bytes: int
    is_lower_bound: bool


def estimate_body_bytes(
    *,
    max_context_bytes: int = 6,
    nursery_fill: int,
    max_sequence_nursery_paths: int | None = None,
) -> BodyEstimate:
    """Fixed floor plus nursery-fill contribution.

    ``nursery_fill`` is the current (or expected) total live nursery entry
    count across all banks -- the quantity that was previously only
    discoverable by running training and reading it back. If
    ``max_sequence_nursery_paths`` is given, ``nursery_fill`` above that
    bank-scaled cap is treated as a config error (nursery cannot exceed its
    own cap) rather than silently estimated.

    ``is_lower_bound`` is True whenever any promotion to mature structure is
    plausible at this fill level -- practically, always: this model has no
    way to know how much mature structure a real corpus would have produced
    by this point, so callers should treat this as a floor to size a budget
    *above*, not an exact prediction to size a budget *at*.
    """
    if nursery_fill < 0:
        raise ValueError("nursery_fill must be non-negative")
    if max_sequence_nursery_paths is not None:
        cap = max_nursery_capacity(max_context_bytes, max_sequence_nursery_paths)
        if nursery_fill > cap:
            raise ValueError(
                f"nursery_fill ({nursery_fill}) exceeds the configured cap "
                f"({cap}) for max_context_bytes={max_context_bytes}, "
                f"max_sequence_nursery_paths={max_sequence_nursery_paths}"
            )
    floor = fixed_floor_bytes(max_context_bytes)
    nursery_bytes = round(nursery_fill * _NURSERY_BYTES_PER_ENTRY)
    return BodyEstimate(
        fixed_floor_bytes=floor,
        nursery_contribution_bytes=nursery_bytes,
        estimated_bytes=floor + nursery_bytes,
        is_lower_bound=True,
    )
