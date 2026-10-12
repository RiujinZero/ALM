"""capacity_estimate.py: validated against direct measurement, not just math.

The whole point of this module is replacing empirical floor-hunting with a
formula -- so the formula itself has to be checked against real
NativeGrowableCortex measurements, not just internally consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from manifoldflow.dendritic_basin.capacity_estimate import (
    estimate_body_bytes,
    fixed_floor_bytes,
    max_nursery_capacity,
)

native_cortex = pytest.importorskip("manifoldflow.dendritic_basin.native_cortex")

pytestmark = pytest.mark.skipif(
    not native_cortex.native_available(),
    reason="alm_native shared library is not built (see cpp/README.md)",
)


@pytest.mark.parametrize(
    "max_context_bytes,measured_cold_bytes",
    [(2, 5923), (4, 5971), (6, 6019), (8, 6067)],
)
def test_fixed_floor_matches_cold_measurement(
    max_context_bytes: int, measured_cold_bytes: int
) -> None:
    predicted = fixed_floor_bytes(max_context_bytes)
    assert predicted == measured_cold_bytes


def test_fixed_floor_matches_a_fresh_cortex_directly() -> None:
    for max_context_bytes in (2, 4, 6, 8):
        cortex = native_cortex.NativeGrowableCortex(max_context_bytes=max_context_bytes)
        assert cortex.embodied_bytes == fixed_floor_bytes(max_context_bytes)


def test_nursery_capacity_matches_measured_saturation_point() -> None:
    # Measured 2026-09-11: max_context_bytes=6 (4 banks) with
    # max_sequence_nursery_paths=2_000_000 saturates nursery fill at
    # exactly 8_000_000 (4 * 2_000_000).
    assert max_nursery_capacity(max_context_bytes=6, max_sequence_nursery_paths=2_000_000) == 8_000_000


def test_estimate_is_within_measured_tolerance_while_nursery_fills() -> None:
    """Real measurement: train with confirmation thresholds high enough to
    suppress mature promotion, so growth is (almost) pure nursery fill, and
    check the estimate stays within a modest relative-error band of the
    actual embodied_bytes at several fill levels below the nursery cap.
    """
    rng = np.random.default_rng(0)
    cortex = native_cortex.NativeGrowableCortex(
        max_context_bytes=6,
        max_sequence_nursery_paths=2_000_000,
        confirmations=100,
        sequence_confirmations=100,
    )
    for _ in range(6):
        stream = rng.integers(0, 256, size=200_000, dtype=np.uint8)
        cortex.observe_chunk_probabilities(stream, learn=True)
        nursery_fill = cortex._stat(3)  # _STAT_SEQUENCE_NURSERY
        if nursery_fill == 0:
            continue
        estimate = estimate_body_bytes(max_context_bytes=6, nursery_fill=nursery_fill)
        actual = cortex.embodied_bytes
        relative_error = abs(estimate.estimated_bytes - actual) / actual
        assert relative_error < 0.05, (
            f"estimate {estimate.estimated_bytes} vs actual {actual} "
            f"at nursery_fill={nursery_fill}: {relative_error:.3%} off"
        )


def test_estimate_rejects_nursery_fill_above_configured_cap() -> None:
    with pytest.raises(ValueError, match="exceeds the configured cap"):
        estimate_body_bytes(
            max_context_bytes=6,
            nursery_fill=10_000_000,
            max_sequence_nursery_paths=2_000_000,
        )
