from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_latest_ring_dot_smoke import (
    generate_episode,
    render_ring_frame,
)


def test_episode_has_constant_direction_and_extra_target() -> None:
    positions, direction = generate_episode(np.random.default_rng(7), 8, 12)
    assert len(positions) == 13
    assert direction in (-1, 1)
    assert np.all((positions[1:] - positions[:-1]) % 8 == direction % 8)


def test_rendered_frame_is_position_specific_but_direction_free() -> None:
    frames = [np.asarray(render_ring_frame(position, 8, 48)) for position in range(8)]
    assert all(frame.shape == (48, 48, 3) for frame in frames)
    assert all(not np.array_equal(frames[0], frame) for frame in frames[1:])
