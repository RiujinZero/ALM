from __future__ import annotations

import numpy as np

from manifoldflow.experiments.dendritic_basin_echolocation_visualizer import (
    STATIC_FILE,
    VisualizerState,
    training_args,
)
from manifoldflow.experiments.dendritic_basin_three_organ_echolocation_smoke import (
    _continuous_episode,
    _fit_preview_head,
)


def test_visualizer_uses_production_three_organ_shape() -> None:
    args = training_args()
    assert args.max_body_bytes == 64_000_000
    assert args.columns == 1
    assert args.visual_capacity == 4 * args.auditory_capacity
    assert args.integration_capacity == args.visual_capacity


def test_visualizer_page_contains_requested_live_surfaces() -> None:
    page = STATIC_FILE.read_text(encoding="utf-8")
    assert 'id="world"' in page
    assert 'id="organism"' in page
    assert 'id="sound"' in page
    assert 'id="reset"' in page
    assert 'id="target"' in page
    assert '<option value="dead_reckoning">Dead reckoning</option>' in page
    assert '<option value="continuous">Continuous playground</option>' in page
    assert 'id="truthDot"' in page
    assert 'id="learning"' in page
    assert 'id="velocity"' in page
    assert "continuous_growth" in page
    assert "VISION · LEARNING" in page
    assert 'id="glossary"' in page
    assert 'id="faq"' in page
    assert "How does ALM learn without backpropagation?" in page
    for protected_detail in ("local-J", "hill steps", "nursery size", "threshold"):
        assert protected_detail not in page
    assert '<option value="0.1">0.1×</option>' in page
    assert '<option value="0.25" selected>0.25×</option>' in page
    assert "ALM prediction" in page
    assert "new EventSource('/events')" in page


def test_continuous_velocity_toggle_doubles_world_motion() -> None:
    normal = _continuous_episode(0.0, steps=3, velocity_scale=1.0)
    fast = _continuous_episode(0.0, steps=3, velocity_scale=2.0)
    normal_step = normal.positions[1] - normal.positions[0]
    fast_step = fast.positions[1] - fast.positions[0]
    assert np.isclose(fast_step, 2.0 * normal_step)


def test_state_replays_latest_organ_growth_to_new_subscriber() -> None:
    state = VisualizerState()
    state.publish({"type": "growth", "organ": "vision", "round": 1})
    state.publish({"type": "growth", "organ": "vision", "round": 2})
    state.publish({"type": "growth", "organ": "audition", "round": 1})
    subscriber = state.subscribe()
    events = [subscriber.get_nowait() for _ in range(3)]
    assert events[0] == {"type": "reset"}
    assert events[1]["organ"] == "vision"
    assert events[1]["round"] == 2
    assert events[2]["organ"] == "audition"


def test_state_reset_clears_retained_run_and_notifies_subscribers() -> None:
    state = VisualizerState()
    state.publish({"type": "complete", "report": {"seed": 0}})
    subscriber = state.subscribe()
    subscriber.get_nowait()
    subscriber.get_nowait()
    assert state.reset()
    assert subscriber.get_nowait() == {"type": "reset"}
    assert state.snapshot() == {"running": False, "events": []}


def test_state_retargets_with_the_retained_frozen_model() -> None:
    state = VisualizerState()
    state.set_predictor(
        lambda target, episode, learning, velocity: {
            "type": "target_update",
            "target": target,
            "episode_name": episode,
            "learning_enabled": learning,
            "velocity_scale": velocity,
        }
    )
    subscriber = state.subscribe()
    subscriber.get_nowait()
    assert state.retarget(0.25, "training", True)
    event = subscriber.get_nowait()
    assert event["target"] == 0.25
    assert event["episode_name"] == "training"
    assert event["learning_enabled"]
    assert event["velocity_scale"] == 1.0


def test_developmental_preview_head_tracks_a_moving_coordinate() -> None:
    position = np.linspace(-1.0, 1.0, 80, dtype=np.float32)
    activations = np.stack((position, position**2, np.sin(position)), axis=1)
    beta = _fit_preview_head(activations, position, np)
    prediction = activations @ beta[:-1] + beta[-1]
    assert np.mean((prediction - position) ** 2) < 1e-4
