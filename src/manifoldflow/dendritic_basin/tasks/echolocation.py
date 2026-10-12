"""Raw-vector echolocation world for the continuous ALM smoke."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

DOT_POSITION, TARGET_POSITION, VISIBLE, RANGE_BEEP, CROSSING_CHIME = range(5)
N_CHANNELS = 5
CHANNEL_NAMES = (
    "dot_position",
    "target_position",
    "visible",
    "range_beep",
    "crossing_chime",
)
AudioMode = Literal["real", "silent"]


@dataclass(frozen=True)
class EcholocationEpisode:
    positions: np.ndarray
    target_position: float
    range_beeps: np.ndarray
    crossing_chimes: np.ndarray
    blackout_start: int

    @property
    def final_position(self) -> float:
        return float(self.positions[-1])


def generate_episode(
    rng: np.random.Generator,
    *,
    steps: int = 40,
    hidden_steps: int = 12,
    velocity_wander: float = 0.012,
) -> EcholocationEpisode:
    """Generate motion and event-only ranging sound without latent shortcuts."""
    if steps < 4:
        raise ValueError("steps must be at least 4")
    if not 1 <= hidden_steps < steps - 1:
        raise ValueError("hidden_steps must be in [1, steps - 2]")
    if velocity_wander < 0.0:
        raise ValueError("velocity_wander cannot be negative")

    target = float(rng.uniform(-0.30, 0.45))
    position = float(max(-0.98, target - rng.uniform(0.55, 0.95)))
    velocity = float(rng.uniform(0.030, 0.055))
    positions = np.empty(steps, dtype=np.float32)
    for index in range(steps):
        velocity = float(
            np.clip(
                0.82 * velocity
                + 0.18 * rng.uniform(0.025, 0.065)
                + rng.normal(0.0, velocity_wander),
                0.008,
                0.085,
            )
        )
        position = float(np.clip(position + velocity, -1.0, 1.0))
        positions[index] = position

    beeps = np.zeros(steps, dtype=np.float32)
    next_beep = 0
    for index, value in enumerate(positions):
        if index >= next_beep:
            beeps[index] = 1.0
            distance = min(1.0, abs(float(value) - target) / 1.25)
            next_beep = index + max(2, int(round(2.0 + 10.0 * distance)))

    crossing = np.zeros(steps, dtype=np.float32)
    signed = positions - target
    indices = np.flatnonzero((signed[:-1] < 0.0) & (signed[1:] >= 0.0))
    if len(indices):
        crossing[int(indices[0]) + 1] = 1.0
    return EcholocationEpisode(positions, target, beeps, crossing, steps - hidden_steps)


def generate_episodes(
    count: int,
    *,
    seed: int,
    steps: int = 40,
    hidden_steps: int = 12,
    velocity_wander: float = 0.012,
) -> list[EcholocationEpisode]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return [
        generate_episode(
            rng, steps=steps, hidden_steps=hidden_steps, velocity_wander=velocity_wander
        )
        for _ in range(count)
    ]


def retarget_episode(
    episode: EcholocationEpisode, target_position: float
) -> EcholocationEpisode:
    """Rebuild only the event soundscape around a new target location."""
    target = float(np.clip(target_position, -0.95, 0.95))
    beeps = np.zeros_like(episode.range_beeps)
    next_beep = 0
    for index, value in enumerate(episode.positions):
        if index >= next_beep:
            beeps[index] = 1.0
            distance = min(1.0, abs(float(value) - target) / 1.25)
            next_beep = index + max(2, int(round(2.0 + 10.0 * distance)))
    crossing = np.zeros_like(episode.crossing_chimes)
    signed = episode.positions - target
    indices = np.flatnonzero(
        ((signed[:-1] < 0.0) & (signed[1:] >= 0.0))
        | ((signed[:-1] > 0.0) & (signed[1:] <= 0.0))
    )
    crossing[indices + 1] = 1.0
    return replace(
        episode,
        target_position=target,
        range_beeps=beeps,
        crossing_chimes=crossing,
    )


def shuffled_ranging_episodes(
    episodes: list[EcholocationEpisode], *, seed: int
) -> list[EcholocationEpisode]:
    """Reassign range-event timelines while preserving all other state."""
    if len(episodes) < 2:
        return list(episodes)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(episodes))
    if np.any(permutation == np.arange(len(episodes))):
        permutation = np.roll(np.arange(len(episodes)), 1)
    return [
        replace(ep, range_beeps=episodes[int(src)].range_beeps.copy())
        for ep, src in zip(episodes, permutation, strict=True)
    ]


def episode_vectors(
    episode: EcholocationEpisode,
    *,
    visible: bool,
    audio_mode: AudioMode = "real",
) -> np.ndarray:
    """Return raw ``[time, channel]`` sensory vectors."""
    vectors = np.zeros((len(episode.positions), N_CHANNELS), dtype=np.float32)
    vectors[:, VISIBLE] = 1.0
    vectors[:, DOT_POSITION] = episode.positions
    vectors[:, TARGET_POSITION] = episode.target_position
    if not visible:
        vectors[episode.blackout_start :, :3] = 0.0
    if audio_mode == "real":
        vectors[:, RANGE_BEEP] = episode.range_beeps
        vectors[:, CROSSING_CHIME] = episode.crossing_chimes
    return vectors


def stack_episode_vectors(
    episodes: list[EcholocationEpisode],
    *,
    visible: bool,
    audio_mode: AudioMode = "real",
) -> np.ndarray:
    return np.stack(
        [episode_vectors(ep, visible=visible, audio_mode=audio_mode) for ep in episodes]
    )


def dead_reckoning_prediction(episode: EcholocationEpisode) -> float:
    stop = episode.blackout_start
    last = float(episode.positions[stop - 1])
    velocity = last - float(episode.positions[stop - 2])
    return float(np.clip(last + (len(episode.positions) - stop) * velocity, -1.0, 1.0))
