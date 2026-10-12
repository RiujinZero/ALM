"""Deterministic raw-byte TinyStories/SmolTalk developmental streams."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

EPISODE_SEPARATOR = b"\n<|endoftext|>\n"


@dataclass(frozen=True)
class MixedStreamReport:
    output: str
    total_bytes: int
    tinystories_bytes: int
    smoltalk_bytes: int
    tinystories_episodes: int
    smoltalk_episodes: int
    smoltalk_train_episodes: int
    smoltalk_valid_episodes: int
    smoltalk_epochs_started: int
    seed: int
    output_sha256: str


def _stable_validation_member(raw_line: bytes, *, validation_fraction: float) -> bool:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    threshold = int(validation_fraction * (1 << 64))
    digest = hashlib.sha256(raw_line.rstrip(b"\r\n")).digest()
    return int.from_bytes(digest[:8], "big") < threshold


def load_smoltalk_episodes(
    path: Path, *, validation_fraction: float = 0.1
) -> tuple[list[bytes], list[bytes]]:
    """Extract raw conversational text and make a content-stable split."""
    train: list[bytes] = []
    valid: list[bytes] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on SmolTalk line {line_number}"
                ) from error
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing text on SmolTalk line {line_number}")
            episode = text.strip().encode("utf-8") + EPISODE_SEPARATOR
            destination = (
                valid
                if _stable_validation_member(
                    raw_line, validation_fraction=validation_fraction
                )
                else train
            )
            destination.append(episode)
    if not train or validation_fraction > 0.0 and not valid:
        raise ValueError("SmolTalk split produced an empty required partition")
    return train, valid


def iter_tinystories_episodes(path: Path) -> Iterator[bytes]:
    """Stream complete TinyStories without loading the 1.9GB corpus."""
    pending = bytearray()
    with path.open("rb") as handle:
        for line in handle:
            if line.strip() == b"<|endoftext|>":
                yield bytes(pending).rstrip() + EPISODE_SEPARATOR
                pending.clear()
            else:
                pending.extend(line)
    if pending:
        yield bytes(pending).rstrip() + EPISODE_SEPARATOR


def _cycled_smoltalk(
    episodes: list[bytes], *, seed: int
) -> Iterator[tuple[bytes, int]]:
    epoch = 0
    while True:
        order = list(range(len(episodes)))
        random.Random(seed + epoch).shuffle(order)
        for index in order:
            yield episodes[index], epoch
        epoch += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_interleaved_stream(
    tinystories_path: Path,
    smoltalk_path: Path,
    output_path: Path,
    *,
    total_bytes: int,
    seed: int = 0,
    validation_fraction: float = 0.1,
    smoltalk_valid_output: Path | None = None,
) -> MixedStreamReport:
    """Alternate whole episodes and stop at an exact matched byte budget."""
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    smol_train, smol_valid = load_smoltalk_episodes(
        smoltalk_path, validation_fraction=validation_fraction
    )
    if smoltalk_valid_output is not None:
        smoltalk_valid_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_valid = smoltalk_valid_output.with_suffix(
            smoltalk_valid_output.suffix + ".tmp"
        )
        with temporary_valid.open("wb") as handle:
            for episode in smol_valid:
                handle.write(episode)
        temporary_valid.replace(smoltalk_valid_output)

    tiny = iter_tinystories_episodes(tinystories_path)
    smol = _cycled_smoltalk(smol_train, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    written = tiny_bytes = smol_bytes = tiny_count = smol_count = 0
    highest_smol_epoch = -1
    turn = seed % 2
    with temporary.open("wb") as handle:
        while written < total_bytes:
            if turn == 0:
                try:
                    episode = next(tiny)
                except StopIteration:
                    tiny = iter_tinystories_episodes(tinystories_path)
                    episode = next(tiny)
                source = "tiny"
            else:
                episode, epoch = next(smol)
                highest_smol_epoch = max(highest_smol_epoch, epoch)
                source = "smol"
            payload = episode[: total_bytes - written]
            handle.write(payload)
            written += len(payload)
            if source == "tiny":
                tiny_bytes += len(payload)
                tiny_count += 1
            else:
                smol_bytes += len(payload)
                smol_count += 1
            turn = 1 - turn
    temporary.replace(output_path)
    return MixedStreamReport(
        output=str(output_path),
        total_bytes=written,
        tinystories_bytes=tiny_bytes,
        smoltalk_bytes=smol_bytes,
        tinystories_episodes=tiny_count,
        smoltalk_episodes=smol_count,
        smoltalk_train_episodes=len(smol_train),
        smoltalk_valid_episodes=len(smol_valid),
        smoltalk_epochs_started=highest_smol_epoch + 1,
        seed=seed,
        output_sha256=_sha256(output_path),
    )
