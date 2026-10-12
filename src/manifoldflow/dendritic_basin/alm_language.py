"""A generic, reusable training/generation core for ALM as a language model.

Configure the mechanism, hand it raw bytes to train on, and call
``generate`` with a cue to get text back. ALM has no separate train/
inference split and no backprop: every ``train()`` call develops the
organism's anatomy directly from the bytes it sees, and no GPU is needed
at any point (see cpp/README.md for the native backend's measured
speedups, or run in pure Python without building anything).

Parsing/tokenization is deliberately *not* part of this core: ALM operates
on raw bytes (UTF-8 text, or anything else), so turning your source data
into a byte stream -- reading files, stripping markup, deciding how to
chunk a corpus -- is up to you. This module only needs ``np.uint8`` arrays
(bytes/bytearray/str are accepted too and converted automatically).

Quick start::

    from manifoldflow.dendritic_basin.alm_language import (
        ALMLanguageConfig, ALMLanguageModel,
    )

    model = ALMLanguageModel(ALMLanguageConfig(native=False))  # pure Python
    model.train(open("corpus.txt", "rb").read())
    print(model.generate("Once upon a time"))

Two training modes, both driven by the same config:

- ``columns=1`` (the default): a single in-process cortex trains directly
  on the stream, sleeping (byte-budget pruning) every ``sleep_bytes``.
  Simple, no subprocess/pickling overhead -- the right default for
  in-memory experimentation.
- ``columns>1``: the same parallel wake/sleep/consolidate cycle the real
  production harness (dendritic_basin_phase126_three_epoch_generation.py)
  uses -- ``columns`` parallel wake columns each develop a disjoint slice
  of every ``sleep_bytes``-sized round via a process pool, merge back via
  ``consolidate_delta`` (ancestry-safe: only what each column learned past
  the shared ancestor), then prune to ``max_body_bytes`` via
  ``consolidate_to_byte_budget``. Requires ``native=True`` (that merge
  path is only meant for the bit-exact native backend at real scale).
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ALMLanguageConfig:
    """Every knob shared by both backends (Python reference and the native
    C++ port), plus the orchestration knobs for multi-column training.

    See opponent_byte_stream.py's SequenceCortexOpponentLearner and
    native_cortex.py's NativeGrowableCortex for what each learner-level
    field actually does; the defaults here match theirs.
    """

    # Core learner hyperparameters.
    prior: float = 0.1
    confirmations: int = 3
    context_confirmations: int = 3
    surprise_bits: float = 2.0
    basal_strength: float = 4.0
    max_context_bytes: int = 6
    sequence_confirmations: int = 3
    parent_strength: float = 16.0
    max_sequence_nursery_paths: int = 50_000

    # Contextual competition / consequence-sensitive learning (opt-in).
    competition_exponent: float = 1.0
    adaptive_competition: bool = False
    trust_gamma: float = 0.5
    trust_learning_rate: float = 0.2
    attribution_gate_threshold: float = 0.5

    # Sleep-time pruning-priority mechanism (opt-in). Validated to beat
    # count-based pruning at matched capacity on both TinyStories and
    # Wikipedia -- see tests/test_dendritic_basin_connectivity_pruning.py
    # and docs/ALM_V1_Architecture.md. Does not change growth, promotion,
    # or the probability math, and costs nothing extra in memory.
    connectivity_pruning: bool = False

    # Body-size ceiling and wake/sleep orchestration.
    max_body_bytes: int = 64_000_000
    sleep_bytes: int = 256_000_000
    columns: int = 1
    chunk_bytes: int = 4_000_000

    # Backend choice. native=True requires the alm_native shared library
    # (see cpp/README.md); native=False always works, in pure Python.
    native: bool = True

    def __post_init__(self) -> None:
        if self.columns < 1:
            raise ValueError("columns must be positive")
        if self.columns > 1 and not self.native:
            raise ValueError(
                "columns > 1 requires native=True -- the multi-column "
                "consolidate_delta merge path is only exercised (and only "
                "makes sense performance-wise) for the native backend"
            )
        if self.sleep_bytes < self.columns:
            raise ValueError("sleep_bytes must provide work to every column")
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")

    def learner_kwargs(self) -> dict[str, Any]:
        """The subset of fields SequenceCortexOpponentLearner/
        NativeGrowableCortex's shared constructor signature accepts."""
        return {
            "prior": self.prior,
            "confirmations": self.confirmations,
            "context_confirmations": self.context_confirmations,
            "surprise_bits": self.surprise_bits,
            "basal_strength": self.basal_strength,
            "max_context_bytes": self.max_context_bytes,
            "sequence_confirmations": self.sequence_confirmations,
            "parent_strength": self.parent_strength,
            "max_sequence_nursery_paths": self.max_sequence_nursery_paths,
            "competition_exponent": self.competition_exponent,
            "adaptive_competition": self.adaptive_competition,
            "trust_gamma": self.trust_gamma,
            "trust_learning_rate": self.trust_learning_rate,
            "attribution_gate_threshold": self.attribution_gate_threshold,
            "connectivity_pruning": self.connectivity_pruning,
        }


def _as_byte_array(data: bytes | bytearray | str | np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _build_learner(config: ALMLanguageConfig):
    if config.native:
        from manifoldflow.dendritic_basin.native_cortex import (
            NativeGrowableCortex,
            native_available,
        )

        if not native_available():
            raise RuntimeError(
                "native=True requires the alm_native shared library; "
                "see cpp/README.md, or pass native=False to run in pure "
                "Python instead"
            )
        return NativeGrowableCortex(**config.learner_kwargs())

    from manifoldflow.dendritic_basin.opponent_byte_stream import (
        SequenceCortexOpponentLearner,
    )

    return SequenceCortexOpponentLearner(packed_paths=True, **config.learner_kwargs())


def _sample_byte(
    distribution: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    probability = np.asarray(distribution, dtype=np.float64).reshape(-1)
    if probability.shape != (256,):
        raise ValueError("the output distribution must contain 256 byte probabilities")
    if temperature == 0.0 or top_k == 1:
        return int(np.argmax(probability))
    logits = np.log(np.maximum(probability, 1e-300)) / temperature
    if 0 < top_k < 256:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full(256, -np.inf, dtype=np.float64)
        masked[keep] = logits[keep]
        logits = masked
    logits -= np.max(logits)
    probability = np.exp(logits)
    probability /= probability.sum()
    if top_p < 1.0:
        order = np.argsort(probability)[::-1]
        cumulative = np.cumsum(probability[order])
        keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        keep = order[:keep_count]
        nucleus = np.zeros(256, dtype=np.float64)
        nucleus[keep] = probability[keep]
        probability = nucleus / nucleus.sum()
    return int(rng.choice(256, p=probability))


def _worker_train_column(
    args: tuple[bytes, bytes, int],
) -> bytes:
    """Runs in a separate process: load the pickled ancestor, train on this
    column's byte slice, return the pickled result. Top-level (not a
    closure/lambda) so ProcessPoolExecutor can pickle it on Windows spawn.
    """
    ancestor_bytes, data_slice, _ = args
    learner = pickle.loads(ancestor_bytes)
    learner.begin_lineage_tracking()
    learner.observe_chunk_probabilities(
        np.frombuffer(data_slice, dtype=np.uint8), learn=True
    )
    return pickle.dumps(learner)


def _partition(total: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, parts)
    offsets = []
    start = 0
    for index in range(parts):
        length = base + (1 if index < extra else 0)
        offsets.append((start, length))
        start += length
    return offsets


class ALMLanguageModel:
    """A configurable, trainable ALM core for byte-level language streams."""

    def __init__(self, config: ALMLanguageConfig | None = None) -> None:
        self.config = config or ALMLanguageConfig()
        self._learner = _build_learner(self.config)
        self.rounds_trained = 0
        self.total_bytes_trained = 0

    @property
    def learner(self) -> Any:
        """The underlying SequenceCortexOpponentLearner or
        NativeGrowableCortex, for anything not exposed here directly."""
        return self._learner

    @property
    def embodied_bytes(self) -> int:
        return int(self._learner.embodied_bytes)

    @property
    def live_sequence_count(self) -> int:
        return int(self._learner.live_sequence_count)

    def train(self, data: bytes | bytearray | str | np.ndarray) -> dict[str, object]:
        """Train on one pass of `data`, sleeping every `sleep_bytes`.

        Returns a small report: bytes trained, bits_per_byte over this
        call's own predictions (prequential -- each byte is scored before
        it is learned from), sleep_cycles, and the resulting body size.
        """
        stream = _as_byte_array(data)
        total = len(stream)
        if total == 0:
            return {
                "bytes_trained": 0,
                "bits_per_byte": float("nan"),
                "sleep_cycles": 0,
                "embodied_bytes": self.embodied_bytes,
                "live_sequence_count": self.live_sequence_count,
            }

        sleep_cycles = 0
        bit_sum = 0.0
        scored = 0
        for offset in range(0, total, self.config.sleep_bytes):
            round_bytes = stream[offset : offset + self.config.sleep_bytes]
            if self.config.columns == 1:
                probability = self._learner.observe_chunk_probabilities(
                    round_bytes, learn=True
                )
            else:
                probability = self._train_round_multicolumn(round_bytes)
            finite = probability[np.isfinite(probability)]
            bit_sum += float(np.sum(-np.log2(np.maximum(finite, 1e-300))))
            scored += len(finite)
            self._learner.consolidate_to_byte_budget(self.config.max_body_bytes)
            sleep_cycles += 1

        self.rounds_trained += sleep_cycles
        self.total_bytes_trained += total
        return {
            "bytes_trained": total,
            "bits_per_byte": bit_sum / max(scored, 1),
            "sleep_cycles": sleep_cycles,
            "embodied_bytes": self.embodied_bytes,
            "live_sequence_count": self.live_sequence_count,
        }

    def _train_round_multicolumn(self, round_bytes: np.ndarray) -> np.ndarray:
        ancestor = self._learner
        ancestor.reset_episode()
        ancestor_bytes = pickle.dumps(ancestor)
        offsets = _partition(len(round_bytes), self.config.columns)
        arguments = [
            (ancestor_bytes, round_bytes[start : start + length].tobytes(), index)
            for index, (start, length) in enumerate(offsets)
        ]
        with ProcessPoolExecutor(max_workers=self.config.columns) as pool:
            column_bytes = list(pool.map(_worker_train_column, arguments))

        master = pickle.loads(pickle.dumps(ancestor))
        for payload in column_bytes:
            descendant = pickle.loads(payload)
            master.consolidate_delta(descendant, ancestor)
        self._learner = master

        # The per-round "probability" report is prequential across the
        # whole round's bytes, computed once more on the merged master so
        # it reflects the anatomy actually kept -- cheap relative to
        # training this same data already cost.
        master.reset_episode()
        return master.observe_chunk_probabilities(round_bytes, learn=False)

    def evaluate_bits_per_byte(self, data: bytes | bytearray | str | np.ndarray) -> float:
        """Held-out bits/byte on `data`, without learning from it."""
        stream = _as_byte_array(data)
        self._learner.reset_episode()
        probability = self._learner.observe_chunk_probabilities(stream, learn=False)
        finite = probability[np.isfinite(probability)]
        return float(np.mean(-np.log2(np.maximum(finite, 1e-300))))

    def generate(
        self,
        cue: bytes | str = "",
        *,
        max_new_bytes: int = 200,
        temperature: float = 0.8,
        top_k: int = 32,
        top_p: float = 1.0,
        seed: int = 0,
        as_text: bool = True,
    ) -> str | bytes:
        """Generate a continuation from `cue`. Returns str by default
        (UTF-8 decoded with `errors="replace"`, since a byte-level
        organism can emit invalid UTF-8); pass as_text=False for raw
        bytes."""
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must be non-negative")
        prompt = cue.encode("utf-8") if isinstance(cue, str) else bytes(cue)
        self._learner.reset_episode()
        if prompt:
            self._learner.observe_chunk_probabilities(
                np.frombuffer(prompt, dtype=np.uint8), learn=False
            )
        rng = np.random.default_rng(seed)
        generated = bytearray()
        for _ in range(max_new_bytes):
            byte_id = _sample_byte(
                self._learner.predict_next_distribution(),
                rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated.append(byte_id)
            self._learner.advance_observation(byte_id, learn=False)
        result = prompt + bytes(generated)
        return result.decode("utf-8", errors="replace") if as_text else result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "learner": self._learner,
                    "rounds_trained": self.rounds_trained,
                    "total_bytes_trained": self.total_bytes_trained,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ALMLanguageModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = cls.__new__(cls)
        model.config = ALMLanguageConfig(**payload["config"])
        model._learner = payload["learner"]
        model.rounds_trained = payload.get("rounds_trained", 0)
        model.total_bytes_trained = payload.get("total_bytes_trained", 0)
        return model
