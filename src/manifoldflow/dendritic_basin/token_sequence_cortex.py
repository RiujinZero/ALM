"""Lossless lexical transduction and a vocabulary-sized sequence cortex.

This module is deliberately small.  It exists to test whether ALM's temporal
learning changes materially when the sensory assemblies correspond to words,
punctuation, and whitespace instead of individual bytes.  The tokenizer is
injective: lexical units outside the learned vocabulary are represented by a
raw-byte escape, so likelihood can still be reported in bits per source byte.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from heapq import nsmallest
from math import log2

import numpy as np

_UNIT_PATTERN = re.compile(rb"[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+|[ \t\r\n]|.")


@dataclass(frozen=True)
class LosslessLexicalCodec:
    """Words/punctuation/whitespace with an exact raw-byte fallback."""

    id_to_unit: tuple[bytes | None, ...]

    RAW_BEGIN = 256
    RAW_END = 257
    FIRST_LEXICAL = 258

    @classmethod
    def build(cls, source: bytes, *, vocab_size: int) -> LosslessLexicalCodec:
        if vocab_size < cls.FIRST_LEXICAL + 1:
            raise ValueError("vocab_size must leave room for lexical units")
        counts = Counter(_UNIT_PATTERN.findall(source))
        # Frequency decides survival; bytes break ties for deterministic builds.
        survivors = sorted(counts, key=lambda unit: (-counts[unit], unit))[
            : vocab_size - cls.FIRST_LEXICAL
        ]
        return cls((None,) * cls.FIRST_LEXICAL + tuple(survivors))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_unit)

    @property
    def embodied_bytes(self) -> int:
        # Only the ordered lexicon is necessary to reconstruct both maps.
        return len(pickle.dumps(self.id_to_unit, protocol=5))

    @property
    def unit_to_id(self) -> dict[bytes, int]:
        return {
            unit: token_id
            for token_id, unit in enumerate(self.id_to_unit)
            if unit is not None
        }

    def encode(self, source: bytes) -> np.ndarray:
        lookup = self.unit_to_id
        encoded: list[int] = []
        for unit in _UNIT_PATTERN.findall(source):
            token_id = lookup.get(unit)
            if token_id is not None:
                encoded.append(token_id)
            else:
                encoded.append(self.RAW_BEGIN)
                encoded.extend(unit)
                encoded.append(self.RAW_END)
        return np.asarray(encoded, dtype=np.int32)

    def decode(self, tokens: np.ndarray | list[int]) -> bytes:
        output = bytearray()
        raw = False
        for value in tokens:
            token_id = int(value)
            if raw:
                if token_id == self.RAW_END:
                    raw = False
                elif 0 <= token_id < 256:
                    output.append(token_id)
                else:
                    raise ValueError("invalid token inside raw-byte escape")
            elif token_id == self.RAW_BEGIN:
                raw = True
            elif token_id < self.FIRST_LEXICAL:
                raise ValueError("raw byte or terminator outside an escape")
            elif token_id < self.vocab_size:
                unit = self.id_to_unit[token_id]
                assert unit is not None
                output.extend(unit)
            else:
                raise ValueError("token outside vocabulary")
        if raw:
            raise ValueError("unterminated raw-byte escape")
        return bytes(output)


class TokenSequenceCortex:
    """Vocabulary-neutral analogue of the surprise-grown sequence cortex.

    Every token is a sensory assembly.  Repeated tokens become living cortical
    assemblies; transitions grow immediately, surprising two-token contexts
    grow through a nursery, and deeper temporal paths survive after repeated
    observation.  Prediction backs off through the same finite-strength parent
    mixture used by :class:`SequenceCortexOpponentLearner`.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        sequence_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        parent_strength: float = 16.0,
        max_context_tokens: int = 6,
        max_nursery_paths: int = 50_000,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if min(confirmations, context_confirmations, sequence_confirmations) < 1:
            raise ValueError("confirmation counts must be positive")
        if max_context_tokens < 2:
            raise ValueError("max_context_tokens must be at least two")
        self.vocab_size = int(vocab_size)
        self.prior = float(prior)
        self.confirmations = int(confirmations)
        self.context_confirmations = int(context_confirmations)
        self.sequence_confirmations = int(sequence_confirmations)
        self.surprise_bits = float(surprise_bits)
        self.basal_strength = float(basal_strength)
        self.parent_strength = float(parent_strength)
        self.max_context_tokens = int(max_context_tokens)
        self.max_nursery_paths = int(max_nursery_paths)

        self.assembly_confirmations = np.zeros(vocab_size, dtype=np.uint16)
        self.live_assemblies = np.zeros(vocab_size, dtype=np.bool_)
        self.unigram_counts = np.zeros(vocab_size, dtype=np.int64)
        self.transition_counts: list[dict[int, int]] = [
            {} for _ in range(vocab_size)
        ]
        self.context_totals = np.zeros(vocab_size, dtype=np.int64)
        self.context_counts: dict[tuple[int, int], dict[int, int]] = {}
        self.context_totals_2: dict[tuple[int, int], int] = {}
        self.context_nursery: dict[tuple[int, int], dict[int, int]] = {}
        # Index zero is a three-token context, matching the byte cortex banks.
        self.sequence_counts: list[dict[tuple[int, ...], dict[int, int]]] = [
            {} for _ in range(max_context_tokens - 2)
        ]
        self.sequence_totals: list[dict[tuple[int, ...], int]] = [
            {} for _ in self.sequence_counts
        ]
        self.sequence_nursery: list[dict[tuple[int, ...], dict[int, int]]] = [
            {} for _ in self.sequence_counts
        ]
        self.history: list[int] = []
        self.steps = 0
        self.context_births = 0
        self.sequence_births = 0
        self.nursery_evictions = 0

    @property
    def embodied_bytes(self) -> int:
        return len(pickle.dumps(self.__dict__, protocol=5))

    @property
    def live_sequence_count(self) -> int:
        return sum(len(bank) for bank in self.sequence_counts)

    def reset_episode(self) -> None:
        self.history.clear()

    def consolidate_to_byte_budget(self, max_bytes: int) -> dict[str, int]:
        """Shed provisional and weak deep paths until the body fits."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        before = self.embodied_bytes
        self.reset_episode()
        shed_nursery = sum(len(bank) for bank in self.sequence_nursery)
        self.sequence_nursery = [{} for _ in self.sequence_nursery]
        current = self.embodied_bytes
        pruned = 0
        passes = 0
        while current > max_bytes:
            live = self.live_sequence_count
            if live < 1:
                raise ValueError(
                    "fixed lexical cortical anatomy exceeds the byte budget"
                )
            mean_path_bytes = max(8.0, current / live)
            needed = max(
                1,
                min(live, int((current - max_bytes) / mean_path_bytes * 1.1) + 1),
            )

            def candidates():
                for bank_index, bank in enumerate(self.sequence_counts):
                    for context, row in bank.items():
                        yield (
                            max(row.values()),
                            self.sequence_totals[bank_index][context],
                            -bank_index,
                            context,
                            bank_index,
                        )

            weakest = nsmallest(needed, candidates())
            if not weakest:
                raise RuntimeError("no lexical temporal paths remain to prune")
            for *_rank, context, bank_index in weakest:
                del self.sequence_counts[bank_index][context]
                del self.sequence_totals[bank_index][context]
            pruned += len(weakest)
            passes += 1
            current = self.embodied_bytes
        self.sequence_births = self.live_sequence_count
        return {
            "body_bytes_before_sleep": before,
            "body_bytes_after_sleep": current,
            "shed_nursery_paths": shed_nursery,
            "pruned_mature_paths": pruned,
            "competition_passes": passes,
        }

    def _basal_probability(self, target: int) -> float:
        if not self.history:
            total = int(self.unigram_counts.sum()) + self.vocab_size * self.prior
            return float((self.unigram_counts[target] + self.prior) / total)
        previous = self.history[-1]
        total = int(self.context_totals[previous]) + self.vocab_size * self.prior
        return float(
            (self.transition_counts[previous].get(target, 0) + self.prior) / total
        )

    def observed_probability(self, target: int) -> float:
        probability = self._basal_probability(target)
        if len(self.history) >= 2:
            key = tuple(self.history[-2:])
            row = self.context_counts.get(key)
            if row is not None:
                probability = float(
                    (row.get(target, 0) + self.basal_strength * probability)
                    / (self.context_totals_2[key] + self.basal_strength)
                )
        for depth in range(3, min(len(self.history), self.max_context_tokens) + 1):
            key = tuple(self.history[-depth:])
            row = self.sequence_counts[depth - 3].get(key)
            if row is not None:
                probability = float(
                    (row.get(target, 0) + self.parent_strength * probability)
                    / (self.sequence_totals[depth - 3][key] + self.parent_strength)
                )
        return probability

    def distribution(self) -> np.ndarray:
        if not self.history:
            result = self.unigram_counts.astype(np.float64) + self.prior
            result /= result.sum()
        else:
            previous = self.history[-1]
            total = self.context_totals[previous] + self.vocab_size * self.prior
            result = np.full(self.vocab_size, self.prior / total)
            for target, count in self.transition_counts[previous].items():
                result[target] = (count + self.prior) / total
        if len(self.history) >= 2:
            key = tuple(self.history[-2:])
            row = self.context_counts.get(key)
            if row is not None:
                result *= self.basal_strength
                for target, count in row.items():
                    result[target] += count
                result /= self.context_totals_2[key] + self.basal_strength
        for depth in range(3, min(len(self.history), self.max_context_tokens) + 1):
            key = tuple(self.history[-depth:])
            row = self.sequence_counts[depth - 3].get(key)
            if row is not None:
                result *= self.parent_strength
                for target, count in row.items():
                    result[target] += count
                result /= self.sequence_totals[depth - 3][key] + self.parent_strength
        return result

    @staticmethod
    def _increment(row: dict[int, int], target: int) -> None:
        row[target] = row.get(target, 0) + 1

    def _update_context(self, key: tuple[int, int], target: int, basal: float) -> None:
        row = self.context_counts.get(key)
        if row is not None:
            self._increment(row, target)
            self.context_totals_2[key] += 1
            return
        if -log2(max(basal, 1e-300)) < self.surprise_bits:
            return
        row = self.context_nursery.setdefault(key, {})
        self._increment(row, target)
        if sum(row.values()) >= self.context_confirmations:
            self.context_counts[key] = self.context_nursery.pop(key)
            self.context_totals_2[key] = sum(row.values())
            self.context_births += 1

    def _update_sequences(self, target: int) -> None:
        for depth in range(3, min(len(self.history), self.max_context_tokens) + 1):
            index = depth - 3
            key = tuple(self.history[-depth:])
            row = self.sequence_counts[index].get(key)
            if row is not None:
                self._increment(row, target)
                self.sequence_totals[index][key] += 1
                continue
            nursery = self.sequence_nursery[index]
            if key not in nursery and len(nursery) >= self.max_nursery_paths:
                del nursery[next(iter(nursery))]
                self.nursery_evictions += 1
            candidate = nursery.setdefault(key, {})
            self._increment(candidate, target)
            if sum(candidate.values()) >= self.sequence_confirmations:
                self.sequence_counts[index][key] = nursery.pop(key)
                self.sequence_totals[index][key] = sum(candidate.values())
                self.sequence_births += 1

    def observe(self, token_id: int, *, learn: bool = True) -> float | None:
        target = int(token_id)
        if not 0 <= target < self.vocab_size:
            raise ValueError("token_id outside vocabulary")
        probability = None if self.steps == 0 else self.observed_probability(target)
        basal = self._basal_probability(target)
        if learn:
            live = bool(self.live_assemblies[target])
            if not live:
                self.assembly_confirmations[target] += 1
                live = self.assembly_confirmations[target] >= self.confirmations
                if live:
                    self.live_assemblies[target] = True
            if live:
                if len(self.history) >= 2:
                    self._update_context(tuple(self.history[-2:]), target, basal)
                self._update_sequences(target)
                self.unigram_counts[target] += 1
                if self.history:
                    previous = self.history[-1]
                    self._increment(self.transition_counts[previous], target)
                    self.context_totals[previous] += 1
                self.history.append(target)
                del self.history[:-self.max_context_tokens]
            else:
                self.history.clear()
            self.steps += 1
        elif self.live_assemblies[target]:
            self.history.append(target)
            del self.history[:-self.max_context_tokens]
        else:
            self.history.clear()
        return probability

    def observe_chunk_probabilities(
        self, token_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        result = np.full(len(values), np.nan)
        for index, token_id in enumerate(values):
            probability = self.observe(int(token_id), learn=learn)
            if probability is not None:
                result[index] = probability
        return result
