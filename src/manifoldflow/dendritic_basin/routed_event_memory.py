"""Selectively routed recurrent motif memory for dendritic ALM."""

from __future__ import annotations

import pickle
from math import ceil, log2
from typing import Any

import numpy as np

from manifoldflow.dendritic_basin.goal_attractor import (
    SparseHomeostaticGoalAttractor,
    SparsePredictiveBasinAttractor,
)
from manifoldflow.dendritic_basin.opponent_byte_stream import (
    OpponentStep,
    PredictiveMotifOpponentLearner,
    SequenceCortexOpponentLearner,
)
from manifoldflow.dendritic_basin.transient_binding import FastSparseBindingIndex


class RoutedMotifOpponentLearner(PredictiveMotifOpponentLearner):
    """Factorized temporal-role receptors above prediction-born motifs.

    Each lag is a reusable dendritic receptor role. A branch binds one source
    motif to one role and predicts the next motif form. Branch utility is the
    local information gain over the organism's motif prior; useful active
    branches compete to provide the one apical row seen by the byte cortex.
    No byte value, delimiter, or language-specific role is privileged.
    """

    def __init__(
        self,
        *,
        route_lags: tuple[int, ...] = (1, 2, 4, 8, 16),
        route_confirmations: int = 3,
        route_prior_strength: float = 8.0,
        route_utility_decay: float = 0.99,
        minimum_route_utility_bits: float = 0.05,
        max_route_paths: int = 100_000,
        max_route_nursery_paths: int = 50_000,
        max_alm_bytes: int | None = 64_000_000,
        byte_budget_check_interval: int = 5_000_000,
        max_motif_assemblies: int | None = None,
        motif_confirmations: int = 2,
        max_motif_nursery: int = 50_000,
        motif_activity_decay: float = 0.99999,
        max_motif_recycles_per_check: int = 128,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not route_lags or any(lag < 1 for lag in route_lags):
            raise ValueError("route lags must contain positive values")
        if route_confirmations < 1:
            raise ValueError("route confirmations must be positive")
        if route_prior_strength <= 0.0:
            raise ValueError("route prior strength must be positive")
        if not 0.0 <= route_utility_decay < 1.0:
            raise ValueError("route utility decay must be in [0, 1)")
        if minimum_route_utility_bits < 0.0:
            raise ValueError("minimum route utility must be non-negative")
        if max_route_paths < 1 or max_route_nursery_paths < 1:
            raise ValueError("route capacities must be positive")
        if max_alm_bytes is not None and max_alm_bytes < 1:
            raise ValueError("max ALM bytes must be positive")
        if byte_budget_check_interval < 1:
            raise ValueError("byte budget check interval must be positive")
        if max_motif_assemblies is not None and max_motif_assemblies < 1:
            raise ValueError("max motif assemblies must be positive")
        if motif_confirmations < 1 or max_motif_nursery < 1:
            raise ValueError("motif nursery settings must be positive")
        if not 0.0 < motif_activity_decay <= 1.0:
            raise ValueError("motif activity decay must be in (0, 1]")
        if max_motif_recycles_per_check < 1:
            raise ValueError("motif recycle budget must be positive")
        self.route_lags = tuple(sorted(set(route_lags)))
        self.route_confirmations = route_confirmations
        self.route_prior_strength = route_prior_strength
        self.route_utility_decay = route_utility_decay
        self.minimum_route_utility_bits = minimum_route_utility_bits
        self.max_route_paths = max_route_paths
        self.max_route_nursery_paths = max_route_nursery_paths
        self.max_alm_bytes = max_alm_bytes
        self.byte_budget_check_interval = byte_budget_check_interval
        self.max_motif_assemblies = (
            max_motif_assemblies
            if max_motif_assemblies is not None
            else (
                max(1, max_alm_bytes // 256)
                if max_alm_bytes is not None
                else 1_000_000_000
            )
        )
        self.motif_confirmations = motif_confirmations
        self.max_motif_nursery = max_motif_nursery
        self.motif_activity_decay = motif_activity_decay
        self.max_motif_recycles_per_check = max_motif_recycles_per_check
        self.motif_activity: list[float] = []
        self.motif_last_seen: list[int] = []
        self.motif_nursery: dict[bytes, tuple[int, int]] = {}
        self.motif_nursery_evictions = 0
        self.motif_recycles = 0
        self.route_history: list[int] = []
        self.route_counts: dict[tuple[int, int], dict[int, int]] = {}
        self.route_totals: dict[tuple[int, int], int] = {}
        self.route_utility: dict[tuple[int, int], float] = {}
        self.route_nursery: dict[tuple[int, int], dict[int, int]] = {}
        self.route_target_counts: dict[int, int] = {}
        self.route_target_total = 0
        self.route_births = 0
        self.route_nursery_evictions = 0
        self.route_capacity_rejections = 0
        self.budget_checks = 0
        self.budget_evictions = 0
        self.budget_mature_replacements = 0
        self.budget_nursery_losses = 0
        self.budget_evictions_by_tissue: dict[str, int] = {}
        self.competition_history: list[tuple[int, int, int, int]] = []
        self.last_budget_check_step = 0
        self.growth_frozen = False

    @property
    def live_route_count(self) -> int:
        return len(self.route_counts)

    @property
    def live_route_synapse_count(self) -> int:
        return sum(len(row) for row in self.route_counts.values())

    @property
    def route_nursery_count(self) -> int:
        return len(self.route_nursery)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.route_lags,
            self.route_confirmations,
            self.route_prior_strength,
            self.route_utility_decay,
            self.minimum_route_utility_bits,
            self.max_route_paths,
            self.max_route_nursery_paths,
            self.max_alm_bytes,
            self.byte_budget_check_interval,
            self.max_motif_assemblies,
            self.motif_confirmations,
            self.max_motif_nursery,
            self.motif_activity_decay,
            self.max_motif_recycles_per_check,
            self.motif_activity,
            self.motif_last_seen,
            self.motif_nursery,
            self.motif_nursery_evictions,
            self.motif_recycles,
            self.route_history,
            self.route_counts,
            self.route_totals,
            self.route_utility,
            self.route_nursery,
            self.route_target_counts,
            self.route_target_total,
            self.route_births,
            self.route_nursery_evictions,
            self.route_capacity_rejections,
            self.budget_checks,
            self.budget_evictions,
            self.budget_mature_replacements,
            self.budget_nursery_losses,
            self.budget_evictions_by_tissue,
            self.competition_history,
            self.last_budget_check_step,
            self.growth_frozen,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    @property
    def byte_occupancy(self) -> float | None:
        if self.max_alm_bytes is None:
            return None
        return self.serialized_size_bytes() / self.max_alm_bytes

    def serialized_size_bytes(self) -> int:
        """Exact durable learner bytes, excluding an experiment report wrapper."""
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def _word_id(self, word: bytes, *, create: bool) -> int | None:
        word_id = self.word_ids.get(word)
        if word_id is not None or not create:
            return word_id
        if getattr(self, "_consolidating_mature_forms", False):
            if len(self.word_forms) >= self.max_motif_assemblies:
                return None
            word_id = super()._word_id(word, create=True)
            assert word_id is not None
            self.motif_activity.append(1.0)
            self.motif_last_seen.append(self.steps)
            return word_id
        if self.growth_frozen:
            self.route_capacity_rejections += 1
            return None
        confirmations, _ = self.motif_nursery.get(word, (0, self.steps))
        confirmations += 1
        if confirmations < self.motif_confirmations:
            if (
                word not in self.motif_nursery
                and len(self.motif_nursery) >= self.max_motif_nursery
            ):
                weakest = min(
                    self.motif_nursery,
                    key=lambda form: (
                        self.motif_nursery[form][0],
                        self.motif_nursery[form][1],
                    ),
                )
                del self.motif_nursery[weakest]
                self.motif_nursery_evictions += 1
            self.motif_nursery[word] = confirmations, self.steps
            return None
        if len(self.word_forms) < self.max_motif_assemblies:
            self.motif_nursery.pop(word, None)
            word_id = super()._word_id(word, create=True)
            assert word_id is not None
            self.motif_activity.append(float(confirmations))
            self.motif_last_seen.append(self.steps)
            return word_id
        self.motif_nursery[word] = confirmations, self.steps
        self.route_capacity_rejections += 1
        return None

    def _decayed_motif_activity(self, word_id: int) -> float:
        age = max(0, self.steps - self.motif_last_seen[word_id])
        return self.motif_activity[word_id] * self.motif_activity_decay**age

    def _touch_motif(self, word_id: int) -> None:
        self.motif_activity[word_id] = self._decayed_motif_activity(word_id) + 1.0
        self.motif_last_seen[word_id] = self.steps

    def _recycle_motif(
        self, word_id: int, new_form: bytes, initial_activity: float
    ) -> None:
        old_form = self.word_forms[word_id]
        del self.word_ids[old_form]

        for bank_index, bank in enumerate(self.event_counts):
            depth = bank_index + 1
            for context, row in list(bank.items()):
                if word_id in self._unpack_event_context(context, depth):
                    del bank[context]
                    del self.event_totals[bank_index][context]
                    continue
                row.pop(word_id, None)
                if not row:
                    del bank[context]
                    del self.event_totals[bank_index][context]
                else:
                    self.event_totals[bank_index][context] = sum(row.values())
        for bank_index, bank in enumerate(self.event_nursery):
            depth = bank_index + 1
            for context, row in list(bank.items()):
                if word_id in self._unpack_event_context(context, depth):
                    del bank[context]
                    continue
                row.pop(word_id, None)
                if not row:
                    del bank[context]

        for key, row in list(self.route_counts.items()):
            if key[1] == word_id:
                del self.route_counts[key]
                del self.route_totals[key]
                del self.route_utility[key]
                continue
            row.pop(word_id, None)
            if not row:
                del self.route_counts[key]
                del self.route_totals[key]
                del self.route_utility[key]
            else:
                self.route_totals[key] = sum(row.values())
        for key, row in list(self.route_nursery.items()):
            if key[1] == word_id:
                del self.route_nursery[key]
                continue
            row.pop(word_id, None)
            if not row:
                del self.route_nursery[key]

        self.route_target_total -= self.route_target_counts.pop(word_id, 0)
        self.word_forms[word_id] = new_form
        self.word_ids[new_form] = word_id
        self.motif_activity[word_id] = initial_activity
        self.motif_last_seen[word_id] = self.steps
        self.event_history.clear()
        self.route_history.clear()
        self.active_word_candidates = None
        self.event_receptor_cache.clear()
        getattr(self, "_predictive_route_signature_cache", {}).clear()
        self.motif_recycles += 1

    def _compete_motif_nursery(self) -> int:
        recycled = 0
        challengers = sorted(
            self.motif_nursery.items(),
            key=lambda item: (item[1][0], item[1][1]),
            reverse=True,
        )
        for form, (confirmations, _) in challengers:
            if recycled >= self.max_motif_recycles_per_check:
                break
            if form in self.word_ids:
                self.motif_nursery.pop(form, None)
                continue
            if len(self.word_forms) < self.max_motif_assemblies:
                continue
            weakest_id = min(
                range(len(self.word_forms)), key=self._decayed_motif_activity
            )
            if confirmations <= self._decayed_motif_activity(weakest_id):
                break
            self.motif_nursery.pop(form, None)
            self._recycle_motif(weakest_id, form, float(confirmations))
            recycled += 1
        return recycled

    def _active_route_keys(self) -> tuple[tuple[int, int], ...]:
        keys = []
        for lag in self.route_lags:
            if len(self.route_history) < lag:
                continue
            source = self.route_history[-lag]
            if source >= 0:
                keys.append((lag, source))
        return tuple(keys)

    def _learning_route_keys(self) -> tuple[tuple[int, int], ...]:
        """Return only causes allowed to create durable route evidence."""
        return self._active_route_keys()

    def _selected_route_key(self) -> tuple[int, int] | None:
        candidates = [
            key
            for key in self._active_route_keys()
            if key in self.route_counts
            and self.route_utility[key] > self.minimum_route_utility_bits
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda key: (
                self.route_utility[key],
                self.route_totals[key],
                -key[0],
            ),
        )

    def _active_event_row(self) -> dict[int, int] | None:
        key = self._selected_route_key()
        if key is not None:
            return self.route_counts[key]
        return super()._active_event_row()

    def _target_prior(self, target_id: int) -> float:
        vocabulary = max(len(self.word_forms), 1)
        return (self.route_target_counts.get(target_id, 0) + 1.0) / (
            self.route_target_total + vocabulary
        )

    def _route_probability(self, key: tuple[int, int], target_id: int) -> float:
        prior = self._target_prior(target_id)
        return (
            self.route_counts[key].get(target_id, 0) + self.route_prior_strength * prior
        ) / (self.route_totals[key] + self.route_prior_strength)

    def _admit_route(self, key: tuple[int, int], row: dict[int, int]) -> None:
        if len(self.route_counts) >= self.max_route_paths:
            self.route_capacity_rejections += 1
            return
        self.route_counts[key] = row
        self.route_totals[key] = sum(row.values())
        self.route_utility[key] = 0.0
        self.route_births += 1

    def _update_routes(self, keys: tuple[tuple[int, int], ...], target_id: int) -> None:
        prior = self._target_prior(target_id)
        for key in keys:
            row = self.route_counts.get(key)
            if row is not None:
                probability = self._route_probability(key, target_id)
                gain = log2(max(probability, 1e-300) / max(prior, 1e-300))
                self.route_utility[key] = (
                    self.route_utility_decay * self.route_utility[key]
                    + (1.0 - self.route_utility_decay) * gain
                )
                row[target_id] = row.get(target_id, 0) + 1
                self.route_totals[key] += 1
                continue
            if key not in self.route_nursery:
                if len(self.route_nursery) >= self.max_route_nursery_paths:
                    del self.route_nursery[next(iter(self.route_nursery))]
                    self.route_nursery_evictions += 1
                self.route_nursery[key] = {}
            nursery = self.route_nursery[key]
            nursery[target_id] = nursery.get(target_id, 0) + 1
            if sum(nursery.values()) >= self.route_confirmations:
                del self.route_nursery[key]
                self._admit_route(key, nursery)
        self.route_target_counts[target_id] = (
            self.route_target_counts.get(target_id, 0) + 1
        )
        self.route_target_total += 1

    def _row_information(
        self,
        row: dict[int, int],
        *,
        word_targets: bool,
        total: int | None = None,
        word_prior_denominator: float | None = None,
        byte_priors: Any = None,
    ) -> float:
        total = sum(row.values()) if total is None else total
        if total <= 0:
            return 0.0
        inverse_total = 1.0 / total
        if word_targets:
            denominator = (
                float(self.route_target_total + max(len(self.word_forms), 1))
                if word_prior_denominator is None
                else word_prior_denominator
            )
            target_counts = self.route_target_counts
            return sum(
                (count * inverse_total)
                * log2(
                    (count * inverse_total)
                    / ((target_counts.get(target, 0) + 1.0) / denominator)
                )
                for target, count in row.items()
            )
        if byte_priors is None:
            base_total = float(self.unigram_counts.sum()) + 256.0
            byte_priors = (self.unigram_counts.astype(float) + 1.0) / base_total
        return sum(
            (count * inverse_total)
            * log2((count * inverse_total) / float(byte_priors[target]))
            for target, count in row.items()
        )

    def _competition_candidates(
        self,
    ) -> list[tuple[float, int, str, int, object]]:
        candidates: list[tuple[float, int, str, int, object]] = []
        word_prior_denominator = float(
            self.route_target_total + max(len(self.word_forms), 1)
        )
        byte_prior_denominator = float(self.unigram_counts.sum()) + 256.0
        byte_priors = (self.unigram_counts.astype(float) + 1.0) / byte_prior_denominator

        def add(
            tissue: str,
            bank_index: int,
            key: object,
            row: dict[int, int],
            *,
            word_targets: bool,
            mature: bool,
            utility: float | None = None,
        ) -> None:
            total = sum(row.values())
            empirical_information = self._row_information(
                row,
                word_targets=word_targets,
                total=total,
                word_prior_denominator=word_prior_denominator,
                byte_priors=byte_priors,
            )
            information = (
                max(utility, 0.0) if utility is not None else empirical_information
            )
            # Candidate ranking only needs a consistent local density estimate;
            # the whole-body ceiling below is still checked by exact pickle size.
            key_words = len(key) if isinstance(key, tuple) else 1
            serialized = 48 + 8 * key_words + 12 * len(row)
            maturity = 1.05 if mature else 1.0
            score = maturity * information * log2(total + 1.0) / serialized
            candidates.append((score, serialized, tissue, bank_index, key))

        for key, row in self.context_nursery.items():
            add("context_nursery", -1, key, row, word_targets=False, mature=False)
        for key, row in self.residual_counts.items():
            add("context", -1, key, row, word_targets=False, mature=True)
        for bank_index, bank in enumerate(self.sequence_nursery):
            for key, row in bank.items():
                add(
                    "sequence_nursery",
                    bank_index,
                    key,
                    row,
                    word_targets=False,
                    mature=False,
                )
        for bank_index, bank in enumerate(self.sequence_counts):
            for key, row in bank.items():
                add(
                    "sequence",
                    bank_index,
                    key,
                    row,
                    word_targets=False,
                    mature=True,
                )
        for bank_index, bank in enumerate(self.event_nursery):
            for key, row in bank.items():
                add(
                    "event_nursery",
                    bank_index,
                    key,
                    row,
                    word_targets=True,
                    mature=False,
                )
        for bank_index, bank in enumerate(self.event_counts):
            for key, row in bank.items():
                add(
                    "event",
                    bank_index,
                    key,
                    row,
                    word_targets=True,
                    mature=True,
                )
        for key, row in self.route_nursery.items():
            add("route_nursery", -1, key, row, word_targets=True, mature=False)
        for key, row in self.route_counts.items():
            add(
                "route",
                -1,
                key,
                row,
                word_targets=True,
                mature=True,
                utility=self.route_utility[key],
            )
        return candidates

    def _evict_competitor(self, tissue: str, bank_index: int, key: object) -> None:
        if tissue == "context_nursery":
            del self.context_nursery[key]
        elif tissue == "context":
            del self.residual_counts[key]
            del self.residual_totals[key]
        elif tissue == "sequence_nursery":
            del self.sequence_nursery[bank_index][key]
        elif tissue == "sequence":
            del self.sequence_counts[bank_index][key]
            del self.sequence_totals[bank_index][key]
        elif tissue == "event_nursery":
            del self.event_nursery[bank_index][key]
        elif tissue == "event":
            del self.event_counts[bank_index][key]
            del self.event_totals[bank_index][key]
        elif tissue == "route_nursery":
            del self.route_nursery[key]
        elif tissue == "route":
            del self.route_counts[key]
            del self.route_totals[key]
            del self.route_utility[key]
        else:  # pragma: no cover - internal exhaustive guard
            raise AssertionError(f"unknown competing tissue {tissue}")
        self.budget_evictions += 1
        if tissue.endswith("_nursery"):
            self.budget_nursery_losses += 1
        else:
            self.budget_mature_replacements += 1
        self.budget_evictions_by_tissue[tissue] = (
            self.budget_evictions_by_tissue.get(tissue, 0) + 1
        )

    def enforce_byte_budget(self, *, target_fraction: float = 0.95) -> int:
        """Run cross-tissue competition until the exact body fits its class."""
        if not 0.0 < target_fraction <= 1.0:
            raise ValueError("target fraction must be in (0, 1]")
        before_evictions = self.budget_evictions
        mature_before = self.budget_mature_replacements
        nursery_before = self.budget_nursery_losses
        self._compete_motif_nursery()
        if self.max_alm_bytes is None:
            return self.budget_evictions - before_evictions
        self.budget_checks += 1
        body = self.serialized_size_bytes()
        trigger = int(self.max_alm_bytes * 0.98)
        if body <= trigger:
            return self.budget_evictions - before_evictions
        target = min(self.max_alm_bytes, int(self.max_alm_bytes * target_fraction))
        while body > target:
            candidates = sorted(self._competition_candidates())
            if not candidates:
                self.growth_frozen = True
                break
            excess = body - target
            average_bytes = sum(item[1] for item in candidates) / len(candidates)
            count = min(
                len(candidates),
                max(1, ceil(excess / max(average_bytes, 1.0))),
            )
            for _, _, tissue, bank_index, key in candidates[:count]:
                self._evict_competitor(tissue, bank_index, key)
            self.event_receptor_cache.clear()
            body = self.serialized_size_bytes()
        mature_delta = self.budget_mature_replacements - mature_before
        nursery_delta = self.budget_nursery_losses - nursery_before
        if mature_delta or nursery_delta:
            self.competition_history.append(
                (self.steps, mature_delta, nursery_delta, body)
            )
            if len(self.competition_history) > 256:
                del self.competition_history[:-256]
            body = self.serialized_size_bytes()
            if body > self.max_alm_bytes:
                self.competition_history.pop()
                body = self.serialized_size_bytes()
        if body > self.max_alm_bytes:
            self.growth_frozen = True
            raise RuntimeError(
                f"serialized ALM body is {body:,} bytes, exceeding its "
                f"{self.max_alm_bytes:,}-byte class after competition"
            )
        return self.budget_evictions - before_evictions

    def _complete_word(self, *, learn: bool) -> None:
        if not self.current_word:
            return
        form = self._normalize_event_form(bytes(self.current_word))
        keys = self._learning_route_keys()
        recycles_before = self.motif_recycles
        super()._complete_word(learn=learn)
        target_id = self.word_ids.get(form)
        if learn and target_id is not None:
            self._touch_motif(target_id)
            if self.motif_recycles == recycles_before:
                self._update_routes(keys, target_id)
        self.route_history.append(-1 if target_id is None else target_id)
        excess = len(self.route_history) - max(self.route_lags)
        if excess > 0:
            del self.route_history[:excess]
        if (
            learn
            and self.max_alm_bytes is not None
            and self.steps - self.last_budget_check_step
            >= self.byte_budget_check_interval
        ):
            self.last_budget_check_step = self.steps
            self.enforce_byte_budget()

    def reset_episode(self) -> None:
        super().reset_episode()
        self.route_history.clear()

    def shed_route_nursery(self) -> int:
        shed = len(self.route_nursery)
        self.route_nursery.clear()
        return shed

    def consolidate_route_anatomy(
        self,
        *,
        minimum_observations: int = 3,
        minimum_utility_bits: float | None = None,
    ) -> tuple[int, int]:
        if minimum_observations < 1:
            raise ValueError("minimum observations must be positive")
        threshold = (
            self.minimum_route_utility_bits
            if minimum_utility_bits is None
            else minimum_utility_bits
        )
        doomed = [
            key
            for key, total in self.route_totals.items()
            if total < minimum_observations or self.route_utility[key] <= threshold
        ]
        synapses = sum(len(self.route_counts[key]) for key in doomed)
        for key in doomed:
            del self.route_counts[key]
            del self.route_totals[key]
            del self.route_utility[key]
        self.route_births = len(self.route_counts)
        return len(doomed), synapses

    def consolidate(self, other: RoutedMotifOpponentLearner) -> None:
        configuration = (
            "route_lags",
            "route_confirmations",
            "route_prior_strength",
            "route_utility_decay",
            "minimum_route_utility_bits",
            "max_route_paths",
            "max_route_nursery_paths",
            "max_alm_bytes",
            "byte_budget_check_interval",
            "max_motif_assemblies",
            "motif_confirmations",
            "max_motif_nursery",
            "motif_activity_decay",
            "max_motif_recycles_per_check",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError(
                "cannot consolidate routed cortices with different configuration"
            )
        self._consolidating_mature_forms = True
        try:
            super().consolidate(other)
        finally:
            self._consolidating_mature_forms = False
        incoming_to_local = [self.word_ids.get(form) for form in other.word_forms]

        def remap_row(row: dict[int, int]) -> dict[int, int]:
            mapped: dict[int, int] = {}
            for target, observations in row.items():
                local_target = incoming_to_local[target]
                if local_target is None:
                    continue
                mapped[local_target] = mapped.get(local_target, 0) + observations
            return mapped

        for (lag, source), incoming in other.route_counts.items():
            local_source = incoming_to_local[source]
            if local_source is None:
                continue
            key = lag, local_source
            mapped = remap_row(incoming)
            incoming_total = sum(mapped.values())
            if not incoming_total:
                continue
            if key in self.route_counts:
                local_total = self.route_totals[key]
                combined = local_total + incoming_total
                self.route_utility[key] = (
                    self.route_utility[key] * local_total
                    + other.route_utility[(lag, source)] * incoming_total
                ) / combined
                row = self.route_counts[key]
                for target, observations in mapped.items():
                    row[target] = row.get(target, 0) + observations
                self.route_totals[key] = combined
            elif len(self.route_counts) < self.max_route_paths:
                self.route_counts[key] = mapped
                self.route_totals[key] = incoming_total
                self.route_utility[key] = other.route_utility[(lag, source)]
            else:
                self.route_capacity_rejections += 1
        for (lag, source), incoming in other.route_nursery.items():
            local_source = incoming_to_local[source]
            if local_source is None:
                continue
            key = lag, local_source
            mapped = remap_row(incoming)
            if not mapped:
                continue
            if key in self.route_counts:
                row = self.route_counts[key]
                for target, observations in mapped.items():
                    row[target] = row.get(target, 0) + observations
                self.route_totals[key] += sum(mapped.values())
                continue
            if key not in self.route_nursery:
                if len(self.route_nursery) >= self.max_route_nursery_paths:
                    del self.route_nursery[next(iter(self.route_nursery))]
                    self.route_nursery_evictions += 1
                self.route_nursery[key] = {}
            row = self.route_nursery[key]
            for target, observations in mapped.items():
                row[target] = row.get(target, 0) + observations
            if sum(row.values()) >= self.route_confirmations:
                del self.route_nursery[key]
                self._admit_route(key, row)
        for target, observations in other.route_target_counts.items():
            local_target = incoming_to_local[target]
            if local_target is None:
                continue
            self.route_target_counts[local_target] = (
                self.route_target_counts.get(local_target, 0) + observations
            )
            self.route_target_total += observations
        self.route_births = len(self.route_counts)
        self.route_nursery_evictions += other.route_nursery_evictions
        self.route_capacity_rejections += other.route_capacity_rejections
        self.budget_checks += other.budget_checks
        self.budget_evictions += other.budget_evictions
        self.budget_mature_replacements += other.budget_mature_replacements
        self.budget_nursery_losses += other.budget_nursery_losses
        self.competition_history.extend(other.competition_history)
        self.competition_history = self.competition_history[-256:]
        for tissue, count in other.budget_evictions_by_tissue.items():
            self.budget_evictions_by_tissue[tissue] = (
                self.budget_evictions_by_tissue.get(tissue, 0) + count
            )
        self.last_budget_check_step = max(
            self.last_budget_check_step, other.last_budget_check_step
        )
        self.growth_frozen |= other.growth_frozen
        self.reset_episode()
        self.enforce_byte_budget()


class TemporalBasisMotifOpponentLearner(RoutedMotifOpponentLearner):
    """Event-specific working traces with logarithmic temporal receptors.

    Exact delay lines overwrite event identity when delays vary. This cortex
    instead keeps a bounded set of recently active motif identities. Each
    identity projects through one age-band receptor, allowing the same local
    route branch to remain addressable across nearby variable delays.
    """

    def __init__(
        self,
        *,
        temporal_bins: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        max_active_temporal_traces: int = 64,
        **kwargs: object,
    ) -> None:
        if not temporal_bins or any(bound < 1 for bound in temporal_bins):
            raise ValueError("temporal bins must contain positive upper bounds")
        if max_active_temporal_traces < 1:
            raise ValueError("active temporal trace capacity must be positive")
        self.temporal_bins = tuple(sorted(set(temporal_bins)))
        self.max_active_temporal_traces = max_active_temporal_traces
        super().__init__(route_lags=self.temporal_bins, **kwargs)
        self.temporal_clock = 0
        self.temporal_last_seen: dict[int, int] = {}
        self.temporal_recalled_sources: set[int] = set()
        self.temporal_trace_evictions = 0

    @property
    def live_temporal_trace_count(self) -> int:
        return len(self.temporal_last_seen)

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.temporal_bins,
            self.max_active_temporal_traces,
            self.temporal_clock,
            self.temporal_last_seen,
            self.temporal_trace_evictions,
        )
        return super().embodied_bytes + len(pickle.dumps(payload, protocol=5))

    def _temporal_bin(self, age: int) -> int | None:
        return next((bound for bound in self.temporal_bins if age <= bound), None)

    def _active_route_keys(self) -> tuple[tuple[int, int], ...]:
        keys = []
        for source, last_seen in self.temporal_last_seen.items():
            age = self.temporal_clock - last_seen + 1
            receptor = self._temporal_bin(age)
            if receptor is not None:
                keys.append((receptor, source))
        return tuple(keys)

    def _learning_route_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            key
            for key in self._active_route_keys()
            if key[1] not in getattr(self, "temporal_recalled_sources", set())
        )

    def _advance_temporal_trace(self, target_id: int | None) -> None:
        self.temporal_clock += 1
        oldest_age = self.temporal_bins[-1]
        stale = [
            source
            for source, last_seen in self.temporal_last_seen.items()
            if self.temporal_clock - last_seen >= oldest_age
        ]
        for source in stale:
            del self.temporal_last_seen[source]
            getattr(self, "temporal_recalled_sources", set()).discard(source)
            self.temporal_trace_evictions += 1
        if target_id is None:
            return
        getattr(self, "temporal_recalled_sources", set()).discard(target_id)
        if (
            target_id not in self.temporal_last_seen
            and len(self.temporal_last_seen) >= self.max_active_temporal_traces
        ):
            oldest = min(self.temporal_last_seen, key=self.temporal_last_seen.get)
            del self.temporal_last_seen[oldest]
            getattr(self, "temporal_recalled_sources", set()).discard(oldest)
            self.temporal_trace_evictions += 1
        self.temporal_last_seen[target_id] = self.temporal_clock

    def _complete_word(self, *, learn: bool) -> None:
        if not self.current_word:
            return
        form = self._normalize_event_form(bytes(self.current_word))
        super()._complete_word(learn=learn)
        self._advance_temporal_trace(self.word_ids.get(form))

    def _recycle_motif(
        self, word_id: int, new_form: bytes, initial_activity: float
    ) -> None:
        self.temporal_last_seen.pop(word_id, None)
        getattr(self, "temporal_recalled_sources", set()).discard(word_id)
        super()._recycle_motif(word_id, new_form, initial_activity)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.temporal_clock = 0
        self.temporal_last_seen.clear()
        getattr(self, "temporal_recalled_sources", set()).clear()


class GoalConditionedTemporalBasisMotifOpponentLearner(
    TemporalBasisMotifOpponentLearner
):
    """Temporal cortex with a bounded restoration-conditioned apical winner."""

    def __init__(
        self,
        *,
        goal_apical_weight: float = 0.05,
        goal_conditioned_gain: float = 0.5,
        maximum_goal_byte_weight: float = 0.0,
        goal_basin_width: float = 0.125,
        max_goal_assemblies: int = 256,
        goal_route_eligibility_decay: float = 0.9,
        maximum_goal_route_bonus: float = 0.5,
        goal_selection_utility_decay: float = 0.99,
        minimum_goal_selection_utility: float = 0.0,
        goal_permission_trial_limit: int = 0,
        minimum_goal_permission_evidence: float = 0.25,
        goal_third_component: str = "capacity_pressure",
        goal_route_context: str = "deficit",
        goal_permission_context: str = "direction",
        goal_reserve_activation: bool = False,
        maximum_goal_route_byte_weight: float = 0.0,
        goal_route_projection_policy: str = "switched_winner",
        goal_route_timing: str = "linear_age",
        goal_route_alignment: str = "target_basin_mass",
        goal_flow_polarity: str = "selected_goal",
        goal_lifecycle: str = "homeostatic",
        goal_failure_patience: int = 4,
        goal_target_alignment_weight: float = 1.0,
        predictive_goal_basins: int = 0,
        predictive_goal_dimensions: int = 8,
        predictive_goal_birth_similarity: float = 0.75,
        predictive_goal_hysteresis: float = 0.05,
        predictive_goal_weight: float = 0.25,
        binding_index_cells: int = 0,
        binding_role_assemblies: int = 32,
        binding_cells_per_association: int = 2,
        binding_activity_decay: float = 0.98,
        binding_minimum_activity: float = 0.05,
        binding_role_birth_similarity: float = 0.8,
        binding_surprise_bits: float = 2.0,
        binding_role_trial_limit: int = 16,
        binding_minimum_role_utility: float = 0.0,
        binding_address_context: str = "structural_basin",
        goal_flow_mode_gate: str = "all",
        goal_flow_mode_trials: int = 32,
        goal_surprise_scope: str = "alphabetic",
        repetition_window: int = 16,
        **kwargs: object,
    ) -> None:
        if not 0.0 <= goal_apical_weight <= 16.0:
            raise ValueError("goal apical weight must be in [0, 16]")
        if repetition_window < 2:
            raise ValueError("repetition window must be at least two")
        if not 0.0 <= maximum_goal_byte_weight <= 0.5:
            raise ValueError("maximum goal byte weight must be in [0, 0.5]")
        if not 0.0 < goal_basin_width <= 1.0:
            raise ValueError("goal basin width must be in (0, 1]")
        if not 0.0 < goal_route_eligibility_decay < 1.0:
            raise ValueError("goal route eligibility decay must be in (0, 1)")
        if not 0.0 <= maximum_goal_route_bonus <= 1.0:
            raise ValueError("maximum goal route bonus must be in [0, 1]")
        if not 0.0 <= goal_selection_utility_decay < 1.0:
            raise ValueError("goal selection utility decay must be in [0, 1)")
        if goal_permission_trial_limit < 0:
            raise ValueError("goal permission trial limit must be non-negative")
        if minimum_goal_permission_evidence < 0.0:
            raise ValueError("minimum goal permission evidence must be non-negative")
        if goal_third_component not in {"capacity_pressure", "temporal_disagreement"}:
            raise ValueError("unknown third goal component")
        if goal_route_context not in {"deficit", "restorative_basin"}:
            raise ValueError("unknown goal route context")
        if goal_permission_context not in {"direction", "goal_basin"}:
            raise ValueError("unknown goal permission context")
        if not 0.0 <= maximum_goal_route_byte_weight <= 0.5:
            raise ValueError("maximum goal route byte weight must be in [0, 0.5]")
        if goal_route_projection_policy not in {"switched_winner", "aligned_flow"}:
            raise ValueError("unknown goal route projection policy")
        if goal_route_timing not in {"linear_age", "learned_hazard"}:
            raise ValueError("unknown goal route timing")
        if goal_route_alignment not in {"target_basin_mass", "restoration_signature"}:
            raise ValueError("unknown goal route alignment")
        if goal_flow_polarity not in {
            "selected_goal",
            "regulatory_mode",
            "regulatory_goal",
            "regulatory_basin",
        }:
            raise ValueError("unknown goal flow polarity")
        if goal_lifecycle not in {"homeostatic", "target_or_failure"}:
            raise ValueError("unknown goal lifecycle")
        if goal_failure_patience < 1:
            raise ValueError("goal failure patience must be positive")
        if not 0.0 <= goal_target_alignment_weight <= 1.0:
            raise ValueError("goal target alignment weight must be in [0, 1]")
        if predictive_goal_basins < 0:
            raise ValueError("predictive goal basin capacity must be non-negative")
        if not 2 <= predictive_goal_dimensions <= 8:
            raise ValueError("predictive goal dimensions must be in [2, 8]")
        if not 0.0 <= predictive_goal_weight <= 1.0:
            raise ValueError("predictive goal weight must be in [0, 1]")
        if binding_index_cells < 0:
            raise ValueError("binding index capacity must be non-negative")
        if binding_role_assemblies < 2:
            raise ValueError("binding role capacity must be at least two")
        if binding_cells_per_association < 1:
            raise ValueError("binding cells per association must be positive")
        if binding_index_cells and binding_cells_per_association > binding_index_cells:
            raise ValueError("binding cells per association exceed index capacity")
        if not 0.0 < binding_activity_decay <= 1.0:
            raise ValueError("binding activity decay must be in (0, 1]")
        if not 0.0 <= binding_minimum_activity < 1.0:
            raise ValueError("binding minimum activity must be in [0, 1)")
        if not -1.0 <= binding_role_birth_similarity <= 1.0:
            raise ValueError("binding role birth similarity must be in [-1, 1]")
        if binding_surprise_bits < 0.0:
            raise ValueError("binding surprise threshold must be non-negative")
        if binding_role_trial_limit < 0:
            raise ValueError("binding role trial limit must be non-negative")
        if binding_address_context not in {
            "structural_basin",
            "query_only",
            "cortical_query",
            "raw_query",
        }:
            raise ValueError("unknown binding address context")
        if goal_flow_mode_gate not in {"all", "learned_utility"}:
            raise ValueError("unknown goal flow mode gate")
        if goal_flow_mode_trials < 0:
            raise ValueError("goal flow mode trials must be non-negative")
        if goal_surprise_scope not in {"alphabetic", "all_bytes"}:
            raise ValueError("unknown goal surprise scope")
        super().__init__(**kwargs)
        self.goal_apical_weight = float(goal_apical_weight)
        self.maximum_goal_byte_weight = float(maximum_goal_byte_weight)
        self.goal_basin_width = float(goal_basin_width)
        self.repetition_window = int(repetition_window)
        self.goal_route_eligibility_decay = float(goal_route_eligibility_decay)
        self.maximum_goal_route_bonus = float(maximum_goal_route_bonus)
        self.goal_selection_utility_decay = float(goal_selection_utility_decay)
        self.minimum_goal_selection_utility = float(minimum_goal_selection_utility)
        self.goal_permission_trial_limit = int(goal_permission_trial_limit)
        self.minimum_goal_permission_evidence = float(minimum_goal_permission_evidence)
        self.goal_third_component = goal_third_component
        self.goal_route_context = goal_route_context
        self.goal_permission_context = goal_permission_context
        self.goal_reserve_activation = bool(goal_reserve_activation)
        self.maximum_goal_route_byte_weight = float(maximum_goal_route_byte_weight)
        self.goal_route_projection_policy = goal_route_projection_policy
        self.goal_route_timing = goal_route_timing
        self.goal_route_alignment = goal_route_alignment
        self.goal_flow_polarity = goal_flow_polarity
        self.goal_lifecycle = goal_lifecycle
        self.goal_failure_patience = int(goal_failure_patience)
        self.goal_target_alignment_weight = float(goal_target_alignment_weight)
        self.predictive_goal_basins = int(predictive_goal_basins)
        self.predictive_goal_dimensions = int(predictive_goal_dimensions)
        self.predictive_goal_birth_similarity = float(predictive_goal_birth_similarity)
        self.predictive_goal_hysteresis = float(predictive_goal_hysteresis)
        self.predictive_goal_weight = float(predictive_goal_weight)
        self.binding_index_cells = int(binding_index_cells)
        self.binding_role_assemblies = int(binding_role_assemblies)
        self.binding_cells_per_association = int(binding_cells_per_association)
        self.binding_activity_decay = float(binding_activity_decay)
        self.binding_minimum_activity = float(binding_minimum_activity)
        self.binding_role_birth_similarity = float(binding_role_birth_similarity)
        self.binding_surprise_bits = float(binding_surprise_bits)
        self.binding_role_trial_limit = int(binding_role_trial_limit)
        self.binding_minimum_role_utility = float(binding_minimum_role_utility)
        self.binding_address_context = binding_address_context
        self.goal_flow_mode_gate = goal_flow_mode_gate
        self.goal_flow_mode_trials = int(goal_flow_mode_trials)
        self.goal_surprise_scope = goal_surprise_scope
        self.motif_age_exposure = np.zeros(
            self.maximum_motif_events + 1, dtype=np.uint64
        )
        self.motif_age_endings = np.zeros(
            self.maximum_motif_events + 1, dtype=np.uint64
        )
        self.goal_organ = SparseHomeostaticGoalAttractor(
            conditioned_gain=goal_conditioned_gain,
            max_eligibility=self.max_active_temporal_traces,
            max_values=max_goal_assemblies,
            failure_patience=goal_failure_patience,
            component_count=3,
        )
        self.predictive_goal_organ = (
            SparsePredictiveBasinAttractor(
                component_count=predictive_goal_dimensions,
                max_basins=predictive_goal_basins,
                birth_similarity=predictive_goal_birth_similarity,
                hysteresis=predictive_goal_hysteresis,
            )
            if predictive_goal_basins >= 2
            else None
        )
        if binding_index_cells and len(self.temporal_bins) < 2:
            raise ValueError("binding roles require at least two temporal receptors")
        self.binding_role_organ = (
            SparsePredictiveBasinAttractor(
                component_count=self.predictive_goal_dimensions,
                max_basins=binding_role_assemblies,
                birth_similarity=binding_role_birth_similarity,
                hysteresis=0.0,
            )
            if binding_index_cells
            else None
        )
        self.binding_index = (
            FastSparseBindingIndex(
                max_cells=binding_index_cells,
                cells_per_binding=binding_cells_per_association,
                activity_decay=binding_activity_decay,
                minimum_activity=binding_minimum_activity,
            )
            if binding_index_cells
            else None
        )
        self.binding_role_utility_sum = np.zeros(
            binding_role_assemblies, dtype=np.float64
        )
        self.binding_role_utility_evidence = np.zeros(
            binding_role_assemblies, dtype=np.uint64
        )
        self.binding_role_trials = np.zeros(
            binding_role_assemblies, dtype=np.uint64
        )
        self._binding_pending_credit: tuple[int, int, int] | None = None
        self._binding_last_reinstatement: tuple[int, int, int, int] | None = None
        self.binding_raw_window = bytearray()
        self.binding_recent_raw_queries: list[int] = []
        self._predictive_route_signature_cache: dict[
            tuple[int, int], tuple[int, np.ndarray]
        ] = {}
        self.goal_recent_motifs: list[int] = []
        self.goal_surprise_sum = 0.0
        self.goal_surprise_count = 0
        self.goal_surprise_max = 0.0
        self.goal_selections = 0
        self.goal_route_restoration_sum: dict[tuple[int, int], list[float]] = {}
        self.goal_route_restoration_evidence: dict[tuple[int, int], list[float]] = {}
        self.goal_route_eligibility: dict[tuple[int, int], float] = {}
        self.goal_selection_utility: dict[tuple[int, int], float] = {}
        self.goal_selection_evidence: dict[tuple[int, int], int] = {}
        self.goal_permission_restoration_sum: dict[object, float] = {}
        self.goal_permission_restoration_evidence: dict[object, float] = {}
        self.goal_permission_trials: dict[object, int] = {}
        self.goal_permission_eligibility: dict[tuple[int, ...], float] = {}
        self.goal_permission_active_trials: set[tuple[int, ...]] = set()
        self._goal_learning_active = False
        self._last_goal_route_conditioned = False
        self._goal_route_projection_cache: dict[int, float] | None = None
        self.goal_route_context_calls = 0
        self.goal_route_proposals = 0
        self.goal_route_denials = 0
        self.goal_route_switches = 0
        self.goal_route_flow_uses = 0
        self.goal_flow_mode_utility = np.zeros(3, dtype=np.float64)
        self.goal_flow_mode_evidence = np.zeros(3, dtype=np.uint64)
        self.goal_flow_mode_candidates = np.zeros(3, dtype=np.uint64)
        self.goal_flow_mode_applications = np.zeros(3, dtype=np.uint64)
        self.binding_role_observations = 0
        self.binding_formation_events = 0
        self.binding_recall_attempts = 0
        self.binding_reinstatements = 0

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.goal_apical_weight,
            self.maximum_goal_byte_weight,
            self.goal_basin_width,
            self.repetition_window,
            self.goal_route_eligibility_decay,
            self.maximum_goal_route_bonus,
            self.goal_selection_utility_decay,
            self.minimum_goal_selection_utility,
            getattr(self, "goal_permission_trial_limit", 0),
            getattr(self, "minimum_goal_permission_evidence", 0.25),
            getattr(self, "goal_third_component", "capacity_pressure"),
            getattr(self, "goal_route_context", "deficit"),
            getattr(self, "goal_permission_context", "direction"),
            getattr(self, "goal_reserve_activation", False),
            getattr(self, "maximum_goal_route_byte_weight", 0.0),
            getattr(self, "goal_route_projection_policy", "switched_winner"),
            getattr(self, "goal_route_timing", "linear_age"),
            getattr(self, "goal_route_alignment", "target_basin_mass"),
            getattr(self, "goal_flow_polarity", "selected_goal"),
            getattr(self, "goal_lifecycle", "homeostatic"),
            getattr(self, "goal_failure_patience", 4),
            getattr(self, "goal_target_alignment_weight", 1.0),
            getattr(self, "predictive_goal_basins", 0),
            getattr(self, "predictive_goal_dimensions", 8),
            getattr(self, "predictive_goal_birth_similarity", 0.75),
            getattr(self, "predictive_goal_hysteresis", 0.05),
            getattr(self, "predictive_goal_weight", 0.25),
            getattr(self, "binding_index_cells", 0),
            getattr(self, "binding_role_assemblies", 32),
            getattr(self, "binding_cells_per_association", 2),
            getattr(self, "binding_activity_decay", 0.98),
            getattr(self, "binding_minimum_activity", 0.05),
            getattr(self, "binding_role_birth_similarity", 0.8),
            getattr(self, "binding_surprise_bits", 2.0),
            getattr(self, "binding_role_trial_limit", 16),
            getattr(self, "binding_minimum_role_utility", 0.0),
            getattr(self, "binding_address_context", "structural_basin"),
            getattr(self, "binding_role_utility_sum", None),
            getattr(self, "binding_role_utility_evidence", None),
            getattr(self, "binding_role_trials", None),
            getattr(self, "goal_flow_mode_gate", "all"),
            getattr(self, "goal_flow_mode_trials", 32),
            getattr(self, "goal_surprise_scope", "alphabetic"),
            getattr(self, "motif_age_exposure", None),
            getattr(self, "motif_age_endings", None),
            self.goal_route_restoration_sum,
            self.goal_route_restoration_evidence,
            self.goal_selection_utility,
            self.goal_selection_evidence,
            getattr(self, "goal_permission_restoration_sum", {}),
            getattr(self, "goal_permission_restoration_evidence", {}),
            getattr(self, "goal_permission_trials", {}),
            getattr(self, "goal_flow_mode_utility", np.zeros(3, dtype=np.float64)),
            getattr(self, "goal_flow_mode_evidence", np.zeros(3, dtype=np.uint64)),
            getattr(self, "predictive_goal_organ", None),
            getattr(self, "binding_role_organ", None),
        )
        return (
            super().embodied_bytes
            + self.goal_organ.embodied_bytes
            + len(pickle.dumps(payload, protocol=5))
        )

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()
        state["_predictive_route_signature_cache"] = {}
        return state

    @property
    def goal_transient_bytes(self) -> int:
        state = (
            self.goal_route_eligibility,
            getattr(self, "goal_permission_eligibility", {}),
            getattr(self, "goal_permission_active_trials", set()),
            getattr(self, "binding_raw_window", bytearray()),
            getattr(self, "binding_recent_raw_queries", []),
        )
        binding = getattr(self, "binding_index", None)
        binding_bytes = 0 if binding is None else binding.transient_bytes
        return (
            self.goal_organ.transient_bytes
            + binding_bytes
            + len(pickle.dumps(state, protocol=5))
        )

    def _binding_role_signature(
        self, event_row: dict[int, int] | None
    ) -> np.ndarray:
        """Describe a cue by learned future geometry, not its sensory identity."""
        return self._predictive_successor_signature(event_row)

    def _binding_filler(
        self,
        target_id: int,
        active_route_keys: tuple[tuple[int, int], ...],
    ) -> tuple[int, int] | None:
        """Let distinctive active cortical assemblies compete as fillers."""
        candidates = {
            (receptor, source)
            for receptor, source in active_route_keys
            if source != target_id
            and source not in getattr(self, "temporal_recalled_sources", set())
            and 0 <= source < len(self.word_forms)
        }
        if not candidates:
            return None

        def novelty(candidate: tuple[int, int]) -> tuple[float, int, int]:
            receptor, source = candidate
            activity = max(float(self.motif_activity[source]), 1e-12)
            return (
                len(self.word_forms[source]) / np.sqrt(activity),
                receptor,
                -source,
            )

        return max(candidates, key=novelty)

    def _binding_query_sources(
        self,
        target_id: int,
        active_route_keys: tuple[tuple[int, int], ...],
        *,
        limit: int = 4,
    ) -> tuple[int, ...]:
        """Select a few locally active assemblies as possible read addresses."""
        candidates = {
            (receptor, source)
            for receptor, source in active_route_keys
            if source != target_id
            and source not in getattr(self, "temporal_recalled_sources", set())
            and 0 <= source < len(self.word_forms)
            and receptor <= 8
        }
        ranked = sorted(
            candidates,
            key=lambda item: (
                item[0],
                float(self.motif_activity[item[1]])
                / max(len(self.word_forms[item[1]]), 1),
                item[1],
            ),
        )
        return tuple(source for _, source in ranked[:limit])

    def _binding_content_receptor(self, filler: int) -> int:
        """Choose a learned output-effective receptor for recalled content."""
        candidates = [
            receptor
            for receptor in self.temporal_bins
            if self.route_utility.get((receptor, filler), 0.0)
            > self.minimum_route_utility_bits
        ]
        if not candidates:
            return int(self.temporal_bins[0])
        return int(
            max(
                candidates,
                key=lambda receptor: (
                    self.route_utility[(receptor, filler)],
                    self.route_totals[(receptor, filler)],
                ),
            )
        )

    def _advance_cortical_query_binding(
        self,
        target_id: int,
        active_route_keys: tuple[tuple[int, int], ...],
        mean_surprise_bits: float,
    ) -> None:
        """Bind arriving content to active cortical queries by coincidence."""
        index = self.binding_index
        if index is None or mean_surprise_bits < self.binding_surprise_bits:
            return
        queries = self._binding_query_sources(target_id, active_route_keys)
        if not queries:
            return
        self.binding_recall_attempts += len(queries)
        recalls = [
            (index.recall(query, 0, exclude_filler=target_id), query)
            for query in queries
        ]
        viable = [
            (recall, query)
            for recall, query in recalls
            if recall.filler is not None
            and recall.cortical_role is not None
            and recall.filler not in self.temporal_last_seen
            and 0 <= recall.filler < len(self.word_forms)
        ]
        if viable:
            recalled, query = max(
                viable,
                key=lambda item: (
                    item[0].confidence,
                    -float(self.motif_activity[item[1]]),
                    -item[1],
                ),
            )
            filler = int(recalled.filler)
            receptor = int(recalled.cortical_role)
            self.temporal_last_seen[filler] = self.temporal_clock - receptor + 1
            if not hasattr(self, "temporal_recalled_sources"):
                self.temporal_recalled_sources = set()
            self.temporal_recalled_sources.add(filler)
            self.binding_reinstatements += 1
            self._binding_last_reinstatement = query, 0, filler, receptor
        # This is episode working state, not durable cortical learning. It must
        # remain writable from observed bytes during frozen evaluation.
        receptor = self._binding_content_receptor(target_id)
        for query in queries:
            index.bind(query, 0, target_id, cortical_role=receptor)
            self.binding_formation_events += 1

    def _bind_raw_queries_to_content(self, target_id: int) -> None:
        """Write fast sensory addresses only for output-effective content."""
        index = self.binding_index
        if index is None:
            return
        receptor = self._binding_content_receptor(target_id)
        if self.route_utility.get((receptor, target_id), 0.0) <= (
            self.minimum_route_utility_bits
        ):
            return
        # Three-byte-spaced receptors retain alignment while spanning 24 bytes.
        queries = tuple(dict.fromkeys(self.binding_recent_raw_queries[-24::3]))
        for query in queries:
            index.bind(query, 0, target_id, cortical_role=receptor)
            self.binding_formation_events += 1

    def _advance_raw_binding_query(self, byte_id: int) -> None:
        """Use a compact partial sensory cue before a slow motif completes."""
        if (
            self.binding_index is None
            or getattr(self, "binding_address_context", "structural_basin")
            != "raw_query"
        ):
            return
        self.binding_raw_window.append(int(byte_id))
        if len(self.binding_raw_window) > 3:
            del self.binding_raw_window[0]
        if len(self.binding_raw_window) < 3:
            return
        query = int.from_bytes(self.binding_raw_window, "big")
        self.binding_recent_raw_queries.append(query)
        if len(self.binding_recent_raw_queries) > 32:
            del self.binding_recent_raw_queries[0]
        self.binding_recall_attempts += 1
        recalled = self.binding_index.recall(query, 0)
        filler = recalled.filler
        receptor = recalled.cortical_role
        if (
            filler is None
            or receptor is None
            or filler in self.temporal_last_seen
            or not 0 <= filler < len(self.word_forms)
        ):
            return
        self.temporal_last_seen[filler] = self.temporal_clock - receptor + 1
        if not hasattr(self, "temporal_recalled_sources"):
            self.temporal_recalled_sources = set()
        self.temporal_recalled_sources.add(filler)
        self.binding_reinstatements += 1
        self._binding_last_reinstatement = query, 0, filler, receptor

    def _binding_role_utility(self, role: int) -> float:
        return float(
            self.binding_role_utility_sum[role]
            / (self.binding_role_utility_evidence[role] + 1.0)
        )

    def _binding_role_is_permitted(self, role: int) -> bool:
        return bool(
            self.binding_role_trials[role] < self.binding_role_trial_limit
            or self._binding_role_utility(role) > self.binding_minimum_role_utility
        )

    def _settle_binding_credit(
        self,
        target_id: int,
        active_route_keys: tuple[tuple[int, int], ...],
        *,
        learn: bool,
    ) -> None:
        pending = self._binding_pending_credit
        self._binding_pending_credit = None
        if pending is None or not learn:
            return
        role, filler, receptor = pending
        prior = self._target_prior(target_id)
        gains = [
            log2(
                max(self._route_probability(key, target_id), 1e-300)
                / max(prior, 1e-300)
            )
            for key in active_route_keys
            if key == (receptor, filler) and key in self.route_counts
        ]
        gain = max(gains, default=0.0)
        self.binding_role_utility_sum[role] += gain
        self.binding_role_utility_evidence[role] += 1

    def _advance_transient_binding(
        self,
        target_id: int,
        active_route_keys: tuple[tuple[int, int], ...],
        event_row: dict[int, int] | None,
        mean_surprise_bits: float,
        *,
        learn: bool,
    ) -> None:
        role_organ = getattr(self, "binding_role_organ", None)
        index = getattr(self, "binding_index", None)
        basin_organ = getattr(self, "predictive_goal_organ", None)
        if role_organ is None or index is None or basin_organ is None:
            return
        self._binding_last_reinstatement = None
        if (
            getattr(self, "binding_address_context", "structural_basin")
            == "cortical_query"
        ):
            role_organ.observe(
                self._binding_role_signature(event_row), learn=learn
            )
            self._advance_cortical_query_binding(
                target_id,
                active_route_keys,
                mean_surprise_bits,
            )
            return
        if (
            getattr(self, "binding_address_context", "structural_basin")
            == "raw_query"
        ):
            role_organ.observe(
                self._binding_role_signature(event_row), learn=learn
            )
            if getattr(self, "goal_surprise_max", 0.0) >= self.binding_surprise_bits:
                self._bind_raw_queries_to_content(target_id)
            return
        self._settle_binding_credit(target_id, active_route_keys, learn=learn)
        signature = self._binding_role_signature(event_row)
        role = role_organ.observe(signature, learn=learn)
        basin = (
            0
            if getattr(self, "binding_address_context", "structural_basin")
            == "query_only"
            else basin_organ.current_basin
        )
        if role is None or basin is None:
            return
        self.binding_role_observations += 1
        if mean_surprise_bits < self.binding_surprise_bits:
            return
        self.binding_recall_attempts += 1
        self.binding_role_trials[role] += 1
        if not self._binding_role_is_permitted(role):
            binding_filler = self._binding_filler(target_id, active_route_keys)
            if binding_filler is not None:
                receptor, filler = binding_filler
                index.bind(basin, role, filler, cortical_role=receptor)
                self.binding_formation_events += 1
            return
        recalled = index.recall(basin, role, exclude_filler=target_id)
        filler = recalled.filler
        receptor = recalled.cortical_role
        if (
            filler is not None
            and receptor is not None
            and 0 <= filler < len(self.word_forms)
            and filler not in self.temporal_last_seen
        ):
            self.temporal_last_seen[filler] = self.temporal_clock - receptor + 1
            if not hasattr(self, "temporal_recalled_sources"):
                self.temporal_recalled_sources = set()
            self.temporal_recalled_sources.add(filler)
            self.binding_reinstatements += 1
            self._binding_pending_credit = role, filler, receptor
            self._binding_last_reinstatement = basin, role, filler, receptor
        binding_filler = self._binding_filler(target_id, active_route_keys)
        if binding_filler is None:
            return
        receptor, filler = binding_filler
        index.bind(basin, role, filler, cortical_role=receptor)
        self.binding_formation_events += 1

    def _goal_basin(self, assembly: int) -> int | None:
        signature = self.goal_organ.restoration_signature(assembly)
        if len(signature) and float(max(signature)) > 0.0:
            return int(max(range(len(signature)), key=signature.__getitem__))
        value = self.goal_organ.value(assembly)
        if value < self.goal_organ.minimum_goal_value:
            return None
        return int(value / self.goal_basin_width + 0.5)

    def _same_goal_basin(self, left: int, right: int) -> bool:
        basin = self._goal_basin(left)
        return basin is not None and basin == self._goal_basin(right)

    def _goal_route_value(self, key: tuple[int, int], mode: int) -> float:
        sums = self.goal_route_restoration_sum.get(key)
        evidence = self.goal_route_restoration_evidence.get(key)
        if sums is None or evidence is None:
            return 0.0
        return sums[mode] / (evidence[mode] + self.goal_organ.value_prior)

    def _route_goal_basin_mass(self, key: tuple[int, int], goal: int) -> float:
        """Return locally predicted mass that shares the goal's restoration basin."""
        basin = self._goal_basin(goal)
        row = self.route_counts.get(key)
        if basin is None or not row:
            return 0.0
        matched = sum(
            observations
            for target, observations in row.items()
            if self._goal_basin(target) == basin
        )
        return matched / self.route_totals[key]

    def _predictive_successor_signature(self, row: dict[int, int] | None) -> np.ndarray:
        dimensions = getattr(self, "predictive_goal_dimensions", 8)
        signature = np.zeros(dimensions, dtype=np.float64)
        if not row:
            return signature
        total = 0
        for target, observations in row.items():
            if observations <= 0 or not 0 <= target < len(self.word_forms):
                continue
            form = self.word_forms[target]
            if not form:
                continue
            byte_id = form[0]
            for component in range(dimensions):
                signature[component] += observations * (
                    1.0 if byte_id & (1 << component) else -1.0
                )
            total += observations
        if total:
            signature /= total
        return signature

    def _predictive_route_signature(self, key: tuple[int, int]) -> np.ndarray:
        total = self.route_totals.get(key, 0)
        cache = getattr(self, "_predictive_route_signature_cache", None)
        if cache is None:
            cache = self._predictive_route_signature_cache = {}
        cached = cache.get(key)
        if cached is not None and cached[0] == total:
            return cached[1]
        signature = self._predictive_successor_signature(self.route_counts.get(key))
        cache[key] = total, signature
        return signature

    def _route_goal_alignment(self, key: tuple[int, int], goal: int | None) -> float:
        polarity = getattr(self, "goal_flow_polarity", "selected_goal")
        if polarity in {"regulatory_mode", "regulatory_goal", "regulatory_basin"}:
            mode = self.goal_organ.current_mode
            sums = self.goal_route_restoration_sum.get(key)
            evidence = self.goal_route_restoration_evidence.get(key)
            if mode is None or sums is None or evidence is None:
                return 0.0
            route_signature = np.maximum(
                np.asarray(sums) / (np.asarray(evidence) + self.goal_organ.value_prior),
                0.0,
            )
            norm = float(np.linalg.norm(route_signature))
            if norm == 0.0:
                return 0.0
            mode_alignment = float(route_signature[mode] / norm)
            if polarity == "regulatory_mode":
                return mode_alignment
            if polarity == "regulatory_basin":
                organ = getattr(self, "predictive_goal_organ", None)
                basin_weight = getattr(self, "predictive_goal_weight", 0.25)
                if organ is None or organ.current_basin is None:
                    return mode_alignment * (1.0 - basin_weight)
                basin_alignment = max(
                    0.0, organ.alignment(self._predictive_route_signature(key))
                )
                return mode_alignment * (
                    (1.0 - basin_weight) + basin_weight * basin_alignment
                )
            target_weight = getattr(self, "goal_target_alignment_weight", 1.0)
            if goal is None:
                return mode_alignment * (1.0 - target_weight)
            goal_signature = np.maximum(
                self.goal_organ.restoration_signature(goal), 0.0
            )
            goal_norm = float(np.linalg.norm(goal_signature))
            if goal_norm == 0.0:
                return 0.0
            goal_alignment = float(
                np.dot(route_signature, goal_signature) / (norm * goal_norm)
            )
            return mode_alignment * (
                (1.0 - target_weight) + target_weight * goal_alignment
            )
        if goal is None:
            return 0.0
        if (
            getattr(self, "goal_route_alignment", "target_basin_mass")
            == "target_basin_mass"
        ):
            return self._route_goal_basin_mass(key, goal)
        goal_signature = np.maximum(self.goal_organ.restoration_signature(goal), 0.0)
        sums = self.goal_route_restoration_sum.get(key)
        evidence = self.goal_route_restoration_evidence.get(key)
        if sums is None or evidence is None or not np.any(goal_signature):
            return 0.0
        route_signature = np.maximum(
            np.asarray(sums) / (np.asarray(evidence) + self.goal_organ.value_prior),
            0.0,
        )
        goal_norm = float(np.linalg.norm(goal_signature))
        route_norm = float(np.linalg.norm(route_signature))
        if goal_norm == 0.0 or route_norm == 0.0:
            return 0.0
        return float(np.dot(goal_signature, route_signature) / (goal_norm * route_norm))

    def _route_winners(
        self,
        candidates: list[tuple[int, int]],
        mode: int,
        *,
        baseline_candidates: list[tuple[int, int]] | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        baseline_pool = (
            candidates if baseline_candidates is None else baseline_candidates
        )
        baseline = max(
            baseline_pool,
            key=lambda key: (self.route_utility[key], self.route_totals[key], -key[0]),
        )
        authority = self.goal_apical_weight / (1.0 + self.goal_apical_weight)
        goal = self.goal_organ.current_goal

        def context_value(key: tuple[int, int]) -> float:
            if (
                getattr(self, "goal_route_context", "deficit") == "restorative_basin"
                and goal is not None
            ):
                return self._route_goal_alignment(key, goal)
            return self._goal_route_value(key, mode)

        conditioned = max(
            candidates,
            key=lambda key: (
                self.route_utility[key]
                + self.maximum_goal_route_bonus * authority * context_value(key),
                self.route_totals[key],
                -key[0],
            ),
        )
        return baseline, conditioned

    def _conditioned_route_candidates(
        self,
        route_keys: tuple[tuple[int, int], ...],
        ordinary: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if (
            getattr(self, "goal_route_context", "deficit") != "restorative_basin"
            or self.goal_organ.current_goal is None
            or not getattr(self, "goal_reserve_activation", False)
        ):
            return ordinary
        return [
            key
            for key in route_keys
            if key in self.route_counts and self.route_utility[key] > 0.0
        ]

    def _selected_route_key(self) -> tuple[int, int] | None:
        self._last_goal_route_conditioned = False
        route_keys = self._active_route_keys()
        ordinary = [
            key
            for key in route_keys
            if key in self.route_counts
            and self.route_utility[key] > self.minimum_route_utility_bits
        ]
        mode = getattr(self.goal_organ, "current_mode", None)
        if not ordinary:
            if (
                mode is None
                or self.goal_apical_weight == 0.0
                or getattr(self, "goal_route_context", "deficit") != "restorative_basin"
                or self.goal_organ.current_goal is None
                or not getattr(self, "goal_reserve_activation", False)
            ):
                return None
            candidates = self._conditioned_route_candidates(route_keys, ordinary)
            if not candidates:
                return None
            authority = self.goal_apical_weight / (1.0 + self.goal_apical_weight)
            conditioned = max(
                candidates,
                key=lambda key: (
                    self.route_utility[key]
                    + self.maximum_goal_route_bonus
                    * authority
                    * self._route_goal_alignment(key, self.goal_organ.current_goal),
                    self.route_totals[key],
                    -key[0],
                ),
            )
            conditioned_score = self.route_utility[
                conditioned
            ] + self.maximum_goal_route_bonus * authority * self._route_goal_alignment(
                conditioned, self.goal_organ.current_goal
            )
            if conditioned_score <= self.minimum_route_utility_bits:
                return None
            self.goal_route_context_calls = (
                getattr(self, "goal_route_context_calls", 0) + 1
            )
            self.goal_route_proposals = getattr(self, "goal_route_proposals", 0) + 1
            permission = (mode, 0)
            if not self._goal_route_is_permitted(permission):
                self.goal_route_denials = getattr(self, "goal_route_denials", 0) + 1
                return None
            if self._goal_learning_active:
                eligibility_key, _ = self._goal_permission_keys(permission)
                self.goal_permission_eligibility[eligibility_key] = 1.0
            self.goal_route_switches = getattr(self, "goal_route_switches", 0) + 1
            self._last_goal_route_conditioned = True
            return conditioned
        baseline = max(
            ordinary,
            key=lambda key: (self.route_utility[key], self.route_totals[key], -key[0]),
        )
        if mode is None or self.goal_apical_weight == 0.0:
            return baseline
        self.goal_route_context_calls = getattr(self, "goal_route_context_calls", 0) + 1
        candidates = self._conditioned_route_candidates(route_keys, ordinary)
        _, conditioned = self._route_winners(
            candidates, mode, baseline_candidates=ordinary
        )
        if conditioned == baseline:
            return baseline
        self.goal_route_proposals = getattr(self, "goal_route_proposals", 0) + 1
        permission = self._goal_selection_key(mode, baseline, conditioned)
        if not self._goal_route_is_permitted(permission):
            self.goal_route_denials = getattr(self, "goal_route_denials", 0) + 1
            return baseline
        if self._goal_learning_active:
            eligibility_key, _ = self._goal_permission_keys(permission)
            self.goal_permission_eligibility[eligibility_key] = 1.0
        self.goal_route_switches = getattr(self, "goal_route_switches", 0) + 1
        self._last_goal_route_conditioned = True
        return conditioned

    def _goal_route_is_permitted(self, permission: tuple[int, int]) -> bool:
        trial_limit = getattr(self, "goal_permission_trial_limit", 0)
        if trial_limit <= 0:
            return (
                self.goal_selection_utility.get(permission, 0.0)
                > self.minimum_goal_selection_utility
            )
        eligibility_key, learning_key = self._goal_permission_keys(permission)
        evidence = self.goal_permission_restoration_evidence.get(learning_key, 0.0)
        restoration = self.goal_permission_restoration_sum.get(learning_key, 0.0)
        if (
            evidence >= getattr(self, "minimum_goal_permission_evidence", 0.25)
            and restoration > 0.0
        ):
            return True
        if not self._goal_learning_active:
            return False
        if eligibility_key in self.goal_permission_active_trials:
            return True
        trials = self.goal_permission_trials.get(learning_key, 0)
        if trials >= trial_limit:
            return False
        self.goal_permission_trials[learning_key] = trials + 1
        self.goal_permission_active_trials.add(eligibility_key)
        return True

    def _goal_permission_keys(
        self, permission: tuple[int, int]
    ) -> tuple[tuple[int, ...], object]:
        mode, temporal_direction = permission
        if getattr(self, "goal_permission_context", "direction") == "goal_basin":
            goal = self.goal_organ.current_goal
            basin = -1 if goal is None else self._goal_basin(goal)
            basin = -1 if basin is None else basin
            return (mode, temporal_direction, basin), (basin, temporal_direction)
        return permission, temporal_direction

    @staticmethod
    def _goal_selection_key(
        mode: int,
        baseline: tuple[int, int],
        conditioned: tuple[int, int],
    ) -> tuple[int, int]:
        temporal_direction = (conditioned[0] > baseline[0]) - (
            conditioned[0] < baseline[0]
        )
        return mode, temporal_direction

    def _update_goal_selection_utility(
        self,
        keys: tuple[tuple[int, int], ...],
        target_id: int,
        mode: int | None,
    ) -> None:
        if mode is None:
            return
        ordinary = [
            key
            for key in keys
            if key in self.route_counts
            and self.route_utility[key] > self.minimum_route_utility_bits
        ]
        if not ordinary:
            return
        candidates = self._conditioned_route_candidates(keys, ordinary)
        if len(candidates) < 2:
            return
        baseline, conditioned = self._route_winners(
            candidates, mode, baseline_candidates=ordinary
        )
        if conditioned == baseline:
            return
        permission = self._goal_selection_key(mode, baseline, conditioned)
        baseline_probability = self._route_probability(baseline, target_id)
        conditioned_probability = self._route_probability(conditioned, target_id)
        gain = log2(
            max(conditioned_probability, 1e-300) / max(baseline_probability, 1e-300)
        )
        decay = self.goal_selection_utility_decay
        self.goal_selection_utility[permission] = (
            decay * self.goal_selection_utility.get(permission, 0.0)
            + (1.0 - decay) * gain
        )
        self.goal_selection_evidence[permission] = (
            self.goal_selection_evidence.get(permission, 0) + 1
        )

    def _advance_goal_route_credit(
        self,
        keys: tuple[tuple[int, int], ...],
        signals: object,
        *,
        learn: bool,
    ) -> None:
        self.goal_route_eligibility = {
            key: value * self.goal_route_eligibility_decay
            for key, value in self.goal_route_eligibility.items()
            if key in self.route_counts
            and value * self.goal_route_eligibility_decay > 1e-6
        }
        for key in keys:
            if key in self.route_counts:
                self.goal_route_eligibility[key] = 1.0
        if not learn:
            getattr(self, "goal_permission_active_trials", set()).clear()
            return
        component_signals = tuple(float(value) for value in signals)
        for key, eligibility in self.goal_route_eligibility.items():
            sums = self.goal_route_restoration_sum.setdefault(key, [0.0] * 3)
            evidence = self.goal_route_restoration_evidence.setdefault(key, [0.0] * 3)
            for mode, signal in enumerate(component_signals):
                sums[mode] += eligibility * signal
                evidence[mode] += eligibility * abs(signal)
        if getattr(self, "goal_permission_trial_limit", 0) <= 0:
            getattr(self, "goal_permission_active_trials", set()).clear()
            return
        self.goal_permission_eligibility = {
            permission: eligibility * self.goal_route_eligibility_decay
            for permission, eligibility in self.goal_permission_eligibility.items()
            if eligibility * self.goal_route_eligibility_decay > 1e-6
        }
        for permission, eligibility in self.goal_permission_eligibility.items():
            mode, temporal_direction = permission[:2]
            learning_key: object = temporal_direction
            if len(permission) == 3:
                learning_key = permission[2], temporal_direction
            signal = component_signals[mode]
            self.goal_permission_restoration_sum[learning_key] = (
                self.goal_permission_restoration_sum.get(learning_key, 0.0)
                + eligibility * signal
            )
            self.goal_permission_restoration_evidence[learning_key] = (
                self.goal_permission_restoration_evidence.get(learning_key, 0.0)
                + eligibility * abs(signal)
            )
        self.goal_permission_active_trials.clear()

    def _evict_competitor(self, tissue: str, bank_index: int, key: object) -> None:
        if tissue == "route":
            route_key = key
            getattr(self, "goal_route_restoration_sum", {}).pop(route_key, None)
            getattr(self, "goal_route_restoration_evidence", {}).pop(route_key, None)
            getattr(self, "goal_route_eligibility", {}).pop(route_key, None)
        super()._evict_competitor(tissue, bank_index, key)

    def _recycle_motif(
        self, word_id: int, new_form: bytes, initial_activity: float
    ) -> None:
        stale = [key for key in self.goal_route_restoration_sum if key[1] == word_id]
        for key in stale:
            self.goal_route_restoration_sum.pop(key, None)
            self.goal_route_restoration_evidence.pop(key, None)
            self.goal_route_eligibility.pop(key, None)
        self.goal_organ.restoration_sum.pop(word_id, None)
        self.goal_organ.restoration_evidence.pop(word_id, None)
        self.goal_organ.component_restoration_sum.pop(word_id, None)
        self.goal_organ.component_restoration_evidence.pop(word_id, None)
        super()._recycle_motif(word_id, new_form, initial_activity)

    def _goal_byte_distribution(
        self, row: dict[int, int] | None = None
    ) -> dict[int, float]:
        goal = self.goal_organ.current_goal
        if goal is None or not 0 <= goal < len(self.word_forms):
            return {}
        basin = self._goal_basin(goal)
        if basin is None:
            return {}
        if row is None:
            row = super()._active_event_row()
        if not row:
            return {}
        prefix = bytes(self.current_word)
        projected: dict[int, float] = {}
        for candidate, accessibility in row.items():
            if (
                accessibility <= 0
                or not 0 <= candidate < len(self.word_forms)
                or self._goal_basin(candidate) != basin
            ):
                continue
            form = self.word_forms[candidate]
            target: int | None = None
            if form.startswith(prefix) and len(prefix) < len(form):
                target = form[len(prefix)]
            elif len(prefix) >= self.minimum_motif_events and form:
                target = form[0]
            if target is not None:
                weight = accessibility * self.goal_organ.value(candidate)
                projected[target] = projected.get(target, 0.0) + weight
        total = sum(projected.values())
        if total <= 0.0:
            return {}
        return {target: weight / total for target, weight in projected.items()}

    def _goal_byte_weight(self) -> float:
        gain = self.goal_apical_weight
        return self.maximum_goal_byte_weight * gain / (1.0 + gain)

    def _goal_route_next_motif_distribution(self) -> dict[int, float]:
        """Project a selected local route toward the first event of its successor."""
        if (
            len(self.current_word) < self.minimum_motif_events
            or self.goal_apical_weight == 0.0
            or self.maximum_goal_route_byte_weight == 0.0
        ):
            return {}
        cached = getattr(self, "_goal_route_projection_cache", None)
        if cached is not None:
            return cached
        if (
            getattr(self, "goal_route_projection_policy", "switched_winner")
            == "aligned_flow"
        ):
            goal = self.goal_organ.current_goal
            polarity = getattr(self, "goal_flow_polarity", "selected_goal")
            mode = self.goal_organ.current_mode
            if (
                (polarity == "selected_goal" and goal is None)
                or (polarity == "regulatory_mode" and mode is None)
                or (polarity == "regulatory_goal" and mode is None)
                or (polarity == "regulatory_basin" and mode is None)
            ):
                self._goal_route_projection_cache = {}
                return {}
            self.goal_route_context_calls = (
                getattr(self, "goal_route_context_calls", 0) + 1
            )
            projected: dict[int, float] = {}
            for key in self._active_route_keys():
                if (
                    key not in self.route_counts
                    or self.route_utility[key] <= self.minimum_route_utility_bits
                ):
                    continue
                flow = self.route_utility[key] * self._route_goal_alignment(key, goal)
                if flow <= 0.0:
                    continue
                row = self.route_counts[key]
                total = self.route_totals[key]
                for candidate, observations in row.items():
                    if observations <= 0 or not 0 <= candidate < len(self.word_forms):
                        continue
                    form = self.word_forms[candidate]
                    if form:
                        projected[form[0]] = projected.get(form[0], 0.0) + (
                            flow * observations / total
                        )
            total_flow = sum(projected.values())
            if total_flow <= 0.0:
                self._goal_route_projection_cache = {}
                return {}
            self.goal_route_flow_uses = getattr(self, "goal_route_flow_uses", 0) + 1
            self._goal_route_projection_cache = {
                target: value / total_flow for target, value in projected.items()
            }
            return self._goal_route_projection_cache
        key = self._selected_route_key()
        if key is None or not self._last_goal_route_conditioned:
            self._goal_route_projection_cache = {}
            return {}
        projected: dict[int, float] = {}
        for candidate, observations in self.route_counts[key].items():
            if observations <= 0 or not 0 <= candidate < len(self.word_forms):
                continue
            form = self.word_forms[candidate]
            if form:
                projected[form[0]] = projected.get(form[0], 0.0) + observations
        total = sum(projected.values())
        if total <= 0.0:
            self._goal_route_projection_cache = {}
            return {}
        self._goal_route_projection_cache = {
            target: count / total for target, count in projected.items()
        }
        return self._goal_route_projection_cache

    def _goal_route_next_motif_weight(self) -> float:
        span = self.maximum_motif_events - self.minimum_motif_events + 1
        progress = min(
            1.0,
            max(0, len(self.current_word) - self.minimum_motif_events + 1) / span,
        )
        if getattr(self, "goal_route_timing", "linear_age") == "learned_hazard":
            age = min(len(self.current_word), self.maximum_motif_events)
            exposure = float(self.motif_age_exposure[age])
            endings = float(self.motif_age_endings[age])
            prior_strength = 8.0
            progress = (endings + prior_strength * progress) / (
                exposure + prior_strength
            )
        authority = self.goal_apical_weight / (1.0 + self.goal_apical_weight)
        return self.maximum_goal_route_byte_weight * authority * progress

    def _goal_flow_mode_is_permitted(self, mode: int | None) -> bool:
        """Let local causal experience decide which regulatory flows may act."""
        if getattr(self, "goal_flow_mode_gate", "all") == "all":
            return True
        if mode is None or not 0 <= mode < 3:
            return False
        evidence = int(
            getattr(self, "goal_flow_mode_evidence", np.zeros(3, dtype=np.uint64))[mode]
        )
        if evidence < getattr(self, "goal_flow_mode_trials", 32):
            return True
        return (
            float(
                getattr(self, "goal_flow_mode_utility", np.zeros(3, dtype=np.float64))[
                    mode
                ]
            )
            > 0.0
        )

    def _record_goal_flow_mode_outcome(
        self, mode: int | None, baseline: float, candidate: float, *, applied: bool
    ) -> None:
        """Update one mode-local permission synapse after observing its outcome."""
        if mode is None or not 0 <= mode < 3:
            return
        if not hasattr(self, "goal_flow_mode_utility"):
            self.goal_flow_mode_utility = np.zeros(3, dtype=np.float64)
            self.goal_flow_mode_evidence = np.zeros(3, dtype=np.uint64)
            self.goal_flow_mode_candidates = np.zeros(3, dtype=np.uint64)
            self.goal_flow_mode_applications = np.zeros(3, dtype=np.uint64)
        self.goal_flow_mode_candidates[mode] += 1
        if applied:
            self.goal_flow_mode_applications[mode] += 1
        if not self._goal_learning_active:
            return
        bit_gain = log2(max(candidate, 1e-300) / max(baseline, 1e-300))
        self.goal_flow_mode_utility[mode] += bit_gain
        self.goal_flow_mode_evidence[mode] += 1

    def _distribution(self):
        distribution = super()._distribution()
        goal_distribution = self._goal_byte_distribution()
        weight = self._goal_byte_weight()
        if goal_distribution and weight != 0.0:
            distribution *= 1.0 - weight
            for target, probability in goal_distribution.items():
                distribution[target] += weight * probability
        route_distribution = self._goal_route_next_motif_distribution()
        route_weight = self._goal_route_next_motif_weight()
        mode = getattr(self.goal_organ, "current_mode", None)
        if (
            route_distribution
            and route_weight != 0.0
            and self._goal_flow_mode_is_permitted(mode)
        ):
            distribution *= 1.0 - route_weight
            for target, probability in route_distribution.items():
                distribution[target] += route_weight * probability
        return distribution

    def _observed_probability(self, byte_id: int) -> float:
        probability = super()._observed_probability(byte_id)
        goal_distribution = self._goal_byte_distribution()
        weight = self._goal_byte_weight()
        if goal_distribution and weight != 0.0:
            probability = (1.0 - weight) * probability + weight * (
                goal_distribution.get(byte_id, 0.0)
            )
        route_distribution = self._goal_route_next_motif_distribution()
        route_weight = self._goal_route_next_motif_weight()
        if route_distribution and route_weight != 0.0:
            mode = getattr(self.goal_organ, "current_mode", None)
            baseline = probability
            candidate = (1.0 - route_weight) * probability + route_weight * (
                route_distribution.get(byte_id, 0.0)
            )
            permitted = self._goal_flow_mode_is_permitted(mode)
            self._record_goal_flow_mode_outcome(
                mode, baseline, candidate, applied=permitted
            )
            if permitted:
                probability = candidate
        return probability

    def _temporal_disagreement(self, route_keys: tuple[tuple[int, int], ...]) -> float:
        winner_counts: dict[int, int] = {}
        predictors = 0
        for key in route_keys:
            if (
                key not in self.route_counts
                or self.route_utility[key] <= self.minimum_route_utility_bits
            ):
                continue
            row = self.route_counts[key]
            if not row:
                continue
            winner = max(row, key=lambda target: (row[target], -target))
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            predictors += 1
        if predictors < 2:
            return 0.0
        agreement = max(winner_counts.values()) / predictors
        return 1.0 - agreement

    def _goal_viability_components(
        self,
        target_id: int,
        route_keys: tuple[tuple[int, int], ...] = (),
    ) -> tuple[float, float, float]:
        surprise = self.goal_surprise_sum / max(1, self.goal_surprise_count)
        normalized_surprise = min(1.0, surprise / 8.0)
        repetition = self.goal_recent_motifs.count(target_id) / max(
            1, len(self.goal_recent_motifs)
        )
        if getattr(self, "goal_third_component", "capacity_pressure") == (
            "temporal_disagreement"
        ):
            third_component = self._temporal_disagreement(route_keys)
        else:
            route_pressure = len(self.route_counts) / max(1, self.max_route_paths)
            motif_pressure = len(self.word_forms) / max(1, self.max_motif_assemblies)
            third_component = 0.5 * (route_pressure + motif_pressure)
        return (
            normalized_surprise,
            repetition,
            third_component,
        )

    def _goal_viability_deviation(self, target_id: int) -> float:
        return sum(self._goal_viability_components(target_id))

    def _goal_scores_byte(self, byte_id: int) -> bool:
        if getattr(self, "goal_surprise_scope", "alphabetic") == "all_bytes":
            return True
        return 65 <= byte_id <= 90 or 97 <= byte_id <= 122

    def _complete_word(self, *, learn: bool) -> None:
        if not self.current_word:
            return
        mean_surprise_bits = self.goal_surprise_sum / max(1, self.goal_surprise_count)
        if learn:
            length = min(len(self.current_word), self.maximum_motif_events)
            self.motif_age_exposure[self.minimum_motif_events : length + 1] += 1
            self.motif_age_endings[length] += 1
        self._goal_route_projection_cache = None
        form = self._normalize_event_form(bytes(self.current_word))
        active_route_keys = self._active_route_keys()
        target_before_update = self.word_ids.get(form)
        mode = getattr(self.goal_organ, "current_mode", None)
        if learn and target_before_update is not None:
            self._update_goal_selection_utility(
                active_route_keys, target_before_update, mode
            )
        super()._complete_word(learn=learn)
        target_id = self.word_ids.get(form)
        if target_id is None:
            self.goal_surprise_sum = 0.0
            self.goal_surprise_count = 0
            self.goal_surprise_max = 0.0
            return
        predictive_goal = getattr(self, "predictive_goal_organ", None)
        event_row = None
        if predictive_goal is not None:
            event_row = PredictiveMotifOpponentLearner._active_event_row(self)
            predictive_goal.observe(
                self._predictive_successor_signature(event_row), learn=learn
            )
        self._advance_transient_binding(
            target_id,
            active_route_keys,
            event_row,
            mean_surprise_bits,
            learn=learn,
        )
        if getattr(self, "goal_lifecycle", "homeostatic") == "target_or_failure":
            self.goal_organ.achieve(target_id)
        self.goal_organ.observe_event(target_id, learn=learn)
        components = self._goal_viability_components(target_id, active_route_keys)
        deviation = sum(components)
        self.goal_organ.observe_deviation(
            deviation,
            learn=learn,
            dissolve_on_restoration=(
                getattr(self, "goal_lifecycle", "homeostatic") == "homeostatic"
            ),
        )
        signals = self.goal_organ.observe_components(components, learn=learn)
        self._advance_goal_route_credit(active_route_keys, signals, learn=learn)
        self.goal_recent_motifs.append(target_id)
        if len(self.goal_recent_motifs) > self.repetition_window:
            del self.goal_recent_motifs[0]
        if self.goal_organ.current_goal is None:
            row = super()._active_event_row()
            if row:
                total = sum(row.values())
                candidates = {key: value / total for key, value in row.items()}
                if self.goal_organ.select_goal(candidates) is not None:
                    self.goal_selections += 1
        self.goal_surprise_sum = 0.0
        self.goal_surprise_count = 0
        self.goal_surprise_max = 0.0

    def _observe_probability_only(self, byte_id: int, *, learn: bool) -> float | None:
        self._goal_learning_active = learn
        probability = None if self.steps == 0 else self._observed_probability(byte_id)
        super()._observe_probability_only(byte_id, learn=learn)
        if probability is not None and self._goal_scores_byte(byte_id):
            surprise = -log2(max(probability, 1e-300))
            self.goal_surprise_sum += surprise
            self.goal_surprise_count += 1
            self.goal_surprise_max = max(self.goal_surprise_max, surprise)
        self._advance_raw_binding_query(byte_id)
        return probability

    def advance_generated_observation(self, byte_id: int) -> float | None:
        """Advance autonomous state with one, rather than repeated, readout."""
        byte_id = int(byte_id)
        self._goal_learning_active = False
        probability = None if self.steps == 0 else self._observed_probability(byte_id)
        motif_baseline = SequenceCortexOpponentLearner._observed_probability(
            self, byte_id
        )
        previous = self.previous_assembly
        if self.live_assemblies[byte_id]:
            self.older_assembly = previous
            self.previous_assembly = byte_id
            self._advance_sequence_history(byte_id)
        else:
            self.older_assembly = None
            self.previous_assembly = None
            self.sequence_history.clear()
        self._advance_motif_state(byte_id, motif_baseline, learn=False)
        if probability is not None and self._goal_scores_byte(byte_id):
            surprise = -log2(max(probability, 1e-300))
            self.goal_surprise_sum += surprise
            self.goal_surprise_count += 1
            self.goal_surprise_max = max(self.goal_surprise_max, surprise)
        self._advance_raw_binding_query(byte_id)
        return probability

    def observe(self, byte_id: int, *, learn: bool = True) -> OpponentStep:
        self._goal_learning_active = learn
        observed = None if self.steps == 0 else self._observed_probability(byte_id)
        step = super().observe(byte_id, learn=learn)
        if observed is not None and self._goal_scores_byte(byte_id):
            surprise = -log2(max(observed, 1e-300))
            self.goal_surprise_sum += surprise
            self.goal_surprise_count += 1
            self.goal_surprise_max = max(self.goal_surprise_max, surprise)
        self._advance_raw_binding_query(byte_id)
        return OpponentStep(observed_probability=observed, prediction=step.prediction)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.goal_organ.reset_episode()
        predictive_goal = getattr(self, "predictive_goal_organ", None)
        if predictive_goal is not None:
            predictive_goal.reset_episode()
        binding_role = getattr(self, "binding_role_organ", None)
        if binding_role is not None:
            binding_role.reset_episode()
        binding_index = getattr(self, "binding_index", None)
        if binding_index is not None:
            binding_index.reset_episode()
        self._binding_pending_credit = None
        self._binding_last_reinstatement = None
        getattr(self, "binding_raw_window", bytearray()).clear()
        getattr(self, "binding_recent_raw_queries", []).clear()
        self.goal_recent_motifs.clear()
        self.goal_route_eligibility.clear()
        getattr(self, "goal_permission_eligibility", {}).clear()
        getattr(self, "goal_permission_active_trials", set()).clear()
        self._goal_learning_active = False
        self._last_goal_route_conditioned = False
        self._goal_route_projection_cache = None
        self.goal_surprise_sum = 0.0
        self.goal_surprise_count = 0
        self.goal_surprise_max = 0.0

    def consolidate(
        self, other: GoalConditionedTemporalBasisMotifOpponentLearner
    ) -> None:
        configuration = (
            "goal_apical_weight",
            "maximum_goal_byte_weight",
            "goal_basin_width",
            "repetition_window",
            "goal_route_eligibility_decay",
            "maximum_goal_route_bonus",
            "goal_selection_utility_decay",
            "minimum_goal_selection_utility",
            "goal_permission_trial_limit",
            "minimum_goal_permission_evidence",
            "goal_third_component",
            "goal_route_context",
            "goal_permission_context",
            "goal_reserve_activation",
            "maximum_goal_route_byte_weight",
            "goal_route_projection_policy",
            "goal_route_timing",
            "goal_route_alignment",
            "goal_flow_polarity",
            "goal_lifecycle",
            "goal_failure_patience",
            "goal_target_alignment_weight",
            "predictive_goal_basins",
            "predictive_goal_dimensions",
            "predictive_goal_birth_similarity",
            "predictive_goal_hysteresis",
            "predictive_goal_weight",
            "binding_index_cells",
            "binding_role_assemblies",
            "binding_cells_per_association",
            "binding_activity_decay",
            "binding_minimum_activity",
            "binding_role_birth_similarity",
            "binding_surprise_bits",
            "binding_role_trial_limit",
            "binding_minimum_role_utility",
            "binding_address_context",
            "goal_flow_mode_gate",
            "goal_flow_mode_trials",
            "goal_surprise_scope",
        )
        if any(getattr(self, name) != getattr(other, name) for name in configuration):
            raise ValueError("cannot consolidate goal cortices with different settings")
        incoming_forms = tuple(other.word_forms)
        incoming_goal = {
            index: (
                other.goal_organ.restoration_sum.get(index, 0.0),
                other.goal_organ.restoration_evidence.get(index, 0.0),
                other.goal_organ.component_restoration_sum.get(index),
                other.goal_organ.component_restoration_evidence.get(index),
            )
            for index in range(len(incoming_forms))
        }
        incoming_route_credit = tuple(
            (
                key,
                tuple(values),
                tuple(other.goal_route_restoration_evidence[key]),
            )
            for key, values in other.goal_route_restoration_sum.items()
            if key in other.goal_route_restoration_evidence
        )
        incoming_selection_utility = tuple(
            (key, utility, other.goal_selection_evidence.get(key, 0))
            for key, utility in other.goal_selection_utility.items()
        )
        incoming_permission_credit = tuple(
            (
                permission,
                restoration,
                other.goal_permission_restoration_evidence.get(permission, 0.0),
                other.goal_permission_trials.get(permission, 0),
            )
            for permission, restoration in (
                other.goal_permission_restoration_sum.items()
            )
        )
        incoming_motif_age_exposure = other.motif_age_exposure.copy()
        incoming_motif_age_endings = other.motif_age_endings.copy()
        incoming_flow_mode_utility = getattr(
            other, "goal_flow_mode_utility", np.zeros(3, dtype=np.float64)
        ).copy()
        incoming_flow_mode_evidence = getattr(
            other, "goal_flow_mode_evidence", np.zeros(3, dtype=np.uint64)
        ).copy()
        incoming_flow_mode_candidates = getattr(
            other, "goal_flow_mode_candidates", np.zeros(3, dtype=np.uint64)
        ).copy()
        incoming_flow_mode_applications = getattr(
            other, "goal_flow_mode_applications", np.zeros(3, dtype=np.uint64)
        ).copy()
        incoming_predictive_goal = getattr(other, "predictive_goal_organ", None)
        incoming_binding_role = getattr(other, "binding_role_organ", None)
        incoming_binding_role_utility_sum = getattr(
            other,
            "binding_role_utility_sum",
            np.zeros(self.binding_role_assemblies, dtype=np.float64),
        ).copy()
        incoming_binding_role_utility_evidence = getattr(
            other,
            "binding_role_utility_evidence",
            np.zeros(self.binding_role_assemblies, dtype=np.uint64),
        ).copy()
        incoming_binding_role_trials = getattr(
            other,
            "binding_role_trials",
            np.zeros(self.binding_role_assemblies, dtype=np.uint64),
        ).copy()
        super().consolidate(other)
        self.motif_age_exposure += incoming_motif_age_exposure
        self.motif_age_endings += incoming_motif_age_endings
        if not hasattr(self, "goal_flow_mode_utility"):
            self.goal_flow_mode_utility = np.zeros(3, dtype=np.float64)
            self.goal_flow_mode_evidence = np.zeros(3, dtype=np.uint64)
            self.goal_flow_mode_candidates = np.zeros(3, dtype=np.uint64)
            self.goal_flow_mode_applications = np.zeros(3, dtype=np.uint64)
        self.goal_flow_mode_utility += incoming_flow_mode_utility
        self.goal_flow_mode_evidence += incoming_flow_mode_evidence
        self.goal_flow_mode_candidates += incoming_flow_mode_candidates
        self.goal_flow_mode_applications += incoming_flow_mode_applications
        if (
            self.predictive_goal_organ is not None
            and incoming_predictive_goal is not None
        ):
            self.predictive_goal_organ.consolidate(incoming_predictive_goal)
        if self.binding_role_organ is not None and incoming_binding_role is not None:
            role_mapping = self.binding_role_organ.consolidate(incoming_binding_role)
            for incoming_role, local_role in role_mapping.items():
                self.binding_role_utility_sum[local_role] += (
                    incoming_binding_role_utility_sum[incoming_role]
                )
                self.binding_role_utility_evidence[local_role] += (
                    incoming_binding_role_utility_evidence[incoming_role]
                )
                self.binding_role_trials[local_role] += incoming_binding_role_trials[
                    incoming_role
                ]
        incoming_to_local = [self.word_ids.get(form) for form in incoming_forms]
        for incoming_id, local_id in enumerate(incoming_to_local):
            if local_id is None:
                continue
            scalar, evidence, component_sum, component_evidence = incoming_goal[
                incoming_id
            ]
            if not self.goal_organ._admit_value(local_id, abs(scalar)):
                continue
            self.goal_organ.restoration_sum[local_id] += scalar
            self.goal_organ.restoration_evidence[local_id] += evidence
            if component_sum is not None and component_evidence is not None:
                local_sum = self.goal_organ.component_restoration_sum.setdefault(
                    local_id, component_sum.copy() * 0.0
                )
                local_evidence = (
                    self.goal_organ.component_restoration_evidence.setdefault(
                        local_id, component_evidence.copy() * 0.0
                    )
                )
                local_sum += component_sum
                local_evidence += component_evidence
        for (receptor, source), sums, evidence in incoming_route_credit:
            if not 0 <= source < len(incoming_to_local):
                continue
            local_source = incoming_to_local[source]
            if local_source is None:
                continue
            key = receptor, local_source
            if key not in self.route_counts:
                continue
            local_sums = self.goal_route_restoration_sum.setdefault(key, [0.0] * 3)
            local_evidence = self.goal_route_restoration_evidence.setdefault(
                key, [0.0] * 3
            )
            for mode in range(3):
                local_sums[mode] += sums[mode]
                local_evidence[mode] += evidence[mode]
        self.goal_organ.local_updates += other.goal_organ.local_updates
        self.goal_organ.goal_births += other.goal_organ.goal_births
        self.goal_organ.goal_dissolutions += other.goal_organ.goal_dissolutions
        self.goal_organ.mode_births += other.goal_organ.mode_births
        self.goal_organ.mode_dissolutions += other.goal_organ.mode_dissolutions
        for counter in (
            "goal_route_context_calls",
            "goal_route_proposals",
            "goal_route_denials",
            "goal_route_switches",
            "goal_route_flow_uses",
            "binding_role_observations",
            "binding_formation_events",
            "binding_recall_attempts",
            "binding_reinstatements",
        ):
            setattr(
                self,
                counter,
                getattr(self, counter, 0) + getattr(other, counter, 0),
            )
        for permission, utility, incoming_evidence in incoming_selection_utility:
            local_evidence = self.goal_selection_evidence.get(permission, 0)
            total = local_evidence + incoming_evidence
            if total:
                self.goal_selection_utility[permission] = (
                    self.goal_selection_utility.get(permission, 0.0) * local_evidence
                    + utility * incoming_evidence
                ) / total
                self.goal_selection_evidence[permission] = total
        for permission, restoration, evidence, trials in incoming_permission_credit:
            self.goal_permission_restoration_sum[permission] = (
                self.goal_permission_restoration_sum.get(permission, 0.0) + restoration
            )
            self.goal_permission_restoration_evidence[permission] = (
                self.goal_permission_restoration_evidence.get(permission, 0.0)
                + evidence
            )
            self.goal_permission_trials[permission] = (
                self.goal_permission_trials.get(permission, 0) + trials
            )
        self.reset_episode()
