"""ctypes binding for the native ALM cortex (``cpp/``).

Two bodies are exposed, matching the wake/sleep lifecycle:

``NativeGrowableCortex``
    The developing cortex. Grows context and sequence anatomy from the
    stream exactly as ``SequenceCortexOpponentLearner`` does, including
    nursery promotion, FIFO nursery eviction, and assembly development.

``NativePackedCortex``
    The frozen mature phenotype. No nursery, no spare capacity, every level
    stored as one contiguous block. Scores only -- it cannot develop.

Both reproduce the Python learner's probabilities bit-for-bit; the oracle
test in ``tests/test_native_cortex_matches_python.py`` is what actually
holds that claim up.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_double, c_int32, c_int64, c_size_t, c_uint8
from pathlib import Path

import numpy as np

_STAT_STEPS = 0
_STAT_LIVE_SEQUENCES = 1
_STAT_LIVE_CONTEXTS = 2
_STAT_SEQUENCE_NURSERY = 3
_STAT_SEQUENCE_BIRTHS = 4
_STAT_CONTEXT_BIRTHS = 5
_STAT_SEQUENCE_EVICTIONS = 6
_STAT_ASSEMBLY_BIRTHS = 7
_STAT_MEMORY_BYTES = 8
_STAT_LONG_RANGE_ENTRIES = 9


class _AlmSleepResult(ctypes.Structure):
    _fields_ = [
        ("body_bytes_before_sleep", c_int64),
        ("body_bytes_after_nursery_shed", c_int64),
        ("body_bytes_after_sleep", c_int64),
        ("shed_nursery_paths", c_int64),
        ("pruned_mature_paths", c_int64),
        ("competition_passes", c_int64),
    ]


class _AlmConfig(ctypes.Structure):
    _fields_ = [
        ("prior", c_double),
        ("surprise_bits", c_double),
        ("basal_strength", c_double),
        ("parent_strength", c_double),
        ("confirmations", c_int32),
        ("context_confirmations", c_int32),
        ("sequence_confirmations", c_int32),
        ("max_context_bytes", c_int32),
        ("max_sequence_nursery_paths", c_int64),
        ("competition_exponent", c_double),
        ("adaptive_competition", c_int32),
        ("trust_gamma", c_double),
        ("trust_learning_rate", c_double),
        ("attribution_gate_threshold", c_double),
        ("trust_delta_clip", c_double),
        ("trust_floor", c_double),
        ("trust_ceiling", c_double),
        ("connectivity_pruning", c_int32),
        ("burst_surprise_signal", c_int32),
        ("burst_ema_rate", c_double),
        ("long_range_bank", c_int32),
        ("long_range_context_bytes", c_int32),
        ("long_range_strength", c_double),
        ("long_range_gate_on_burst", c_int32),
        ("long_range_gate_threshold", c_double),
        ("long_range_max_entries", c_int64),
        ("long_range_bands", c_int32),
        ("long_range_band_depths", c_int32 * 4),
        ("long_range_band_count", c_int32),
        ("long_range_rolling_hash", c_int32),
        ("long_range_band_confirmations", c_int32),
        ("long_range_importance_bits", c_double),
        ("long_range_importance_bonus", c_int32),
        ("long_range_band_max_entries", c_int64),
        ("long_range_use_combined_salience", c_int32),
        ("long_range_salience_span_bytes", c_int32),
        ("long_range_salience_degree_bank", c_int32),
        ("long_range_use_affective_intensity", c_int32),
        ("long_range_ai_threshold", c_double),
        ("long_range_ai_bypass_burst", c_double),
        ("long_range_replay_queue_capacity", c_int32),
        ("long_range_consolidated_max_entries", c_int64),
        ("long_range_permanent_max_entries", c_int64),
        ("long_range_permanent_recurrence_threshold", c_int64),
        ("long_range_allow_permanent_tier", c_int32),
        ("episodic_enabled", c_int32),
        ("episodic_k", c_int32),
        ("episodic_s", c_int32),
        ("episodic_bucket_bits", c_int32),
        ("episodic_chain_slots", c_int32),
        ("episodic_tier3_refresh_threshold", c_int32),
        ("episodic_strength", c_double),
    ]


def _library_candidates() -> list[Path]:
    # Override for validating a not-yet-promoted build (e.g. cpp/build_experiment)
    # without touching cpp/build's DLL, which a separate live run may have
    # locked. Unset in the normal case, so default behavior is unchanged.
    override = os.environ.get("ALM_NATIVE_DLL")
    if override:
        return [Path(override)]
    root = Path(__file__).resolve().parents[3] / "cpp" / "build"
    names = (
        ["alm_native.dll"]
        if sys.platform == "win32"
        else ["libalm_native.so", "libalm_native.dylib"]
    )
    candidates: list[Path] = []
    for name in names:
        candidates.append(root / "bin" / "Release" / name)
        candidates.append(root / "bin" / name)
        candidates.append(root / "Release" / name)
        candidates.append(root / name)
    return candidates


_LIBRARY = None


def _load() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    for candidate in _library_candidates():
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            _bind(library)
            _LIBRARY = library
            return library
    raise FileNotFoundError(
        "alm_native shared library not found; build it with:\n"
        "  cmake -S cpp -B cpp/build -G \"Visual Studio 17 2022\" -A x64\n"
        "  cmake --build cpp/build --config Release"
    )


class _AlmError(RuntimeError):
    pass


def _bind(library: ctypes.CDLL) -> None:
    library.alm_cortex_create.argtypes = [POINTER(_AlmConfig)]
    library.alm_cortex_create.restype = ctypes.c_void_p
    library.alm_cortex_destroy.argtypes = [ctypes.c_void_p]
    library.alm_cortex_destroy.restype = None
    library.alm_cortex_observe_chunk.argtypes = [
        ctypes.c_void_p,
        POINTER(c_uint8),
        c_size_t,
        c_int32,
        POINTER(c_double),
    ]
    library.alm_cortex_observe_chunk.restype = None
    library.alm_cortex_predict_distribution.argtypes = [
        ctypes.c_void_p,
        POINTER(c_double),
    ]
    library.alm_cortex_predict_distribution.restype = None
    library.alm_cortex_reset_episode.argtypes = [ctypes.c_void_p]
    library.alm_cortex_reset_episode.restype = None
    library.alm_cortex_begin_lineage_tracking.argtypes = [ctypes.c_void_p]
    library.alm_cortex_begin_lineage_tracking.restype = None
    library.alm_cortex_shed_sequence_nursery.argtypes = [ctypes.c_void_p]
    library.alm_cortex_shed_sequence_nursery.restype = c_int64
    library.alm_cortex_stat.argtypes = [ctypes.c_void_p, c_int32]
    library.alm_cortex_stat.restype = c_int64
    library.alm_cortex_burst_fraction.argtypes = [ctypes.c_void_p]
    library.alm_cortex_burst_fraction.restype = c_double
    library.alm_cortex_context_degree.argtypes = [ctypes.c_void_p, c_int32, ctypes.c_uint64]
    library.alm_cortex_context_degree.restype = c_int64
    library.alm_cortex_current_context_degree.argtypes = [ctypes.c_void_p, c_int32]
    library.alm_cortex_current_context_degree.restype = c_int64
    library.alm_cortex_current_context_frequency.argtypes = [ctypes.c_void_p, c_int32]
    library.alm_cortex_current_context_frequency.restype = c_double
    library.alm_cortex_replay_reinforce.argtypes = [ctypes.c_void_p]
    library.alm_cortex_replay_reinforce.restype = None
    library.alm_cortex_replay_queue_size.argtypes = [ctypes.c_void_p]
    library.alm_cortex_replay_queue_size.restype = c_int64
    library.alm_cortex_max_count_value.argtypes = [ctypes.c_void_p]
    library.alm_cortex_max_count_value.restype = c_int64
    library.alm_cortex_pack.argtypes = [ctypes.c_void_p]
    library.alm_cortex_pack.restype = ctypes.c_void_p
    library.alm_packed_destroy.argtypes = [ctypes.c_void_p]
    library.alm_packed_destroy.restype = None
    library.alm_packed_score_chunk.argtypes = [
        ctypes.c_void_p,
        POINTER(c_uint8),
        c_size_t,
        POINTER(c_double),
    ]
    library.alm_packed_score_chunk.restype = None
    library.alm_packed_reset_episode.argtypes = [ctypes.c_void_p]
    library.alm_packed_reset_episode.restype = None
    library.alm_packed_stat.argtypes = [ctypes.c_void_p, c_int32]
    library.alm_packed_stat.restype = c_int64
    library.alm_cortex_save.argtypes = [ctypes.c_void_p, POINTER(c_size_t)]
    library.alm_cortex_save.restype = POINTER(c_uint8)
    library.alm_free_buffer.argtypes = [POINTER(c_uint8)]
    library.alm_free_buffer.restype = None
    library.alm_cortex_load.argtypes = [POINTER(c_uint8), c_size_t, POINTER(c_int32)]
    library.alm_cortex_load.restype = ctypes.c_void_p
    library.alm_cortex_set_debug_ignore_tier_at_or_above.argtypes = [
        ctypes.c_void_p,
        c_int32,
    ]
    library.alm_cortex_set_debug_ignore_tier_at_or_above.restype = None
    library.alm_cortex_set_debug_freeze_inference_adaptation.argtypes = [
        ctypes.c_void_p,
        c_int32,
    ]
    library.alm_cortex_set_debug_freeze_inference_adaptation.restype = None
    library.alm_cortex_deep_permanent_member_count.argtypes = [ctypes.c_void_p]
    library.alm_cortex_deep_permanent_member_count.restype = c_int64
    library.alm_cortex_enable_episodic_layer.argtypes = [ctypes.c_void_p]
    library.alm_cortex_enable_episodic_layer.restype = None
    library.alm_cortex_configure_episodic_layer.argtypes = [
        ctypes.c_void_p, c_int32, c_int32, c_int32,
    ]
    library.alm_cortex_configure_episodic_layer.restype = c_int32
    library.alm_cortex_set_trust_dynamics.argtypes = [
        ctypes.c_void_p, c_double, c_double, c_double,
    ]
    library.alm_cortex_set_trust_dynamics.restype = None
    library.alm_cortex_set_long_range_allow_permanent_tier.argtypes = [
        ctypes.c_void_p, c_int32,
    ]
    library.alm_cortex_set_long_range_allow_permanent_tier.restype = None
    library.alm_cortex_set_episodic_strength.argtypes = [ctypes.c_void_p, c_double]
    library.alm_cortex_set_episodic_strength.restype = c_int32
    library.alm_cortex_set_debug_deep_band_anchor_gate.argtypes = [
        ctypes.c_void_p,
        c_int32,
        c_int32,
        c_int32,
    ]
    library.alm_cortex_set_debug_deep_band_anchor_gate.restype = c_int32
    library.alm_cortex_trust_summary.argtypes = [
        ctypes.c_void_p,
        POINTER(c_int64),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int64),
        POINTER(c_int64),
        POINTER(c_int64),
    ]
    library.alm_cortex_trust_summary.restype = c_int32
    library.alm_cortex_consolidate_delta.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.alm_cortex_consolidate_delta.restype = c_int32
    library.alm_cortex_consolidate_to_byte_budget.argtypes = [
        ctypes.c_void_p,
        c_int64,
        POINTER(_AlmSleepResult),
    ]
    library.alm_cortex_consolidate_to_byte_budget.restype = c_int32
    library.alm_cortex_estimated_bytes.argtypes = [ctypes.c_void_p]
    library.alm_cortex_estimated_bytes.restype = c_int64
    library.alm_version.argtypes = []
    library.alm_version.restype = c_char_p


def native_available() -> bool:
    try:
        _load()
    except (FileNotFoundError, OSError):
        return False
    return True


def native_version() -> str:
    return _load().alm_version().decode("ascii")


def _as_stream(byte_ids: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(byte_ids, dtype=np.uint8).reshape(-1)


class NativePackedCortex:
    """Frozen mature phenotype: scores a stream, never develops."""

    def __init__(self, handle: int) -> None:
        self._library = _load()
        self._handle = handle

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.alm_packed_destroy(ctypes.c_void_p(handle))
            self._handle = None

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = False
    ) -> np.ndarray:
        if learn:
            raise ValueError("a packed cortex cannot learn; pack after development")
        values = _as_stream(byte_ids)
        observed = np.empty(len(values), dtype=np.float64)
        if len(values):
            self._library.alm_packed_score_chunk(
                ctypes.c_void_p(self._handle),
                values.ctypes.data_as(POINTER(c_uint8)),
                c_size_t(len(values)),
                observed.ctypes.data_as(POINTER(c_double)),
            )
        return observed

    def reset_episode(self) -> None:
        self._library.alm_packed_reset_episode(ctypes.c_void_p(self._handle))

    def _stat(self, stat: int) -> int:
        return int(self._library.alm_packed_stat(ctypes.c_void_p(self._handle), stat))

    @property
    def steps(self) -> int:
        return self._stat(_STAT_STEPS)

    @property
    def live_sequence_count(self) -> int:
        return self._stat(_STAT_LIVE_SEQUENCES)

    @property
    def live_context_count(self) -> int:
        return self._stat(_STAT_LIVE_CONTEXTS)

    @property
    def memory_bytes(self) -> int:
        return self._stat(_STAT_MEMORY_BYTES)


class NativeGrowableCortex:
    """Developing cortex; mirrors ``SequenceCortexOpponentLearner``."""

    def __init__(
        self,
        *,
        prior: float = 0.1,
        confirmations: int = 3,
        context_confirmations: int = 3,
        surprise_bits: float = 2.0,
        basal_strength: float = 4.0,
        max_context_bytes: int = 6,
        sequence_confirmations: int = 3,
        parent_strength: float = 16.0,
        max_sequence_nursery_paths: int = 50_000,
        competition_exponent: float = 1.0,
        adaptive_competition: bool = False,
        trust_gamma: float = 0.5,
        trust_learning_rate: float = 0.2,
        attribution_gate_threshold: float = 0.5,
        trust_delta_clip: float = 2.0,
        trust_floor: float = 0.1,
        trust_ceiling: float = 10.0,
        connectivity_pruning: bool = False,
        burst_surprise_signal: bool = False,
        burst_ema_rate: float = 0.001,
        long_range_bank: bool = False,
        long_range_context_bytes: int = 24,
        long_range_strength: float = 8.0,
        long_range_gate_on_burst: bool = False,
        long_range_gate_threshold: float = 0.1,
        long_range_max_entries: int = 200_000,
        long_range_bands: bool = False,
        long_range_band_depths: tuple[int, int, int, int] = (12, 16, 24, 32),
        long_range_band_count: int = 4,
        long_range_rolling_hash: bool = False,
        long_range_band_confirmations: int = 3,
        long_range_importance_bits: float = 4.0,
        long_range_importance_bonus: int = 3,
        long_range_band_max_entries: int = 200_000,
        long_range_use_combined_salience: bool = False,
        long_range_salience_span_bytes: int = 6,
        long_range_salience_degree_bank: int = 0,
        long_range_use_affective_intensity: bool = False,
        long_range_ai_threshold: float = 2.0,
        long_range_ai_bypass_burst: float = 0.5,
        long_range_replay_queue_capacity: int = 64,
        long_range_consolidated_max_entries: int = 8_000,
        long_range_permanent_max_entries: int = 800,
        long_range_permanent_recurrence_threshold: int = 100,
        long_range_allow_permanent_tier: bool = False,
        episodic_enabled: bool = False,
        episodic_k: int = 14,
        episodic_s: int = 6,
        episodic_bucket_bits: int = 20,
        episodic_chain_slots: int = 4,
        episodic_tier3_refresh_threshold: int = 3,
        episodic_strength: float = 4.0,
    ) -> None:
        if max_context_bytes < 2 or max_context_bytes > 8:
            raise ValueError("max_context_bytes must be in [2, 8]")
        if long_range_gate_on_burst and not burst_surprise_signal:
            raise ValueError(
                "long_range_gate_on_burst requires burst_surprise_signal=True "
                "(otherwise burst_fraction never leaves 0.0 and every byte is "
                "gated out, silently disabling the long-range bank entirely)"
            )
        if long_range_band_count < 0 or long_range_band_count > 4:
            raise ValueError("long_range_band_count must be in [0, 4]")
        self._library = _load()
        config = _AlmConfig(
            prior=prior,
            surprise_bits=surprise_bits,
            basal_strength=basal_strength,
            parent_strength=parent_strength,
            confirmations=confirmations,
            context_confirmations=context_confirmations,
            sequence_confirmations=sequence_confirmations,
            max_context_bytes=max_context_bytes,
            max_sequence_nursery_paths=max_sequence_nursery_paths,
            competition_exponent=competition_exponent,
            adaptive_competition=1 if adaptive_competition else 0,
            trust_gamma=trust_gamma,
            trust_learning_rate=trust_learning_rate,
            attribution_gate_threshold=attribution_gate_threshold,
            trust_delta_clip=trust_delta_clip,
            trust_floor=trust_floor,
            trust_ceiling=trust_ceiling,
            connectivity_pruning=1 if connectivity_pruning else 0,
            burst_surprise_signal=1 if burst_surprise_signal else 0,
            burst_ema_rate=burst_ema_rate,
            long_range_bank=1 if long_range_bank else 0,
            long_range_context_bytes=long_range_context_bytes,
            long_range_strength=long_range_strength,
            long_range_gate_on_burst=1 if long_range_gate_on_burst else 0,
            long_range_gate_threshold=long_range_gate_threshold,
            long_range_max_entries=long_range_max_entries,
            long_range_bands=1 if long_range_bands else 0,
            long_range_band_depths=(c_int32 * 4)(*long_range_band_depths),
            long_range_band_count=long_range_band_count,
            long_range_rolling_hash=1 if long_range_rolling_hash else 0,
            long_range_band_confirmations=long_range_band_confirmations,
            long_range_importance_bits=long_range_importance_bits,
            long_range_importance_bonus=long_range_importance_bonus,
            long_range_band_max_entries=long_range_band_max_entries,
            long_range_use_combined_salience=1 if long_range_use_combined_salience else 0,
            long_range_salience_span_bytes=long_range_salience_span_bytes,
            long_range_salience_degree_bank=long_range_salience_degree_bank,
            long_range_use_affective_intensity=1 if long_range_use_affective_intensity else 0,
            long_range_ai_threshold=long_range_ai_threshold,
            long_range_ai_bypass_burst=long_range_ai_bypass_burst,
            long_range_replay_queue_capacity=long_range_replay_queue_capacity,
            long_range_consolidated_max_entries=long_range_consolidated_max_entries,
            long_range_permanent_max_entries=long_range_permanent_max_entries,
            long_range_permanent_recurrence_threshold=long_range_permanent_recurrence_threshold,
            long_range_allow_permanent_tier=1 if long_range_allow_permanent_tier else 0,
            episodic_enabled=1 if episodic_enabled else 0,
            episodic_k=episodic_k,
            episodic_s=episodic_s,
            episodic_bucket_bits=episodic_bucket_bits,
            episodic_chain_slots=episodic_chain_slots,
            episodic_tier3_refresh_threshold=episodic_tier3_refresh_threshold,
            episodic_strength=episodic_strength,
        )
        handle = self._library.alm_cortex_create(ctypes.byref(config))
        if not handle:
            raise MemoryError("could not allocate a native cortex")
        self._handle = handle

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.alm_cortex_destroy(ctypes.c_void_p(handle))
            self._handle = None

    def observe_chunk_probabilities(
        self, byte_ids: np.ndarray, *, learn: bool = True
    ) -> np.ndarray:
        values = _as_stream(byte_ids)
        observed = np.empty(len(values), dtype=np.float64)
        if len(values):
            self._library.alm_cortex_observe_chunk(
                ctypes.c_void_p(self._handle),
                values.ctypes.data_as(POINTER(c_uint8)),
                c_size_t(len(values)),
                c_int32(1 if learn else 0),
                observed.ctypes.data_as(POINTER(c_double)),
            )
        return observed

    def predict_next_distribution(self) -> np.ndarray:
        """Full 256-way prediction at the current context; no side effects.

        Matches ``predict_next_distribution`` on the Python cortex -- used
        by the generation loop to sample a byte before it is known, not to
        score one that already is.
        """
        out = np.empty(256, dtype=np.float64)
        self._library.alm_cortex_predict_distribution(
            ctypes.c_void_p(self._handle), out.ctypes.data_as(POINTER(c_double))
        )
        return out

    def advance_observation(self, byte_id: int, *, learn: bool = False) -> float | None:
        """Advance state by one byte without building the full output vector.

        Matches the Python cortex's method of the same name (used by the
        generation loop after sampling): equivalent to
        ``observe_chunk_probabilities([byte_id], learn=learn)[0]``.
        """
        result = self.observe_chunk_probabilities(
            np.array([byte_id], dtype=np.uint8), learn=learn
        )
        value = float(result[0])
        return None if np.isnan(value) else value

    def reset_episode(self) -> None:
        self._library.alm_cortex_reset_episode(ctypes.c_void_p(self._handle))

    def begin_lineage_tracking(self) -> None:
        self._library.alm_cortex_begin_lineage_tracking(ctypes.c_void_p(self._handle))

    def set_debug_ignore_tier_at_or_above(self, threshold: int) -> None:
        """Ablation-only: see alm_cortex_set_debug_ignore_tier_at_or_above
        in c_api.h. threshold<=0 disables; 4 blocks permanent only; 3
        blocks consolidated+permanent; 2 blocks every deep-band match.
        Not persisted by to_bytes()/from_bytes()."""
        self._library.alm_cortex_set_debug_ignore_tier_at_or_above(
            ctypes.c_void_p(self._handle), c_int32(threshold)
        )

    def set_debug_freeze_inference_adaptation(self, freeze: bool) -> None:
        """Make ``learn=False`` traversal read-only for adaptive state.

        The normal deployment path continues to update its burst EMA and
        context-trust learner while durable anatomy is frozen. Counterfactual
        evaluation reuses one cortex across several candidate branches, where
        those updates would make results depend on branch order. This
        runtime-only debug switch freezes those two updates while retaining
        ordinary episode-history advance. It is not checkpointed.
        """
        self._library.alm_cortex_set_debug_freeze_inference_adaptation(
            ctypes.c_void_p(self._handle), c_int32(1 if freeze else 0)
        )

    def deep_permanent_member_count(self) -> int:
        """Debug-only: total deep-band keys currently classified tier 4
        (permanent), across all bands."""
        return int(
            self._library.alm_cortex_deep_permanent_member_count(
                ctypes.c_void_p(self._handle)
            )
        )

    def trust_summary(self) -> dict:
        """Summary of the learned adaptive-competition trust distribution
        (per-context multiplicative weight, clipped to [0.1, 10.0], 1.0 =
        never updated). Answers whether trust has developed real
        differentiation or stayed near-flat."""
        count = c_int64()
        min_value = c_double()
        max_value = c_double()
        mean_value = c_double()
        count_at_floor = c_int64()
        count_at_ceiling = c_int64()
        count_near_neutral = c_int64()
        self._library.alm_cortex_trust_summary(
            ctypes.c_void_p(self._handle),
            ctypes.byref(count),
            ctypes.byref(min_value),
            ctypes.byref(max_value),
            ctypes.byref(mean_value),
            ctypes.byref(count_at_floor),
            ctypes.byref(count_at_ceiling),
            ctypes.byref(count_near_neutral),
        )
        return {
            "count": count.value,
            "min": min_value.value,
            "max": max_value.value,
            "mean": mean_value.value,
            "count_at_floor": count_at_floor.value,
            "count_at_ceiling": count_at_ceiling.value,
            "count_near_neutral": count_near_neutral.value,
        }

    def enable_episodic_layer(self) -> None:
        """Turns the sparse episodic layer on for an already-constructed
        (e.g. resumed/deserialized) cortex built with episodic_enabled
        False -- lets an in-progress run retrofit episodic memory without
        restarting from scratch. No-op if already enabled."""
        self._library.alm_cortex_enable_episodic_layer(ctypes.c_void_p(self._handle))

    def configure_episodic_layer(self, enabled: bool, k: int, s: int) -> None:
        """Reconfigures the episodic layer's k/s anchor width, ALWAYS
        re-applying it (unlike enable_episodic_layer, a no-op once already
        enabled). episodic_k/s/enabled are not part of the checkpoint
        format -- a training loop resuming from a checkpoint every epoch
        must call this every epoch too, or the width silently reverts to
        the compiled default (k=14/s=6) after the first epoch. Raises on
        an invalid k/s (see GrowableCortex::init_episodic_state)."""
        result = self._library.alm_cortex_configure_episodic_layer(
            ctypes.c_void_p(self._handle), c_int32(1 if enabled else 0),
            c_int32(k), c_int32(s),
        )
        if result != 0:
            raise RuntimeError(f"configure_episodic_layer failed for k={k}, s={s}")

    def set_trust_dynamics(self, delta_clip: float, floor: float, ceiling: float) -> None:
        """trust_delta_clip/floor/ceiling are not part of the checkpoint
        format -- same reapply-every-epoch reasoning as
        configure_episodic_layer above."""
        self._library.alm_cortex_set_trust_dynamics(
            ctypes.c_void_p(self._handle),
            c_double(delta_clip), c_double(floor), c_double(ceiling),
        )

    def set_long_range_allow_permanent_tier(self, allow: bool) -> None:
        """long_range_allow_permanent_tier is not part of the checkpoint
        format -- same reapply-every-epoch reasoning as
        configure_episodic_layer above. Matters more than the others:
        without reapplying, a run intended to allow permanent-tier
        promotion silently falls back to the privacy-gated default (False)
        after its first epoch, for the rest of the run."""
        self._library.alm_cortex_set_long_range_allow_permanent_tier(
            ctypes.c_void_p(self._handle), c_int32(1 if allow else 0)
        )

    def set_episodic_strength(self, strength: float) -> None:
        """episodic_strength is not part of the checkpoint format -- same
        reapply-every-epoch reasoning as the setters above, and a genuine
        tuning knob: a strength sweep at w=2 found a sharp threshold (0%
        exact-match generation recall of a planted code at the old default
        4.0, 100% at every value <=1.0, no throughput cost either way,
        since strength is pure blend arithmetic). Raises if strength<=0."""
        result = self._library.alm_cortex_set_episodic_strength(
            ctypes.c_void_p(self._handle), c_double(strength)
        )
        if result != 0:
            raise RuntimeError(f"set_episodic_strength failed for strength={strength}")

    def set_debug_deep_band_anchor_gate(
        self, enabled: bool, k: int = 6, s: int = 3
    ) -> None:
        """Ablation-only: gates deep-band lookup/grow to only fire at
        syncmer-anchor positions (a separate, narrower k/s test than the
        episodic layer's own k=14/s=6 -- swept via k/s here), leaving the
        O(1) rolling-hash history update dense every byte. Raises if any
        active long_range_band_depths entry is below k, or s/k are out of
        range. Not persisted by to_bytes()/from_bytes()."""
        result = self._library.alm_cortex_set_debug_deep_band_anchor_gate(
            ctypes.c_void_p(self._handle), c_int32(1 if enabled else 0),
            c_int32(k), c_int32(s),
        )
        if result != 0:
            raise RuntimeError(
                "set_debug_deep_band_anchor_gate failed -- likely a "
                f"long_range_band_depths entry below k={k}, or invalid k/s"
            )

    @property
    def deep_band_total_promotions(self) -> int:
        """Cumulative nursery->mature promotions ever, across all deep
        bands. Unlike deep_band_mature_count (a live snapshot capped at
        long_range_band_max_entries and reduced by eviction), this never
        decreases -- use it for a genuine stabilization rate once the live
        count has saturated its cap. Not serialized by to_bytes()/
        from_bytes(); resets to 0 on load."""
        return self._stat(13)

    @property
    def deep_band_total_consolidations(self) -> int:
        """Cumulative tier 2->3 (consolidated) promotions ever. Much rarer
        than deep_band_total_promotions (tier 2 is high-volume churn) --
        closer to a genuine stabilization signal. Resets to 0 on load."""
        return self._stat(14)

    @property
    def deep_band_total_permanent_promotions(self) -> int:
        """Cumulative tier 3->4 (permanent) promotions ever -- the rarest
        tier (single/double digits observed over tens of GB of real text).
        The closest existing analogue to "this became worth a durable
        episodic anchor." Resets to 0 on load."""
        return self._stat(15)

    def shed_sequence_nursery(self) -> int:
        return int(
            self._library.alm_cortex_shed_sequence_nursery(ctypes.c_void_p(self._handle))
        )

    def to_bytes(self) -> bytes:
        """Native binary checkpoint (see cpp/include/alm/serialize.hpp).

        Same-machine, same-build only -- this is a fast replacement for
        pickling the Python object graph, not a portable exchange format.
        """
        size = c_size_t()
        buffer = self._library.alm_cortex_save(
            ctypes.c_void_p(self._handle), ctypes.byref(size)
        )
        if not buffer:
            raise _AlmError("failed to serialize native cortex")
        try:
            return ctypes.string_at(buffer, size.value)
        finally:
            self._library.alm_free_buffer(buffer)

    @classmethod
    def from_bytes(cls, data: bytes) -> NativeGrowableCortex:
        library = _load()
        payload = (c_uint8 * len(data)).from_buffer_copy(data)
        error = c_int32()
        handle = library.alm_cortex_load(
            payload, c_size_t(len(data)), ctypes.byref(error)
        )
        if not handle or error.value:
            raise _AlmError("failed to load native cortex checkpoint")
        instance = cls.__new__(cls)
        instance._library = library
        instance._handle = handle
        return instance

    def __reduce__(self) -> tuple:
        return (NativeGrowableCortex.from_bytes, (self.to_bytes(),))

    def consolidate_delta(
        self, descendant: NativeGrowableCortex, ancestor: NativeGrowableCortex
    ) -> None:
        """Merge `descendant`'s anatomy past its shared `ancestor` into self.

        Matches ``SequenceCortexOpponentLearner.consolidate_delta``: several
        wake columns cloned from one ancestor are merged back without
        double-counting what the ancestor already knew.
        """
        result = self._library.alm_cortex_consolidate_delta(
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(descendant._handle),
            ctypes.c_void_p(ancestor._handle),
        )
        if result != 0:
            raise ValueError("cannot consolidate incompatible cortical lineage")

    def consolidate_to_byte_budget(self, max_bytes: int) -> dict[str, int]:
        """Sleep: shed nursery anatomy, prune weak sequence paths to fit.

        Matches ``SequenceCortexOpponentLearner.consolidate_to_byte_budget``,
        except the byte budget is measured against this checkpoint's own
        native serialized size, not a Python pickle byte count -- the two
        are different units, so a native body at a given cap holds more
        learned paths than a Python body at the same cap.
        """
        result = _AlmSleepResult()
        status = self._library.alm_cortex_consolidate_to_byte_budget(
            ctypes.c_void_p(self._handle), c_int64(max_bytes), ctypes.byref(result)
        )
        if status != 0:
            raise ValueError(
                "cannot sleep to byte budget "
                "(budget too small for the fixed cortical anatomy, or no "
                "anatomy remains to prune)"
            )
        return {
            "body_bytes_before_sleep": result.body_bytes_before_sleep,
            "body_bytes_after_nursery_shed": result.body_bytes_after_nursery_shed,
            "body_bytes_after_sleep": result.body_bytes_after_sleep,
            "shed_nursery_paths": result.shed_nursery_paths,
            "pruned_mature_paths": result.pruned_mature_paths,
            "competition_passes": result.competition_passes,
        }

    @property
    def embodied_bytes(self) -> int:
        """Named to match the Python cortex's property of the same name, so
        this object is a drop-in replacement wherever harness code reads
        ``cortex.embodied_bytes`` (checkpoint size reporting, the
        byte-budget sleep step). See :meth:`consolidate_to_byte_budget`'s
        docstring for why the number itself is not directly comparable to
        the Python cortex's pickle-based measure.
        """
        return int(
            self._library.alm_cortex_estimated_bytes(ctypes.c_void_p(self._handle))
        )

    def pack(self) -> NativePackedCortex:
        """Freeze the developed anatomy into the mature inference body."""
        handle = self._library.alm_cortex_pack(ctypes.c_void_p(self._handle))
        if not handle:
            raise MemoryError("could not allocate a packed cortex")
        return NativePackedCortex(handle)

    def _stat(self, stat: int) -> int:
        return int(self._library.alm_cortex_stat(ctypes.c_void_p(self._handle), stat))

    @property
    def steps(self) -> int:
        return self._stat(_STAT_STEPS)

    @property
    def live_sequence_count(self) -> int:
        return self._stat(_STAT_LIVE_SEQUENCES)

    @property
    def live_context_count(self) -> int:
        return self._stat(_STAT_LIVE_CONTEXTS)

    @property
    def long_range_entry_count(self) -> int:
        return self._stat(_STAT_LONG_RANGE_ENTRIES)

    @property
    def burst_fraction(self) -> float:
        return float(self._library.alm_cortex_burst_fraction(ctypes.c_void_p(self._handle)))

    def context_degree(self, bank: int, context: int) -> int:
        """Salience-signal prototype: how many mature bank+1 rows currently
        use `context` (a bank-depth suffix) as their trailing suffix."""
        return int(
            self._library.alm_cortex_context_degree(
                ctypes.c_void_p(self._handle), c_int32(bank), ctypes.c_uint64(context)
            )
        )

    def current_context_degree(self, bank: int) -> int:
        return int(
            self._library.alm_cortex_current_context_degree(
                ctypes.c_void_p(self._handle), c_int32(bank)
            )
        )

    def current_context_frequency(self, bank: int) -> float:
        return float(
            self._library.alm_cortex_current_context_frequency(
                ctypes.c_void_p(self._handle), c_int32(bank)
            )
        )

    def replay_reinforce(self) -> None:
        """Affective Intensity's other half: reprocess every queued replay
        candidate as if observed again. Call once per sleep cycle."""
        self._library.alm_cortex_replay_reinforce(ctypes.c_void_p(self._handle))

    @property
    def replay_queue_size(self) -> int:
        return int(self._library.alm_cortex_replay_queue_size(ctypes.c_void_p(self._handle)))

    @property
    def max_count_value(self) -> int:
        return int(self._library.alm_cortex_max_count_value(ctypes.c_void_p(self._handle)))

    @property
    def sequence_nursery_count(self) -> int:
        return self._stat(_STAT_SEQUENCE_NURSERY)

    @property
    def sequence_births(self) -> int:
        return self._stat(_STAT_SEQUENCE_BIRTHS)

    @property
    def context_births(self) -> int:
        return self._stat(_STAT_CONTEXT_BIRTHS)

    @property
    def sequence_nursery_evictions(self) -> int:
        return self._stat(_STAT_SEQUENCE_EVICTIONS)

    @property
    def assembly_births(self) -> int:
        return self._stat(_STAT_ASSEMBLY_BIRTHS)

    @property
    def memory_bytes(self) -> int:
        return self._stat(_STAT_MEMORY_BYTES)
