"""One-shot conversion: a Python ``SequenceCortexOpponentLearner`` -> the
native binary checkpoint format (``cpp/include/alm/serialize.hpp``).

Existing checkpoints on disk (the ``alm_phase122_*_full_64mb_4c`` bodies)
were pickled by the pure-Python cortex before the native backend existed.
This is the bridge that lets a continued run pick that anatomy up and carry
it forward natively: encode once here, then everything downstream (columns,
``consolidate_delta``, further checkpoints) stays in the native format.

The encoder does not touch probabilities at all -- it walks count tables and
repacks them, so there is nothing to get subtly wrong about IEEE-754 bits.
What it must get exactly right is the *shape* of that data (dict key
packing, array layout) matching what ``alm/serialize.hpp`` expects to read.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from manifoldflow.dendritic_basin.native_cortex import NativeGrowableCortex

if TYPE_CHECKING:
    from manifoldflow.dendritic_basin.opponent_byte_stream import (
        SequenceCortexOpponentLearner,
    )

_MAGIC = 0x414C4D31
_VERSION = 1

_HEADER = struct.Struct("<II")
# prior(d) confirmations(i) context_confirmations(i) surprise_bits(d)
# basal_strength(d) max_context_bytes(i) sequence_confirmations(i)
# parent_strength(d) max_sequence_nursery_paths(q) competition_exponent(d)
# adaptive_competition(B) trust_gamma(d) trust_learning_rate(d)
# attribution_gate_threshold(d) connectivity_pruning(B) burst_surprise_signal(B)
# burst_ema_rate(d) long_range_bank(B) long_range_context_bytes(i)
# long_range_strength(d) long_range_gate_on_burst(B) long_range_gate_threshold(d)
# long_range_max_entries(q) long_range_bands(B) long_range_band_count(i)
# long_range_band_depths[4](iiii) long_range_rolling_hash(B)
# long_range_band_confirmations(i) long_range_importance_bits(d)
# long_range_importance_bonus(i) long_range_band_max_entries(q)
# long_range_use_combined_salience(B) long_range_salience_span_bytes(i)
# long_range_salience_degree_bank(i) long_range_use_affective_intensity(B)
# long_range_ai_threshold(d) long_range_ai_bypass_burst(d)
# long_range_replay_queue_capacity(i) long_range_consolidated_max_entries(q)
# long_range_permanent_max_entries(q) long_range_permanent_recurrence_threshold(q)
# -- order matches CortexIO::write_config exactly (every bool field is one
# raw byte, no struct padding). The pure-Python learner never had deep
# bands, so this import path always writes their struct defaults (the
# native side treats an imported checkpoint as if these fields were never
# touched, same as a fresh GrowableCortex would).
_CONFIG = struct.Struct("<diiddiidqdBdddBBdBidBdq" + "BiiiiiBidiqBiiBddiqqq")
# previous(i) older(i) steps(q) unigram_sum(q) context_births(q)
# sequence_births(q) sequence_evictions(q) assembly_births(q)
# history_length(i) packed_history(Q) burst_fraction(d)
# burst_last_birth_total(q) -- matches CortexIO::save's scalar write order
# exactly.
_SCALARS = struct.Struct("<iiqqqqqqiQdq")
_VECTOR_HEADER = struct.Struct("<Q")
_ROW_HEADER = struct.Struct("<H")
_ROW_ENTRY = struct.Struct("<Bq")
_LEVEL_KEY = struct.Struct("<Q")


def _encode_row(row: dict[int, int]) -> bytes:
    parts = [_ROW_HEADER.pack(len(row))]
    for key, count in row.items():
        parts.append(_ROW_ENTRY.pack(key, count))
    return b"".join(parts)


def _encode_level(level: dict[int, dict[int, int]]) -> bytes:
    parts = [_VECTOR_HEADER.pack(len(level))]
    for key, row in level.items():
        parts.append(_LEVEL_KEY.pack(key))
        parts.append(_encode_row(row))
    return b"".join(parts)


def _encode_vector_i64(values) -> bytes:
    values = list(values)
    return _VECTOR_HEADER.pack(len(values)) + struct.pack(f"<{len(values)}q", *values)


def _encode_vector_u8(values) -> bytes:
    values = list(values)
    return _VECTOR_HEADER.pack(len(values)) + struct.pack(
        f"<{len(values)}B", *(int(bool(v)) for v in values)
    )


def _encode_vector_u16(values) -> bytes:
    values = list(values)
    return _VECTOR_HEADER.pack(len(values)) + struct.pack(f"<{len(values)}H", *values)


def _packed_history(sequence_history: list[int]) -> int:
    packed = 0
    for offset, byte_id in enumerate(reversed(sequence_history)):
        packed |= int(byte_id) << (8 * offset)
    return packed


def encode_native_checkpoint(learner: SequenceCortexOpponentLearner) -> bytes:
    """Repack a live Python cortex's anatomy as a native checkpoint blob."""
    if not learner.packed_paths:
        raise ValueError(
            "native import requires packed_paths=True "
            "(the default, and the only mode the harness actually uses)"
        )

    parts: list[bytes] = [
        _HEADER.pack(_MAGIC, _VERSION),
        _CONFIG.pack(
            float(learner.prior),
            int(learner.confirmations),
            int(learner.context_confirmations),
            float(learner.surprise_bits),
            float(learner.basal_strength),
            int(learner.max_context_bytes),
            int(learner.sequence_confirmations),
            float(learner.parent_strength),
            int(learner.max_sequence_nursery_paths),
            float(getattr(learner, "competition_exponent", 1.0)),
            1 if getattr(learner, "adaptive_competition", False) else 0,
            float(getattr(learner, "trust_gamma", 0.5)),
            float(
                getattr(getattr(learner, "context_value", None), "learning_rate", 0.2)
            ),
            float(getattr(learner, "attribution_gate_threshold", 0.5)),
            1 if getattr(learner, "connectivity_pruning", False) else 0,
            1 if getattr(learner, "burst_surprise_signal", False) else 0,
            float(getattr(learner, "burst_ema_rate", 0.001)),
            1 if getattr(learner, "long_range_bank", False) else 0,
            int(getattr(learner, "long_range_context_bytes", 24)),
            float(getattr(learner, "long_range_strength", 8.0)),
            1 if getattr(learner, "long_range_gate_on_burst", False) else 0,
            float(getattr(learner, "long_range_gate_threshold", 0.1)),
            int(getattr(learner, "long_range_max_entries", 200_000)),
            0,  # long_range_bands
            4,  # long_range_band_count
            12, 16, 24, 32,  # long_range_band_depths
            0,  # long_range_rolling_hash
            3,  # long_range_band_confirmations
            4.0,  # long_range_importance_bits
            3,  # long_range_importance_bonus
            200_000,  # long_range_band_max_entries
            0,  # long_range_use_combined_salience
            6,  # long_range_salience_span_bytes
            0,  # long_range_salience_degree_bank
            0,  # long_range_use_affective_intensity
            2.0,  # long_range_ai_threshold
            0.5,  # long_range_ai_bypass_burst
            64,  # long_range_replay_queue_capacity
            8_000,  # long_range_consolidated_max_entries
            800,  # long_range_permanent_max_entries
            100,  # long_range_permanent_recurrence_threshold
        ),
        _SCALARS.pack(
            -1 if learner.previous_assembly is None else int(learner.previous_assembly),
            -1 if learner.older_assembly is None else int(learner.older_assembly),
            int(learner.steps),
            int(learner.unigram_counts.sum()),
            int(learner.context_births),
            int(learner.sequence_births),
            int(learner.sequence_nursery_evictions),
            int(learner.assembly_births),
            len(learner.sequence_history),
            _packed_history(learner.sequence_history),
            float(getattr(learner, "burst_fraction", 0.0)),
            int(getattr(learner, "_burst_last_birth_total", 0)),
        ),
        _encode_vector_i64(learner.unigram_counts),
        _encode_vector_i64(learner.context_totals),
        _encode_vector_u8(learner.live_assemblies),
        _encode_vector_u16(learner.assembly_confirmations),
        _encode_vector_u16(learner.pair_confirmations.reshape(-1)),
        _encode_vector_u8(learner.pair_live.reshape(-1)),
        _encode_vector_u16(learner.quartet_confirmations.reshape(-1)),
        _encode_vector_u8(learner.quartet_live.reshape(-1)),
    ]
    for index in range(256):
        parts.append(_encode_row(learner.transition_counts[index]))

    packed_residual = {
        (older << 8) | previous: row
        for (older, previous), row in learner.residual_counts.items()
    }
    packed_context_nursery = {
        (older << 8) | previous: row
        for (older, previous), row in learner.context_nursery.items()
    }
    parts.append(_encode_level(packed_residual))
    parts.append(_encode_level(packed_context_nursery))

    bank_count = max(0, learner.max_context_bytes - 2)
    parts.append(struct.pack("<i", bank_count))
    for bank in range(bank_count):
        parts.append(_encode_level(learner.sequence_counts[bank]))
    for bank in range(bank_count):
        nursery = learner.sequence_nursery[bank]
        parts.append(_encode_level(nursery))
        order = list(nursery.keys())
        parts.append(_VECTOR_HEADER.pack(len(order)))
        parts.append(struct.pack(f"<{len(order)}Q", *order) if order else b"")

    parts.append(_encode_trust(learner))
    parts.append(_encode_long_range(learner))
    # Deep bands (long_range_bands) never existed in the pure-Python
    # learner, so this import path always writes band_count=0, matching
    # CortexIO::write_deep_bands' own early return when the feature is off.
    parts.append(struct.pack("<i", 0))
    # No live begin_lineage_tracking() state to carry over -- a freshly
    # imported cortex is not mid-wake-round, matching CortexIO's own
    # tracking_=false default. One zero byte, per write_lineage_tracking.
    parts.append(struct.pack("<B", 0))
    return b"".join(parts)


def _encode_long_range(learner: SequenceCortexOpponentLearner) -> bytes:
    """Matches CortexIO::write_long_range exactly. long_range_counts is
    keyed on real byte tuples (not a packed uint64 like the ordinary
    banks), and the FIFO eviction order native tracks explicitly
    (long_range_order_, a deque) is just Python dict insertion order here
    -- list(long_range_counts) reproduces it without a separate structure.
    """
    counts: dict[tuple[int, ...], dict[int, int]] = getattr(
        learner, "long_range_counts", {}
    )
    history: list[int] = list(getattr(learner, "_long_range_history", []))

    parts = [_VECTOR_HEADER.pack(len(counts))]
    for key, row in counts.items():
        key_bytes = bytes(key)
        parts.append(_VECTOR_HEADER.pack(len(key_bytes)))
        parts.append(key_bytes)
        parts.append(_encode_row(row))
    parts.append(_VECTOR_HEADER.pack(len(history)))
    parts.append(bytes(history))
    order = list(counts.keys())
    parts.append(_VECTOR_HEADER.pack(len(order)))
    for key in order:
        key_bytes = bytes(key)
        parts.append(_VECTOR_HEADER.pack(len(key_bytes)))
        parts.append(key_bytes)
    return b"".join(parts)


def _pack_context_key(key: tuple[int, int]) -> int:
    older, previous = key
    return (int(older) << 8) | int(previous)


def _encode_trust(learner: SequenceCortexOpponentLearner) -> bytes:
    """Matches CortexIO::write_trust exactly (see cpp/include/alm/serialize.hpp).

    Only present when ``adaptive_competition`` was ever turned on for this
    object -- ``getattr`` defaults cover the pre-Phase-153 checkpoints that
    predate this state entirely.
    """
    parts: list[bytes] = []

    context_trust: dict = getattr(learner, "context_trust", {})
    parts.append(_VECTOR_HEADER.pack(len(context_trust)))
    for key, value in context_trust.items():
        parts.append(_LEVEL_KEY.pack(_pack_context_key(key)))
        parts.append(struct.pack("<d", float(value)))

    context_value = getattr(learner, "context_value", None)
    values = dict(getattr(context_value, "values", {}))
    parts.append(_VECTOR_HEADER.pack(len(values)))
    for key, value in values.items():
        parts.append(_LEVEL_KEY.pack(_pack_context_key(key)))
        parts.append(struct.pack("<d", float(value)))

    attribution = getattr(learner, "context_attribution", None)
    self_scale = getattr(attribution, "self_scale", None)
    other_scale = getattr(attribution, "other_scale", None)
    parts.append(
        struct.pack(
            "<dqdqqqq",
            float(getattr(self_scale, "sum_squares", 0.0)),
            int(getattr(self_scale, "count", 0)),
            float(getattr(other_scale, "sum_squares", 0.0)),
            int(getattr(other_scale, "count", 0)),
            int(getattr(attribution, "observations", 0)),
            int(getattr(attribution, "confirmed_self", 0)),
            int(getattr(attribution, "confirmed_other", 0)),
        )
    )

    pending = getattr(learner, "_pending_context_credit", None)
    if pending is None:
        parts.append(struct.pack("<B", 0))
    else:
        key, utility = pending
        parts.append(struct.pack("<B", 1))
        parts.append(_LEVEL_KEY.pack(_pack_context_key(key)))
        parts.append(struct.pack("<d", float(utility)))

    return b"".join(parts)


def native_cortex_from_python(
    learner: SequenceCortexOpponentLearner,
) -> NativeGrowableCortex:
    """Convert a live Python cortex into an equivalent native one."""
    return NativeGrowableCortex.from_bytes(encode_native_checkpoint(learner))
