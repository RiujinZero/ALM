// Native binary checkpoint format for GrowableCortex.
//
// Same-machine, same-build use only (raw struct layout, host endianness) --
// this is not a portable exchange format, it is a fast replacement for
// pickling a `dict`-of-`dict`-of-`dict` Python object graph. A row's
// (successor, count) pairs are written in arbitrary order and rebuilt with
// ordinary `increment` calls on load, which is safe because neither the
// probabilities nor the dense/sparse storage choice depend on insertion
// order -- only the *set* of counts does.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "alm/cortex.hpp"

namespace alm {

struct CortexIO {
    static constexpr uint32_t kMagic = 0x414C4D31;  // "ALM1"
    static constexpr uint32_t kVersion = 1;

    static std::vector<uint8_t> save(const GrowableCortex& cortex) {
        std::vector<uint8_t> out;
        out.reserve(1 << 20);
        write_u32(out, kMagic);
        write_u32(out, kVersion);
        write_config(out, cortex.config_);

        write_i32(out, cortex.previous_);
        write_i32(out, cortex.older_);
        write_i64(out, cortex.steps_);
        write_i64(out, cortex.unigram_sum_);
        write_i64(out, cortex.context_births_);
        write_i64(out, cortex.sequence_births_);
        write_i64(out, cortex.sequence_evictions_);
        write_i64(out, cortex.assembly_births_);
        write_i32(out, cortex.history_length_);
        write_u64(out, cortex.packed_history_);
        write_f64(out, cortex.burst_fraction_);
        write_i64(out, cortex.burst_last_birth_total_);

        write_vector(out, cortex.unigram_counts_);
        write_vector(out, cortex.context_totals_);
        write_vector(out, cortex.live_assemblies_);
        write_vector(out, cortex.assembly_confirmations_);
        write_vector(out, cortex.pair_confirmations_);
        write_vector(out, cortex.pair_live_);
        write_vector(out, cortex.quartet_confirmations_);
        write_vector(out, cortex.quartet_live_);

        for (int index = 0; index < 256; ++index) {
            write_row(out, cortex.arena_, cortex.transition_rows_[index]);
        }
        write_level(out, cortex.arena_, cortex.residual_map_);
        write_level(out, cortex.arena_, cortex.context_nursery_);

        write_i32(out, cortex.bank_count_);
        for (int32_t bank = 0; bank < cortex.bank_count_; ++bank) {
            if (cortex.is_deep_bank(bank)) {
                write_level(out, cortex.arena_deep_bank_, cortex.sequence_maps_[bank]);
            } else {
                write_level(out, cortex.arena_, cortex.sequence_maps_[bank]);
            }
        }
        for (int32_t bank = 0; bank < cortex.bank_count_; ++bank) {
            const NurseryBank& nursery = cortex.sequence_nursery_[bank];
            if (cortex.is_deep_bank(bank)) {
                write_level(out, cortex.arena_deep_bank_, nursery.map);
            } else {
                write_level(out, cortex.arena_, nursery.map);
            }
            write_u64(out, static_cast<uint64_t>(nursery.order.size()));
            for (uint64_t key : nursery.order) write_u64(out, key);
        }
        write_trust(out, cortex.trust_);
        write_long_range(out, cortex);
        write_deep_bands(out, cortex);
        write_lineage_tracking(out, cortex);
        return out;
    }

    static GrowableCortex load(const uint8_t* data, size_t size) {
        Reader reader{data, size, 0};
        if (read_u32(reader) != kMagic) throw std::runtime_error("bad ALM checkpoint magic");
        if (read_u32(reader) != kVersion)
            throw std::runtime_error("unsupported ALM checkpoint version");
        CortexConfig config = read_config(reader);
        GrowableCortex cortex(config);

        cortex.previous_ = read_i32(reader);
        cortex.older_ = read_i32(reader);
        cortex.steps_ = read_i64(reader);
        cortex.unigram_sum_ = read_i64(reader);
        cortex.context_births_ = read_i64(reader);
        cortex.sequence_births_ = read_i64(reader);
        cortex.sequence_evictions_ = read_i64(reader);
        cortex.assembly_births_ = read_i64(reader);
        cortex.history_length_ = read_i32(reader);
        cortex.packed_history_ = read_u64(reader);
        cortex.burst_fraction_ = read_f64(reader);
        cortex.burst_last_birth_total_ = read_i64(reader);

        read_vector(reader, cortex.unigram_counts_);
        read_vector(reader, cortex.context_totals_);
        read_vector(reader, cortex.live_assemblies_);
        read_vector(reader, cortex.assembly_confirmations_);
        read_vector(reader, cortex.pair_confirmations_);
        read_vector(reader, cortex.pair_live_);
        read_vector(reader, cortex.quartet_confirmations_);
        read_vector(reader, cortex.quartet_live_);

        for (int index = 0; index < 256; ++index) {
            read_row(reader, cortex.arena_, cortex.transition_rows_[index]);
        }
        read_level(reader, cortex.arena_, cortex.residual_map_);
        read_level(reader, cortex.arena_, cortex.context_nursery_);

        const int32_t bank_count = read_i32(reader);
        if (bank_count != cortex.bank_count_)
            throw std::runtime_error("ALM checkpoint bank count does not match config");
        for (int32_t bank = 0; bank < bank_count; ++bank) {
            if (cortex.is_deep_bank(bank)) {
                read_level(reader, cortex.arena_deep_bank_, cortex.sequence_maps_[bank]);
            } else {
                read_level(reader, cortex.arena_, cortex.sequence_maps_[bank]);
            }
        }
        for (int32_t bank = 0; bank < bank_count; ++bank) {
            NurseryBank& nursery = cortex.sequence_nursery_[bank];
            if (cortex.is_deep_bank(bank)) {
                read_level(reader, cortex.arena_deep_bank_, nursery.map);
            } else {
                read_level(reader, cortex.arena_, nursery.map);
            }
            const uint64_t order_size = read_u64(reader);
            for (uint64_t i = 0; i < order_size; ++i) nursery.order.push_back(read_u64(reader));
        }
        read_trust(reader, cortex.trust_);
        read_long_range(reader, cortex);
        read_deep_bands(reader, cortex);
        read_lineage_tracking(reader, cortex);
        return cortex;
    }

private:
    struct Reader {
        const uint8_t* data;
        size_t size;
        size_t offset;
    };

    static void need(const Reader& reader, size_t bytes) {
        if (reader.offset + bytes > reader.size)
            throw std::runtime_error("truncated ALM checkpoint");
    }

    template <typename T>
    static void write_pod(std::vector<uint8_t>& out, T value) {
        const size_t at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &value, sizeof(T));
    }
    template <typename T>
    static T read_pod(Reader& reader) {
        need(reader, sizeof(T));
        T value;
        std::memcpy(&value, reader.data + reader.offset, sizeof(T));
        reader.offset += sizeof(T);
        return value;
    }

    static void write_u32(std::vector<uint8_t>& out, uint32_t v) { write_pod(out, v); }
    static void write_i32(std::vector<uint8_t>& out, int32_t v) { write_pod(out, v); }
    static void write_u64(std::vector<uint8_t>& out, uint64_t v) { write_pod(out, v); }
    static void write_i64(std::vector<uint8_t>& out, int64_t v) { write_pod(out, v); }
    static void write_f64(std::vector<uint8_t>& out, double v) { write_pod(out, v); }
    static uint32_t read_u32(Reader& r) { return read_pod<uint32_t>(r); }
    static int32_t read_i32(Reader& r) { return read_pod<int32_t>(r); }
    static uint64_t read_u64(Reader& r) { return read_pod<uint64_t>(r); }
    static int64_t read_i64(Reader& r) { return read_pod<int64_t>(r); }
    static double read_f64(Reader& r) { return read_pod<double>(r); }

    static void write_config(std::vector<uint8_t>& out, const CortexConfig& c) {
        write_f64(out, c.prior);
        write_i32(out, c.confirmations);
        write_i32(out, c.context_confirmations);
        write_f64(out, c.surprise_bits);
        write_f64(out, c.basal_strength);
        write_i32(out, c.max_context_bytes);
        write_i32(out, c.sequence_confirmations);
        write_f64(out, c.parent_strength);
        write_i64(out, c.max_sequence_nursery_paths);
        write_f64(out, c.competition_exponent);
        write_pod<uint8_t>(out, c.adaptive_competition ? 1 : 0);
        write_f64(out, c.trust_gamma);
        write_f64(out, c.trust_learning_rate);
        write_f64(out, c.attribution_gate_threshold);
        write_pod<uint8_t>(out, c.connectivity_pruning ? 1 : 0);
        write_pod<uint8_t>(out, c.burst_surprise_signal ? 1 : 0);
        write_f64(out, c.burst_ema_rate);
        write_pod<uint8_t>(out, c.long_range_bank ? 1 : 0);
        write_i32(out, c.long_range_context_bytes);
        write_f64(out, c.long_range_strength);
        write_pod<uint8_t>(out, c.long_range_gate_on_burst ? 1 : 0);
        write_f64(out, c.long_range_gate_threshold);
        write_i64(out, c.long_range_max_entries);
        // Deep bands (long_range_bands) and everything built on top of it
        // (combined_salience, Affective Intensity, replay, tiered
        // persistence) were previously never serialized at all -- every
        // one of these flags silently reset to its struct default (off)
        // on the very next save/load round trip, which the wake/sleep
        // harness does every epoch for every column. That made the whole
        // mechanism a no-op past epoch 1 for any run that enabled it.
        write_pod<uint8_t>(out, c.long_range_bands ? 1 : 0);
        write_i32(out, c.long_range_band_count);
        for (int32_t b = 0; b < CortexConfig::kMaxLongRangeBands; ++b) {
            write_i32(out, c.long_range_band_depths[b]);
        }
        write_pod<uint8_t>(out, c.long_range_rolling_hash ? 1 : 0);
        write_i32(out, c.long_range_band_confirmations);
        write_f64(out, c.long_range_importance_bits);
        write_i32(out, c.long_range_importance_bonus);
        write_i64(out, c.long_range_band_max_entries);
        write_pod<uint8_t>(out, c.long_range_use_combined_salience ? 1 : 0);
        write_i32(out, c.long_range_salience_span_bytes);
        write_i32(out, c.long_range_salience_degree_bank);
        write_pod<uint8_t>(out, c.long_range_use_affective_intensity ? 1 : 0);
        write_f64(out, c.long_range_ai_threshold);
        write_f64(out, c.long_range_ai_bypass_burst);
        write_i32(out, c.long_range_replay_queue_capacity);
        write_i64(out, c.long_range_consolidated_max_entries);
        write_i64(out, c.long_range_permanent_max_entries);
        write_i64(out, c.long_range_permanent_recurrence_threshold);
    }
    static CortexConfig read_config(Reader& r) {
        CortexConfig c;
        c.prior = read_f64(r);
        c.confirmations = read_i32(r);
        c.context_confirmations = read_i32(r);
        c.surprise_bits = read_f64(r);
        c.basal_strength = read_f64(r);
        c.max_context_bytes = read_i32(r);
        c.sequence_confirmations = read_i32(r);
        c.parent_strength = read_f64(r);
        c.max_sequence_nursery_paths = read_i64(r);
        c.competition_exponent = read_f64(r);
        c.adaptive_competition = read_pod<uint8_t>(r) != 0;
        c.trust_gamma = read_f64(r);
        c.trust_learning_rate = read_f64(r);
        c.attribution_gate_threshold = read_f64(r);
        // Added alongside connectivity_pruning: checkpoints written before
        // this field existed do not have this trailing byte, and there is
        // no safe way to tell "old checkpoint, field absent" apart from
        // "field present but happens to run out of buffer" from here --
        // the sections after config (nursery, lineage tracking) have their
        // own data immediately following in the same stream, so silently
        // defaulting on a truncated read would misalign every read after
        // this one instead of failing cleanly. A checkpoint saved before
        // this field existed is not loadable by this build; start a fresh
        // run rather than resume across this specific format change.
        c.connectivity_pruning = read_pod<uint8_t>(r) != 0;
        c.burst_surprise_signal = read_pod<uint8_t>(r) != 0;
        c.burst_ema_rate = read_f64(r);
        c.long_range_bank = read_pod<uint8_t>(r) != 0;
        c.long_range_context_bytes = read_i32(r);
        c.long_range_strength = read_f64(r);
        c.long_range_gate_on_burst = read_pod<uint8_t>(r) != 0;
        c.long_range_gate_threshold = read_f64(r);
        c.long_range_max_entries = read_i64(r);
        c.long_range_bands = read_pod<uint8_t>(r) != 0;
        c.long_range_band_count = read_i32(r);
        for (int32_t b = 0; b < CortexConfig::kMaxLongRangeBands; ++b) {
            c.long_range_band_depths[b] = read_i32(r);
        }
        c.long_range_rolling_hash = read_pod<uint8_t>(r) != 0;
        c.long_range_band_confirmations = read_i32(r);
        c.long_range_importance_bits = read_f64(r);
        c.long_range_importance_bonus = read_i32(r);
        c.long_range_band_max_entries = read_i64(r);
        c.long_range_use_combined_salience = read_pod<uint8_t>(r) != 0;
        c.long_range_salience_span_bytes = read_i32(r);
        c.long_range_salience_degree_bank = read_i32(r);
        c.long_range_use_affective_intensity = read_pod<uint8_t>(r) != 0;
        c.long_range_ai_threshold = read_f64(r);
        c.long_range_ai_bypass_burst = read_f64(r);
        c.long_range_replay_queue_capacity = read_i32(r);
        c.long_range_consolidated_max_entries = read_i64(r);
        c.long_range_permanent_max_entries = read_i64(r);
        c.long_range_permanent_recurrence_threshold = read_i64(r);
        return c;
    }

    static void write_trust(std::vector<uint8_t>& out, const TrustState& trust) {
        write_u64(out, static_cast<uint64_t>(trust.context_trust.size()));
        for (const auto& [key, value] : trust.context_trust) {
            write_u64(out, key);
            write_f64(out, value);
        }
        write_u64(out, static_cast<uint64_t>(trust.context_value.values().size()));
        for (const auto& [key, value] : trust.context_value.values()) {
            write_u64(out, key);
            write_f64(out, value);
        }
        const SelfCausalAttribution& attribution = trust.context_attribution;
        write_f64(out, attribution.self_sum_squares());
        write_i64(out, attribution.self_count());
        write_f64(out, attribution.other_sum_squares());
        write_i64(out, attribution.other_count());
        write_i64(out, attribution.observations());
        write_i64(out, attribution.confirmed_self());
        write_i64(out, attribution.confirmed_other());
        write_pod<uint8_t>(out, trust.pending_context_credit.has_value() ? 1 : 0);
        if (trust.pending_context_credit) {
            write_u64(out, trust.pending_context_credit->first);
            write_f64(out, trust.pending_context_credit->second);
        }
    }
    static void read_trust(Reader& reader, TrustState& trust) {
        const uint64_t trust_count = read_u64(reader);
        for (uint64_t i = 0; i < trust_count; ++i) {
            const uint64_t key = read_u64(reader);
            const double value = read_f64(reader);
            trust.context_trust[key] = value;
        }
        const uint64_t value_count = read_u64(reader);
        for (uint64_t i = 0; i < value_count; ++i) {
            const uint64_t key = read_u64(reader);
            const double value = read_f64(reader);
            trust.context_value.values()[key] = value;
        }
        const double self_sum_squares = read_f64(reader);
        const int64_t self_count = read_i64(reader);
        const double other_sum_squares = read_f64(reader);
        const int64_t other_count = read_i64(reader);
        const int64_t observations = read_i64(reader);
        const int64_t confirmed_self = read_i64(reader);
        const int64_t confirmed_other = read_i64(reader);
        trust.context_attribution.set_scale_state(self_sum_squares, self_count,
                                                   other_sum_squares, other_count,
                                                   observations, confirmed_self,
                                                   confirmed_other);
        const bool has_pending = read_pod<uint8_t>(reader) != 0;
        if (has_pending) {
            const uint64_t key = read_u64(reader);
            const double value = read_f64(reader);
            trust.pending_context_credit = std::make_pair(key, value);
        } else {
            trust.pending_context_credit.reset();
        }
    }

    // long_range_map_ is keyed on a fixed-size byte key (LongRangeKey, not
    // a packed uint64 like the ordinary banks -- see cortex.hpp for why:
    // a std::string key meant a heap allocation on every processed byte,
    // which profiling showed dominated long_range_bank's cost), and
    // long_range_order_ is the FIFO eviction queue in insertion order --
    // both need to round-trip for the same reason nursery.order does: the
    // wake/sleep harness always saves a column and reloads it before
    // merging. Wire format is unchanged from the std::string version this
    // replaced (u64 length + raw bytes) for compatibility with the Python
    // importer (native_cortex_import.py's _encode_long_range).
    static void write_long_range_key(std::vector<uint8_t>& out, const LongRangeKey& key) {
        write_u64(out, static_cast<uint64_t>(key.length));
        const size_t at = out.size();
        out.resize(at + key.length);
        if (key.length) std::memcpy(out.data() + at, key.bytes, key.length);
    }
    static LongRangeKey read_long_range_key(Reader& reader) {
        const uint64_t size = read_u64(reader);
        if (size > LongRangeKey::kMaxBytes) {
            throw std::runtime_error("long-range key in checkpoint exceeds max length");
        }
        need(reader, size);
        LongRangeKey key;
        key.length = static_cast<uint8_t>(size);
        if (size) std::memcpy(key.bytes, reader.data + reader.offset, size);
        reader.offset += size;
        return key;
    }

    static void write_long_range(std::vector<uint8_t>& out, const GrowableCortex& cortex) {
        write_u64(out, static_cast<uint64_t>(cortex.long_range_map_.size()));
        for (const auto& [key, slot] : cortex.long_range_map_) {
            write_long_range_key(out, key);
            write_row(out, cortex.arena_, slot);
        }
        write_u64(out, static_cast<uint64_t>(cortex.long_range_history_.size()));
        for (uint8_t byte_value : cortex.long_range_history_) write_pod<uint8_t>(out, byte_value);
        write_u64(out, static_cast<uint64_t>(cortex.long_range_order_.size()));
        for (const LongRangeKey& key : cortex.long_range_order_) write_long_range_key(out, key);
    }
    static void read_long_range(Reader& reader, GrowableCortex& cortex) {
        const uint64_t entry_count = read_u64(reader);
        for (uint64_t i = 0; i < entry_count; ++i) {
            LongRangeKey key = read_long_range_key(reader);
            const uint32_t slot = cortex.arena_.create_row();
            read_row(reader, cortex.arena_, slot);
            cortex.long_range_map_.emplace(key, slot);
        }
        const uint64_t history_size = read_u64(reader);
        for (uint64_t i = 0; i < history_size; ++i) {
            cortex.long_range_history_.push_back(read_pod<uint8_t>(reader));
        }
        const uint64_t order_size = read_u64(reader);
        for (uint64_t i = 0; i < order_size; ++i) {
            cortex.long_range_order_.push_back(read_long_range_key(reader));
        }
    }

    // Deep bands (long_range_bands): the sparse multi-depth structure
    // built on top of long_range_bank above -- previously never
    // serialized at all, so every learned association, and the tier-3/4
    // persistence state built on it, silently reset on the very next
    // save/load round trip (which the wake/sleep harness does every
    // epoch, for every column -- see write_config's comment). Tier is
    // written per mature entry (2 is the common case and costs one byte,
    // not worth special-casing away); nursery/order state round-trips for
    // the same reason nursery.order does elsewhere in this file. Inline
    // nursery rows write a flag byte then either the inline (key,count)
    // pairs directly or -- if overflowed -- the real arena32_ row via the
    // same write_row/read_row every mature row already uses.
    static void write_inline_nursery_row(std::vector<uint8_t>& out,
                                         const RowArena<int16_t>& arena32,
                                         const InlineNurseryRow& row) {
        write_pod<uint8_t>(out, row.overflowed() ? 1 : 0);
        if (row.overflowed()) {
            write_row(out, arena32, row.overflow_slot);
            return;
        }
        write_pod<uint8_t>(out, row.occupied);
        for (int32_t i = 0; i < row.occupied; ++i) {
            write_pod<uint8_t>(out, row.keys[i]);
            write_i64(out, row.counts[i]);
        }
    }
    static InlineNurseryRow read_inline_nursery_row(Reader& reader, RowArena<int16_t>& arena32) {
        InlineNurseryRow row;
        const uint8_t overflowed = read_pod<uint8_t>(reader);
        if (overflowed) {
            row.overflow_slot = arena32.create_row();
            read_row(reader, arena32, row.overflow_slot);
            return row;
        }
        row.occupied = read_pod<uint8_t>(reader);
        for (int32_t i = 0; i < row.occupied; ++i) {
            row.keys[i] = read_pod<uint8_t>(reader);
            row.counts[i] = static_cast<int16_t>(read_i64(reader));
        }
        return row;
    }

    static void write_deep_bands(std::vector<uint8_t>& out, const GrowableCortex& cortex) {
        const int32_t band_count =
            cortex.config_.long_range_bands ? cortex.config_.long_range_band_count : 0;
        write_i32(out, band_count);
        if (band_count == 0) return;

        for (int32_t b = 0; b < band_count; ++b) {
            write_u64(out, static_cast<uint64_t>(cortex.deep_mature_hash_[b].size()));
            cortex.deep_mature_hash_[b].for_each([&](uint64_t key, uint32_t slot) {
                write_u64(out, key);
                write_row(out, cortex.arena32_, slot);
                const uint32_t tier = cortex.deep_tier_hash_[b].find(key);
                write_pod<uint8_t>(out, tier == IndexMap::kEmpty ? 2 : static_cast<uint8_t>(tier));
            });
            write_u64(out, static_cast<uint64_t>(cortex.deep_nursery_hash_[b].map.size()));
            cortex.deep_nursery_hash_[b].map.for_each([&](uint64_t key, uint32_t inline_slot) {
                write_u64(out, key);
                write_inline_nursery_row(out, cortex.arena32_,
                                         cortex.deep_nursery_hash_[b].arena.row(inline_slot));
            });
            write_u64(out, static_cast<uint64_t>(cortex.deep_nursery_hash_[b].order.size()));
            for (uint64_t key : cortex.deep_nursery_hash_[b].order) write_u64(out, key);
            write_u64(out, static_cast<uint64_t>(cortex.deep_mature_order_hash_[b].size()));
            for (uint64_t key : cortex.deep_mature_order_hash_[b]) write_u64(out, key);

            write_u64(out, static_cast<uint64_t>(cortex.deep_mature_exact_[b].size()));
            for (const auto& [key, slot] : cortex.deep_mature_exact_[b]) {
                write_long_range_key(out, key);
                write_row(out, cortex.arena32_, slot);
                const auto tier_it = cortex.deep_tier_exact_[b].find(key);
                write_pod<uint8_t>(
                    out, tier_it == cortex.deep_tier_exact_[b].end() ? 2 : tier_it->second);
            }
            write_u64(out, static_cast<uint64_t>(cortex.deep_nursery_exact_[b].size()));
            for (const auto& [key, slot] : cortex.deep_nursery_exact_[b]) {
                write_long_range_key(out, key);
                write_row(out, cortex.arena32_, slot);
            }
            write_u64(out, static_cast<uint64_t>(cortex.deep_nursery_order_exact_[b].size()));
            for (const LongRangeKey& key : cortex.deep_nursery_order_exact_[b]) {
                write_long_range_key(out, key);
            }
            write_u64(out, static_cast<uint64_t>(cortex.deep_mature_order_exact_[b].size()));
            for (const LongRangeKey& key : cortex.deep_mature_order_exact_[b]) {
                write_long_range_key(out, key);
            }
        }

        // Rolling incremental hash state -- needed for bit-exact key
        // construction continuity right after resume, same reason
        // long_range_history_ above round-trips.
        write_u64(out, static_cast<uint64_t>(cortex.deep_band_history_.size()));
        for (uint8_t byte_value : cortex.deep_band_history_) write_pod<uint8_t>(out, byte_value);
        for (int32_t b = 0; b < CortexConfig::kMaxLongRangeBands; ++b) {
            write_u64(out, cortex.deep_band_hash_[b]);
        }
        write_i64(out, cortex.deep_band_steps_);

        write_u64(out, static_cast<uint64_t>(cortex.span_surprise_window_.size()));
        for (double value : cortex.span_surprise_window_) write_f64(out, value);
        write_f64(out, cortex.span_surprise_sum_);

        write_u64(out, static_cast<uint64_t>(cortex.replay_queue_.size()));
        for (const ReplayCandidate& candidate : cortex.replay_queue_) {
            write_i32(out, candidate.band);
            write_long_range_key(out, candidate.exact_key);
            write_u64(out, candidate.hash_key);
            write_pod<uint8_t>(out, candidate.target);
        }
    }

    static void read_deep_bands(Reader& reader, GrowableCortex& cortex) {
        const int32_t band_count = read_i32(reader);
        if (band_count == 0) return;
        if (band_count != static_cast<int32_t>(cortex.deep_mature_hash_.size())) {
            throw std::runtime_error("ALM checkpoint deep-band count does not match config");
        }

        for (int32_t b = 0; b < band_count; ++b) {
            const uint64_t mature_hash_count = read_u64(reader);
            for (uint64_t i = 0; i < mature_hash_count; ++i) {
                const uint64_t key = read_u64(reader);
                const uint32_t slot = cortex.arena32_.create_row();
                read_row(reader, cortex.arena32_, slot);
                const uint8_t tier = read_pod<uint8_t>(reader);
                cortex.deep_mature_hash_[b].insert(key, slot);
                if (tier == 3) {
                    cortex.deep_tier_hash_[b].insert(key, 3);
                    cortex.deep_consolidated_members_hash_[b].insert(key);
                } else if (tier == 4) {
                    cortex.deep_tier_hash_[b].insert(key, 4);
                    cortex.deep_permanent_members_hash_[b].insert(key);
                }
            }
            const uint64_t nursery_hash_count = read_u64(reader);
            for (uint64_t i = 0; i < nursery_hash_count; ++i) {
                const uint64_t key = read_u64(reader);
                InlineNurseryRow row = read_inline_nursery_row(reader, cortex.arena32_);
                const uint32_t inline_slot = cortex.deep_nursery_hash_[b].arena.create_row();
                cortex.deep_nursery_hash_[b].arena.row(inline_slot) = row;
                cortex.deep_nursery_hash_[b].map.insert(key, inline_slot);
            }
            const uint64_t nursery_order_count = read_u64(reader);
            for (uint64_t i = 0; i < nursery_order_count; ++i) {
                cortex.deep_nursery_hash_[b].order.push_back(read_u64(reader));
            }
            const uint64_t mature_order_count = read_u64(reader);
            for (uint64_t i = 0; i < mature_order_count; ++i) {
                cortex.deep_mature_order_hash_[b].push_back(read_u64(reader));
            }

            const uint64_t mature_exact_count = read_u64(reader);
            for (uint64_t i = 0; i < mature_exact_count; ++i) {
                LongRangeKey key = read_long_range_key(reader);
                const uint32_t slot = cortex.arena32_.create_row();
                read_row(reader, cortex.arena32_, slot);
                const uint8_t tier = read_pod<uint8_t>(reader);
                cortex.deep_mature_exact_[b].emplace(key, slot);
                if (tier == 3) {
                    cortex.deep_tier_exact_[b][key] = 3;
                    cortex.deep_consolidated_members_exact_[b].insert(key);
                } else if (tier == 4) {
                    cortex.deep_tier_exact_[b][key] = 4;
                    cortex.deep_permanent_members_exact_[b].insert(key);
                }
            }
            const uint64_t nursery_exact_count = read_u64(reader);
            for (uint64_t i = 0; i < nursery_exact_count; ++i) {
                LongRangeKey key = read_long_range_key(reader);
                const uint32_t slot = cortex.arena32_.create_row();
                read_row(reader, cortex.arena32_, slot);
                cortex.deep_nursery_exact_[b].emplace(key, slot);
            }
            const uint64_t nursery_order_exact_count = read_u64(reader);
            for (uint64_t i = 0; i < nursery_order_exact_count; ++i) {
                cortex.deep_nursery_order_exact_[b].push_back(read_long_range_key(reader));
            }
            const uint64_t mature_order_exact_count = read_u64(reader);
            for (uint64_t i = 0; i < mature_order_exact_count; ++i) {
                cortex.deep_mature_order_exact_[b].push_back(read_long_range_key(reader));
            }
        }

        const uint64_t history_size = read_u64(reader);
        for (uint64_t i = 0; i < history_size; ++i) {
            cortex.deep_band_history_.push_back(read_pod<uint8_t>(reader));
        }
        for (int32_t b = 0; b < CortexConfig::kMaxLongRangeBands; ++b) {
            cortex.deep_band_hash_[b] = read_u64(reader);
        }
        cortex.deep_band_steps_ = read_i64(reader);

        const uint64_t span_window_size = read_u64(reader);
        for (uint64_t i = 0; i < span_window_size; ++i) {
            cortex.span_surprise_window_.push_back(read_f64(reader));
        }
        cortex.span_surprise_sum_ = read_f64(reader);

        const uint64_t replay_count = read_u64(reader);
        for (uint64_t i = 0; i < replay_count; ++i) {
            const int32_t band = read_i32(reader);
            LongRangeKey exact_key = read_long_range_key(reader);
            const uint64_t hash_key = read_u64(reader);
            const uint8_t target = read_pod<uint8_t>(reader);
            cortex.replay_queue_.push_back(ReplayCandidate{band, exact_key, hash_key, target});
        }
    }

    // The touched-key sets `begin_lineage_tracking()` builds are exactly
    // what `consolidate_delta` needs, and the real wake/sleep harness
    // always saves a column to disk and reloads it before merging -- so
    // this state has to survive a round trip, not just live in memory for
    // the object that called begin_lineage_tracking() directly. Losing it
    // silently was a real crash (CortexConsolidate indexing an empty,
    // wrongly-sized touched_sequences_ vector); this is the fix.
    static void write_lineage_tracking(std::vector<uint8_t>& out,
                                       const GrowableCortex& cortex) {
        write_pod<uint8_t>(out, cortex.tracking_ ? 1 : 0);
        if (!cortex.tracking_) return;
        write_u64(out, static_cast<uint64_t>(cortex.touched_contexts_.size()));
        for (uint64_t key : cortex.touched_contexts_) write_u64(out, key);
        write_i32(out, static_cast<int32_t>(cortex.touched_sequences_.size()));
        for (const auto& bank_set : cortex.touched_sequences_) {
            write_u64(out, static_cast<uint64_t>(bank_set.size()));
            for (uint64_t key : bank_set) write_u64(out, key);
        }
    }
    static void read_lineage_tracking(Reader& reader, GrowableCortex& cortex) {
        const bool tracking = read_pod<uint8_t>(reader) != 0;
        cortex.tracking_ = tracking;
        if (!tracking) return;
        const uint64_t context_count = read_u64(reader);
        cortex.touched_contexts_.clear();
        for (uint64_t i = 0; i < context_count; ++i) {
            cortex.touched_contexts_.insert(read_u64(reader));
        }
        const int32_t bank_count = read_i32(reader);
        cortex.touched_sequences_.assign(bank_count, {});
        for (int32_t bank = 0; bank < bank_count; ++bank) {
            const uint64_t key_count = read_u64(reader);
            for (uint64_t i = 0; i < key_count; ++i) {
                cortex.touched_sequences_[bank].insert(read_u64(reader));
            }
        }
    }

    template <typename T>
    static void write_vector(std::vector<uint8_t>& out, const std::vector<T>& values) {
        write_u64(out, static_cast<uint64_t>(values.size()));
        if (!values.empty()) {
            const size_t at = out.size();
            out.resize(at + values.size() * sizeof(T));
            std::memcpy(out.data() + at, values.data(), values.size() * sizeof(T));
        }
    }
    template <typename T>
    static void read_vector(Reader& reader, std::vector<T>& values) {
        const uint64_t count = read_u64(reader);
        need(reader, count * sizeof(T));
        values.resize(count);
        if (count) {
            std::memcpy(values.data(), reader.data + reader.offset, count * sizeof(T));
            reader.offset += count * sizeof(T);
        }
    }

    // Templated on the arena's count width (int64_t for most banks, the
    // deepest ordinary sequence bank's int32_t arena_deep_bank_ -- see
    // cortex.hpp) -- but the WIRE FORMAT is unchanged either way: a row's
    // counts are always written as int64_t on disk regardless of which
    // in-memory arena they came from (for_each's callback already
    // widens), so this needs no checkpoint version bump and existing
    // checkpoints stay loadable. Only the in-memory routing (which arena
    // a given bank's rows load into) changes.
    template <typename CountT>
    static void write_row(std::vector<uint8_t>& out, const RowArena<CountT>& arena, uint32_t slot) {
        const uint16_t size = arena.size(slot);
        write_pod<uint16_t>(out, size);
        arena.for_each(slot, [&](uint8_t key, int64_t count) {
            write_pod<uint8_t>(out, key);
            write_i64(out, count);
        });
    }
    template <typename CountT>
    static void read_row(Reader& reader, RowArena<CountT>& arena, uint32_t slot) {
        const uint16_t size = read_pod<uint16_t>(reader);
        for (uint16_t i = 0; i < size; ++i) {
            const uint8_t key = read_pod<uint8_t>(reader);
            const int64_t count = read_i64(reader);
            arena.increment(slot, key, count);
            // Compatibility with checkpoints written before RowArena stopped
            // counting unseen keys after a narrow row saturated. Those files
            // declared the phantom zero cells in `size`, but dense for_each()
            // did not emit them. Positive stored counts sum to the saturated
            // total, so reaching the ceiling marks the actual end of the row.
            if constexpr (sizeof(CountT) < sizeof(int64_t)) {
                if (arena.total(slot) == std::numeric_limits<CountT>::max()) break;
            }
        }
    }

    template <typename CountT>
    static void write_level(std::vector<uint8_t>& out, const RowArena<CountT>& arena,
                            const IndexMap& map) {
        write_u64(out, static_cast<uint64_t>(map.size()));
        map.for_each([&](uint64_t key, uint32_t slot) {
            write_u64(out, key);
            write_row(out, arena, slot);
        });
    }
    template <typename CountT>
    static void read_level(Reader& reader, RowArena<CountT>& arena, IndexMap& map) {
        const uint64_t count = read_u64(reader);
        map.reset(count ? count * 2 : 16);
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t key = read_u64(reader);
            const uint32_t slot = arena.create_row();
            read_row(reader, arena, slot);
            map.insert(key, slot);
        }
    }
};

}  // namespace alm
