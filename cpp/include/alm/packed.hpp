// Packed cortex: the mature phenotype, frozen for inference.
//
// A growable cortex carries everything it needs in order to keep developing:
// spare row capacity, free lists, nursery banks, insertion-order queues. Once
// a body has been consolidated none of that is load-bearing, so the packed
// form drops it and stores each level as one contiguous CSR block. Same
// arithmetic, same bits, a fraction of the memory and far fewer cache misses.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "alm/cortex.hpp"
#include "alm/hash_map.hpp"

namespace alm {

// One frozen level: key -> (successor keys, counts, total).
struct PackedLevel {
    IndexMap index;                  // context key -> row slot
    std::vector<uint32_t> offsets;   // row slot -> start in keys/counts
    std::vector<uint16_t> sizes;     // distinct successors in the row
    std::vector<int64_t> totals;     // row sum
    std::vector<uint8_t> keys;       // successor bytes, all rows concatenated
    std::vector<int64_t> counts;     // parallel to keys

    size_t memory_bytes() const noexcept {
        return index.memory_bytes() + offsets.capacity() * sizeof(uint32_t) +
               sizes.capacity() * sizeof(uint16_t) +
               totals.capacity() * sizeof(int64_t) + keys.capacity() +
               counts.capacity() * sizeof(int64_t);
    }

    int64_t get(uint32_t slot, uint8_t key) const noexcept {
        const uint32_t start = offsets[slot];
        const uint16_t count = sizes[slot];
        const uint8_t* row = keys.data() + start;
        uint16_t position = 0;
#if defined(_MSC_VER) || defined(__SSE2__)
        const __m128i needle = _mm_set1_epi8(static_cast<char>(key));
        for (; position + 16 <= count; position += 16) {
            const __m128i block =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + position));
            const int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(block, needle));
            if (mask) {
#if defined(_MSC_VER)
                unsigned long bit;
                _BitScanForward(&bit, static_cast<unsigned long>(mask));
                return counts[start + position + bit];
#else
                return counts[start + position +
                              __builtin_ctz(static_cast<unsigned>(mask))];
#endif
            }
        }
#endif
        for (; position < count; ++position) {
            if (row[position] == key) return counts[start + position];
        }
        return 0;
    }
};

class PackedCortex {
public:
    static PackedCortex from_growable(const GrowableCortex& source) {
        PackedCortex packed;
        packed.config_ = source.config();
        packed.bank_count_ = source.bank_count();
        packed.history_mask_ =
            packed.config_.max_context_bytes >= 8
                ? ~0ULL
                : ((1ULL << (8 * packed.config_.max_context_bytes)) - 1ULL);
        packed.context_masks_.resize(packed.bank_count_);
        for (int32_t bank = 0; bank < packed.bank_count_; ++bank) {
            const int32_t depth = bank + 3;
            packed.context_masks_[bank] =
                depth >= 8 ? ~0ULL : ((1ULL << (8 * depth)) - 1ULL);
        }
        packed.unigram_counts_ = source.unigram_counts();
        packed.unigram_sum_ = source.unigram_sum();
        packed.context_totals_ = source.context_totals();
        packed.live_assemblies_ = source.live_assemblies();
        packed.steps_ = source.steps();

        const RowArena<int64_t>& arena = source.arena();

        // Transitions are dense by construction: 256 rows, one per assembly.
        packed.transitions_.offsets.resize(256);
        packed.transitions_.sizes.resize(256);
        packed.transitions_.totals.resize(256);
        for (int index = 0; index < 256; ++index) {
            const uint32_t row = source.transition_row(index);
            packed.transitions_.offsets[index] =
                static_cast<uint32_t>(packed.transitions_.keys.size());
            uint16_t written = 0;
            arena.for_each(row, [&](uint8_t key, int64_t count) {
                packed.transitions_.keys.push_back(key);
                packed.transitions_.counts.push_back(count);
                ++written;
            });
            packed.transitions_.sizes[index] = written;
            packed.transitions_.totals[index] = arena.total(row);
        }

        packed.freeze_level(packed.residual_, source.residual_map(), arena);
        packed.sequences_.resize(packed.bank_count_);
        const RowArena<int32_t>& deep_arena = source.deep_bank_arena();
        for (int32_t bank = 0; bank < packed.bank_count_; ++bank) {
            if (source.bank_is_deep(bank)) {
                packed.freeze_level(packed.sequences_[bank], source.sequence_map(bank),
                                    deep_arena);
            } else {
                packed.freeze_level(packed.sequences_[bank], source.sequence_map(bank),
                                    arena);
            }
        }
        return packed;
    }

    // Frozen scoring. Mirrors the learn=False path of the growable loop, so
    // the two produce identical bits on identical anatomy.
    void score_chunk(const uint8_t* data, size_t count, double* out) {
        const double prior = config_.prior;
        const double basal_strength = config_.basal_strength;
        const double parent_strength = config_.parent_strength;
        const int32_t max_context = config_.max_context_bytes;

        uint64_t contexts[8];

        for (size_t index = 0; index < count; ++index) {
            const uint8_t target = data[index];

            const int32_t available =
                history_length_ < max_context ? history_length_ : max_context;
            const int32_t active_banks = available > 2 ? available - 2 : 0;
            for (int32_t bank = 0; bank < active_banks; ++bank) {
                contexts[bank] = packed_history_ & context_masks_[bank];
                sequences_[bank].index.prefetch(contexts[bank]);
            }

            double probability;
            if (previous_ < 0) {
                const double basal_total =
                    static_cast<double>(unigram_sum_) + 256.0 * prior;
                probability =
                    (static_cast<double>(unigram_counts_[target]) + prior) / basal_total;
            } else {
                const double basal_total =
                    static_cast<double>(context_totals_[previous_]) + 256.0 * prior;
                probability =
                    (static_cast<double>(transitions_.get(
                         static_cast<uint32_t>(previous_), target)) +
                     prior) /
                    basal_total;
            }

            if (older_ >= 0 && previous_ >= 0) {
                const uint64_t key = (static_cast<uint64_t>(older_) << 8) |
                                     static_cast<uint64_t>(previous_);
                const uint32_t slot = residual_.index.find(key);
                if (slot != IndexMap::kEmpty) {
                    probability = (static_cast<double>(residual_.get(slot, target)) +
                                   basal_strength * probability) /
                                  (static_cast<double>(residual_.totals[slot]) +
                                   basal_strength);
                }
            }

            for (int32_t bank = 0; bank < active_banks; ++bank) {
                const PackedLevel& level = sequences_[bank];
                const uint32_t slot = level.index.find(contexts[bank]);
                if (slot != IndexMap::kEmpty) {
                    probability = (static_cast<double>(level.get(slot, target)) +
                                   parent_strength * probability) /
                                  (static_cast<double>(level.totals[slot]) +
                                   parent_strength);
                }
            }

            if (steps_ != 0) out[index] = probability;

            if (live_assemblies_[target]) {
                older_ = previous_;
                previous_ = target;
                packed_history_ =
                    ((packed_history_ << 8) | static_cast<uint64_t>(target)) &
                    history_mask_;
                if (history_length_ < max_context) ++history_length_;
            } else {
                older_ = -1;
                previous_ = -1;
                packed_history_ = 0;
                history_length_ = 0;
            }
        }
    }

    void reset_episode() {
        older_ = -1;
        previous_ = -1;
        packed_history_ = 0;
        history_length_ = 0;
    }

    size_t memory_bytes() const noexcept {
        size_t bytes = transitions_.memory_bytes() + residual_.memory_bytes();
        for (const auto& level : sequences_) bytes += level.memory_bytes();
        bytes += unigram_counts_.capacity() * sizeof(int64_t);
        bytes += context_totals_.capacity() * sizeof(int64_t);
        bytes += live_assemblies_.capacity();
        return bytes;
    }

    int64_t live_sequence_count() const noexcept {
        int64_t total = 0;
        for (const auto& level : sequences_)
            total += static_cast<int64_t>(level.index.size());
        return total;
    }
    int64_t live_context_count() const noexcept {
        return static_cast<int64_t>(residual_.index.size());
    }
    int64_t steps() const noexcept { return steps_; }
    const CortexConfig& config() const noexcept { return config_; }

private:
    // Templated on the SOURCE arena's count width (int64_t for most banks,
    // int32_t for the deepest ordinary bank's arena_deep_bank_ -- see
    // cortex.hpp); the packed/frozen level itself always stores int64_t
    // regardless, since for_each already widens on read.
    template <typename CountT>
    void freeze_level(PackedLevel& level, const IndexMap& source,
                      const RowArena<CountT>& arena) {
        const size_t entries = source.size();
        level.index.reset(entries ? entries * 2 : 16);
        level.offsets.reserve(entries);
        level.sizes.reserve(entries);
        level.totals.reserve(entries);
        source.for_each([&](uint64_t key, uint32_t row) {
            const uint32_t slot = static_cast<uint32_t>(level.offsets.size());
            level.offsets.push_back(static_cast<uint32_t>(level.keys.size()));
            uint16_t written = 0;
            arena.for_each(row, [&](uint8_t successor, int64_t count) {
                level.keys.push_back(successor);
                level.counts.push_back(count);
                ++written;
            });
            level.sizes.push_back(written);
            level.totals.push_back(arena.total(row));
            level.index.insert(key, slot);
        });
    }

    CortexConfig config_;
    PackedLevel transitions_;
    PackedLevel residual_;
    std::vector<PackedLevel> sequences_;
    std::vector<uint64_t> context_masks_;
    std::vector<int64_t> unigram_counts_;
    std::vector<int64_t> context_totals_;
    std::vector<uint8_t> live_assemblies_;

    uint64_t history_mask_ = 0;
    uint64_t packed_history_ = 0;
    int32_t history_length_ = 0;
    int32_t bank_count_ = 0;
    int32_t previous_ = -1;
    int32_t older_ = -1;
    int64_t steps_ = 0;
    int64_t unigram_sum_ = 0;
};

}  // namespace alm
