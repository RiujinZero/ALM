// Successor rows: a byte -> count table, one per learned context.
//
// Python stores these as dict[int, int], one heap object per context. There
// are millions of them, most holding only a handful of successors, so the
// per-dict overhead dominates both memory and lookup cost. Here every row
// lives in one shared arena: sparse rows are a run of (key, count) pairs
// scanned 16 keys at a time with SSE2, and a row only becomes a dense
// 256-entry table once it has enough distinct successors to earn it.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#include <emmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace alm {

// Above this many distinct successors a row switches to a direct-indexed
// dense table. Sparse scanning stays cheaper below it, and the vast majority
// of deep-context rows never get close.
constexpr uint32_t kDenseThreshold = 48;
constexpr uint32_t kSizeClasses = 8;  // capacities 4, 8, 16, 32, 64 ...

// CountT is a mixed-precision sizing knob: int64_t (the original, always
// safe) or a narrower type for rows that measurement shows stay well
// within its range for the row population's realistic lifetime -- see
// max_count_value/max_sequence_bank_count in cortex.hpp, which is how
// that was actually checked (bank 3 tops out around 1.4% of int32 range
// after 2 production epochs; banks 0-2 were already 38-49% of it and are
// NOT safe to narrow this way).
template <typename CountT = int64_t>
struct Row {
    uint32_t offset = 0;   // index into sparse arena, or into dense arena
    uint16_t size = 0;     // distinct successors
    uint16_t capacity = 0; // sparse slots reserved (0 once dense)
    CountT total = 0;      // sum of counts, maintained incrementally
    bool dense = false;
};

template <typename CountT = int64_t>
class RowArena {
public:
    RowArena() {
        rows_.reserve(1024);
        sparse_keys_.reserve(1 << 16);
        sparse_counts_.reserve(1 << 16);
        free_lists_.resize(kSizeClasses);
    }

    uint32_t create_row() {
        if (!free_rows_.empty()) {
            const uint32_t index = free_rows_.back();
            free_rows_.pop_back();
            rows_[index] = Row<CountT>{};
            return index;
        }
        rows_.push_back(Row<CountT>{});
        return static_cast<uint32_t>(rows_.size() - 1);
    }

    // Returns a row's storage to the arena. Nursery paths are evicted in
    // large numbers over a long wake generation, so without this the arena
    // would only ever grow.
    void release_row(uint32_t index) {
        Row<CountT>& r = rows_[index];
        if (r.dense) {
            dense_free_.push_back(r.offset);
        } else if (r.capacity) {
            free_lists_[size_class(r.capacity)].push_back(r.offset);
        }
        r = Row<CountT>{};
        free_rows_.push_back(index);
    }

    size_t row_count() const noexcept { return rows_.size(); }
    Row<CountT>& row(uint32_t index) noexcept { return rows_[index]; }
    const Row<CountT>& row(uint32_t index) const noexcept { return rows_[index]; }
    int64_t total(uint32_t index) const noexcept {
        return static_cast<int64_t>(rows_[index].total);
    }
    uint16_t size(uint32_t index) const noexcept { return rows_[index].size; }

    int64_t get(uint32_t index, uint8_t key) const noexcept {
        const Row<CountT>& r = rows_[index];
        if (r.dense) return static_cast<int64_t>(dense_counts_[r.offset + key]);
        const int position = find_sparse(r, key);
        return position < 0 ? 0 : static_cast<int64_t>(sparse_counts_[r.offset + position]);
    }

    // Saturating: clamps at CountT's max instead of wrapping on overflow.
    // Cheap insurance for a narrower CountT -- the actual safety margin
    // comes from measuring real max counts before choosing CountT (see
    // the type-level comment above), not from this clamp masking a wrong
    // choice, but a loud precision ceiling beats silent wraparound.
    void increment(uint32_t index, uint8_t key, int64_t amount = 1) {
        Row<CountT>& r = rows_[index];
        const CountT max_value = std::numeric_limits<CountT>::max();
        // Once the row total is saturated there is no representable mass to
        // assign. In particular, do not register a previously unseen key:
        // dense for_each() skips zero-valued cells, so incrementing `size`
        // for a zero increment makes the serialized row declare more cells
        // than write_row() emits and corrupts every section that follows it.
        if (amount <= 0 || r.total == max_value) return;
        const CountT clamped_amount =
            (static_cast<int64_t>(max_value) - static_cast<int64_t>(r.total) < amount)
                ? static_cast<CountT>(max_value - r.total)
                : static_cast<CountT>(amount);
        r.total += clamped_amount;
        if (r.dense) {
            CountT& cell = dense_counts_[r.offset + key];
            if (cell == 0) ++r.size;
            cell = (max_value - cell < clamped_amount) ? max_value
                                                        : static_cast<CountT>(cell + clamped_amount);
            return;
        }
        const int position = find_sparse(r, key);
        if (position >= 0) {
            CountT& cell = sparse_counts_[r.offset + position];
            cell = (max_value - cell < clamped_amount) ? max_value
                                                        : static_cast<CountT>(cell + clamped_amount);
            return;
        }
        if (static_cast<uint32_t>(r.size) + 1u > kDenseThreshold) {
            promote_dense(r);
            CountT& cell = dense_counts_[r.offset + key];
            if (cell == 0) ++r.size;
            cell = (max_value - cell < clamped_amount) ? max_value
                                                        : static_cast<CountT>(cell + clamped_amount);
            return;
        }
        if (r.size == r.capacity) {
            reserve_sparse(r, r.capacity ? static_cast<uint16_t>(r.capacity * 2) : 4);
        }
        sparse_keys_[r.offset + r.size] = key;
        sparse_counts_[r.offset + r.size] = clamped_amount;
        ++r.size;
    }

    template <typename Fn>
    void for_each(uint32_t index, Fn&& fn) const {
        const Row<CountT>& r = rows_[index];
        if (r.dense) {
            for (uint32_t key = 0; key < 256; ++key) {
                const CountT count = dense_counts_[r.offset + key];
                if (count) fn(static_cast<uint8_t>(key), static_cast<int64_t>(count));
            }
            return;
        }
        for (uint16_t slot = 0; slot < r.size; ++slot) {
            fn(sparse_keys_[r.offset + slot], static_cast<int64_t>(sparse_counts_[r.offset + slot]));
        }
    }

    void copy_row_into(uint32_t index, RowArena<CountT>& target, uint32_t target_index) const {
        for_each(index, [&](uint8_t key, int64_t count) {
            target.increment(target_index, key, count);
        });
    }

    size_t memory_bytes() const noexcept {
        return rows_.capacity() * sizeof(Row<CountT>) +
               sparse_keys_.capacity() * sizeof(uint8_t) +
               sparse_counts_.capacity() * sizeof(CountT) +
               dense_counts_.capacity() * sizeof(CountT);
    }

    // Mixed-precision sizing check: the largest single count/total value
    // currently stored, live cells only (sparse_counts_/dense_counts_ may
    // hold stale/freed slots). Tells us how much headroom a narrower
    // integer type would actually have on real, at-scale data before
    // committing to changing anything.
    // `skip_rows` excludes the first N row indices (e.g. the 256
    // always-dense transition_rows_, created first and fed on nearly
    // every byte across the whole training history -- a fixed-size,
    // high-traffic outlier unrepresentative of the millions of far
    // sparser deep-context rows that actually dominate total memory).
    int64_t max_count_value(uint32_t skip_rows = 0) const noexcept {
        int64_t max_value = 0;
        for (uint32_t idx = skip_rows; idx < rows_.size(); ++idx) {
            const Row<CountT>& row = rows_[idx];
            if (static_cast<int64_t>(row.total) > max_value) max_value = row.total;
            if (row.dense) {
                for (uint32_t i = 0; i < 256; ++i) {
                    const int64_t v = static_cast<int64_t>(dense_counts_[row.offset + i]);
                    if (v > max_value) max_value = v;
                }
            } else {
                for (uint32_t i = 0; i < row.size; ++i) {
                    const int64_t v = static_cast<int64_t>(sparse_counts_[row.offset + i]);
                    if (v > max_value) max_value = v;
                }
            }
        }
        return max_value;
    }

private:
    static uint32_t size_class(uint16_t capacity) noexcept {
        uint32_t cls = 0;
        uint32_t width = 4;
        while (width < capacity && cls + 1 < kSizeClasses) {
            width <<= 1;
            ++cls;
        }
        return cls;
    }

    int find_sparse(const Row<CountT>& r, uint8_t key) const noexcept {
        const uint8_t* keys = sparse_keys_.data() + r.offset;
        const uint16_t count = r.size;
        uint16_t slot = 0;
#if defined(_MSC_VER) || defined(__SSE2__)
        const __m128i needle = _mm_set1_epi8(static_cast<char>(key));
        for (; slot + 16 <= count; slot += 16) {
            const __m128i block =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(keys + slot));
            const int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(block, needle));
            if (mask) {
#if defined(_MSC_VER)
                unsigned long bit;
                _BitScanForward(&bit, static_cast<unsigned long>(mask));
                return static_cast<int>(slot + bit);
#else
                return static_cast<int>(slot + __builtin_ctz(static_cast<unsigned>(mask)));
#endif
            }
        }
#endif
        for (; slot < count; ++slot) {
            if (keys[slot] == key) return static_cast<int>(slot);
        }
        return -1;
    }

    void reserve_sparse(Row<CountT>& r, uint16_t capacity) {
        const uint32_t cls = size_class(capacity);
        uint32_t offset;
        if (!free_lists_[cls].empty()) {
            offset = free_lists_[cls].back();
            free_lists_[cls].pop_back();
        } else {
            offset = static_cast<uint32_t>(sparse_keys_.size());
            const uint32_t width = 4u << cls;
            sparse_keys_.resize(offset + width);
            sparse_counts_.resize(offset + width);
        }
        if (r.size) {
            std::memcpy(sparse_keys_.data() + offset, sparse_keys_.data() + r.offset,
                        r.size * sizeof(uint8_t));
            std::memcpy(sparse_counts_.data() + offset, sparse_counts_.data() + r.offset,
                        r.size * sizeof(CountT));
            free_lists_[size_class(r.capacity)].push_back(r.offset);
        }
        r.offset = offset;
        r.capacity = static_cast<uint16_t>(4u << cls);
    }

    void promote_dense(Row<CountT>& r) {
        uint32_t offset;
        if (!dense_free_.empty()) {
            offset = dense_free_.back();
            dense_free_.pop_back();
            std::memset(dense_counts_.data() + offset, 0, 256 * sizeof(CountT));
        } else {
            offset = static_cast<uint32_t>(dense_counts_.size());
            dense_counts_.resize(offset + 256, 0);
        }
        for (uint16_t slot = 0; slot < r.size; ++slot) {
            dense_counts_[offset + sparse_keys_[r.offset + slot]] =
                sparse_counts_[r.offset + slot];
        }
        if (r.capacity) free_lists_[size_class(r.capacity)].push_back(r.offset);
        r.offset = offset;
        r.capacity = 0;
        r.dense = true;
    }

    std::vector<Row<CountT>> rows_;
    std::vector<uint8_t> sparse_keys_;
    std::vector<CountT> sparse_counts_;
    std::vector<CountT> dense_counts_;
    std::vector<std::vector<uint32_t>> free_lists_;
    std::vector<uint32_t> dense_free_;
    std::vector<uint32_t> free_rows_;
};

}  // namespace alm
