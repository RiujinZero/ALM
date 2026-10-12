// Open-addressing maps sized for ALM's context/sequence anatomy.
//
// std::unordered_map is a node-per-entry structure: every lookup is a pointer
// chase into a separately allocated node, which is precisely the cost the
// Python dict version already pays. These maps keep keys and payload indices
// in flat arrays so a lookup touches one or two cache lines.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace alm {

inline uint64_t mix64(uint64_t value) noexcept {
    // splitmix64 finalizer: cheap, and strong enough that packed byte
    // contexts (which share long low-order runs) do not cluster.
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Maps a uint64 key to a uint32 payload index. Key 0 is representable: an
// explicit occupancy flag word is kept instead of reserving a sentinel key,
// because context 0 (a run of NUL bytes) is a legitimate packed context.
class IndexMap {
public:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

    IndexMap() { reset(1024); }

    void reset(size_t capacity_hint) {
        size_t capacity = 16;
        while (capacity * 3 < capacity_hint * 4) capacity <<= 1;
        keys_.assign(capacity, 0);
        values_.assign(capacity, kEmpty);
        size_ = 0;
        mask_ = capacity - 1;
    }

    size_t size() const noexcept { return size_; }
    size_t bucket_count() const noexcept { return keys_.size(); }

    // Returns kEmpty when absent.
    uint32_t find(uint64_t key) const noexcept {
        size_t slot = mix64(key) & mask_;
        while (true) {
            const uint32_t value = values_[slot];
            if (value == kEmpty) return kEmpty;
            if (keys_[slot] == key) return value;
            slot = (slot + 1) & mask_;
        }
    }

    void prefetch(uint64_t key) const noexcept {
        const size_t slot = mix64(key) & mask_;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&keys_[slot], 0, 3);
        __builtin_prefetch(&values_[slot], 0, 3);
#elif defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(&keys_[slot]), 1 /*_MM_HINT_T0*/);
        _mm_prefetch(reinterpret_cast<const char*>(&values_[slot]), 1);
#else
        (void)slot;
#endif
    }

    void insert(uint64_t key, uint32_t value) {
        if ((size_ + 1) * 4 >= keys_.size() * 3) grow();
        size_t slot = mix64(key) & mask_;
        while (values_[slot] != kEmpty) {
            if (keys_[slot] == key) {
                values_[slot] = value;
                return;
            }
            slot = (slot + 1) & mask_;
        }
        keys_[slot] = key;
        values_[slot] = value;
        ++size_;
    }

    // Backward-shift deletion keeps probe chains intact without tombstones.
    bool erase(uint64_t key) {
        size_t slot = mix64(key) & mask_;
        while (true) {
            const uint32_t value = values_[slot];
            if (value == kEmpty) return false;
            if (keys_[slot] == key) break;
            slot = (slot + 1) & mask_;
        }
        size_t hole = slot;
        size_t probe = (slot + 1) & mask_;
        while (values_[probe] != kEmpty) {
            const size_t ideal = mix64(keys_[probe]) & mask_;
            const size_t hole_distance = (hole - ideal) & mask_;
            const size_t probe_distance = (probe - ideal) & mask_;
            if (hole_distance < probe_distance) {
                keys_[hole] = keys_[probe];
                values_[hole] = values_[probe];
                hole = probe;
            }
            probe = (probe + 1) & mask_;
        }
        values_[hole] = kEmpty;
        keys_[hole] = 0;
        --size_;
        return true;
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (size_t slot = 0; slot < keys_.size(); ++slot) {
            if (values_[slot] != kEmpty) fn(keys_[slot], values_[slot]);
        }
    }

    size_t memory_bytes() const noexcept {
        return keys_.size() * sizeof(uint64_t) + values_.size() * sizeof(uint32_t);
    }

private:
    void grow() {
        std::vector<uint64_t> old_keys;
        std::vector<uint32_t> old_values;
        old_keys.swap(keys_);
        old_values.swap(values_);
        const size_t capacity = old_keys.size() * 2;
        keys_.assign(capacity, 0);
        values_.assign(capacity, kEmpty);
        mask_ = capacity - 1;
        size_ = 0;
        for (size_t slot = 0; slot < old_keys.size(); ++slot) {
            if (old_values[slot] != kEmpty) insert(old_keys[slot], old_values[slot]);
        }
    }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> values_;
    size_t size_ = 0;
    size_t mask_ = 0;
};

}  // namespace alm
