#include <cassert>
#include <cstdint>
#include <limits>

#include "alm/row_arena.hpp"

int main() {
    alm::RowArena<int16_t> arena;
    const uint32_t row = arena.create_row();

    // Force the row into dense storage, then saturate its narrow total.
    for (uint16_t key = 0; key <= alm::kDenseThreshold; ++key) {
        arena.increment(row, static_cast<uint8_t>(key));
    }
    assert(arena.size(row) == alm::kDenseThreshold + 1);
    arena.increment(row, 0, std::numeric_limits<int16_t>::max());
    assert(arena.total(row) == std::numeric_limits<int16_t>::max());

    // This used to create a phantom distinct key with a zero count. The
    // serializer wrote `size` but for_each skipped that zero-valued cell.
    const uint16_t size_before = arena.size(row);
    arena.increment(row, 255);
    assert(arena.size(row) == size_before);
    assert(arena.get(row, 255) == 0);

    uint16_t emitted = 0;
    arena.for_each(row, [&](uint8_t, int64_t count) {
        assert(count > 0);
        ++emitted;
    });
    assert(emitted == arena.size(row));
    return 0;
}
