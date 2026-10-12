// Ancestry-safe delta merge: the native equivalent of
// SequenceCortexOpponentLearner.consolidate_delta.
//
// Several parallel wake columns are cloned from one shared ancestor and each
// develop a disjoint slice of the stream. Merging their raw counts back
// together would double-count everything the ancestor already knew; this
// merges only what each descendant added past that shared ancestor.
//
// The context/sequence levels use `begin_lineage_tracking()`'s touched-key
// sets rather than diffing the full anatomy -- the same wake-time mutation
// journal the Python version uses, and the reason this merge stays cheap on
// a body with millions of learned paths when only thousands moved this
// round. The 256-row transition table is small enough to diff in full.
#pragma once

#include <cstdint>
#include <stdexcept>

#include "alm/cortex.hpp"

namespace alm {

struct CortexConsolidate {
    static void merge_delta(GrowableCortex& self, const GrowableCortex& descendant,
                            const GrowableCortex& ancestor) {
        require_same_configuration(self, descendant, ancestor);
        if (!descendant.tracking_ ||
            static_cast<int32_t>(descendant.touched_sequences_.size()) !=
                descendant.bank_count_) {
            throw std::runtime_error(
                "descendant has no lineage tracking to merge from -- call "
                "begin_lineage_tracking() before it develops, and make sure "
                "it survived any save/load round trip");
        }

        for (int i = 0; i < 256; ++i) {
            const int64_t delta =
                descendant.unigram_counts_[i] - ancestor.unigram_counts_[i];
            if (delta < 0) throw std::runtime_error("descendant anatomy predates its ancestor");
            self.unigram_counts_[i] += delta;
            self.unigram_sum_ += delta;
        }

        for (int i = 0; i < 256; ++i) {
            merge_row_delta(self.arena_, self.transition_rows_[i], descendant.arena_,
                            descendant.transition_rows_[i], ancestor.arena_,
                            ancestor.transition_rows_[i]);
            self.context_totals_[i] = self.arena_.total(self.transition_rows_[i]);
        }

        merge_confirmations(self.pair_confirmations_, descendant.pair_confirmations_,
                            ancestor.pair_confirmations_);
        or_live(self.pair_live_, descendant.pair_live_);
        merge_confirmations(self.quartet_confirmations_, descendant.quartet_confirmations_,
                            ancestor.quartet_confirmations_);
        or_live(self.quartet_live_, descendant.quartet_live_);
        merge_confirmations(self.assembly_confirmations_, descendant.assembly_confirmations_,
                            ancestor.assembly_confirmations_);
        int64_t assembly_births = 0;
        for (size_t i = 0; i < self.live_assemblies_.size(); ++i) {
            self.live_assemblies_[i] = self.live_assemblies_[i] | descendant.live_assemblies_[i];
            assembly_births += self.live_assemblies_[i] != 0;
        }
        self.assembly_births_ = assembly_births;

        merge_context_level(self, descendant, ancestor);
        for (int32_t bank = 0; bank < self.bank_count_; ++bank) {
            merge_sequence_bank(self, descendant, ancestor, bank);
        }
        if (self.config_.long_range_bank) {
            merge_long_range(self, descendant, ancestor);
        }

        const int64_t eviction_delta =
            descendant.sequence_evictions_ - ancestor.sequence_evictions_;
        if (eviction_delta < 0) throw std::runtime_error("descendant anatomy predates its ancestor");
        self.sequence_evictions_ += eviction_delta;
        self.sequence_births_ = self.live_sequence_count();
        self.context_births_ = self.live_context_count();

        const int64_t step_delta = descendant.steps_ - ancestor.steps_;
        if (step_delta < 0) throw std::runtime_error("descendant anatomy predates its ancestor");
        self.steps_ += step_delta;

        self.clear_history();
    }

private:
    static void require_same_configuration(const GrowableCortex& self,
                                           const GrowableCortex& descendant,
                                           const GrowableCortex& ancestor) {
        const CortexConfig& a = self.config_;
        const CortexConfig& b = descendant.config_;
        const CortexConfig& c = ancestor.config_;
        const bool ok = a.prior == b.prior && a.prior == c.prior &&
                        a.confirmations == b.confirmations &&
                        a.confirmations == c.confirmations &&
                        a.context_confirmations == b.context_confirmations &&
                        a.context_confirmations == c.context_confirmations &&
                        a.surprise_bits == b.surprise_bits &&
                        a.surprise_bits == c.surprise_bits &&
                        a.basal_strength == b.basal_strength &&
                        a.basal_strength == c.basal_strength &&
                        a.max_context_bytes == b.max_context_bytes &&
                        a.max_context_bytes == c.max_context_bytes &&
                        a.sequence_confirmations == b.sequence_confirmations &&
                        a.sequence_confirmations == c.sequence_confirmations &&
                        a.parent_strength == b.parent_strength &&
                        a.parent_strength == c.parent_strength &&
                        a.max_sequence_nursery_paths == b.max_sequence_nursery_paths &&
                        a.max_sequence_nursery_paths == c.max_sequence_nursery_paths;
        if (!ok) throw std::runtime_error("cannot consolidate incompatible cortical lineage");
    }

    template <typename CountT>
    static void merge_row_delta(RowArena<CountT>& self_arena, uint32_t self_slot,
                                const RowArena<CountT>& descendant_arena, uint32_t descendant_slot,
                                const RowArena<CountT>& ancestor_arena, uint32_t ancestor_slot) {
        descendant_arena.for_each(descendant_slot, [&](uint8_t key, int64_t count) {
            const int64_t ancestor_count =
                ancestor_slot == IndexMap::kEmpty ? 0 : ancestor_arena.get(ancestor_slot, key);
            const int64_t delta = count - ancestor_count;
            if (delta < 0) throw std::runtime_error("descendant anatomy predates its ancestor");
            if (delta > 0) self_arena.increment(self_slot, key, delta);
        });
    }

    // A nursery row's growth is only monotonic *between* evictions: once a
    // key is evicted by a column's own max_sequence_nursery_paths FIFO cap
    // during its wake period, a later observation of the same context
    // starts a brand-new row via grow_sequence_nursery (cortex.hpp),
    // counting from zero rather than continuing where the evicted row left
    // off. So a descendant's current row can legitimately show a lower
    // count for some byte than the ancestor's row had -- not lineage
    // corruption, just "this column forgot what the ancestor remembered
    // here, then possibly relearned a fraction of it." There is no
    // reliable delta to extract in that case, so the whole row is skipped
    // (mirrors opponent_byte_stream.py's _sparse_row_delta) rather than
    // throwing -- unlike merge_row_delta above, which stays strict for the
    // unigram transition table, where no nursery/eviction mechanism exists
    // and a decrease really would mean corruption.
    template <typename CountT>
    static void merge_row_delta_tolerant(RowArena<CountT>& self_arena, uint32_t self_slot,
                                         const RowArena<CountT>& descendant_arena,
                                         uint32_t descendant_slot,
                                         const RowArena<CountT>& ancestor_arena,
                                         uint32_t ancestor_slot) {
        bool forgotten = false;
        descendant_arena.for_each(descendant_slot, [&](uint8_t key, int64_t count) {
            const int64_t ancestor_count =
                ancestor_slot == IndexMap::kEmpty ? 0 : ancestor_arena.get(ancestor_slot, key);
            if (count < ancestor_count) forgotten = true;
        });
        if (forgotten) return;
        descendant_arena.for_each(descendant_slot, [&](uint8_t key, int64_t count) {
            const int64_t ancestor_count =
                ancestor_slot == IndexMap::kEmpty ? 0 : ancestor_arena.get(ancestor_slot, key);
            const int64_t delta = count - ancestor_count;
            if (delta > 0) self_arena.increment(self_slot, key, delta);
        });
    }

    static void merge_confirmations(std::vector<uint16_t>& self_values,
                                    const std::vector<uint16_t>& descendant_values,
                                    const std::vector<uint16_t>& ancestor_values) {
        for (size_t i = 0; i < self_values.size(); ++i) {
            const int32_t increment =
                static_cast<int32_t>(descendant_values[i]) - static_cast<int32_t>(ancestor_values[i]);
            if (increment < 0) throw std::runtime_error("descendant anatomy predates its ancestor");
            const int32_t total = static_cast<int32_t>(self_values[i]) + increment;
            self_values[i] = static_cast<uint16_t>(total > 65535 ? 65535 : total);
        }
    }

    static void or_live(std::vector<uint8_t>& self_values,
                        const std::vector<uint8_t>& descendant_values) {
        for (size_t i = 0; i < self_values.size(); ++i) self_values[i] |= descendant_values[i];
    }

    // Finds the row currently holding data for `key` in a cortex where a key
    // lives in exactly one of {nursery, mature} at a time.
    static uint32_t find_combined(const IndexMap& nursery, const IndexMap& mature,
                                  uint64_t key) {
        const uint32_t slot = nursery.find(key);
        return slot != IndexMap::kEmpty ? slot : mature.find(key);
    }

    static void merge_context_level(GrowableCortex& self, const GrowableCortex& descendant,
                                    const GrowableCortex& ancestor) {
        for (uint64_t key : descendant.touched_contexts_) {
            const uint32_t descendant_slot =
                find_combined(descendant.context_nursery_, descendant.residual_map_, key);
            if (descendant_slot == IndexMap::kEmpty) continue;
            const uint32_t ancestor_slot =
                find_combined(ancestor.context_nursery_, ancestor.residual_map_, key);

            uint32_t self_slot = self.residual_map_.find(key);
            const bool self_in_residual = self_slot != IndexMap::kEmpty;
            if (!self_in_residual) {
                self_slot = self.context_nursery_.find(key);
                if (self_slot == IndexMap::kEmpty) {
                    self_slot = self.arena_.create_row();
                    self.context_nursery_.insert(key, self_slot);
                }
            }

            merge_row_delta_tolerant(self.arena_, self_slot, descendant.arena_, descendant_slot,
                                     ancestor.arena_, ancestor_slot);

            const bool descendant_live = descendant.residual_map_.find(key) != IndexMap::kEmpty;
            const int64_t total = self.arena_.total(self_slot);
            if (!self_in_residual &&
                (descendant_live || total >= self.config_.context_confirmations)) {
                self.context_nursery_.erase(key);
                self.residual_map_.insert(key, self_slot);
            }
        }
    }

    // Bank-aware dispatch: the deepest ordinary sequence bank lives in a
    // separate int32 arena (cortex.hpp's arena_deep_bank_, measured safe
    // on real production data), every other bank stays on the wide int64
    // arena_. require_same_configuration already guarantees bank_count_
    // (and so is_deep_bank(bank)) agrees across self/descendant/ancestor,
    // so one check picks the right arena triple for all three.
    static void merge_row_delta_tolerant_for_bank(GrowableCortex& self, uint32_t self_slot,
                                                  const GrowableCortex& descendant,
                                                  uint32_t descendant_slot,
                                                  const GrowableCortex& ancestor,
                                                  uint32_t ancestor_slot, int32_t bank) {
        if (self.is_deep_bank(bank)) {
            merge_row_delta_tolerant(self.arena_deep_bank_, self_slot,
                                     descendant.arena_deep_bank_, descendant_slot,
                                     ancestor.arena_deep_bank_, ancestor_slot);
        } else {
            merge_row_delta_tolerant(self.arena_, self_slot, descendant.arena_, descendant_slot,
                                     ancestor.arena_, ancestor_slot);
        }
    }

    static void merge_sequence_bank(GrowableCortex& self, const GrowableCortex& descendant,
                                    const GrowableCortex& ancestor, int32_t bank) {
        for (uint64_t key : descendant.touched_sequences_[bank]) {
            const uint32_t descendant_slot = find_combined(
                descendant.sequence_nursery_[bank].map, descendant.sequence_maps_[bank], key);
            if (descendant_slot == IndexMap::kEmpty) continue;
            const uint32_t ancestor_slot = find_combined(
                ancestor.sequence_nursery_[bank].map, ancestor.sequence_maps_[bank], key);

            IndexMap& self_mature = self.sequence_maps_[bank];
            NurseryBank& self_nursery = self.sequence_nursery_[bank];
            uint32_t self_slot = self_mature.find(key);
            const bool self_in_mature = self_slot != IndexMap::kEmpty;
            if (!self_in_mature) {
                self_slot = self_nursery.map.find(key);
                if (self_slot == IndexMap::kEmpty) {
                    // Mirrors grow_sequence_nursery's FIFO cap enforcement
                    // (cortex.hpp): the per-byte observation path evicts
                    // the oldest nursery entry once at max_sequence_nursery_paths,
                    // and this merge path has to match it -- previously
                    // masked because the nursery was shed unconditionally
                    // on every sleep (Python's shed_sequence_nursery(),
                    // called from consolidate_to_byte_budget), so an
                    // uncapped nursery here never had a chance to be
                    // visible. Now that the nursery persists, a
                    // generation's delta rows across several columns can
                    // number in the tens of thousands of novel contexts.
                    if (static_cast<int64_t>(self_nursery.map.size()) >=
                            self.config_.max_sequence_nursery_paths &&
                        !self_nursery.empty_order()) {
                        const uint64_t victim = self_nursery.oldest();
                        const uint32_t victim_slot = self_nursery.map.find(victim);
                        if (victim_slot != IndexMap::kEmpty &&
                            self_nursery.map.erase(victim)) {
                            self.bank_release_row(bank, victim_slot);
                            self_nursery.drop_front();
                            ++self.sequence_evictions_;
                        }
                    }
                    self_slot = self.bank_create_row(bank);
                    self_nursery.map.insert(key, self_slot);
                    self_nursery.note_insertion(key);
                }
            }

            merge_row_delta_tolerant_for_bank(self, self_slot, descendant, descendant_slot,
                                              ancestor, ancestor_slot, bank);

            const bool descendant_live =
                descendant.sequence_maps_[bank].find(key) != IndexMap::kEmpty;
            const int64_t total = self.bank_total(bank, self_slot);
            if (!self_in_mature &&
                (descendant_live || total >= self.config_.sequence_confirmations)) {
                self_nursery.map.erase(key);
                self_mature.insert(key, self_slot);
            }
        }
    }

    // Without this, long_range_bank is silently inert in the real
    // multi-column harness: every wake round starts each column from a
    // fresh copy of the ancestor, so anything a column's long_range_map_
    // grew during its own wake period would otherwise be discarded the
    // instant the round's columns are merged back -- found and fixed
    // before this mechanism was ever run through the harness for real,
    // mirroring opponent_byte_stream.py's _merge_long_range_delta.
    //
    // Full diff over descendant.long_range_map_ rather than a
    // touched-key set (unlike merge_context_level/merge_sequence_bank
    // above): long_range_map_ is already bounded by long_range_max_entries
    // (an ordinary run's context/sequence anatomy is not), so an O(size)
    // diff here stays cheap without needing a second tracked-key set.
    //
    // Deliberately does not re-apply the burst gate here (unlike
    // grow_long_range, which does for ordinary per-byte growth): the
    // descendant already made that novelty call correctly, using its own
    // burst_fraction_, when it decided to create the key. self (the
    // master) never trains directly, only merges -- a master seeded from
    // a cold ancestor has burst_fraction_ == 0.0 forever, always below any
    // positive threshold, which silently discarded every descendant's
    // long-range growth on every single merge until this was caught (a
    // real production run where long_range_bank stayed inert through the
    // whole multi-column harness despite working in every single-process
    // test). The FIFO cap below still applies -- that bounds this table's
    // size, unlike the gate, which was answering "was this novel" with
    // the wrong learner's state.
    static void merge_long_range(GrowableCortex& self, const GrowableCortex& descendant,
                                  const GrowableCortex& ancestor) {
        for (const auto& [key, descendant_slot] : descendant.long_range_map_) {
            auto ancestor_it = ancestor.long_range_map_.find(key);
            const bool has_ancestor = ancestor_it != ancestor.long_range_map_.end();

            bool forgotten = false;
            descendant.arena_.for_each(descendant_slot, [&](uint8_t byte_value, int64_t count) {
                const int64_t ancestor_count =
                    has_ancestor ? ancestor.arena_.get(ancestor_it->second, byte_value) : 0;
                if (count < ancestor_count) forgotten = true;
            });
            if (forgotten) continue;

            auto self_it = self.long_range_map_.find(key);
            const bool is_new_key = self_it == self.long_range_map_.end();

            uint32_t self_slot;
            if (is_new_key) {
                if (static_cast<int64_t>(self.long_range_map_.size()) >=
                    self.config_.long_range_max_entries) {
                    while (!self.long_range_order_.empty()) {
                        const LongRangeKey& victim = self.long_range_order_.front();
                        auto victim_it = self.long_range_map_.find(victim);
                        if (victim_it != self.long_range_map_.end()) {
                            self.arena_.release_row(victim_it->second);
                            self.long_range_map_.erase(victim_it);
                            self.long_range_order_.pop_front();
                            break;
                        }
                        self.long_range_order_.pop_front();
                    }
                }
                self_slot = self.arena_.create_row();
                self.long_range_map_.emplace(key, self_slot);
                self.long_range_order_.push_back(key);
            } else {
                self_slot = self_it->second;
            }

            descendant.arena_.for_each(descendant_slot, [&](uint8_t byte_value, int64_t count) {
                const int64_t ancestor_count =
                    has_ancestor ? ancestor.arena_.get(ancestor_it->second, byte_value) : 0;
                const int64_t delta = count - ancestor_count;
                if (delta > 0) self.arena_.increment(self_slot, byte_value, delta);
            });
        }
    }
};

}  // namespace alm
