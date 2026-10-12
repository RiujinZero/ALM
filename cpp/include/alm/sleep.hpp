// Sleep: prune the weakest mature sequence paths until the body fits a
// byte budget. Native equivalent of
// SequenceCortexOpponentLearner.consolidate_to_byte_budget.
//
// Residual/context rows are never pruned here, matching Python exactly --
// only the sequence banks compete for the budget.
//
// The nursery is no longer shed here (matching a Python-side fix): it used
// to be wiped unconditionally on every sleep, which discarded provisional
// candidates before they had a real chance to accumulate
// sequence_confirmations worth of repeated observations -- merge_sequence_bank
// (consolidate.hpp) already merges nursery counts across columns correctly
// and promotes on the merged total, so a persistent nursery lets that
// mechanism actually do something. Once a candidate promotes, it pushes
// embodied_bytes over budget and the loop below evicts the weakest existing
// mature path to make room -- the same "new candidate competes for a slot"
// dynamic the vision organism's utility-ranked eviction has, without a
// separate scoring scheme.
//
// One deliberate departure from Python: "embodied bytes" here is the native
// checkpoint's own serialized size (CortexIO::save), not a Python pickle
// byte count. The two are not comparable measures, so a native body at a
// given byte cap holds more learned paths than a Python body at the same
// cap -- same budget, more anatomy, because there is no per-object Python
// overhead being counted. That is a real, worth-knowing consequence of
// changing backend, not a bug.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "alm/cortex.hpp"
#include "alm/serialize.hpp"

namespace alm {

struct SleepResult {
    int64_t body_bytes_before_sleep = 0;
    int64_t body_bytes_after_nursery_shed = 0;
    int64_t body_bytes_after_sleep = 0;
    int64_t shed_nursery_paths = 0;
    int64_t pruned_mature_paths = 0;
    int64_t competition_passes = 0;
};

struct CortexSleep {
    // (strongest, total, -bank_index, context) -- Python's exact ranking
    // key, lexicographic. heapq.nsmallest keeps the smallest N; the bounded
    // max-heap below keeps the same N by evicting its current largest
    // whenever a smaller candidate arrives, which is the standard
    // find-N-smallest-without-sorting-everything construction.
    struct Candidate {
        int64_t strongest;
        int64_t total;
        int32_t neg_bank;
        uint64_t context;
        int32_t bank_index;

        // "less" for this candidate as a prune priority (smaller = weaker
        // = prune first) is Python's tuple order.
        bool operator<(const Candidate& other) const noexcept {
            if (strongest != other.strongest) return strongest < other.strongest;
            if (total != other.total) return total < other.total;
            if (neg_bank != other.neg_bank) return neg_bank < other.neg_bank;
            return context < other.context;
        }
        // For a max-heap ranked by "how not-weak" (std::priority_queue is a
        // max-heap by operator<, so its top() is the strongest candidate
        // currently held -- the one to evict first when a weaker one shows
        // up needing room).
    };

    // (degree, strongest, total, -bank_index, context) -- the byte-cortex
    // analog of the vision organism's relation-degree utility
    // (degree*(capacity+1) + observations*confidence in
    // continual_vision.py's sleep()). A well-connected vision assembly is
    // one many others co-activate with, so removing it disrupts many
    // relationships at once; the direct equivalent here is a context many
    // longer, deeper mature contexts share as their trailing backoff
    // suffix -- a hub in the backoff chain. Degree is the primary ranking
    // key (ascending, least-connected pruned first); strongest/total/
    // bank/context are only tiebreakers among equally-connected contexts,
    // matching how vision's formula uses observations*confidence only to
    // break ties within the same degree.
    struct ConnectivityCandidate {
        int32_t degree;
        int64_t strongest;
        int64_t total;
        int32_t neg_bank;
        uint64_t context;
        int32_t bank_index;

        bool operator<(const ConnectivityCandidate& other) const noexcept {
            if (degree != other.degree) return degree < other.degree;
            if (strongest != other.strongest) return strongest < other.strongest;
            if (total != other.total) return total < other.total;
            if (neg_bank != other.neg_bank) return neg_bank < other.neg_bank;
            return context < other.context;
        }
    };

    static int64_t estimated_bytes(const GrowableCortex& cortex) {
        return static_cast<int64_t>(CortexIO::save(cortex).size());
    }

    static SleepResult consolidate_to_byte_budget(GrowableCortex& cortex,
                                                   int64_t max_bytes) {
        if (max_bytes < 1) throw std::runtime_error("max_bytes must be positive");
        SleepResult result;
        result.body_bytes_before_sleep = estimated_bytes(cortex);
        cortex.clear_history();
        // Affective Intensity's replay half: reprocess whatever the
        // burst-bypass queued during wake, before this sleep's own
        // consolidation/pruning runs (was previously exposed via the C
        // API but never actually called from any real sleep path -- a
        // real gap, not by design). Cheap when the queue is empty.
        if (cortex.config_.long_range_use_affective_intensity) {
            cortex.replay_reinforce();
        }
        // Refresh combined_salience's degree cache once per sleep instead
        // of leaving it stale from construction -- see
        // refresh_degree_cache's own comment for why an on-demand
        // per-byte version was ruinously slow on real text.
        if (cortex.config_.long_range_use_combined_salience) {
            cortex.refresh_degree_cache();
        }
        result.shed_nursery_paths = 0;
        result.body_bytes_after_nursery_shed = estimated_bytes(cortex);
        int64_t current = result.body_bytes_after_nursery_shed;

        while (current > max_bytes) {
            const int64_t live = cortex.live_sequence_count();
            if (live < 1) {
                // Ordinary sequence banks are exhausted but still over
                // budget -- deep bands (previously had no budget-aware
                // eviction at all, only a fixed per-band entry-count cap
                // that measurement showed can still correspond to many
                // times max_bytes on real diverse text) absorb further
                // pressure here. Batched the same way as the ordinary-bank
                // loop below: estimated_bytes() re-serializes the whole
                // checkpoint, so recomputing it after every single evicted
                // entry would itself be the kind of per-item-O(whole-body)
                // trap this codebase has already been burned by once
                // (context_degree's original on-demand scan).
                const int64_t deep_live = cortex.deep_band_total_entries();
                if (deep_live < 1) {
                    throw std::runtime_error(
                        "fixed cortical anatomy exceeds the requested byte budget");
                }
                const double mean_entry_bytes = std::max(
                    8.0, static_cast<double>(current) / static_cast<double>(deep_live));
                const int64_t needed = std::max(
                    int64_t{1},
                    std::min(deep_live, static_cast<int64_t>(std::ceil(
                                            (current - max_bytes) / mean_entry_bytes * 1.05))));
                int64_t evicted = 0;
                for (int64_t i = 0; i < needed; ++i) {
                    if (!cortex.evict_one_deep_band_entry()) break;
                    ++evicted;
                }
                current = estimated_bytes(cortex);
                if (evicted == 0) {
                    throw std::runtime_error(
                        "fixed cortical anatomy exceeds the requested byte budget");
                }
                continue;  // outer while re-checks current > max_bytes
            }
            const double mean_path_bytes =
                std::max(8.0, static_cast<double>(current) / static_cast<double>(live));
            const int64_t needed = std::max(
                int64_t{1},
                std::min(live, static_cast<int64_t>(std::ceil(
                                    (current - max_bytes) / mean_path_bytes * 1.05))));

            if (cortex.config_.connectivity_pruning) {
                const std::vector<ConnectivityCandidate> weakest =
                    connectivity_weakest_candidates(cortex, needed);
                if (weakest.empty()) {
                    throw std::runtime_error(
                        "no temporal anatomy remains available to prune");
                }
                for (const ConnectivityCandidate& candidate : weakest) {
                    const uint32_t slot =
                        cortex.sequence_maps_[candidate.bank_index].find(candidate.context);
                    if (slot != IndexMap::kEmpty) {
                        cortex.bank_release_row(candidate.bank_index, slot);
                        cortex.sequence_maps_[candidate.bank_index].erase(candidate.context);
                    }
                }
                result.pruned_mature_paths += static_cast<int64_t>(weakest.size());
                ++result.competition_passes;
                current = estimated_bytes(cortex);
                continue;
            }

            const std::vector<Candidate> weakest = weakest_candidates(cortex, needed);
            if (weakest.empty()) {
                throw std::runtime_error("no temporal anatomy remains available to prune");
            }
            for (const Candidate& candidate : weakest) {
                const uint32_t slot =
                    cortex.sequence_maps_[candidate.bank_index].find(candidate.context);
                if (slot != IndexMap::kEmpty) {
                    cortex.bank_release_row(candidate.bank_index, slot);
                    cortex.sequence_maps_[candidate.bank_index].erase(candidate.context);
                }
            }
            result.pruned_mature_paths += static_cast<int64_t>(weakest.size());
            ++result.competition_passes;
            current = estimated_bytes(cortex);
        }
        cortex.sequence_births_ = cortex.live_sequence_count();
        result.body_bytes_after_sleep = current;
        return result;
    }

private:
    static std::vector<Candidate> weakest_candidates(const GrowableCortex& cortex,
                                                      int64_t needed) {
        std::priority_queue<Candidate> heap;  // max-heap: top() is strongest held
        for (int32_t bank = 0; bank < cortex.bank_count_; ++bank) {
            cortex.sequence_maps_[bank].for_each([&](uint64_t context, uint32_t slot) {
                int64_t strongest = 0;
                cortex.bank_for_each(bank, slot, [&](uint8_t, int64_t count) {
                    if (count > strongest) strongest = count;
                });
                const Candidate candidate{strongest, cortex.bank_total(bank, slot),
                                          -bank, context, bank};
                if (static_cast<int64_t>(heap.size()) < needed) {
                    heap.push(candidate);
                } else if (candidate < heap.top()) {
                    heap.pop();
                    heap.push(candidate);
                }
            });
        }
        std::vector<Candidate> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            result.push_back(heap.top());
            heap.pop();
        }
        return result;
    }

    static std::vector<ConnectivityCandidate> connectivity_weakest_candidates(
        const GrowableCortex& cortex, int64_t needed) {
        // degree[bank][context] = how many mature contexts in bank+1 have
        // `context` as their trailing (bank_depth)-byte suffix. Native
        // always uses packed integer contexts (no tuple-path oracle mode),
        // so this is always the mask-and-compare form of Python's suffix
        // check.
        std::vector<std::unordered_map<uint64_t, int32_t>> degree(
            static_cast<size_t>(cortex.bank_count_));
        for (int32_t bank = 0; bank + 1 < cortex.bank_count_; ++bank) {
            const int32_t depth = bank + 3;
            const uint64_t mask = (depth >= 8) ? ~0ULL : ((1ULL << (8 * depth)) - 1ULL);
            std::unordered_map<uint64_t, int32_t>& parent_degree = degree[bank];
            cortex.sequence_maps_[bank + 1].for_each(
                [&](uint64_t child_context, uint32_t) {
                    const uint64_t suffix = child_context & mask;
                    ++parent_degree[suffix];
                });
        }

        std::priority_queue<ConnectivityCandidate> heap;
        for (int32_t bank = 0; bank < cortex.bank_count_; ++bank) {
            const std::unordered_map<uint64_t, int32_t>& parent_degree = degree[bank];
            cortex.sequence_maps_[bank].for_each([&](uint64_t context, uint32_t slot) {
                int64_t strongest = 0;
                cortex.bank_for_each(bank, slot, [&](uint8_t, int64_t count) {
                    if (count > strongest) strongest = count;
                });
                const auto found = parent_degree.find(context);
                const int32_t context_degree =
                    found == parent_degree.end() ? 0 : found->second;
                const ConnectivityCandidate candidate{
                    context_degree, strongest, cortex.bank_total(bank, slot),
                    -bank, context, bank};
                if (static_cast<int64_t>(heap.size()) < needed) {
                    heap.push(candidate);
                } else if (candidate < heap.top()) {
                    heap.pop();
                    heap.push(candidate);
                }
            });
        }
        std::vector<ConnectivityCandidate> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            result.push_back(heap.top());
            heap.pop();
        }
        return result;
    }
};

}  // namespace alm
