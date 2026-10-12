// ALM sequence cortex: growable (wake) and packed (mature) implementations.
//
// Both reproduce SequenceCortexOpponentLearner.observe_chunk_probabilities
// bit-for-bit. The nested backoff is evaluated in exactly the Python
// operation order so IEEE-754 doubles land on identical bits; nothing here
// may be compiled with fast-math or FMA contraction.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "alm/adaptive_trust.hpp"
#include "alm/hash_map.hpp"
#include "alm/row_arena.hpp"

namespace alm {

struct CortexConfig {
    double prior = 0.1;
    int32_t confirmations = 3;
    int32_t context_confirmations = 3;
    double surprise_bits = 2.0;
    double basal_strength = 4.0;
    int32_t max_context_bytes = 6;
    int32_t sequence_confirmations = 3;
    double parent_strength = 16.0;
    int64_t max_sequence_nursery_paths = 50000;

    // Contextual-competition / consequence-sensitive-learning extension
    // (opponent_byte_stream.py's competition_exponent/adaptive_competition).
    // competition_exponent == 1.0 with no learned trust reproduces the
    // original fixed backoff exactly -- these five fields are additive.
    double competition_exponent = 1.0;
    bool adaptive_competition = false;
    double trust_gamma = 0.5;
    double trust_learning_rate = 0.2;
    double attribution_gate_threshold = 0.5;
    // Formerly hardcoded inside TrustState::credit -- a real run showed
    // the default (2.0) saturating hard (49.8% of learned trust values
    // sitting at the exact ceiling). Tightening trust_delta_clip is the
    // direct lever to test whether trust can settle on graded,
    // intermediate values instead of swinging to an extreme after a
    // handful of updates. Defaults match the original hardcoded
    // constants exactly, so existing configs are unaffected.
    double trust_delta_clip = 2.0;
    double trust_floor = 0.1;
    double trust_ceiling = 10.0;

    // Sleep-time-only pruning-priority alternative (opponent_byte_stream.py's
    // connectivity_pruning): does not change growth, promotion, or the
    // probability math at all, only which mature rows a byte-budget
    // overflow evicts first. See sleep.hpp for what it ranks by.
    bool connectivity_pruning = false;

    // Population-level novelty signal (opponent_byte_stream.py's
    // burst_surprise_signal): an EMA of "did anything just get born"
    // across context_births/sequence_births. Feeds two independent uses,
    // both opt-in: damping adaptive_competition's per-context trust gain
    // while the population is actively bursting, and gating long_range_bank
    // growth below (see docs/HTM_Inspired_Followup_Experiments.md).
    bool burst_surprise_signal = false;
    double burst_ema_rate = 0.001;

    // HTM-inspired long-range context bank (opponent_byte_stream.py's
    // long_range_bank): a second context bank keyed on far more trailing
    // bytes than the ordinary sequence banks (which top out at
    // max_context_bytes), blended in exactly like they are. Validated to
    // carry identity-specific information across a gap the ordinary banks
    // cannot see across at all (53% -> 95% correct on a controlled probe),
    // reusing the same context-matching mechanism rather than a separate
    // memory organ.
    bool long_range_bank = false;
    int32_t long_range_context_bytes = 24;
    double long_range_strength = 8.0;
    // Gates only the *creation* of a new long-range key on burst_fraction,
    // not reinforcement of an existing one -- gating both cost real recall
    // confidence (95% -> 63% on the same controlled probe) because an EMA
    // novelty signal cannot tell "boring because already established"
    // apart from "boring because quiet". Requires burst_surprise_signal.
    bool long_range_gate_on_burst = false;
    double long_range_gate_threshold = 0.1;
    // FIFO cap: nothing prunes long_range_map_ (unlike the ordinary
    // sequence banks, which consolidate_to_byte_budget can evict from),
    // so without a cap it grows without bound over a long run.
    int64_t long_range_max_entries = 200000;

    // Idea 2 continued: sparse multi-depth long-range banks, additive and
    // independent of the single-depth long_range_bank above (both can be
    // enabled at once for comparison; neither changes the other's
    // behavior). Where long_range_bank creates a row the instant a new key
    // is seen (gated only by burst_fraction), these banks require genuine
    // repeated observation before promotion -- a nursery stage, same
    // "arena_.total(slot) >= confirmations" contract grow_sequence_nursery
    // already uses for the ordinary banks -- so long-range recall is
    // reached by the same kind of evidence accumulation as short-range
    // recall, not a separate one-shot-create rule.
    bool long_range_bands = false;
    static constexpr int32_t kMaxLongRangeBands = 4;
    // Sparse, not dense: adjacent depths would multiply per-byte hash-map
    // work for little extra reach. Tunable; only the first
    // long_range_band_count entries are active.
    int32_t long_range_band_depths[4] = {12, 16, 24, 32};
    int32_t long_range_band_count = 4;
    // Backend under comparison: false = exact fixed-width key (reuses
    // LongRangeKey, one map per band, no collision risk, capped at 32
    // bytes by LongRangeKey::kMaxBytes); true = O(1)-per-byte rolling hash
    // into a uint64_t key, reusing the same fast IndexMap/NurseryBank the
    // ordinary depth-3..8 banks already use, at the cost of possible hash
    // collisions between two different long contexts.
    bool long_range_rolling_hash = false;
    // Shared across all bands for this comparison; promotion happens once
    // a band's nursery row accumulates this many total observations.
    int32_t long_range_band_confirmations = 3;
    // Importance-bonus shortcut: an observation whose pre-band probability
    // (everything the shallower banks already predicted) was surprising
    // enough -- -log2(probability) >= long_range_importance_bits -- counts
    // as long_range_importance_bonus extra confirmations on top of its own
    // +1, so a sufficiently surprising first sighting can promote
    // immediately instead of waiting for ordinary repetition. Never
    // applies to reinforcing an already-mature row (matching the
    // gate-creation-not-reinforcement lesson from long_range_gate_on_burst
    // below), only to nursery promotion.
    double long_range_importance_bits = 4.0;
    int32_t long_range_importance_bonus = 3;
    // Per-band FIFO cap on the mature map (same eviction shape as
    // long_range_max_entries above, applied independently per band).
    int64_t long_range_band_max_entries = 200000;

    // Salience-signal prototype: gate deep-band nursery promotion on
    // span_surprise * log2(2+degree) instead of single-byte surprise
    // alone. Validated (controlled probe, 7 hub vs 7 isolated suffix
    // pairs, real wiki text): raw single-byte surprise does not reliably
    // separate "surprising because this attaches to something the
    // organism already treats as important" from "surprising because
    // nothing here has any structure at all" -- the two can go either
    // direction. Summing surprise over a short span reduces per-byte
    // noise; multiplying by log-scaled hub degree (connectivity_pruning's
    // existing "how many things route through this context" measure,
    // queried on demand -- see context_degree) preferentially keeps facts
    // that attach to already-important structure. log, not linear,
    // scaling on degree: linear let one very-high-degree example dominate
    // the whole signal in testing. Degree is only queried on a nursery
    // key's first sighting (bounded cost), never on reinforcement.
    bool long_range_use_combined_salience = false;
    int32_t long_range_salience_span_bytes = 6;
    int32_t long_range_salience_degree_bank = 0;

    // Affective Intensity (AI): a second, independent path to promotion,
    // alongside (not instead of) surprise x degree -- catches events that
    // matter because a huge amount of processing was mobilized RIGHT NOW
    // (a population-wide burst: burst_fraction_, already built for a
    // different purpose), not because they're novel or hub-adjacent. A
    // familiar, unsurprising, low-hub-degree event (a name you've heard a
    // thousand times) can still trigger AI if it coincides with a burst.
    // AI = burst_fraction_ * context_frequency (a cheap O(1) re-encounter
    // estimate, NOT context_degree's O(|bank+1|) scan) -- but re-encounter
    // probability is a MULTIPLIER, never a hard requirement: a bear chase
    // has near-zero re-encounter probability yet must still be
    // unforgettable, so sufficiently extreme burst alone bypasses the
    // probability term entirely.
    bool long_range_use_affective_intensity = false;
    double long_range_ai_threshold = 2.0;
    double long_range_ai_bypass_burst = 0.5;

    // Replay: an AI-triggered event enters a small bounded FIFO queue and
    // gets RE-reinforced (as if observed again) once per subsequent sleep
    // call, riding the existing wake/sleep split rather than a new
    // consolidation mechanism -- "thinking about it more" after the fact
    // compounds the same way real rehearsal does. Entries age out purely
    // by FIFO capacity (a first pass, not priority-ranked eviction).
    int32_t long_range_replay_queue_capacity = 64;

    // Tiered persistence (systems consolidation): a deep-band entry that
    // reaches ordinary mature status (tier 2, the existing FIFO-evicted
    // map) can graduate further. Tier 3 ("consolidated") is reached only
    // via a replay hit -- reinforce_deep_band_key only ever fires for
    // AI-triggered keys, so arriving here already implies the entry once
    // mattered enough to bypass -- and is evicted weakest-first (lowest
    // accumulated count) instead of oldest-first, from its own smaller
    // pool. Tier 4 ("permanent") is reached from tier 3 by ordinary
    // continued recurrence crossing a high count threshold, and is not
    // evicted by sleep pressure at all (only an emergency weakest-first
    // demotion back to tier 3 if its own tiny cap is ever exceeded).
    // Deliberately small caps: this tier is meant to hold the rare few
    // things that both mattered intensely once (replay) and kept
    // recurring since, not to become a second copy of the whole band.
    int64_t long_range_consolidated_max_entries = 8000;
    int64_t long_range_permanent_max_entries = 800;
    int64_t long_range_permanent_recurrence_threshold = 100;
    // Privacy gate on tier 4 specifically: permanent entries are never
    // evicted by ordinary sleep pressure and persist for the organism's
    // entire lifetime, unlike every other tier (all eventually forgettable
    // under budget pressure). Something a specific user states enough
    // times could otherwise get baked into permanent, effectively
    // unforgettable memory -- defaults to false so an operator has to
    // explicitly opt in to that retention rather than getting it for
    // free. Consolidated (tier 3) and below are unaffected; only the
    // tier 3 -> tier 4 promotion itself is gated.
    bool long_range_allow_permanent_tier = false;

    // Sparse episodic layer (docs/ALM_V1_Architecture.md, "A proposed
    // sparse episodic layer" / "rolling window" design): content-addressed
    // recall over arbitrary distance, distinct in kind from the deep bands
    // above -- those are dense, exact-match, globally-durable (tier 4
    // requires 100+ recurrences across the WHOLE corpus); this is sparse
    // (syncmer-selected, ~1-in-20 positions), per-episode, and decays
    // unless re-referenced. Prototyped and validated standalone
    // (episodic_bench*.cpp, not checked in) before this integration:
    // closed-syncmer selection is context-independent (same content always
    // selects the same anchor regardless of surrounding text, unlike a
    // large-window minimizer); recall is 100% and distance-independent
    // from 1KB to 200M bytes once an anchor exists; a write-once entry
    // survives real bucket contention only ~50-60% of the time, while one
    // refreshed periodically survives ~100%; crossing
    // episodic_tier3_refresh_threshold refreshes gives a sharp step to
    // nearly-guaranteed survival (a real phase transition, not a gradual
    // improvement) by exempting the entry from ordinary LRU eviction.
    bool episodic_enabled = false;
    int32_t episodic_k = 14;              // k-mer width: both the anchor-selection input and the storage key
    int32_t episodic_s = 6;                // s-mer sub-window width for closed-syncmer selection
    int32_t episodic_bucket_bits = 20;     // ~1M buckets by default
    int32_t episodic_chain_slots = 4;      // entries per bucket before eviction pressure begins
    int32_t episodic_tier3_refresh_threshold = 3;  // refreshes to become eviction-immune (mirrors long_range_band_confirmations)
    double episodic_strength = 4.0;        // blend weight into predict_distribution, same role as long_range_strength
};

// r = d_i / sum(d_j) at kappa=0, the exact-partition case _backoff_weights
// always uses. Returns nullopt exactly when Python's _backoff_weights would
// return None (exponent==1 and child_gain==1), so every caller falls back
// to the original fused formula bit-for-bit at those (overwhelmingly
// common) defaults.
inline std::optional<std::pair<double, double>> competition_weights(
    double parent_evidence, double child_evidence, double child_gain, double exponent) {
    if (exponent == 1.0 && child_gain == 1.0) return std::nullopt;
    const double d0 = std::pow(parent_evidence, exponent);
    const double d1 = std::pow(child_evidence * child_gain, exponent);
    const double denominator = d0 + d1;
    if (denominator <= 0.0) return std::make_pair(0.0, 0.0);
    return std::make_pair(d0 / denominator, d1 / denominator);
}

// One nursery bank: a key -> row map plus the insertion order Python dicts
// give for free, because eviction takes the oldest surviving key and the
// promotion decisions that follow depend on exactly which key that is.
struct NurseryBank {
    IndexMap map;
    std::deque<uint64_t> order;

    void note_insertion(uint64_t key) {
        order.push_back(key);
        // Promoted keys leave the map but not the order queue; compact once
        // the stale share dominates so the queue tracks the map's size.
        if (order.size() > 4 * map.size() + 1024) compact();
    }

    // Mirrors `del bank[next(iter(bank))]`.
    uint64_t oldest() {
        while (!order.empty()) {
            const uint64_t key = order.front();
            if (map.find(key) != IndexMap::kEmpty) return key;
            order.pop_front();
        }
        return 0;
    }

    bool empty_order() const noexcept { return order.empty(); }

    void drop_front() {
        if (!order.empty()) order.pop_front();
    }

    void clear() {
        map.reset(1024);
        order.clear();
    }

private:
    void compact() {
        std::deque<uint64_t> live;
        for (uint64_t key : order) {
            if (map.find(key) != IndexMap::kEmpty) live.push_back(key);
        }
        order.swap(live);
    }
};

// A newborn deep-band context typically holds only a few distinct
// successor bytes before either promoting or being evicted (measured on
// real diverse text: ~97% of nursery touches create a brand-new row, and
// the common case for a row that survives is only 1-4 distinct
// successors before confirmations is reached). An arena32_ allocation
// plus a second dependent dereference for that is real, avoidable
// overhead -- the row lives entirely inline until it would need a 5th
// distinct successor, at which point it "overflows" into a real arena32_
// row (same counters, same confirmations/promotion contract -- a storage
// format transition, not a different maturity or learning rule).
struct InlineNurseryRow {
    static constexpr int32_t kCapacity = 4;
    uint8_t keys[kCapacity] = {};
    int16_t counts[kCapacity] = {};
    uint8_t occupied = 0;
    uint32_t overflow_slot = IndexMap::kEmpty;

    bool overflowed() const noexcept { return overflow_slot != IndexMap::kEmpty; }
};

// Owns InlineNurseryRow storage with the same create/release free-list
// shape RowArena uses for its own rows, so slots are reused rather than
// left to accumulate as nursery entries promote or get evicted.
class InlineNurseryArena {
public:
    uint32_t create_row() {
        if (!free_.empty()) {
            const uint32_t slot = free_.back();
            free_.pop_back();
            rows_[slot] = InlineNurseryRow{};
            return slot;
        }
        rows_.emplace_back();
        return static_cast<uint32_t>(rows_.size() - 1);
    }
    void release_row(uint32_t slot) { free_.push_back(slot); }
    InlineNurseryRow& row(uint32_t slot) { return rows_[slot]; }
    const InlineNurseryRow& row(uint32_t slot) const { return rows_[slot]; }

private:
    std::vector<InlineNurseryRow> rows_;
    std::vector<uint32_t> free_;
};

// Same map+order-FIFO shape as NurseryBank (identical interface, reused
// deliberately to keep call sites structurally unchanged), except the
// map's value indexes into `arena` (InlineNurseryRow storage) instead of
// a shared arena32_ slot.
struct InlineNurseryBank {
    IndexMap map;
    std::deque<uint64_t> order;
    InlineNurseryArena arena;

    void note_insertion(uint64_t key) {
        order.push_back(key);
        if (order.size() > 4 * map.size() + 1024) compact();
    }
    uint64_t oldest() {
        while (!order.empty()) {
            const uint64_t key = order.front();
            if (map.find(key) != IndexMap::kEmpty) return key;
            order.pop_front();
        }
        return 0;
    }
    bool empty_order() const noexcept { return order.empty(); }
    void drop_front() {
        if (!order.empty()) order.pop_front();
    }

private:
    void compact() {
        std::deque<uint64_t> live;
        for (uint64_t key : order) {
            if (map.find(key) != IndexMap::kEmpty) live.push_back(key);
        }
        order.swap(live);
    }
};

// Fixed-capacity, non-allocating key for the long-range bank. A std::string
// key (the first cut) meant a heap allocation on every single processed
// byte once long_range_bank is on -- most keys exceed small-string
// optimization's inline buffer at typical long_range_context_bytes values
// (24 by default) -- and profiling confirmed this dominates the cost:
// long_range_bank alone measured 2.75x slower than baseline even before
// any scoring/read step is considered, since growth needs the same key
// construction on every byte regardless of how often the blended score
// is actually read. This type reproduces the exact same equality Python's
// tuple keys give (no collision risk, unlike a rolling hash) with zero
// per-byte allocation, at the cost of a fixed maximum key length.
struct LongRangeKey {
    static constexpr size_t kMaxBytes = 32;
    uint8_t bytes[kMaxBytes] = {};
    uint8_t length = 0;

    bool operator==(const LongRangeKey& other) const noexcept {
        return length == other.length &&
               std::memcmp(bytes, other.bytes, length) == 0;
    }
};

struct LongRangeKeyHash {
    size_t operator()(const LongRangeKey& key) const noexcept {
        // FNV-1a over the live bytes only; cheap and adequate since real
        // collisions are handled by operator== regardless (this only
        // needs to distribute keys across buckets, not guarantee
        // uniqueness).
        uint64_t hash = 1469598103934665603ULL;
        for (uint8_t i = 0; i < key.length; ++i) {
            hash ^= key.bytes[i];
            hash *= 1099511628211ULL;
        }
        return static_cast<size_t>(hash);
    }
};

// A deep-band key queued for sleep-time replay because Affective Intensity
// (burst-driven salience) fired at encoding time. Holds both backends'
// key representations since which one is meaningful depends on
// long_range_rolling_hash at the time it was queued.
struct ReplayCandidate {
    int32_t band;
    LongRangeKey exact_key;
    uint64_t hash_key;
    uint8_t target;
};

class GrowableCortex {
public:
    // Serialization, consolidate_delta, and the Python-anatomy importer all
    // need to poke at raw state that has no business being public API, so
    // they reach in as friends instead of forcing a getter/setter for every
    // field of a structure they are allowed to know the layout of.
    friend struct CortexIO;
    friend struct CortexConsolidate;
    friend struct CortexSleep;

    explicit GrowableCortex(const CortexConfig& config)
        : config_(config), trust_(config.trust_learning_rate) {
        if (config_.long_range_bank &&
            (config_.long_range_context_bytes < 1 ||
             static_cast<size_t>(config_.long_range_context_bytes) >
                 LongRangeKey::kMaxBytes)) {
            throw std::runtime_error(
                "long_range_context_bytes must be in [1, 32] (fixed-capacity, "
                "non-allocating key)");
        }
        const int32_t depth = config_.max_context_bytes;
        bank_count_ = depth > 2 ? depth - 2 : 0;
        history_mask_ = depth >= 8 ? ~0ULL : ((1ULL << (8 * depth)) - 1ULL);
        context_masks_.resize(bank_count_);
        for (int32_t bank = 0; bank < bank_count_; ++bank) {
            const int32_t bank_depth = bank + 3;
            context_masks_[bank] =
                bank_depth >= 8 ? ~0ULL : ((1ULL << (8 * bank_depth)) - 1ULL);
        }
        sequence_maps_.resize(bank_count_);
        sequence_nursery_.resize(bank_count_);
        unigram_counts_.assign(256, 0);
        context_totals_.assign(256, 0);
        live_assemblies_.assign(256, 0);
        assembly_confirmations_.assign(256, 0);
        pair_confirmations_.assign(16, 0);
        pair_live_.assign(16, 0);
        quartet_confirmations_.assign(32, 0);
        quartet_live_.assign(32, 0);
        transition_rows_.resize(256);
        for (int i = 0; i < 256; ++i) transition_rows_[i] = arena_.create_row();

        if (config_.long_range_bands) {
            const int32_t band_count = config_.long_range_band_count;
            if (band_count < 0 || band_count > CortexConfig::kMaxLongRangeBands) {
                throw std::runtime_error(
                    "long_range_band_count must be in [0, kMaxLongRangeBands]");
            }
            for (int32_t b = 0; b < band_count; ++b) {
                const int32_t d = config_.long_range_band_depths[b];
                if (d < 1 || (!config_.long_range_rolling_hash &&
                              static_cast<size_t>(d) > LongRangeKey::kMaxBytes)) {
                    throw std::runtime_error(
                        "long_range_band_depths entries must be in [1, 32] for the "
                        "exact backend (fixed-capacity key)");
                }
            }
            deep_mature_hash_.resize(band_count);
            deep_nursery_hash_.resize(band_count);
            deep_mature_order_hash_.resize(band_count);
            deep_mature_exact_.resize(band_count);
            deep_nursery_exact_.resize(band_count);
            deep_mature_order_exact_.resize(band_count);
            deep_nursery_order_exact_.resize(band_count);
            deep_tier_hash_.resize(band_count);
            deep_tier_exact_.resize(band_count);
            deep_consolidated_members_hash_.resize(band_count);
            deep_consolidated_members_exact_.resize(band_count);
            deep_permanent_members_hash_.resize(band_count);
            deep_permanent_members_exact_.resize(band_count);
            for (int32_t b = 0; b < band_count; ++b) {
                deep_band_hash_[b] = 0;
                uint64_t pow = 1;
                for (int32_t i = 1; i < config_.long_range_band_depths[b]; ++i) pow *= kDeepBandHashBase;
                deep_band_pow_[b] = pow;
            }
        }

        if (config_.episodic_enabled) init_episodic_state();

        // Anchor-gate rolling-hash powers at the default k=6/s=3 width;
        // recomputed by set_debug_deep_band_anchor_gate if a different
        // width is requested. Computed unconditionally (cheap) so the gate
        // can be flipped on at any point without a separate init path.
        recompute_anchor_gate_pow();
    }

    const CortexConfig& config() const noexcept { return config_; }

    // Turns the episodic layer on for an ALREADY-CONSTRUCTED (including
    // deserialized/resumed) cortex that was built with episodic_enabled
    // false -- the constructor only allocates episodic_buckets_ etc. if
    // the flag was already set at construction time, and CortexIO
    // deserializes whatever config_.episodic_enabled the checkpoint was
    // originally saved with (false, for any checkpoint from before this
    // layer existed, or from a run that hadn't opted in). This is what
    // lets an in-progress, already-mature run retrofit episodic memory
    // without restarting from scratch: config_.episodic_k/s/bucket_bits/
    // etc. already hold sensible defaults regardless of whether the flag
    // was ever on (read_config only overwrites fields write_config
    // actually serialized; anything else keeps CortexConfig's in-class
    // defaults), so this just needs to flip the flag and run the same
    // init the constructor would have. No-op if already enabled.
    void enable_episodic_layer() {
        if (config_.episodic_enabled) return;
        config_.episodic_enabled = true;
        init_episodic_state();
    }

    // Reconfigures the episodic layer's k/s anchor width, ALWAYS
    // re-applying it (unlike enable_episodic_layer, a one-time opt-in
    // that's a no-op once already enabled) -- because episodic_k/s/
    // enabled are not part of serialize.hpp (a checkpoint round-trip
    // silently reverts them to CortexConfig's compiled defaults), and a
    // multi-epoch training loop that resumes from a checkpoint every
    // epoch needs to reapply its intended width every time, not just on
    // the very first construction. Safe mid-run: episodic content is
    // already wiped on every reset_episode() call (once per wake round in
    // the phase156 training loop), so reinitializing here loses nothing
    // that would have survived anyway.
    void configure_episodic_layer(bool enabled, int32_t k, int32_t s) {
        config_.episodic_enabled = false;
        config_.episodic_k = k;
        config_.episodic_s = s;
        if (enabled) {
            config_.episodic_enabled = true;
            init_episodic_state();
        }
    }

    // trust_delta_clip/floor/ceiling are likewise not part of
    // serialize.hpp -- same reapply-every-epoch reasoning as above.
    void set_trust_dynamics(double delta_clip, double floor, double ceiling) {
        config_.trust_delta_clip = delta_clip;
        config_.trust_floor = floor;
        config_.trust_ceiling = ceiling;
    }

    // Ablation/evaluation-only: make learn=false a genuinely read-only
    // traversal.  The deployed inference path normally keeps adapting the
    // burst EMA and residual-context trust even when durable anatomy is
    // frozen.  Counterfactual candidate scoring replays several branches
    // through one cortex, so those updates would otherwise make the answer
    // depend on branch order.  This flag is deliberately runtime-only and
    // leaves ordinary training and inference behavior unchanged by default.
    void set_debug_freeze_inference_adaptation(bool freeze) {
        debug_freeze_inference_adaptation_ = freeze;
    }

    // long_range_allow_permanent_tier is likewise not part of
    // serialize.hpp -- same reapply-every-epoch reasoning as above. Matters
    // more than the others: without reapplying this, a run intended to
    // allow permanent-tier promotion silently falls back to the privacy-
    // gated default (false) after its first epoch, permanently losing
    // access to tier 4 for the rest of the run.
    void set_long_range_allow_permanent_tier(bool allow) {
        config_.long_range_allow_permanent_tier = allow;
    }

    // episodic_strength is likewise not part of serialize.hpp -- same
    // reapply-every-epoch reasoning as above. Also a genuine tuning knob,
    // not just a persistence fix: a strength-sweep at w=2 (k=6/s=5) found a
    // sharp threshold, not a gradual tradeoff -- strength=4.0 (the old
    // default) gave 0/20 exact-match generation recall of a planted code,
    // while EVERY tested value <=1.0 gave 20/20, with no throughput cost
    // either way (strength is pure blend arithmetic, no control-flow
    // change). The math: with a single occurrence (count=1, total=1), the
    // boosted probability is (prior+1)/(1+strength) -- at strength<=1.0
    // that's >=0.5 always, which the combined mass of every competing
    // candidate (squeezed to <=0.5) essentially can't beat. See
    // docs/ALM_V1_Architecture.md.
    void set_episodic_strength(double strength) {
        if (strength <= 0.0) {
            throw std::runtime_error("episodic_strength must be positive");
        }
        config_.episodic_strength = strength;
    }

private:
    void init_episodic_state() {
        if (config_.episodic_k < 1 || config_.episodic_k > kMaxEpisodicKmerWidth) {
            throw std::runtime_error("episodic_k must be in [1, kMaxEpisodicKmerWidth]");
        }
        if (config_.episodic_s < 1 || config_.episodic_s >= config_.episodic_k ||
            config_.episodic_s > kMaxEpisodicKmerWidth) {
            throw std::runtime_error("episodic_s must be in [1, episodic_k)");
        }
        const int32_t syncmer_window = config_.episodic_k - config_.episodic_s + 1;
        if (syncmer_window > kMaxEpisodicSyncmerWindow) {
            throw std::runtime_error("episodic_k - episodic_s + 1 exceeds kMaxEpisodicSyncmerWindow");
        }
        if (config_.episodic_chain_slots < 1 ||
            config_.episodic_chain_slots > kMaxEpisodicChainSlots) {
            throw std::runtime_error("episodic_chain_slots must be in [1, kMaxEpisodicChainSlots]");
        }
        uint64_t kmer_pow = 1;
        for (int32_t i = 1; i < config_.episodic_k; ++i) kmer_pow *= kDeepBandHashBase;
        episodic_kmer_pow_ = kmer_pow;
        uint64_t smer_pow = 1;
        for (int32_t i = 1; i < config_.episodic_s; ++i) smer_pow *= kDeepBandHashBase;
        episodic_smer_pow_ = smer_pow;
        const size_t bucket_count = size_t(1) << config_.episodic_bucket_bits;
        episodic_buckets_.assign(bucket_count, EpisodicBucket{});
        episodic_bucket_mask_ = bucket_count - 1;
    }

public:

    // Scores every byte in `data` and, when learning, develops anatomy in the
    // same interleaved order as the Python fused loop: each backoff level is
    // read at its pre-update counts and then incremented before the next
    // level is consulted.
    void observe_chunk(const uint8_t* data, size_t count, bool learn, double* out) {
        const double prior = config_.prior;
        const double basal_strength = config_.basal_strength;
        const double parent_strength = config_.parent_strength;
        const double surprise_bits = config_.surprise_bits;
        const int32_t context_confirmations = config_.context_confirmations;
        const int32_t sequence_confirmations = config_.sequence_confirmations;
        const int64_t max_nursery = config_.max_sequence_nursery_paths;
        const int32_t max_context = config_.max_context_bytes;

        uint64_t contexts[8];

        for (size_t index = 0; index < count; ++index) {
            const uint8_t target = data[index];
            const bool fast_live_update = learn && live_assemblies_[target] != 0;

            // Reads context_births_/sequence_births_ as they stood before
            // this byte's own growth below -- matches Python's
            // _update_burst_fraction() being called at the top of
            // _observed_probability, before _learn_and_advance runs.
            if (config_.burst_surprise_signal &&
                (learn || !debug_freeze_inference_adaptation_)) {
                update_burst_fraction();
            }

            LongRangeKey long_range_k;
            const bool has_long_range_key =
                config_.long_range_bank && long_range_key(long_range_k);
            std::unordered_map<LongRangeKey, uint32_t, LongRangeKeyHash>::iterator
                long_range_found = long_range_map_.end();
            if (has_long_range_key) long_range_found = long_range_map_.find(long_range_k);

            LongRangeKey deep_exact_keys[CortexConfig::kMaxLongRangeBands];
            bool deep_key_ready[CortexConfig::kMaxLongRangeBands] = {};
            if (config_.long_range_bands) {
                const int32_t band_count = config_.long_range_band_count;
                // Ablation-only: when set, every band's lookup/grow this
                // byte is skipped unless the current position is an
                // anchor-gate syncmer -- see set_debug_deep_band_anchor_gate
                // and the anchor_gate_* field comments. advance_deep_band_
                // history's O(1) rolling-hash bookkeeping below is
                // deliberately NOT gated by this, only the expensive
                // hash-map lookup/insert (deep_band_mature_lookup/
                // deep_band_grow) is, via deep_key_ready.
                const bool anchor_gate_ok = !debug_deep_band_anchor_gate_enabled_ ||
                                             anchor_gate_current_is_syncmer();
                if (config_.long_range_rolling_hash) {
                    for (int32_t b = 0; b < band_count; ++b) {
                        deep_key_ready[b] = anchor_gate_ok &&
                                            deep_band_steps_ >= config_.long_range_band_depths[b];
                    }
                    // Every ready band's rolling-hash key is already
                    // finalized (by the previous byte's
                    // advance_deep_band_history), so both maps
                    // deep_band_blend/deep_band_grow actually probe below
                    // -- mature (always) and nursery (whenever mature
                    // misses, measured to be the common case on diverse
                    // real text) -- can be requested now, hidden behind
                    // the ordinary-bank work in between, same pattern as
                    // sequence_maps_[bank].prefetch above. Prefetching
                    // both costs a little wasted bandwidth on whichever
                    // one a given byte doesn't need, cheap next to a cold
                    // miss on either.
                    for (int32_t b = 0; b < band_count; ++b) {
                        if (!deep_key_ready[b]) continue;
                        deep_mature_hash_[b].prefetch(deep_band_hash_[b]);
                        deep_nursery_hash_[b].map.prefetch(deep_band_hash_[b]);
                    }
                } else {
                    for (int32_t b = 0; b < band_count; ++b) {
                        deep_key_ready[b] = anchor_gate_ok && deep_band_key(b, deep_exact_keys[b]);
                    }
                }
            }

            // Sequence contexts depend only on history, so their buckets can
            // be pulled into cache while the basal levels are evaluated.
            const int32_t available =
                history_length_ < max_context ? history_length_ : max_context;
            const int32_t active_banks = available > 2 ? available - 2 : 0;
            for (int32_t bank = 0; bank < active_banks; ++bank) {
                contexts[bank] = packed_history_ & context_masks_[bank];
                sequence_maps_[bank].prefetch(contexts[bank]);
            }

            double basal;
            if (previous_ < 0) {
                const double basal_total =
                    static_cast<double>(unigram_sum_) + 256.0 * prior;
                basal = (static_cast<double>(unigram_counts_[target]) + prior) /
                        basal_total;
            } else {
                const double basal_total =
                    static_cast<double>(context_totals_[previous_]) + 256.0 * prior;
                basal = (static_cast<double>(
                             arena_.get(transition_rows_[previous_], target)) +
                         prior) /
                        basal_total;
            }
            double probability = basal;

            const bool has_context = older_ >= 0 && previous_ >= 0;
            const uint64_t context_key =
                has_context ? (static_cast<uint64_t>(older_) << 8) |
                                  static_cast<uint64_t>(previous_)
                            : 0;
            if (has_context) {
                const uint32_t slot = residual_map_.find(context_key);
                if (slot != IndexMap::kEmpty) {
                    const int64_t cell = arena_.get(slot, target);
                    const int64_t total = arena_.total(slot);
                    if (!config_.adaptive_competition && config_.competition_exponent == 1.0) {
                        probability = (static_cast<double>(cell) + basal_strength * basal) /
                                      (static_cast<double>(total) + basal_strength);
                    } else {
                        double trust =
                            config_.adaptive_competition ? trust_.trust_of(context_key) : 1.0;
                        if (config_.burst_surprise_signal) {
                            trust *= std::max(0.05, 1.0 - burst_fraction_);
                        }
                        const double child_probability =
                            static_cast<double>(cell) / static_cast<double>(total);
                        const auto weights = competition_weights(
                            basal_strength, static_cast<double>(total), trust,
                            config_.competition_exponent);
                        probability = weights
                                          ? weights->first * basal + weights->second * child_probability
                                          : (static_cast<double>(cell) + basal_strength * basal) /
                                                (static_cast<double>(total) + basal_strength);
                        if (config_.adaptive_competition &&
                            (learn || !debug_freeze_inference_adaptation_)) {
                            trust_.credit(context_key, basal, child_probability,
                                         config_.trust_gamma,
                                         config_.attribution_gate_threshold,
                                         config_.trust_delta_clip,
                                         config_.trust_floor,
                                         config_.trust_ceiling);
                        }
                    }
                    if (fast_live_update) arena_.increment(slot, target);
                } else if (fast_live_update &&
                           -std::log2(basal > 1e-300 ? basal : 1e-300) >=
                               surprise_bits) {
                    grow_context_nursery(context_key, target, context_confirmations);
                }
                if (fast_live_update && tracking_) touched_contexts_.insert(context_key);
            }

            if (fast_live_update) {
                unigram_counts_[target] += 1;
                unigram_sum_ += 1;
                if (previous_ >= 0) {
                    arena_.increment(transition_rows_[previous_], target);
                    context_totals_[previous_] += 1;
                }
            }

            for (int32_t bank = 0; bank < active_banks; ++bank) {
                const uint64_t context = contexts[bank];
                const uint32_t slot = sequence_maps_[bank].find(context);
                if (slot != IndexMap::kEmpty) {
                    const int64_t cell = bank_get(bank, slot, target);
                    probability =
                        (static_cast<double>(cell) + parent_strength * probability) /
                        (static_cast<double>(bank_total(bank, slot)) + parent_strength);
                    if (fast_live_update) {
                        bank_increment(bank, slot, target);
                        if (tracking_) touched_sequences_[bank].insert(context);
                    }
                } else if (fast_live_update) {
                    if (tracking_) touched_sequences_[bank].insert(context);
                    grow_sequence_nursery(bank, context, target, sequence_confirmations,
                                          max_nursery);
                }
            }

            if (long_range_found != long_range_map_.end()) {
                const uint32_t slot = long_range_found->second;
                const int64_t cell = arena_.get(slot, target);
                const int64_t total = arena_.total(slot);
                const double strength = config_.long_range_strength;
                probability = (static_cast<double>(cell) + strength * probability) /
                              (static_cast<double>(total) + strength);
            }

            int64_t deep_mature_slot[CortexConfig::kMaxLongRangeBands];
            double deep_probability_before[CortexConfig::kMaxLongRangeBands];
            if (config_.long_range_bands) {
                const int32_t band_count = config_.long_range_band_count;
                uint32_t found_slot[CortexConfig::kMaxLongRangeBands];
                for (int32_t b = 0; b < band_count; ++b) {
                    found_slot[b] = deep_band_mature_lookup(
                        b, deep_exact_keys[b], deep_band_hash_[b], deep_key_ready[b]);
                    if (!deep_key_ready[b]) continue;
                    if (found_slot[b] == IndexMap::kEmpty) {
                        ++deep_band_mature_misses_;
                    } else {
                        ++deep_band_mature_hits_;
                    }
                }
                for (int32_t b = 0; b < band_count; ++b) {
                    deep_probability_before[b] = probability;
                    deep_mature_slot[b] = deep_band_combine(found_slot[b], target, probability);
                }
            }

            uint64_t episodic_bucket_id = 0;
            bool episodic_is_syncmer = false;
            if (config_.episodic_enabled) {
                const uint32_t episodic_slot = episodic_lookup(&episodic_bucket_id, &episodic_is_syncmer);
                episodic_combine(episodic_slot, target, probability);
            }

            if (steps_ != 0) out[index] = probability;

            if (config_.long_range_bands && config_.long_range_use_combined_salience) {
                const double byte_surprise =
                    -std::log2(probability > 1e-300 ? probability : 1e-300);
                span_surprise_window_.push_back(byte_surprise);
                span_surprise_sum_ += byte_surprise;
                while (static_cast<int32_t>(span_surprise_window_.size()) >
                       config_.long_range_salience_span_bytes) {
                    span_surprise_sum_ -= span_surprise_window_.front();
                    span_surprise_window_.pop_front();
                }
            }

            if (learn) {
                const bool is_live = fast_live_update || develop(target);
                if (is_live) {
                    if (!fast_live_update) {
                        catch_up(target, basal, has_context, context_key, contexts,
                                 active_banks);
                    }
                    if (config_.long_range_bank) {
                        if (has_long_range_key) {
                            grow_long_range(long_range_k, long_range_found, target);
                        }
                        advance_long_range_history(target);
                    }
                    if (config_.long_range_bands) {
                        const int32_t band_count = config_.long_range_band_count;
                        for (int32_t b = 0; b < band_count; ++b) {
                            deep_band_grow(b, deep_exact_keys[b], deep_band_hash_[b],
                                          deep_key_ready[b], target,
                                          deep_probability_before[b], deep_mature_slot[b]);
                        }
                        advance_deep_band_history(target);
                        if (debug_deep_band_anchor_gate_enabled_) advance_anchor_gate(target);
                    }
                    older_ = previous_;
                    previous_ = target;
                    packed_history_ =
                        ((packed_history_ << 8) | static_cast<uint64_t>(target)) &
                        history_mask_;
                    if (history_length_ < max_context) ++history_length_;
                } else {
                    clear_history();
                }
                ++steps_;
            } else if (live_assemblies_[target]) {
                if (config_.long_range_bank) advance_long_range_history(target);
                if (config_.long_range_bands) {
                    advance_deep_band_history(target);
                    if (debug_deep_band_anchor_gate_enabled_) advance_anchor_gate(target);
                }
                older_ = previous_;
                previous_ = target;
                packed_history_ =
                    ((packed_history_ << 8) | static_cast<uint64_t>(target)) &
                    history_mask_;
                if (history_length_ < max_context) ++history_length_;
            } else {
                clear_history();
            }

            // Deliberately OUTSIDE the learn/is_live/live_assemblies_
            // branching above: memory and learning are decoupled by
            // design. "Learning" here means rewiring the durable cortex
            // (assembly formation, deep-band nursery/mature growth) --
            // whether that happens is what the branches above gate. The
            // episodic layer instead tracks what was actually said in
            // this episode, which should keep happening even when the
            // organism is frozen (learn=false) or this particular byte's
            // assembly hasn't developed -- otherwise a deployed, frozen
            // instance could look up memories but could never form new
            // ones during generation, defeating the point of an
            // always-on conversational memory.
            if (config_.episodic_enabled) {
                if (episodic_is_syncmer) episodic_write(episodic_bucket_id, target);
                advance_episodic(target);
            }
        }
    }

    void reset_episode() {
        clear_history();
        clear_episodic_store();
    }

    // Full 256-way prediction at the current context, with no side effects
    // -- no learning, no episode advance, no trust credit (mirrors Python's
    // predict_next_distribution()/`_distribution()`, which query only).
    void predict_distribution(double* out) const {
        const double prior = config_.prior;
        const double basal_strength = config_.basal_strength;
        const double parent_strength = config_.parent_strength;
        const int32_t max_context = config_.max_context_bytes;

        if (previous_ < 0) {
            const double total = static_cast<double>(unigram_sum_) + 256.0 * prior;
            for (int t = 0; t < 256; ++t) {
                out[t] = (static_cast<double>(unigram_counts_[t]) + prior) / total;
            }
        } else {
            const double total =
                static_cast<double>(context_totals_[previous_]) + 256.0 * prior;
            for (int t = 0; t < 256; ++t) out[t] = prior / total;
            arena_.for_each(transition_rows_[previous_], [&](uint8_t key, int64_t count) {
                out[key] = (static_cast<double>(count) + prior) / total;
            });
        }

        if (older_ >= 0 && previous_ >= 0) {
            const uint64_t context_key =
                (static_cast<uint64_t>(older_) << 8) | static_cast<uint64_t>(previous_);
            const uint32_t slot = residual_map_.find(context_key);
            if (slot != IndexMap::kEmpty) {
                const int64_t total = arena_.total(slot);
                double trust = config_.adaptive_competition ? trust_.trust_of(context_key) : 1.0;
                if (config_.burst_surprise_signal) {
                    trust *= std::max(0.05, 1.0 - burst_fraction_);
                }
                const auto weights = competition_weights(
                    basal_strength, static_cast<double>(total), trust,
                    config_.competition_exponent);
                if (!weights) {
                    // Matches Python's exact operation order (scale, add
                    // raw counts, divide once at the end) rather than
                    // dividing per term -- algebraically equal, not
                    // necessarily bit-equal, and this path is the one that
                    // must stay bit-exact at the (overwhelmingly common)
                    // fixed-backoff default.
                    const double denominator = static_cast<double>(total) + basal_strength;
                    for (int t = 0; t < 256; ++t) out[t] *= basal_strength;
                    arena_.for_each(slot, [&](uint8_t key, int64_t count) {
                        out[key] += static_cast<double>(count);
                    });
                    for (int t = 0; t < 256; ++t) out[t] /= denominator;
                } else {
                    const double parent_weight = weights->first;
                    const double child_weight = weights->second;
                    for (int t = 0; t < 256; ++t) out[t] *= parent_weight;
                    arena_.for_each(slot, [&](uint8_t key, int64_t count) {
                        out[key] += child_weight * (static_cast<double>(count) /
                                                    static_cast<double>(total));
                    });
                }
            }
        }

        const int32_t available =
            history_length_ < max_context ? history_length_ : max_context;
        const int32_t active_banks = available > 2 ? available - 2 : 0;
        for (int32_t bank = 0; bank < active_banks; ++bank) {
            const uint64_t context = packed_history_ & context_masks_[bank];
            const uint32_t slot = sequence_maps_[bank].find(context);
            if (slot == IndexMap::kEmpty) continue;
            const int64_t total = bank_total(bank, slot);
            const double denominator = static_cast<double>(total) + parent_strength;
            for (int t = 0; t < 256; ++t) out[t] *= parent_strength;
            bank_for_each(bank, slot, [&](uint8_t key, int64_t count) {
                out[key] += static_cast<double>(count);
            });
            for (int t = 0; t < 256; ++t) out[t] /= denominator;
        }

        if (config_.long_range_bank &&
            static_cast<int32_t>(long_range_history_.size()) >=
                config_.long_range_context_bytes) {
            LongRangeKey key;
            key.length = static_cast<uint8_t>(long_range_history_.size());
            size_t key_index = 0;
            for (uint8_t byte_value : long_range_history_) key.bytes[key_index++] = byte_value;
            auto found = long_range_map_.find(key);
            if (found != long_range_map_.end()) {
                const uint32_t slot = found->second;
                const int64_t total = arena_.total(slot);
                const double strength = config_.long_range_strength;
                const double denominator = static_cast<double>(total) + strength;
                for (int t = 0; t < 256; ++t) out[t] *= strength;
                arena_.for_each(slot, [&](uint8_t key, int64_t count) {
                    out[key] += static_cast<double>(count);
                });
                for (int t = 0; t < 256; ++t) out[t] /= denominator;
            }
        }

        if (config_.long_range_bands) {
            const int32_t band_count = config_.long_range_band_count;
            for (int32_t b = 0; b < band_count; ++b) {
                const int32_t depth = config_.long_range_band_depths[b];
                if (deep_band_steps_ < depth) continue;
                uint32_t slot = IndexMap::kEmpty;
                if (config_.long_range_rolling_hash) {
                    slot = deep_mature_hash_[b].find(deep_band_hash_[b]);
                    if (slot != IndexMap::kEmpty && debug_ignore_tier_at_or_above_ > 0) {
                        const uint32_t stored_tier = deep_tier_hash_[b].find(deep_band_hash_[b]);
                        const int32_t effective_tier =
                            (stored_tier == IndexMap::kEmpty) ? 2 : static_cast<int32_t>(stored_tier);
                        if (effective_tier >= debug_ignore_tier_at_or_above_) slot = IndexMap::kEmpty;
                    }
                } else {
                    LongRangeKey key;
                    key.length = static_cast<uint8_t>(depth);
                    const size_t start = deep_band_history_.size() - static_cast<size_t>(depth);
                    for (int32_t i = 0; i < depth; ++i) {
                        key.bytes[i] = deep_band_history_[start + static_cast<size_t>(i)];
                    }
                    const auto it = deep_mature_exact_[b].find(key);
                    if (it != deep_mature_exact_[b].end()) {
                        slot = it->second;
                        if (debug_ignore_tier_at_or_above_ > 0) {
                            const auto tier_it = deep_tier_exact_[b].find(key);
                            const int32_t effective_tier = (tier_it != deep_tier_exact_[b].end())
                                                                ? static_cast<int32_t>(tier_it->second)
                                                                : 2;
                            if (effective_tier >= debug_ignore_tier_at_or_above_) slot = IndexMap::kEmpty;
                        }
                    }
                }
                if (slot == IndexMap::kEmpty) continue;
                const int64_t total = arena32_.total(slot);
                const double strength = config_.long_range_strength;
                const double denominator = static_cast<double>(total) + strength;
                for (int t = 0; t < 256; ++t) out[t] *= strength;
                arena32_.for_each(slot, [&](uint8_t key, int64_t count) {
                    out[key] += static_cast<double>(count);
                });
                for (int t = 0; t < 256; ++t) out[t] /= denominator;
            }
        }

        if (config_.episodic_enabled && episodic_current_is_syncmer()) {
            const uint64_t bucket_id = mix64(episodic_kmer_hash_) & episodic_bucket_mask_;
            const EpisodicBucket& b = episodic_buckets_[bucket_id];
            uint32_t slot = IndexMap::kEmpty;
            for (int32_t i = 0; i < b.used; ++i) {
                if (b.key[i] == episodic_kmer_hash_) { slot = b.arena_slot[i]; break; }
            }
            if (slot != IndexMap::kEmpty) {
                const int64_t total = arena32_.total(slot);
                const double strength = config_.episodic_strength;
                const double denominator = static_cast<double>(total) + strength;
                for (int t = 0; t < 256; ++t) out[t] *= strength;
                arena32_.for_each(slot, [&](uint8_t key, int64_t count) {
                    out[key] += static_cast<double>(count);
                });
                for (int t = 0; t < 256; ++t) out[t] /= denominator;
            }
        }
    }

    void begin_lineage_tracking() {
        tracking_ = true;
        touched_contexts_.clear();
        touched_sequences_.assign(bank_count_, {});
    }

    int64_t steps() const noexcept { return steps_; }
    int64_t context_births() const noexcept { return context_births_; }
    int64_t sequence_births() const noexcept { return sequence_births_; }
    int64_t sequence_evictions() const noexcept { return sequence_evictions_; }
    int64_t assembly_births() const noexcept { return assembly_births_; }
    double burst_fraction() const noexcept { return burst_fraction_; }
    int64_t long_range_entry_count() const noexcept {
        return static_cast<int64_t>(long_range_map_.size());
    }
    // Debug-only: total mature entries across all deep bands (whichever
    // backend), for verifying promotion actually happened during
    // development of this feature.
    int64_t deep_band_mature_count() const noexcept {
        int64_t total = 0;
        if (config_.long_range_rolling_hash) {
            for (const auto& m : deep_mature_hash_) total += static_cast<int64_t>(m.size());
        } else {
            for (const auto& m : deep_mature_exact_) total += static_cast<int64_t>(m.size());
        }
        return total;
    }
    // Debug-only: total keys currently classified tier 4 (permanent)
    // across all deep bands -- lets a caller confirm an ablation using
    // debug_ignore_permanent_tier_ actually has something to block,
    // rather than trivially matching an unablated run because nothing
    // has graduated yet.
    int64_t deep_permanent_member_count() const noexcept {
        int64_t total = 0;
        if (config_.long_range_rolling_hash) {
            for (const auto& m : deep_permanent_members_hash_) total += static_cast<int64_t>(m.size());
        } else {
            for (const auto& m : deep_permanent_members_exact_) total += static_cast<int64_t>(m.size());
        }
        return total;
    }

    // Summary of the learned adaptive-competition trust distribution
    // (TrustState::context_trust, adaptive_trust.hpp) -- values are
    // multiplicatively updated and clipped to [0.1, 10.0], with 1.0 the
    // default/never-updated value. Answers "has this actually developed
    // meaningful, non-trivial differentiation, or stayed near-flat" --
    // count_near_neutral close to count means most visited contexts never
    // moved meaningfully from their starting weight; count_at_floor/
    // count_at_ceiling saturating means updates are hitting the clip
    // rather than settling at an intermediate value.
    struct TrustSummary {
        int64_t count = 0;
        double min_value = 0.0;
        double max_value = 0.0;
        double mean_value = 0.0;
        int64_t count_at_floor = 0;
        int64_t count_at_ceiling = 0;
        int64_t count_near_neutral = 0;  // within [0.99, 1.01]
    };
    TrustSummary trust_summary() const {
        TrustSummary s;
        double min_value = 1e300, max_value = -1e300, sum = 0.0;
        const double floor_eps = config_.trust_floor + 1e-9;
        const double ceiling_eps = config_.trust_ceiling - 1e-9;
        for (const auto& kv : trust_.context_trust) {
            const double value = kv.second;
            ++s.count;
            sum += value;
            if (value < min_value) min_value = value;
            if (value > max_value) max_value = value;
            if (value <= floor_eps) ++s.count_at_floor;
            if (value >= ceiling_eps) ++s.count_at_ceiling;
            if (value >= 0.99 && value <= 1.01) ++s.count_near_neutral;
        }
        s.min_value = s.count ? min_value : 0.0;
        s.max_value = s.count ? max_value : 0.0;
        s.mean_value = s.count ? sum / static_cast<double>(s.count) : 0.0;
        return s;
    }
    int64_t deep_band_steps_debug() const noexcept { return deep_band_steps_; }
    int64_t deep_band_nursery_count() const noexcept {
        int64_t total = 0;
        if (config_.long_range_rolling_hash) {
            for (const auto& n : deep_nursery_hash_) total += static_cast<int64_t>(n.map.size());
        } else {
            for (const auto& n : deep_nursery_exact_) total += static_cast<int64_t>(n.size());
        }
        return total;
    }

    // Affective Intensity's other half: reprocess every queued replay
    // candidate as if observed again, riding the existing wake/sleep
    // split rather than a new consolidation mechanism. Call once per
    // sleep cycle, before consolidate_to_byte_budget -- a replayed key
    // that crosses its confirmation threshold here promotes immediately,
    // exactly like a real repeated observation would, since a queued
    // key (a bear chase) may never naturally recur again on its own.
    void replay_reinforce() {
        const int64_t weight = 1 + config_.long_range_importance_bonus;
        for (const auto& candidate : replay_queue_) {
            reinforce_deep_band_key(candidate.band, candidate.exact_key, candidate.hash_key,
                                    candidate.target, weight);
        }
    }
    int64_t replay_queue_size() const noexcept {
        return static_cast<int64_t>(replay_queue_.size());
    }

    // On-demand version of connectivity_pruning's degree computation
    // (sleep.hpp): how many mature rows in bank+1 currently have `context`
    // (a bank-depth context, i.e. depth = bank+3) as their trailing suffix,
    // right now -- not a periodic sleep-time sweep. O(|bank+1|) per call,
    // fine for a query/experiment, not yet cheap enough for a per-byte
    // production decision. Salience-signal prototype: this is the
    // "does this context already matter to the organism" half of a
    // surprise x hub-degree combined score.
    int64_t context_degree(int32_t bank, uint64_t context) const {
        if (bank < 0 || bank + 1 >= bank_count_) return 0;
        const int32_t depth = bank + 3;
        const uint64_t mask = (depth >= 8) ? ~0ULL : ((1ULL << (8 * depth)) - 1ULL);
        int64_t degree = 0;
        sequence_maps_[static_cast<size_t>(bank + 1)].for_each(
            [&](uint64_t child_context, uint32_t) {
                if ((child_context & mask) == context) ++degree;
            });
        return degree;
    }

    // Convenience: degree of whatever context is CURRENTLY active at the
    // given bank depth (packed_history_ masked the same way observe_chunk
    // computes it), for querying "how important is my current context"
    // without the caller needing to reconstruct the mask/packing logic.
    int64_t current_context_degree(int32_t bank) const {
        if (bank < 0 || bank >= bank_count_) return 0;
        const int32_t available = history_length_ < config_.max_context_bytes
                                       ? history_length_
                                       : config_.max_context_bytes;
        const int32_t active_banks = available > 2 ? available - 2 : 0;
        if (bank >= active_banks) return 0;
        const uint64_t context = packed_history_ & context_masks_[static_cast<size_t>(bank)];
        return context_degree(bank, context);
    }

    // Production-viable version of the above: context_degree/
    // current_context_degree are O(|bank+1|) per call, and measured (not
    // assumed) to make salience_weight's per-byte use of them ruinously
    // slow on real diverse text (a 400K-byte test took 3+ minutes) --
    // first-sighting events, the trigger for this lookup, turn out to be
    // the COMMON case on real text, not rare. Fix: cache degree for every
    // context, refreshed once per sleep cycle (refresh_degree_cache,
    // called from consolidate_to_byte_budget) instead of recomputed fresh
    // on every first-sighting. Reads are O(1). The cache is current as of
    // the last sleep, not the current byte -- an approximation, not exact
    // recall, but salience is already a coarse "how important is this"
    // signal, not something that needs byte-fresh precision.
    int64_t cached_context_degree(int32_t bank, uint64_t context) const {
        if (bank < 0 || bank >= static_cast<int32_t>(cached_degree_.size())) return 0;
        const auto& degree_map = cached_degree_[static_cast<size_t>(bank)];
        const auto it = degree_map.find(context);
        return it == degree_map.end() ? 0 : it->second;
    }
    int64_t current_cached_context_degree(int32_t bank) const {
        if (bank < 0 || bank >= bank_count_) return 0;
        const int32_t available = history_length_ < config_.max_context_bytes
                                       ? history_length_
                                       : config_.max_context_bytes;
        const int32_t active_banks = available > 2 ? available - 2 : 0;
        if (bank >= active_banks) return 0;
        const uint64_t context = packed_history_ & context_masks_[static_cast<size_t>(bank)];
        return cached_context_degree(bank, context);
    }
    void refresh_degree_cache() {
        cached_degree_.assign(static_cast<size_t>(bank_count_), {});
        for (int32_t bank = 0; bank + 1 < bank_count_; ++bank) {
            const int32_t depth = bank + 3;
            const uint64_t mask = (depth >= 8) ? ~0ULL : ((1ULL << (8 * depth)) - 1ULL);
            std::unordered_map<uint64_t, int32_t>& parent_degree =
                cached_degree_[static_cast<size_t>(bank)];
            sequence_maps_[static_cast<size_t>(bank + 1)].for_each(
                [&](uint64_t child_context, uint32_t) {
                    ++parent_degree[child_context & mask];
                });
        }
    }

    // Cheap (O(1), one hash lookup -- NOT context_degree's O(|bank+1|)
    // scan) re-encounter estimate for the current context at `bank`: its
    // own accumulated observation count relative to total steps taken.
    // This is the frequentist reading of the same posterior the ordinary
    // backoff formula already computes, just exposed directly for AI's
    // burst x re-encounter-probability score.
    double current_context_frequency(int32_t bank) const {
        if (bank < 0 || bank >= bank_count_ || steps_ <= 0) return 0.0;
        const int32_t available = history_length_ < config_.max_context_bytes
                                       ? history_length_
                                       : config_.max_context_bytes;
        const int32_t active_banks = available > 2 ? available - 2 : 0;
        if (bank >= active_banks) return 0.0;
        const uint64_t context = packed_history_ & context_masks_[static_cast<size_t>(bank)];
        const uint32_t slot = sequence_maps_[static_cast<size_t>(bank)].find(context);
        if (slot == IndexMap::kEmpty) return 0.0;
        return static_cast<double>(bank_total(bank, slot)) / static_cast<double>(steps_);
    }

    int64_t live_sequence_count() const noexcept {
        int64_t total = 0;
        for (const auto& map : sequence_maps_) total += static_cast<int64_t>(map.size());
        return total;
    }
    int64_t live_sequence_count_at_bank(int32_t bank) const noexcept {
        if (bank < 0 || bank >= bank_count_) return 0;
        return static_cast<int64_t>(sequence_maps_[static_cast<size_t>(bank)].size());
    }
    int64_t live_context_count() const noexcept {
        return static_cast<int64_t>(residual_map_.size());
    }
    int64_t sequence_nursery_count() const noexcept {
        int64_t total = 0;
        for (const auto& bank : sequence_nursery_)
            total += static_cast<int64_t>(bank.map.size());
        return total;
    }

    // Mixed-precision sizing check: see RowArena::max_count_value.
    int64_t max_count_value(uint32_t skip_rows = 0) const noexcept {
        return arena_.max_count_value(skip_rows);
    }
    // Mixed-precision sizing check for deep bands specifically -- these
    // have never run at real production scale, so this is how to check
    // whether a narrower CountT (see arena32_'s declaration) is actually
    // safe before trusting small-corpus tests that never reach anywhere
    // near a narrow type's ceiling.
    int64_t max_deep_band_count_value() const noexcept {
        return arena32_.max_count_value();
    }

    // Mixed-precision sizing check: max count within just the ordinary
    // sequence banks (depth 3+, the millions of rows that actually
    // dominate total memory), separate from the coarse unigram/
    // transition/residual levels which see cumulative whole-corpus
    // traffic and can't be cleanly excluded by row index alone.
    int64_t max_sequence_bank_count(int32_t bank) const noexcept {
        if (bank < 0 || bank >= bank_count_) return 0;
        int64_t max_value = 0;
        sequence_maps_[static_cast<size_t>(bank)].for_each(
            [&](uint64_t, uint32_t slot) {
                const int64_t t = bank_total(bank, slot);
                if (t > max_value) max_value = t;
            });
        return max_value;
    }

    // Real, per-row-measured estimate of what bank `bank` would cost at a
    // given count-storage width, using each live row's actual dense/
    // sparse state and capacity (not a rough average) -- so this is the
    // real answer for "how much smaller would this bank be at int32 vs
    // int16 vs int64", not a guess. Row-struct overhead itself also
    // shrinks with a narrower CountT (offset+size+capacity+total+dense,
    // padded to alignment), included per row. `count_bytes` is the width
    // to price (8 for int64_t, 4 for int32_t, 2 for int16_t, 1 for int8_t).
    int64_t estimate_bank_bytes(int32_t bank, int32_t count_bytes) const noexcept {
        if (bank < 0 || bank >= bank_count_) return 0;
        // Row<CountT>: uint32_t offset, uint16_t size, uint16_t capacity,
        // CountT total, bool dense -- aligned to CountT's own alignment
        // (or 4 bytes, whichever is larger, matching typical struct
        // packing rules).
        const int64_t align = count_bytes > 4 ? count_bytes : 4;
        const int64_t raw_row_size = 4 + 2 + 2 + count_bytes + 1;
        const int64_t row_struct_bytes =
            ((raw_row_size + align - 1) / align) * align;
        int64_t total_bytes = 0;
        const bool deep = is_deep_bank(bank);
        sequence_maps_[static_cast<size_t>(bank)].for_each(
            [&](uint64_t, uint32_t slot) {
                const bool dense = deep ? arena_deep_bank_.row(slot).dense
                                         : arena_.row(slot).dense;
                const uint16_t capacity = deep ? arena_deep_bank_.row(slot).capacity
                                                : arena_.row(slot).capacity;
                total_bytes += row_struct_bytes;
                if (dense) {
                    total_bytes += 256 * static_cast<int64_t>(count_bytes);
                } else {
                    total_bytes += static_cast<int64_t>(capacity) *
                                   (count_bytes + 1);  // + 1-byte sparse key
                }
            });
        return total_bytes;
    }

    size_t memory_bytes() const noexcept {
        size_t bytes = arena_.memory_bytes() + arena32_.memory_bytes() +
                       residual_map_.memory_bytes() + context_nursery_.memory_bytes();
        for (const auto& map : sequence_maps_) bytes += map.memory_bytes();
        for (const auto& bank : sequence_nursery_) {
            bytes += bank.map.memory_bytes() + bank.order.size() * sizeof(uint64_t);
        }
        return bytes;
    }

    // Drops every unconfirmed nursery path, the sleep step that precedes
    // consolidation in the wake/sleep lifecycle.
    int64_t shed_sequence_nursery() {
        int64_t shed = 0;
        for (int32_t bank = 0; bank < static_cast<int32_t>(sequence_nursery_.size()); ++bank) {
            NurseryBank& nursery = sequence_nursery_[static_cast<size_t>(bank)];
            shed += static_cast<int64_t>(nursery.map.size());
            nursery.map.for_each(
                [&](uint64_t, uint32_t slot) { bank_release_row(bank, slot); });
            nursery.clear();
        }
        return shed;
    }

    // Accessors used by the packer and the serializer.
    int32_t bank_count() const noexcept { return bank_count_; }
    // Public wrapper for non-friend callers (e.g. PackedCortex) that need
    // to route a bank to the right arena themselves.
    bool bank_is_deep(int32_t bank) const noexcept { return is_deep_bank(bank); }
    const IndexMap& sequence_map(int32_t bank) const { return sequence_maps_[bank]; }
    const IndexMap& residual_map() const { return residual_map_; }
    const RowArena<int64_t>& arena() const { return arena_; }
    const RowArena<int32_t>& deep_bank_arena() const { return arena_deep_bank_; }
    const std::vector<int64_t>& unigram_counts() const { return unigram_counts_; }
    const std::vector<int64_t>& context_totals() const { return context_totals_; }
    const std::vector<uint8_t>& live_assemblies() const { return live_assemblies_; }
    uint32_t transition_row(int index) const { return transition_rows_[index]; }
    int64_t unigram_sum() const noexcept { return unigram_sum_; }
    int32_t previous() const noexcept { return previous_; }
    int32_t older() const noexcept { return older_; }

private:
    void clear_history() {
        older_ = -1;
        previous_ = -1;
        packed_history_ = 0;
        history_length_ = 0;
        long_range_history_.clear();
        if (config_.long_range_bands) {
            deep_band_history_.clear();
            deep_band_steps_ = 0;
            for (int32_t b = 0; b < config_.long_range_band_count; ++b) deep_band_hash_[b] = 0;
            if (config_.long_range_use_combined_salience) {
                span_surprise_window_.clear();
                span_surprise_sum_ = 0.0;
            }
            // Reset in lockstep with deep_band_history_ above -- both must
            // see exactly the same set of "live" bytes for the anchor
            // gate's same-window-same-decision guarantee to hold.
            if (debug_deep_band_anchor_gate_enabled_) {
                anchor_gate_kmer_hash_ = 0;
                anchor_gate_kmer_pos_ = 0;
                anchor_gate_kmer_filled_ = 0;
                anchor_gate_smer_hash_ = 0;
                anchor_gate_smer_pos_ = 0;
                anchor_gate_smer_filled_ = 0;
                anchor_gate_syncmer_head_ = 0;
                anchor_gate_syncmer_count_ = 0;
            }
        }
        // Deliberately NOT resetting any episodic state here, unlike the
        // deep bands above: clear_history() also runs on the far more
        // frequent per-byte "this assembly never developed" fallback in
        // observe_chunk, not just on a genuine reset_episode() call.
        // Memory and learning are decoupled by design (see the
        // observe_chunk comment at the unconditional episodic block) --
        // resetting the episodic rolling-hash state here would wipe its
        // in-progress k-mer window on almost every byte for any
        // lightly-trained cortex (most bytes' assemblies aren't "live"
        // yet), so it could never accumulate the depth needed to become
        // ready at all. See clear_episodic_store() for the real episode
        // boundary, called only from reset_episode().
    }

    // The expensive half of an episode boundary: wipes every stored
    // episodic entry (and releases its arena32_ row) rather than just the
    // rolling-hash state clear_history() resets on every "not live" byte.
    // Called only from reset_episode() -- see the comment there.
    void clear_episodic_store() {
        if (!config_.episodic_enabled) return;
        for (auto& bucket : episodic_buckets_) {
            for (int32_t i = 0; i < bucket.used; ++i) arena32_.release_row(bucket.arena_slot[i]);
            bucket.used = 0;
        }
        episodic_kmer_hash_ = 0;
        episodic_kmer_pos_ = 0;
        episodic_kmer_filled_ = 0;
        episodic_smer_hash_ = 0;
        episodic_smer_pos_ = 0;
        episodic_smer_filled_ = 0;
        episodic_syncmer_head_ = 0;
        episodic_syncmer_count_ = 0;
        episodic_steps_ = 0;
    }

    void update_burst_fraction() {
        const int64_t current_total = context_births_ + sequence_births_;
        const double indicator = current_total > burst_last_birth_total_ ? 1.0 : 0.0;
        burst_last_birth_total_ = current_total;
        burst_fraction_ += config_.burst_ema_rate * (indicator - burst_fraction_);
    }

    // Present key, if the history buffer is already full -- mirrors
    // Python's _long_range_key() exactly, including returning "no key yet"
    // while history is still filling. Fixed-size, stack-only: no
    // allocation, unlike the std::string version this replaced.
    bool long_range_key(LongRangeKey& out) const {
        if (!config_.long_range_bank ||
            static_cast<int32_t>(long_range_history_.size()) <
                config_.long_range_context_bytes) {
            return false;
        }
        out.length = static_cast<uint8_t>(long_range_history_.size());
        size_t i = 0;
        for (uint8_t byte_value : long_range_history_) out.bytes[i++] = byte_value;
        return true;
    }

    void advance_long_range_history(uint8_t target) {
        if (!config_.long_range_bank) return;
        long_range_history_.push_back(target);
        while (static_cast<int32_t>(long_range_history_.size()) >
               config_.long_range_context_bytes) {
            long_range_history_.pop_front();
        }
    }

    // Mirrors Python's _grow_long_range exactly: the burst-gate applies
    // only to *creating* a new key, never to reinforcing one that already
    // exists (gating both cost real recall confidence -- see cortex.hpp's
    // CortexConfig comment on long_range_gate_on_burst). Takes the read
    // step's already-computed find() result so growth never repeats a
    // lookup the caller already did.
    void grow_long_range(
        const LongRangeKey& key,
        std::unordered_map<LongRangeKey, uint32_t, LongRangeKeyHash>::iterator found,
        uint8_t target) {
        const bool is_new_key = found == long_range_map_.end();
        if (is_new_key && config_.long_range_gate_on_burst &&
            burst_fraction_ < config_.long_range_gate_threshold) {
            return;
        }
        uint32_t slot;
        if (is_new_key) {
            if (static_cast<int64_t>(long_range_map_.size()) >=
                config_.long_range_max_entries) {
                while (!long_range_order_.empty()) {
                    const LongRangeKey& victim = long_range_order_.front();
                    auto victim_it = long_range_map_.find(victim);
                    if (victim_it != long_range_map_.end()) {
                        arena_.release_row(victim_it->second);
                        long_range_map_.erase(victim_it);
                        long_range_order_.pop_front();
                        break;
                    }
                    long_range_order_.pop_front();
                }
            }
            slot = arena_.create_row();
            long_range_map_.emplace(key, slot);
            long_range_order_.push_back(key);
        } else {
            slot = found->second;
        }
        arena_.increment(slot, target);
    }

    // ---- Idea 2 continued: sparse multi-depth long-range bands ----
    //
    // One shared raw-byte ring buffer (deep_band_history_, sized to the
    // deepest active band) serves both backends: the exact backend copies
    // its trailing d bytes straight out of it; the rolling-hash backend
    // maintains an O(1)-per-byte incremental hash alongside it and never
    // reads the buffer's older bytes directly except to compute the value
    // sliding out of each band's own window. Advanced from the same two
    // call sites long_range_history_ already is, for the same reason: the
    // context should reflect what was actually learned from, not bytes
    // seen during a non-live replay.
    void advance_deep_band_history(uint8_t target) {
        if (!config_.long_range_bands || config_.long_range_band_count <= 0) return;
        const int32_t band_count = config_.long_range_band_count;
        if (config_.long_range_rolling_hash) {
            const int64_t pre_push_len = static_cast<int64_t>(deep_band_history_.size());
            for (int32_t b = 0; b < band_count; ++b) {
                const int32_t depth = config_.long_range_band_depths[b];
                if (pre_push_len >= depth) {
                    const uint8_t outgoing = deep_band_history_[static_cast<size_t>(pre_push_len - depth)];
                    deep_band_hash_[b] = (deep_band_hash_[b] -
                                          (static_cast<uint64_t>(outgoing) + 1) * deep_band_pow_[b]) *
                                             kDeepBandHashBase +
                                         (static_cast<uint64_t>(target) + 1);
                } else {
                    deep_band_hash_[b] =
                        deep_band_hash_[b] * kDeepBandHashBase + (static_cast<uint64_t>(target) + 1);
                }
            }
        }
        deep_band_history_.push_back(target);
        ++deep_band_steps_;
        int32_t max_depth = 0;
        for (int32_t b = 0; b < band_count; ++b) {
            max_depth = std::max(max_depth, config_.long_range_band_depths[b]);
        }
        while (static_cast<int32_t>(deep_band_history_.size()) > max_depth) {
            deep_band_history_.pop_front();
        }
    }

    // ---- Sparse episodic layer (see CortexConfig::episodic_* above) ----
    //
    // True iff the CURRENT (pre-target) episodic_k-byte window is fully
    // formed AND is a closed syncmer: the minimum s-mer hash among its
    // (k-s+1) sub-windows sits at either end. Context-independent by
    // construction -- depends only on the k bytes themselves, not on
    // anything before or after them, unlike a large-window minimizer
    // (which the standalone prototype confirmed collapses to near-zero
    // recall once surrounding context differs between two occurrences of
    // the same short marker).
    bool episodic_current_is_syncmer() const noexcept {
        const int32_t window = config_.episodic_k - config_.episodic_s + 1;
        if (episodic_kmer_filled_ < config_.episodic_k) return false;
        if (episodic_syncmer_count_ < window) return false;
        uint64_t min_val = episodic_syncmer_ring_[0];
        int32_t min_idx = 0;
        for (int32_t i = 1; i < window; ++i) {
            if (episodic_syncmer_ring_[i] < min_val) { min_val = episodic_syncmer_ring_[i]; min_idx = i; }
        }
        const int32_t last_idx = (episodic_syncmer_head_ - 1 + window) % window;
        return min_idx == episodic_syncmer_head_ || min_idx == last_idx;
    }

    // Advances the k-mer/s-mer rolling hashes and syncmer ring by one byte.
    // Call AFTER using the pre-advance state for lookup/write, exactly
    // mirroring advance_deep_band_history's own before/after contract.
    void advance_episodic(uint8_t target) {
        const int32_t k = config_.episodic_k;
        if (episodic_kmer_filled_ < k) {
            episodic_kmer_hash_ = episodic_kmer_hash_ * kDeepBandHashBase + (static_cast<uint64_t>(target) + 1);
            episodic_kmer_history_[episodic_kmer_pos_] = target;
            episodic_kmer_pos_ = (episodic_kmer_pos_ + 1) % k;
            ++episodic_kmer_filled_;
        } else {
            const uint8_t outgoing = episodic_kmer_history_[episodic_kmer_pos_];
            episodic_kmer_hash_ = (episodic_kmer_hash_ -
                                   (static_cast<uint64_t>(outgoing) + 1) * episodic_kmer_pow_) *
                                      kDeepBandHashBase +
                                  (static_cast<uint64_t>(target) + 1);
            episodic_kmer_history_[episodic_kmer_pos_] = target;
            episodic_kmer_pos_ = (episodic_kmer_pos_ + 1) % k;
        }
        const int32_t s = config_.episodic_s;
        if (episodic_smer_filled_ < s) {
            episodic_smer_hash_ = episodic_smer_hash_ * kDeepBandHashBase + (static_cast<uint64_t>(target) + 1);
            episodic_smer_history_[episodic_smer_pos_] = target;
            episodic_smer_pos_ = (episodic_smer_pos_ + 1) % s;
            ++episodic_smer_filled_;
        } else {
            const uint8_t outgoing = episodic_smer_history_[episodic_smer_pos_];
            episodic_smer_hash_ = (episodic_smer_hash_ -
                                   (static_cast<uint64_t>(outgoing) + 1) * episodic_smer_pow_) *
                                      kDeepBandHashBase +
                                  (static_cast<uint64_t>(target) + 1);
            episodic_smer_history_[episodic_smer_pos_] = target;
            episodic_smer_pos_ = (episodic_smer_pos_ + 1) % s;
        }
        if (episodic_smer_filled_ >= s) {
            const int32_t window = k - s + 1;
            episodic_syncmer_ring_[episodic_syncmer_head_] = episodic_smer_hash_;
            episodic_syncmer_head_ = (episodic_syncmer_head_ + 1) % window;
            if (episodic_syncmer_count_ < window) ++episodic_syncmer_count_;
        }
        ++episodic_steps_;
    }

    // Recomputes anchor_gate_kmer_pow_/anchor_gate_smer_pow_ for the
    // current anchor_gate_k_/anchor_gate_s_. Called from the constructor
    // (default width) and from set_debug_deep_band_anchor_gate whenever a
    // different width is requested.
    void recompute_anchor_gate_pow() {
        uint64_t pow = 1;
        for (int32_t i = 1; i < anchor_gate_k_; ++i) pow *= kDeepBandHashBase;
        anchor_gate_kmer_pow_ = pow;
        uint64_t spow = 1;
        for (int32_t i = 1; i < anchor_gate_s_; ++i) spow *= kDeepBandHashBase;
        anchor_gate_smer_pow_ = spow;
    }

    // Ablation-only: same closed-syncmer test as episodic_current_is_syncmer,
    // at the separate, runtime-configurable anchor_gate_k_/s_ width (see the
    // field comments above).
    bool anchor_gate_current_is_syncmer() const noexcept {
        if (anchor_gate_kmer_filled_ < anchor_gate_k_) return false;
        if (anchor_gate_syncmer_count_ < anchor_gate_window_) return false;
        uint64_t min_val = anchor_gate_syncmer_ring_[0];
        int32_t min_idx = 0;
        for (int32_t i = 1; i < anchor_gate_window_; ++i) {
            if (anchor_gate_syncmer_ring_[i] < min_val) { min_val = anchor_gate_syncmer_ring_[i]; min_idx = i; }
        }
        const int32_t last_idx = (anchor_gate_syncmer_head_ - 1 + anchor_gate_window_) % anchor_gate_window_;
        return min_idx == anchor_gate_syncmer_head_ || min_idx == last_idx;
    }

    // Advances the anchor-gate kmer/smer rolling hashes and syncmer ring by
    // one byte. Must be called from exactly the same sites as
    // advance_deep_band_history (and reset alongside it in clear_history())
    // so its "how many live bytes have I seen" state stays in lockstep --
    // otherwise a repeated window could see the deep-band history intact
    // while this test's own window is offset by however many bytes were
    // skipped, breaking the same-window-same-decision guarantee.
    void advance_anchor_gate(uint8_t target) {
        const int32_t k = anchor_gate_k_;
        if (anchor_gate_kmer_filled_ < k) {
            anchor_gate_kmer_hash_ = anchor_gate_kmer_hash_ * kDeepBandHashBase + (static_cast<uint64_t>(target) + 1);
            anchor_gate_kmer_history_[anchor_gate_kmer_pos_] = target;
            anchor_gate_kmer_pos_ = (anchor_gate_kmer_pos_ + 1) % k;
            ++anchor_gate_kmer_filled_;
        } else {
            const uint8_t outgoing = anchor_gate_kmer_history_[anchor_gate_kmer_pos_];
            anchor_gate_kmer_hash_ = (anchor_gate_kmer_hash_ -
                                     (static_cast<uint64_t>(outgoing) + 1) * anchor_gate_kmer_pow_) *
                                        kDeepBandHashBase +
                                    (static_cast<uint64_t>(target) + 1);
            anchor_gate_kmer_history_[anchor_gate_kmer_pos_] = target;
            anchor_gate_kmer_pos_ = (anchor_gate_kmer_pos_ + 1) % k;
        }
        const int32_t s = anchor_gate_s_;
        if (anchor_gate_smer_filled_ < s) {
            anchor_gate_smer_hash_ = anchor_gate_smer_hash_ * kDeepBandHashBase + (static_cast<uint64_t>(target) + 1);
            anchor_gate_smer_history_[anchor_gate_smer_pos_] = target;
            anchor_gate_smer_pos_ = (anchor_gate_smer_pos_ + 1) % s;
            ++anchor_gate_smer_filled_;
        } else {
            const uint8_t outgoing = anchor_gate_smer_history_[anchor_gate_smer_pos_];
            anchor_gate_smer_hash_ = (anchor_gate_smer_hash_ -
                                     (static_cast<uint64_t>(outgoing) + 1) * anchor_gate_smer_pow_) *
                                        kDeepBandHashBase +
                                    (static_cast<uint64_t>(target) + 1);
            anchor_gate_smer_history_[anchor_gate_smer_pos_] = target;
            anchor_gate_smer_pos_ = (anchor_gate_smer_pos_ + 1) % s;
        }
        if (anchor_gate_smer_filled_ >= s) {
            anchor_gate_syncmer_ring_[anchor_gate_syncmer_head_] = anchor_gate_smer_hash_;
            anchor_gate_syncmer_head_ = (anchor_gate_syncmer_head_ + 1) % anchor_gate_window_;
            if (anchor_gate_syncmer_count_ < anchor_gate_window_) ++anchor_gate_syncmer_count_;
        }
    }

    // Looks up the current (pre-target) anchor if this position is a
    // syncmer. Returns the arena32_ slot holding its byte-count
    // distribution, or IndexMap::kEmpty if absent/not-a-syncmer -- always
    // fills *out_bucket (valid whenever the position IS a syncmer, so the
    // write-side call below never has to recompute mix64 or re-run the
    // syncmer test).
    uint32_t episodic_lookup(uint64_t* out_bucket, bool* out_is_syncmer) const noexcept {
        *out_is_syncmer = episodic_current_is_syncmer();
        if (!*out_is_syncmer) return IndexMap::kEmpty;
        const uint64_t bucket_id = mix64(episodic_kmer_hash_) & episodic_bucket_mask_;
        *out_bucket = bucket_id;
        const EpisodicBucket& b = episodic_buckets_[bucket_id];
        for (int32_t i = 0; i < b.used; ++i) {
            if (b.key[i] == episodic_kmer_hash_) return b.arena_slot[i];
        }
        return IndexMap::kEmpty;
    }

    // Writes/refreshes the current anchor with `target`. Rolling-window
    // semantics: a fresh key gets a new arena32_ row; an existing key gets
    // reinforced AND has its LRU clock reset (a reference resets decay,
    // validated standalone: a write-once entry survived real bucket
    // contention only ~50-60% of the time at a fixed later distance, one
    // refreshed periodically survived ~100%). Crossing
    // episodic_tier3_refresh_threshold refreshes exempts the entry from
    // ordinary eviction (tier3[i] = true) -- validated as a sharp step
    // (56% vs 100% survival at 1 vs 5 refreshes, not a gradual gradient),
    // mirroring how the deep bands' own tiers are a real phase transition,
    // not a continuous dial. Eviction always prefers a non-tier3 slot;
    // only if every slot in the bucket has already been promoted does it
    // fall back to evicting the globally-oldest-touched one.
    void episodic_write(uint64_t bucket_id, uint8_t target) {
        EpisodicBucket& b = episodic_buckets_[bucket_id];
        const int32_t chain_slots = config_.episodic_chain_slots;
        for (int32_t i = 0; i < b.used; ++i) {
            if (b.key[i] == episodic_kmer_hash_) {
                arena32_.increment(b.arena_slot[i], target);
                b.last_touch[i] = episodic_steps_;
                ++b.refresh_count[i];
                if (!b.tier3[i] && b.refresh_count[i] >= config_.episodic_tier3_refresh_threshold) {
                    b.tier3[i] = true;
                }
                return;
            }
        }
        const uint32_t new_slot = arena32_.create_row();
        arena32_.increment(new_slot, target);
        if (b.used < chain_slots) {
            const int32_t i = b.used++;
            b.key[i] = episodic_kmer_hash_; b.arena_slot[i] = new_slot;
            b.last_touch[i] = episodic_steps_; b.refresh_count[i] = 1; b.tier3[i] = false;
        } else {
            int32_t victim = -1;
            for (int32_t i = 0; i < chain_slots; ++i) {
                if (b.tier3[i]) continue;
                if (victim < 0 || b.last_touch[i] < b.last_touch[victim]) victim = i;
            }
            if (victim < 0) {
                victim = 0;
                for (int32_t i = 1; i < chain_slots; ++i) {
                    if (b.last_touch[i] < b.last_touch[victim]) victim = i;
                }
            }
            arena32_.release_row(b.arena_slot[victim]);
            b.key[victim] = episodic_kmer_hash_; b.arena_slot[victim] = new_slot;
            b.last_touch[victim] = episodic_steps_; b.refresh_count[victim] = 1; b.tier3[victim] = false;
        }
    }

    // True once band `b` has seen enough history to form a key; fills
    // `out` for the exact backend only (the rolling-hash backend reads
    // deep_band_hash_[b] directly, no key object needed).
    bool deep_band_key(int32_t b, LongRangeKey& out) const {
        const int32_t depth = config_.long_range_band_depths[b];
        if (deep_band_steps_ < depth) return false;
        out.length = static_cast<uint8_t>(depth);
        const size_t start = deep_band_history_.size() - static_cast<size_t>(depth);
        for (int32_t i = 0; i < depth; ++i) out.bytes[i] = deep_band_history_[start + static_cast<size_t>(i)];
        return true;
    }

    // Blends band `b` into `probability` if it has a mature row, mirroring
    // the single-band blend formula exactly (same Dirichlet-style shape
    // every other bank in this file uses). Returns the slot to reinforce
    // via deep_band_grow, or kEmpty-equivalent (-1) if there is no mature
    // row yet (nothing to reinforce, only a nursery candidate to grow).
    // Split from the combine step below so observe_chunk can fire every
    // band's find() back-to-back before touching any of them -- otherwise
    // band 1's lookup doesn't even issue until band 0's combine math (and
    // its dependent arena32_ reads) fully completes, which left most of
    // the memory-level parallelism prefetching made available unused.
    // Independent of the exact/rolling-hash choice, this is a pure lookup
    // with no side effects, safe to call for all bands up front.
    uint32_t deep_band_mature_lookup(int32_t b, const LongRangeKey& exact_key,
                                     uint64_t hash_key, bool key_ready) const {
        if (!key_ready) return IndexMap::kEmpty;
        if (config_.long_range_rolling_hash) {
            return deep_mature_hash_[b].find(hash_key);
        }
        const auto it = deep_mature_exact_[b].find(exact_key);
        return it != deep_mature_exact_[b].end() ? it->second : IndexMap::kEmpty;
    }

    // The actual blend: cheap arithmetic on an already-resolved slot, plus
    // the genuine sequential dependency (each band's result feeds the next
    // band's `probability` prior) that lookup batching above cannot remove
    // -- but by the time this runs the row data is far more likely already
    // in cache, since every band's find() (and therefore every row's
    // location) was already touched a few instructions earlier.
    int64_t deep_band_combine(uint32_t slot, uint8_t target, double& probability) const {
        if (slot == IndexMap::kEmpty) return -1;
        const int64_t cell = arena32_.get(slot, target);
        const int64_t total = arena32_.total(slot);
        const double strength = config_.long_range_strength;
        probability = (static_cast<double>(cell) + strength * probability) /
                      (static_cast<double>(total) + strength);
        return static_cast<int64_t>(slot);
    }

    // Same blend shape as deep_band_combine, using episodic_strength.
    void episodic_combine(uint32_t slot, uint8_t target, double& probability) const {
        if (slot == IndexMap::kEmpty) return;
        const int64_t cell = arena32_.get(slot, target);
        const int64_t total = arena32_.total(slot);
        const double strength = config_.episodic_strength;
        probability = (static_cast<double>(cell) + strength * probability) /
                      (static_cast<double>(total) + strength);
    }

    // Weight for a nursery increment: single-byte surprise threshold by
    // default (the original mechanism), or -- when
    // long_range_use_combined_salience is set -- span_surprise *
    // log2(2+degree), OR (independently -- either path can fire) Affective
    // Intensity: burst_fraction_ * context_frequency, with sufficiently
    // extreme burst alone bypassing the frequency term entirely (a bear
    // chase has near-zero re-encounter probability but must still be
    // unforgettable). Degree/frequency are only queried when `is_new_key`
    // (a nursery key's first sighting), never on reinforcement of an
    // existing one -- context_degree is O(|bank+1|) per call, and
    // first-sighting events are far rarer than routine reinforcement.
    // Sets `out_ai_triggered` so the caller can enqueue this key for sleep-
    // time replay when AI (not just surprise x degree) is what fired.
    int64_t salience_weight(bool is_new_key, double probability_before, bool& out_ai_triggered) {
        out_ai_triggered = false;
        if (!config_.long_range_use_combined_salience &&
            !config_.long_range_use_affective_intensity) {
            const double surprise =
                -std::log2(probability_before > 1e-300 ? probability_before : 1e-300);
            return surprise >= config_.long_range_importance_bits
                       ? 1 + config_.long_range_importance_bonus
                       : 1;
        }
        if (!is_new_key) return 1;
        bool triggered = false;
        if (config_.long_range_use_combined_salience) {
            const int64_t degree =
                current_cached_context_degree(config_.long_range_salience_degree_bank);
            const double combined =
                span_surprise_sum_ * std::log2(2.0 + static_cast<double>(degree));
            triggered = combined >= config_.long_range_importance_bits;
        }
        if (!triggered && config_.long_range_use_affective_intensity) {
            const double frequency =
                current_context_frequency(config_.long_range_salience_degree_bank);
            const double ai_score = burst_fraction_ * frequency;
            const bool ai_fires = ai_score >= config_.long_range_ai_threshold ||
                                   burst_fraction_ >= config_.long_range_ai_bypass_burst;
            if (ai_fires) {
                triggered = true;
                out_ai_triggered = true;
            }
        }
        return triggered ? 1 + config_.long_range_importance_bonus : 1;
    }

    // Re-applies a growth increment to whatever slot a deep-band key
    // currently occupies (mature or nursery), promoting it if this
    // reinforcement crosses the confirmation threshold -- the sleep-time
    // replay path uses this exactly like a repeated real observation
    // would, since a queued key may not naturally recur again for a long
    // time (if ever).
    // Tier-3 ("consolidated") membership scan for the weakest current
    // member, by accumulated arena32_ count. O(tier size), not O(band
    // size) -- safe because both new tiers are capacity-bounded to a
    // small size by construction (the same reasoning that made
    // refresh_degree_cache necessary for the much bigger ordinary bands).
    bool find_weakest_member_hash(int32_t b, const std::unordered_set<uint64_t>& members,
                                  uint64_t& out_key, uint32_t& out_slot) const {
        bool found = false;
        int64_t weakest_total = 0;
        for (uint64_t key : members) {
            const uint32_t slot = deep_mature_hash_[b].find(key);
            if (slot == IndexMap::kEmpty) continue;
            const int64_t total = arena32_.total(slot);
            if (!found || total < weakest_total) {
                found = true;
                weakest_total = total;
                out_key = key;
                out_slot = slot;
            }
        }
        return found;
    }
    bool find_weakest_member_exact(int32_t b,
                                   const std::unordered_set<LongRangeKey, LongRangeKeyHash>& members,
                                   LongRangeKey& out_key, uint32_t& out_slot) const {
        bool found = false;
        int64_t weakest_total = 0;
        for (const LongRangeKey& key : members) {
            const auto it = deep_mature_exact_[b].find(key);
            if (it == deep_mature_exact_[b].end()) continue;
            const int64_t total = arena32_.total(it->second);
            if (!found || total < weakest_total) {
                found = true;
                weakest_total = total;
                out_key = key;
                out_slot = it->second;
            }
        }
        return found;
    }

    // Mature (tier 2) -> consolidated (tier 3): only reached via a replay
    // hit in reinforce_deep_band_key below, which only ever fires for
    // AI-triggered keys -- so arriving here already implies this key once
    // mattered enough to bypass. If the tier is full, the weakest current
    // member is demoted back to tier 2 (re-enters the ordinary FIFO,
    // eligible for eviction again) rather than refusing the promotion.
    void promote_to_consolidated_hash(int32_t b, uint64_t key) {
        auto& members = deep_consolidated_members_hash_[b];
        if (members.count(key)) return;
        if (static_cast<int64_t>(members.size()) >= config_.long_range_consolidated_max_entries) {
            uint64_t weakest_key;
            uint32_t weakest_slot;
            if (find_weakest_member_hash(b, members, weakest_key, weakest_slot)) {
                members.erase(weakest_key);
                deep_tier_hash_[b].erase(weakest_key);  // absence == tier 2
                deep_mature_order_hash_[b].push_back(weakest_key);
            }
        }
        members.insert(key);
        deep_tier_hash_[b].insert(key, 3);
        ++deep_band_total_consolidations_;
    }
    void promote_to_consolidated_exact(int32_t b, const LongRangeKey& key) {
        auto& members = deep_consolidated_members_exact_[b];
        if (members.count(key)) return;
        if (static_cast<int64_t>(members.size()) >= config_.long_range_consolidated_max_entries) {
            LongRangeKey weakest_key;
            uint32_t weakest_slot;
            if (find_weakest_member_exact(b, members, weakest_key, weakest_slot)) {
                members.erase(weakest_key);
                deep_tier_exact_[b].erase(weakest_key);
                deep_mature_order_exact_[b].push_back(weakest_key);
            }
        }
        members.insert(key);
        deep_tier_exact_[b][key] = 3;
        ++deep_band_total_consolidations_;
    }

    // Consolidated (tier 3) -> permanent (tier 4): reached from ordinary
    // continued recurrence (checked in deep_band_grow's already-mature
    // path), not from a second replay event -- replay only ever fires
    // once per key (at nursery creation), so requiring another one would
    // make tier 4 unreachable in practice. Gating on accumulated count
    // means permanence requires BOTH having mattered once (tier 3) AND
    // recurring enough since to prove it wasn't a one-off.
    void promote_to_permanent_hash(int32_t b, uint64_t key) {
        if (!config_.long_range_allow_permanent_tier) return;
        auto& consolidated = deep_consolidated_members_hash_[b];
        auto& permanent = deep_permanent_members_hash_[b];
        if (permanent.count(key)) return;
        if (static_cast<int64_t>(permanent.size()) >= config_.long_range_permanent_max_entries) {
            uint64_t weakest_key;
            uint32_t weakest_slot;
            if (find_weakest_member_hash(b, permanent, weakest_key, weakest_slot)) {
                permanent.erase(weakest_key);
                deep_tier_hash_[b].insert(weakest_key, 3);  // demote, not evict
                consolidated.insert(weakest_key);
            }
        }
        consolidated.erase(key);
        permanent.insert(key);
        deep_tier_hash_[b].insert(key, 4);
        ++deep_band_total_permanent_promotions_;
    }
    void promote_to_permanent_exact(int32_t b, const LongRangeKey& key) {
        if (!config_.long_range_allow_permanent_tier) return;
        auto& consolidated = deep_consolidated_members_exact_[b];
        auto& permanent = deep_permanent_members_exact_[b];
        if (permanent.count(key)) return;
        if (static_cast<int64_t>(permanent.size()) >= config_.long_range_permanent_max_entries) {
            LongRangeKey weakest_key;
            uint32_t weakest_slot;
            if (find_weakest_member_exact(b, permanent, weakest_key, weakest_slot)) {
                permanent.erase(weakest_key);
                deep_tier_exact_[b][weakest_key] = 3;
                consolidated.insert(weakest_key);
            }
        }
        consolidated.erase(key);
        permanent.insert(key);
        deep_tier_exact_[b][key] = 4;
        ++deep_band_total_permanent_promotions_;
    }

    void maybe_promote_recurrence_hash(int32_t b, uint64_t key, uint32_t slot) {
        if (deep_tier_hash_[b].find(key) != 3) return;  // only tier 3 can earn permanence
        if (arena32_.total(slot) >= config_.long_range_permanent_recurrence_threshold) {
            promote_to_permanent_hash(b, key);
        }
    }
    void maybe_promote_recurrence_exact(int32_t b, const LongRangeKey& key, uint32_t slot) {
        const auto it = deep_tier_exact_[b].find(key);
        if (it == deep_tier_exact_[b].end() || it->second != 3) return;
        if (arena32_.total(slot) >= config_.long_range_permanent_recurrence_threshold) {
            promote_to_permanent_exact(b, key);
        }
    }

    void reinforce_deep_band_key(int32_t b, const LongRangeKey& exact_key, uint64_t hash_key,
                                 uint8_t target, int64_t weight) {
        const int32_t confirmations = config_.long_range_band_confirmations;
        const int64_t max_entries = config_.long_range_band_max_entries;
        if (config_.long_range_rolling_hash) {
            const uint32_t mature_slot = deep_mature_hash_[b].find(hash_key);
            if (mature_slot != IndexMap::kEmpty) {
                arena32_.increment(mature_slot, target, weight);
                if (config_.long_range_use_affective_intensity) {
                    promote_to_consolidated_hash(b, hash_key);
                }
                return;
            }
            InlineNurseryBank& nursery = deep_nursery_hash_[b];
            const uint32_t inline_slot = nursery.map.find(hash_key);
            if (inline_slot == IndexMap::kEmpty) return;  // already evicted -- nothing to replay
            InlineNurseryRow& row = nursery.arena.row(inline_slot);
            const int64_t promote_slot =
                inline_nursery_increment_and_maybe_ready(row, target, weight, confirmations);
            if (promote_slot >= 0) {
                if (static_cast<int64_t>(deep_mature_hash_[b].size()) >= max_entries) {
                    evict_oldest_deep_mature_hash(b);
                }
                deep_mature_hash_[b].insert(hash_key, static_cast<uint32_t>(promote_slot));
                ++deep_band_total_promotions_;
                // Without this, evict_oldest_deep_mature_hash always walks
                // an empty queue and silently no-ops -- the per-band cap
                // above was never actually enforced for the rolling-hash
                // backend (the exact-key backend's equivalent promotion
                // sites already do this correctly).
                deep_mature_order_hash_[b].push_back(hash_key);
                nursery.map.erase(hash_key);
                nursery.arena.release_row(inline_slot);
            }
        } else {
            const auto mature_it = deep_mature_exact_[b].find(exact_key);
            if (mature_it != deep_mature_exact_[b].end()) {
                arena32_.increment(mature_it->second, target, weight);
                if (config_.long_range_use_affective_intensity) {
                    promote_to_consolidated_exact(b, exact_key);
                }
                return;
            }
            auto& nursery = deep_nursery_exact_[b];
            const auto it = nursery.find(exact_key);
            if (it == nursery.end()) return;  // already evicted -- nothing to replay
            const uint32_t slot = it->second;
            arena32_.increment(slot, target, weight);
            if (arena32_.total(slot) >= confirmations) {
                if (static_cast<int64_t>(deep_mature_exact_[b].size()) >= max_entries) {
                    evict_oldest_deep_mature_exact(b);
                }
                deep_mature_exact_[b].emplace(exact_key, slot);
                ++deep_band_total_promotions_;
                deep_mature_order_exact_[b].push_back(exact_key);
                nursery.erase(exact_key);
            }
        }
    }

    void enqueue_replay(int32_t band, const LongRangeKey& exact_key, uint64_t hash_key,
                        uint8_t target) {
        if (static_cast<int32_t>(replay_queue_.size()) >=
            config_.long_range_replay_queue_capacity) {
            replay_queue_.pop_front();
        }
        replay_queue_.push_back(ReplayCandidate{band, exact_key, hash_key, target});
    }

    // Applies `weight` to `target`'s count in an inline nursery row,
    // transparently overflowing to a real arena32_ row if it would need a
    // 5th distinct successor, and materializing one too if confirmations
    // is reached while still inline (mature entries always live in
    // arena32_ -- the hot blend/grow path already assumes that for every
    // mature lookup, so inline storage is nursery-only). Returns the
    // arena32_ slot to promote into if ready this call, else -1. Never
    // mutates `row` in the "reached confirmations while inline" case --
    // the caller discards the inline row entirely on promotion.
    int64_t inline_nursery_increment_and_maybe_ready(InlineNurseryRow& row, uint8_t target,
                                                      int64_t weight, int32_t confirmations) {
        constexpr int64_t kMax16 = std::numeric_limits<int16_t>::max();
        if (row.overflowed()) {
            arena32_.increment(row.overflow_slot, target, weight);
            return arena32_.total(row.overflow_slot) >= confirmations
                       ? static_cast<int64_t>(row.overflow_slot)
                       : -1;
        }
        int32_t found = -1;
        for (int32_t i = 0; i < row.occupied; ++i) {
            if (row.keys[i] == target) {
                found = i;
                break;
            }
        }
        if (found >= 0) {
            const int64_t updated = static_cast<int64_t>(row.counts[found]) + weight;
            row.counts[found] = static_cast<int16_t>(std::min<int64_t>(updated, kMax16));
        } else if (row.occupied < InlineNurseryRow::kCapacity) {
            row.keys[row.occupied] = target;
            row.counts[row.occupied] = static_cast<int16_t>(std::min<int64_t>(weight, kMax16));
            ++row.occupied;
        } else {
            // This row already has kCapacity distinct successors and just
            // saw a (kCapacity+1)-th: overflow to arena32_, preserving
            // every inline count exactly, then apply this increment.
            const uint32_t new_slot = arena32_.create_row();
            for (int32_t i = 0; i < row.occupied; ++i) {
                arena32_.increment(new_slot, row.keys[i], row.counts[i]);
            }
            arena32_.increment(new_slot, target, weight);
            row.overflow_slot = new_slot;
            row.occupied = 0;
            return arena32_.total(new_slot) >= confirmations ? static_cast<int64_t>(new_slot) : -1;
        }
        int64_t total = 0;
        for (int32_t i = 0; i < row.occupied; ++i) total += row.counts[i];
        if (total < confirmations) return -1;
        const uint32_t new_slot = arena32_.create_row();
        for (int32_t i = 0; i < row.occupied; ++i) {
            arena32_.increment(new_slot, row.keys[i], row.counts[i]);
        }
        return static_cast<int64_t>(new_slot);
    }

    // Growth/promotion for band `b`: nursery-staged, same
    // "arena total >= confirmations" contract the ordinary sequence banks
    // use, with an importance-bonus weight on surprising nursery
    // increments only (never on reinforcing an already-mature row -- the
    // same reason long_range_gate_on_burst only gates creation).
    void deep_band_grow(int32_t b, const LongRangeKey& exact_key, uint64_t hash_key,
                        bool key_ready, uint8_t target, double probability_before,
                        int64_t mature_slot) {
        if (!key_ready) return;
        if (mature_slot >= 0) {
            const uint32_t slot = static_cast<uint32_t>(mature_slot);
            arena32_.increment(slot, target);
            if (config_.long_range_use_affective_intensity) {
                if (config_.long_range_rolling_hash) {
                    maybe_promote_recurrence_hash(b, hash_key, slot);
                } else {
                    maybe_promote_recurrence_exact(b, exact_key, slot);
                }
            }
            return;
        }
        const int32_t confirmations = config_.long_range_band_confirmations;
        const int64_t max_entries = config_.long_range_band_max_entries;
        if (config_.long_range_rolling_hash) {
            InlineNurseryBank& nursery = deep_nursery_hash_[b];
            uint32_t inline_slot = nursery.map.find(hash_key);
            const bool is_new_key = inline_slot == IndexMap::kEmpty;
            if (is_new_key) ++deep_band_new_rows_; else ++deep_band_reused_rows_;
            bool ai_triggered = false;
            const int64_t weight = salience_weight(is_new_key, probability_before, ai_triggered);
            if (ai_triggered) enqueue_replay(b, exact_key, hash_key, target);
            if (is_new_key) {
                if (static_cast<int64_t>(nursery.map.size()) >= config_.max_sequence_nursery_paths &&
                    !nursery.empty_order()) {
                    const uint64_t victim = nursery.oldest();
                    const uint32_t victim_slot = nursery.map.find(victim);
                    if (victim_slot != IndexMap::kEmpty && nursery.map.erase(victim)) {
                        const InlineNurseryRow& victim_row = nursery.arena.row(victim_slot);
                        if (victim_row.overflowed()) arena32_.release_row(victim_row.overflow_slot);
                        nursery.arena.release_row(victim_slot);
                        nursery.drop_front();
                    }
                }
                inline_slot = nursery.arena.create_row();
                nursery.map.insert(hash_key, inline_slot);
                nursery.note_insertion(hash_key);
            }
            InlineNurseryRow& row = nursery.arena.row(inline_slot);
            const int64_t promote_slot =
                inline_nursery_increment_and_maybe_ready(row, target, weight, confirmations);
            if (promote_slot >= 0) {
                if (static_cast<int64_t>(deep_mature_hash_[b].size()) >= max_entries) {
                    evict_oldest_deep_mature_hash(b);
                }
                deep_mature_hash_[b].insert(hash_key, static_cast<uint32_t>(promote_slot));
                ++deep_band_total_promotions_;
                // Without this, evict_oldest_deep_mature_hash always walks
                // an empty queue and silently no-ops -- the per-band cap
                // above was never actually enforced for the rolling-hash
                // backend (the exact-key backend's equivalent promotion
                // sites already do this correctly).
                deep_mature_order_hash_[b].push_back(hash_key);
                nursery.map.erase(hash_key);
                // Safe regardless of whether the row was inline or already
                // overflowed: InlineNurseryArena's free-list only manages
                // InlineNurseryRow objects, never arena32_ -- an
                // overflow_slot just transferred to deep_mature_hash_ above
                // is untouched by this release.
                nursery.arena.release_row(inline_slot);
            }
        } else {
            auto& nursery = deep_nursery_exact_[b];
            auto& nursery_order = deep_nursery_order_exact_[b];
            auto it = nursery.find(exact_key);
            const bool is_new_key = it == nursery.end();
            bool ai_triggered = false;
            const int64_t weight = salience_weight(is_new_key, probability_before, ai_triggered);
            if (ai_triggered) enqueue_replay(b, exact_key, hash_key, target);
            uint32_t slot;
            if (is_new_key) {
                if (static_cast<int64_t>(nursery.size()) >= config_.max_sequence_nursery_paths) {
                    evict_oldest_deep_nursery_exact(b);
                }
                slot = arena32_.create_row();
                nursery.emplace(exact_key, slot);
                nursery_order.push_back(exact_key);
            } else {
                slot = it->second;
            }
            arena32_.increment(slot, target, weight);
            if (arena32_.total(slot) >= confirmations) {
                if (static_cast<int64_t>(deep_mature_exact_[b].size()) >= max_entries) {
                    evict_oldest_deep_mature_exact(b);
                }
                deep_mature_exact_[b].emplace(exact_key, slot);
                ++deep_band_total_promotions_;
                deep_mature_order_exact_[b].push_back(exact_key);
                nursery.erase(exact_key);
            }
        }
    }

    void evict_oldest_deep_mature_hash(int32_t b) {
        auto& order = deep_mature_order_hash_[b];
        auto& mature = deep_mature_hash_[b];
        auto& tier = deep_tier_hash_[b];
        while (!order.empty()) {
            const uint64_t victim = order.front();
            order.pop_front();
            // Promoted past tier 2 since this FIFO entry was pushed --
            // stale reference, not evictable here (see promote_to_*_hash,
            // which re-pushes an entry onto this queue only when it is
            // actually demoted back to tier 2).
            if (tier.find(victim) != IndexMap::kEmpty) continue;
            const uint32_t slot = mature.find(victim);
            if (slot != IndexMap::kEmpty) {
                arena32_.release_row(slot);
                mature.erase(victim);
                return;
            }
        }
    }

    void evict_oldest_deep_mature_exact(int32_t b) {
        auto& order = deep_mature_order_exact_[b];
        auto& mature = deep_mature_exact_[b];
        auto& tier = deep_tier_exact_[b];
        while (!order.empty()) {
            const LongRangeKey victim = order.front();
            order.pop_front();
            if (tier.find(victim) != tier.end()) continue;  // stale -- promoted since
            const auto it = mature.find(victim);
            if (it != mature.end()) {
                arena32_.release_row(it->second);
                mature.erase(it);
                return;
            }
        }
    }

    void evict_oldest_deep_nursery_exact(int32_t b) {
        auto& order = deep_nursery_order_exact_[b];
        auto& nursery = deep_nursery_exact_[b];
        while (!order.empty()) {
            const LongRangeKey victim = order.front();
            order.pop_front();
            const auto it = nursery.find(victim);
            if (it != nursery.end()) {
                arena32_.release_row(it->second);
                nursery.erase(it);
                return;
            }
        }
    }

    // Byte-budget-pressure eviction (called from CortexSleep::
    // consolidate_to_byte_budget, sleep.hpp, only once the ordinary
    // sequence banks have nothing left to prune). Deep bands previously
    // had no budget-aware eviction at all -- only the fixed per-band
    // entry-count caps above, which measurement showed can still
    // correspond to many hundreds of MB to GB-scale serialized data on
    // real diverse text, well past max_body_bytes, leaving sleep with
    // nowhere left to turn. Priority mirrors the tiering ladder's own
    // durability intent: nursery (provisional) sheds first, then tier 2
    // (ordinary FIFO mature), then tier 3 (consolidated), and only as a
    // last resort tier 4 (permanent) -- budget pressure erodes the least
    // durable memory first, same principle the tiers exist to encode.
    int64_t deep_band_total_entries() const {
        if (!config_.long_range_bands) return 0;
        int64_t total = 0;
        const int32_t band_count = config_.long_range_band_count;
        for (int32_t b = 0; b < band_count; ++b) {
            if (config_.long_range_rolling_hash) {
                total += static_cast<int64_t>(deep_mature_hash_[b].size());
                total += static_cast<int64_t>(deep_nursery_hash_[b].map.size());
            } else {
                total += static_cast<int64_t>(deep_mature_exact_[b].size());
                total += static_cast<int64_t>(deep_nursery_exact_[b].size());
            }
        }
        return total;
    }

    bool shed_one_deep_nursery() {
        const int32_t band_count = config_.long_range_bands ? config_.long_range_band_count : 0;
        for (int32_t b = 0; b < band_count; ++b) {
            if (config_.long_range_rolling_hash) {
                InlineNurseryBank& nursery = deep_nursery_hash_[b];
                if (nursery.map.size() == 0) continue;
                const uint64_t victim = nursery.oldest();
                const uint32_t inline_slot = nursery.map.find(victim);
                if (inline_slot != IndexMap::kEmpty && nursery.map.erase(victim)) {
                    const InlineNurseryRow& row = nursery.arena.row(inline_slot);
                    if (row.overflowed()) arena32_.release_row(row.overflow_slot);
                    nursery.arena.release_row(inline_slot);
                    nursery.drop_front();
                    return true;
                }
            } else {
                auto& nursery = deep_nursery_exact_[b];
                auto& order = deep_nursery_order_exact_[b];
                while (!order.empty()) {
                    const LongRangeKey victim = order.front();
                    order.pop_front();
                    const auto it = nursery.find(victim);
                    if (it != nursery.end()) {
                        arena32_.release_row(it->second);
                        nursery.erase(it);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // tier == 2 reuses the existing oldest-first evictors (which already
    // skip stale/promoted entries correctly); tier == 3 or 4 does a full
    // removal, not the tiering system's own cap-driven demotion (which
    // only reclassifies a slot back to tier 2 -- budget pressure needs to
    // actually free the row).
    bool evict_weakest_deep_mature_tier(int32_t tier) {
        const int32_t band_count = config_.long_range_bands ? config_.long_range_band_count : 0;
        for (int32_t b = 0; b < band_count; ++b) {
            if (tier == 2) {
                const size_t before = config_.long_range_rolling_hash
                                           ? deep_mature_hash_[b].size()
                                           : deep_mature_exact_[b].size();
                if (before == 0) continue;
                if (config_.long_range_rolling_hash) {
                    evict_oldest_deep_mature_hash(b);
                    if (deep_mature_hash_[b].size() < before) return true;
                } else {
                    evict_oldest_deep_mature_exact(b);
                    if (deep_mature_exact_[b].size() < before) return true;
                }
                continue;
            }
            if (config_.long_range_rolling_hash) {
                auto& members = tier == 3 ? deep_consolidated_members_hash_[b]
                                          : deep_permanent_members_hash_[b];
                uint64_t weakest_key;
                uint32_t weakest_slot;
                if (!find_weakest_member_hash(b, members, weakest_key, weakest_slot)) continue;
                members.erase(weakest_key);
                deep_tier_hash_[b].erase(weakest_key);
                deep_mature_hash_[b].erase(weakest_key);
                arena32_.release_row(weakest_slot);
                return true;
            } else {
                auto& members = tier == 3 ? deep_consolidated_members_exact_[b]
                                          : deep_permanent_members_exact_[b];
                LongRangeKey weakest_key;
                uint32_t weakest_slot;
                if (!find_weakest_member_exact(b, members, weakest_key, weakest_slot)) continue;
                members.erase(weakest_key);
                deep_tier_exact_[b].erase(weakest_key);
                deep_mature_exact_[b].erase(weakest_key);
                arena32_.release_row(weakest_slot);
                return true;
            }
        }
        return false;
    }

    // Tries the least-durable population first; false only when deep-band
    // memory is genuinely empty across every band and every tier.
    bool evict_one_deep_band_entry() {
        if (shed_one_deep_nursery()) return true;
        if (evict_weakest_deep_mature_tier(2)) return true;
        if (evict_weakest_deep_mature_tier(3)) return true;
        if (evict_weakest_deep_mature_tier(4)) return true;
        return false;
    }

    void grow_context_nursery(uint64_t key, uint8_t target, int32_t confirmations) {
        uint32_t slot = context_nursery_.find(key);
        if (slot == IndexMap::kEmpty) {
            slot = arena_.create_row();
            context_nursery_.insert(key, slot);
        }
        arena_.increment(slot, target);
        if (arena_.total(slot) >= confirmations) {
            residual_map_.insert(key, slot);
            context_nursery_.erase(key);
            ++context_births_;
        }
    }

    // Mixed-precision dispatch: the deepest ordinary sequence bank (index
    // bank_count_-1) lives in arena_deep_bank_ (int32, measured safe --
    // see that member's declaration comment); every other bank stays on
    // the wide arena_ (int64). Nursery rows for that bank must dispatch
    // the same way as mature rows: a slot index is only valid in the
    // arena that actually created it, so a promoted nursery row has to
    // already live in the arena its bank's mature map will read from.
    bool is_deep_bank(int32_t bank) const noexcept {
        return bank_count_ > 0 && bank == bank_count_ - 1;
    }
    int64_t bank_total(int32_t bank, uint32_t slot) const noexcept {
        return is_deep_bank(bank) ? arena_deep_bank_.total(slot) : arena_.total(slot);
    }
    int64_t bank_get(int32_t bank, uint32_t slot, uint8_t key) const noexcept {
        return is_deep_bank(bank) ? arena_deep_bank_.get(slot, key) : arena_.get(slot, key);
    }
    void bank_increment(int32_t bank, uint32_t slot, uint8_t key, int64_t amount = 1) {
        if (is_deep_bank(bank)) arena_deep_bank_.increment(slot, key, amount);
        else arena_.increment(slot, key, amount);
    }
    uint32_t bank_create_row(int32_t bank) {
        return is_deep_bank(bank) ? arena_deep_bank_.create_row() : arena_.create_row();
    }
    void bank_release_row(int32_t bank, uint32_t slot) {
        if (is_deep_bank(bank)) arena_deep_bank_.release_row(slot);
        else arena_.release_row(slot);
    }
    template <typename Fn>
    void bank_for_each(int32_t bank, uint32_t slot, Fn&& fn) const {
        if (is_deep_bank(bank)) arena_deep_bank_.for_each(slot, std::forward<Fn>(fn));
        else arena_.for_each(slot, std::forward<Fn>(fn));
    }

    void grow_sequence_nursery(int32_t bank, uint64_t context, uint8_t target,
                               int32_t confirmations, int64_t max_nursery) {
        NurseryBank& nursery = sequence_nursery_[bank];
        uint32_t slot = nursery.map.find(context);
        if (slot == IndexMap::kEmpty) {
            if (static_cast<int64_t>(nursery.map.size()) >= max_nursery &&
                !nursery.empty_order()) {
                const uint64_t victim = nursery.oldest();
                const uint32_t victim_slot = nursery.map.find(victim);
                if (victim_slot != IndexMap::kEmpty && nursery.map.erase(victim)) {
                    bank_release_row(bank, victim_slot);
                    nursery.drop_front();
                    ++sequence_evictions_;
                }
            }
            slot = bank_create_row(bank);
            nursery.map.insert(context, slot);
            nursery.note_insertion(context);
        }
        bank_increment(bank, slot, target);
        if (bank_total(bank, slot) >= confirmations) {
            sequence_maps_[bank].insert(context, slot);
            nursery.map.erase(context);
            ++sequence_births_;
        }
    }

    // The updates a newly born assembly missed while its probability was read.
    void catch_up(uint8_t target, double basal, bool has_context, uint64_t context_key,
                  const uint64_t* contexts, int32_t active_banks) {
        if (has_context) {
            if (tracking_) touched_contexts_.insert(context_key);
            const uint32_t slot = residual_map_.find(context_key);
            if (slot != IndexMap::kEmpty) {
                arena_.increment(slot, target);
            } else if (-std::log2(basal > 1e-300 ? basal : 1e-300) >=
                       config_.surprise_bits) {
                grow_context_nursery(context_key, target, config_.context_confirmations);
            }
        }
        unigram_counts_[target] += 1;
        unigram_sum_ += 1;
        if (previous_ >= 0) {
            arena_.increment(transition_rows_[previous_], target);
            context_totals_[previous_] += 1;
        }
        for (int32_t bank = 0; bank < active_banks; ++bank) {
            const uint64_t context = contexts[bank];
            if (tracking_) touched_sequences_[bank].insert(context);
            const uint32_t slot = sequence_maps_[bank].find(context);
            if (slot != IndexMap::kEmpty) {
                bank_increment(bank, slot, target);
                continue;
            }
            grow_sequence_nursery(bank, context, target, config_.sequence_confirmations,
                                  config_.max_sequence_nursery_paths);
        }
    }

    bool develop(uint8_t byte_id) {
        if (live_assemblies_[byte_id]) return true;
        const int32_t confirmations = config_.confirmations;
        int32_t pair_states[4];
        for (int32_t position = 0; position < 4; ++position) {
            pair_states[position] = (byte_id >> (2 * position)) & 3;
            const int32_t cell = position * 4 + pair_states[position];
            if (!pair_live_[cell]) {
                if (++pair_confirmations_[cell] >= confirmations) pair_live_[cell] = 1;
            }
        }
        const int32_t quartet_states[2] = {pair_states[0] + 4 * pair_states[1],
                                           pair_states[2] + 4 * pair_states[3]};
        for (int32_t position = 0; position < 2; ++position) {
            const int32_t pair_offset = 2 * position;
            const int32_t cell = position * 16 + quartet_states[position];
            if (pair_live_[pair_offset * 4 + pair_states[pair_offset]] &&
                pair_live_[(pair_offset + 1) * 4 + pair_states[pair_offset + 1]] &&
                !quartet_live_[cell]) {
                if (++quartet_confirmations_[cell] >= confirmations)
                    quartet_live_[cell] = 1;
            }
        }
        const int32_t assembly = quartet_states[0] + 16 * quartet_states[1];
        if (quartet_live_[quartet_states[0]] &&
            quartet_live_[16 + quartet_states[1]] && !live_assemblies_[assembly]) {
            if (++assembly_confirmations_[assembly] >= confirmations) {
                live_assemblies_[assembly] = 1;
                ++assembly_births_;
            }
        }
        return live_assemblies_[assembly] != 0;
    }

    CortexConfig config_;
    TrustState trust_;
    RowArena<int64_t> arena_;
    // Mixed-precision: deep bands only (measured safe -- see
    // max_sequence_bank_count/max_count_value in cortex.hpp and the
    // per-bank measurement this was based on). NOT used for the ordinary
    // sequence banks except the deepest (see arena_deep_bank_ below):
    // bank 0 alone was already at ~49% of int32 range after 2 production
    // epochs, so banks 0..bank_count_-2 stay on the wide arena.
    // Empirically validated (not assumed): real wiki text pushed deep-band
    // counts up to 2,336 within 32MB alone -- already 18x past int8_t's
    // 127 ceiling, so int8 is unsafe. int16_t (32,767 ceiling) held ~14x
    // headroom at that same measurement point, with growth visibly
    // plateauing as FIFO eviction caps how long any one entry accumulates
    // before being evicted -- the better footprint/safety tradeoff of the
    // three widths actually measured.
    RowArena<int16_t> arena32_;
    // The deepest ordinary sequence bank (index bank_count_-1, depth
    // max_context_bytes): measured on a real production checkpoint at
    // max count 30,307,634 -- 1.4% of int32 range, comfortably safe, and
    // this bank alone holds 53% of all sequence rows (1.29M of 2.43M),
    // so it's both the safest AND the largest single memory win among the
    // ordinary banks. int16 is NOT safe here (that same 30.3M measured max
    // is 925x past int16's 32,767 ceiling) -- this is int32 only.
    RowArena<int32_t> arena_deep_bank_;
    IndexMap residual_map_;
    IndexMap context_nursery_;
    std::vector<IndexMap> sequence_maps_;
    std::vector<NurseryBank> sequence_nursery_;
    std::vector<uint64_t> context_masks_;
    std::vector<uint32_t> transition_rows_;
    std::vector<int64_t> unigram_counts_;
    std::vector<int64_t> context_totals_;
    std::vector<uint8_t> live_assemblies_;
    std::vector<uint16_t> assembly_confirmations_;
    std::vector<uint16_t> pair_confirmations_;
    std::vector<uint8_t> pair_live_;
    std::vector<uint16_t> quartet_confirmations_;
    std::vector<uint8_t> quartet_live_;

    uint64_t history_mask_ = 0;
    uint64_t packed_history_ = 0;
    int32_t history_length_ = 0;
    int32_t bank_count_ = 0;
    int32_t previous_ = -1;
    int32_t older_ = -1;
    int64_t steps_ = 0;
    int64_t unigram_sum_ = 0;
    int64_t context_births_ = 0;
    int64_t sequence_births_ = 0;
    int64_t sequence_evictions_ = 0;
    int64_t assembly_births_ = 0;

    bool tracking_ = false;
    std::unordered_set<uint64_t> touched_contexts_;
    std::vector<std::unordered_set<uint64_t>> touched_sequences_;

    // Idea 1 (burst_surprise_signal): EMA of "did anything just get born".
    double burst_fraction_ = 0.0;
    int64_t burst_last_birth_total_ = 0;

    // Idea 2 (long_range_bank): a second context bank on far more trailing
    // bytes than the ordinary sequence banks reach, using the same arena
    // row abstraction (key -> row of 256-way counts) those banks use.
    // Real byte-string keys, not a hashed/packed integer: unlike the
    // ordinary banks (capped at 8 bytes so they fit a uint64_t),
    // long_range_context_bytes has no such cap, and Python's exact tuple
    // keys have no collision risk to reproduce unfaithfully.
    std::unordered_map<LongRangeKey, uint32_t, LongRangeKeyHash> long_range_map_;
    std::deque<uint8_t> long_range_history_;
    std::deque<LongRangeKey> long_range_order_;  // FIFO eviction queue

    // Idea 2 continued: sparse multi-depth long-range bands
    // (long_range_bands), additive alongside long_range_bank above.
    static constexpr uint64_t kDeepBandHashBase = 1099511628211ULL;  // FNV prime
    std::deque<uint8_t> deep_band_history_;  // shared raw bytes, sized to max active depth
    int64_t deep_band_steps_ = 0;
    uint64_t deep_band_hash_[CortexConfig::kMaxLongRangeBands] = {};
    uint64_t deep_band_pow_[CortexConfig::kMaxLongRangeBands] = {};  // base^(depth-1) per band

    // Ablation-only anchor gate for deep-band lookup/grow (see
    // set_debug_deep_band_anchor_gate below) -- a SEPARATE, smaller
    // closed-syncmer test from the episodic layer's own (k=14/s=6), which
    // deliberately does NOT share episodic's state or width. episodic_k
    // can exceed a configured band depth (e.g. the default 12-byte band),
    // and if it did, the same matched window could get a DIFFERENT anchor
    // decision at its two occurrences depending on the two bytes outside
    // the 12-byte span that differ between them -- silently dropping real
    // matches instead of just relocating them to a shared anchor offset.
    // Fixed at k=6/s=3 so it fits within any band depth >= 6 (set_debug_
    // deep_band_anchor_gate throws otherwise); mirrors the episodic
    // kmer/smer/syncmer-ring layout exactly, just narrower.
    // k/s are runtime-configurable (see set_debug_deep_band_anchor_gate)
    // within [1, kMaxAnchorGateWidth] so the throughput/quality tradeoff
    // can be swept without rebuilding; arrays are sized to the max width
    // rather than the active k/s.
    static constexpr int32_t kMaxAnchorGateWidth = 16;
    int32_t anchor_gate_k_ = 6;
    int32_t anchor_gate_s_ = 3;
    int32_t anchor_gate_window_ = 4;  // k - s + 1
    uint64_t anchor_gate_kmer_hash_ = 0;
    uint64_t anchor_gate_kmer_pow_ = 0;
    uint8_t anchor_gate_kmer_history_[kMaxAnchorGateWidth] = {};
    int32_t anchor_gate_kmer_pos_ = 0;
    int32_t anchor_gate_kmer_filled_ = 0;
    uint64_t anchor_gate_smer_hash_ = 0;
    uint64_t anchor_gate_smer_pow_ = 0;
    uint8_t anchor_gate_smer_history_[kMaxAnchorGateWidth] = {};
    int32_t anchor_gate_smer_pos_ = 0;
    int32_t anchor_gate_smer_filled_ = 0;
    uint64_t anchor_gate_syncmer_ring_[kMaxAnchorGateWidth] = {};
    int32_t anchor_gate_syncmer_head_ = 0;
    int32_t anchor_gate_syncmer_count_ = 0;
    bool debug_deep_band_anchor_gate_enabled_ = false;

    // Sparse episodic layer (see CortexConfig::episodic_* above). Own
    // rolling-hash state at width episodic_k, independent of the deep
    // bands' depths -- plus a small width-episodic_s ring for closed-
    // syncmer selection (min s-mer hash at either end of its window).
    // Fixed-size array, not std::deque: a large-window (128-512 byte)
    // minimizer using std::deque for the same purpose measured
    // catastrophically slow in the standalone prototype's abandoned first
    // attempt (hours for 500MB, never finished) -- the syncmer window is
    // tiny (k-s+1, typically 9 elements) by design specifically so a
    // linear-scan fixed array stays cheap; do not reintroduce a deque here.
    static constexpr int32_t kMaxEpisodicKmerWidth = 32;
    static constexpr int32_t kMaxEpisodicSyncmerWindow = 32;
    static constexpr int32_t kMaxEpisodicChainSlots = 8;
    uint64_t episodic_kmer_hash_ = 0;
    uint64_t episodic_kmer_pow_ = 0;
    uint8_t episodic_kmer_history_[kMaxEpisodicKmerWidth] = {};
    int32_t episodic_kmer_pos_ = 0;
    int32_t episodic_kmer_filled_ = 0;
    uint64_t episodic_smer_hash_ = 0;
    uint64_t episodic_smer_pow_ = 0;
    uint8_t episodic_smer_history_[kMaxEpisodicKmerWidth] = {};
    int32_t episodic_smer_pos_ = 0;
    int32_t episodic_smer_filled_ = 0;
    uint64_t episodic_syncmer_ring_[kMaxEpisodicSyncmerWindow] = {};
    int32_t episodic_syncmer_head_ = 0;
    int32_t episodic_syncmer_count_ = 0;
    int64_t episodic_steps_ = 0;  // this episode's byte clock, used as the LRU "now"

    struct EpisodicBucket {
        uint64_t key[kMaxEpisodicChainSlots];
        uint32_t arena_slot[kMaxEpisodicChainSlots];
        int64_t last_touch[kMaxEpisodicChainSlots];
        int32_t refresh_count[kMaxEpisodicChainSlots];
        bool tier3[kMaxEpisodicChainSlots] = {};
        int32_t used = 0;
    };
    std::vector<EpisodicBucket> episodic_buckets_;
    uint64_t episodic_bucket_mask_ = 0;

    // Salience-signal prototype: rolling span-surprise window (sum of
    // -log2(final blended probability) over the last
    // long_range_salience_span_bytes bytes), maintained every byte at O(1)
    // amortized cost so deep_band_grow can read "how surprising was this
    // whole recent span" instead of just the current byte.
    std::deque<double> span_surprise_window_;
    double span_surprise_sum_ = 0.0;

    // Affective Intensity: bounded FIFO replay queue, reprocessed once per
    // sleep call (see replay_reinforce()).
    std::deque<ReplayCandidate> replay_queue_;

    // Sleep-refreshed degree cache (see refresh_degree_cache/
    // cached_context_degree above) -- makes combined_salience's per-byte
    // degree lookup O(1) instead of an O(|bank+1|) scan on every
    // first-sighting event, which measurement showed was ruinously slow
    // on real diverse text.
    std::vector<std::unordered_map<uint64_t, int32_t>> cached_degree_;

    // Rolling-hash backend: reuses the same fast IndexMap/NurseryBank the
    // ordinary depth-3..8 banks use, keyed on the incremental hash above.
    std::vector<IndexMap> deep_mature_hash_;
    std::vector<InlineNurseryBank> deep_nursery_hash_;
    std::vector<std::deque<uint64_t>> deep_mature_order_hash_;  // FIFO eviction on mature

    // Exact backend: fixed-width byte keys, no collision risk, capped at
    // 32 bytes per band by LongRangeKey::kMaxBytes.
    std::vector<std::unordered_map<LongRangeKey, uint32_t, LongRangeKeyHash>> deep_mature_exact_;
    std::vector<std::unordered_map<LongRangeKey, uint32_t, LongRangeKeyHash>> deep_nursery_exact_;
    std::vector<std::deque<LongRangeKey>> deep_mature_order_exact_;
    std::vector<std::deque<LongRangeKey>> deep_nursery_order_exact_;

    // Tiered persistence. Absence from deep_tier_hash_/deep_tier_exact_
    // means tier 2 (ordinary mature, the common case) -- only entries
    // that graduated further carry an entry, so cost is proportional to
    // how much actually consolidates, not to mature-map size. The member
    // sets exist so weakest-first eviction within a tier is O(tier size)
    // instead of an O(band size) scan (the same trap refresh_degree_cache
    // fixed for combined_salience -- see cached_degree_ above).
    std::vector<IndexMap> deep_tier_hash_;  // key -> tier (3 or 4)
    std::vector<std::unordered_map<LongRangeKey, uint8_t, LongRangeKeyHash>> deep_tier_exact_;
    std::vector<std::unordered_set<uint64_t>> deep_consolidated_members_hash_;
    std::vector<std::unordered_set<LongRangeKey, LongRangeKeyHash>> deep_consolidated_members_exact_;
    std::vector<std::unordered_set<uint64_t>> deep_permanent_members_hash_;
    std::vector<std::unordered_set<LongRangeKey, LongRangeKeyHash>> deep_permanent_members_exact_;

    // Throughput-investigation instrumentation only: distinguishes paying
    // for a failed lookup alone from paying for a failed lookup plus row
    // allocation/initialization, since those need different fixes (cheap
    // rejection vs. inline storage). Rolling-hash backend only, matching
    // production config; not meant as a permanent per-request API.
public:
    int64_t deep_band_mature_hits_ = 0;
    int64_t deep_band_mature_misses_ = 0;
    int64_t deep_band_new_rows_ = 0;
    int64_t deep_band_reused_rows_ = 0;

    // Cumulative count of nursery->mature promotions ever, across all deep
    // bands -- unlike deep_band_mature_count() (a live snapshot capped at
    // long_range_band_max_entries and reduced by eviction), this never
    // decreases, so it gives a genuine stabilization RATE (promotions per
    // byte/MB processed) even once the live count has saturated its cap.
    // NOTE: tier 2 (mature) is the high-volume, constantly-churning tier
    // (confirmed as the source of generation attractor-collapse) -- this
    // counter measures ordinary mature-tier turnover, NOT the rare
    // "worth a durable episodic anchor" signal; use
    // deep_band_total_consolidations_/deep_band_total_permanent_promotions_
    // below for that.
    int64_t deep_band_total_promotions_ = 0;
    // Cumulative tier 2 -> tier 3 (consolidated) promotions ever.
    int64_t deep_band_total_consolidations_ = 0;
    // Cumulative tier 3 -> tier 4 (permanent) promotions ever -- the
    // rarest tier (single/double digits observed over tens of GB), the
    // closest existing analogue to a genuine stabilization/"worth an
    // episodic anchor" event.
    int64_t deep_band_total_permanent_promotions_ = 0;

    // Ablation-only: when > 0, predict_distribution() treats a deep-band
    // match as a miss (falls through to whatever shallower signal is
    // available) if that key's effective tier (absence from
    // deep_tier_hash_/deep_tier_exact_ == tier 2, the common case) is >=
    // this threshold. 4 blocks only permanent; 3 blocks
    // consolidated+permanent; 2 blocks every deep-band match regardless
    // of tier. Deliberately NOT part of Config/serialize.hpp -- this is a
    // runtime toggle for probing an already-loaded checkpoint (which tier
    // actually explains a dominant attractor response), not a trained
    // property. 0 (default) disables the ablation.
    int32_t debug_ignore_tier_at_or_above_ = 0;

    // Counterfactual-evaluation toggle.  When true, learn=false input may
    // advance the episode history needed for scoring but cannot update the
    // burst EMA or adaptive-trust state.  Not serialized; defaults off.
    bool debug_freeze_inference_adaptation_ = false;

    // Ablation-only: gates deep-band lookup/grow (NOT the O(1) rolling-
    // hash history update, which stays dense every byte) to only fire when
    // the current position is a syncmer anchor under a separate, smaller
    // k/s test than the episodic layer's own -- see the anchor_gate_* field
    // comments for why this can't just reuse episodic's k/s. k/s are
    // swept-over ablation parameters (not part of Config/serialize.hpp,
    // same reasoning as debug_ignore_tier_at_or_above_ above -- a
    // throughput/quality tradeoff knob, not a trained property), defaulting
    // to the validated 6/3 width. Requires every active band depth >= k
    // for the same-window-same-decision guarantee to hold, s in [1, k), and
    // both within [1, kMaxAnchorGateWidth]; throws otherwise rather than
    // silently running unsafe.
    void set_debug_deep_band_anchor_gate(bool enabled, int32_t k = 6, int32_t s = 3) {
        if (enabled) {
            if (k < 1 || k > kMaxAnchorGateWidth) {
                throw std::runtime_error("anchor gate k must be in [1, kMaxAnchorGateWidth]");
            }
            if (s < 1 || s >= k) {
                throw std::runtime_error("anchor gate s must be in [1, k)");
            }
            if (config_.long_range_bands) {
                for (int32_t b = 0; b < config_.long_range_band_count; ++b) {
                    if (config_.long_range_band_depths[b] < k) {
                        throw std::runtime_error(
                            "debug_deep_band_anchor_gate requires every active "
                            "long_range_band_depths entry >= k");
                    }
                }
            }
            anchor_gate_k_ = k;
            anchor_gate_s_ = s;
            anchor_gate_window_ = k - s + 1;
            recompute_anchor_gate_pow();
            anchor_gate_kmer_pos_ = 0;
            anchor_gate_kmer_filled_ = 0;
            anchor_gate_smer_pos_ = 0;
            anchor_gate_smer_filled_ = 0;
            anchor_gate_syncmer_head_ = 0;
            anchor_gate_syncmer_count_ = 0;
        }
        debug_deep_band_anchor_gate_enabled_ = enabled;
    }

    bool debug_deep_band_anchor_gate_enabled() const noexcept {
        return debug_deep_band_anchor_gate_enabled_;
    }

private:
};

}  // namespace alm
