// Flat C ABI for the ALM cortex, called from Python via ctypes.
//
// The unit of work is a whole chunk (megabytes), so call overhead is
// irrelevant and a plain C ABI avoids coupling the build to any particular
// CPython ABI or binding library.
#ifndef ALM_C_API_H
#define ALM_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define ALM_EXPORT __declspec(dllexport)
#else
#define ALM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AlmCortex AlmCortex;
typedef struct AlmPacked AlmPacked;

typedef struct AlmConfig {
    double prior;
    double surprise_bits;
    double basal_strength;
    double parent_strength;
    int32_t confirmations;
    int32_t context_confirmations;
    int32_t sequence_confirmations;
    int32_t max_context_bytes;
    int64_t max_sequence_nursery_paths;
    double competition_exponent;
    int32_t adaptive_competition;
    double trust_gamma;
    double trust_learning_rate;
    double attribution_gate_threshold;
    double trust_delta_clip;
    double trust_floor;
    double trust_ceiling;
    int32_t connectivity_pruning;
    int32_t burst_surprise_signal;
    double burst_ema_rate;
    int32_t long_range_bank;
    int32_t long_range_context_bytes;
    double long_range_strength;
    int32_t long_range_gate_on_burst;
    double long_range_gate_threshold;
    int64_t long_range_max_entries;
    int32_t long_range_bands;
    int32_t long_range_band_depths[4];
    int32_t long_range_band_count;
    int32_t long_range_rolling_hash;
    int32_t long_range_band_confirmations;
    double long_range_importance_bits;
    int32_t long_range_importance_bonus;
    int64_t long_range_band_max_entries;
    int32_t long_range_use_combined_salience;
    int32_t long_range_salience_span_bytes;
    int32_t long_range_salience_degree_bank;
    int32_t long_range_use_affective_intensity;
    double long_range_ai_threshold;
    double long_range_ai_bypass_burst;
    int32_t long_range_replay_queue_capacity;
    int64_t long_range_consolidated_max_entries;
    int64_t long_range_permanent_max_entries;
    int64_t long_range_permanent_recurrence_threshold;
    int32_t long_range_allow_permanent_tier;
    int32_t episodic_enabled;
    int32_t episodic_k;
    int32_t episodic_s;
    int32_t episodic_bucket_bits;
    int32_t episodic_chain_slots;
    int32_t episodic_tier3_refresh_threshold;
    double episodic_strength;
} AlmConfig;

enum AlmStat {
    ALM_STAT_STEPS = 0,
    ALM_STAT_LIVE_SEQUENCES = 1,
    ALM_STAT_LIVE_CONTEXTS = 2,
    ALM_STAT_SEQUENCE_NURSERY = 3,
    ALM_STAT_SEQUENCE_BIRTHS = 4,
    ALM_STAT_CONTEXT_BIRTHS = 5,
    ALM_STAT_SEQUENCE_EVICTIONS = 6,
    ALM_STAT_ASSEMBLY_BIRTHS = 7,
    ALM_STAT_MEMORY_BYTES = 8,
    ALM_STAT_LONG_RANGE_ENTRIES = 9,
    ALM_STAT_DEEP_BAND_MATURE = 10,
    ALM_STAT_DEEP_BAND_NURSERY = 11,
    ALM_STAT_DEEP_BAND_STEPS = 12,
    // Cumulative nursery->mature promotions ever (never decreases, unlike
    // ALM_STAT_DEEP_BAND_MATURE which is a live, cap-limited snapshot) --
    // use this for a genuine stabilization rate once the live count has
    // saturated long_range_band_max_entries.
    ALM_STAT_DEEP_BAND_TOTAL_PROMOTIONS = 13,
    // Cumulative tier 2->3 and tier 3->4 promotions ever -- see the
    // comment on deep_band_total_consolidations_/
    // deep_band_total_permanent_promotions_ in cortex.hpp for why these,
    // not ALM_STAT_DEEP_BAND_TOTAL_PROMOTIONS, are the closer analogue to
    // a genuine stabilization/"worth an episodic anchor" rate.
    ALM_STAT_DEEP_BAND_TOTAL_CONSOLIDATIONS = 14,
    ALM_STAT_DEEP_BAND_TOTAL_PERMANENT_PROMOTIONS = 15
};

ALM_EXPORT AlmCortex* alm_cortex_create(const AlmConfig* config);
ALM_EXPORT void alm_cortex_destroy(AlmCortex* cortex);

// Writes one probability per input byte. Positions scored before the learner
// has taken a step are left as NaN, matching the Python contract.
ALM_EXPORT void alm_cortex_observe_chunk(AlmCortex* cortex, const uint8_t* data,
                                         size_t count, int learn, double* out);
// Writes exactly 256 probabilities for the current context, with no side
// effects (no learning, no episode advance) -- the generation-time query
// Python's predict_next_distribution() answers.
ALM_EXPORT void alm_cortex_predict_distribution(const AlmCortex* cortex, double* out);
ALM_EXPORT void alm_cortex_reset_episode(AlmCortex* cortex);
ALM_EXPORT void alm_cortex_begin_lineage_tracking(AlmCortex* cortex);
ALM_EXPORT int64_t alm_cortex_shed_sequence_nursery(AlmCortex* cortex);
ALM_EXPORT int64_t alm_cortex_stat(const AlmCortex* cortex, int32_t stat);
ALM_EXPORT double alm_cortex_burst_fraction(const AlmCortex* cortex);

// Salience-signal prototype: how many mature bank+1 rows currently have
// `context` (a bank-depth suffix) as their trailing suffix. On-demand,
// O(|bank+1|) per call -- see GrowableCortex::context_degree in cortex.hpp.
ALM_EXPORT int64_t alm_cortex_context_degree(const AlmCortex* cortex, int32_t bank,
                                             uint64_t context);
// Degree of whatever context is currently active at the given bank depth.
ALM_EXPORT int64_t alm_cortex_current_context_degree(const AlmCortex* cortex, int32_t bank);
ALM_EXPORT double alm_cortex_current_context_frequency(const AlmCortex* cortex, int32_t bank);

// Affective Intensity's replay path: reprocess every queued candidate
// (call once per sleep cycle, before consolidate_to_byte_budget).
ALM_EXPORT void alm_cortex_replay_reinforce(AlmCortex* cortex);
ALM_EXPORT int64_t alm_cortex_replay_queue_size(const AlmCortex* cortex);

// Throughput-investigation instrumentation only -- writes 4 int64_t to
// out4: [mature_hits, mature_misses, new_rows, reused_rows]. Not a
// permanent diagnostic API.
ALM_EXPORT void alm_cortex_deep_band_stats(const AlmCortex* cortex, int64_t* out4);

// Mixed-precision sizing check: largest count/total currently stored.
ALM_EXPORT int64_t alm_cortex_max_count_value(const AlmCortex* cortex);
// Same, excluding the first `skip_rows` rows (e.g. 256 to skip the
// always-dense transition_rows_, a fixed-size high-traffic outlier
// unrepresentative of the millions of far sparser deep-context rows that
// actually dominate total memory). Separate export, not an overload of
// the above, so the two already-built DLL copies in play right now can't
// end up mismatched with native_cortex.py's existing single-arg binding.
ALM_EXPORT int64_t alm_cortex_max_count_value_skip(const AlmCortex* cortex, int32_t skip_rows);
// Max count within just ordinary sequence bank `bank` (depth bank+3) --
// the millions of rows that dominate total memory, isolated from the
// coarse unigram/transition/residual levels.
ALM_EXPORT int64_t alm_cortex_max_sequence_bank_count(const AlmCortex* cortex, int32_t bank);
ALM_EXPORT int64_t alm_cortex_live_sequence_count_at_bank(const AlmCortex* cortex, int32_t bank);
ALM_EXPORT int64_t alm_cortex_max_deep_band_count_value(const AlmCortex* cortex);
// Real per-row-measured byte estimate for bank `bank` at `count_bytes`
// (8/4/2/1) count-storage width -- see estimate_bank_bytes in cortex.hpp.
ALM_EXPORT int64_t alm_cortex_estimate_bank_bytes(const AlmCortex* cortex, int32_t bank,
                                                  int32_t count_bytes);

ALM_EXPORT AlmPacked* alm_cortex_pack(const AlmCortex* cortex);
ALM_EXPORT void alm_packed_destroy(AlmPacked* packed);
ALM_EXPORT void alm_packed_score_chunk(AlmPacked* packed, const uint8_t* data,
                                       size_t count, double* out);
ALM_EXPORT void alm_packed_reset_episode(AlmPacked* packed);
ALM_EXPORT int64_t alm_packed_stat(const AlmPacked* packed, int32_t stat);

// Native binary checkpoint (see alm/serialize.hpp): same-machine, same-build
// only. `alm_cortex_save` allocates the returned buffer; free it with
// `alm_free_buffer`. `alm_cortex_load` returns NULL and leaves `error`
// nonzero on a truncated or foreign buffer.
ALM_EXPORT uint8_t* alm_cortex_save(const AlmCortex* cortex, size_t* out_size);
ALM_EXPORT void alm_free_buffer(uint8_t* buffer);
ALM_EXPORT AlmCortex* alm_cortex_load(const uint8_t* data, size_t size, int32_t* error);

// Ablation-only debug toggle: when enabled, predict_distribution() treats
// a deep-band match as a miss if that key currently carries tier 4
// (permanent) -- for probing whether the permanent tier's eviction-
// immunity (not the trained associations themselves) explains a dominant
// generation attractor. Not persisted by save/load; defaults off.
ALM_EXPORT void alm_cortex_set_debug_ignore_tier_at_or_above(AlmCortex* cortex, int32_t threshold);
ALM_EXPORT int64_t alm_cortex_deep_permanent_member_count(const AlmCortex* cortex);

// Ablation/evaluation-only: when enabled, learn=false traversal advances
// history but does not update the burst EMA or adaptive context-trust state.
// This makes counterfactual branch scoring order-independent. Runtime-only;
// not persisted by save/load and disabled by default.
ALM_EXPORT void alm_cortex_set_debug_freeze_inference_adaptation(
    AlmCortex* cortex, int32_t freeze);

// Turns the sparse episodic layer on for an already-constructed (e.g.
// resumed/deserialized) cortex that was built with episodic_enabled
// false -- lets an in-progress run retrofit episodic memory without
// restarting from scratch. No-op if already enabled. See
// GrowableCortex::enable_episodic_layer in cortex.hpp.
ALM_EXPORT void alm_cortex_enable_episodic_layer(AlmCortex* cortex);

// Reconfigures the episodic layer's k/s width, ALWAYS re-applying it
// (unlike alm_cortex_enable_episodic_layer, a no-op once already enabled).
// episodic_k/s/enabled are not part of the checkpoint format -- a training
// loop that resumes from a checkpoint every epoch must call this every
// epoch too, or the width silently reverts to CortexConfig's compiled
// default after the first. Throws (via the usual exception-to-error-code
// path) if k/s are out of range. See
// GrowableCortex::configure_episodic_layer in cortex.hpp.
ALM_EXPORT int32_t alm_cortex_configure_episodic_layer(AlmCortex* cortex, int32_t enabled,
                                                       int32_t k, int32_t s);

// trust_delta_clip/floor/ceiling and long_range_allow_permanent_tier are
// likewise not part of the checkpoint format -- same reapply-every-epoch
// reasoning as alm_cortex_configure_episodic_layer above.
ALM_EXPORT void alm_cortex_set_trust_dynamics(AlmCortex* cortex, double delta_clip,
                                              double floor, double ceiling);
ALM_EXPORT void alm_cortex_set_long_range_allow_permanent_tier(AlmCortex* cortex, int32_t allow);

// Also not part of the checkpoint format -- same reapply-every-epoch
// reasoning as the setters above, and a real tuning knob: a strength
// sweep at w=2 found a sharp threshold (0% exact-match generation recall
// at the old default 4.0, 100% at every value <=1.0, no throughput cost
// either way). Throws (via the usual exception-to-error-code path) if
// strength <= 0. See GrowableCortex::set_episodic_strength in cortex.hpp.
ALM_EXPORT int32_t alm_cortex_set_episodic_strength(AlmCortex* cortex, double strength);

// Ablation-only: gates deep-band lookup/grow to only fire at syncmer-anchor
// positions (a separate, narrower k/s test than the episodic layer's own,
// default k=6/s=3, swept via the k/s params here), leaving the O(1)
// rolling-hash history update dense every byte. Throws (via the usual
// exception-to-error-code path) if any active band depth is below k, or
// s/k are out of range. Not persisted by save/load; defaults off.
// See GrowableCortex::set_debug_deep_band_anchor_gate in cortex.hpp.
ALM_EXPORT int32_t alm_cortex_set_debug_deep_band_anchor_gate(AlmCortex* cortex, int32_t enabled,
                                                              int32_t k, int32_t s);

// Summary of the learned adaptive-competition trust distribution -- see
// GrowableCortex::trust_summary in cortex.hpp. All out-params optional
// (pass NULL to skip); returns 0 on success, nonzero if cortex is NULL.
ALM_EXPORT int32_t alm_cortex_trust_summary(const AlmCortex* cortex, int64_t* count,
                                            double* min_value, double* max_value,
                                            double* mean_value, int64_t* count_at_floor,
                                            int64_t* count_at_ceiling,
                                            int64_t* count_near_neutral);

// Merges `descendant`'s anatomy past its shared `ancestor` into `self`,
// exactly the way several wake columns cloned from one ancestor are merged
// back together. Returns 0 on success, nonzero (and leaves `self`
// unchanged) if the three cortices have incompatible configuration or the
// descendant does not extend the stated ancestor.
ALM_EXPORT int32_t alm_cortex_consolidate_delta(AlmCortex* self,
                                                const AlmCortex* descendant,
                                                const AlmCortex* ancestor);

typedef struct AlmSleepResult {
    int64_t body_bytes_before_sleep;
    int64_t body_bytes_after_nursery_shed;
    int64_t body_bytes_after_sleep;
    int64_t shed_nursery_paths;
    int64_t pruned_mature_paths;
    int64_t competition_passes;
} AlmSleepResult;

// Native equivalent of consolidate_to_byte_budget: shed nursery anatomy,
// then prune the weakest sequence paths until the body's own serialized
// size (not a Python pickle byte count -- see alm/sleep.hpp) fits
// `max_bytes`. Returns 0 on success.
ALM_EXPORT int32_t alm_cortex_consolidate_to_byte_budget(AlmCortex* cortex,
                                                         int64_t max_bytes,
                                                         AlmSleepResult* out_result);
ALM_EXPORT int64_t alm_cortex_estimated_bytes(const AlmCortex* cortex);

ALM_EXPORT const char* alm_version(void);

#ifdef __cplusplus
}
#endif

#endif  // ALM_C_API_H
