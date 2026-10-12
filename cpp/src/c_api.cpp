#include "alm/c_api.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <new>
#include <utility>

#include "alm/consolidate.hpp"
#include "alm/cortex.hpp"
#include "alm/packed.hpp"
#include "alm/serialize.hpp"
#include "alm/sleep.hpp"

struct AlmCortex {
    explicit AlmCortex(const alm::CortexConfig& config) : cortex(config) {}
    explicit AlmCortex(alm::GrowableCortex&& loaded) : cortex(std::move(loaded)) {}
    alm::GrowableCortex cortex;
};

struct AlmPacked {
    explicit AlmPacked(alm::PackedCortex&& source) : packed(std::move(source)) {}
    alm::PackedCortex packed;
};

namespace {

alm::CortexConfig translate(const AlmConfig* config) {
    alm::CortexConfig out;
    if (!config) return out;
    out.prior = config->prior;
    out.surprise_bits = config->surprise_bits;
    out.basal_strength = config->basal_strength;
    out.parent_strength = config->parent_strength;
    out.confirmations = config->confirmations;
    out.context_confirmations = config->context_confirmations;
    out.sequence_confirmations = config->sequence_confirmations;
    out.max_context_bytes = config->max_context_bytes;
    out.max_sequence_nursery_paths = config->max_sequence_nursery_paths;
    out.competition_exponent = config->competition_exponent;
    out.adaptive_competition = config->adaptive_competition != 0;
    out.trust_gamma = config->trust_gamma;
    out.trust_learning_rate = config->trust_learning_rate;
    out.attribution_gate_threshold = config->attribution_gate_threshold;
    out.trust_delta_clip = config->trust_delta_clip;
    out.trust_floor = config->trust_floor;
    out.trust_ceiling = config->trust_ceiling;
    out.connectivity_pruning = config->connectivity_pruning != 0;
    out.burst_surprise_signal = config->burst_surprise_signal != 0;
    out.burst_ema_rate = config->burst_ema_rate;
    out.long_range_bank = config->long_range_bank != 0;
    out.long_range_context_bytes = config->long_range_context_bytes;
    out.long_range_strength = config->long_range_strength;
    out.long_range_gate_on_burst = config->long_range_gate_on_burst != 0;
    out.long_range_gate_threshold = config->long_range_gate_threshold;
    out.long_range_max_entries = config->long_range_max_entries;
    out.long_range_bands = config->long_range_bands != 0;
    for (int32_t b = 0; b < alm::CortexConfig::kMaxLongRangeBands; ++b) {
        out.long_range_band_depths[b] = config->long_range_band_depths[b];
    }
    out.long_range_band_count = config->long_range_band_count;
    out.long_range_rolling_hash = config->long_range_rolling_hash != 0;
    out.long_range_band_confirmations = config->long_range_band_confirmations;
    out.long_range_importance_bits = config->long_range_importance_bits;
    out.long_range_importance_bonus = config->long_range_importance_bonus;
    out.long_range_band_max_entries = config->long_range_band_max_entries;
    out.long_range_use_combined_salience = config->long_range_use_combined_salience != 0;
    out.long_range_salience_span_bytes = config->long_range_salience_span_bytes;
    out.long_range_salience_degree_bank = config->long_range_salience_degree_bank;
    out.long_range_use_affective_intensity = config->long_range_use_affective_intensity != 0;
    out.long_range_ai_threshold = config->long_range_ai_threshold;
    out.long_range_ai_bypass_burst = config->long_range_ai_bypass_burst;
    out.long_range_replay_queue_capacity = config->long_range_replay_queue_capacity;
    out.long_range_consolidated_max_entries = config->long_range_consolidated_max_entries;
    out.long_range_permanent_max_entries = config->long_range_permanent_max_entries;
    out.long_range_permanent_recurrence_threshold =
        config->long_range_permanent_recurrence_threshold;
    out.long_range_allow_permanent_tier = config->long_range_allow_permanent_tier != 0;
    out.episodic_enabled = config->episodic_enabled != 0;
    out.episodic_k = config->episodic_k;
    out.episodic_s = config->episodic_s;
    out.episodic_bucket_bits = config->episodic_bucket_bits;
    out.episodic_chain_slots = config->episodic_chain_slots;
    out.episodic_tier3_refresh_threshold = config->episodic_tier3_refresh_threshold;
    out.episodic_strength = config->episodic_strength;
    return out;
}

void fill_nan(double* out, size_t count) {
    std::fill(out, out + count, std::numeric_limits<double>::quiet_NaN());
}

}  // namespace

extern "C" {

AlmCortex* alm_cortex_create(const AlmConfig* config) {
    return new (std::nothrow) AlmCortex(translate(config));
}

void alm_cortex_destroy(AlmCortex* cortex) { delete cortex; }

void alm_cortex_observe_chunk(AlmCortex* cortex, const uint8_t* data, size_t count,
                              int learn, double* out) {
    if (!cortex || !data || !out || count == 0) return;
    fill_nan(out, count);
    cortex->cortex.observe_chunk(data, count, learn != 0, out);
}

void alm_cortex_predict_distribution(const AlmCortex* cortex, double* out) {
    if (!cortex || !out) return;
    cortex->cortex.predict_distribution(out);
}

void alm_cortex_reset_episode(AlmCortex* cortex) {
    if (cortex) cortex->cortex.reset_episode();
}

void alm_cortex_begin_lineage_tracking(AlmCortex* cortex) {
    if (cortex) cortex->cortex.begin_lineage_tracking();
}

int64_t alm_cortex_shed_sequence_nursery(AlmCortex* cortex) {
    return cortex ? cortex->cortex.shed_sequence_nursery() : 0;
}

int64_t alm_cortex_stat(const AlmCortex* cortex, int32_t stat) {
    if (!cortex) return 0;
    const alm::GrowableCortex& core = cortex->cortex;
    switch (stat) {
        case ALM_STAT_STEPS: return core.steps();
        case ALM_STAT_LIVE_SEQUENCES: return core.live_sequence_count();
        case ALM_STAT_LIVE_CONTEXTS: return core.live_context_count();
        case ALM_STAT_SEQUENCE_NURSERY: return core.sequence_nursery_count();
        case ALM_STAT_SEQUENCE_BIRTHS: return core.sequence_births();
        case ALM_STAT_CONTEXT_BIRTHS: return core.context_births();
        case ALM_STAT_SEQUENCE_EVICTIONS: return core.sequence_evictions();
        case ALM_STAT_ASSEMBLY_BIRTHS: return core.assembly_births();
        case ALM_STAT_MEMORY_BYTES: return static_cast<int64_t>(core.memory_bytes());
        case ALM_STAT_LONG_RANGE_ENTRIES: return core.long_range_entry_count();
        case ALM_STAT_DEEP_BAND_MATURE: return core.deep_band_mature_count();
        case ALM_STAT_DEEP_BAND_NURSERY: return core.deep_band_nursery_count();
        case ALM_STAT_DEEP_BAND_STEPS: return core.deep_band_steps_debug();
        case ALM_STAT_DEEP_BAND_TOTAL_PROMOTIONS: return core.deep_band_total_promotions_;
        case ALM_STAT_DEEP_BAND_TOTAL_CONSOLIDATIONS: return core.deep_band_total_consolidations_;
        case ALM_STAT_DEEP_BAND_TOTAL_PERMANENT_PROMOTIONS: return core.deep_band_total_permanent_promotions_;
        default: return 0;
    }
}

double alm_cortex_burst_fraction(const AlmCortex* cortex) {
    if (!cortex) return 0.0;
    return cortex->cortex.burst_fraction();
}

int64_t alm_cortex_context_degree(const AlmCortex* cortex, int32_t bank, uint64_t context) {
    if (!cortex) return 0;
    return cortex->cortex.context_degree(bank, context);
}

int64_t alm_cortex_current_context_degree(const AlmCortex* cortex, int32_t bank) {
    if (!cortex) return 0;
    return cortex->cortex.current_context_degree(bank);
}

double alm_cortex_current_context_frequency(const AlmCortex* cortex, int32_t bank) {
    if (!cortex) return 0.0;
    return cortex->cortex.current_context_frequency(bank);
}

void alm_cortex_replay_reinforce(AlmCortex* cortex) {
    if (!cortex) return;
    cortex->cortex.replay_reinforce();
}

int64_t alm_cortex_replay_queue_size(const AlmCortex* cortex) {
    if (!cortex) return 0;
    return cortex->cortex.replay_queue_size();
}

// Throughput-investigation instrumentation only (see cortex.hpp's
// deep_band_mature_hits_ etc.) -- not a permanent diagnostic API.
void alm_cortex_deep_band_stats(const AlmCortex* cortex, int64_t* out4) {
    if (!cortex || !out4) return;
    out4[0] = cortex->cortex.deep_band_mature_hits_;
    out4[1] = cortex->cortex.deep_band_mature_misses_;
    out4[2] = cortex->cortex.deep_band_new_rows_;
    out4[3] = cortex->cortex.deep_band_reused_rows_;
}

int64_t alm_cortex_max_count_value(const AlmCortex* cortex) {
    if (!cortex) return 0;
    return cortex->cortex.max_count_value();
}

int64_t alm_cortex_max_count_value_skip(const AlmCortex* cortex, int32_t skip_rows) {
    if (!cortex) return 0;
    return cortex->cortex.max_count_value(static_cast<uint32_t>(skip_rows));
}

int64_t alm_cortex_max_sequence_bank_count(const AlmCortex* cortex, int32_t bank) {
    if (!cortex) return 0;
    return cortex->cortex.max_sequence_bank_count(bank);
}

int64_t alm_cortex_live_sequence_count_at_bank(const AlmCortex* cortex, int32_t bank) {
    if (!cortex) return 0;
    return cortex->cortex.live_sequence_count_at_bank(bank);
}

int64_t alm_cortex_max_deep_band_count_value(const AlmCortex* cortex) {
    if (!cortex) return 0;
    return cortex->cortex.max_deep_band_count_value();
}

int64_t alm_cortex_estimate_bank_bytes(const AlmCortex* cortex, int32_t bank, int32_t count_bytes) {
    if (!cortex) return 0;
    return cortex->cortex.estimate_bank_bytes(bank, count_bytes);
}

AlmPacked* alm_cortex_pack(const AlmCortex* cortex) {
    if (!cortex) return nullptr;
    return new (std::nothrow)
        AlmPacked(alm::PackedCortex::from_growable(cortex->cortex));
}

void alm_packed_destroy(AlmPacked* packed) { delete packed; }

void alm_packed_score_chunk(AlmPacked* packed, const uint8_t* data, size_t count,
                            double* out) {
    if (!packed || !data || !out || count == 0) return;
    fill_nan(out, count);
    packed->packed.score_chunk(data, count, out);
}

void alm_packed_reset_episode(AlmPacked* packed) {
    if (packed) packed->packed.reset_episode();
}

int64_t alm_packed_stat(const AlmPacked* packed, int32_t stat) {
    if (!packed) return 0;
    const alm::PackedCortex& core = packed->packed;
    switch (stat) {
        case ALM_STAT_STEPS: return core.steps();
        case ALM_STAT_LIVE_SEQUENCES: return core.live_sequence_count();
        case ALM_STAT_LIVE_CONTEXTS: return core.live_context_count();
        case ALM_STAT_MEMORY_BYTES: return static_cast<int64_t>(core.memory_bytes());
        default: return 0;
    }
}

uint8_t* alm_cortex_save(const AlmCortex* cortex, size_t* out_size) {
    if (!cortex || !out_size) return nullptr;
    std::vector<uint8_t> encoded = alm::CortexIO::save(cortex->cortex);
    uint8_t* buffer = static_cast<uint8_t*>(::operator new(encoded.size(), std::nothrow));
    if (!buffer) {
        *out_size = 0;
        return nullptr;
    }
    std::copy(encoded.begin(), encoded.end(), buffer);
    *out_size = encoded.size();
    return buffer;
}

void alm_free_buffer(uint8_t* buffer) { ::operator delete(buffer); }

AlmCortex* alm_cortex_load(const uint8_t* data, size_t size, int32_t* error) {
    if (error) *error = 0;
    if (!data || size == 0) {
        if (error) *error = 1;
        return nullptr;
    }
    try {
        alm::GrowableCortex loaded = alm::CortexIO::load(data, size);
        return new (std::nothrow) AlmCortex(std::move(loaded));
    } catch (const std::exception&) {
        if (error) *error = 1;
        return nullptr;
    }
}

void alm_cortex_set_debug_ignore_tier_at_or_above(AlmCortex* cortex, int32_t threshold) {
    if (!cortex) return;
    cortex->cortex.debug_ignore_tier_at_or_above_ = threshold;
}

void alm_cortex_set_debug_freeze_inference_adaptation(AlmCortex* cortex,
                                                       int32_t freeze) {
    if (!cortex) return;
    cortex->cortex.set_debug_freeze_inference_adaptation(freeze != 0);
}

int64_t alm_cortex_deep_permanent_member_count(const AlmCortex* cortex) {
    if (!cortex) return 0;
    return cortex->cortex.deep_permanent_member_count();
}

void alm_cortex_enable_episodic_layer(AlmCortex* cortex) {
    if (!cortex) return;
    cortex->cortex.enable_episodic_layer();
}

int32_t alm_cortex_configure_episodic_layer(AlmCortex* cortex, int32_t enabled,
                                            int32_t k, int32_t s) {
    if (!cortex) return 1;
    try {
        cortex->cortex.configure_episodic_layer(enabled != 0, k, s);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[alm_cortex_configure_episodic_layer] %s\n", e.what());
        return 1;
    }
}

void alm_cortex_set_trust_dynamics(AlmCortex* cortex, double delta_clip,
                                   double floor, double ceiling) {
    if (!cortex) return;
    cortex->cortex.set_trust_dynamics(delta_clip, floor, ceiling);
}

void alm_cortex_set_long_range_allow_permanent_tier(AlmCortex* cortex, int32_t allow) {
    if (!cortex) return;
    cortex->cortex.set_long_range_allow_permanent_tier(allow != 0);
}

int32_t alm_cortex_set_episodic_strength(AlmCortex* cortex, double strength) {
    if (!cortex) return 1;
    try {
        cortex->cortex.set_episodic_strength(strength);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[alm_cortex_set_episodic_strength] %s\n", e.what());
        return 1;
    }
}

int32_t alm_cortex_set_debug_deep_band_anchor_gate(AlmCortex* cortex, int32_t enabled,
                                                   int32_t k, int32_t s) {
    if (!cortex) return 1;
    try {
        cortex->cortex.set_debug_deep_band_anchor_gate(enabled != 0, k, s);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[alm_cortex_set_debug_deep_band_anchor_gate] %s\n", e.what());
        return 1;
    }
}

int32_t alm_cortex_trust_summary(const AlmCortex* cortex, int64_t* count,
                                 double* min_value, double* max_value,
                                 double* mean_value, int64_t* count_at_floor,
                                 int64_t* count_at_ceiling, int64_t* count_near_neutral) {
    if (!cortex) return 1;
    const auto s = cortex->cortex.trust_summary();
    if (count) *count = s.count;
    if (min_value) *min_value = s.min_value;
    if (max_value) *max_value = s.max_value;
    if (mean_value) *mean_value = s.mean_value;
    if (count_at_floor) *count_at_floor = s.count_at_floor;
    if (count_at_ceiling) *count_at_ceiling = s.count_at_ceiling;
    if (count_near_neutral) *count_near_neutral = s.count_near_neutral;
    return 0;
}

int32_t alm_cortex_consolidate_delta(AlmCortex* self, const AlmCortex* descendant,
                                     const AlmCortex* ancestor) {
    if (!self || !descendant || !ancestor) return 1;
    try {
        alm::CortexConsolidate::merge_delta(self->cortex, descendant->cortex,
                                            ancestor->cortex);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int32_t alm_cortex_consolidate_to_byte_budget(AlmCortex* cortex, int64_t max_bytes,
                                              AlmSleepResult* out_result) {
    if (!cortex) return 1;
    try {
        const alm::SleepResult result =
            alm::CortexSleep::consolidate_to_byte_budget(cortex->cortex, max_bytes);
        if (out_result) {
            out_result->body_bytes_before_sleep = result.body_bytes_before_sleep;
            out_result->body_bytes_after_nursery_shed = result.body_bytes_after_nursery_shed;
            out_result->body_bytes_after_sleep = result.body_bytes_after_sleep;
            out_result->shed_nursery_paths = result.shed_nursery_paths;
            out_result->pruned_mature_paths = result.pruned_mature_paths;
            out_result->competition_passes = result.competition_passes;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[alm_cortex_consolidate_to_byte_budget] %s\n", e.what());
        return 1;
    }
}

int64_t alm_cortex_estimated_bytes(const AlmCortex* cortex) {
    return cortex ? alm::CortexSleep::estimated_bytes(cortex->cortex) : 0;
}

const char* alm_version(void) { return "alm-native 0.1.0"; }

}  // extern "C"
