#include "alm/vision_c_api.h"

#include <algorithm>
#include <new>
#include <vector>

#include "alm/vision.hpp"
#include "alm/vision_serialize.hpp"

struct AlmVision {
    explicit AlmVision(const alm::VisionConfig& config) : organism(config) {}
    explicit AlmVision(alm::GrowingVisionOrganism&& loaded) : organism(std::move(loaded)) {}
    alm::GrowingVisionOrganism organism;
};

namespace {

alm::VisionConfig translate(const AlmVisionConfig* config) {
    alm::VisionConfig out;
    if (!config) return out;
    out.classes = config->classes;
    out.image_size = config->image_size;
    out.columns = config->columns;
    out.context_decay = config->context_decay;
    out.sleep_interval = config->sleep_interval;
    out.max_body_bytes = config->max_body_bytes;
    out.learning_rate = config->learning_rate;
    out.maximum_synapses = config->maximum_synapses;
    out.maximum_visual_assemblies = config->maximum_visual_assemblies;
    out.visual_birth_activation = config->visual_birth_activation;
    out.visual_routing_width = config->visual_routing_width;
    out.relation_sketch_words = config->relation_sketch_words;
    out.relation_turnover_fraction = config->relation_turnover_fraction;
    out.episodic_enabled = config->episodic_enabled != 0;
    out.episodic_lsh_bits = config->episodic_lsh_bits;
    out.episodic_probe_radius = config->episodic_probe_radius;
    out.episodic_chain_slots = config->episodic_chain_slots;
    out.episodic_tier3_refresh_threshold = config->episodic_tier3_refresh_threshold;
    out.episodic_strength = config->episodic_strength;
    return out;
}

}  // namespace

extern "C" {

AlmVision* alm_vision_create(const AlmVisionConfig* config) {
    try {
        return new (std::nothrow) AlmVision(translate(config));
    } catch (const std::exception&) {
        return nullptr;
    }
}

void alm_vision_destroy(AlmVision* vision) { delete vision; }

void alm_vision_enable_episodic_layer(AlmVision* vision) {
    if (vision) vision->organism.enable_episodic_layer();
}

int32_t alm_vision_dimensions(const AlmVision* vision) {
    return vision ? vision->organism.dimensions() : 0;
}

int32_t alm_vision_classes(const AlmVision* vision) {
    return vision ? vision->organism.classes() : 0;
}

void alm_vision_evidence(AlmVision* vision, const float* rgb, float* out_evidence) {
    if (!vision || !rgb || !out_evidence) return;
    vision->organism.evidence(rgb, out_evidence);
}

int32_t alm_vision_predict(const AlmVision* vision, const float* evidence,
                           float* out_votes) {
    if (!vision || !evidence || !out_votes) return -1;
    return vision->organism.predict_evidence(evidence, out_votes);
}

int32_t alm_vision_predict_single_column(const AlmVision* vision, const float* evidence,
                                         float* out_votes) {
    if (!vision || !evidence || !out_votes) return -1;
    return vision->organism.predict_evidence_single_column(evidence, out_votes);
}

void alm_vision_observe_external(AlmVision* vision, const float* evidence,
                                 int32_t label) {
    if (!vision || !evidence) return;
    vision->organism.observe_external(evidence, label);
}

void alm_vision_reset_episode(AlmVision* vision) {
    if (vision) vision->organism.reset_episode();
}

void alm_vision_sleep(AlmVision* vision) {
    if (vision) vision->organism.sleep();
}

uint8_t* alm_vision_save(const AlmVision* vision, size_t* out_size) {
    if (!vision || !out_size) return nullptr;
    std::vector<uint8_t> encoded = alm::VisionIO::save(vision->organism);
    uint8_t* buffer = static_cast<uint8_t*>(::operator new(encoded.size(), std::nothrow));
    if (!buffer) {
        *out_size = 0;
        return nullptr;
    }
    std::copy(encoded.begin(), encoded.end(), buffer);
    *out_size = encoded.size();
    return buffer;
}

void alm_vision_free_buffer(uint8_t* buffer) { ::operator delete(buffer); }

AlmVision* alm_vision_load(const uint8_t* data, size_t size, int32_t* error) {
    if (error) *error = 0;
    if (!data || size == 0) {
        if (error) *error = 1;
        return nullptr;
    }
    try {
        alm::GrowingVisionOrganism loaded = alm::VisionIO::load(data, size);
        return new (std::nothrow) AlmVision(std::move(loaded));
    } catch (const std::exception&) {
        if (error) *error = 1;
        return nullptr;
    }
}

int64_t alm_vision_stat(const AlmVision* vision, int32_t stat) {
    if (!vision) return 0;
    const alm::GrowingVisionOrganism& o = vision->organism;
    switch (stat) {
        case ALM_VISION_STAT_STEPS: return o.steps();
        case ALM_VISION_STAT_LIVE_ASSEMBLIES: return o.live_assemblies();
        case ALM_VISION_STAT_SLEEP_CYCLES: return o.sleep_cycles();
        case ALM_VISION_STAT_VISUAL_BIRTHS: return o.visual_births();
        case ALM_VISION_STAT_RELATION_TURNOVER_RELEASES: return o.relation_turnover_releases();
        case ALM_VISION_STAT_MEMORY_BYTES: return static_cast<int64_t>(o.memory_bytes());
        default: return 0;
    }
}

}  // extern "C"
