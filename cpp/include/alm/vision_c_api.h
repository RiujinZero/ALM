// Flat C ABI for the native growing-vision organism, called from Python via
// ctypes. See vision.hpp for the correctness contract (numerically close,
// not bit-exact -- PyTorch's reduction order/topk tie-breaking are not a
// portable target).
#ifndef ALM_VISION_C_API_H
#define ALM_VISION_C_API_H

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

typedef struct AlmVision AlmVision;

typedef struct AlmVisionConfig {
    int32_t classes;
    int32_t image_size;
    int32_t columns;
    double context_decay;
    int32_t sleep_interval;
    int64_t max_body_bytes;
    double learning_rate;
    int32_t maximum_synapses;
    int32_t maximum_visual_assemblies;
    double visual_birth_activation;
    int32_t visual_routing_width;
    int32_t relation_sketch_words;
    double relation_turnover_fraction;

    // Sparse episodic recall for continuous evidence -- see VisionConfig in
    // vision.hpp for the design rationale (SimHash key, Hamming-ball probed
    // lookup, validated standalone before being wired in).
    int32_t episodic_enabled;
    int32_t episodic_lsh_bits;
    int32_t episodic_probe_radius;
    int32_t episodic_chain_slots;
    int32_t episodic_tier3_refresh_threshold;
    double episodic_strength;
} AlmVisionConfig;

ALM_EXPORT AlmVision* alm_vision_create(const AlmVisionConfig* config);
ALM_EXPORT void alm_vision_destroy(AlmVision* vision);

// Turns episodic recall on for an already-constructed organism (e.g. one
// loaded from a checkpoint saved before this was enabled). No-op if already
// enabled. See GrowingVisionOrganism::enable_episodic_layer in vision.hpp.
ALM_EXPORT void alm_vision_enable_episodic_layer(AlmVision* vision);

ALM_EXPORT int32_t alm_vision_dimensions(const AlmVision* vision);
ALM_EXPORT int32_t alm_vision_classes(const AlmVision* vision);

// `rgb` is [image_size, image_size, 3] float32 in [0, 1] (already decoded
// and resized in Python). Writes `dimensions()` floats to `out_evidence`
// and advances the slow context trace.
ALM_EXPORT void alm_vision_evidence(AlmVision* vision, const float* rgb,
                                    float* out_evidence);

// Writes `classes` floats to `out_votes`; returns the argmax prediction.
ALM_EXPORT int32_t alm_vision_predict(const AlmVision* vision, const float* evidence,
                                      float* out_votes);

// Diagnostic-only: same signature as alm_vision_predict, but votes from just
// the currently-awake column instead of ensembling all columns. See
// GrowingVisionOrganism::predict_evidence_single_column in vision.hpp.
ALM_EXPORT int32_t alm_vision_predict_single_column(const AlmVision* vision,
                                                    const float* evidence,
                                                    float* out_votes);

ALM_EXPORT void alm_vision_observe_external(AlmVision* vision, const float* evidence,
                                            int32_t label);
ALM_EXPORT void alm_vision_reset_episode(AlmVision* vision);
ALM_EXPORT void alm_vision_sleep(AlmVision* vision);

enum AlmVisionStat {
    ALM_VISION_STAT_STEPS = 0,
    ALM_VISION_STAT_LIVE_ASSEMBLIES = 1,
    ALM_VISION_STAT_SLEEP_CYCLES = 2,
    ALM_VISION_STAT_VISUAL_BIRTHS = 3,
    ALM_VISION_STAT_RELATION_TURNOVER_RELEASES = 4,
    ALM_VISION_STAT_MEMORY_BYTES = 5
};
ALM_EXPORT int64_t alm_vision_stat(const AlmVision* vision, int32_t stat);

// Same-machine, same-build checkpoint format only (see vision_serialize.hpp).
// `alm_vision_save` allocates the returned buffer; free it with
// `alm_vision_free_buffer`. `alm_vision_load` returns NULL and leaves
// `error` nonzero on any failure (corrupt buffer, wrong version, or a
// config mismatch against the stored dimensions/column count).
ALM_EXPORT uint8_t* alm_vision_save(const AlmVision* vision, size_t* out_size);
ALM_EXPORT void alm_vision_free_buffer(uint8_t* buffer);
ALM_EXPORT AlmVision* alm_vision_load(const uint8_t* data, size_t size, int32_t* error);

#ifdef __cplusplus
}
#endif

#endif  // ALM_VISION_C_API_H
