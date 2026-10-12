// Native port of continual_vision.py's ALM64MB4CGrowingVision.
//
// Unlike the byte cortex, this is not a bit-exact port: PyTorch's internal
// reduction order (pairwise summation, etc.) and topk's tie-breaking under
// equal values are not literally reproducible from a from-scratch
// implementation without porting PyTorch's own kernels. The bar here is
// numerically matching within a tight tolerance and functionally identical
// growth/update decisions, verified against the Python/PyTorch reference on
// real data -- disclosed, not quietly downgraded.
//
// Image decode (JPEG/PNG) and the bilinear resize to image_size x image_size
// stay in Python/PIL, an already-optimized C library and not the identified
// bottleneck. This module starts from the resized RGB array: the receptor's
// gradient/vector math, and the entire growing-assembly column system
// (activation, voting, growth, local update, relation-sketch bookkeeping,
// sleep/merge), all as dense arrays with no per-sample allocation and no
// GPU round trip.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace alm {

struct VisionConfig {
    int32_t classes = 2;
    int32_t image_size = 24;
    int32_t columns = 4;
    double context_decay = 0.94;
    int32_t sleep_interval = 4096;
    int64_t max_body_bytes = 64'000'000;
    double learning_rate = 0.5;
    int32_t maximum_synapses = 128;
    int32_t maximum_visual_assemblies = 50'000;
    double visual_birth_activation = 0.72;
    int32_t visual_routing_width = 16;
    int32_t relation_sketch_words = 4;
    double relation_turnover_fraction = 0.0;

    // Sparse episodic recall for continuous evidence -- the vision-side
    // analogue of the byte cortex's episodic layer (cortex.hpp), validated
    // standalone (vision_lsh_recall_probe.py) before being wired in here.
    // Unlike byte sequences, two occurrences of "the same thing" are never
    // bit-identical in a continuous evidence vector (camera noise, pose,
    // lighting), so the key is a locality-sensitive sign-hash of a fixed
    // random projection (SimHash), not an exact hash -- and lookup probes
    // a small Hamming ball around the query key rather than requiring an
    // exact bucket match, which the standalone probe found necessary:
    // requiring all `episodic_lsh_bits` sign bits to agree is effectively
    // an exact-match requirement in disguise (0% recall under any real
    // perturbation), while probing radius 2 with 20 bits held 100/100/98%
    // recall at zero/small/moderate perturbation and degraded gracefully
    // (not catastrophically) out to a 50,000-instance gap.
    bool episodic_enabled = false;
    int32_t episodic_lsh_bits = 16;      // bucket table has 2^this entries
    int32_t episodic_probe_radius = 2;   // Hamming-ball radius checked at query time
    int32_t episodic_chain_slots = 4;    // distinct labels remembered per bucket
    int32_t episodic_tier3_refresh_threshold = 3;  // mirrors the byte cortex's own
    // Blend weight: out_votes[label] gets an additive boost of
    // refresh/(refresh+strength) when a bucket recalls `label`. Mirrors the
    // byte cortex's episodic_strength shape and its validated finding --
    // LOWER strength lets a real recalled observation dominate MORE, not
    // less (a single real occurrence's boosted weight is
    // refresh/(refresh+strength), so strength<=1 keeps that >=0.5 even
    // after only one refresh). Not yet swept for vision specifically;
    // defaults straight to the byte-side's validated regime rather than
    // the naive starting point that regime replaced.
    double episodic_strength = 0.1;
};

// Grayscale + horizontal/vertical local contrast + the raw RGB sheet,
// concatenated and L2-normalized. Matches RawImageReceptor.__call__ exactly
// in operation order (the input is already the resized [size,size,3] float
// image in [0,1], matching `pixels` in the Python source).
class VisionReceptor {
public:
    explicit VisionReceptor(int32_t image_size) : size_(image_size) {
        if (size_ < 1) throw std::runtime_error("image_size must be positive");
    }

    int32_t dimensions() const noexcept { return 5 * size_ * size_; }

    // `rgb` is [size, size, 3] row-major, values in [0, 1]. `out` must have
    // room for `dimensions()` floats.
    void transform(const float* rgb, float* out) const {
        const int32_t n = size_ * size_;
        std::vector<float> grey(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            const float r = rgb[3 * i + 0];
            const float g = rgb[3 * i + 1];
            const float b = rgb[3 * i + 2];
            grey[static_cast<size_t>(i)] = (r + g + b) / 3.0f;
        }
        // pixels.reshape(-1): row-major [size,size,3] already matches `rgb`.
        std::memcpy(out, rgb, static_cast<size_t>(3 * n) * sizeof(float));
        float* horizontal = out + 3 * n;
        float* vertical = out + 4 * n;
        std::fill(horizontal, horizontal + n, 0.0f);
        std::fill(vertical, vertical + n, 0.0f);
        for (int32_t row = 0; row < size_; ++row) {
            for (int32_t col = 1; col < size_; ++col) {
                const int32_t idx = row * size_ + col;
                horizontal[idx] = grey[static_cast<size_t>(idx)] -
                                   grey[static_cast<size_t>(idx - 1)];
            }
        }
        for (int32_t row = 1; row < size_; ++row) {
            for (int32_t col = 0; col < size_; ++col) {
                const int32_t idx = row * size_ + col;
                vertical[idx] = grey[static_cast<size_t>(idx)] -
                                 grey[static_cast<size_t>(idx - size_)];
            }
        }
        const int32_t total = dimensions();
        double sum_sq = 0.0;
        for (int32_t i = 0; i < total; ++i) sum_sq += static_cast<double>(out[i]) * out[i];
        const double norm = std::sqrt(sum_sq);
        if (norm > 0.0) {
            const float inv = static_cast<float>(1.0 / norm);
            for (int32_t i = 0; i < total; ++i) out[i] *= inv;
        }
    }

private:
    int32_t size_;
};

inline int32_t popcount64(uint64_t value) noexcept {
#if defined(_MSC_VER)
    return static_cast<int32_t>(__popcnt64(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int32_t count = 0;
    while (value) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

namespace detail {

// Indices of the `count` largest values by `key(i)`, ties broken by lower
// index (a specific, documented choice -- PyTorch's own tie-break order
// under equal keys is not a portable contract to match against).
template <typename KeyFn>
std::vector<int32_t> top_indices(int32_t n, int32_t count, KeyFn key) {
    count = std::min(count, n);
    std::vector<int32_t> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + count, idx.end(),
                       [&](int32_t a, int32_t b) {
                           const double ka = key(a);
                           const double kb = key(b);
                           if (ka != kb) return ka > kb;
                           return a < b;
                       });
    idx.resize(static_cast<size_t>(count));
    return idx;
}

}  // namespace detail

// One sparse, growing population of visual assemblies (_GrowingVisualColumn).
class VisionColumn {
public:
    friend struct VisionIO;

    VisionColumn(const VisionConfig& config, int32_t dimensions)
        : config_(config), dimensions_(dimensions) {
        const size_t capacity = static_cast<size_t>(config_.maximum_visual_assemblies);
        const size_t synapses = static_cast<size_t>(config_.maximum_synapses);
        indices_.assign(capacity * synapses, 0);
        weights_.assign(capacity * synapses, 0.0f);
        label_evidence_.assign(capacity * static_cast<size_t>(config_.classes), 0.0f);
        observations_.assign(capacity, 0);
        relation_sketch_.assign(
            capacity * static_cast<size_t>(config_.relation_sketch_words), 0ULL);
    }

    int32_t live() const noexcept { return live_; }
    int64_t births() const noexcept { return births_; }
    int64_t rebirths() const noexcept { return rebirths_; }

    // Per-assembly activation for the given evidence vector; empty if none
    // live. This is the dominant cost at scale: O(live_ * synapses) gathers
    // per call, and live_ grows toward maximum_visual_assemblies over a
    // long run -- unlike alm_native's bit-exact byte cortex, this module's
    // correctness contract is already numerically-close, not bit-exact
    // (see the file header and cpp/README.md), so trading the previous
    // double-precision accumulation and strict left-to-right summation
    // order for AVX2 gather + float32 accumulation is a real reassociation
    // but stays within the same float32-noise-floor tolerance already
    // measured and accepted by tests/test_native_vision_matches_python.py.
    std::vector<float> activations(const float* evidence) const {
        std::vector<float> result(static_cast<size_t>(live_));
        const int32_t synapses = config_.maximum_synapses;
#if defined(__AVX2__)
        for (int32_t a = 0; a < live_; ++a) {
            const int32_t* row_idx = &indices_[static_cast<size_t>(a) * synapses];
            const float* row_w = &weights_[static_cast<size_t>(a) * synapses];
            __m256 acc = _mm256_setzero_ps();
            int32_t s = 0;
            for (; s + 8 <= synapses; s += 8) {
                __m256i idx = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row_idx + s));
                __m256 w = _mm256_loadu_ps(row_w + s);
                __m256 e = _mm256_i32gather_ps(evidence, idx, 4);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(w, e));
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sums = _mm_add_ps(sum4, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float total = _mm_cvtss_f32(sums);
            for (; s < synapses; ++s) total += row_w[s] * evidence[row_idx[s]];
            float clamped = total < 0.0f ? 0.0f : (total > 1.0f ? 1.0f : total);
            result[static_cast<size_t>(a)] = clamped;
        }
#else
        for (int32_t a = 0; a < live_; ++a) {
            float sum = 0.0f;
            const int32_t* row_idx = &indices_[static_cast<size_t>(a) * synapses];
            const float* row_w = &weights_[static_cast<size_t>(a) * synapses];
            for (int32_t s = 0; s < synapses; ++s) {
                sum += row_w[s] * evidence[row_idx[s]];
            }
            float clamped = sum < 0.0f ? 0.0f : (sum > 1.0f ? 1.0f : sum);
            result[static_cast<size_t>(a)] = clamped;
        }
#endif
        return result;
    }

    // Class votes plus the raw activation vector (needed by the caller for
    // growth/relation decisions), matching `votes()`.
    void votes(const float* evidence, float* out_votes,
               std::vector<float>* out_activity) const {
        std::fill(out_votes, out_votes + config_.classes, 0.0f);
        std::vector<float> activity = activations(evidence);
        if (out_activity) *out_activity = activity;
        if (activity.empty()) return;
        const int32_t width = std::min(config_.visual_routing_width, live_);
        std::vector<int32_t> top =
            detail::top_indices(live_, width, [&](int32_t i) { return activity[static_cast<size_t>(i)]; });
        for (int32_t assembly : top) {
            const float strength = activity[static_cast<size_t>(assembly)];
            if (strength <= 0.0f) continue;
            const float* labels =
                &label_evidence_[static_cast<size_t>(assembly) * config_.classes];
            double total = 0.0;
            for (int32_t c = 0; c < config_.classes; ++c) total += labels[c];
            total = std::max(total, 1.0);
            for (int32_t c = 0; c < config_.classes; ++c) {
                out_votes[c] += static_cast<float>(strength * labels[c] / total);
            }
        }
    }

    // Returns true if a new assembly was born.
    bool observe_external(const float* evidence, int32_t label) {
        if (config_.classes > 256) throw std::runtime_error("classes too large for stack buffer");
        float class_votes[256];
        std::vector<float> activity;
        votes(evidence, class_votes, &activity);

        int32_t prediction = -1;
        float best = -1.0f;
        bool any_vote = false;
        for (int32_t c = 0; c < config_.classes; ++c) {
            if (class_votes[c] != 0.0f) any_vote = true;
            if (class_votes[c] > best) {
                best = class_votes[c];
                prediction = c;
            }
        }
        if (!any_vote) prediction = -1;

        float peak = 0.0f;
        for (float v : activity) peak = std::max(peak, v);

        if (peak < static_cast<float>(config_.visual_birth_activation) ||
            prediction != label) {
            const int32_t born = grow(evidence, label);
            strengthen_relations(activity, born);
            if (born < 0 && !activity.empty()) {
                // Capacity full: grow() silently declined. The closest
                // existing assembly's synapse *indices* were fixed at
                // whatever birth first picked them -- reweighting alone
                // (update_winner) can only nudge its weights on those same
                // fixed positions, which does nothing if that index set is
                // simply a poor fit for what's recurring here now. Rebirth
                // it in place instead, with the exact same selection rule
                // grow() uses (top-K by |evidence|, fresh unit-energy
                // weights): this is the same "surprise" trigger that
                // created assemblies in the first place (low peak
                // activation or a wrong prediction), just applied to an
                // existing slot instead of a new one since there's no room
                // left to append.
                const int32_t winner = static_cast<int32_t>(
                    std::max_element(activity.begin(), activity.end()) - activity.begin());
                rebirth(winner, evidence, label);
            }
            return born >= 0;
        }

        const int32_t winner = static_cast<int32_t>(
            std::max_element(activity.begin(), activity.end()) - activity.begin());
        update_winner(evidence, winner, label);
        strengthen_relations(activity, -1);
        return false;
    }

    void load_rows(int32_t count, const int32_t* indices, const float* weights,
                  const float* labels, const int32_t* observations,
                  const uint64_t* relations) {
        if (count > config_.maximum_visual_assemblies) {
            throw std::runtime_error("merged visual tissue exceeds row capacity");
        }
        const int32_t synapses = config_.maximum_synapses;
        const int32_t classes = config_.classes;
        const int32_t words = config_.relation_sketch_words;
        std::fill(indices_.begin(), indices_.end(), 0);
        std::fill(weights_.begin(), weights_.end(), 0.0f);
        std::fill(label_evidence_.begin(), label_evidence_.end(), 0.0f);
        std::fill(observations_.begin(), observations_.end(), 0);
        std::fill(relation_sketch_.begin(), relation_sketch_.end(), 0ULL);
        std::memcpy(indices_.data(), indices,
                    static_cast<size_t>(count) * synapses * sizeof(int32_t));
        std::memcpy(weights_.data(), weights,
                    static_cast<size_t>(count) * synapses * sizeof(float));
        std::memcpy(label_evidence_.data(), labels,
                    static_cast<size_t>(count) * classes * sizeof(float));
        std::memcpy(observations_.data(), observations,
                    static_cast<size_t>(count) * sizeof(int32_t));
        std::memcpy(relation_sketch_.data(), relations,
                    static_cast<size_t>(count) * words * sizeof(uint64_t));
        live_ = count;
    }

    // Accessors for merging (sleep()) across columns, and for serialization.
    const int32_t* indices_data() const noexcept { return indices_.data(); }
    const float* weights_data() const noexcept { return weights_.data(); }
    const float* labels_data() const noexcept { return label_evidence_.data(); }
    const int32_t* observations_data() const noexcept { return observations_.data(); }
    const uint64_t* relations_data() const noexcept { return relation_sketch_.data(); }
    int32_t synapses() const noexcept { return config_.maximum_synapses; }
    int32_t classes() const noexcept { return config_.classes; }
    int32_t relation_words() const noexcept { return config_.relation_sketch_words; }
    void set_births(int64_t value) noexcept { births_ = value; }

    size_t memory_bytes() const noexcept {
        return indices_.size() * sizeof(int32_t) + weights_.size() * sizeof(float) +
               label_evidence_.size() * sizeof(float) +
               observations_.size() * sizeof(int32_t) +
               relation_sketch_.size() * sizeof(uint64_t);
    }

private:
    int32_t grow(const float* evidence, int32_t label) {
        if (live_ >= config_.maximum_visual_assemblies) return -1;
        const int32_t assembly = live_;
        const int32_t synapses = config_.maximum_synapses;
        std::vector<int32_t> selected = detail::top_indices(
            dimensions_, synapses, [&](int32_t i) { return std::fabs(evidence[i]); });
        double energy = 0.0;
        for (int32_t idx : selected) energy += static_cast<double>(evidence[idx]) * evidence[idx];
        energy = std::max(energy, 1e-12);
        int32_t* row_idx = &indices_[static_cast<size_t>(assembly) * synapses];
        float* row_w = &weights_[static_cast<size_t>(assembly) * synapses];
        for (size_t s = 0; s < selected.size(); ++s) {
            row_idx[s] = selected[s];
            row_w[s] = static_cast<float>(evidence[selected[s]] / energy);
        }
        for (size_t s = selected.size(); s < static_cast<size_t>(synapses); ++s) {
            row_idx[s] = 0;
            row_w[s] = 0.0f;
        }
        label_evidence_[static_cast<size_t>(assembly) * config_.classes + label] = 1.0f;
        observations_[static_cast<size_t>(assembly)] = 1;
        ++live_;
        ++births_;
        return assembly;
    }

    void update_winner(const float* evidence, int32_t winner, int32_t label) {
        const int32_t synapses = config_.maximum_synapses;
        int32_t* row_idx = &indices_[static_cast<size_t>(winner) * synapses];
        float* row_w = &weights_[static_cast<size_t>(winner) * synapses];
        double energy = 0.0;
        std::vector<double> values(static_cast<size_t>(synapses));
        for (int32_t s = 0; s < synapses; ++s) {
            values[static_cast<size_t>(s)] = evidence[row_idx[s]];
            energy += values[static_cast<size_t>(s)] * values[static_cast<size_t>(s)];
        }
        energy = std::max(energy, 1e-12);
        const double rate = config_.learning_rate /
                            (1.0 + std::sqrt(static_cast<double>(observations_[static_cast<size_t>(winner)])));
        for (int32_t s = 0; s < synapses; ++s) {
            const double target = values[static_cast<size_t>(s)] / energy;
            row_w[s] = static_cast<float>(row_w[s] + rate * (target - row_w[s]));
        }
        label_evidence_[static_cast<size_t>(winner) * config_.classes + label] += 1.0f;
        observations_[static_cast<size_t>(winner)] += 1;
    }

    // Same index-selection and weight-init rule as grow(), but writes into
    // an existing assembly slot rather than appending a new one. Fully
    // replaces that assembly's anatomy (indices AND weights, not just
    // weights) and resets its label distribution -- a genuine rebirth, not
    // a nudge, so a slot whose original random synapse subset turned out to
    // be a poor fit for what recurs there can pick a better one.
    void rebirth(int32_t assembly, const float* evidence, int32_t label) {
        const int32_t synapses = config_.maximum_synapses;
        const int32_t classes = config_.classes;
        std::vector<int32_t> selected = detail::top_indices(
            dimensions_, synapses, [&](int32_t i) { return std::fabs(evidence[i]); });
        double energy = 0.0;
        for (int32_t idx : selected) energy += static_cast<double>(evidence[idx]) * evidence[idx];
        energy = std::max(energy, 1e-12);
        int32_t* row_idx = &indices_[static_cast<size_t>(assembly) * synapses];
        float* row_w = &weights_[static_cast<size_t>(assembly) * synapses];
        for (size_t s = 0; s < selected.size(); ++s) {
            row_idx[s] = selected[s];
            row_w[s] = static_cast<float>(evidence[selected[s]] / energy);
        }
        for (size_t s = selected.size(); s < static_cast<size_t>(synapses); ++s) {
            row_idx[s] = 0;
            row_w[s] = 0.0f;
        }
        float* labels = &label_evidence_[static_cast<size_t>(assembly) * classes];
        std::fill(labels, labels + classes, 0.0f);
        labels[label] = 1.0f;
        observations_[static_cast<size_t>(assembly)] = 1;
        ++rebirths_;
    }

    void strengthen_relations(const std::vector<float>& activity, int32_t born) {
        const int32_t width = std::min(config_.visual_routing_width,
                                       static_cast<int32_t>(activity.size()));
        std::vector<int32_t> assemblies;
        if (width > 0) {
            std::vector<int32_t> top = detail::top_indices(
                static_cast<int32_t>(activity.size()), width,
                [&](int32_t i) { return activity[static_cast<size_t>(i)]; });
            for (int32_t a : top) {
                if (activity[static_cast<size_t>(a)] > 0.0f) assemblies.push_back(a);
            }
        }
        if (born >= 0) assemblies.push_back(born);
        std::sort(assemblies.begin(), assemblies.end());
        assemblies.erase(std::unique(assemblies.begin(), assemblies.end()), assemblies.end());
        if (assemblies.size() < 2) return;

        const int32_t words = config_.relation_sketch_words;
        std::vector<uint64_t> signature(static_cast<size_t>(words), 0ULL);
        for (int32_t assembly : assemblies) {
            const int32_t word = assembly % words;
            const int32_t bit = (assembly / words) % 62;
            signature[static_cast<size_t>(word)] |= (1ULL << bit);
        }
        for (int32_t assembly : assemblies) {
            uint64_t* row = &relation_sketch_[static_cast<size_t>(assembly) * words];
            for (int32_t w = 0; w < words; ++w) row[w] |= signature[static_cast<size_t>(w)];
        }
    }

    VisionConfig config_;
    int32_t dimensions_;
    int32_t live_ = 0;
    int64_t births_ = 0;
    int64_t rebirths_ = 0;
    std::vector<int32_t> indices_;
    std::vector<float> weights_;
    std::vector<float> label_evidence_;
    std::vector<int32_t> observations_;
    std::vector<uint64_t> relation_sketch_;
};

// Four wake lineages plus the slow context trace and sleep/merge lifecycle
// (ALM64MB4CGrowingVision).
class GrowingVisionOrganism {
public:
    friend struct VisionIO;

    explicit GrowingVisionOrganism(const VisionConfig& config)
        : config_(config), receptor_(config.image_size) {
        dimensions_ = 2 * receptor_.dimensions();
        for (int32_t c = 0; c < config_.columns; ++c) {
            columns_.emplace_back(config_, dimensions_);
        }
        slow_context_.assign(static_cast<size_t>(receptor_.dimensions()), 0.0f);
        label_counts_.assign(static_cast<size_t>(config_.classes), 0.0f);
        if (config_.episodic_enabled) init_episodic_state();
    }

    // Turns episodic recall on for an already-constructed organism (e.g.
    // loaded from a checkpoint saved before this was enabled). No-op if
    // already enabled. Mirrors GrowableCortex::enable_episodic_layer.
    void enable_episodic_layer() {
        if (config_.episodic_enabled) return;
        config_.episodic_enabled = true;
        init_episodic_state();
    }

    void reset_episode() { std::fill(slow_context_.begin(), slow_context_.end(), 0.0f); }

    // `rgb` is the already-resized [image_size, image_size, 3] float image
    // in [0, 1]. Fills `out_evidence` (dimensions_ floats): basal receptor
    // vector concatenated with the slow apical trace, L2-normalized: then
    // advances the trace exactly as Python's evidence() does.
    void evidence(const float* rgb, float* out_evidence) {
        const int32_t basal_dim = receptor_.dimensions();
        // receptor_.transform() already L2-normalizes its own output (unit
        // norm on its own, matching RawImageReceptor.__call__). That value
        // -- BEFORE the second, combined-vector normalization below -- is
        // what Python's slow_context update uses, so it has to be saved
        // before out_evidence[0:basal_dim] gets rescaled again.
        receptor_.transform(rgb, out_evidence);
        std::vector<float> basal(out_evidence, out_evidence + basal_dim);
        std::memcpy(out_evidence + basal_dim, slow_context_.data(),
                    static_cast<size_t>(basal_dim) * sizeof(float));
        double sum_sq = 0.0;
        for (int32_t i = 0; i < dimensions_; ++i) {
            sum_sq += static_cast<double>(out_evidence[i]) * out_evidence[i];
        }
        const double norm = std::max(std::sqrt(sum_sq), 1e-12);
        const float inv = static_cast<float>(1.0 / norm);
        for (int32_t i = 0; i < dimensions_; ++i) out_evidence[i] *= inv;

        const double decay = config_.context_decay;
        for (int32_t i = 0; i < basal_dim; ++i) {
            slow_context_[static_cast<size_t>(i)] =
                static_cast<float>(decay * slow_context_[static_cast<size_t>(i)] +
                                   (1.0 - decay) * basal[static_cast<size_t>(i)]);
        }
    }

    // Fills out_votes (classes floats) and returns the argmax prediction.
    //
    // Measured (profiling, not assumed): at CLEAR-10's full 50,000-assembly
    // capacity, this was 3.5x slower than observe_external() despite doing
    // *less* work per call (no growth/weight-update) -- because it scans
    // all `columns_` columns, each a full O(live * synapses) AVX2 gather,
    // where observe_external() only ever touches the single currently-awake
    // column. And since the benchmark harness calls predict() *and*
    // observe_external() on every training sample (not just at eval time),
    // this redundant per-column scan was the dominant cost of training
    // itself. Each column's votes() is const and reads only that column's
    // own arrays plus the shared (also read-only during this call) evidence
    // vector -- genuinely independent, so this parallelizes across columns
    // with no behavior change: same math, same result, concurrent instead
    // of sequential.
    int32_t predict_evidence(const float* evidence, float* out_votes) const {
        const int32_t classes = config_.classes;
        const size_t column_count = columns_.size();
        std::vector<float> per_column(column_count * static_cast<size_t>(classes), 0.0f);
        if (column_count > 1) {
            std::vector<std::thread> workers;
            workers.reserve(column_count);
            for (size_t i = 0; i < column_count; ++i) {
                workers.emplace_back([this, evidence, &per_column, classes, i]() {
                    columns_[i].votes(
                        evidence, per_column.data() + i * static_cast<size_t>(classes), nullptr);
                });
            }
            for (std::thread& worker : workers) worker.join();
        } else if (column_count == 1) {
            columns_[0].votes(evidence, per_column.data(), nullptr);
        }

        std::fill(out_votes, out_votes + classes, 0.0f);
        bool any = false;
        for (size_t i = 0; i < column_count; ++i) {
            const float* local = per_column.data() + i * static_cast<size_t>(classes);
            for (int32_t c = 0; c < classes; ++c) {
                out_votes[c] += local[c];
                if (local[c] != 0.0f) any = true;
            }
        }
        if (!any) {
            std::copy(label_counts_.begin(), label_counts_.end(), out_votes);
        }
        if (config_.episodic_enabled) episodic_recall(evidence, out_votes);
        return static_cast<int32_t>(
            std::max_element(out_votes, out_votes + config_.classes) - out_votes);
    }

    // Diagnostic-only counterpart to predict_evidence(): votes from just the
    // currently-awake column (the same one observe_external() would train),
    // instead of summing all columns. Used to measure whether the 4-column
    // ensemble is earning its cost over a single column, since between
    // sleep() merges the non-awake columns are mostly stale copies of the
    // same base rows plus whatever each grew independently.
    int32_t predict_evidence_single_column(const float* evidence, float* out_votes) const {
        const int32_t classes = config_.classes;
        const VisionColumn& awake = columns_[static_cast<size_t>(steps_ % config_.columns)];
        awake.votes(evidence, out_votes, nullptr);
        bool any = false;
        for (int32_t c = 0; c < classes; ++c) {
            if (out_votes[c] != 0.0f) any = true;
        }
        if (!any) {
            std::copy(label_counts_.begin(), label_counts_.end(), out_votes);
        }
        if (config_.episodic_enabled) episodic_recall(evidence, out_votes);
        return static_cast<int32_t>(
            std::max_element(out_votes, out_votes + config_.classes) - out_votes);
    }

    void observe_external(const float* evidence, int32_t label) {
        if (label < 0 || label >= config_.classes) {
            throw std::runtime_error("label outside configured classes");
        }
        VisionColumn& awake = columns_[static_cast<size_t>(steps_ % config_.columns)];
        const bool born = awake.observe_external(evidence, label);
        visual_births_ += born ? 1 : 0;
        label_counts_[static_cast<size_t>(label)] += 1.0f;
        // Deliberately unconditional on column growth/update above, same
        // decoupling reasoning as the byte cortex's own episodic layer:
        // this tracks what was actually seen and labeled, not whether the
        // column anatomy happened to grow or update from it.
        if (config_.episodic_enabled) episodic_write(evidence, label);
        ++steps_;
        ++external_updates_;
        if (steps_ % config_.sleep_interval == 0) sleep();
    }

    void sleep() {
        const int32_t synapses = config_.maximum_synapses;
        const int32_t classes = config_.classes;
        const int32_t words = config_.relation_sketch_words;
        const int32_t base = base_rows_;

        // Merge: base rows averaged across columns (weights/labels/obs) with
        // relation sketches OR'd; new rows appended per column beyond base.
        std::vector<int32_t> merged_indices;
        std::vector<float> merged_weights;
        std::vector<float> merged_labels;
        std::vector<int32_t> merged_observations;
        std::vector<uint64_t> merged_relations;

        if (base > 0) {
            const int32_t* base_idx = columns_[0].indices_data();
            merged_indices.assign(base_idx, base_idx + static_cast<size_t>(base) * synapses);
            merged_weights.assign(static_cast<size_t>(base) * synapses, 0.0f);
            merged_labels.assign(static_cast<size_t>(base) * classes, 0.0f);
            std::vector<double> observation_sum(static_cast<size_t>(base), 0.0);
            merged_relations.assign(
                columns_[0].relations_data(),
                columns_[0].relations_data() + static_cast<size_t>(base) * words);
            for (const auto& column : columns_) {
                const float* w = column.weights_data();
                const float* l = column.labels_data();
                const int32_t* o = column.observations_data();
                const uint64_t* r = column.relations_data();
                for (size_t i = 0; i < static_cast<size_t>(base) * synapses; ++i) {
                    merged_weights[i] += w[i];
                }
                for (size_t i = 0; i < static_cast<size_t>(base) * classes; ++i) {
                    merged_labels[i] += l[i];
                }
                for (int32_t row = 0; row < base; ++row) {
                    observation_sum[static_cast<size_t>(row)] += o[row];
                }
                if (&column != &columns_[0]) {
                    for (size_t i = 0; i < static_cast<size_t>(base) * words; ++i) {
                        merged_relations[i] |= r[i];
                    }
                }
            }
            const float column_count = static_cast<float>(config_.columns);
            for (float& v : merged_weights) v /= column_count;
            for (float& v : merged_labels) v /= column_count;
            merged_observations.resize(static_cast<size_t>(base));
            for (int32_t row = 0; row < base; ++row) {
                merged_observations[static_cast<size_t>(row)] = static_cast<int32_t>(
                    observation_sum[static_cast<size_t>(row)] / config_.columns);
            }
        }

        for (const auto& column : columns_) {
            const int32_t live = column.live();
            for (int32_t row = base; row < live; ++row) {
                const int32_t* idx_row = column.indices_data() + static_cast<size_t>(row) * synapses;
                const float* w_row = column.weights_data() + static_cast<size_t>(row) * synapses;
                const float* l_row = column.labels_data() + static_cast<size_t>(row) * classes;
                const uint64_t* r_row = column.relations_data() + static_cast<size_t>(row) * words;
                merged_indices.insert(merged_indices.end(), idx_row, idx_row + synapses);
                merged_weights.insert(merged_weights.end(), w_row, w_row + synapses);
                merged_labels.insert(merged_labels.end(), l_row, l_row + classes);
                merged_observations.push_back(column.observations_data()[row]);
                merged_relations.insert(merged_relations.end(), r_row, r_row + words);
            }
        }

        int32_t total = static_cast<int32_t>(merged_observations.size());
        const int32_t capacity = config_.maximum_visual_assemblies;
        int32_t release = 0;
        if (total >= capacity) {
            release = static_cast<int32_t>(capacity * config_.relation_turnover_fraction);
        }
        int32_t retain_count = std::min(total, capacity - release);
        if (total > retain_count) {
            std::vector<double> utility(static_cast<size_t>(total));
            for (int32_t row = 0; row < total; ++row) {
                const uint64_t* r_row = &merged_relations[static_cast<size_t>(row) * words];
                int32_t degree = 0;
                for (int32_t w = 0; w < words; ++w) degree += popcount64(r_row[w]);
                const float* l_row = &merged_labels[static_cast<size_t>(row) * classes];
                double label_sum = 0.0, label_max = 0.0;
                for (int32_t c = 0; c < classes; ++c) {
                    label_sum += l_row[c];
                    label_max = std::max(label_max, static_cast<double>(l_row[c]));
                }
                const double confidence = label_max / std::max(label_sum, 1.0);
                utility[static_cast<size_t>(row)] =
                    degree * (capacity + 1.0) +
                    static_cast<double>(merged_observations[static_cast<size_t>(row)]) * confidence;
            }
            // Deliberately NOT re-sorted to index order: Python's
            // torch.topk(...).indices comes back utility-descending, and
            // that row order feeds the assembly-index-dependent relation
            // signature bits computed on every future strengthen_relations
            // call -- reordering here would silently diverge behavior over
            // subsequent sleeps, not just cosmetically relabel rows.
            std::vector<int32_t> retained = detail::top_indices(
                total, retain_count, [&](int32_t i) { return utility[static_cast<size_t>(i)]; });
            releases_ += total - retain_count;

            std::vector<int32_t> new_indices(static_cast<size_t>(retain_count) * synapses);
            std::vector<float> new_weights(static_cast<size_t>(retain_count) * synapses);
            std::vector<float> new_labels(static_cast<size_t>(retain_count) * classes);
            std::vector<int32_t> new_observations(static_cast<size_t>(retain_count));
            std::vector<uint64_t> new_relations(static_cast<size_t>(retain_count) * words);
            for (int32_t out_row = 0; out_row < retain_count; ++out_row) {
                const int32_t src = retained[static_cast<size_t>(out_row)];
                std::memcpy(&new_indices[static_cast<size_t>(out_row) * synapses],
                            &merged_indices[static_cast<size_t>(src) * synapses],
                            static_cast<size_t>(synapses) * sizeof(int32_t));
                std::memcpy(&new_weights[static_cast<size_t>(out_row) * synapses],
                            &merged_weights[static_cast<size_t>(src) * synapses],
                            static_cast<size_t>(synapses) * sizeof(float));
                std::memcpy(&new_labels[static_cast<size_t>(out_row) * classes],
                            &merged_labels[static_cast<size_t>(src) * classes],
                            static_cast<size_t>(classes) * sizeof(float));
                new_observations[static_cast<size_t>(out_row)] =
                    merged_observations[static_cast<size_t>(src)];
                std::memcpy(&new_relations[static_cast<size_t>(out_row) * words],
                            &merged_relations[static_cast<size_t>(src) * words],
                            static_cast<size_t>(words) * sizeof(uint64_t));
            }
            merged_indices.swap(new_indices);
            merged_weights.swap(new_weights);
            merged_labels.swap(new_labels);
            merged_observations.swap(new_observations);
            merged_relations.swap(new_relations);
            total = retain_count;
        }

        for (auto& column : columns_) {
            column.load_rows(total, merged_indices.data(), merged_weights.data(),
                             merged_labels.data(), merged_observations.data(),
                             merged_relations.data());
        }
        base_rows_ = total;
        ++sleep_cycles_;
    }

    int32_t dimensions() const noexcept { return dimensions_; }
    int32_t classes() const noexcept { return config_.classes; }
    int32_t live_assemblies() const noexcept { return columns_[0].live(); }
    int64_t steps() const noexcept { return steps_; }
    int64_t sleep_cycles() const noexcept { return sleep_cycles_; }
    int64_t visual_births() const noexcept { return visual_births_; }
    int64_t relation_turnover_releases() const noexcept { return releases_; }

    size_t memory_bytes() const noexcept {
        // Only column 0 is unique mature phenotype after a sleep (all
        // columns hold identical rows immediately post-sleep); report the
        // single consolidated body, matching embodied_bytes's own framing.
        return columns_.empty() ? 0 : columns_[0].memory_bytes();
    }

    const VisionColumn& column(int32_t index) const { return columns_[static_cast<size_t>(index)]; }
    VisionColumn& column(int32_t index) { return columns_[static_cast<size_t>(index)]; }
    int32_t column_count() const noexcept { return config_.columns; }
    int32_t base_rows() const noexcept { return base_rows_; }
    void set_base_rows(int32_t value) noexcept { base_rows_ = value; }

private:
    static constexpr int32_t kMaxEpisodicChainSlots = 8;

    void init_episodic_state() {
        if (config_.episodic_lsh_bits < 1 || config_.episodic_lsh_bits > 30) {
            throw std::runtime_error("episodic_lsh_bits must be in [1, 30]");
        }
        if (config_.episodic_chain_slots < 1 ||
            config_.episodic_chain_slots > kMaxEpisodicChainSlots) {
            throw std::runtime_error("episodic_chain_slots must be in [1, kMaxEpisodicChainSlots]");
        }
        if (config_.episodic_probe_radius < 0 || config_.episodic_probe_radius > 2) {
            throw std::runtime_error("episodic_probe_radius must be in [0, 2]");
        }
        const uint32_t bucket_count = 1u << config_.episodic_lsh_bits;
        episodic_bucket_mask_ = bucket_count - 1u;
        // Fixed seed: the projection must hash the same way after a
        // checkpoint reload as it did when originally constructed (it is
        // regenerated from config on load, not itself serialized -- see
        // VisionIO::write_config/read_config).
        std::mt19937_64 rng(0x564953494f4e4c53ULL);  // "VISIONLS"
        std::normal_distribution<float> normal(0.0f, 1.0f);
        episodic_projection_.assign(
            static_cast<size_t>(config_.episodic_lsh_bits) * static_cast<size_t>(dimensions_), 0.0f);
        for (float& v : episodic_projection_) v = normal(rng);
        const size_t slots =
            static_cast<size_t>(bucket_count) * static_cast<size_t>(config_.episodic_chain_slots);
        episodic_label_.assign(slots, -1);
        episodic_last_touch_.assign(slots, 0);
        episodic_refresh_.assign(slots, 0);
        episodic_used_.assign(bucket_count, 0);
        episodic_step_ = 0;
    }

    // SimHash: sign of each of episodic_lsh_bits random projections of the
    // evidence vector, packed into an integer. Locality-sensitive by
    // construction (nearby vectors agree on most bits), unlike an exact
    // hash of the same data.
    uint32_t episodic_key(const float* evidence) const {
        uint32_t key = 0;
        const int32_t bits = config_.episodic_lsh_bits;
        for (int32_t b = 0; b < bits; ++b) {
            const float* row =
                &episodic_projection_[static_cast<size_t>(b) * static_cast<size_t>(dimensions_)];
            double dot = 0.0;
            for (int32_t i = 0; i < dimensions_; ++i) {
                dot += static_cast<double>(row[i]) * static_cast<double>(evidence[i]);
            }
            if (dot > 0.0) key |= (1u << b);
        }
        return key;
    }

    // Rolling-window semantics identical in spirit to the byte cortex's
    // episodic_write: a fresh (bucket, label) pair gets a new slot; an
    // existing one is refreshed (touch reset, refresh count incremented,
    // crossing episodic_tier3_refresh_threshold makes it eviction-immune).
    // Eviction always prefers a non-tier3 slot, oldest-touched first.
    void episodic_write(const float* evidence, int32_t label) {
        const uint32_t key = episodic_key(evidence) & episodic_bucket_mask_;
        ++episodic_step_;
        const int32_t slots = config_.episodic_chain_slots;
        const size_t base = static_cast<size_t>(key) * static_cast<size_t>(slots);
        int32_t& used = episodic_used_[key];
        for (int32_t i = 0; i < used; ++i) {
            const size_t si = base + static_cast<size_t>(i);
            if (episodic_label_[si] == label) {
                episodic_last_touch_[si] = episodic_step_;
                ++episodic_refresh_[si];
                return;
            }
        }
        if (used < slots) {
            const size_t si = base + static_cast<size_t>(used);
            episodic_label_[si] = label;
            episodic_last_touch_[si] = episodic_step_;
            episodic_refresh_[si] = 1;
            ++used;
            return;
        }
        int32_t victim = -1;
        for (int32_t i = 0; i < slots; ++i) {
            const size_t si = base + static_cast<size_t>(i);
            if (episodic_refresh_[si] >= config_.episodic_tier3_refresh_threshold) continue;
            if (victim < 0 || episodic_last_touch_[si] <
                                  episodic_last_touch_[base + static_cast<size_t>(victim)]) {
                victim = i;
            }
        }
        if (victim < 0) {
            victim = 0;
            for (int32_t i = 1; i < slots; ++i) {
                if (episodic_last_touch_[base + static_cast<size_t>(i)] <
                    episodic_last_touch_[base + static_cast<size_t>(victim)]) {
                    victim = i;
                }
            }
        }
        const size_t vi = base + static_cast<size_t>(victim);
        episodic_label_[vi] = label;
        episodic_last_touch_[vi] = episodic_step_;
        episodic_refresh_[vi] = 1;
    }

    // Adds a recall boost to out_votes[label] for every distinct label
    // found in any bucket within episodic_probe_radius Hamming distance of
    // the query key -- validated standalone (vision_lsh_recall_probe.py)
    // as necessary: requiring the exact key (radius 0) behaves like an
    // exact-match hash in practice (0% recall under any real perturbation
    // once noise is measured correctly), while radius 2 with a reasonable
    // bit count held up gracefully across both increasing perturbation and
    // increasing gap size. Multiple candidate buckets accumulate additively
    // (a label recalled from several nearby buckets earns more support),
    // same pseudo-count shape as the byte cortex's episodic_combine.
    void episodic_recall(const float* evidence, float* out_votes) const {
        const uint32_t key = episodic_key(evidence) & episodic_bucket_mask_;
        const int32_t bits = config_.episodic_lsh_bits;
        const int32_t slots = config_.episodic_chain_slots;
        const double strength = config_.episodic_strength;
        const int32_t classes = config_.classes;
        auto apply_bucket = [&](uint32_t raw_key) {
            const uint32_t k = raw_key & episodic_bucket_mask_;
            const int32_t used = episodic_used_[k];
            const size_t base = static_cast<size_t>(k) * static_cast<size_t>(slots);
            for (int32_t i = 0; i < used; ++i) {
                const size_t si = base + static_cast<size_t>(i);
                const int32_t label = episodic_label_[si];
                if (label < 0 || label >= classes) continue;
                const double refresh = static_cast<double>(episodic_refresh_[si]);
                out_votes[label] += static_cast<float>(refresh / (refresh + strength));
            }
        };
        apply_bucket(key);
        if (config_.episodic_probe_radius >= 1) {
            for (int32_t b = 0; b < bits; ++b) apply_bucket(key ^ (1u << b));
        }
        if (config_.episodic_probe_radius >= 2) {
            for (int32_t b1 = 0; b1 < bits; ++b1) {
                for (int32_t b2 = b1 + 1; b2 < bits; ++b2) {
                    apply_bucket(key ^ (1u << b1) ^ (1u << b2));
                }
            }
        }
    }

    VisionConfig config_;
    VisionReceptor receptor_;
    int32_t dimensions_;
    std::vector<VisionColumn> columns_;
    std::vector<float> slow_context_;
    std::vector<float> label_counts_;
    int64_t steps_ = 0;
    int64_t sleep_cycles_ = 0;
    int64_t external_updates_ = 0;
    int64_t visual_births_ = 0;
    int64_t releases_ = 0;
    int32_t base_rows_ = 0;

    // Sparse episodic recall state (see VisionConfig::episodic_* above).
    // episodic_projection_ is regenerated from config on checkpoint load,
    // not itself serialized; the bucket table IS serialized (VisionIO),
    // since -- unlike the byte cortex's per-round episodic memory -- this
    // is meant to persist across the whole continual-learning stream, only
    // cleared at genuine episode boundaries (reset_episode(), which fires
    // at pass boundaries in the CLEAR-10 harness, not per-sample).
    std::vector<float> episodic_projection_;
    std::vector<int32_t> episodic_label_;
    std::vector<int64_t> episodic_last_touch_;
    std::vector<int32_t> episodic_refresh_;
    std::vector<int32_t> episodic_used_;
    uint32_t episodic_bucket_mask_ = 0;
    int64_t episodic_step_ = 0;
};

}  // namespace alm
