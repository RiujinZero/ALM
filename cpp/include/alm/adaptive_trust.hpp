// Native port of the Python adaptive-competition layer
// (opponent_byte_stream.py's _backoff_weights/_credit_context_trust,
// temporal_difference_credit.LocalValueEstimate, causal_attribution's
// SelfCausalAttribution). Same equations, same update order, so that a
// context-level backoff decision lands on the same bits either
// implementation makes it in.
//
// Only the residual/context level uses this -- SequenceCortexOpponentLearner
// never passes a non-default child_gain into its own sequence-bank backoff,
// so trust has exactly one attachment point, matching the Python source.
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace alm {

constexpr double kPi = 3.14159265358979323846;

// LocalValueEstimate: a per-key running value, TD(0)-updated.
class LocalValueEstimate {
public:
    explicit LocalValueEstimate(double learning_rate = 0.1, double initial_value = 0.0)
        : learning_rate_(learning_rate), initial_value_(initial_value) {}

    double value(uint64_t key) const {
        const auto it = values_.find(key);
        return it == values_.end() ? initial_value_ : it->second;
    }

    // next_key == UINT64_MAX means "no bootstrap" (Python's next_key=None).
    double update(uint64_t key, double outcome, uint64_t next_key, bool has_next,
                  double gamma) {
        const double current = value(key);
        const double bootstrap = has_next ? value(next_key) : 0.0;
        const double delta = outcome + gamma * bootstrap - current;
        values_[key] = current + learning_rate_ * delta;
        return delta;
    }

    const std::unordered_map<uint64_t, double>& values() const noexcept { return values_; }
    std::unordered_map<uint64_t, double>& values() noexcept { return values_; }

private:
    double learning_rate_;
    double initial_value_;
    std::unordered_map<uint64_t, double> values_;
};

// Bayesian self-vs-other attribution over a non-negative effect size.
class SelfCausalAttribution {
public:
    SelfCausalAttribution(double prior_self = 0.5, double self_prior_scale = 1.0,
                          double other_prior_scale = 0.1)
        : prior_self_(prior_self),
          self_prior_scale_(self_prior_scale),
          other_prior_scale_(other_prior_scale) {}

    double posterior_self(double d) const {
        d = d < 0.0 ? 0.0 : d;
        const double log_self = std::log(prior_self_) + half_normal_log_density(d, self_scale());
        const double log_other =
            std::log(1.0 - prior_self_) + half_normal_log_density(d, other_scale());
        const double peak = log_self > log_other ? log_self : log_other;
        const double total = std::exp(log_self - peak) + std::exp(log_other - peak);
        return std::exp(log_self - peak) / total;
    }

    void confirm(double d, bool is_self) {
        d = d < 0.0 ? 0.0 : d;
        ++observations_;
        if (is_self) {
            self_sum_squares_ += d * d;
            ++self_count_;
            ++confirmed_self_;
        } else {
            other_sum_squares_ += d * d;
            ++other_count_;
            ++confirmed_other_;
        }
    }

    double self_scale() const {
        return self_count_ == 0 ? self_prior_scale_
                                : (std::sqrt(self_sum_squares_ / self_count_));
    }
    double other_scale() const {
        return other_count_ == 0 ? other_prior_scale_
                                 : (std::sqrt(other_sum_squares_ / other_count_));
    }

    double prior_self() const noexcept { return prior_self_; }
    double self_prior_scale() const noexcept { return self_prior_scale_; }
    double other_prior_scale() const noexcept { return other_prior_scale_; }
    double self_sum_squares() const noexcept { return self_sum_squares_; }
    int64_t self_count() const noexcept { return self_count_; }
    double other_sum_squares() const noexcept { return other_sum_squares_; }
    int64_t other_count() const noexcept { return other_count_; }
    int64_t observations() const noexcept { return observations_; }
    int64_t confirmed_self() const noexcept { return confirmed_self_; }
    int64_t confirmed_other() const noexcept { return confirmed_other_; }

    void set_scale_state(double self_sum_squares, int64_t self_count,
                         double other_sum_squares, int64_t other_count,
                         int64_t observations, int64_t confirmed_self,
                         int64_t confirmed_other) {
        self_sum_squares_ = self_sum_squares;
        self_count_ = self_count;
        other_sum_squares_ = other_sum_squares;
        other_count_ = other_count;
        observations_ = observations;
        confirmed_self_ = confirmed_self;
        confirmed_other_ = confirmed_other;
    }

private:
    static double half_normal_log_density(double d, double scale) {
        return 0.5 * std::log(2.0 / kPi) - std::log(scale) - (d * d) / (2.0 * scale * scale);
    }

    double prior_self_;
    double self_prior_scale_;
    double other_prior_scale_;
    double self_sum_squares_ = 0.0;
    int64_t self_count_ = 0;
    double other_sum_squares_ = 0.0;
    int64_t other_count_ = 0;
    int64_t observations_ = 0;
    int64_t confirmed_self_ = 0;
    int64_t confirmed_other_ = 0;
};

// Everything the residual/context backoff needs to learn and apply trust.
struct TrustState {
    TrustState() : context_value(0.2) {}
    explicit TrustState(double learning_rate) : context_value(learning_rate) {}

    std::unordered_map<uint64_t, double> context_trust;
    LocalValueEstimate context_value;
    SelfCausalAttribution context_attribution;
    std::optional<std::pair<uint64_t, double>> pending_context_credit;

    double trust_of(uint64_t key) const {
        const auto it = context_trust.find(key);
        return it == context_trust.end() ? 1.0 : it->second;
    }

    // Mirrors _credit_context_trust exactly at delta_clip=2.0/floor=0.1/
    // ceiling=10.0 (the original hardcoded constants, kept as defaults):
    // finalizes the PREVIOUS key's credit now that the current key is
    // known (the one-visit-delayed TD bootstrap), gated by causal
    // attribution before trust actually moves. delta_clip/floor/ceiling
    // now configurable (CortexConfig::trust_delta_clip/trust_floor/
    // trust_ceiling) -- a real run showed this saturating hard at the
    // default clip (49.8% of learned values sitting at the exact
    // ceiling, only 0.5% near the neutral 1.0), i.e. the multiplicative
    // update swings most contexts to an extreme after a handful of
    // updates rather than settling on a graded value. Tightening
    // delta_clip is the direct lever to test whether that's a property
    // of the update rule (fixable) or the underlying signal (not).
    void credit(uint64_t key, double basal, double child_probability, double gamma,
               double attribution_gate_threshold, double delta_clip = 2.0,
               double trust_floor = 0.1, double trust_ceiling = 10.0) {
        const double utility =
            std::log2(child_probability > 1e-12 ? child_probability : 1e-12) -
            std::log2(basal > 1e-12 ? basal : 1e-12);
        if (pending_context_credit.has_value()) {
            const auto [previous_key, previous_utility] = *pending_context_credit;
            const double delta =
                context_value.update(previous_key, previous_utility, key, true, gamma);
            const double effect = previous_utility < 0.0 ? -previous_utility : previous_utility;
            context_attribution.confirm(effect, delta > 0.0);
            if (context_attribution.posterior_self(effect) > attribution_gate_threshold) {
                const double clipped = delta < -delta_clip ? -delta_clip : (delta > delta_clip ? delta_clip : delta);
                const double multiplier = std::exp(clipped);
                const double previous_trust = trust_of(previous_key);
                double updated = previous_trust * multiplier;
                updated = updated < trust_floor ? trust_floor : (updated > trust_ceiling ? trust_ceiling : updated);
                context_trust[previous_key] = updated;
            }
        }
        pending_context_credit = std::make_pair(key, utility);
    }
};

}  // namespace alm
