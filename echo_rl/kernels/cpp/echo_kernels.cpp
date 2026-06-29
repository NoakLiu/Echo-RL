#include "include/echo_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace echo_rl {

namespace {

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

uint64_t fnv1a_update(uint64_t hash, const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= FNV_PRIME;
    }
    return hash;
}

}  // namespace

EMAPlanTracker::EMAPlanTracker(int dim, double decay)
    : dim_(dim), decay_(decay), ema_(dim, 0.0f), initialized_(false) {}

void EMAPlanTracker::update(const float* plan, int dim) {
    if (dim != dim_) {
        return;
    }
    if (!initialized_) {
        std::memcpy(ema_.data(), plan, static_cast<size_t>(dim) * sizeof(float));
        initialized_ = true;
        return;
    }
    const float one_minus = static_cast<float>(1.0 - decay_);
    const float decay_f = static_cast<float>(decay_);
    for (int i = 0; i < dim_; ++i) {
        ema_[i] = decay_f * ema_[i] + one_minus * plan[i];
    }
}

void EMAPlanTracker::get_ema(float* out, int dim) const {
    if (dim != dim_ || !initialized_) {
        return;
    }
    std::memcpy(out, ema_.data(), static_cast<size_t>(dim) * sizeof(float));
}

float compute_plan_surprise(
    const float* plan,
    const float* ema_plan,
    int dim,
    float reward,
    float surprise_weight,
    float reward_weight) {
    float sq_dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = plan[i] - ema_plan[i];
        sq_dist += diff * diff;
    }
    return surprise_weight * sq_dist + reward_weight * std::fabs(reward);
}

void compute_plan_surprise_batch(
    const float* plans,
    const float* ema_plan,
    const float* rewards,
    int batch_size,
    int dim,
    float surprise_weight,
    float reward_weight,
    float* out_scores) {
    for (int b = 0; b < batch_size; ++b) {
        out_scores[b] = compute_plan_surprise(
            plans + b * dim,
            ema_plan,
            dim,
            rewards[b],
            surprise_weight,
            reward_weight
        );
    }
}

float compute_attention_bandwidth_cost(int seq_len, float scale) {
    if (seq_len <= 0) {
        return 0.0f;
    }
    // Quadratic attention cost over prefix length
    return scale * static_cast<float>(seq_len) * static_cast<float>(seq_len + 1) * 0.5f;
}

void compute_attention_bandwidth_cost_batch(
    const int* seq_lengths,
    int batch_size,
    float scale,
    float* out_costs) {
    for (int i = 0; i < batch_size; ++i) {
        out_costs[i] = compute_attention_bandwidth_cost(seq_lengths[i], scale);
    }
}

void compute_schedule_priorities(
    const float* rewards,
    const float* queue_times,
    int batch_size,
    float epsilon,
    float* out_priorities) {
    for (int i = 0; i < batch_size; ++i) {
        out_priorities[i] = rewards[i] / (queue_times[i] + epsilon);
    }
}

uint64_t hash_state_vector(const float* data, int dim) {
    uint64_t hash = FNV_OFFSET;
    hash = fnv1a_update(hash, data, static_cast<size_t>(dim) * sizeof(float));
    return hash;
}

std::optional<PrefixMatchResult> find_longest_prefix(
    const uint64_t* sequence_hashes,
    int seq_len,
    const std::vector<std::pair<uint64_t, int>>& cache_entries,
    int min_prefix_len) {
    if (seq_len < min_prefix_len || cache_entries.empty()) {
        return std::nullopt;
    }

    int best_prefix = 0;
    int best_cache_index = -1;

    for (const auto& entry : cache_entries) {
        const uint64_t target = entry.first;
        const int cache_index = entry.second;

        for (int prefix_len = seq_len - 1; prefix_len >= min_prefix_len; --prefix_len) {
            if (sequence_hashes[prefix_len - 1] == target) {
                if (prefix_len > best_prefix) {
                    best_prefix = prefix_len;
                    best_cache_index = cache_index;
                }
                break;
            }
        }
    }

    if (best_cache_index < 0) {
        return std::nullopt;
    }
    return PrefixMatchResult{best_cache_index, best_prefix};
}

SampleResult softmax_priority_sample(
    const float* priorities,
    int num_items,
    int sample_size,
    float temperature,
    float importance_beta,
    uint64_t seed) {
    SampleResult result;
    if (num_items <= 0 || sample_size <= 0) {
        return result;
    }

    sample_size = std::min(sample_size, num_items);
    const float temp = std::max(temperature, 1e-8f);

    std::vector<float> scaled(num_items);
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_items; ++i) {
        scaled[i] = priorities[i] / temp;
        max_val = std::max(max_val, scaled[i]);
    }

    std::vector<float> probs(num_items);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_items; ++i) {
        probs[i] = std::exp(scaled[i] - max_val);
        sum_exp += probs[i];
    }
    for (int i = 0; i < num_items; ++i) {
        probs[i] /= sum_exp;
    }

    std::mt19937_64 rng(seed);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    result.indices.reserve(static_cast<size_t>(sample_size));
    result.importance_weights.reserve(static_cast<size_t>(sample_size));

    for (int s = 0; s < sample_size; ++s) {
        const int idx = dist(rng);
        result.indices.push_back(idx);
        const float prob = std::max(probs[idx], 1e-12f);
        const float weight = std::pow(static_cast<float>(num_items) * prob, -importance_beta);
        result.importance_weights.push_back(weight);
    }

    return result;
}

float compute_bandwidth_efficiency(
    float total_reward,
    float rollout_bandwidth_cost,
    float learner_cost) {
    const float denom = rollout_bandwidth_cost + learner_cost;
    if (denom <= 1e-12f) {
        return 0.0f;
    }
    return total_reward / denom;
}

float compute_temporal_kl(
    const float* current_plan,
    const float* previous_plan,
    int dim,
    float sigma_squared) {
    float sq_dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = current_plan[i] - previous_plan[i];
        sq_dist += diff * diff;
    }
    const float denom = std::max(sigma_squared, 1e-12f);
    return sq_dist / (2.0f * denom);
}

void compute_temporal_kl_batch(
    const float* current_plans,
    const float* previous_plans,
    int batch_size,
    int dim,
    float sigma_squared,
    float* out_kl) {
    for (int b = 0; b < batch_size; ++b) {
        out_kl[b] = compute_temporal_kl(
            current_plans + b * dim,
            previous_plans + b * dim,
            dim,
            sigma_squared
        );
    }
}

}  // namespace echo_rl
