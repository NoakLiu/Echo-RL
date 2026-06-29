#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <optional>

namespace echo_rl {

// EMA plan tracker: τ̄_{t+1} = decay * τ̄_t + (1 - decay) * τ_t
class EMAPlanTracker {
public:
    explicit EMAPlanTracker(int dim, double decay = 0.99);

    void update(const float* plan, int dim);
    void get_ema(float* out, int dim) const;
    bool initialized() const { return initialized_; }
    int dim() const { return dim_; }

private:
    int dim_;
    double decay_;
    std::vector<float> ema_;
    bool initialized_;
};

// ||τ - τ̄||² + α * |r|
float compute_plan_surprise(
    const float* plan,
    const float* ema_plan,
    int dim,
    float reward,
    float surprise_weight,
    float reward_weight
);

void compute_plan_surprise_batch(
    const float* plans,
    const float* ema_plan,
    const float* rewards,
    int batch_size,
    int dim,
    float surprise_weight,
    float reward_weight,
    float* out_scores
);

// Attention/KV bandwidth cost: b(s_{1:t}) ≈ Σ_{i=1}^{t} i = t(t+1)/2 (scaled)
float compute_attention_bandwidth_cost(int seq_len, float scale = 1.0f);

void compute_attention_bandwidth_cost_batch(
    const int* seq_lengths,
    int batch_size,
    float scale,
    float* out_costs
);

// priority(i) = reward / (queue_time + epsilon)
void compute_schedule_priorities(
    const float* rewards,
    const float* queue_times,
    int batch_size,
    float epsilon,
    float* out_priorities
);

// Effective bandwidth with KV prefix reuse:
// b_eff(s_{1:t}, t') = b(s_{1:t}) - b(s_{1:t'})
float compute_effective_bandwidth_cost(
    int seq_len,
    int reuse_len,
    float scale = 1.0f
);

// Bandwidth-aware scheduling: priority(i) = reward / (bandwidth + queue_time + epsilon)
void compute_bandwidth_aware_priorities(
    const float* rewards,
    const float* bandwidth_costs,
    const float* queue_times,
    int batch_size,
    float epsilon,
    float* out_priorities
);

// Longest prefix match against cached sequence hashes
struct PrefixMatchResult {
    int cache_index;
    int prefix_length;
};

std::optional<PrefixMatchResult> find_longest_prefix(
    const uint64_t* sequence_hashes,
    int seq_len,
    const std::vector<std::pair<uint64_t, int>>& cache_entries,
    int min_prefix_len = 1
);

uint64_t hash_state_vector(const float* data, int dim);

// Softmax-weighted sampling with importance correction
struct SampleResult {
    std::vector<int> indices;
    std::vector<float> importance_weights;
};

SampleResult softmax_priority_sample(
    const float* priorities,
    int num_items,
    int sample_size,
    float temperature,
    float importance_beta,
    uint64_t seed
);

// Bandwidth efficiency: η_bw = total_reward / (rollout_cost + learner_cost)
float compute_bandwidth_efficiency(
    float total_reward,
    float rollout_bandwidth_cost,
    float learner_cost
);

// Temporal KL: ||τ_t - τ_{t-1}||² / (2σ²)
float compute_temporal_kl(
    const float* current_plan,
    const float* previous_plan,
    int dim,
    float sigma_squared
);

void compute_temporal_kl_batch(
    const float* current_plans,
    const float* previous_plans,
    int batch_size,
    int dim,
    float sigma_squared,
    float* out_kl
);

}  // namespace echo_rl
