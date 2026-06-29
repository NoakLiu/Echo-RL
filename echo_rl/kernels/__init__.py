"""
EchoRL C++ kernel bindings with pure-Python fallbacks.

Accelerates paper-critical paths:
- EMA plan tracking (τ̄)
- Plan surprise scoring: ||τ_t - τ̄||² + α|r_t|
- KV prefix lookup and bandwidth cost b(s_{1:t})
- Reward/latency rollout scheduling
- Softmax prioritized replay sampling
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_KERNELS_AVAILABLE = False

try:
    from echo_rl.kernels._echo_kernels import (
        EMAPlanTracker as _CEMAPlanTracker,
        compute_attention_bandwidth_cost,
        compute_attention_bandwidth_cost_batch,
        compute_bandwidth_aware_priorities,
        compute_bandwidth_efficiency,
        compute_effective_bandwidth_cost,
        compute_plan_surprise,
        compute_plan_surprise_batch,
        compute_schedule_priorities,
        compute_temporal_kl,
        compute_temporal_kl_batch,
        find_longest_prefix,
        hash_state_vector,
        softmax_priority_sample,
    )

    _KERNELS_AVAILABLE = True
except ImportError:
    logger.debug("EchoRL C++ kernels not built; using Python fallbacks")


def kernels_available() -> bool:
    return _KERNELS_AVAILABLE


class EMAPlanTracker:
    """EMA latent plan τ̄ used for surprise scoring and replay prioritization."""

    def __init__(self, dim: int, decay: float = 0.99):
        self.dim = dim
        self.decay = decay
        if _KERNELS_AVAILABLE:
            self._tracker = _CEMAPlanTracker(dim, decay)
        else:
            self._ema: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        if _KERNELS_AVAILABLE:
            return self._tracker.initialized
        return self._ema is not None

    def update(self, plan: np.ndarray) -> None:
        plan = np.asarray(plan, dtype=np.float32).reshape(-1)
        if _KERNELS_AVAILABLE:
            self._tracker.update(plan)
        else:
            if self._ema is None:
                self._ema = plan.copy()
            else:
                self._ema = self.decay * self._ema + (1.0 - self.decay) * plan

    def get_ema(self) -> np.ndarray:
        if _KERNELS_AVAILABLE:
            return self._tracker.get_ema(self.dim)
        if self._ema is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self._ema.copy()


def plan_surprise(
    plan: np.ndarray,
    ema_plan: np.ndarray,
    reward: float,
    surprise_weight: float = 1.0,
    reward_weight: float = 1.0,
) -> float:
    plan = np.asarray(plan, dtype=np.float32)
    ema_plan = np.asarray(ema_plan, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return float(
            compute_plan_surprise(plan, ema_plan, reward, surprise_weight, reward_weight)
        )
    sq_dist = float(np.sum((plan - ema_plan) ** 2))
    return surprise_weight * sq_dist + reward_weight * abs(reward)


def plan_surprise_batch(
    plans: np.ndarray,
    ema_plan: np.ndarray,
    rewards: np.ndarray,
    surprise_weight: float = 1.0,
    reward_weight: float = 1.0,
) -> np.ndarray:
    plans = np.asarray(plans, dtype=np.float32)
    ema_plan = np.asarray(ema_plan, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return compute_plan_surprise_batch(
            plans, ema_plan, rewards, surprise_weight, reward_weight
        )
    diff = plans - ema_plan
    sq_dist = np.sum(diff * diff, axis=1)
    return surprise_weight * sq_dist + reward_weight * np.abs(rewards)


def attention_bandwidth_cost(seq_len: int, scale: float = 1.0) -> float:
    if _KERNELS_AVAILABLE:
        return float(compute_attention_bandwidth_cost(seq_len, scale))
    if seq_len <= 0:
        return 0.0
    return scale * seq_len * (seq_len + 1) * 0.5


def attention_bandwidth_cost_batch(seq_lengths: np.ndarray, scale: float = 1.0) -> np.ndarray:
    seq_lengths = np.asarray(seq_lengths, dtype=np.int32)
    if _KERNELS_AVAILABLE:
        return compute_attention_bandwidth_cost_batch(seq_lengths, scale)
    return np.array(
        [attention_bandwidth_cost(int(s), scale) for s in seq_lengths], dtype=np.float32
    )


def effective_bandwidth_cost(seq_len: int, reuse_len: int = 0, scale: float = 1.0) -> float:
    """Bandwidth cost after KV prefix reuse: b(s_{1:t}) - b(s_{1:t'})."""
    if _KERNELS_AVAILABLE:
        return float(compute_effective_bandwidth_cost(seq_len, reuse_len, scale))
    full = attention_bandwidth_cost(seq_len, scale)
    prefix = attention_bandwidth_cost(reuse_len, scale)
    return max(0.0, full - prefix)


def bandwidth_aware_priorities(
    rewards: np.ndarray,
    bandwidth_costs: np.ndarray,
    queue_times: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """priority(i) = reward / (bandwidth + queue_time + epsilon)."""
    rewards = np.asarray(rewards, dtype=np.float32)
    bandwidth_costs = np.asarray(bandwidth_costs, dtype=np.float32)
    queue_times = np.asarray(queue_times, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return compute_bandwidth_aware_priorities(
            rewards, bandwidth_costs, queue_times, epsilon
        )
    denom = bandwidth_costs + queue_times + epsilon
    return rewards / denom


def schedule_priorities(
    rewards: np.ndarray, queue_times: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    queue_times = np.asarray(queue_times, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return compute_schedule_priorities(rewards, queue_times, epsilon)
    return rewards / (queue_times + epsilon)


def state_hash(state: np.ndarray) -> int:
    state = np.asarray(state, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return int(hash_state_vector(state))
    data = state.tobytes()
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def prefix_match(
    sequence_hashes: np.ndarray,
    cache_entries: List[Tuple[int, int]],
    min_prefix_len: int = 1,
) -> Tuple[int, int]:
    sequence_hashes = np.asarray(sequence_hashes, dtype=np.uint64)
    if _KERNELS_AVAILABLE:
        cache_index, prefix_len = find_longest_prefix(
            sequence_hashes, cache_entries, min_prefix_len
        )
        return int(cache_index), int(prefix_len)

    best_prefix = 0
    best_index = -1
    for target_hash, cache_index in cache_entries:
        for prefix_len in range(len(sequence_hashes) - 1, min_prefix_len - 1, -1):
            if sequence_hashes[prefix_len - 1] == target_hash:
                if prefix_len > best_prefix:
                    best_prefix = prefix_len
                    best_index = cache_index
                break
    return best_index, best_prefix


def priority_sample(
    priorities: np.ndarray,
    sample_size: int,
    temperature: float = 1.0,
    importance_beta: float = 0.4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    priorities = np.asarray(priorities, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        indices, weights = softmax_priority_sample(
            priorities, sample_size, temperature, importance_beta, seed
        )
        return np.asarray(indices, dtype=np.int64), np.asarray(weights, dtype=np.float32)

    temp = max(temperature, 1e-8)
    scaled = priorities / temp
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(priorities), size=min(sample_size, len(priorities)), p=probs)
    weights = (len(priorities) * probs[indices]) ** (-importance_beta)
    return indices.astype(np.int64), weights.astype(np.float32)


def bandwidth_efficiency(
    total_reward: float, rollout_cost: float, learner_cost: float
) -> float:
    if _KERNELS_AVAILABLE:
        return float(compute_bandwidth_efficiency(total_reward, rollout_cost, learner_cost))
    denom = rollout_cost + learner_cost
    if denom <= 1e-12:
        return 0.0
    return total_reward / denom


def temporal_kl(
    current_plan: np.ndarray, previous_plan: np.ndarray, sigma_squared: float
) -> float:
    current_plan = np.asarray(current_plan, dtype=np.float32)
    previous_plan = np.asarray(previous_plan, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return float(compute_temporal_kl(current_plan, previous_plan, sigma_squared))
    sq_dist = float(np.sum((current_plan - previous_plan) ** 2))
    return sq_dist / (2.0 * max(sigma_squared, 1e-12))


def temporal_kl_batch(
    current_plans: np.ndarray, previous_plans: np.ndarray, sigma_squared: float
) -> np.ndarray:
    current_plans = np.asarray(current_plans, dtype=np.float32)
    previous_plans = np.asarray(previous_plans, dtype=np.float32)
    if _KERNELS_AVAILABLE:
        return compute_temporal_kl_batch(current_plans, previous_plans, sigma_squared)
    diff = current_plans - previous_plans
    sq_dist = np.sum(diff * diff, axis=1)
    return sq_dist / (2.0 * max(sigma_squared, 1e-12))


__all__ = [
    "EMAPlanTracker",
    "attention_bandwidth_cost",
    "attention_bandwidth_cost_batch",
    "bandwidth_aware_priorities",
    "bandwidth_efficiency",
    "effective_bandwidth_cost",
    "kernels_available",
    "plan_surprise",
    "plan_surprise_batch",
    "prefix_match",
    "priority_sample",
    "schedule_priorities",
    "state_hash",
    "temporal_kl",
    "temporal_kl_batch",
]
