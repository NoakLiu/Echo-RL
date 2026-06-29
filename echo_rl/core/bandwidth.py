"""
Bandwidth-efficient reinforcement learning for EchoRL.

Core metric:
    η_bw(π) = E[Σ r_t] / (E[Σ b_eff(s_{1:t})] + E_B[w|ℓ_PG|])

where b_eff(s_{1:t}, t') = b(s_{1:t}) - b(s_{1:t'}) accounts for KV prefix reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
from collections import deque

from ..kernels import (
    attention_bandwidth_cost,
    bandwidth_aware_priorities,
    bandwidth_efficiency,
    effective_bandwidth_cost,
)


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth-efficient RL."""

    attention_scale: float = 1.0
    schedule_epsilon: float = 1e-6
    bandwidth_weight: float = 1.0
    queue_weight: float = 1.0
    history_size: int = 1000
    enable_bandwidth_scheduling: bool = True


@dataclass
class BandwidthMetrics:
    """Rolling bandwidth efficiency statistics."""

    total_reward: float = 0.0
    total_rollout_cost: float = 0.0
    total_effective_rollout_cost: float = 0.0
    total_bandwidth_saved: float = 0.0
    total_learner_cost: float = 0.0
    eta_bw: float = 0.0
    kv_reuse_rate: float = 0.0
    num_rollout_steps: int = 0
    num_updates: int = 0


class BandwidthEfficiencyTracker:
    """Tracks learning return per unit effective rollout and learner bandwidth."""

    def __init__(
        self,
        config: Optional[BandwidthConfig] = None,
        history_size: Optional[int] = None,
        attention_scale: Optional[float] = None,
    ):
        self.config = config or BandwidthConfig()
        if history_size is not None:
            self.config.history_size = history_size
        if attention_scale is not None:
            self.config.attention_scale = attention_scale

        self.history: Deque[BandwidthMetrics] = deque(maxlen=self.config.history_size)
        self.cumulative = BandwidthMetrics()
        self._total_reuse_length = 0
        self._total_seq_length = 0

    def record_rollout_step(
        self,
        reward: float,
        seq_len: int,
        reuse_len: int = 0,
    ) -> float:
        """Record one rollout step; returns effective bandwidth cost."""
        full_cost = attention_bandwidth_cost(seq_len, self.config.attention_scale)
        eff_cost = effective_bandwidth_cost(
            seq_len, reuse_len, self.config.attention_scale
        )
        saved = max(0.0, full_cost - eff_cost)

        self.cumulative.total_reward += reward
        self.cumulative.total_rollout_cost += full_cost
        self.cumulative.total_effective_rollout_cost += eff_cost
        self.cumulative.total_bandwidth_saved += saved
        self.cumulative.num_rollout_steps += 1

        self._total_reuse_length += reuse_len
        self._total_seq_length += seq_len
        return eff_cost

    def record_learner_update(self, weighted_pg_loss: float) -> None:
        self.cumulative.total_learner_cost += abs(weighted_pg_loss)

    def snapshot(self) -> BandwidthMetrics:
        rollout_cost = self.cumulative.total_effective_rollout_cost
        eta = bandwidth_efficiency(
            self.cumulative.total_reward,
            rollout_cost,
            self.cumulative.total_learner_cost,
        )
        kv_reuse = (
            self._total_reuse_length / max(self._total_seq_length, 1)
        )
        metrics = BandwidthMetrics(
            total_reward=self.cumulative.total_reward,
            total_rollout_cost=self.cumulative.total_rollout_cost,
            total_effective_rollout_cost=rollout_cost,
            total_bandwidth_saved=self.cumulative.total_bandwidth_saved,
            total_learner_cost=self.cumulative.total_learner_cost,
            eta_bw=eta,
            kv_reuse_rate=kv_reuse,
            num_rollout_steps=self.cumulative.num_rollout_steps,
            num_updates=self.cumulative.num_updates + 1,
        )
        self.history.append(metrics)
        self.cumulative.num_updates += 1
        return metrics

    def get_summary(self) -> Dict[str, float]:
        if not self.history:
            return {
                "eta_bw": 0.0,
                "total_reward": self.cumulative.total_reward,
                "total_rollout_cost": self.cumulative.total_rollout_cost,
                "total_effective_rollout_cost": self.cumulative.total_effective_rollout_cost,
                "total_bandwidth_saved": self.cumulative.total_bandwidth_saved,
                "total_learner_cost": self.cumulative.total_learner_cost,
                "kv_reuse_rate": 0.0,
            }
        recent = self.history[-1]
        return {
            "eta_bw": recent.eta_bw,
            "total_reward": recent.total_reward,
            "total_rollout_cost": recent.total_rollout_cost,
            "total_effective_rollout_cost": recent.total_effective_rollout_cost,
            "total_bandwidth_saved": recent.total_bandwidth_saved,
            "total_learner_cost": recent.total_learner_cost,
            "kv_reuse_rate": recent.kv_reuse_rate,
            "num_snapshots": len(self.history),
        }


class BandwidthAwareScheduler:
    """
    Bandwidth-aware rollout scheduler.

    priority(i) = r_i / (w_b * b_eff(s_{1:t}) + w_q * q_i + ε)
    """

    def __init__(self, config: Optional[BandwidthConfig] = None):
        self.config = config or BandwidthConfig()

    def compute_priority(
        self,
        reward: float,
        seq_len: int,
        queue_time: float,
        reuse_len: int = 0,
    ) -> float:
        eff_cost = effective_bandwidth_cost(
            seq_len, reuse_len, self.config.attention_scale
        )
        weighted_bw = self.config.bandwidth_weight * eff_cost
        weighted_queue = self.config.queue_weight * queue_time
        denom = weighted_bw + weighted_queue + self.config.schedule_epsilon
        return reward / denom

    def compute_priorities_batch(
        self,
        rewards: np.ndarray,
        seq_lengths: np.ndarray,
        queue_times: np.ndarray,
        reuse_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float32)
        seq_lengths = np.asarray(seq_lengths, dtype=np.int32)
        queue_times = np.asarray(queue_times, dtype=np.float32)
        n = len(rewards)

        if reuse_lengths is None:
            reuse_lengths = np.zeros(n, dtype=np.int32)
        else:
            reuse_lengths = np.asarray(reuse_lengths, dtype=np.int32)

        bandwidth_costs = np.array(
            [
                effective_bandwidth_cost(
                    int(seq_lengths[i]),
                    int(reuse_lengths[i]),
                    self.config.attention_scale,
                )
                for i in range(n)
            ],
            dtype=np.float32,
        )
        bandwidth_costs *= self.config.bandwidth_weight
        queue_times = queue_times * self.config.queue_weight

        return bandwidth_aware_priorities(
            rewards, bandwidth_costs, queue_times, self.config.schedule_epsilon
        )
