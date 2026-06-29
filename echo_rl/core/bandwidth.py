"""
Bandwidth efficiency metrics from EchoRL paper.

η_bw(π) = E[Σ r_t] / (E[Σ b(s_{1:t})] + E_B[w|ℓ_PG|])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List
from collections import deque

from ..kernels import attention_bandwidth_cost, bandwidth_efficiency


@dataclass
class BandwidthMetrics:
    """Rolling bandwidth efficiency statistics."""

    total_reward: float = 0.0
    total_rollout_cost: float = 0.0
    total_learner_cost: float = 0.0
    eta_bw: float = 0.0
    num_updates: int = 0


class BandwidthEfficiencyTracker:
    """Tracks learning return per unit rollout and learner bandwidth."""

    def __init__(self, history_size: int = 1000, attention_scale: float = 1.0):
        self.attention_scale = attention_scale
        self.history: Deque[BandwidthMetrics] = deque(maxlen=history_size)
        self.cumulative = BandwidthMetrics()

    def record_rollout_step(self, reward: float, seq_len: int) -> None:
        cost = attention_bandwidth_cost(seq_len, self.attention_scale)
        self.cumulative.total_reward += reward
        self.cumulative.total_rollout_cost += cost

    def record_learner_update(self, weighted_pg_loss: float) -> None:
        self.cumulative.total_learner_cost += abs(weighted_pg_loss)

    def snapshot(self) -> BandwidthMetrics:
        eta = bandwidth_efficiency(
            self.cumulative.total_reward,
            self.cumulative.total_rollout_cost,
            self.cumulative.total_learner_cost,
        )
        metrics = BandwidthMetrics(
            total_reward=self.cumulative.total_reward,
            total_rollout_cost=self.cumulative.total_rollout_cost,
            total_learner_cost=self.cumulative.total_learner_cost,
            eta_bw=eta,
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
                "total_learner_cost": self.cumulative.total_learner_cost,
            }
        recent = self.history[-1]
        return {
            "eta_bw": recent.eta_bw,
            "total_reward": recent.total_reward,
            "total_rollout_cost": recent.total_rollout_cost,
            "total_learner_cost": recent.total_learner_cost,
            "num_snapshots": len(self.history),
        }
