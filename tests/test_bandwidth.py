"""Tests for bandwidth-efficient RL components."""

import numpy as np

from echo_rl.core.bandwidth import (
    BandwidthAwareScheduler,
    BandwidthConfig,
    BandwidthEfficiencyTracker,
)
from echo_rl.kernels import (
    bandwidth_aware_priorities,
    effective_bandwidth_cost,
)


def test_effective_bandwidth_cost_no_reuse():
    full = effective_bandwidth_cost(10, reuse_len=0)
    assert full == 55.0  # 10 * 11 / 2


def test_effective_bandwidth_cost_with_reuse():
    full = effective_bandwidth_cost(10, reuse_len=0)
    partial = effective_bandwidth_cost(10, reuse_len=5)
    prefix = effective_bandwidth_cost(5, reuse_len=0)
    assert abs(partial - (full - prefix)) < 1e-5
    assert partial < full


def test_bandwidth_aware_priorities():
    rewards = np.array([2.0, 1.0], dtype=np.float32)
    bandwidth = np.array([10.0, 5.0], dtype=np.float32)
    queue = np.array([1.0, 3.0], dtype=np.float32)
    priorities = bandwidth_aware_priorities(rewards, bandwidth, queue, epsilon=1e-6)
    assert priorities[0] > priorities[1]


def test_bandwidth_aware_scheduler():
    scheduler = BandwidthAwareScheduler(BandwidthConfig())
    high_reuse = scheduler.compute_priority(
        reward=1.0, seq_len=100, queue_time=0.5, reuse_len=90
    )
    low_reuse = scheduler.compute_priority(
        reward=1.0, seq_len=100, queue_time=0.5, reuse_len=0
    )
    assert high_reuse > low_reuse


def test_bandwidth_efficiency_tracker():
    tracker = BandwidthEfficiencyTracker(BandwidthConfig())
    tracker.record_rollout_step(reward=1.0, seq_len=10, reuse_len=5)
    tracker.record_rollout_step(reward=0.5, seq_len=8, reuse_len=4)
    tracker.record_learner_update(0.1)
    snapshot = tracker.snapshot()

    assert snapshot.total_reward == 1.5
    assert snapshot.total_effective_rollout_cost > 0
    assert snapshot.total_bandwidth_saved > 0
    assert snapshot.eta_bw > 0
    assert snapshot.kv_reuse_rate > 0


def test_bandwidth_scheduler_batch():
    scheduler = BandwidthAwareScheduler(BandwidthConfig())
    rewards = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    seq_lengths = np.array([64, 32, 128], dtype=np.int32)
    queue_times = np.array([0.1, 0.5, 0.2], dtype=np.float32)
    reuse = np.array([48, 0, 64], dtype=np.int32)

    priorities = scheduler.compute_priorities_batch(
        rewards, seq_lengths, queue_times, reuse
    )
    assert priorities.shape == (3,)
    assert np.all(priorities > 0)
