"""Tests for EchoRL C++ kernels and Python fallbacks."""

import numpy as np
import pytest

from echo_rl.kernels import (
    EMAPlanTracker,
    attention_bandwidth_cost,
    bandwidth_efficiency,
    kernels_available,
    plan_surprise,
    plan_surprise_batch,
    prefix_match,
    priority_sample,
    schedule_priorities,
    state_hash,
    temporal_kl,
)


def test_ema_plan_tracker():
    tracker = EMAPlanTracker(dim=4, decay=0.9)
    plan = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    tracker.update(plan)
    assert tracker.initialized
    ema = tracker.get_ema()
    np.testing.assert_allclose(ema, plan, rtol=1e-5)

    tracker.update(np.zeros(4, dtype=np.float32))
    ema2 = tracker.get_ema()
    assert ema2[0] < 1.0


def test_plan_surprise():
    plan = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ema = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    score = plan_surprise(plan, ema, reward=0.5, surprise_weight=1.0, reward_weight=0.1)
    expected = 14.0 + 0.05
    assert abs(score - expected) < 1e-4


def test_plan_surprise_batch():
    plans = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ema = np.zeros(2, dtype=np.float32)
    rewards = np.array([1.0, -1.0], dtype=np.float32)
    scores = plan_surprise_batch(plans, ema, rewards)
    assert scores.shape == (2,)
    assert scores[0] > scores[1] or scores[0] == scores[1]


def test_attention_bandwidth_cost():
    assert attention_bandwidth_cost(0) == 0.0
    assert attention_bandwidth_cost(3) == 6.0


def test_schedule_priorities():
    rewards = np.array([2.0, 1.0], dtype=np.float32)
    queue = np.array([1.0, 3.0], dtype=np.float32)
    priorities = schedule_priorities(rewards, queue, epsilon=1e-6)
    assert priorities[0] > priorities[1]


def test_prefix_match():
    hashes = np.array([10, 20, 30], dtype=np.uint64)
    cache = [(20, 0), (10, 1)]
    idx, length = prefix_match(hashes, cache)
    assert idx == 0
    assert length == 2


def test_priority_sample():
    priorities = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    indices, weights = priority_sample(priorities, sample_size=4, seed=42)
    assert len(indices) == 4
    assert len(weights) == 4
    assert all(w > 0 for w in weights)


def test_temporal_kl():
    current = np.array([1.0, 0.0], dtype=np.float32)
    previous = np.array([0.0, 0.0], dtype=np.float32)
    kl = temporal_kl(current, previous, sigma_squared=0.01)
    assert abs(kl - 50.0) < 1e-3


def test_bandwidth_efficiency():
    eta = bandwidth_efficiency(total_reward=10.0, rollout_cost=5.0, learner_cost=5.0)
    assert eta == 1.0


def test_effective_bandwidth_cost():
    from echo_rl.kernels import effective_bandwidth_cost

    assert effective_bandwidth_cost(0, 0) == 0.0
    full = effective_bandwidth_cost(3, 0)
    assert full == 6.0
    partial = effective_bandwidth_cost(3, 2)
    assert partial == 3.0  # 6 - 3


def test_bandwidth_aware_priorities_kernel():
    from echo_rl.kernels import bandwidth_aware_priorities

    rewards = np.array([2.0, 1.0], dtype=np.float32)
    bandwidth = np.array([4.0, 2.0], dtype=np.float32)
    queue = np.array([1.0, 1.0], dtype=np.float32)
    priorities = bandwidth_aware_priorities(rewards, bandwidth, queue)
    assert priorities[0] > priorities[1]  # 2/(4+1) > 1/(2+1)


def test_state_hash_deterministic():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert state_hash(vec) == state_hash(vec)


@pytest.mark.skipif(not kernels_available(), reason="C++ kernels not built")
def test_kernels_available():
    assert kernels_available()
