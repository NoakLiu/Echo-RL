#!/usr/bin/env python3
"""Demonstrate bandwidth-efficient scheduling and η_bw tracking in EchoRL."""

import numpy as np

from echo_rl.core.bandwidth import (
    BandwidthAwareScheduler,
    BandwidthConfig,
    BandwidthEfficiencyTracker,
)
from echo_rl.kernels import (
    attention_bandwidth_cost,
    bandwidth_efficiency,
    effective_bandwidth_cost,
    kernels_available,
)


def main() -> None:
    print("EchoRL Bandwidth-Efficient RL Demo")
    print(f"C++ kernels available: {kernels_available()}\n")

    seq_len = 128
    reuse_len = 96
    scale = 1.0

    full_cost = attention_bandwidth_cost(seq_len, scale)
    eff_cost = effective_bandwidth_cost(seq_len, reuse_len, scale)
    saved = full_cost - eff_cost

    print("=== KV Prefix Reuse ===")
    print(f"Sequence length:     {seq_len}")
    print(f"Reused prefix:       {reuse_len}")
    print(f"Full bandwidth:      {full_cost:.2f}")
    print(f"Effective bandwidth: {eff_cost:.2f}")
    print(f"Bandwidth saved:     {saved:.2f} ({100 * saved / full_cost:.1f}%)\n")

    config = BandwidthConfig(bandwidth_weight=1.0, queue_weight=1.0)
    scheduler = BandwidthAwareScheduler(config)

    rewards = np.array([2.0, 1.0, 1.5], dtype=np.float32)
    seq_lengths = np.array([128, 64, 96], dtype=np.int32)
    queue_times = np.array([0.5, 0.2, 1.0], dtype=np.float32)
    reuse_lengths = np.array([96, 0, 48], dtype=np.int32)

    priorities = scheduler.compute_priorities_batch(
        rewards, seq_lengths, queue_times, reuse_lengths
    )

    print("=== Bandwidth-Aware Scheduling ===")
    print("priority(i) = r_i / (b_eff + q_i + ε)\n")
    for i, p in enumerate(priorities):
        print(
            f"  Request {i}: r={rewards[i]:.1f}, "
            f"b_eff={effective_bandwidth_cost(int(seq_lengths[i]), int(reuse_lengths[i])):.1f}, "
            f"q={queue_times[i]:.1f} → priority={p:.4f}"
        )

    print("\n=== η_bw Tracking ===")
    tracker = BandwidthEfficiencyTracker(config)
    for step in range(20):
        r = np.random.uniform(0.0, 1.0)
        t = np.random.randint(32, 128)
        t_prime = np.random.randint(0, t // 2 + 1)
        tracker.record_rollout_step(r, seq_len=t, reuse_len=t_prime)
        if step % 5 == 4:
            tracker.record_learner_update(weighted_pg_loss=0.01 + np.random.uniform(0, 0.02))

    snapshot = tracker.snapshot()
    summary = tracker.get_summary()
    print(f"Total reward:              {summary['total_reward']:.3f}")
    print(f"Effective rollout cost:      {summary['total_effective_rollout_cost']:.2f}")
    print(f"Total bandwidth saved:     {summary['total_bandwidth_saved']:.2f}")
    print(f"KV reuse rate:             {summary['kv_reuse_rate']:.2%}")
    print(f"η_bw:                      {summary['eta_bw']:.6f}")

    eta_manual = bandwidth_efficiency(
        snapshot.total_reward,
        snapshot.total_effective_rollout_cost,
        snapshot.total_learner_cost,
    )
    print(f"η_bw (verified):           {eta_manual:.6f}")


if __name__ == "__main__":
    main()
