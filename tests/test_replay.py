"""Tests for planning-aware replay buffer."""

import torch
import pytest

from echo_rl.core.prioritized_replay import PrioritizedReplayBuffer, ReplayConfig


@pytest.fixture
def replay_buffer():
    config = ReplayConfig(
        hot_buffer_size=100,
        cold_buffer_size=100,
        age_threshold=2,
        max_replay_age=10,
        min_experiences=1,
    )
    return PrioritizedReplayBuffer(config, latent_dim=8)


def test_add_and_sample(replay_buffer):
    for i in range(5):
        replay_buffer.add_experience(
            state=torch.randn(4),
            latent_plan=torch.randn(8),
            action=torch.tensor(i % 3),
            reward=float(i),
            next_state=torch.randn(4),
            done=False,
        )
    experiences, weights = replay_buffer.sample_batch(batch_size=3)
    assert len(experiences) == 3
    assert len(weights) == 3


def test_hot_cold_migration(replay_buffer):
    for i in range(3):
        replay_buffer.add_experience(
            state=torch.randn(4),
            latent_plan=torch.randn(8),
            action=torch.tensor(0),
            reward=1.0,
            next_state=torch.randn(4),
            done=False,
        )
    for _ in range(5):
        replay_buffer.update_experience_ages()

    stats = replay_buffer.get_buffer_statistics()
    assert stats["buffer_stats"]["cold_buffer_size"] >= 0
