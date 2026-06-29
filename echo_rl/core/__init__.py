"""
Core EchoRL Components

This module contains the three key innovations of EchoRL:
1. Latent Planning Optimization
2. Asynchronous Execution Engine  
3. Prioritized Replay Buffer
"""

from .latent_planning import LatentPlanningOptimizer, TrajectoryEncoder, SoftPrefixAdapter
from .async_execution import AsyncExecutionEngine, KVCacheManager, LatencyScheduler
from .prioritized_replay import PrioritizedReplayBuffer, HotColdBuffer, SurpriseCalculator
from .ppo_learner import PPOLearner, GAECalculator, PolicyNetwork
from .bandwidth import (
    BandwidthAwareScheduler,
    BandwidthConfig,
    BandwidthEfficiencyTracker,
    BandwidthMetrics,
)

__all__ = [
    "LatentPlanningOptimizer",
    "TrajectoryEncoder",
    "SoftPrefixAdapter",
    "AsyncExecutionEngine",
    "KVCacheManager",
    "LatencyScheduler",
    "PrioritizedReplayBuffer",
    "HotColdBuffer",
    "SurpriseCalculator",
    "PPOLearner",
    "GAECalculator",
    "PolicyNetwork",
    "BandwidthEfficiencyTracker",
    "BandwidthMetrics",
    "BandwidthConfig",
    "BandwidthAwareScheduler",
]
