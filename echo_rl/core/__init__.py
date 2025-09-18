"""
Core EchoRL Components

This module contains the three key innovations of EchoRL:
1. Latent Planning Optimization
2. Asynchronous Execution Engine  
3. Prioritized Replay Buffer
"""

from .latent_planning import LatentPlanningOptimizer, TrajectoryEncoder
from .async_execution import AsyncExecutionEngine, KVCacheManager, LatencyScheduler
from .prioritized_replay import PrioritizedReplayBuffer, HotColdBuffer, SurpriseCalculator
from .ppo_learner import PPOLearner, GAECalculator, PolicyNetwork

__all__ = [
    "LatentPlanningOptimizer",
    "TrajectoryEncoder",
    "AsyncExecutionEngine", 
    "KVCacheManager",
    "LatencyScheduler",
    "PrioritizedReplayBuffer",
    "HotColdBuffer",
    "SurpriseCalculator",
    "PPOLearner",
    "GAECalculator",
    "PolicyNetwork"
]
