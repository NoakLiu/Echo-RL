"""
EchoRL: Learning to Plan through Experience for Bandwidth-Efficient Reinforcement Learning

A system framework that bridges reaction and planning in real-time RL through 
experience-grounded infrastructure with bandwidth-efficient execution.

Key Components:
1. Latent Planning Optimization - structured rollout with continuation-based reasoning
2. Asynchronous Execution Engine - KV-cache sharing, bandwidth-aware scheduling
3. Prioritized Replay Buffer - stratified hot/cold buffers for improved RL training efficiency
4. Bandwidth Efficiency Tracking - η_bw metric with effective rollout bandwidth
"""

__version__ = "1.0.0"
__author__ = "EchoRL Team"

from .core.latent_planning import LatentPlanningOptimizer, TrajectoryEncoder
from .core.async_execution import AsyncExecutionEngine, KVCacheManager
from .core.prioritized_replay import PrioritizedReplayBuffer, HotColdBuffer
from .core.ppo_learner import PPOLearner, GAECalculator
from .environments.base import EchoRLEnvironment
from .environments.alfworld import ALFWorldEnvironment
from .environments.webshop import WebShopEnvironment
from .environments.cruxeval import CRUXEvalEnvironment
from .core.bandwidth import BandwidthEfficiencyTracker, BandwidthAwareScheduler
from .training.trainer import EchoRLTrainer
from .evaluation.benchmark import EchoRLBenchmark
from .utils.monitoring import PerformanceMonitor, MetricsCollector

__all__ = [
    "LatentPlanningOptimizer",
    "TrajectoryEncoder", 
    "AsyncExecutionEngine",
    "KVCacheManager",
    "PrioritizedReplayBuffer",
    "HotColdBuffer",
    "PPOLearner",
    "GAECalculator",
    "EchoRLEnvironment",
    "ALFWorldEnvironment",
    "WebShopEnvironment", 
    "CRUXEvalEnvironment",
    "EchoRLTrainer",
    "EchoRLBenchmark",
    "PerformanceMonitor",
    "MetricsCollector",
    "BandwidthEfficiencyTracker",
    "BandwidthAwareScheduler",
]
