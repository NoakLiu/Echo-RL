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

from .core.latent_planning import LatentPlanningOptimizer, PlanningConfig, TrajectoryEncoder
from .core.async_execution import AsyncExecutionEngine, ExecutionConfig, KVCacheManager
from .core.prioritized_replay import HotColdBuffer, PrioritizedReplayBuffer, ReplayConfig
from .core.ppo_learner import GAECalculator, PPOConfig, PPOLearner
from .core.bandwidth import BandwidthAwareScheduler, BandwidthEfficiencyTracker
from .environments.base import EchoRLEnvironment
from .environments.alfworld import ALFWorldEnvironment
from .environments.webshop import WebShopEnvironment
from .environments.cruxeval import CRUXEvalEnvironment
from .environments.arc import ARCEnvironment
from .environments.minigrid import MiniGridEnvironment
from .training.trainer import EchoRLTrainer, TrainingConfig
from .evaluation.benchmark import BenchmarkConfig, EchoRLBenchmark
from .utils.monitoring import MetricsCollector, PerformanceMonitor

__all__ = [
    "LatentPlanningOptimizer",
    "PlanningConfig",
    "TrajectoryEncoder",
    "AsyncExecutionEngine",
    "ExecutionConfig",
    "KVCacheManager",
    "PrioritizedReplayBuffer",
    "HotColdBuffer",
    "ReplayConfig",
    "PPOLearner",
    "PPOConfig",
    "GAECalculator",
    "EchoRLEnvironment",
    "ALFWorldEnvironment",
    "WebShopEnvironment",
    "CRUXEvalEnvironment",
    "ARCEnvironment",
    "MiniGridEnvironment",
    "EchoRLTrainer",
    "TrainingConfig",
    "EchoRLBenchmark",
    "BenchmarkConfig",
    "PerformanceMonitor",
    "MetricsCollector",
    "BandwidthEfficiencyTracker",
    "BandwidthAwareScheduler",
]
