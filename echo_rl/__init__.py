"""
EchoRL: Learning to Plan through Experience for Efficient Reinforcement Learning

A system framework that bridges reaction and planning in real-time RL through 
experience-grounded infrastructure.

Key Components:
1. Latent Planning Optimization - structured rollout with continuation-based reasoning
2. Asynchronous Execution Engine - KV-cache sharing and token-level dispatch  
3. Prioritized Replay Buffer - stratified hot/cold buffers for improved RL training efficiency
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
    "MetricsCollector"
]
