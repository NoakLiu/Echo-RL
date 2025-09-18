"""
EchoRL Training Infrastructure

This module provides the main training infrastructure for EchoRL including:
- EchoRLTrainer: Main training coordinator
- Distributed training support
- Performance monitoring and metrics collection
- Checkpointing and model management
"""

from .trainer import EchoRLTrainer, TrainingConfig, TrainingMetrics
from .distributed import DistributedTrainer, DistributedConfig
from .monitoring import TrainingMonitor, MetricsLogger, PerformanceTracker

__all__ = [
    "EchoRLTrainer",
    "TrainingConfig", 
    "TrainingMetrics",
    "DistributedTrainer",
    "DistributedConfig",
    "TrainingMonitor",
    "MetricsLogger",
    "PerformanceTracker"
]
