"""
EchoRL Evaluation and Benchmarking System

This module provides comprehensive evaluation and benchmarking capabilities for EchoRL including:
- EchoRLBenchmark: Main benchmarking framework
- Performance evaluation across multiple tasks and backbones
- Comparative analysis with baseline methods
- Automated reporting and visualization
"""

from .benchmark import EchoRLBenchmark, BenchmarkConfig, BenchmarkResult
from .evaluator import EchoRLEvaluator, EvaluationConfig, EvaluationResult
from .comparison import MethodComparison, ComparisonConfig, ComparisonResult
from .reporting import BenchmarkReporter, ReportGenerator

__all__ = [
    "EchoRLBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult", 
    "EchoRLEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "MethodComparison",
    "ComparisonConfig",
    "ComparisonResult",
    "BenchmarkReporter",
    "ReportGenerator"
]
