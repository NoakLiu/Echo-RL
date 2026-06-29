"""
Performance Monitoring and Metrics Collection

Provides comprehensive monitoring capabilities for EchoRL training including:
- Real-time performance tracking
- System resource monitoring
- Metrics collection and aggregation
- Performance analysis and reporting
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    step: int
    episode: int
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    kl_divergence: float = 0.0
    entropy_loss: float = 0.0
    total_reward: float = 0.0
    episode_length: int = 0
    success_rate: float = 0.0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    
    # EchoRL specific metrics
    kv_cache_hit_rate: float = 0.0
    tokens_per_second: float = 0.0
    replay_buffer_size: int = 0
    hot_buffer_hit_rate: float = 0.0
    cold_buffer_hit_rate: float = 0.0
    bandwidth_efficiency: float = 0.0  # η_bw
    rollout_bandwidth_cost: float = 0.0
    
    # Timing metrics
    rollout_time: float = 0.0
    training_time: float = 0.0
    total_time: float = 0.0

class MetricsCollector:
    """
    Collects and aggregates metrics from various EchoRL components
    
    Provides centralized metrics collection with automatic aggregation
    and statistical analysis capabilities.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.metric_aggregators = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
    
    def collect_metrics(self, metrics: PerformanceMetrics):
        """Collect performance metrics"""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Update aggregators
            for field_name, value in metrics.__dict__.items():
                if isinstance(value, (int, float)):
                    self.metric_aggregators[field_name].append(value)
    
    def get_metric_summary(self, metric_name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistical summary for a metric"""
        with self._lock:
            if metric_name not in self.metric_aggregators:
                return {}
            
            values = self.metric_aggregators[metric_name]
            if not values:
                return {}
            
            # Get recent values
            recent_values = values[-window_size:] if len(values) >= window_size else values
            
            return {
                "mean": np.mean(recent_values),
                "std": np.std(recent_values),
                "min": np.min(recent_values),
                "max": np.max(recent_values),
                "median": np.median(recent_values),
                "count": len(recent_values)
            }
    
    def get_all_metrics_summary(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Get summary for all metrics"""
        with self._lock:
            summary = {}
            for metric_name in self.metric_aggregators:
                summary[metric_name] = self.get_metric_summary(metric_name, window_size)
            return summary
    
    def get_trend_analysis(self, metric_name: str, window_size: int = 100) -> Dict[str, Any]:
        """Analyze trend for a metric"""
        with self._lock:
            if metric_name not in self.metric_aggregators:
                return {}
            
            values = self.metric_aggregators[metric_name]
            if len(values) < window_size:
                return {}
            
            recent_values = values[-window_size:]
            
            # Simple trend analysis
            x = np.arange(len(recent_values))
            coeffs = np.polyfit(x, recent_values, 1)
            trend_slope = coeffs[0]
            
            # Trend direction
            if abs(trend_slope) < 0.001:
                trend_direction = "stable"
            elif trend_slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            return {
                "trend_slope": trend_slope,
                "trend_direction": trend_direction,
                "recent_values": recent_values.tolist(),
                "window_size": window_size
            }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        with self._lock:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_metrics": len(self.metrics_history),
                "metrics_history": [
                    {
                        "timestamp": m.timestamp,
                        "step": m.step,
                        "episode": m.episode,
                        **{k: v for k, v in m.__dict__.items() 
                           if k not in ["timestamp", "step", "episode"]}
                    }
                    for m in self.metrics_history
                ],
                "metric_summaries": self.get_all_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")

class SystemMonitor:
    """
    Monitors system resources and performance
    
    Tracks CPU, memory, GPU usage and other system metrics
    for performance analysis and optimization.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # System metrics
        self.cpu_usage_history = deque(maxlen=1000)
        self.memory_usage_history = deque(maxlen=1000)
        self.gpu_usage_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_usage = memory_info.percent
                
                # GPU metrics (if available)
                gpu_usage = 0.0
                gpu_memory = 0.0
                
                if torch.cuda.is_available():
                    try:
                        gpu_usage = torch.cuda.utilization()
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    except Exception as e:
                        logger.debug(f"GPU monitoring error: {e}")
                
                # Store metrics
                with self._lock:
                    self.cpu_usage_history.append(cpu_usage)
                    self.memory_usage_history.append(memory_usage)
                    self.gpu_usage_history.append(gpu_usage)
                    self.gpu_memory_history.append(gpu_memory)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        with self._lock:
            return {
                "cpu_usage": self.cpu_usage_history[-1] if self.cpu_usage_history else 0.0,
                "memory_usage": self.memory_usage_history[-1] if self.memory_usage_history else 0.0,
                "gpu_usage": self.gpu_usage_history[-1] if self.gpu_usage_history else 0.0,
                "gpu_memory": self.gpu_memory_history[-1] if self.gpu_memory_history else 0.0
            }
    
    def get_metrics_summary(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Get summary of system metrics"""
        with self._lock:
            summary = {}
            
            for metric_name, history in [
                ("cpu_usage", self.cpu_usage_history),
                ("memory_usage", self.memory_usage_history),
                ("gpu_usage", self.gpu_usage_history),
                ("gpu_memory", self.gpu_memory_history)
            ]:
                if history:
                    recent_values = list(history)[-window_size:]
                    summary[metric_name] = {
                        "mean": np.mean(recent_values),
                        "std": np.std(recent_values),
                        "min": np.min(recent_values),
                        "max": np.max(recent_values),
                        "current": recent_values[-1]
                    }
            
            return summary

class PerformanceMonitor:
    """
    Main performance monitoring coordinator
    
    Coordinates metrics collection, system monitoring, and performance analysis
    for comprehensive EchoRL performance tracking.
    """
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 system_monitor: Optional[SystemMonitor] = None,
                 log_interval: float = 10.0):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.system_monitor = system_monitor or SystemMonitor()
        self.log_interval = log_interval
        
        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.step_count = 0
        self.episode_count = 0
        
        # Performance callbacks
        self.performance_callbacks = []
        
        # Thread safety
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start all monitoring components"""
        self.system_monitor.start_monitoring()
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        self.system_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
    
    def log_metrics(self, 
                   step: int,
                   episode: int,
                   training_metrics: Dict[str, float],
                   echo_rl_metrics: Dict[str, Any]):
        """Log performance metrics"""
        current_time = time.time()
        
        # Get system metrics
        system_metrics = self.system_monitor.get_current_metrics()
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            timestamp=current_time,
            step=step,
            episode=episode,
            policy_loss=training_metrics.get("policy_loss", 0.0),
            value_loss=training_metrics.get("value_loss", 0.0),
            kl_divergence=training_metrics.get("kl_divergence", 0.0),
            entropy_loss=training_metrics.get("entropy_loss", 0.0),
            total_reward=training_metrics.get("total_reward", 0.0),
            episode_length=training_metrics.get("episode_length", 0),
            success_rate=training_metrics.get("success_rate", 0.0),
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage=system_metrics["memory_usage"],
            gpu_usage=system_metrics["gpu_usage"],
            gpu_memory=system_metrics["gpu_memory"],
            kv_cache_hit_rate=echo_rl_metrics.get("kv_cache_hit_rate", 0.0),
            tokens_per_second=echo_rl_metrics.get("tokens_per_second", 0.0),
            replay_buffer_size=echo_rl_metrics.get("replay_buffer_size", 0),
            hot_buffer_hit_rate=echo_rl_metrics.get("hot_buffer_hit_rate", 0.0),
            cold_buffer_hit_rate=echo_rl_metrics.get("cold_buffer_hit_rate", 0.0),
            bandwidth_efficiency=echo_rl_metrics.get("bandwidth_efficiency", 0.0),
            rollout_bandwidth_cost=echo_rl_metrics.get("rollout_bandwidth_cost", 0.0),
            rollout_time=echo_rl_metrics.get("rollout_time", 0.0),
            training_time=echo_rl_metrics.get("training_time", 0.0),
            total_time=current_time - self.start_time
        )
        
        # Collect metrics
        self.metrics_collector.collect_metrics(perf_metrics)
        
        # Update counters
        with self._lock:
            self.step_count = step
            self.episode_count = episode
        
        # Log periodically
        if current_time - self.last_log_time >= self.log_interval:
            self._log_performance_summary()
            self.last_log_time = current_time
        
        # Call performance callbacks
        for callback in self.performance_callbacks:
            try:
                callback(perf_metrics)
            except Exception as e:
                logger.error(f"Performance callback error: {e}")
    
    def _log_performance_summary(self):
        """Log performance summary"""
        # Get recent metrics summary
        recent_summary = self.metrics_collector.get_all_metrics_summary(window_size=100)
        
        # Get system summary
        system_summary = self.system_monitor.get_metrics_summary()
        
        # Log key metrics
        logger.info("=== Performance Summary ===")
        
        if "policy_loss" in recent_summary:
            policy_summary = recent_summary["policy_loss"]
            logger.info(f"Policy Loss: {policy_summary['mean']:.4f} ± {policy_summary['std']:.4f}")
        
        if "total_reward" in recent_summary:
            reward_summary = recent_summary["total_reward"]
            logger.info(f"Total Reward: {reward_summary['mean']:.2f} ± {reward_summary['std']:.2f}")
        
        if "tokens_per_second" in recent_summary:
            tps_summary = recent_summary["tokens_per_second"]
            logger.info(f"Tokens/sec: {tps_summary['mean']:.1f} ± {tps_summary['std']:.1f}")

        if "bandwidth_efficiency" in recent_summary:
            bw_summary = recent_summary["bandwidth_efficiency"]
            logger.info(f"η_bw: {bw_summary['mean']:.4f} ± {bw_summary['std']:.4f}")
        
        if "cpu_usage" in system_summary:
            cpu_summary = system_summary["cpu_usage"]
            logger.info(f"CPU Usage: {cpu_summary['current']:.1f}% (avg: {cpu_summary['mean']:.1f}%)")
        
        if "memory_usage" in system_summary:
            mem_summary = system_summary["memory_usage"]
            logger.info(f"Memory Usage: {mem_summary['current']:.1f}% (avg: {mem_summary['mean']:.1f}%)")
        
        logger.info(f"Steps: {self.step_count}, Episodes: {self.episode_count}")
        logger.info("==========================")
    
    def add_performance_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add performance monitoring callback"""
        self.performance_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self._lock:
            current_time = time.time()
            
            # Get metrics summaries
            training_summary = self.metrics_collector.get_all_metrics_summary()
            system_summary = self.system_monitor.get_metrics_summary()
            
            # Calculate performance trends
            trends = {}
            for metric_name in ["policy_loss", "total_reward", "tokens_per_second"]:
                if metric_name in training_summary:
                    trends[metric_name] = self.metrics_collector.get_trend_analysis(metric_name)
            
            return {
                "timestamp": current_time,
                "total_training_time": current_time - self.start_time,
                "step_count": self.step_count,
                "episode_count": self.episode_count,
                "training_metrics": training_summary,
                "system_metrics": system_summary,
                "performance_trends": trends,
                "total_metrics_collected": len(self.metrics_collector.metrics_history)
            }
    
    def export_performance_data(self, filepath: str):
        """Export all performance data"""
        # Export metrics
        metrics_file = filepath.replace('.json', '_metrics.json')
        self.metrics_collector.export_metrics(metrics_file)
        
        # Export performance report
        report = self.get_performance_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance data exported to {filepath}")

class MetricsLogger:
    """
    Specialized logger for EchoRL metrics
    
    Provides structured logging of training metrics with automatic
    formatting and analysis capabilities.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = logging.getLogger("echo_rl_metrics")
        
        # Setup logging
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_training_step(self, 
                         step: int,
                         episode: int,
                         metrics: Dict[str, float],
                         echo_rl_metrics: Dict[str, Any]):
        """Log training step metrics"""
        log_data = {
            "step": step,
            "episode": episode,
            "metrics": metrics,
            "echo_rl_metrics": echo_rl_metrics,
            "timestamp": time.time()
        }
        
        self.logger.info(f"Training Step: {json.dumps(log_data)}")
    
    def log_evaluation(self, 
                      step: int,
                      eval_results: Dict[str, Any]):
        """Log evaluation results"""
        log_data = {
            "step": step,
            "evaluation": eval_results,
            "timestamp": time.time()
        }
        
        self.logger.info(f"Evaluation: {json.dumps(log_data)}")
    
    def log_checkpoint(self, 
                      step: int,
                      checkpoint_path: str,
                      metrics: Dict[str, Any]):
        """Log checkpoint save"""
        log_data = {
            "step": step,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        self.logger.info(f"Checkpoint: {json.dumps(log_data)}")
