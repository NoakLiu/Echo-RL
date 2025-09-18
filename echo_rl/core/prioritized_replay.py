"""
Prioritized Replay Buffer with Hot/Cold Stratification

Implements the planning-aware prioritized replay system with stratified memory design.
Maintains hot/cold buffers separated by time threshold and samples trajectories 
based on surprise-reward prioritization.

Key Components:
- HotColdBuffer: Manages hot and cold buffer partitions
- SurpriseCalculator: Computes trajectory surprise metrics
- PrioritizedReplayBuffer: Main replay buffer with softmax-weighted sampling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque
import heapq
import time
import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Represents a single experience tuple"""
    state: torch.Tensor
    latent_plan: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    timestamp: float
    surprise_score: float = 0.0
    priority: float = 0.0
    age: int = 0  # Number of training steps since collection

@dataclass
class ReplayConfig:
    """Configuration for prioritized replay buffer"""
    hot_buffer_size: int = 1000000  # |B_hot|
    cold_buffer_size: int = 10000000  # |B_cold|
    age_threshold: int = 1000  # τ_thresh - threshold for hot/cold separation
    temperature: float = 1.0  # β - temperature for softmax sampling
    surprise_weight: float = 1.0  # α - weight for surprise metric
    reward_weight: float = 1.0  # Weight for reward component
    min_experiences: int = 1000  # Minimum experiences before sampling
    max_experiences: int = 11000000  # Total buffer capacity
    priority_alpha: float = 0.6  # Priority exponent
    importance_sampling_beta: float = 0.4  # Importance sampling correction

class SurpriseCalculator:
    """
    Computes trajectory surprise metrics for experience prioritization
    
    Implements: score(t) = ||τ_t - E[τ]||² + α * r_t
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        
        # Running statistics for surprise calculation
        self.trajectory_mean = None
        self.trajectory_variance = None
        self.trajectory_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def update_statistics(self, latent_plan: torch.Tensor):
        """
        Update running statistics for trajectory surprise calculation
        
        Args:
            latent_plan: [embedding_dim] - latent trajectory plan
        """
        with self._lock:
            if self.trajectory_mean is None:
                # Initialize with first trajectory
                self.trajectory_mean = latent_plan.clone()
                self.trajectory_variance = torch.zeros_like(latent_plan)
                self.trajectory_count = 1
            else:
                # Update running statistics using Welford's algorithm
                self.trajectory_count += 1
                delta = latent_plan - self.trajectory_mean
                self.trajectory_mean += delta / self.trajectory_count
                delta2 = latent_plan - self.trajectory_mean
                self.trajectory_variance += delta * delta2
    
    def compute_surprise_score(self, 
                              latent_plan: torch.Tensor, 
                              reward: float) -> float:
        """
        Compute surprise score for trajectory prioritization
        
        Args:
            latent_plan: [embedding_dim] - latent trajectory plan
            reward: Reward value
            
        Returns:
            surprise_score: Surprise score for prioritization
        """
        with self._lock:
            if self.trajectory_mean is None:
                # No statistics available yet, use simple reward-based score
                return abs(reward) * self.config.reward_weight
            
            # Compute distance from expected trajectory
            trajectory_distance = torch.norm(latent_plan - self.trajectory_mean).item()
            
            # Combine with reward
            surprise_score = (trajectory_distance * self.config.surprise_weight + 
                            abs(reward) * self.config.reward_weight)
            
            return surprise_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current trajectory statistics"""
        with self._lock:
            return {
                "trajectory_count": self.trajectory_count,
                "trajectory_mean_norm": torch.norm(self.trajectory_mean).item() if self.trajectory_mean is not None else 0.0,
                "trajectory_variance_norm": torch.norm(self.trajectory_variance).item() if self.trajectory_variance is not None else 0.0
            }

class HotColdBuffer:
    """
    Manages hot and cold buffer partitions with age-based stratification
    
    Hot buffer: Recent experiences (age ≤ τ_thresh)
    Cold buffer: Older experiences (age > τ_thresh)
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        
        # Buffer partitions
        self.hot_buffer: List[Experience] = []
        self.cold_buffer: List[Experience] = []
        
        # Priority queues for efficient sampling
        self.hot_priorities = []
        self.cold_priorities = []
        
        # Statistics
        self.hot_hits = 0
        self.cold_hits = 0
        self.total_samples = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_experience(self, experience: Experience):
        """
        Add experience to appropriate buffer partition
        
        Args:
            experience: Experience to add
        """
        with self._lock:
            # Determine buffer based on age
            if experience.age <= self.config.age_threshold:
                # Add to hot buffer
                self.hot_buffer.append(experience)
                
                # Add to priority queue (negative for max-heap)
                priority = experience.priority ** self.config.priority_alpha
                heapq.heappush(self.hot_priorities, (-priority, len(self.hot_buffer) - 1))
                
                # Manage hot buffer size
                if len(self.hot_buffer) > self.config.hot_buffer_size:
                    self._evict_hot_experience()
            else:
                # Add to cold buffer
                self.cold_buffer.append(experience)
                
                # Add to priority queue
                priority = experience.priority ** self.config.priority_alpha
                heapq.heappush(self.cold_priorities, (-priority, len(self.cold_buffer) - 1))
                
                # Manage cold buffer size
                if len(self.cold_buffer) > self.config.cold_buffer_size:
                    self._evict_cold_experience()
    
    def _evict_hot_experience(self):
        """Evict lowest priority experience from hot buffer"""
        if not self.hot_priorities:
            return
        
        # Remove lowest priority experience
        _, idx = heapq.heappop(self.hot_priorities)
        
        # Remove from buffer (swap with last element for efficiency)
        if idx < len(self.hot_buffer) - 1:
            self.hot_buffer[idx] = self.hot_buffer[-1]
            # Update priority queue index
            for i, (_, buffer_idx) in enumerate(self.hot_priorities):
                if buffer_idx == len(self.hot_buffer) - 1:
                    self.hot_priorities[i] = (self.hot_priorities[i][0], idx)
                    heapq.heapify(self.hot_priorities)
                    break
        
        self.hot_buffer.pop()
    
    def _evict_cold_experience(self):
        """Evict lowest priority experience from cold buffer"""
        if not self.cold_priorities:
            return
        
        # Remove lowest priority experience
        _, idx = heapq.heappop(self.cold_priorities)
        
        # Remove from buffer (swap with last element for efficiency)
        if idx < len(self.cold_buffer) - 1:
            self.cold_buffer[idx] = self.cold_buffer[-1]
            # Update priority queue index
            for i, (_, buffer_idx) in enumerate(self.cold_priorities):
                if buffer_idx == len(self.cold_buffer) - 1:
                    self.cold_priorities[i] = (self.cold_priorities[i][0], idx)
                    heapq.heapify(self.cold_priorities)
                    break
        
        self.cold_buffer.pop()
    
    def sample_experiences(self, 
                          batch_size: int,
                          temperature: float = 1.0,
                          hot_ratio: float = 0.7) -> Tuple[List[Experience], List[float]]:
        """
        Sample experiences using softmax-weighted distribution
        
        Args:
            batch_size: Number of experiences to sample
            temperature: Temperature for softmax sampling
            hot_ratio: Ratio of samples from hot buffer
            
        Returns:
            (experiences, importance_weights): Sampled experiences and weights
        """
        with self._lock:
            if len(self.hot_buffer) + len(self.cold_buffer) < self.config.min_experiences:
                return [], []
            
            # Determine sample sizes
            hot_samples = int(batch_size * hot_ratio)
            cold_samples = batch_size - hot_samples
            
            # Sample from hot buffer
            hot_experiences, hot_weights = self._sample_from_buffer(
                self.hot_buffer, self.hot_priorities, hot_samples, temperature
            )
            self.hot_hits += len(hot_experiences)
            
            # Sample from cold buffer
            cold_experiences, cold_weights = self._sample_from_buffer(
                self.cold_buffer, self.cold_priorities, cold_samples, temperature
            )
            self.cold_hits += len(cold_experiences)
            
            # Combine samples
            all_experiences = hot_experiences + cold_experiences
            all_weights = hot_weights + cold_weights
            
            self.total_samples += len(all_experiences)
            
            return all_experiences, all_weights
    
    def _sample_from_buffer(self, 
                           buffer: List[Experience],
                           priorities: List[Tuple[float, int]],
                           sample_size: int,
                           temperature: float) -> Tuple[List[Experience], List[float]]:
        """Sample experiences from a single buffer partition"""
        if not buffer or sample_size == 0:
            return [], []
        
        # Get priorities
        buffer_priorities = []
        for _, idx in priorities:
            if idx < len(buffer):
                buffer_priorities.append(buffer[idx].priority)
        
        if not buffer_priorities:
            return [], []
        
        # Convert to numpy for softmax computation
        priorities_array = np.array(buffer_priorities)
        
        # Apply temperature scaling
        scaled_priorities = priorities_array / temperature
        
        # Compute softmax probabilities
        exp_priorities = np.exp(scaled_priorities - np.max(scaled_priorities))
        probabilities = exp_priorities / np.sum(exp_priorities)
        
        # Sample indices
        indices = np.random.choice(
            len(buffer_priorities), 
            size=min(sample_size, len(buffer_priorities)),
            replace=True,
            p=probabilities
        )
        
        # Get experiences and compute importance weights
        experiences = []
        weights = []
        
        for idx in indices:
            buffer_idx = priorities[idx][1]
            if buffer_idx < len(buffer):
                experience = buffer[buffer_idx]
                experiences.append(experience)
                
                # Importance sampling weight
                importance_weight = (len(buffer) * probabilities[idx]) ** (-self.config.importance_sampling_beta)
                weights.append(importance_weight)
        
        return experiences, weights
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for specific experiences"""
        with self._lock:
            # This is a simplified implementation
            # In practice, you'd need to track which buffer each index belongs to
            # and update the corresponding priority queue
            
            for idx, new_priority in zip(indices, new_priorities):
                # Find and update priority in both buffers
                # This is complex due to the heap structure
                # For now, we'll skip this optimization
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            hot_hit_rate = self.hot_hits / max(self.total_samples, 1)
            cold_hit_rate = self.cold_hits / max(self.total_samples, 1)
            
            return {
                "hot_buffer_size": len(self.hot_buffer),
                "cold_buffer_size": len(self.cold_buffer),
                "total_size": len(self.hot_buffer) + len(self.cold_buffer),
                "hot_hits": self.hot_hits,
                "cold_hits": self.cold_hits,
                "total_samples": self.total_samples,
                "hot_hit_rate": hot_hit_rate,
                "cold_hit_rate": cold_hit_rate,
                "hot_buffer_capacity": self.config.hot_buffer_size,
                "cold_buffer_capacity": self.config.cold_buffer_size
            }

class PrioritizedReplayBuffer:
    """
    Main prioritized replay buffer with hot/cold stratification
    
    Implements the complete EchoRL replay system with:
    - Hot/cold buffer management
    - Surprise-weighted sampling
    - Softmax probability distribution
    - Importance sampling correction
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        
        # Core components
        self.hot_cold_buffer = HotColdBuffer(config)
        self.surprise_calculator = SurpriseCalculator(config)
        
        # Experience tracking
        self.total_experiences = 0
        self.experience_age_counter = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_experience(self, 
                      state: torch.Tensor,
                      latent_plan: torch.Tensor,
                      action: torch.Tensor,
                      reward: float,
                      next_state: torch.Tensor,
                      done: bool):
        """
        Add new experience to replay buffer
        
        Args:
            state: Current state
            latent_plan: Latent trajectory plan
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        with self._lock:
            # Update trajectory statistics
            self.surprise_calculator.update_statistics(latent_plan)
            
            # Compute surprise score
            surprise_score = self.surprise_calculator.compute_surprise_score(
                latent_plan, reward
            )
            
            # Create experience
            experience = Experience(
                state=state.clone(),
                latent_plan=latent_plan.clone(),
                action=action.clone(),
                reward=reward,
                next_state=next_state.clone(),
                done=done,
                timestamp=time.time(),
                surprise_score=surprise_score,
                priority=surprise_score,
                age=0
            )
            
            # Add to buffer
            self.hot_cold_buffer.add_experience(experience)
            
            self.total_experiences += 1
    
    def sample_batch(self, 
                    batch_size: int,
                    temperature: Optional[float] = None) -> Tuple[List[Experience], List[float]]:
        """
        Sample batch of experiences using prioritized sampling
        
        Args:
            batch_size: Number of experiences to sample
            temperature: Temperature for softmax sampling (uses config default if None)
            
        Returns:
            (experiences, importance_weights): Sampled experiences and weights
        """
        with self._lock:
            if temperature is None:
                temperature = self.config.temperature
            
            return self.hot_cold_buffer.sample_experiences(
                batch_size, temperature
            )
    
    def update_experience_ages(self):
        """Update age of all experiences (called each training step)"""
        with self._lock:
            self.experience_age_counter += 1
            
            # Update ages in both buffers
            for experience in self.hot_cold_buffer.hot_buffer:
                experience.age += 1
            
            for experience in self.hot_cold_buffer.cold_buffer:
                experience.age += 1
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        with self._lock:
            buffer_stats = self.hot_cold_buffer.get_statistics()
            surprise_stats = self.surprise_calculator.get_statistics()
            
            return {
                "total_experiences": self.total_experiences,
                "experience_age_counter": self.experience_age_counter,
                "buffer_stats": buffer_stats,
                "surprise_stats": surprise_stats,
                "config": {
                    "hot_buffer_size": self.config.hot_buffer_size,
                    "cold_buffer_size": self.config.cold_buffer_size,
                    "age_threshold": self.config.age_threshold,
                    "temperature": self.config.temperature,
                    "min_experiences": self.config.min_experiences
                }
            }
    
    def clear_buffer(self):
        """Clear all experiences from buffer"""
        with self._lock:
            self.hot_cold_buffer = HotColdBuffer(self.config)
            self.surprise_calculator = SurpriseCalculator(self.config)
            self.total_experiences = 0
            self.experience_age_counter = 0
    
    def save_buffer(self, filepath: str):
        """Save buffer to disk"""
        with self._lock:
            buffer_data = {
                "hot_buffer": self.hot_cold_buffer.hot_buffer,
                "cold_buffer": self.hot_cold_buffer.cold_buffer,
                "total_experiences": self.total_experiences,
                "experience_age_counter": self.experience_age_counter,
                "surprise_stats": self.surprise_calculator.get_statistics(),
                "config": self.config
            }
            torch.save(buffer_data, filepath)
            logger.info(f"Saved replay buffer to {filepath}")
    
    def load_buffer(self, filepath: str):
        """Load buffer from disk"""
        with self._lock:
            buffer_data = torch.load(filepath, map_location='cpu')
            
            self.hot_cold_buffer.hot_buffer = buffer_data["hot_buffer"]
            self.hot_cold_buffer.cold_buffer = buffer_data["cold_buffer"]
            self.total_experiences = buffer_data["total_experiences"]
            self.experience_age_counter = buffer_data["experience_age_counter"]
            
            # Restore surprise calculator statistics
            surprise_stats = buffer_data["surprise_stats"]
            if surprise_stats["trajectory_count"] > 0:
                self.surprise_calculator.trajectory_count = surprise_stats["trajectory_count"]
                # Note: Would need to restore mean and variance tensors
            
            logger.info(f"Loaded replay buffer from {filepath}")

class AdaptiveReplayBuffer(PrioritizedReplayBuffer):
    """
    Adaptive version of prioritized replay buffer that adjusts parameters
    based on training progress and performance metrics
    """
    
    def __init__(self, config: ReplayConfig):
        super().__init__(config)
        
        # Adaptive parameters
        self.adaptive_temperature = config.temperature
        self.adaptive_hot_ratio = 0.7
        self.performance_history = deque(maxlen=100)
        
    def update_adaptive_parameters(self, performance_metric: float):
        """
        Update adaptive parameters based on performance
        
        Args:
            performance_metric: Current performance metric (e.g., success rate)
        """
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 10:
            return
        
        # Adjust temperature based on performance trend
        recent_performance = np.mean(list(self.performance_history)[-10:])
        older_performance = np.mean(list(self.performance_history)[-20:-10]) if len(self.performance_history) >= 20 else recent_performance
        
        if recent_performance > older_performance:
            # Performance improving, reduce temperature (more exploitation)
            self.adaptive_temperature = max(0.1, self.adaptive_temperature * 0.95)
        else:
            # Performance stagnating, increase temperature (more exploration)
            self.adaptive_temperature = min(2.0, self.adaptive_temperature * 1.05)
        
        # Adjust hot buffer ratio based on performance
        if recent_performance > 0.8:
            # High performance, focus more on recent experiences
            self.adaptive_hot_ratio = min(0.9, self.adaptive_hot_ratio + 0.01)
        else:
            # Lower performance, use more diverse experiences
            self.adaptive_hot_ratio = max(0.5, self.adaptive_hot_ratio - 0.01)
    
    def sample_batch(self, batch_size: int, temperature: Optional[float] = None) -> Tuple[List[Experience], List[float]]:
        """Sample batch using adaptive parameters"""
        if temperature is None:
            temperature = self.adaptive_temperature
        
        return self.hot_cold_buffer.sample_experiences(
            batch_size, temperature, self.adaptive_hot_ratio
        )
    
    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get statistics including adaptive parameters"""
        base_stats = self.get_buffer_statistics()
        
        adaptive_stats = {
            "adaptive_temperature": self.adaptive_temperature,
            "adaptive_hot_ratio": self.adaptive_hot_ratio,
            "performance_history": list(self.performance_history),
            "recent_performance": np.mean(list(self.performance_history)[-10:]) if self.performance_history else 0.0
        }
        
        return {**base_stats, **adaptive_stats}
