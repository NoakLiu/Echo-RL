"""
Asynchronous Execution Engine with KV-Cache Sharing

Implements the asynchronous rollout system with KV-cache sharing and 
latency-aware scheduling to boost parallelism and reduce computational overhead.

Key Components:
- KVCacheManager: Manages KV cache with prefix sharing
- LatencyScheduler: Priority-based scheduling with reward-cost ratio
- AsyncExecutionEngine: Coordinates async rollout across multiple workers
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import heapq
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..kernels import (
    attention_bandwidth_cost,
    prefix_match,
    schedule_priorities,
    state_hash,
)

logger = logging.getLogger(__name__)

@dataclass
class KVCacheEntry:
    """Represents a cached KV state"""
    key: str  # Cache key (e.g., "s1:t")
    kv_states: torch.Tensor  # Cached KV states
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        # For priority queue ordering (LRU)
        return self.last_access < other.last_access

@dataclass
class RolloutRequest:
    """Represents a rollout request"""
    request_id: str
    state_sequence: torch.Tensor
    priority: float
    timestamp: float
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RolloutResult:
    """Result of a rollout operation"""
    request_id: str
    success: bool
    kv_cache: Optional[torch.Tensor] = None
    execution_time: float = 0.0
    tokens_generated: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionConfig:
    """Configuration for async execution engine"""
    max_cache_size: int = 10000
    cache_cleanup_interval: float = 60.0  # seconds
    max_concurrent_rollouts: int = 128
    priority_queue_size: int = 1000
    kv_reuse_threshold: float = 0.5  # Minimum reuse rate to keep cache entry
    latency_weight: float = 0.1
    reward_weight: float = 1.0
    timeout: float = 30.0  # seconds per rollout
    schedule_epsilon: float = 1e-6  # ε in priority(i) = r_i / (q_i + ε)
    bandwidth_scale: float = 1.0  # scale for b(s_{1:t})

class KVCacheManager:
    """
    Manages KV cache with prefix sharing and LRU eviction
    
    Implements: KV(s1:t) = KV_frozen(s1:t') ∪ KV_rolling(s_{t'+1:t})
    """
    
    def __init__(self, config: ExecutionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Cache storage: key -> entry, plus ordered hash index for prefix lookup
        self.cache: Dict[str, KVCacheEntry] = {}
        self.cache_hash_index: List[Tuple[int, str]] = []  # (hash, cache_key)
        self.access_order = deque()
        
        # Cache statistics
        self.hit_count = 0
        self.miss_count = 0
        self.total_requests = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cache cleanup task"""
        def cleanup_loop():
            while True:
                time.sleep(self.config.cache_cleanup_interval)
                self._cleanup_cache()
        
        self._cleanup_task = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_task.start()
    
    def _cleanup_cache(self):
        """Remove old or unused cache entries"""
        with self._lock:
            current_time = time.time()
            to_remove = []
            
            for key, entry in self.cache.items():
                # Remove entries that haven't been accessed recently
                if (current_time - entry.last_access) > self.config.cache_cleanup_interval:
                    to_remove.append(key)
            
            for key in to_remove:
                del self.cache[key]
                logger.debug(f"Removed cache entry: {key}")
    
    def get_cache_key(self, state_sequence: torch.Tensor) -> str:
        """Generate cache key from state sequence prefix hash."""
        state_np = state_sequence.detach().cpu().numpy().astype(np.float32)
        if state_np.ndim == 2:
            flat = state_np.reshape(-1)
        else:
            flat = state_np
        return f"states_{state_hash(flat)}"

    def _build_prefix_hashes(self, state_sequence: torch.Tensor) -> np.ndarray:
        """Cumulative prefix hashes for longest-prefix KV reuse."""
        seq = state_sequence.detach().cpu().numpy()
        if seq.ndim == 1:
            return np.array([state_hash(seq)], dtype=np.uint64)
        hashes = []
        for i in range(1, seq.shape[0] + 1):
            prefix = seq[:i].reshape(-1).astype(np.float32)
            hashes.append(state_hash(prefix))
        return np.array(hashes, dtype=np.uint64)

    def find_longest_prefix(self, state_sequence: torch.Tensor) -> Optional[Tuple[str, int]]:
        """
        Find longest cached prefix using C++ kernel.

        KV(s_{1:t}) = KV_frozen(s_{1:t'}) ∪ KV_rolling(s_{t'+1:t})
        """
        if state_sequence.shape[0] <= 1 or not self.cache_hash_index:
            return None

        prefix_hashes = self._build_prefix_hashes(state_sequence)
        cache_entries = [
            (int(h), idx) for idx, (h, _) in enumerate(self.cache_hash_index)
        ]
        cache_index, prefix_len = prefix_match(prefix_hashes, cache_entries, min_prefix_len=1)
        if cache_index < 0 or prefix_len <= 0:
            return None

        _, cache_key = self.cache_hash_index[cache_index]
        if cache_key in self.cache:
            return cache_key, prefix_len
        return None
    
    def get_kv_cache(self, state_sequence: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
        """
        Get KV cache for state sequence with prefix sharing
        
        Args:
            state_sequence: [seq_len, state_dim] - input sequence
            
        Returns:
            (kv_cache, reuse_length) - cached KV states and length of reused prefix
        """
        with self._lock:
            self.total_requests += 1
            
            # Find longest cached prefix
            prefix_result = self.find_longest_prefix(state_sequence)
            
            if prefix_result is None:
                # No cache hit
                self.miss_count += 1
                return None, 0
            
            cache_key, prefix_len = prefix_result
            
            # Update cache entry access info
            entry = self.cache[cache_key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            self.hit_count += 1
            
            return entry.kv_states, prefix_len
    
    def store_kv_cache(self, state_sequence: torch.Tensor, kv_states: torch.Tensor):
        """
        Store KV cache for state sequence
        
        Args:
            state_sequence: [seq_len, state_dim] - input sequence
            kv_states: [seq_len, hidden_dim] - computed KV states
        """
        with self._lock:
            cache_key = self.get_cache_key(state_sequence)
            
            # Create cache entry
            entry = KVCacheEntry(
                key=cache_key,
                kv_states=kv_states.clone(),
                timestamp=time.time()
            )
            
            # Store in cache
            self.cache[cache_key] = entry
            prefix_hash = int(state_hash(
                state_sequence.detach().cpu().numpy().reshape(-1).astype(np.float32)
            ))
            self.cache_hash_index.append((prefix_hash, cache_key))

            # Manage cache size
            if len(self.cache) > self.config.max_cache_size:
                self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used cache entry"""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_access)
        
        del self.cache[lru_key]
        self.cache_hash_index = [
            (h, k) for h, k in self.cache_hash_index if k != lru_key
        ]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = self.hit_count / max(self.total_requests, 1)
            return {
                "cache_size": len(self.cache),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "total_requests": self.total_requests,
                "hit_rate": hit_rate,
                "max_cache_size": self.config.max_cache_size
            }

class LatencyScheduler:
    """
    Reward-to-latency scheduler for async rollouts.

    priority(i) = r_i / (q_i + ε)
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.priority_queue = []
        self.request_times = {}
        self.estimated_times = {}
        self._lock = threading.RLock()

    def add_request(self, request: RolloutRequest):
        with self._lock:
            queue_time = len(self.priority_queue) * 0.1
            rewards = np.array([request.priority], dtype=np.float32)
            queue_times = np.array([queue_time], dtype=np.float32)
            priority_arr = schedule_priorities(
                rewards, queue_times, self.config.schedule_epsilon
            )
            priority = float(priority_arr[0])

            self.request_times[request.request_id] = time.time()
            heapq.heappush(self.priority_queue, (-priority, request))
    
    def get_next_request(self) -> Optional[RolloutRequest]:
        """Get next highest priority request"""
        with self._lock:
            if not self.priority_queue:
                return None
            
            _, request = heapq.heappop(self.priority_queue)
            
            # Update estimated execution time
            actual_queue_time = time.time() - self.request_times[request.request_id]
            self.estimated_times[request.request_id] = actual_queue_time
            
            return request
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                "queue_size": len(self.priority_queue),
                "total_requests": len(self.request_times),
                "avg_queue_time": np.mean(list(self.estimated_times.values())) if self.estimated_times else 0.0
            }

class AsyncExecutionEngine:
    """
    Main async execution engine coordinating rollout across multiple workers
    
    Manages KV cache sharing, priority scheduling, and parallel execution
    to maximize throughput and minimize latency.
    """
    
    def __init__(self, 
                 config: ExecutionConfig,
                 model: nn.Module,
                 device: str = "cuda"):
        self.config = config
        self.model = model
        self.device = device
        
        # Core components
        self.kv_cache_manager = KVCacheManager(config, device)
        self.scheduler = LatencyScheduler(config)
        
        # Execution state
        self.active_rollouts = {}
        self.completed_rollouts = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_rollouts)
        
        # Performance tracking
        self.total_tokens_generated = 0
        self.total_execution_time = 0.0
        self.total_bandwidth_cost = 0.0
        self.rollout_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    async def submit_rollout(self, 
                           state_sequence: torch.Tensor,
                           priority: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a rollout request for async execution
        
        Args:
            state_sequence: [seq_len, state_dim] - input state sequence
            priority: Priority score (higher = more important)
            metadata: Optional metadata for the request
            
        Returns:
            request_id: Unique identifier for the request
        """
        request_id = f"rollout_{self.rollout_count}_{int(time.time() * 1000)}"
        
        request = RolloutRequest(
            request_id=request_id,
            state_sequence=state_sequence,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Add to scheduler
        self.scheduler.add_request(request)
        
        # Start async execution
        future = self.executor.submit(self._execute_rollout, request)
        self.active_rollouts[request_id] = future
        
        self.rollout_count += 1
        
        return request_id
    
    def _execute_rollout(self, request: RolloutRequest) -> RolloutResult:
        """
        Execute a single rollout with KV cache optimization
        
        Args:
            request: RolloutRequest to execute
            
        Returns:
            result: RolloutResult with execution details
        """
        start_time = time.time()
        
        try:
            # Get KV cache with prefix sharing
            cached_kv, reuse_length = self.kv_cache_manager.get_kv_cache(
                request.state_sequence
            )
            
            # Compute new KV states for the non-cached portion
            if cached_kv is not None:
                # Use cached prefix and compute rolling portion
                new_sequence = request.state_sequence[reuse_length:]
                if len(new_sequence) > 0:
                    new_kv = self._compute_kv_states(new_sequence)
                    # Combine cached and new KV states
                    kv_states = torch.cat([cached_kv, new_kv], dim=0)
                else:
                    kv_states = cached_kv
            else:
                # No cache hit, compute full sequence
                kv_states = self._compute_kv_states(request.state_sequence)
            
            # Store updated KV cache
            self.kv_cache_manager.store_kv_cache(request.state_sequence, kv_states)
            
            tokens_generated = self._generate_tokens(kv_states, request.state_sequence)
            seq_len = int(request.state_sequence.shape[0])
            bandwidth_cost = attention_bandwidth_cost(seq_len, self.config.bandwidth_scale)

            execution_time = time.time() - start_time

            result = RolloutResult(
                request_id=request.request_id,
                success=True,
                kv_cache=kv_states,
                execution_time=execution_time,
                tokens_generated=tokens_generated,
                metadata={
                    "reuse_length": reuse_length,
                    "cache_hit": cached_kv is not None,
                    "bandwidth_cost": bandwidth_cost,
                    **request.metadata,
                },
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Rollout execution failed: {e}")
            
            result = RolloutResult(
                request_id=request.request_id,
                success=False,
                execution_time=execution_time,
                error=str(e),
                metadata=request.metadata
            )
        
        # Update statistics
        with self._lock:
            self.total_execution_time += execution_time
            if result.success:
                self.total_tokens_generated += result.tokens_generated
                self.total_bandwidth_cost += result.metadata.get("bandwidth_cost", 0.0)
            
            # Move from active to completed
            if request.request_id in self.active_rollouts:
                del self.active_rollouts[request.request_id]
            self.completed_rollouts[request.request_id] = result
        
        return result
    
    def _compute_kv_states(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute KV states for state sequence
        
        Args:
            state_sequence: [seq_len, state_dim] - input sequence
            
        Returns:
            kv_states: [seq_len, hidden_dim] - computed KV states
        """
        # This is a simplified implementation
        # In practice, this would use the actual transformer attention mechanism
        
        with torch.no_grad():
            # Simulate KV computation (replace with actual model forward pass)
            seq_len = state_sequence.shape[0]
            hidden_dim = 512  # Should match model hidden dimension
            
            # For now, return random KV states
            kv_states = torch.randn(seq_len, hidden_dim, device=self.device)
            
            return kv_states
    
    def _generate_tokens(self, kv_states: torch.Tensor, state_sequence: torch.Tensor) -> int:
        """
        Generate tokens using KV states
        
        Args:
            kv_states: [seq_len, hidden_dim] - KV states
            state_sequence: [seq_len, state_dim] - input sequence
            
        Returns:
            tokens_generated: Number of tokens generated
        """
        # Simplified token generation
        # In practice, this would use the actual model generation
        
        with torch.no_grad():
            # Simulate token generation
            tokens_generated = min(50, kv_states.shape[0] * 2)  # Rough estimate
            
            return tokens_generated
    
    async def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[RolloutResult]:
        """
        Get result for a rollout request
        
        Args:
            request_id: Request identifier
            timeout: Optional timeout in seconds
            
        Returns:
            result: RolloutResult if available, None otherwise
        """
        # Check if already completed
        if request_id in self.completed_rollouts:
            return self.completed_rollouts[request_id]
        
        # Wait for completion
        if request_id in self.active_rollouts:
            future = self.active_rollouts[request_id]
            try:
                result = future.result(timeout=timeout or self.config.timeout)
                return result
            except Exception as e:
                logger.error(f"Failed to get result for {request_id}: {e}")
                return None
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self._lock:
            cache_stats = self.kv_cache_manager.get_cache_stats()
            queue_stats = self.scheduler.get_queue_stats()
            
            avg_execution_time = (self.total_execution_time / 
                                max(self.rollout_count, 1))
            
            tokens_per_second = (self.total_tokens_generated / 
                               max(self.total_execution_time, 1e-6))
            
            return {
                "total_rollouts": self.rollout_count,
                "active_rollouts": len(self.active_rollouts),
                "completed_rollouts": len(self.completed_rollouts),
                "total_tokens_generated": self.total_tokens_generated,
                "total_execution_time": self.total_execution_time,
                "total_bandwidth_cost": self.total_bandwidth_cost,
                "avg_execution_time": avg_execution_time,
                "tokens_per_second": tokens_per_second,
                "cache_stats": cache_stats,
                "queue_stats": queue_stats,
            }
    
    def shutdown(self):
        """Shutdown the execution engine"""
        logger.info("Shutting down async execution engine...")
        
        # Cancel active rollouts
        for future in self.active_rollouts.values():
            future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Async execution engine shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()

class DistributedExecutionEngine(AsyncExecutionEngine):
    """
    Distributed version of the execution engine for multi-GPU/multi-node setups
    
    Extends AsyncExecutionEngine with distributed KV cache sharing
    and load balancing across multiple workers.
    """
    
    def __init__(self, 
                 config: ExecutionConfig,
                 model: nn.Module,
                 device: str = "cuda",
                 worker_nodes: Optional[List[str]] = None):
        super().__init__(config, model, device)
        
        self.worker_nodes = worker_nodes or ["localhost"]
        self.current_worker = 0
        
        # Distributed KV cache (simplified)
        self.distributed_cache = {}
        
    def _select_worker(self) -> str:
        """Select next worker node (round-robin)"""
        worker = self.worker_nodes[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.worker_nodes)
        return worker
    
    def _distributed_kv_lookup(self, state_sequence: torch.Tensor) -> Optional[torch.Tensor]:
        """Lookup KV cache across distributed workers"""
        # Simplified distributed lookup
        # In practice, this would use distributed caching (Redis, etc.)
        
        cache_key = self.kv_cache_manager.get_cache_key(state_sequence)
        
        # Check local cache first
        if cache_key in self.distributed_cache:
            return self.distributed_cache[cache_key]
        
        # Check remote caches (simplified)
        for worker in self.worker_nodes:
            if worker != "localhost":
                # Simulate remote cache lookup
                # In practice, this would make network requests
                pass
        
        return None
    
    def get_distributed_metrics(self) -> Dict[str, Any]:
        """Get metrics including distributed statistics"""
        base_metrics = self.get_performance_metrics()
        
        distributed_metrics = {
            "worker_nodes": len(self.worker_nodes),
            "current_worker": self.current_worker,
            "distributed_cache_size": len(self.distributed_cache)
        }
        
        return {**base_metrics, **distributed_metrics}
