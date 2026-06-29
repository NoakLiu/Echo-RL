"""
EchoRL Main Trainer

Implements the main training coordinator that orchestrates all EchoRL components:
- Latent Planning Optimization
- Asynchronous Execution Engine
- Prioritized Replay Buffer
- PPO Learner

Coordinates the training loop and manages the interaction between components.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque
import json
import os

from ..core.latent_planning import LatentPlanningOptimizer, PlanningConfig, TrajectoryPrior
from ..core.async_execution import AsyncExecutionEngine, ExecutionConfig, RolloutRequest
from ..core.prioritized_replay import PrioritizedReplayBuffer, ReplayConfig, Experience
from ..core.ppo_learner import PPOLearner, PPOConfig, PPOTrainingStep
from ..core.bandwidth import BandwidthConfig, BandwidthEfficiencyTracker
from ..environments.base import EchoRLEnvironment, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for EchoRL training"""
    # Environment settings
    env_name: str = "alfworld"
    env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    total_timesteps: int = 1000000
    learning_starts: int = 10000
    train_frequency: int = 4
    target_update_interval: int = 1000
    evaluation_frequency: int = 10000
    save_frequency: int = 50000
    
    # Model parameters
    state_dim: int = 512
    action_dim: int = 20
    latent_dim: int = 512
    
    # Component configurations
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    replay_config: ReplayConfig = field(default_factory=ReplayConfig)
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    bandwidth_config: BandwidthConfig = field(default_factory=BandwidthConfig)
    
    # Training settings
    device: str = "cuda"
    seed: Optional[int] = None
    num_actors: int = 128
    num_learners: int = 2
    batch_size: int = 256
    
    # Logging and monitoring
    log_level: str = "INFO"
    tensorboard_log: str = "./logs/tensorboard"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

@dataclass
class TrainingMetrics:
    """Training metrics and statistics"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    kv_cache_hit_rates: List[float] = field(default_factory=list)
    bandwidth_efficiency: List[float] = field(default_factory=list)
    replay_buffer_stats: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    tokens_per_second: List[float] = field(default_factory=list)
    wall_clock_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Training progress
    total_timesteps: int = 0
    total_episodes: int = 0
    training_time: float = 0.0
    evaluation_results: Dict[str, Any] = field(default_factory=dict)

class EchoRLTrainer:
    """
    Main EchoRL trainer coordinating all components
    
    Orchestrates the training loop integrating:
    - Latent Planning Optimization
    - Asynchronous Execution Engine  
    - Prioritized Replay Buffer
    - PPO Learner
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Set random seed
        if config.seed is not None:
            self._set_seed(config.seed)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        # Environment
        self.env = None
        
        # Async execution state
        self.active_rollouts = {}
        self.rollout_results = {}
        
        # Training buffers
        self.episode_buffer = []
        self.training_buffer = []
        self.state_history: deque = deque(maxlen=64)

        # Bandwidth efficiency tracker (η_bw)
        self.bandwidth_tracker = BandwidthEfficiencyTracker(
            config=self.config.bandwidth_config
        )
        
        logger.info("EchoRL Trainer initialized successfully")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _initialize_components(self):
        """Initialize all EchoRL components"""
        # Latent Planning Optimizer
        self.planning_optimizer = LatentPlanningOptimizer(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            config=self.config.planning_config,
            device=self.device
        )
        
        # Async Execution Engine
        self.execution_engine = AsyncExecutionEngine(
            config=self.config.execution_config,
            model=self.planning_optimizer.policy_network,
            device=self.device
        )
        
        # Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            config=self.config.replay_config,
            latent_dim=self.config.planning_config.embedding_dim,
        )
        
        # PPO Learner
        self.ppo_learner = PPOLearner(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            latent_dim=self.config.planning_config.embedding_dim,
            config=self.config.ppo_config,
            device=self.device
        )
        
        logger.info("All EchoRL components initialized")
    
    def _create_environment(self) -> EchoRLEnvironment:
        """Create environment instance"""
        if self.config.env_name == "alfworld":
            from ..environments.alfworld import ALFWorldEnvironment, ALFWorldConfig
            env_config = ALFWorldConfig(**self.config.env_config)
            return ALFWorldEnvironment(env_config)
        elif self.config.env_name == "webshop":
            from ..environments.webshop import WebShopEnvironment, WebShopConfig
            env_config = WebShopConfig(**self.config.env_config)
            return WebShopEnvironment(env_config)
        elif self.config.env_name == "cruxeval":
            from ..environments.cruxeval import CRUXEvalEnvironment, CRUXEvalConfig
            env_config = CRUXEvalConfig(**self.config.env_config)
            return CRUXEvalEnvironment(env_config)
        elif self.config.env_name == "arc":
            from ..environments.arc import ARCEnvironment, ARCConfig
            env_config = ARCConfig(**self.config.env_config)
            return ARCEnvironment(env_config)
        elif self.config.env_name == "game24":
            from ..environments.game24 import Game24Environment, Game24Config
            env_config = Game24Config(**self.config.env_config)
            return Game24Environment(env_config)
        elif self.config.env_name == "minigrid":
            from ..environments.minigrid import MiniGridEnvironment, MiniGridConfig
            env_config = MiniGridConfig(**self.config.env_config)
            return MiniGridEnvironment(env_config)
        else:
            raise ValueError(f"Unknown environment: {self.config.env_name}")
    
    async def train(self) -> TrainingMetrics:
        """
        Main training loop
        
        Returns:
            metrics: Final training metrics
        """
        logger.info("Starting EchoRL training")
        
        # Create environment
        self.env = self._create_environment()
        
        # Training loop
        while self.training_step < self.config.total_timesteps:
            # Collect experience
            await self._collect_experience()
            
            # Update models
            if self.training_step >= self.config.learning_starts:
                await self._update_models()
            
            # Evaluation
            if self.training_step % self.config.evaluation_frequency == 0:
                await self._evaluate()
            
            # Save checkpoint
            if self.training_step % self.config.save_frequency == 0:
                self._save_checkpoint()
            
            # Update metrics
            self._update_metrics()
            
            # Log progress
            if self.training_step % 1000 == 0:
                self._log_progress()
        
        logger.info("Training completed")
        return self.metrics
    
    async def _collect_experience(self):
        """Collect experience using async execution engine"""
        # Create rollout requests for multiple actors
        rollout_requests = []
        
        for actor_id in range(self.config.num_actors):
            # Reset environment for new episode
            state = self.env.reset()
            
            # Create state sequence for trajectory encoding
            state_window = self._create_state_window(state.observation)
            
            # Submit rollout request
            request_id = await self.execution_engine.submit_rollout(
                state_sequence=state_window,
                priority=1.0,
                metadata={"actor_id": actor_id, "episode": self.episode_count}
            )
            
            rollout_requests.append(request_id)
            self.active_rollouts[request_id] = {
                "actor_id": actor_id,
                "episode_start": time.time(),
                "initial_state": state
            }
        
        # Wait for rollouts to complete
        await self._wait_for_rollouts(rollout_requests)
    
    async def _wait_for_rollouts(self, request_ids: List[str]):
        """Wait for rollout requests to complete"""
        for request_id in request_ids:
            try:
                result = await self.execution_engine.get_result(request_id, timeout=30.0)
                if result and result.success:
                    await self._process_rollout_result(request_id, result)
                else:
                    logger.warning(f"Rollout {request_id} failed")
            except Exception as e:
                logger.error(f"Error waiting for rollout {request_id}: {e}")
    
    async def _process_rollout_result(self, request_id: str, result):
        """Process completed rollout result"""
        rollout_info = self.active_rollouts[request_id]
        
        # Generate trajectory using the result
        trajectory = await self._generate_trajectory(rollout_info, result)
        
        # Add experiences to replay buffer
        for experience in trajectory:
            self.replay_buffer.add_experience(
                state=experience.state,
                latent_plan=experience.latent_plan,
                action=experience.action,
                reward=experience.reward,
                next_state=experience.next_state,
                done=experience.done
            )
        
        # Clean up
        del self.active_rollouts[request_id]
        self.rollout_results[request_id] = result
    
    async def _generate_trajectory(self, rollout_info: Dict, result) -> List[Experience]:
        """Generate trajectory from rollout result"""
        # Simplified trajectory generation
        # In practice, this would use the actual rollout execution
        
        trajectory = []
        initial_state = rollout_info["initial_state"]
        
        # Create a simple trajectory
        for step in range(10):  # Simplified episode length
            # Get state representation
            state = self.env.get_state_representation()
            
            # Encode trajectory
            state_window = self._create_state_window(state)
            trajectory_prior = self.planning_optimizer.encode_trajectory(state_window)
            
            # Sample action
            action, log_prob = self.ppo_learner.get_action_and_log_prob(
                state, trajectory_prior.latent_plan
            )
            
            # Execute action
            next_state = self.env.step(action.item())
            
            # Create experience
            experience = Experience(
                state=state,
                latent_plan=trajectory_prior.latent_plan,
                action=action,
                reward=next_state.reward,
                next_state=self.env.get_state_representation(),
                done=next_state.done,
                timestamp=time.time(),
                surprise_score=self.planning_optimizer.get_trajectory_surprise(
                    trajectory_prior, next_state.reward
                ),
            )
            
            trajectory.append(experience)
            reuse_len = max(0, len(self.state_history) - 1)
            self.bandwidth_tracker.record_rollout_step(
                next_state.reward,
                seq_len=len(self.state_history) + 1,
                reuse_len=reuse_len,
            )
            self.state_history.append(state)

            if next_state.done:
                break

        return trajectory

    def _create_state_window(self, current_state: torch.Tensor) -> torch.Tensor:
        """Create sliding state window s_{t-k:t} for trajectory encoding."""
        self.state_history.append(current_state)
        window_size = self.config.planning_config.state_window_size

        history = list(self.state_history)
        if len(history) <= window_size + 1:
            pad_count = window_size + 1 - len(history)
            pad = [history[0]] * pad_count if history else [current_state]
            window = pad + history
        else:
            window = history[-(window_size + 1):]

        return torch.stack(window)
    
    async def _update_models(self):
        """Update models using collected experience"""
        if self.replay_buffer.total_experiences < self.config.batch_size:
            return
        
        # Sample batch from replay buffer
        experiences, importance_weights = self.replay_buffer.sample_batch(
            batch_size=self.config.batch_size
        )
        
        if not experiences:
            return
        
        # Prepare training data
        training_step = self._prepare_training_data(experiences, importance_weights)
        
        # Update PPO learner
        ppo_metrics = self.ppo_learner.update(training_step)

        weighted_pg = ppo_metrics["policy_loss"] * (
            training_step.importance_weights.mean().item()
            if training_step.importance_weights is not None
            else 1.0
        )
        self.bandwidth_tracker.record_learner_update(weighted_pg)
        bw_snapshot = self.bandwidth_tracker.snapshot()
        self.metrics.bandwidth_efficiency.append(bw_snapshot.eta_bw)
        
        # Update planning optimizer
        planning_metrics = await self._update_planning_optimizer(experiences)
        
        # Update metrics
        self.metrics.policy_losses.append(ppo_metrics["policy_loss"])
        self.metrics.value_losses.append(ppo_metrics["value_loss"])
        self.metrics.kl_divergences.append(ppo_metrics["kl_divergence"])
        self.metrics.entropy_losses.append(ppo_metrics["entropy_loss"])
        
        # Update replay buffer ages
        self.replay_buffer.update_experience_ages()
        
        self.training_step += 1
    
    def _prepare_training_data(self, experiences: List[Experience], 
                             importance_weights: List[float]) -> PPOTrainingStep:
        """Prepare training data for PPO update"""
        # Extract data from experiences
        states = torch.stack([exp.state for exp in experiences])
        latent_plans = torch.stack([exp.latent_plan for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        rewards = torch.tensor([exp.reward for exp in experiences])
        
        # Get value estimates
        values = self.ppo_learner.get_value_estimate(states, latent_plans)
        
        # Get action log probabilities
        _, log_probs = self.ppo_learner.get_action_and_log_prob(states, latent_plans)
        
        # Compute advantages using GAE
        advantages, returns = self.ppo_learner.gae_calculator.compute_advantages(
            rewards.unsqueeze(0), values.unsqueeze(0), 
            torch.zeros_like(rewards).unsqueeze(0)
        )
        
        # Convert importance weights to tensor
        importance_weights_tensor = torch.tensor(importance_weights, device=self.device)
        
        return PPOTrainingStep(
            states=states,
            latent_plans=latent_plans,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            advantages=advantages.squeeze(0),
            returns=returns.squeeze(0),
            importance_weights=importance_weights_tensor
        )
    
    async def _update_planning_optimizer(self, experiences: List[Experience]):
        """Update planning optimizer with trajectory data"""
        if len(experiences) < 2:
            return {}
        
        # Prepare trajectory data
        current_states = torch.stack([exp.state for exp in experiences[:-1]])
        previous_states = torch.stack([exp.state for exp in experiences[1:]])
        
        # Create state windows
        current_windows = torch.stack([self._create_state_window(state) for state in current_states])
        previous_windows = torch.stack([self._create_state_window(state) for state in previous_states])
        
        # Extract other data
        actions = torch.stack([exp.action for exp in experiences[:-1]])
        rewards = torch.tensor([exp.reward for exp in experiences[:-1]])
        
        # Compute advantages (simplified)
        advantages = torch.tensor([exp.reward for exp in experiences[:-1]], device=self.device)
        
        # Compute planning loss
        loss_dict = self.planning_optimizer.compute_planning_loss(
            current_windows, previous_windows, actions, rewards, advantages
        )
        
        # Update planning optimizer
        metrics = self.planning_optimizer.update(loss_dict)
        
        return metrics
    
    async def _evaluate(self):
        """Evaluate current policy"""
        logger.info(f"Evaluating at step {self.training_step}")
        
        # Run evaluation episodes
        eval_rewards = []
        eval_successes = []
        
        for _ in range(10):  # Run 10 evaluation episodes
            state = self.env.reset()
            episode_reward = 0.0
            episode_success = False
            
            for step in range(100):  # Max episode length
                # Get state representation
                state_tensor = self.env.get_state_representation()
                
                # Create state window
                state_window = self._create_state_window(state_tensor)
                
                # Encode trajectory
                trajectory_prior = self.planning_optimizer.encode_trajectory(state_window)
                
                # Sample action
                action, _ = self.ppo_learner.get_action_and_log_prob(
                    state_tensor, trajectory_prior.latent_plan
                )
                
                # Execute action
                next_state = self.env.step(action.item())
                episode_reward += next_state.reward
                
                if next_state.done:
                    episode_success = True
                    break
                
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_successes.append(episode_success)
        
        # Update evaluation metrics
        self.metrics.evaluation_results = {
            "step": self.training_step,
            "avg_reward": np.mean(eval_rewards),
            "success_rate": np.mean(eval_successes),
            "rewards": eval_rewards,
            "successes": eval_successes
        }
        
        logger.info(f"Evaluation - Avg Reward: {np.mean(eval_rewards):.2f}, "
                   f"Success Rate: {np.mean(eval_successes):.2f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_{self.training_step}.pt"
        )
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "planning_optimizer": self.planning_optimizer.trajectory_encoder.state_dict(),
            "ppo_learner": self.ppo_learner.policy_network.state_dict(),
            "metrics": self.metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _update_metrics(self):
        """Update training metrics"""
        # Update execution metrics
        exec_metrics = self.execution_engine.get_performance_metrics()
        self.metrics.execution_metrics = exec_metrics
        self.metrics.kv_cache_hit_rates.append(
            exec_metrics.get("cache_stats", {}).get("hit_rate", 0.0)
        )
        
        # Update replay buffer metrics
        buffer_stats = self.replay_buffer.get_buffer_statistics()
        self.metrics.replay_buffer_stats = buffer_stats
        
        # Update performance metrics
        self.metrics.tokens_per_second.append(exec_metrics.get("tokens_per_second", 0.0))
        self.metrics.wall_clock_times.append(time.time() - self.start_time)
        
        # Update training progress
        self.metrics.total_timesteps = self.training_step
        self.metrics.total_episodes = self.episode_count
        self.metrics.training_time = time.time() - self.start_time
    
    def _log_progress(self):
        """Log training progress"""
        if self.metrics.policy_losses:
            avg_policy_loss = np.mean(self.metrics.policy_losses[-100:])
            avg_value_loss = np.mean(self.metrics.value_losses[-100:])
            avg_kl_div = np.mean(self.metrics.kl_divergences[-100:])
        else:
            avg_policy_loss = avg_value_loss = avg_kl_div = 0.0
        
        logger.info(
            f"Step {self.training_step}/{self.config.total_timesteps} - "
            f"Policy Loss: {avg_policy_loss:.4f}, "
            f"Value Loss: {avg_value_loss:.4f}, "
            f"KL Div: {avg_kl_div:.4f}, "
            f"Episodes: {self.episode_count}"
        )
    
    def get_training_metrics(self) -> TrainingMetrics:
        """Get current training metrics"""
        return self.metrics
    
    def close(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()
        
        if hasattr(self, 'execution_engine'):
            self.execution_engine.shutdown()
        
        logger.info("EchoRL Trainer closed")
