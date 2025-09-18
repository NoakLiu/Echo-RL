"""
PPO Learner with GAE and KL Regularization

Implements Proximal Policy Optimization with Generalized Advantage Estimation
and KL regularization for the EchoRL system. Handles policy updates with
clipped surrogate objective and importance sampling correction.

Key Components:
- GAECalculator: Computes advantages using Generalized Advantage Estimation
- PPOLearner: Main PPO learner with clipped objective and KL penalty
- PolicyNetwork: Policy network conditioned on latent plans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PPOTrainingStep:
    """Represents a single PPO training step"""
    states: torch.Tensor
    latent_plans: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    importance_weights: Optional[torch.Tensor] = None

@dataclass
class PPOConfig:
    """Configuration for PPO learner"""
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1  # KL regularization coefficient
    max_grad_norm: float = 1.0
    gae_lambda: float = 0.95
    gamma: float = 0.99
    num_epochs: int = 4
    batch_size: int = 256
    target_kl: float = 0.01  # Early stopping threshold
    value_clip: bool = True
    normalize_advantages: bool = True

class ValueNetwork(nn.Module):
    """
    Value network for estimating state values
    
    Estimates V(s_t, τ_t) - value function conditioned on state and latent plan
    """
    
    def __init__(self, 
                 state_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 512,
                 device: str = "cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # State and latent plan processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, latent_plan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of value network
        
        Args:
            state: [batch_size, state_dim] - current state
            latent_plan: [batch_size, latent_dim] - latent trajectory plan
            
        Returns:
            values: [batch_size, 1] - estimated state values
        """
        # Encode state and latent plan
        state_repr = self.state_encoder(state)
        latent_repr = self.latent_encoder(latent_plan)
        
        # Combine representations
        combined = torch.cat([state_repr, latent_repr], dim=-1)
        
        # Estimate value
        value = self.value_head(combined)
        
        return value

class GAECalculator:
    """
    Generalized Advantage Estimation calculator
    
    Computes advantages using GAE: A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
    
    def compute_advantages(self,
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          dones: torch.Tensor,
                          next_values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE
        
        Args:
            rewards: [batch_size, seq_len] - rewards for each timestep
            values: [batch_size, seq_len] - value estimates for each timestep
            dones: [batch_size, seq_len] - done flags for each timestep
            next_values: [batch_size] - value estimates for next states (optional)
            
        Returns:
            (advantages, returns): GAE advantages and discounted returns
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device
        
        # Initialize advantage and return tensors
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # Last timestep
                if next_values is not None:
                    next_value = next_values
                else:
                    next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]
            
            # Compute TD error
            delta = rewards[:, t] + self.config.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            
            # Compute advantage using GAE
            if t == seq_len - 1:
                advantages[:, t] = delta
            else:
                advantages[:, t] = delta + self.config.gamma * self.config.gae_lambda * advantages[:, t + 1] * (1 - dones[:, t])
            
            # Compute returns
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def compute_advantages_single_episode(self,
                                        rewards: List[float],
                                        values: List[float],
                                        dones: List[bool],
                                        next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute advantages for a single episode
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state after episode
            
        Returns:
            (advantages, returns): Lists of advantages and returns
        """
        episode_length = len(rewards)
        advantages = [0.0] * episode_length
        returns = [0.0] * episode_length
        
        # Compute advantages using GAE
        for t in reversed(range(episode_length)):
            if t == episode_length - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # Compute TD error
            delta = rewards[t] + self.config.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # Compute advantage using GAE
            if t == episode_length - 1:
                advantages[t] = delta
            else:
                advantages[t] = delta + self.config.gamma * self.config.gae_lambda * advantages[t + 1] * (1 - dones[t])
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

class PPOLearner:
    """
    Main PPO learner implementing clipped surrogate objective with KL regularization
    
    Handles policy and value network updates using PPO with importance sampling
    correction from prioritized replay buffer.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 config: PPOConfig,
                 device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Networks
        self.policy_network = PolicyNetwork(state_dim, action_dim, latent_dim, device=device)
        self.value_network = ValueNetwork(state_dim, latent_dim, device=device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=config.learning_rate
        )
        
        # GAE calculator
        self.gae_calculator = GAECalculator(config)
        
        # Training state
        self.training_step = 0
        self.kl_divergences = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
    def compute_policy_loss(self,
                           states: torch.Tensor,
                           latent_plans: torch.Tensor,
                           actions: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor,
                           importance_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PPO policy loss with clipped surrogate objective
        
        Args:
            states: [batch_size, state_dim] - states
            latent_plans: [batch_size, latent_dim] - latent plans
            actions: [batch_size] - actions
            old_log_probs: [batch_size] - old action log probabilities
            advantages: [batch_size] - advantages
            importance_weights: [batch_size] - importance sampling weights
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        # Get current policy log probabilities
        action_log_probs = self.policy_network.get_action_probs(states, latent_plans)
        action_log_probs = torch.log(action_log_probs.gather(1, actions.unsqueeze(-1)) + 1e-8).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Apply importance sampling correction if provided
        if importance_weights is not None:
            ratio = ratio * importance_weights
        
        # Compute clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute entropy loss for exploration
        action_probs = self.policy_network.get_action_probs(states, latent_plans)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # Compute KL divergence for monitoring
        kl_divergence = (old_log_probs - action_log_probs).mean()
        
        return {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "kl_divergence": kl_divergence,
            "ratio": ratio.mean(),
            "entropy": entropy
        }
    
    def compute_value_loss(self,
                          states: torch.Tensor,
                          latent_plans: torch.Tensor,
                          returns: torch.Tensor,
                          old_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute value function loss
        
        Args:
            states: [batch_size, state_dim] - states
            latent_plans: [batch_size, latent_dim] - latent plans
            returns: [batch_size] - target returns
            old_values: [batch_size] - old value estimates
            
        Returns:
            loss_dict: Dictionary containing value loss components
        """
        # Get current value estimates
        current_values = self.value_network(states, latent_plans).squeeze(-1)
        
        if self.config.value_clip:
            # Clipped value loss
            value_clipped = old_values + torch.clamp(
                current_values - old_values,
                -self.config.clip_epsilon,
                self.config.clip_epsilon
            )
            
            value_loss1 = F.mse_loss(current_values, returns)
            value_loss2 = F.mse_loss(value_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            # Standard value loss
            value_loss = F.mse_loss(current_values, returns)
        
        return {
            "value_loss": value_loss,
            "value_error": torch.abs(current_values - returns).mean(),
            "value_variance": torch.var(current_values)
        }
    
    def update(self, training_step: PPOTrainingStep) -> Dict[str, float]:
        """
        Perform PPO update
        
        Args:
            training_step: PPOTrainingStep containing training data
            
        Returns:
            metrics: Training metrics
        """
        # Normalize advantages if configured
        if self.config.normalize_advantages:
            advantages = (training_step.advantages - training_step.advantages.mean()) / (training_step.advantages.std() + 1e-8)
        else:
            advantages = training_step.advantages
        
        # Prepare data
        states = training_step.states
        latent_plans = training_step.latent_plans
        actions = training_step.actions
        old_log_probs = training_step.log_probs
        returns = training_step.returns
        old_values = training_step.values
        importance_weights = training_step.importance_weights
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_divergence = 0.0
        
        # Multiple epochs of updates
        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            batch_size = len(states)
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_latent_plans = latent_plans[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_importance_weights = importance_weights[batch_indices] if importance_weights is not None else None
                
                # Compute losses
                policy_loss_dict = self.compute_policy_loss(
                    batch_states, batch_latent_plans, batch_actions,
                    batch_old_log_probs, batch_advantages, batch_importance_weights
                )
                
                value_loss_dict = self.compute_value_loss(
                    batch_states, batch_latent_plans, batch_returns, batch_old_values
                )
                
                # Check KL divergence for early stopping
                kl_divergence = policy_loss_dict["kl_divergence"].item()
                if kl_divergence > self.config.target_kl:
                    logger.warning(f"Early stopping due to high KL divergence: {kl_divergence}")
                    break
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss = policy_loss_dict["policy_loss"] + policy_loss_dict["entropy_loss"]
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(),
                    self.config.max_grad_norm
                )
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss = value_loss_dict["value_loss"] * self.config.value_loss_coef
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(),
                    self.config.max_grad_norm
                )
                self.value_optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss_dict["policy_loss"].item()
                total_value_loss += value_loss_dict["value_loss"].item()
                total_entropy_loss += policy_loss_dict["entropy_loss"].item()
                total_kl_divergence += kl_divergence
        
        # Update training state
        self.training_step += 1
        
        # Record metrics
        metrics = {
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy_loss": total_entropy_loss,
            "kl_divergence": total_kl_divergence,
            "training_step": self.training_step,
            "ratio_mean": policy_loss_dict["ratio"].item(),
            "entropy": policy_loss_dict["entropy"].item(),
            "value_error": value_loss_dict["value_error"].item(),
            "value_variance": value_loss_dict["value_variance"].item()
        }
        
        # Store in history
        self.kl_divergences.append(metrics["kl_divergence"])
        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.entropy_losses.append(metrics["entropy_loss"])
        
        return metrics
    
    def get_value_estimate(self, state: torch.Tensor, latent_plan: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state and latent plan"""
        with torch.no_grad():
            return self.value_network(state, latent_plan).squeeze(-1)
    
    def get_action_and_log_prob(self, state: torch.Tensor, latent_plan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability from policy"""
        return self.policy_network.sample_action(state, latent_plan)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        return {
            "training_step": self.training_step,
            "kl_divergences": list(self.kl_divergences),
            "policy_losses": list(self.policy_losses),
            "value_losses": list(self.value_losses),
            "entropy_losses": list(self.entropy_losses),
            "avg_kl_divergence": np.mean(self.kl_divergences) if self.kl_divergences else 0.0,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else 0.0,
            "avg_value_loss": np.mean(self.value_losses) if self.value_losses else 0.0,
            "avg_entropy_loss": np.mean(self.entropy_losses) if self.entropy_losses else 0.0,
            "policy_params": sum(p.numel() for p in self.policy_network.parameters()),
            "value_params": sum(p.numel() for p in self.value_network.parameters())
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "training_step": self.training_step,
            "config": self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved PPO checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded PPO checkpoint from {filepath}")

# Import PolicyNetwork from latent_planning module
from .latent_planning import PolicyNetwork
