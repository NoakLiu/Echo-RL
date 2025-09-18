"""
Latent Planning Optimization Module

Implements the trajectory encoder F_φ and latent planning optimization
as described in the EchoRL paper. This enables structured rollout with 
continuation-based reasoning beyond reactive decoding.

Key Components:
- TrajectoryEncoder: Encodes state sequences into latent plans τ_t
- LatentPlanningOptimizer: Manages KL regularization and trajectory conditioning
- Policy conditioning on latent plans: π_θ(a_t | s_t, τ_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPrior:
    """Represents a latent trajectory prior τ_t"""
    latent_plan: torch.Tensor  # Shape: [d] where d is embedding dimension
    state_window: torch.Tensor  # Shape: [k+1, state_dim] - the input state sequence
    timestamp: int
    kl_divergence: Optional[float] = None
    surprise_score: Optional[float] = None

@dataclass
class PlanningConfig:
    """Configuration for latent planning optimization"""
    embedding_dim: int = 512
    state_window_size: int = 8  # k in the paper
    kl_weight: float = 0.1  # λ_KL
    lipschitz_constant: float = 1.0
    noise_std: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0

class TrajectoryEncoder(nn.Module):
    """
    Trajectory encoder F_φ that maps state sequences to latent plans
    
    Implements: τ_t = F_φ(s_{t-k:t})
    
    The encoder is designed to be Lipschitz continuous to ensure
    trajectory stability and enable KL regularization.
    """
    
    def __init__(self, 
                 state_dim: int,
                 config: PlanningConfig,
                 device: str = "cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.config = config
        self.device = device
        
        # Multi-layer transformer encoder for trajectory encoding
        self.input_projection = nn.Linear(state_dim, config.embedding_dim)
        
        # Positional encoding for temporal structure
        self.pos_encoding = nn.Parameter(
            torch.randn(config.state_window_size + 1, config.embedding_dim) * 0.1
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection to latent plan
        self.output_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim)
        )
        
        # Initialize weights for Lipschitz continuity
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to ensure Lipschitz continuity"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state_window: torch.Tensor) -> torch.Tensor:
        """
        Encode state window into latent plan
        
        Args:
            state_window: [batch_size, k+1, state_dim] - state sequence
            
        Returns:
            latent_plan: [batch_size, embedding_dim] - encoded trajectory prior
        """
        batch_size, seq_len, _ = state_window.shape
        
        # Project input states
        x = self.input_projection(state_window)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer encoding
        encoded = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        
        # Global average pooling to get trajectory representation
        trajectory_repr = encoded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Final projection to latent plan
        latent_plan = self.output_projection(trajectory_repr)
        
        return latent_plan
    
    def compute_kl_divergence(self, 
                              current_plan: torch.Tensor,
                              previous_plan: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between consecutive trajectory priors
        
        Implements: L_KL = D_KL[p_φ(τ_t | s_{1:t}) || p_φ(τ_{t-1} | s_{1:t-1})]
        
        Args:
            current_plan: [batch_size, embedding_dim] - τ_t
            previous_plan: [batch_size, embedding_dim] - τ_{t-1}
            
        Returns:
            kl_divergence: [batch_size] - KL divergence values
        """
        # Treat latent plans as Gaussian distributions
        # Current plan: N(τ_t, σ²I)
        # Previous plan: N(τ_{t-1}, σ²I)
        
        sigma_squared = self.config.noise_std ** 2
        
        # KL divergence between two Gaussians
        # D_KL(N(μ₁, σ²I) || N(μ₂, σ²I)) = ||μ₁ - μ₂||² / (2σ²)
        diff = current_plan - previous_plan
        kl_div = torch.sum(diff ** 2, dim=-1) / (2 * sigma_squared)
        
        return kl_div
    
    def get_lipschitz_bound(self) -> float:
        """Estimate Lipschitz constant of the encoder"""
        # Simple estimation based on layer norms
        total_bound = 1.0
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Frobenius norm of weight matrix
                weight_norm = torch.norm(module.weight, p='fro').item()
                total_bound *= weight_norm
        
        return total_bound

class PolicyNetwork(nn.Module):
    """
    Policy network that conditions on both state and latent plan
    
    Implements: π_θ(a_t | s_t, τ_t)
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 512,
                 device: str = "cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        
        # Combined processing
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor, latent_plan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of policy network
        
        Args:
            state: [batch_size, state_dim] - current state s_t
            latent_plan: [batch_size, latent_dim] - trajectory prior τ_t
            
        Returns:
            logits: [batch_size, action_dim] - action logits
        """
        # Encode state and latent plan
        state_repr = self.state_encoder(state)
        latent_repr = self.latent_encoder(latent_plan)
        
        # Combine representations
        combined = torch.cat([state_repr, latent_repr], dim=-1)
        
        # Generate action logits
        logits = self.combined_processor(combined)
        
        return logits
    
    def get_action_probs(self, state: torch.Tensor, latent_plan: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from policy"""
        logits = self.forward(state, latent_plan)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor, latent_plan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        logits = self.forward(state, latent_plan)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action
        action = torch.multinomial(probs, 1).squeeze(-1)
        
        # Compute log probability
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, action_log_prob

class LatentPlanningOptimizer:
    """
    Main optimizer for latent planning with KL regularization
    
    Manages the trajectory encoder and policy network training,
    implementing the core EchoRL latent planning optimization.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: PlanningConfig,
                 device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Initialize networks
        self.trajectory_encoder = TrajectoryEncoder(state_dim, config, device)
        self.policy_network = PolicyNetwork(state_dim, action_dim, config.embedding_dim, device=device)
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.trajectory_encoder.parameters(),
            lr=config.learning_rate
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.training_step = 0
        self.kl_loss_history = []
        self.planning_loss_history = []
        
    def encode_trajectory(self, state_window: torch.Tensor) -> TrajectoryPrior:
        """
        Encode a state window into a trajectory prior
        
        Args:
            state_window: [batch_size, k+1, state_dim] - state sequence
            
        Returns:
            trajectory_prior: TrajectoryPrior object
        """
        with torch.no_grad():
            latent_plan = self.trajectory_encoder(state_window)
            
            # Add noise for exploration
            noise = torch.randn_like(latent_plan) * self.config.noise_std
            latent_plan = latent_plan + noise
            
            return TrajectoryPrior(
                latent_plan=latent_plan,
                state_window=state_window,
                timestamp=self.training_step
            )
    
    def compute_planning_loss(self,
                             current_state_window: torch.Tensor,
                             previous_state_window: torch.Tensor,
                             actions: torch.Tensor,
                             rewards: torch.Tensor,
                             advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the complete planning loss including KL regularization
        
        Args:
            current_state_window: [batch_size, k+1, state_dim] - s_{t-k:t}
            previous_state_window: [batch_size, k+1, state_dim] - s_{t-k-1:t-1}
            actions: [batch_size] - actions taken
            rewards: [batch_size] - rewards received
            advantages: [batch_size] - computed advantages
            
        Returns:
            loss_dict: Dictionary containing all loss components
        """
        # Encode trajectories
        current_plan = self.trajectory_encoder(current_state_window)
        previous_plan = self.trajectory_encoder(previous_state_window)
        
        # Compute KL divergence
        kl_divergence = self.trajectory_encoder.compute_kl_divergence(current_plan, previous_plan)
        kl_loss = kl_divergence.mean()
        
        # Policy loss (PPO-style)
        current_states = current_state_window[:, -1, :]  # Last state in window
        action_log_probs = self.policy_network.get_action_probs(current_states, current_plan)
        action_log_probs = torch.log(action_log_probs.gather(1, actions.unsqueeze(-1)) + 1e-8).squeeze(-1)
        
        # PPO clipped objective
        old_action_log_probs = action_log_probs.detach()
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Total loss
        total_loss = policy_loss + self.config.kl_weight * kl_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "kl_divergence": kl_divergence.mean(),
            "current_plan": current_plan,
            "previous_plan": previous_plan
        }
    
    def update(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update both trajectory encoder and policy network
        
        Args:
            loss_dict: Loss dictionary from compute_planning_loss
            
        Returns:
            metrics: Training metrics
        """
        total_loss = loss_dict["total_loss"]
        
        # Update trajectory encoder
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.trajectory_encoder.parameters(),
            self.config.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.config.max_grad_norm
        )
        
        self.encoder_optimizer.step()
        self.policy_optimizer.step()
        
        self.training_step += 1
        
        # Record metrics
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": loss_dict["policy_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item(),
            "kl_divergence": loss_dict["kl_divergence"].item(),
            "training_step": self.training_step
        }
        
        self.kl_loss_history.append(metrics["kl_loss"])
        self.planning_loss_history.append(metrics["total_loss"])
        
        return metrics
    
    def get_trajectory_surprise(self, trajectory_prior: TrajectoryPrior) -> float:
        """
        Compute surprise score for trajectory prioritization
        
        Implements: score(t) = ||τ_t - E[τ]||² + α * r_t
        
        Args:
            trajectory_prior: TrajectoryPrior object
            
        Returns:
            surprise_score: Surprise score for replay prioritization
        """
        # Compute distance from expected trajectory
        latent_plan = trajectory_prior.latent_plan
        
        # For now, use simple L2 norm as surprise metric
        # In practice, this would be computed against a running average
        surprise_score = torch.norm(latent_plan).item()
        
        trajectory_prior.surprise_score = surprise_score
        return surprise_score
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            "trajectory_encoder": self.trajectory_encoder.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "training_step": self.training_step,
            "config": self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.trajectory_encoder.load_state_dict(checkpoint["trajectory_encoder"])
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            "training_step": self.training_step,
            "kl_loss_history": self.kl_loss_history[-100:],  # Last 100 steps
            "planning_loss_history": self.planning_loss_history[-100:],
            "lipschitz_bound": self.trajectory_encoder.get_lipschitz_bound(),
            "encoder_params": sum(p.numel() for p in self.trajectory_encoder.parameters()),
            "policy_params": sum(p.numel() for p in self.policy_network.parameters())
        }
