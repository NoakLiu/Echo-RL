"""
Base Environment Interface for EchoRL

Defines the abstract base class and common interfaces for all EchoRL environments.
All task-specific environments should inherit from EchoRLEnvironment.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Base configuration for EchoRL environments"""
    max_steps: int = 1000
    reward_scale: float = 1.0
    success_threshold: float = 0.8
    timeout_penalty: float = -1.0
    device: str = "cuda"
    seed: Optional[int] = None

@dataclass
class EnvironmentResult:
    """Result of environment interaction"""
    success: bool
    reward: float
    total_reward: float
    steps_taken: int
    max_steps: int
    success_rate: float
    episode_length: int
    metadata: Dict[str, Any]

@dataclass
class EnvironmentState:
    """Represents the current state of an environment"""
    observation: torch.Tensor
    reward: float
    done: bool
    info: Dict[str, Any]
    step_count: int
    episode_reward: float

class EchoRLEnvironment(ABC):
    """
    Abstract base class for EchoRL environments
    
    All task-specific environments should inherit from this class and implement
    the required methods for state representation, action execution, and reward computation.
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.device = config.device
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0
        self.total_rewards = []
        self.success_count = 0
        
        # Set random seed if provided
        if config.seed is not None:
            self._set_seed(config.seed)
    
    @abstractmethod
    def reset(self) -> EnvironmentState:
        """
        Reset environment to initial state
        
        Returns:
            initial_state: EnvironmentState with initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, torch.Tensor, str]) -> EnvironmentState:
        """
        Execute action and return next state
        
        Args:
            action: Action to execute (type depends on environment)
            
        Returns:
            next_state: EnvironmentState after action execution
        """
        pass
    
    @abstractmethod
    def get_state_representation(self) -> torch.Tensor:
        """
        Get current state representation as tensor
        
        Returns:
            state_tensor: [state_dim] - current state representation
        """
        pass
    
    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get size of action space
        
        Returns:
            action_space_size: Number of possible actions
        """
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """
        Get dimension of state representation
        
        Returns:
            state_dim: Dimension of state tensor
        """
        pass
    
    def is_done(self) -> bool:
        """Check if episode is done"""
        return self.current_step >= self.config.max_steps
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for current episode"""
        return {
            "episode_count": self.episode_count,
            "current_step": self.current_step,
            "episode_reward": self.episode_reward,
            "max_steps": self.config.max_steps,
            "is_done": self.is_done()
        }
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get overall training statistics"""
        total_episodes = len(self.total_rewards)
        success_rate = self.success_count / max(total_episodes, 1)
        avg_reward = np.mean(self.total_rewards) if self.total_rewards else 0.0
        
        return {
            "total_episodes": total_episodes,
            "success_count": self.success_count,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "total_rewards": self.total_rewards[-100:] if self.total_rewards else []  # Last 100 episodes
        }
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _update_statistics(self, reward: float, success: bool = False):
        """Update internal statistics"""
        self.episode_reward += reward
        if success:
            self.success_count += 1
    
    def _finish_episode(self):
        """Finish current episode and update statistics"""
        self.total_rewards.append(self.episode_reward)
        self.episode_count += 1
        self.current_step = 0
        self.episode_reward = 0.0
    
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render environment (optional implementation)
        
        Args:
            mode: Rendering mode ("human", "rgb_array", etc.)
            
        Returns:
            rendered_frame: Rendered frame (type depends on mode)
        """
        return None
    
    def close(self):
        """Close environment and cleanup resources"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

class EnvironmentWrapper:
    """
    Wrapper class for modifying environment behavior
    
    Provides common functionality like reward shaping, action space modification,
    and state preprocessing.
    """
    
    def __init__(self, env: EchoRLEnvironment, **kwargs):
        self.env = env
        self.kwargs = kwargs
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment"""
        return getattr(self.env, name)
    
    def reset(self) -> EnvironmentState:
        """Reset wrapped environment"""
        return self.env.reset()
    
    def step(self, action: Union[int, torch.Tensor, str]) -> EnvironmentState:
        """Step wrapped environment"""
        return self.env.step(action)
    
    def get_state_representation(self) -> torch.Tensor:
        """Get state representation from wrapped environment"""
        return self.env.get_state_representation()
    
    def get_action_space_size(self) -> int:
        """Get action space size from wrapped environment"""
        return self.env.get_action_space_size()
    
    def get_state_dim(self) -> int:
        """Get state dimension from wrapped environment"""
        return self.env.get_state_dim()

class RewardShapingWrapper(EnvironmentWrapper):
    """Wrapper for reward shaping"""
    
    def __init__(self, env: EchoRLEnvironment, reward_shaper: callable):
        super().__init__(env)
        self.reward_shaper = reward_shaper
    
    def step(self, action: Union[int, torch.Tensor, str]) -> EnvironmentState:
        """Step with reward shaping"""
        state = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = self.reward_shaper(state.observation, state.reward, state.done, state.info)
        state.reward = shaped_reward
        
        return state

class ActionSpaceWrapper(EnvironmentWrapper):
    """Wrapper for action space modification"""
    
    def __init__(self, env: EchoRLEnvironment, action_mapper: callable):
        super().__init__(env)
        self.action_mapper = action_mapper
    
    def step(self, action: Union[int, torch.Tensor, str]) -> EnvironmentState:
        """Step with action mapping"""
        mapped_action = self.action_mapper(action)
        return self.env.step(mapped_action)

class StatePreprocessingWrapper(EnvironmentWrapper):
    """Wrapper for state preprocessing"""
    
    def __init__(self, env: EchoRLEnvironment, state_preprocessor: callable):
        super().__init__(env)
        self.state_preprocessor = state_preprocessor
    
    def get_state_representation(self) -> torch.Tensor:
        """Get preprocessed state representation"""
        raw_state = self.env.get_state_representation()
        return self.state_preprocessor(raw_state)
    
    def reset(self) -> EnvironmentState:
        """Reset with preprocessed state"""
        state = self.env.reset()
        state.observation = self.state_preprocessor(state.observation)
        return state
    
    def step(self, action: Union[int, torch.Tensor, str]) -> EnvironmentState:
        """Step with preprocessed state"""
        state = self.env.step(action)
        state.observation = self.state_preprocessor(state.observation)
        return state
