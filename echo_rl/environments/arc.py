"""
ARC Environment Interface

Implements the ARC (Abstraction and Reasoning Corpus) environment for
abstract reasoning tasks. ARC provides grid-based puzzles that require
pattern recognition and abstract reasoning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class ARCConfig(EnvironmentConfig):
    """Configuration for ARC environment"""
    grid_size: int = 10
    max_colors: int = 10
    task_type: str = "pattern_completion"  # pattern_completion, transformation, etc.
    state_encoding_dim: int = 512
    action_space_size: int = 15  # ARC actions

class ARCEnvironment(EchoRLEnvironment):
    """
    ARC environment for abstract reasoning tasks
    
    Provides grid-based puzzles requiring pattern recognition,
    transformation, and abstract reasoning capabilities.
    """
    
    def __init__(self, config: ARCConfig):
        super().__init__(config)
        self.config = config
        
        # Environment state
        self.input_grid = np.zeros((config.grid_size, config.grid_size), dtype=int)
        self.output_grid = np.zeros((config.grid_size, config.grid_size), dtype=int)
        self.target_grid = np.zeros((config.grid_size, config.grid_size), dtype=int)
        self.current_position = (0, 0)
        self.selected_color = 1
        self.grid_history = []
        
        # Action mapping
        self.action_map = self._create_action_map()
        
        # Task setup
        self._setup_arc_task()
    
    def _create_action_map(self) -> Dict[int, str]:
        """Create mapping from action indices to ARC commands"""
        return {
            0: "move_up",
            1: "move_down",
            2: "move_left",
            3: "move_right",
            4: "select_color",
            5: "paint_cell",
            6: "clear_cell",
            7: "copy_pattern",
            8: "rotate_pattern",
            9: "flip_pattern",
            10: "scale_pattern",
            11: "apply_transformation",
            12: "undo_action",
            13: "reset_grid",
            14: "submit_solution"
        }
    
    def _setup_arc_task(self):
        """Setup ARC task with input-output examples"""
        if self.config.task_type == "pattern_completion":
            self._setup_pattern_completion_task()
        elif self.config.task_type == "transformation":
            self._setup_transformation_task()
        else:
            self._setup_default_task()
    
    def _setup_pattern_completion_task(self):
        """Setup pattern completion task"""
        # Create input pattern
        self.input_grid = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2]
        ])
        
        # Create target pattern (extended version)
        self.target_grid = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1]
        ])
        
        # Initialize output grid
        self.output_grid = np.zeros_like(self.target_grid)
    
    def _setup_transformation_task(self):
        """Setup transformation task"""
        # Create input pattern
        self.input_grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        # Create target pattern (rotated 90 degrees)
        self.target_grid = np.array([
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ])
        
        # Initialize output grid
        self.output_grid = np.zeros_like(self.target_grid)
    
    def _setup_default_task(self):
        """Setup default task"""
        # Simple pattern task
        self.input_grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
        
        self.target_grid = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ])
        
        self.output_grid = np.zeros_like(self.target_grid)
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset grid state
        self.output_grid = np.zeros_like(self.target_grid)
        self.current_position = (0, 0)
        self.selected_color = 1
        self.grid_history = [self.output_grid.copy()]
        
        # Generate initial observation
        observation = self._generate_observation()
        
        # Create initial state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=0.0,
            done=False,
            info={
                "input_grid": self.input_grid.copy(),
                "output_grid": self.output_grid.copy(),
                "target_grid": self.target_grid.copy(),
                "current_position": self.current_position,
                "selected_color": self.selected_color,
                "task_type": self.config.task_type
            },
            step_count=self.current_step,
            episode_reward=self.episode_reward
        )
        
        return state
    
    def step(self, action: int) -> EnvironmentState:
        """Execute action and return next state"""
        self.current_step += 1
        
        # Get action command
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        if action not in self.action_map:
            action = 14  # Default to submit_solution
        
        action_command = self.action_map[action]
        
        # Execute action and get reward
        reward, success = self._execute_action(action_command)
        self.episode_reward += reward
        
        # Check if task is complete
        done = self._is_task_complete() or self.is_done()
        
        # Generate new observation
        observation = self._generate_observation()
        
        # Create next state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=reward,
            done=done,
            info={
                "action_executed": action_command,
                "success": success,
                "input_grid": self.input_grid.copy(),
                "output_grid": self.output_grid.copy(),
                "target_grid": self.target_grid.copy(),
                "current_position": self.current_position,
                "selected_color": self.selected_color,
                "task_progress": self._get_task_progress()
            },
            step_count=self.current_step,
            episode_reward=self.episode_reward
        )
        
        # Update statistics
        if done:
            self._update_statistics(reward, success)
            self._finish_episode()
        
        return state
    
    def _execute_action(self, action_command: str) -> Tuple[float, bool]:
        """Execute action command and return reward and success flag"""
        reward = 0.0
        success = False
        
        if action_command == "move_up":
            reward, success = self._action_move_up()
        elif action_command == "move_down":
            reward, success = self._action_move_down()
        elif action_command == "move_left":
            reward, success = self._action_move_left()
        elif action_command == "move_right":
            reward, success = self._action_move_right()
        elif action_command == "select_color":
            reward, success = self._action_select_color()
        elif action_command == "paint_cell":
            reward, success = self._action_paint_cell()
        elif action_command == "clear_cell":
            reward, success = self._action_clear_cell()
        elif action_command == "copy_pattern":
            reward, success = self._action_copy_pattern()
        elif action_command == "rotate_pattern":
            reward, success = self._action_rotate_pattern()
        elif action_command == "flip_pattern":
            reward, success = self._action_flip_pattern()
        elif action_command == "apply_transformation":
            reward, success = self._action_apply_transformation()
        elif action_command == "undo_action":
            reward, success = self._action_undo_action()
        elif action_command == "reset_grid":
            reward, success = self._action_reset_grid()
        elif action_command == "submit_solution":
            reward, success = self._action_submit_solution()
        else:
            # Default action
            reward = 0.0
            success = True
        
        return reward, success
    
    def _action_move_up(self) -> Tuple[float, bool]:
        """Move cursor up"""
        new_pos = (max(0, self.current_position[0] - 1), self.current_position[1])
        self.current_position = new_pos
        return 0.0, True
    
    def _action_move_down(self) -> Tuple[float, bool]:
        """Move cursor down"""
        new_pos = (min(self.output_grid.shape[0] - 1, self.current_position[0] + 1), self.current_position[1])
        self.current_position = new_pos
        return 0.0, True
    
    def _action_move_left(self) -> Tuple[float, bool]:
        """Move cursor left"""
        new_pos = (self.current_position[0], max(0, self.current_position[1] - 1))
        self.current_position = new_pos
        return 0.0, True
    
    def _action_move_right(self) -> Tuple[float, bool]:
        """Move cursor right"""
        new_pos = (self.current_position[0], min(self.output_grid.shape[1] - 1, self.current_position[1] + 1))
        self.current_position = new_pos
        return 0.0, True
    
    def _action_select_color(self) -> Tuple[float, bool]:
        """Select next color"""
        self.selected_color = (self.selected_color % self.config.max_colors) + 1
        return 0.0, True
    
    def _action_paint_cell(self) -> Tuple[float, bool]:
        """Paint current cell with selected color"""
        if 0 <= self.current_position[0] < self.output_grid.shape[0] and \
           0 <= self.current_position[1] < self.output_grid.shape[1]:
            self.output_grid[self.current_position] = self.selected_color
            self._save_grid_state()
            return 0.1, True
        return -0.1, False
    
    def _action_clear_cell(self) -> Tuple[float, bool]:
        """Clear current cell"""
        if 0 <= self.current_position[0] < self.output_grid.shape[0] and \
           0 <= self.current_position[1] < self.output_grid.shape[1]:
            self.output_grid[self.current_position] = 0
            self._save_grid_state()
            return 0.1, True
        return -0.1, False
    
    def _action_copy_pattern(self) -> Tuple[float, bool]:
        """Copy pattern from input to output"""
        # Simple pattern copying
        min_size = min(self.input_grid.shape[0], self.output_grid.shape[0])
        min_cols = min(self.input_grid.shape[1], self.output_grid.shape[1])
        
        for i in range(min_size):
            for j in range(min_cols):
                self.output_grid[i, j] = self.input_grid[i, j]
        
        self._save_grid_state()
        return 0.5, True
    
    def _action_rotate_pattern(self) -> Tuple[float, bool]:
        """Rotate pattern 90 degrees"""
        # Rotate the input pattern and apply to output
        rotated_input = np.rot90(self.input_grid)
        
        min_size = min(rotated_input.shape[0], self.output_grid.shape[0])
        min_cols = min(rotated_input.shape[1], self.output_grid.shape[1])
        
        for i in range(min_size):
            for j in range(min_cols):
                self.output_grid[i, j] = rotated_input[i, j]
        
        self._save_grid_state()
        return 0.5, True
    
    def _action_flip_pattern(self) -> Tuple[float, bool]:
        """Flip pattern horizontally"""
        # Flip the input pattern and apply to output
        flipped_input = np.fliplr(self.input_grid)
        
        min_size = min(flipped_input.shape[0], self.output_grid.shape[0])
        min_cols = min(flipped_input.shape[1], self.output_grid.shape[1])
        
        for i in range(min_size):
            for j in range(min_cols):
                self.output_grid[i, j] = flipped_input[i, j]
        
        self._save_grid_state()
        return 0.5, True
    
    def _action_apply_transformation(self) -> Tuple[float, bool]:
        """Apply transformation based on task type"""
        if self.config.task_type == "transformation":
            # Apply rotation transformation
            return self._action_rotate_pattern()
        elif self.config.task_type == "pattern_completion":
            # Apply pattern extension
            return self._action_extend_pattern()
        else:
            return self._action_copy_pattern()
    
    def _action_extend_pattern(self) -> Tuple[float, bool]:
        """Extend pattern (for pattern completion tasks)"""
        # Simple pattern extension: repeat the input pattern
        input_h, input_w = self.input_grid.shape
        output_h, output_w = self.output_grid.shape
        
        for i in range(output_h):
            for j in range(output_w):
                self.output_grid[i, j] = self.input_grid[i % input_h, j % input_w]
        
        self._save_grid_state()
        return 0.7, True
    
    def _action_undo_action(self) -> Tuple[float, bool]:
        """Undo last action"""
        if len(self.grid_history) > 1:
            self.grid_history.pop()
            self.output_grid = self.grid_history[-1].copy()
            return 0.1, True
        return -0.1, False
    
    def _action_reset_grid(self) -> Tuple[float, bool]:
        """Reset output grid"""
        self.output_grid = np.zeros_like(self.target_grid)
        self._save_grid_state()
        return 0.0, True
    
    def _action_submit_solution(self) -> Tuple[float, bool]:
        """Submit current solution"""
        # Check if solution matches target
        if np.array_equal(self.output_grid, self.target_grid):
            return 10.0, True  # Large reward for correct solution
        else:
            # Partial reward based on similarity
            similarity = np.mean(self.output_grid == self.target_grid)
            return similarity * 2.0, False
    
    def _save_grid_state(self):
        """Save current grid state to history"""
        self.grid_history.append(self.output_grid.copy())
        if len(self.grid_history) > 10:  # Limit history size
            self.grid_history.pop(0)
    
    def _is_task_complete(self) -> bool:
        """Check if ARC task is complete"""
        return np.array_equal(self.output_grid, self.target_grid)
    
    def _get_task_progress(self) -> float:
        """Get task completion progress"""
        if self.target_grid.size == 0:
            return 0.0
        
        correct_cells = np.sum(self.output_grid == self.target_grid)
        total_cells = self.target_grid.size
        return correct_cells / total_cells
    
    def _generate_observation(self) -> str:
        """Generate text observation of current state"""
        obs_parts = []
        
        # Grid information
        obs_parts.append(f"Grid size: {self.output_grid.shape}")
        obs_parts.append(f"Current position: {self.current_position}")
        obs_parts.append(f"Selected color: {self.selected_color}")
        
        # Task progress
        progress = self._get_task_progress()
        obs_parts.append(f"Progress: {progress:.2f}")
        
        # Grid visualization (small grids only)
        if self.output_grid.shape[0] <= 6 and self.output_grid.shape[1] <= 6:
            obs_parts.append("Output grid:")
            for row in self.output_grid:
                obs_parts.append(" ".join(map(str, row)))
        
        return '\n'.join(obs_parts)
    
    def get_state_representation(self) -> torch.Tensor:
        """Get current state as tensor representation"""
        # Flatten grids and create feature vector
        features = []
        
        # Input grid features
        input_flat = self.input_grid.flatten()
        features.extend(input_flat.tolist())
        
        # Output grid features
        output_flat = self.output_grid.flatten()
        features.extend(output_flat.tolist())
        
        # Target grid features
        target_flat = self.target_grid.flatten()
        features.extend(target_flat.tolist())
        
        # Position features
        features.append(self.current_position[0] / self.output_grid.shape[0])
        features.append(self.current_position[1] / self.output_grid.shape[1])
        
        # Color features
        features.append(self.selected_color / self.config.max_colors)
        
        # Progress features
        features.append(self._get_task_progress())
        
        # Pad or truncate to fixed size
        target_size = self.config.state_encoding_dim
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return self.config.action_space_size
    
    def get_state_dim(self) -> int:
        """Get dimension of state representation"""
        return self.config.state_encoding_dim
    
    def get_arc_info(self) -> Dict[str, Any]:
        """Get ARC-specific information"""
        return {
            "task_type": self.config.task_type,
            "grid_size": self.output_grid.shape,
            "input_grid": self.input_grid.copy(),
            "output_grid": self.output_grid.copy(),
            "target_grid": self.target_grid.copy(),
            "current_position": self.current_position,
            "selected_color": self.selected_color,
            "grid_history": [grid.copy() for grid in self.grid_history]
        }
    
    def render(self, mode: str = "text") -> str:
        """Render environment state as text"""
        if mode == "text":
            return self._generate_observation()
        else:
            return super().render(mode)
