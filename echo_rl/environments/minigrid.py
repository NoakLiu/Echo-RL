"""
MiniGrid Environment Interface

Implements the MiniGrid environment for grid-world planning tasks.
MiniGrid provides 2D grid environments with navigation, object manipulation,
and goal-oriented tasks requiring planning and reasoning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class MiniGridConfig(EnvironmentConfig):
    """Configuration for MiniGrid environment"""
    grid_size: int = 8
    num_objects: int = 3
    task_type: str = "key_door"  # key_door, multi_object, navigation
    state_encoding_dim: int = 512
    action_space_size: int = 7  # MiniGrid actions

class MiniGridEnvironment(EchoRLEnvironment):
    """
    MiniGrid environment for grid-world planning tasks
    
    Provides 2D grid environments with objects, goals, and navigation
    challenges requiring planning and spatial reasoning.
    """
    
    def __init__(self, config: MiniGridConfig):
        super().__init__(config)
        self.config = config
        
        # Environment state
        self.grid = np.zeros((config.grid_size, config.grid_size), dtype=int)
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # 0: right, 1: down, 2: left, 3: up
        self.carrying = None
        self.objects = {}
        self.goals = []
        self.walls = set()
        self.doors = {}
        self.keys = {}
        
        # Action mapping
        self.action_map = self._create_action_map()
        
        # Task setup
        self._setup_minigrid_task()
    
    def _create_action_map(self) -> Dict[int, str]:
        """Create mapping from action indices to MiniGrid commands"""
        return {
            0: "move_forward",
            1: "turn_left",
            2: "turn_right",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done"
        }
    
    def _setup_minigrid_task(self):
        """Setup MiniGrid task based on task type"""
        if self.config.task_type == "key_door":
            self._setup_key_door_task()
        elif self.config.task_type == "multi_object":
            self._setup_multi_object_task()
        elif self.config.task_type == "navigation":
            self._setup_navigation_task()
        else:
            self._setup_default_task()
    
    def _setup_key_door_task(self):
        """Setup key-door task"""
        # Create walls
        for i in range(self.config.grid_size):
            self.walls.add((0, i))  # Top wall
            self.walls.add((self.config.grid_size - 1, i))  # Bottom wall
            self.walls.add((i, 0))  # Left wall
            self.walls.add((i, self.config.grid_size - 1))  # Right wall
        
        # Add internal wall with door
        wall_y = self.config.grid_size // 2
        for x in range(1, self.config.grid_size - 1):
            if x != self.config.grid_size // 2:  # Leave space for door
                self.walls.add((x, wall_y))
        
        # Place door
        door_pos = (self.config.grid_size // 2, wall_y)
        self.doors[door_pos] = {"locked": True, "color": "red"}
        
        # Place key
        key_pos = (1, 1)
        self.keys[key_pos] = {"color": "red"}
        
        # Place goal
        goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self.goals.append(goal_pos)
        
        # Initialize grid
        self._update_grid()
    
    def _setup_multi_object_task(self):
        """Setup multi-object collection task"""
        # Create simple room
        for i in range(self.config.grid_size):
            self.walls.add((0, i))
            self.walls.add((self.config.grid_size - 1, i))
            self.walls.add((i, 0))
            self.walls.add((i, self.config.grid_size - 1))
        
        # Place objects to collect
        object_positions = [
            (2, 2), (4, 2), (2, 4), (4, 4)
        ]
        
        for i, pos in enumerate(object_positions):
            self.objects[pos] = {"type": "ball", "color": f"color_{i}"}
        
        # Place goal
        goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self.goals.append(goal_pos)
        
        # Initialize grid
        self._update_grid()
    
    def _setup_navigation_task(self):
        """Setup navigation task"""
        # Create maze-like environment
        for i in range(self.config.grid_size):
            self.walls.add((0, i))
            self.walls.add((self.config.grid_size - 1, i))
            self.walls.add((i, 0))
            self.walls.add((i, self.config.grid_size - 1))
        
        # Add internal walls
        internal_walls = [
            (2, 2), (3, 2), (4, 2),
            (2, 4), (3, 4), (4, 4),
            (5, 3), (5, 4), (5, 5)
        ]
        
        for wall in internal_walls:
            self.walls.add(wall)
        
        # Place goal
        goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self.goals.append(goal_pos)
        
        # Initialize grid
        self._update_grid()
    
    def _setup_default_task(self):
        """Setup default task"""
        # Simple room with goal
        for i in range(self.config.grid_size):
            self.walls.add((0, i))
            self.walls.add((self.config.grid_size - 1, i))
            self.walls.add((i, 0))
            self.walls.add((i, self.config.grid_size - 1))
        
        # Place goal
        goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self.goals.append(goal_pos)
        
        # Initialize grid
        self._update_grid()
    
    def _update_grid(self):
        """Update grid representation"""
        self.grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=int)
        
        # Mark walls
        for wall in self.walls:
            self.grid[wall] = 1
        
        # Mark doors
        for door_pos in self.doors:
            self.grid[door_pos] = 2
        
        # Mark keys
        for key_pos in self.keys:
            self.grid[key_pos] = 3
        
        # Mark objects
        for obj_pos in self.objects:
            self.grid[obj_pos] = 4
        
        # Mark goals
        for goal_pos in self.goals:
            self.grid[goal_pos] = 5
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset agent state
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.carrying = None
        
        # Reset task-specific state
        if self.config.task_type == "key_door":
            # Reset door state
            for door_pos in self.doors:
                self.doors[door_pos]["locked"] = True
        
        # Update grid
        self._update_grid()
        
        # Generate initial observation
        observation = self._generate_observation()
        
        # Create initial state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=0.0,
            done=False,
            info={
                "agent_pos": self.agent_pos,
                "agent_dir": self.agent_dir,
                "carrying": self.carrying,
                "grid": self.grid.copy(),
                "task_type": self.config.task_type,
                "goals": self.goals.copy()
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
            action = 6  # Default to done
        
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
                "agent_pos": self.agent_pos,
                "agent_dir": self.agent_dir,
                "carrying": self.carrying,
                "grid": self.grid.copy(),
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
        
        if action_command == "move_forward":
            reward, success = self._action_move_forward()
        elif action_command == "turn_left":
            reward, success = self._action_turn_left()
        elif action_command == "turn_right":
            reward, success = self._action_turn_right()
        elif action_command == "pickup":
            reward, success = self._action_pickup()
        elif action_command == "drop":
            reward, success = self._action_drop()
        elif action_command == "toggle":
            reward, success = self._action_toggle()
        elif action_command == "done":
            reward, success = self._action_done()
        else:
            # Default action
            reward = 0.0
            success = True
        
        return reward, success
    
    def _action_move_forward(self) -> Tuple[float, bool]:
        """Move agent forward"""
        # Calculate new position
        dx, dy = self._get_direction_vector()
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
            self._update_grid()
            return 0.1, True
        else:
            return -0.1, False
    
    def _action_turn_left(self) -> Tuple[float, bool]:
        """Turn agent left"""
        self.agent_dir = (self.agent_dir - 1) % 4
        return 0.0, True
    
    def _action_turn_right(self) -> Tuple[float, bool]:
        """Turn agent right"""
        self.agent_dir = (self.agent_dir + 1) % 4
        return 0.0, True
    
    def _action_pickup(self) -> Tuple[float, bool]:
        """Pick up object at current position"""
        if self.carrying is not None:
            return -0.1, False  # Already carrying something
        
        if self.agent_pos in self.keys:
            # Pick up key
            self.carrying = self.keys[self.agent_pos]
            del self.keys[self.agent_pos]
            self._update_grid()
            return 1.0, True
        elif self.agent_pos in self.objects:
            # Pick up object
            self.carrying = self.objects[self.agent_pos]
            del self.objects[self.agent_pos]
            self._update_grid()
            return 0.5, True
        else:
            return -0.1, False
    
    def _action_drop(self) -> Tuple[float, bool]:
        """Drop carried object"""
        if self.carrying is None:
            return -0.1, False
        
        # Drop object at current position
        if self.agent_pos not in self.walls and self.agent_pos not in self.doors:
            if self.carrying.get("type") == "key":
                self.keys[self.agent_pos] = self.carrying
            else:
                self.objects[self.agent_pos] = self.carrying
            
            self.carrying = None
            self._update_grid()
            return 0.2, True
        else:
            return -0.1, False
    
    def _action_toggle(self) -> Tuple[float, bool]:
        """Toggle door or switch"""
        if self.agent_pos in self.doors:
            door = self.doors[self.agent_pos]
            if door["locked"] and self.carrying and self.carrying.get("color") == door["color"]:
                # Unlock door with matching key
                door["locked"] = False
                self.carrying = None  # Key is consumed
                self._update_grid()
                return 2.0, True
            elif not door["locked"]:
                # Door is already open
                return 0.0, True
            else:
                return -0.1, False
        else:
            return -0.1, False
    
    def _action_done(self) -> Tuple[float, bool]:
        """Mark task as done"""
        if self._is_task_complete():
            return 10.0, True
        else:
            return -1.0, False
    
    def _get_direction_vector(self) -> Tuple[int, int]:
        """Get direction vector based on agent direction"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        return directions[self.agent_dir]
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not wall or locked door)"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= self.config.grid_size or y < 0 or y >= self.config.grid_size:
            return False
        
        # Check walls
        if pos in self.walls:
            return False
        
        # Check locked doors
        if pos in self.doors and self.doors[pos]["locked"]:
            return False
        
        return True
    
    def _is_task_complete(self) -> bool:
        """Check if task is complete"""
        if self.config.task_type == "key_door":
            # Task complete if agent reaches goal
            return self.agent_pos in self.goals
        elif self.config.task_type == "multi_object":
            # Task complete if all objects collected and agent reaches goal
            return len(self.objects) == 0 and self.agent_pos in self.goals
        elif self.config.task_type == "navigation":
            # Task complete if agent reaches goal
            return self.agent_pos in self.goals
        else:
            return self.agent_pos in self.goals
    
    def _get_task_progress(self) -> float:
        """Get task completion progress"""
        if self.config.task_type == "key_door":
            # Progress based on distance to goal and key collection
            goal_dist = self._get_distance_to_goal()
            max_dist = self.config.grid_size * 2
            distance_progress = 1.0 - (goal_dist / max_dist)
            
            key_progress = 0.0
            if self.carrying and self.carrying.get("type") == "key":
                key_progress = 1.0
            
            return (distance_progress + key_progress) / 2.0
        
        elif self.config.task_type == "multi_object":
            # Progress based on objects collected
            total_objects = self.config.num_objects
            collected_objects = total_objects - len(self.objects)
            return collected_objects / total_objects
        
        else:
            # Progress based on distance to goal
            goal_dist = self._get_distance_to_goal()
            max_dist = self.config.grid_size * 2
            return 1.0 - (goal_dist / max_dist)
    
    def _get_distance_to_goal(self) -> float:
        """Get Manhattan distance to nearest goal"""
        if not self.goals:
            return 0.0
        
        min_dist = float('inf')
        for goal in self.goals:
            dist = abs(self.agent_pos[0] - goal[0]) + abs(self.agent_pos[1] - goal[1])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _generate_observation(self) -> str:
        """Generate text observation of current state"""
        obs_parts = []
        
        # Agent state
        obs_parts.append(f"Agent position: {self.agent_pos}")
        obs_parts.append(f"Agent direction: {self.agent_dir}")
        obs_parts.append(f"Carrying: {self.carrying}")
        
        # Task progress
        progress = self._get_task_progress()
        obs_parts.append(f"Progress: {progress:.2f}")
        
        # Grid visualization (small grids only)
        if self.config.grid_size <= 8:
            obs_parts.append("Grid:")
            for i in range(self.config.grid_size):
                row = []
                for j in range(self.config.grid_size):
                    if (i, j) == self.agent_pos:
                        row.append("A")
                    elif (i, j) in self.walls:
                        row.append("#")
                    elif (i, j) in self.doors:
                        row.append("D")
                    elif (i, j) in self.keys:
                        row.append("K")
                    elif (i, j) in self.objects:
                        row.append("O")
                    elif (i, j) in self.goals:
                        row.append("G")
                    else:
                        row.append(".")
                obs_parts.append(" ".join(row))
        
        return '\n'.join(obs_parts)
    
    def get_state_representation(self) -> torch.Tensor:
        """Get current state as tensor representation"""
        # Create feature vector from environment state
        features = []
        
        # Grid features (flattened)
        grid_flat = self.grid.flatten()
        features.extend(grid_flat.tolist())
        
        # Agent features
        features.append(self.agent_pos[0] / self.config.grid_size)
        features.append(self.agent_pos[1] / self.config.grid_size)
        features.append(self.agent_dir / 4.0)
        
        # Carrying features
        if self.carrying:
            features.append(1.0)
            features.append(self.carrying.get("type", "unknown") == "key")
        else:
            features.extend([0.0, 0.0])
        
        # Goal features
        goal_dist = self._get_distance_to_goal()
        features.append(goal_dist / (self.config.grid_size * 2))
        
        # Task progress
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
    
    def get_minigrid_info(self) -> Dict[str, Any]:
        """Get MiniGrid-specific information"""
        return {
            "task_type": self.config.task_type,
            "grid_size": self.config.grid_size,
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "carrying": self.carrying,
            "grid": self.grid.copy(),
            "walls": list(self.walls),
            "doors": self.doors.copy(),
            "keys": self.keys.copy(),
            "objects": self.objects.copy(),
            "goals": self.goals.copy()
        }
    
    def render(self, mode: str = "text") -> str:
        """Render environment state as text"""
        if mode == "text":
            return self._generate_observation()
        else:
            return super().render(mode)
