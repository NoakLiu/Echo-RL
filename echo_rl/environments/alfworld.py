"""
ALFWorld Environment Interface

Implements the ALFWorld text-world control environment for EchoRL.
ALFWorld provides interactive text-based environments where agents must
navigate and manipulate objects to complete tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import re

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class ALFWorldConfig(EnvironmentConfig):
    """Configuration for ALFWorld environment"""
    task_type: str = "pick_and_place"  # pick_and_place, clean_and_place, etc.
    max_objects: int = 10
    room_size: int = 5
    use_vision: bool = False
    text_encoding_dim: int = 512
    action_space_size: int = 20  # Common ALFWorld actions

class ALFWorldEnvironment(EchoRLEnvironment):
    """
    ALFWorld environment for text-world control tasks
    
    Provides a simplified interface to ALFWorld tasks focusing on
    object manipulation and navigation in text-based environments.
    """
    
    def __init__(self, config: ALFWorldConfig):
        super().__init__(config)
        self.config = config
        
        # Environment state
        self.current_room = None
        self.inventory = []
        self.object_locations = {}
        self.task_objects = []
        self.task_description = ""
        self.current_observation = ""
        
        # Action mapping
        self.action_map = self._create_action_map()
        
        # Task-specific setup
        self._setup_task()
    
    def _create_action_map(self) -> Dict[int, str]:
        """Create mapping from action indices to ALFWorld commands"""
        return {
            0: "go to",
            1: "take",
            2: "put",
            3: "open",
            4: "close",
            5: "toggle",
            6: "clean",
            7: "heat",
            8: "cool",
            9: "examine",
            10: "use",
            11: "look",
            12: "inventory",
            13: "north",
            14: "south", 
            15: "east",
            16: "west",
            17: "up",
            18: "down",
            19: "wait"
        }
    
    def _setup_task(self):
        """Setup task-specific configuration"""
        if self.config.task_type == "pick_and_place":
            self.task_objects = ["apple", "banana", "orange"]
            self.task_description = "Pick up objects and place them in the correct locations"
        elif self.config.task_type == "clean_and_place":
            self.task_objects = ["dirty_plate", "sponge", "sink"]
            self.task_description = "Clean dirty objects and place them appropriately"
        else:
            self.task_objects = ["object1", "object2", "object3"]
            self.task_description = "Complete the assigned task"
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Initialize room and objects
        self._initialize_room()
        self._place_objects()
        
        # Generate initial observation
        self.current_observation = self._generate_observation()
        
        # Create initial state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=0.0,
            done=False,
            info={
                "task_description": self.task_description,
                "inventory": self.inventory.copy(),
                "room_description": self._get_room_description()
            },
            step_count=self.current_step,
            episode_reward=self.episode_reward
        )
        
        return state
    
    def _initialize_room(self):
        """Initialize room layout"""
        self.current_room = {
            "name": "kitchen",
            "objects": [],
            "containers": ["counter", "sink", "table"],
            "connections": ["living_room", "dining_room"]
        }
        
        # Reset inventory and object locations
        self.inventory = []
        self.object_locations = {}
    
    def _place_objects(self):
        """Place task objects in the environment"""
        locations = ["counter", "sink", "table", "floor"]
        
        for i, obj in enumerate(self.task_objects):
            location = locations[i % len(locations)]
            self.object_locations[obj] = location
            self.current_room["objects"].append(obj)
    
    def _generate_observation(self) -> str:
        """Generate text observation of current state"""
        room_desc = self._get_room_description()
        inventory_desc = f"Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}"
        
        observation = f"{room_desc}\n{inventory_desc}\nTask: {self.task_description}"
        return observation
    
    def _get_room_description(self) -> str:
        """Get description of current room"""
        objects_desc = ", ".join(self.current_room["objects"])
        containers_desc = ", ".join(self.current_room["containers"])
        
        return f"You are in the {self.current_room['name']}. You can see: {objects_desc}. Available surfaces: {containers_desc}."
    
    def step(self, action: int) -> EnvironmentState:
        """Execute action and return next state"""
        self.current_step += 1
        
        # Get action command
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        if action not in self.action_map:
            action = 19  # Default to wait
        
        action_command = self.action_map[action]
        
        # Execute action and get reward
        reward, success = self._execute_action(action_command)
        self.episode_reward += reward
        
        # Check if task is complete
        done = self._is_task_complete() or self.is_done()
        
        # Generate new observation
        self.current_observation = self._generate_observation()
        
        # Create next state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=reward,
            done=done,
            info={
                "action_executed": action_command,
                "success": success,
                "inventory": self.inventory.copy(),
                "room_description": self._get_room_description(),
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
        
        # Parse action command
        if action_command == "take":
            reward, success = self._action_take()
        elif action_command == "put":
            reward, success = self._action_put()
        elif action_command == "go to":
            reward, success = self._action_go_to()
        elif action_command == "examine":
            reward, success = self._action_examine()
        elif action_command == "inventory":
            reward, success = self._action_inventory()
        elif action_command in ["north", "south", "east", "west", "up", "down"]:
            reward, success = self._action_move(action_command)
        else:
            # Default action (wait, look, etc.)
            reward = 0.0
            success = True
        
        return reward, success
    
    def _action_take(self) -> Tuple[float, bool]:
        """Take an object"""
        # Simplified: take first available object
        available_objects = [obj for obj in self.current_room["objects"] 
                           if obj not in self.inventory]
        
        if available_objects:
            obj = available_objects[0]
            self.inventory.append(obj)
            self.current_room["objects"].remove(obj)
            return 1.0, True
        
        return -0.1, False  # Penalty for failed action
    
    def _action_put(self) -> Tuple[float, bool]:
        """Put an object somewhere"""
        if not self.inventory:
            return -0.1, False
        
        # Simplified: put first object in inventory
        obj = self.inventory[0]
        self.inventory.remove(obj)
        self.current_room["objects"].append(obj)
        
        # Check if this completes part of the task
        if self._is_object_correctly_placed(obj):
            return 2.0, True
        
        return 0.5, True
    
    def _action_go_to(self) -> Tuple[float, bool]:
        """Go to a location"""
        # Simplified: just return small reward for movement
        return 0.1, True
    
    def _action_examine(self) -> Tuple[float, bool]:
        """Examine current location"""
        return 0.1, True
    
    def _action_inventory(self) -> Tuple[float, bool]:
        """Check inventory"""
        return 0.0, True
    
    def _action_move(self, direction: str) -> Tuple[float, bool]:
        """Move in specified direction"""
        return 0.1, True
    
    def _is_object_correctly_placed(self, obj: str) -> bool:
        """Check if object is correctly placed for task"""
        # Simplified task completion logic
        if self.config.task_type == "pick_and_place":
            # Check if object is on a surface (not on floor)
            return obj in self.current_room["objects"]
        return True
    
    def _is_task_complete(self) -> bool:
        """Check if task is complete"""
        if self.config.task_type == "pick_and_place":
            # Task complete if all objects are placed correctly
            return len(self.inventory) == 0 and len(self.current_room["objects"]) == len(self.task_objects)
        return False
    
    def _get_task_progress(self) -> float:
        """Get task completion progress"""
        if self.config.task_type == "pick_and_place":
            total_objects = len(self.task_objects)
            placed_objects = len(self.current_room["objects"])
            return placed_objects / total_objects
        return 0.0
    
    def get_state_representation(self) -> torch.Tensor:
        """Get current state as tensor representation"""
        # Convert text observation to tensor
        # This is a simplified implementation - in practice, you'd use a proper text encoder
        
        # Create feature vector from environment state
        features = []
        
        # Room features
        features.append(len(self.current_room["objects"]))
        features.append(len(self.current_room["containers"]))
        features.append(len(self.inventory))
        
        # Object location features
        for obj in self.task_objects:
            if obj in self.inventory:
                features.append(1.0)  # In inventory
            elif obj in self.current_room["objects"]:
                features.append(0.5)  # In room
            else:
                features.append(0.0)  # Not found
        
        # Task progress
        features.append(self._get_task_progress())
        
        # Pad or truncate to fixed size
        target_size = self.config.text_encoding_dim
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return self.config.action_space_size
    
    def get_state_dim(self) -> int:
        """Get dimension of state representation"""
        return self.config.text_encoding_dim
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task-specific information"""
        return {
            "task_type": self.config.task_type,
            "task_description": self.task_description,
            "task_objects": self.task_objects,
            "current_room": self.current_room,
            "object_locations": self.object_locations,
            "inventory": self.inventory
        }
    
    def render(self, mode: str = "text") -> str:
        """Render environment state as text"""
        if mode == "text":
            return self.current_observation
        else:
            return super().render(mode)
