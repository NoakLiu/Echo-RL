"""
EchoRL Environment Interfaces

This module provides environment interfaces for various tasks mentioned in the EchoRL paper:
- ALFWorld: Text-world control tasks
- WebShop: Web-based shopping agent tasks  
- CRUXEval: Code repair and debugging tasks
- ARC: Abstract reasoning tasks
- MiniGrid: Grid-world planning tasks
"""

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentResult
from .alfworld import ALFWorldEnvironment, ALFWorldConfig
from .webshop import WebShopEnvironment, WebShopConfig
from .cruxeval import CRUXEvalEnvironment, CRUXEvalConfig
from .arc import ARCEnvironment, ARCConfig
from .minigrid import MiniGridEnvironment, MiniGridConfig

__all__ = [
    "EchoRLEnvironment",
    "EnvironmentConfig", 
    "EnvironmentResult",
    "ALFWorldEnvironment",
    "ALFWorldConfig",
    "WebShopEnvironment",
    "WebShopConfig",
    "CRUXEvalEnvironment",
    "CRUXEvalConfig",
    "ARCEnvironment",
    "ARCConfig",
    "MiniGridEnvironment",
    "MiniGridConfig"
]
