"""
Game24 Environment Interface

Arithmetic planning benchmark from the EchoRL paper. Given four numbers,
the agent must combine them with +, -, *, / to reach 24.
"""

import itertools
import random
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)


@dataclass
class Game24Config(EnvironmentConfig):
    """Configuration for Game24 environment"""
    state_encoding_dim: int = 512
    action_space_size: int = 20
    allow_fractions: bool = True
    max_expression_length: int = 64


class Game24Environment(EchoRLEnvironment):
    """Game24 arithmetic planning environment."""

    OPERATORS = ["+", "-", "*", "/"]

    def __init__(self, config: Game24Config):
        super().__init__(config)
        self.config = config
        self.numbers: List[int] = []
        self.expression: str = ""
        self.action_map = self._create_action_map()

    def _create_action_map(self) -> Dict[int, str]:
        actions = {
            0: "append_0", 1: "append_1", 2: "append_2", 3: "append_3",
            4: "append_4", 5: "append_5", 6: "append_6", 7: "append_7",
            8: "append_8", 9: "append_9",
            10: "append_plus", 11: "append_minus", 12: "append_times",
            13: "append_div", 14: "append_lparen", 15: "append_rparen",
            16: "append_space", 17: "backspace", 18: "submit", 19: "reset_expr",
        }
        return actions

    def _sample_numbers(self) -> List[int]:
        while True:
            nums = [random.randint(1, 9) for _ in range(4)]
            if self._has_solution(nums):
                return nums

    @staticmethod
    def _has_solution(numbers: List[int]) -> bool:
        for perm in set(itertools.permutations(numbers)):
            for ops in itertools.product(Game24Environment.OPERATORS, repeat=3):
                exprs = [
                    f"(({perm[0]}{ops[0]}{perm[1]}){ops[1]}{perm[2]}){ops[2]}{perm[3]}",
                    f"({perm[0]}{ops[0]}{perm[1]}{ops[1]}{perm[2]}){ops[2]}{perm[3]}",
                    f"{perm[0]}{ops[0]}({perm[1]}{ops[1]}{perm[2]}{ops[2]}{perm[3]})",
                    f"({perm[0]}{ops[0]}{perm[1]}){ops[1]}({perm[2]}{ops[2]}{perm[3]})",
                ]
                for expr in exprs:
                    try:
                        if abs(eval(expr) - 24) < 1e-6:  # noqa: S307 - controlled puzzle eval
                            return True
                    except ZeroDivisionError:
                        continue
        return False

    def reset(self) -> EnvironmentState:
        self.current_step = 0
        self.episode_reward = 0.0
        self.numbers = self._sample_numbers()
        self.expression = ""
        obs = self._encode_observation()
        return EnvironmentState(
            observation=obs,
            reward=0.0,
            done=False,
            info={"numbers": self.numbers, "expression": self.expression},
            step_count=0,
            episode_reward=0.0,
        )

    def step(self, action: int) -> EnvironmentState:
        self.current_step += 1
        reward = -0.01
        done = False
        info: Dict[str, Any] = {}

        token = self.action_map.get(action, "")
        if token == "backspace":
            self.expression = self.expression[:-1]
        elif token == "reset_expr":
            self.expression = ""
        elif token == "submit":
            reward, done, info = self._evaluate_expression()
        elif token.startswith("append_"):
            ch = token.replace("append_", "")
            mapping = {
                "plus": "+", "minus": "-", "times": "*", "div": "/",
                "lparen": "(", "rparen": ")", "space": " ",
            }
            self.expression += mapping.get(ch, ch)

        if self.current_step >= self.config.max_steps:
            done = True
            reward += self.config.timeout_penalty

        self.episode_reward += reward
        obs = self._encode_observation()
        return EnvironmentState(
            observation=obs,
            reward=reward,
            done=done,
            info={**info, "numbers": self.numbers, "expression": self.expression},
            step_count=self.current_step,
            episode_reward=self.episode_reward,
        )

    def _evaluate_expression(self) -> Tuple[float, bool, Dict[str, Any]]:
        expr = self.expression.strip()
        if not expr:
            return -0.5, False, {"valid": False, "reason": "empty"}

        used_digits = [int(d) for d in re.findall(r"\d", expr)]
        if sorted(used_digits) != sorted(self.numbers):
            return -0.5, False, {"valid": False, "reason": "wrong_numbers"}

        try:
            value = eval(expr)  # noqa: S307
            if abs(value - 24) < 1e-6:
                return 1.0, True, {"valid": True, "value": value, "success": True}
            return -0.2, True, {"valid": True, "value": value, "success": False}
        except Exception as exc:
            return -0.3, False, {"valid": False, "reason": str(exc)}

    def _encode_observation(self) -> torch.Tensor:
        obs = torch.zeros(self.config.state_encoding_dim)
        for i, n in enumerate(self.numbers[:4]):
            obs[i] = n / 9.0
        expr_bytes = self.expression.encode("utf-8")[: self.config.max_expression_length]
        for i, b in enumerate(expr_bytes):
            if i + 8 < self.config.state_encoding_dim:
                obs[i + 8] = b / 255.0
        obs[-1] = self.current_step / max(self.config.max_steps, 1)
        return obs.to(self.device)

    def get_state_representation(self) -> torch.Tensor:
        return self._encode_observation()

    def get_action_space_size(self) -> int:
        return self.config.action_space_size

    def get_observation_space_size(self) -> int:
        return self.config.state_encoding_dim

    def close(self):
        pass
