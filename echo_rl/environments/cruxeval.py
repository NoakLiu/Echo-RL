"""
CRUXEval Environment Interface

Implements the CRUXEval environment for code repair and debugging tasks.
CRUXEval provides realistic programming scenarios where agents must identify
bugs, write fixes, and validate solutions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import ast
import subprocess
import tempfile
import os

from .base import EchoRLEnvironment, EnvironmentConfig, EnvironmentState

logger = logging.getLogger(__name__)

@dataclass
class CRUXEvalConfig(EnvironmentConfig):
    """Configuration for CRUXEval environment"""
    language: str = "python"  # python, javascript, java, etc.
    max_code_length: int = 1000
    max_test_cases: int = 10
    timeout_seconds: int = 5
    state_encoding_dim: int = 512
    action_space_size: int = 30  # CRUXEval actions

class CRUXEvalEnvironment(EchoRLEnvironment):
    """
    CRUXEval environment for code repair and debugging tasks
    
    Simulates realistic programming scenarios including bug identification,
    code fixing, and test validation.
    """
    
    def __init__(self, config: CRUXEvalConfig):
        super().__init__(config)
        self.config = config
        
        # Environment state
        self.current_code = ""
        self.buggy_code = ""
        self.test_cases = []
        self.failed_tests = []
        self.passed_tests = []
        self.error_messages = []
        self.code_history = []
        self.current_line = 0
        self.selected_text = ""
        
        # Action mapping
        self.action_map = self._create_action_map()
        
        # Code templates and bugs
        self.code_templates = self._create_code_templates()
        self.bug_patterns = self._create_bug_patterns()
        
        # Task setup
        self._setup_coding_task()
    
    def _create_action_map(self) -> Dict[int, str]:
        """Create mapping from action indices to CRUXEval commands"""
        return {
            0: "insert_line",
            1: "delete_line",
            2: "modify_line",
            3: "add_function",
            4: "remove_function",
            5: "fix_syntax",
            6: "fix_logic",
            7: "add_import",
            8: "remove_import",
            9: "add_variable",
            10: "modify_variable",
            11: "add_condition",
            12: "modify_condition",
            13: "add_loop",
            14: "modify_loop",
            15: "add_exception",
            16: "fix_exception",
            17: "run_tests",
            18: "debug_step",
            19: "add_print",
            20: "remove_print",
            21: "format_code",
            22: "optimize_code",
            23: "add_comments",
            24: "remove_comments",
            25: "select_text",
            26: "copy_text",
            27: "paste_text",
            28: "undo_change",
            29: "redo_change"
        }
    
    def _create_code_templates(self) -> Dict[str, str]:
        """Create code templates for different tasks"""
        return {
            "sort_function": '''
def sort_list(lst):
    """Sort a list in ascending order"""
    # TODO: Implement sorting
    return lst

# Test cases
test_cases = [
    ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
]
''',
            "fibonacci": '''
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    # TODO: Implement Fibonacci
    return 0

# Test cases
test_cases = [
    (0, 0),
    (1, 1),
    (2, 1),
    (5, 5),
    (10, 55)
]
''',
            "palindrome": '''
def is_palindrome(s):
    """Check if string is palindrome"""
    # TODO: Implement palindrome check
    return False

# Test cases
test_cases = [
    ("racecar", True),
    ("hello", False),
    ("", True),
    ("a", True),
    ("ab", False)
]
'''
        }
    
    def _create_bug_patterns(self) -> Dict[str, List[str]]:
        """Create common bug patterns"""
        return {
            "syntax_errors": [
                "missing_colon",
                "missing_parentheses",
                "incorrect_indentation",
                "undefined_variable"
            ],
            "logic_errors": [
                "off_by_one",
                "wrong_condition",
                "missing_edge_case",
                "incorrect_algorithm"
            ],
            "runtime_errors": [
                "division_by_zero",
                "index_out_of_bounds",
                "type_error",
                "attribute_error"
            ]
        }
    
    def _setup_coding_task(self):
        """Setup coding task with buggy code"""
        # Select random template and introduce bugs
        template_name = "sort_function"  # Simplified: always use sort function
        self.buggy_code = self.code_templates[template_name]
        
        # Introduce bugs
        self.buggy_code = self._introduce_bugs(self.buggy_code)
        self.current_code = self.buggy_code
        
        # Parse test cases
        self._parse_test_cases()
    
    def _introduce_bugs(self, code: str) -> str:
        """Introduce bugs into code"""
        # Simple bug introduction: remove the actual implementation
        lines = code.split('\n')
        buggy_lines = []
        
        for line in lines:
            if '# TODO: Implement' in line:
                # Replace with buggy implementation
                buggy_lines.append('    return []  # Bug: always returns empty list')
            else:
                buggy_lines.append(line)
        
        return '\n'.join(buggy_lines)
    
    def _parse_test_cases(self):
        """Parse test cases from code"""
        # Extract test cases from code (simplified)
        self.test_cases = [
            ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
            ([], []),
            ([1], [1]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
        ]
        self.failed_tests = []
        self.passed_tests = []
    
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset code state
        self.current_code = self.buggy_code
        self.code_history = [self.buggy_code]
        self.current_line = 0
        self.selected_text = ""
        self.error_messages = []
        
        # Reset test results
        self.failed_tests = []
        self.passed_tests = []
        
        # Generate initial observation
        observation = self._generate_observation()
        
        # Create initial state
        state = EnvironmentState(
            observation=self.get_state_representation(),
            reward=0.0,
            done=False,
            info={
                "current_code": self.current_code,
                "test_cases_count": len(self.test_cases),
                "failed_tests": len(self.failed_tests),
                "passed_tests": len(self.passed_tests),
                "error_messages": self.error_messages
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
            action = 29  # Default to redo_change
        
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
                "current_code": self.current_code,
                "test_cases_count": len(self.test_cases),
                "failed_tests": len(self.failed_tests),
                "passed_tests": len(self.passed_tests),
                "error_messages": self.error_messages,
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
        
        if action_command == "insert_line":
            reward, success = self._action_insert_line()
        elif action_command == "delete_line":
            reward, success = self._action_delete_line()
        elif action_command == "modify_line":
            reward, success = self._action_modify_line()
        elif action_command == "add_function":
            reward, success = self._action_add_function()
        elif action_command == "fix_syntax":
            reward, success = self._action_fix_syntax()
        elif action_command == "fix_logic":
            reward, success = self._action_fix_logic()
        elif action_command == "run_tests":
            reward, success = self._action_run_tests()
        elif action_command == "debug_step":
            reward, success = self._action_debug_step()
        elif action_command == "add_print":
            reward, success = self._action_add_print()
        elif action_command == "undo_change":
            reward, success = self._action_undo_change()
        else:
            # Default action
            reward = 0.0
            success = True
        
        return reward, success
    
    def _action_insert_line(self) -> Tuple[float, bool]:
        """Insert a new line of code"""
        # Simplified: insert a basic line
        lines = self.current_code.split('\n')
        if self.current_line < len(lines):
            lines.insert(self.current_line, '    pass  # New line')
            self.current_code = '\n'.join(lines)
            self._save_code_state()
            return 0.1, True
        return -0.1, False
    
    def _action_delete_line(self) -> Tuple[float, bool]:
        """Delete current line"""
        lines = self.current_code.split('\n')
        if 0 <= self.current_line < len(lines):
            del lines[self.current_line]
            self.current_code = '\n'.join(lines)
            self._save_code_state()
            return 0.1, True
        return -0.1, False
    
    def _action_modify_line(self) -> Tuple[float, bool]:
        """Modify current line"""
        lines = self.current_code.split('\n')
        if 0 <= self.current_line < len(lines):
            # Simple modification: add comment
            lines[self.current_line] += '  # Modified'
            self.current_code = '\n'.join(lines)
            self._save_code_state()
            return 0.2, True
        return -0.1, False
    
    def _action_add_function(self) -> Tuple[float, bool]:
        """Add a new function"""
        new_function = '''
def helper_function():
    """Helper function"""
    return True
'''
        self.current_code += new_function
        self._save_code_state()
        return 0.3, True
    
    def _action_fix_syntax(self) -> Tuple[float, bool]:
        """Fix syntax errors"""
        # Simple syntax fix: ensure proper indentation
        lines = self.current_code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                fixed_lines.append(line)
            elif line.strip() and not line.startswith('    '):
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)
        
        self.current_code = '\n'.join(fixed_lines)
        self._save_code_state()
        return 0.5, True
    
    def _action_fix_logic(self) -> Tuple[float, bool]:
        """Fix logic errors"""
        # Simple logic fix: replace buggy return with proper implementation
        if 'return []  # Bug: always returns empty list' in self.current_code:
            self.current_code = self.current_code.replace(
                'return []  # Bug: always returns empty list',
                'return sorted(lst)  # Fixed: proper sorting'
            )
            self._save_code_state()
            return 1.0, True
        return 0.0, True
    
    def _action_run_tests(self) -> Tuple[float, bool]:
        """Run test cases"""
        # Simulate running tests
        try:
            # Create temporary file with test code
            test_code = self.current_code + '''
# Run tests
for i, (input_val, expected) in enumerate(test_cases):
    try:
        result = sort_list(input_val)
        if result == expected:
            print(f"Test {i+1}: PASSED")
        else:
            print(f"Test {i+1}: FAILED - Expected {expected}, got {result}")
    except Exception as e:
        print(f"Test {i+1}: ERROR - {e}")
'''
            
            # Execute test code (simplified simulation)
            self.failed_tests = []
            self.passed_tests = []
            
            for i, (input_val, expected) in enumerate(self.test_cases):
                try:
                    # Simulate function execution
                    if 'return sorted(lst)' in self.current_code:
                        result = sorted(input_val)
                        if result == expected:
                            self.passed_tests.append(i)
                        else:
                            self.failed_tests.append(i)
                    else:
                        self.failed_tests.append(i)
                except Exception as e:
                    self.failed_tests.append(i)
                    self.error_messages.append(f"Test {i+1}: {str(e)}")
            
            # Calculate reward based on test results
            total_tests = len(self.test_cases)
            passed_count = len(self.passed_tests)
            reward = passed_count / total_tests
            
            return reward, True
            
        except Exception as e:
            self.error_messages.append(f"Test execution error: {str(e)}")
            return -0.5, False
    
    def _action_debug_step(self) -> Tuple[float, bool]:
        """Debug step through code"""
        self.current_line = (self.current_line + 1) % len(self.current_code.split('\n'))
        return 0.1, True
    
    def _action_add_print(self) -> Tuple[float, bool]:
        """Add print statement for debugging"""
        lines = self.current_code.split('\n')
        if self.current_line < len(lines):
            lines.insert(self.current_line, '    print("Debug:", lst)')
            self.current_code = '\n'.join(lines)
            self._save_code_state()
            return 0.2, True
        return -0.1, False
    
    def _action_undo_change(self) -> Tuple[float, bool]:
        """Undo last change"""
        if len(self.code_history) > 1:
            self.code_history.pop()
            self.current_code = self.code_history[-1]
            return 0.1, True
        return -0.1, False
    
    def _save_code_state(self):
        """Save current code state to history"""
        self.code_history.append(self.current_code)
        if len(self.code_history) > 10:  # Limit history size
            self.code_history.pop(0)
    
    def _is_task_complete(self) -> bool:
        """Check if coding task is complete"""
        return len(self.passed_tests) == len(self.test_cases) and len(self.failed_tests) == 0
    
    def _get_task_progress(self) -> float:
        """Get task completion progress"""
        if not self.test_cases:
            return 0.0
        
        total_tests = len(self.test_cases)
        passed_tests = len(self.passed_tests)
        return passed_tests / total_tests
    
    def _generate_observation(self) -> str:
        """Generate text observation of current state"""
        obs_parts = []
        
        # Code status
        obs_parts.append(f"Code lines: {len(self.current_code.split('\\n'))}")
        obs_parts.append(f"Current line: {self.current_line}")
        
        # Test results
        obs_parts.append(f"Tests: {len(self.passed_tests)}/{len(self.test_cases)} passed")
        
        # Error messages
        if self.error_messages:
            obs_parts.append(f"Errors: {len(self.error_messages)}")
        
        # Code snippet (first few lines)
        code_lines = self.current_code.split('\n')[:5]
        obs_parts.append("Code:")
        obs_parts.extend(code_lines)
        
        return '\n'.join(obs_parts)
    
    def get_state_representation(self) -> torch.Tensor:
        """Get current state as tensor representation"""
        # Create feature vector from environment state
        features = []
        
        # Code features
        code_lines = self.current_code.split('\n')
        features.append(len(code_lines) / 100.0)  # Normalized line count
        features.append(self.current_line / max(len(code_lines), 1))  # Current line position
        
        # Test features
        total_tests = len(self.test_cases)
        if total_tests > 0:
            features.append(len(self.passed_tests) / total_tests)  # Pass rate
            features.append(len(self.failed_tests) / total_tests)  # Fail rate
        else:
            features.extend([0.0, 0.0])
        
        # Error features
        features.append(len(self.error_messages) / 10.0)  # Normalized error count
        
        # Code quality features (simplified)
        features.append(1.0 if 'def ' in self.current_code else 0.0)  # Has function
        features.append(1.0 if 'return ' in self.current_code else 0.0)  # Has return
        features.append(1.0 if 'sorted(' in self.current_code else 0.0)  # Uses sorted
        
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
    
    def get_coding_info(self) -> Dict[str, Any]:
        """Get coding-specific information"""
        return {
            "language": self.config.language,
            "current_code": self.current_code,
            "buggy_code": self.buggy_code,
            "test_cases": self.test_cases,
            "failed_tests": self.failed_tests,
            "passed_tests": self.passed_tests,
            "error_messages": self.error_messages,
            "code_history": self.code_history
        }
    
    def render(self, mode: str = "text") -> str:
        """Render environment state as text"""
        if mode == "text":
            return self._generate_observation()
        else:
            return super().render(mode)
