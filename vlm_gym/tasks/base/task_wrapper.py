#!/usr/bin/env python3
"""Base task wrapper for all VLM tasks"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import logging
import hashlib

logger = logging.getLogger(__name__)


class BaseTaskWrapper(ABC):
    """Base class for all task wrappers"""
    
    # Task type identifier
    task_type = "base"
    
    def __init__(self, task_id: str, task_data: dict, **kwargs):
        """Initialize task wrapper
        
        Args:
            task_id: Unique task identifier
            task_data: Task data including question, answer, image_path
            **kwargs: Additional configuration
        """
        self.task_id = task_id
        self.task_data = task_data
        
        # Core task data
        self.current_image = task_data.get('image_path')
        self.current_question = task_data.get('question', '')
        self.answer = task_data.get('answer', '')
        
        # Configuration
        self.max_steps = kwargs.get('max_steps', 10)
        self.current_step = 0
        
        # Reflection support
        self.enable_reflection = kwargs.get('enable_reflection', False)
        self.max_attempts = kwargs.get('max_attempts', 3)
        self.current_attempt = 0
        self.attempt_history = []
        self.conversation_history = []
        
        # Output format
        self.use_structured_output = kwargs.get('use_structured_output', True)
        
        # Debug
        self.debug = kwargs.get('debug', False)
        
        # Tool support
        self.enable_deepeyes = kwargs.get('enable_deepeyes', False)
        self.enable_grounding_dino = kwargs.get('enable_grounding_dino', False)
        self.enable_chartmoe = kwargs.get('enable_chartmoe', False)
        self.enable_sam2 = kwargs.get('enable_sam2', False)
        self.enable_sympy_geometry = kwargs.get('enable_sympy_geometry', False)
        self.enable_multimath_server = kwargs.get('enable_multimath_server', False)
        
        # Tool histories
        self.deepeyes_history = []
        self.grounding_dino_history = []
        self.chartmoe_history = []
        self.sam2_history = []
        self.sympy_geometry_history = []
        self.multimath_history = []
        
        # Action set (if using action system)
        self.action_set = kwargs.get('action_set')
        
        # Components (will be initialized by subclasses)
        self.components = self._initialize_components(kwargs)
        
        # Additional initialization
        self._post_init(kwargs)
    
    def _initialize_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize task components
        
        Returns:
            Dictionary of component instances
        """
        components = self.get_components(config)
        
        # Validate required components
        required = ['evaluator', 'extractor', 'classifier']
        for comp in required:
            if comp not in components:
                raise ValueError(f"Missing required component: {comp}")
        
        return components
    
    @abstractmethod
    def get_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get task-specific components
        
        Should return a dictionary with:
        - evaluator: Answer evaluator
        - extractor: Answer extractor
        - classifier: Question classifier
        - feedback: Feedback generator (optional)
        - formatter: Output formatter (optional)
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of component instances
        """
        pass
    
    def _post_init(self, config: Dict[str, Any]):
        """Post-initialization hook for subclasses"""
        pass
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """Setup method - required by environment"""
        message = f"{self.task_type} task initialized. Question: {self.current_question}"
        info = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "has_actions": self.action_set is not None,
            "question_type": self.classify_question(),
            "reflection_enabled": self.enable_reflection,
            "max_attempts": self.max_attempts if self.enable_reflection else 1,
            "use_structured_output": self.use_structured_output,
            "deepeyes_enabled": self.enable_deepeyes,
            "grounding_dino_enabled": self.enable_grounding_dino,
            "chartmoe_enabled": self.enable_chartmoe,
            "sam2_enabled": self.enable_sam2,
            "sympy_geometry_enabled": self.enable_sympy_geometry,
            "multimath_enabled": self.enable_multimath_server
        }
        return message, info
    
    def teardown(self):
        """Teardown method"""
        pass
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset task state"""
        self.current_step = 0
        self.current_attempt = 0
        self.attempt_history = []
        self.conversation_history = []
        self.deepeyes_history = []
        self.grounding_dino_history = []
        self.chartmoe_history = []
        self.sam2_history = []
        self.sympy_geometry_history = []
        self.multimath_history = []
        
        obs = self.get_initial_observation()
        info = self.get_initial_info()
        
        return obs, info
    
    def get_initial_observation(self) -> Dict[str, Any]:
        """Get initial observation"""
        obs = {
            "image_path": self.current_image,
            "question": self.current_question,
            "task_description": self.get_task_description(),
            "conversation_history": [],
            "output_format_instruction": self.get_output_format_instruction() if self.use_structured_output else "",
            "deepeyes_enabled": self.enable_deepeyes,
            "grounding_dino_enabled": self.enable_grounding_dino,
            "chartmoe_enabled": self.enable_chartmoe,
            "sam2_enabled": self.enable_sam2,
            "sympy_geometry_enabled": self.enable_sympy_geometry,
            "multimath_enabled": self.enable_multimath_server,
            "is_visual_question": True,
            "task_type": self.task_type
        }
        return obs
    
    def get_initial_info(self) -> Dict[str, Any]:
        """Get initial info"""
        info = {
            "task_id": self.task_id,
            "dataset": self.task_type,
            "ground_truth": self.answer,
            "attempt": 1,
            "max_attempts": self.max_attempts if self.enable_reflection else 1,
            "question_type": self.classify_question()
        }
        return info
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation for the task"""
        obs = {
            "image_path": self.current_image,
            "question": self.current_question,
            "task_id": self.task_id,
            "attempt": self.current_attempt + 1,
            "conversation_history": self.conversation_history,
            "output_format_instruction": self.get_output_format_instruction() if self.use_structured_output else "",
            "use_structured_output": self.use_structured_output,
            "deepeyes_enabled": self.enable_deepeyes,
            "deepeyes_history": self.deepeyes_history,
            "grounding_dino_enabled": self.enable_grounding_dino,
            "grounding_dino_history": self.grounding_dino_history,
            "chartmoe_enabled": self.enable_chartmoe,
            "chartmoe_history": self.chartmoe_history,
            "sam2_enabled": self.enable_sam2,
            "sam2_history": self.sam2_history,
            "sympy_geometry_enabled": self.enable_sympy_geometry,
            "sympy_geometry_history": self.sympy_geometry_history,
            "multimath_enabled": self.enable_multimath_server,
            "multimath_history": self.multimath_history,
            "is_visual_question": True,
            "task_type": self.task_type
        }
        
        # Add reflection-specific fields if this is a retry
        if self.enable_reflection and self.current_attempt > 0 and self.attempt_history:
            obs.update(self.get_reflection_observation())
        
        return obs
    
    def get_reflection_observation(self) -> Dict[str, Any]:
        """Get reflection-specific observation fields"""
        last_attempt = self.attempt_history[-1]
        obs = {
            "feedback": last_attempt.get("feedback", ""),
            "previous_answer": last_attempt.get("answer", ""),
            "attempts_remaining": self.max_attempts - self.current_attempt,
            "max_attempts": self.max_attempts,
            "previous_attempt_failed": True
        }
        
        # Add hint on last attempt
        if self.current_attempt == self.max_attempts - 1:
            obs["hint"] = self.generate_hint()
        
        # Check if should force tool usage
        force_tool = self.should_force_tool()
        if force_tool:
            obs["must_use_tool"] = True
            obs["tool_to_use"] = force_tool
            obs["reflection_format_reminder"] = self.get_tool_force_reminder(force_tool)
        else:
            obs["reflection_format_reminder"] = "Remember to use the <think>...</think> and <answer>...</answer> format in your response."
        
        return obs
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Process action"""
        self.current_step += 1
        
        if self.debug:
            logger.debug(f"[{self.task_type}] Step {self.current_step}, Attempt {self.current_attempt + 1}")
        
        # Record tool usage
        self.record_tool_usage(action)
        
        # Extract answer
        answer = self.extract_answer(action)
        
        # Evaluate answer
        is_correct = self.evaluate_answer(answer)
        
        # Update attempt history
        attempt_data = self.create_attempt_data(action, answer, is_correct)
        
        # Increment attempt counter
        self.current_attempt += 1
        self.attempt_history.append(attempt_data)
        
        # Check if should continue (reflection logic)
        should_continue = (not is_correct and 
                          self.enable_reflection and 
                          self.current_attempt < self.max_attempts)
        
        if should_continue:
            return self.handle_reflection(answer, attempt_data)
        else:
            return self.handle_completion(answer, is_correct)
    
    def handle_reflection(self, answer: str, attempt_data: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handle reflection for incorrect answer"""
        # Generate feedback
        feedback = self.generate_feedback(answer)
        attempt_data["feedback"] = feedback
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": attempt_data.get("action", "")
        })
        self.conversation_history.append({
            "role": "system",
            "content": feedback
        })
        
        # Prepare observation for next attempt
        obs = self.get_observation()
        
        done = False
        reward = 0.0
        truncated = False
        
        info = {
            "prediction": answer,
            "ground_truth": self.answer,
            "correct": False,
            "attempt": self.current_attempt,
            "attempt_history": self.attempt_history,
            "done_reason": "continuing",
            "question_type": self.classify_question(),
            "message": f"Incorrect answer. Attempt {self.current_attempt}/{self.max_attempts}. {feedback[:100]}..."
        }
        
        return obs, reward, done, truncated, info
    
    def handle_completion(self, answer: str, is_correct: bool) -> Tuple[Dict, float, bool, bool, Dict]:
        """Handle task completion"""
        done = True
        reward = 1.0 if is_correct else 0.0
        truncated = (not is_correct and 
                    self.enable_reflection and 
                    self.current_attempt >= self.max_attempts)
        
        obs = {
            "message": f"Task completed. Final answer: {answer}. Correct: {is_correct}",
            "final_answer": answer,
            "attempts_used": self.current_attempt,
            "success": is_correct
        }
        
        info = {
            "prediction": answer,
            "ground_truth": self.answer,
            "correct": is_correct,
            "attempt": self.current_attempt,
            "attempt_history": self.attempt_history if self.enable_reflection else None,
            "done_reason": "correct" if is_correct else ("max_attempts" if truncated else "single_attempt"),
            "question_type": self.classify_question(),
            "message": f"Task completed. Attempts: {self.current_attempt}/{self.max_attempts}. Correct: {is_correct}"
        }
        
        # Check max steps truncation
        if not done and self.current_step >= self.max_steps:
            truncated = True
            done = True
            obs["message"] = "Truncated by max steps limit"
            info["done_reason"] = "max_steps"
        
        return obs, reward, done, truncated, info
    
    def validate(self, chat_history, observation, full_history=None) -> Tuple[float, bool, str, Dict[str, Any]]:
        """Validate method required by environment"""
        # Extract answer from observation
        answer = self.extract_answer_from_observation(observation)
        
        # Evaluate
        is_correct = self.evaluate_answer(answer)
        
        # Determine if done
        if is_correct:
            done = True
            reward = 1.0
        elif not self.enable_reflection:
            done = True
            reward = 0.0
        elif self.current_attempt >= self.max_attempts:
            done = True
            reward = 0.0
        else:
            done = False
            reward = 0.0
        
        # Generate message
        message_parts = []
        if self.enable_reflection and len(self.attempt_history) > 0:
            message_parts.append(f"Attempt {self.current_attempt} of {self.max_attempts}.")
        else:
            message_parts.append("Task evaluation.")
        
        message_parts.append(f"Answer: {answer}")
        message_parts.append(f"Correct: {is_correct}")
        
        if not done and self.enable_reflection:
            message_parts.append(f"Attempts remaining: {self.max_attempts - self.current_attempt}")
        
        message = " ".join(message_parts)
        
        info = {
            "prediction": answer,
            "ground_truth": self.answer,
            "correct": is_correct,
            "current_attempt": self.current_attempt,
            "max_attempts": self.max_attempts,
            "should_continue": not done and self.enable_reflection,
            "attempt_history": self.attempt_history if self.enable_reflection else None,
            "question_type": self.classify_question()
        }
        
        return reward, done, message, info
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Get task description"""
        pass
    
    @abstractmethod
    def get_output_format_instruction(self) -> str:
        """Get output format instruction"""
        pass
    
    @abstractmethod
    def classify_question(self) -> str:
        """Classify question type"""
        pass
    
    @abstractmethod
    def extract_answer(self, action: str) -> str:
        """Extract answer from action"""
        pass
    
    @abstractmethod
    def evaluate_answer(self, answer: str) -> bool:
        """Evaluate if answer is correct"""
        pass
    
    @abstractmethod
    def generate_feedback(self, answer: str) -> str:
        """Generate feedback for incorrect answer"""
        pass
    
    @abstractmethod
    def generate_hint(self) -> str:
        """Generate hint for final attempt"""
        pass
    
    # Helper methods
    
    def record_tool_usage(self, action: str):
        """Record tool usage from action"""
        if isinstance(action, str):
            if '<tool_call>' in action:
                if '"tool": "chartmoe"' in action:
                    self.chartmoe_history.append({
                        "step": self.current_step,
                        "action": action
                    })
                elif '"tool": "grounding_dino"' in action:
                    self.grounding_dino_history.append({
                        "step": self.current_step,
                        "action": action
                    })
                elif '"name": "image_zoom_in_tool"' in action:
                    self.deepeyes_history.append({
                        "step": self.current_step,
                        "action": action
                    })
                elif '"tool": "sam2"' in action:
                    self.sam2_history.append({
                        "step": self.current_step,
                        "action": action
                    })
                elif '"tool": "sympy_geometry"' in action:
                    self.sympy_geometry_history.append({
                        "step": self.current_step,
                        "action": action
                    })
                elif '"tool": "multimath_server"' in action:
                    self.multimath_history.append({
                        "step": self.current_step,
                        "action": action
                    })
    
    def create_attempt_data(self, action: str, answer: str, is_correct: bool) -> Dict[str, Any]:
        """Create attempt data record"""
        return {
            "attempt": self.current_attempt + 1,
            "answer": answer,
            "correct": is_correct,
            "action": action,
            "question_type": self.classify_question(),
            "deepeyes_used": len([h for h in self.deepeyes_history if h["step"] == self.current_step]) > 0,
            "grounding_dino_used": len([h for h in self.grounding_dino_history if h["step"] == self.current_step]) > 0,
            "chartmoe_used": len([h for h in self.chartmoe_history if h["step"] == self.current_step]) > 0,
            "sam2_used": len([h for h in self.sam2_history if h["step"] == self.current_step]) > 0,
            "sympy_used": len([h for h in self.sympy_geometry_history if h["step"] == self.current_step]) > 0,
            "multimath_used": len([h for h in self.multimath_history if h["step"] == self.current_step]) > 0
        }
    
    def should_force_tool(self) -> Optional[str]:
        """Check if should force tool usage in reflection"""
        # Subclasses can override this
        return None
    
    def get_tool_force_reminder(self, tool_name: str) -> str:
        """Get reminder for forced tool usage"""
        # Subclasses can override this
        return f"YOU MUST use {tool_name} tool first!"

    
    def get_expected_format_tags(self) -> Dict[str, List[str]]:
        """Get expected format tags for this task type
        
        Returns:
            Dictionary with 'reasoning' and 'answer' keys, 
            each containing list of acceptable tags
        """
        # 默认支持多种格式
        return {
            "reasoning": ["<think>", "<reasoning>"],  # 支持两种格式
            "answer": ["<answer>"]
        }
    
    
    def extract_answer_from_observation(self, observation: Any) -> str:
        """Extract answer from observation (for validate method)"""
        if isinstance(observation, dict):
            answer = observation.get('final_answer',
                                   observation.get('answer',
                                                observation.get('content', '')))
        else:
            answer = str(observation)
        
        return answer