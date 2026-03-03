#!/usr/bin/env python3
"""Tool-aware Agent that extends BaseAgent with tool capabilities"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import json
import hashlib
from enum import Enum

# Import from parent module
from ...base import BaseAgent, AgentConfig

# Import tool system components
from ..tools.registry import ToolRegistry
from ..tools.base import ToolResult
from ..strategies.workflow_manager import WorkflowManager
from ..strategies.tool_selection import ToolSelector
from ..strategies.reflection import ReflectionStrategy
from ..classifiers.question_classifier import QuestionClassifier
from ..classifiers.task_detector import TaskDetector
from ..utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class ToolCallState(Enum):
    """State of tool calls"""
    IDLE = "idle"
    CALLING = "calling"
    WAITING = "waiting_for_response"
    PROCESSING = "processing_response"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolAwareAgentConfig(AgentConfig):
    """Extended configuration for tool-aware agents"""
    # Tool system configuration
    enable_tools: bool = True
    max_tool_calls: int = 3
    tool_selection_strategy: str = "auto"  # auto, manual, workflow
    tool_response_mode: str = "auto"  # auto, direct, formatted
    
    # Enabled tools list
    enabled_tools: List[str] = field(default_factory=lambda: [
        "chartmoe", "multimath_server", "sam2", "sympy_geometry",
        "grounding_dino", "easyocr", "deepeyes", "diagram_formalizer"
    ])
    
    # Workflow configuration
    enable_workflows: bool = True
    preferred_workflows: List[str] = field(default_factory=lambda: [
        "geometry_workflow", "vision_workflow", "chart_workflow"
    ])
    
    # Reflection configuration
    enable_reflection: bool = True
    max_reflection_attempts: int = 2
    
    # Debug configuration
    debug: bool = False
    log_tool_calls: bool = True


class ToolAwareAgent(BaseAgent):
    """Agent with tool awareness and coordination capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tool-aware agent
        
        Args:
            config: Configuration dictionary
        """
        # Parse configuration
        if isinstance(config, dict):
            if 'agent' in config:
                base_config = config['agent']
            else:
                base_config = config
            
            # Create ToolAwareAgentConfig
            self.config = ToolAwareAgentConfig(**base_config)
        else:
            self.config = config
            
        # Initialize parent
        super().__init__(self.config.__dict__)
        
        # Tool system components
        self.tool_registry = ToolRegistry()
        self.workflow_manager = WorkflowManager()
        self.tool_selector = ToolSelector()
        self.reflection_strategy = ReflectionStrategy()
        
        # Classifiers
        self.question_classifier = QuestionClassifier()
        self.task_detector = TaskDetector()
        
        # Response formatter
        self.response_formatter = ResponseFormatter()
        
        # Tool call tracking
        self.current_tool_calls = 0
        self.tool_call_state = ToolCallState.IDLE
        self.tool_call_history = {}  # question_hash -> set of tools used
        self.tool_context = {}  # Store tool outputs for coordination
        self.current_workflow_context = None
        
        # Performance tracking
        self.tool_performance = {
            tool: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0
            }
            for tool in self.config.enabled_tools
        }
        
        # Initialize tools (will be done by environment)
        self.tool_instances = {}
        
        logger.info(f"Initialized ToolAwareAgent with {len(self.config.enabled_tools)} tools enabled")
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent with tool support
        
        Args:
            observation: Dictionary containing task information
            
        Returns:
            action: The agent's response/answer or tool call
            extra_info: Additional information about the action
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Check if this is a new task
        if self._is_new_task(observation):
            self._reset_task_state()
        
        # Detect task type and requirements
        task_type = self.task_detector.detect(observation)
        question_type = self.question_classifier.classify(observation.get("question", ""))
        
        # Add classifications to observation
        observation["task_type"] = task_type
        observation["question_type"] = question_type
        
        # Check if we're handling tool feedback
        if observation.get("requires_response") and "tool_feedback" in observation:
            return self._handle_tool_feedback(observation)
        
        # Check for reflection/retry
        if self._is_reflection_attempt(observation):
            return self._handle_reflection(observation)
        
        # Check if tools are disabled or unavailable
        if not self.config.enable_tools or not self._has_available_tools():
            return self._generate_direct_answer(observation)
        
        # Check if we've reached tool call limit
        if self.current_tool_calls >= self.config.max_tool_calls:
            return self._generate_forced_final_answer(observation)
        
        # Try workflow-based approach first
        if self.config.enable_workflows:
            workflow = self.workflow_manager.select_workflow(observation)
            if workflow:
                return self._execute_workflow(workflow, observation)
        
        # Fall back to individual tool selection
        tool = self.tool_selector.select_tool(
            observation, 
            self.tool_registry.get_available_tools(),
            self.tool_call_history
        )
        
        if tool:
            return self._call_tool(tool, observation)
        
        # No tool selected, generate direct answer
        return self._generate_direct_answer(observation)
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """Check if this is a new task"""
        current_task_id = observation.get("task_id")
        
        if hasattr(self, "_last_task_id"):
            if self._last_task_id != current_task_id:
                self._last_task_id = current_task_id
                return True
        else:
            self._last_task_id = current_task_id
            return True
            
        # Check question change
        current_question = observation.get("question", "")
        if hasattr(self, "_last_question"):
            if self._last_question != current_question:
                self._last_question = current_question
                return True
        else:
            self._last_question = current_question
            
        return False
    
    def _reset_task_state(self):
        """Reset state for new task"""
        self.current_tool_calls = 0
        self.tool_call_state = ToolCallState.IDLE
        self.tool_call_history = {}
        self.tool_context = {}
        self.current_workflow_context = None
        
        if self.config.debug:
            logger.debug("Task state reset")
    
    def _is_reflection_attempt(self, observation: Dict[str, Any]) -> bool:
        """Check if this is a reflection/retry attempt"""
        return (
            observation.get("previous_attempt_failed", False) or
            observation.get("feedback") is not None or
            observation.get("attempt", 1) > 1
        )
    
    def _has_available_tools(self) -> bool:
        """Check if any tools are available"""
        return len(self.tool_registry.get_available_tools()) > 0
    
    def _execute_workflow(self, workflow, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Execute a workflow"""
        self.current_workflow_context = workflow.create_context(observation)
        
        # Get next tool call from workflow
        tool_call = workflow.get_next_tool_call(self.current_workflow_context)
        
        if tool_call:
            self.tool_call_state = ToolCallState.CALLING
            self.current_tool_calls += 1
            
            extra_info = {
                "action_type": "workflow_tool_call",
                "workflow": workflow.name,
                "tool": tool_call.get("tool"),
                "tool_call_count": self.current_tool_calls
            }
            
            return json.dumps(tool_call), extra_info
        
        # Workflow complete, generate answer
        return workflow.generate_answer(self.current_workflow_context)
    
    def _call_tool(self, tool, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Call a single tool"""
        # Build tool prompt
        tool_call = tool.build_prompt(observation)
        
        # Track the call
        question_hash = self._hash_question(observation.get("question", ""))
        if question_hash not in self.tool_call_history:
            self.tool_call_history[question_hash] = set()
        self.tool_call_history[question_hash].add(tool.name)
        
        self.tool_call_state = ToolCallState.CALLING
        self.current_tool_calls += 1
        
        # Update performance tracking
        self.tool_performance[tool.name]["attempts"] += 1
        
        extra_info = {
            "action_type": "tool_call",
            "tool": tool.name,
            "tool_call_count": self.current_tool_calls,
            "confidence": tool.last_confidence if hasattr(tool, 'last_confidence') else 0
        }
        
        if self.config.log_tool_calls:
            logger.info(f"Calling tool: {tool.name} (attempt {self.current_tool_calls})")
        
        return tool_call, extra_info
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle feedback from tool execution"""
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        # Store tool result in context
        self.tool_context[tool_name] = tool_feedback
        
        # Update performance tracking
        if tool_feedback.get("success"):
            self.tool_performance[tool_name]["successes"] += 1
        else:
            self.tool_performance[tool_name]["failures"] += 1
        
        # If in workflow, continue workflow
        if self.current_workflow_context:
            workflow = self.workflow_manager.get_active_workflow()
            if workflow:
                workflow.process_tool_result(self.current_workflow_context, tool_feedback)
                return self._execute_workflow(workflow, observation)
        
        # Get the tool handler
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            # Process the result
            result = tool.process_result(tool_feedback, observation)
            
            # Format for answer
            formatted_prompt = tool.format_for_answer(result, observation)
            
            # Generate final answer using base model
            enhanced_observation = observation.copy()
            enhanced_observation["output_format_instruction"] = formatted_prompt
            
            response, base_info = super().act(enhanced_observation)
            
            # Ensure proper formatting
            response = self.response_formatter.ensure_answer_format(response, observation)
            
            extra_info = base_info.copy()
            extra_info.update({
                "action_type": "tool_response",
                "tool_used": tool_name,
                "tool_success": result.success
            })
            
            return response, extra_info
        
        # Fallback to direct answer if tool not found
        return self._generate_direct_answer(observation)
    
    def _handle_reflection(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Handle reflection/retry attempt"""
        strategy = self.reflection_strategy.get_strategy(observation)
        
        if strategy == "retry_tool":
            # Try a different tool or approach
            alternative_tool = self.tool_selector.select_alternative_tool(
                observation,
                self.tool_registry.get_available_tools(),
                self.tool_call_history
            )
            
            if alternative_tool:
                return self._call_tool(alternative_tool, observation)
        
        elif strategy == "refine_answer":
            # Refine the previous answer
            enhanced_observation = observation.copy()
            enhanced_observation["instruction"] = "Refine your previous answer based on the feedback"
            return super().act(enhanced_observation)
        
        # Default: generate new answer
        return self._generate_direct_answer(observation)
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate answer without tools"""
        # Use parent class's act method
        response, base_info = super().act(observation)
        
        # Ensure proper formatting
        response = self.response_formatter.ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "direct_answer",
            "tools_used": False
        })
        
        return response, extra_info
    
    def _generate_forced_final_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate final answer when tool limit reached"""
        enhanced_observation = observation.copy()
        enhanced_observation["output_format_instruction"] = (
            "You have reached the maximum number of tool calls. "
            "Based on all available information, provide your final answer. "
            "You MUST output your answer inside <answer> tags."
        )
        
        response, base_info = super().act(enhanced_observation)
        
        # Ensure proper formatting
        response = self.response_formatter.ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "forced_final_answer",
            "reason": "tool_limit_reached",
            "tool_calls_made": self.current_tool_calls
        })
        
        return response, extra_info
    
    def _hash_question(self, question: str) -> str:
        """Generate hash for question tracking"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def register_tool_instance(self, name: str, instance: Any):
        """Register a tool instance (called by environment)"""
        self.tool_instances[name] = instance
        logger.info(f"Registered tool instance: {name}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = super().get_state()
        stats.update({
            "tool_performance": self.tool_performance,
            "total_tool_calls": sum(t["attempts"] for t in self.tool_performance.values()),
            "successful_tool_calls": sum(t["successes"] for t in self.tool_performance.values()),
            "failed_tool_calls": sum(t["failures"] for t in self.tool_performance.values())
        })
        return stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self._reset_task_state()
        logger.info("ToolAwareAgent state reset")