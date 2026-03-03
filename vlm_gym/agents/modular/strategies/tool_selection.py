#!/usr/bin/env python3
"""Tool selection strategy module - Capability-based approach"""

from typing import Dict, Any, List, Optional, Set
import re
import logging
import hashlib

logger = logging.getLogger(__name__)


class ToolSelector:
    """Strategy for selecting appropriate tools based on task capabilities"""
    
    def __init__(self):
        self.logger = logger
        
        # Task to capabilities mapping (parallel capabilities)
        self.task_capabilities = {
            "chartqa": ["chart_analysis", "table_extraction", "data_visualization", "text_recognition"],
            "geometry": ["geometric_reasoning", "formal_proof", "calculation", "diagram_understanding"],
            "medical_vqa": ["medical_segmentation", "organ_detection", "abnormality_detection"],
            "text_extraction": ["text_recognition", "formula_extraction"],
            "object_detection": ["object_localization", "counting", "spatial_analysis"],
            "general_vqa": ["visual_understanding", "reasoning"]
        }
        
        # Capability to tools mapping
        self.capability_tools = {
            # Chart capabilities
            "chart_analysis": ["chartmoe"],
            "table_extraction": ["chartmoe", "easyocr"],
            "data_visualization": ["chartmoe"],
            "text_recognition": ["easyocr", "chartmoe"],
            
            # Geometry capabilities
            "geometric_reasoning": ["sympy_geometry", "multimath_server"],
            "formal_proof": ["diagram_formalizer", "multimath_server"],
            "calculation": ["multimath_server", "sympy_geometry"],
            "diagram_understanding": ["diagram_formalizer", "deepeyes"],
            
            # Medical capabilities
            "medical_segmentation": ["sam2"],
            "organ_detection": ["grounding_dino", "sam2"],
            "abnormality_detection": ["sam2", "grounding_dino"],
            
            # Vision capabilities
            "text_recognition": ["easyocr"],
            "formula_extraction": ["easyocr", "diagram_formalizer"],
            "object_localization": ["grounding_dino"],
            "counting": ["grounding_dino", "chartmoe"],
            "spatial_analysis": ["grounding_dino", "sam2"],
            "visual_understanding": ["deepeyes"],
            "reasoning": []  # Base VLM capability
        }
    
    def select_tool(
        self, 
        observation: Dict[str, Any], 
        available_tools: List[Any],
        tool_history: Dict[str, Set[str]]
    ) -> Optional[Any]:
        """Select the most appropriate tool based on capabilities needed
        
        Args:
            observation: Current observation/task
            available_tools: List of available tool instances
            tool_history: History of tools used for each question
            
        Returns:
            Selected tool instance or None
        """
        # Get question hash for history tracking
        question = observation.get("question", "")
        current_attempt = observation.get("attempt", 1)
        question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
        used_tools = tool_history.get(question_key, set())
        
        # Check for forced tool usage
        forced_tool = self._check_forced_usage(observation, available_tools, used_tools)
        if forced_tool:
            return forced_tool
        
        # Get task type from observation
        task_type = observation.get("task_type", "general_vqa")
        
        # Analyze what capabilities are needed
        needed_capabilities = self._analyze_needed_capabilities(observation, task_type)
        
        if not needed_capabilities:
            self.logger.debug("No specific capabilities identified")
            return None
        
        # Get candidate tools that provide needed capabilities
        candidate_tools = self._get_capable_tools(needed_capabilities, available_tools, used_tools)
        
        if not candidate_tools:
            self.logger.debug("No suitable tools available")
            return None
        
        # Let each tool evaluate its fitness for this specific task
        tool_scores = []
        for tool in candidate_tools:
            can_handle, confidence = tool.can_handle(observation)
            if can_handle:
                tool_scores.append((tool, confidence))
        
        if not tool_scores:
            self.logger.debug("No tools can handle this task")
            return None
        
        # Select tool with highest confidence
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        selected_tool, confidence = tool_scores[0]
        
        self.logger.info(f"Selected {selected_tool.name} with confidence {confidence:.2f}")
        self.logger.debug(f"Capabilities needed: {needed_capabilities}")
        self.logger.debug(f"All scores: {[(t.name, c) for t, c in tool_scores]}")
        
        return selected_tool
    
    def select_alternative_tool(
        self,
        observation: Dict[str, Any],
        available_tools: List[Any],
        tool_history: Dict[str, Set[str]]
    ) -> Optional[Any]:
        """Select an alternative tool for reflection/retry
        
        Args:
            observation: Current observation with feedback
            available_tools: List of available tool instances
            tool_history: History of tools used
            
        Returns:
            Alternative tool instance or None
        """
        feedback = observation.get("feedback", "").lower()
        task_type = observation.get("task_type", "general_vqa")
        
        # Get used tools for current attempt
        question = observation.get("question", "")
        current_attempt = observation.get("attempt", 1)
        question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
        used_tools = tool_history.get(question_key, set())
        
        # Analyze what went wrong and what alternative capabilities might help
        alternative_capabilities = self._analyze_alternative_capabilities(
            observation, feedback, task_type, used_tools
        )
        
        if not alternative_capabilities:
            # No specific alternatives, try any unused capable tool
            return self._get_any_unused_capable_tool(observation, available_tools, used_tools)
        
        # Get tools that provide alternative capabilities
        candidate_tools = self._get_capable_tools(alternative_capabilities, available_tools, used_tools)
        
        if not candidate_tools:
            return None
        
        # Evaluate each candidate
        tool_scores = []
        for tool in candidate_tools:
            can_handle, confidence = tool.can_handle(observation)
            if can_handle:
                # Boost confidence for tools specifically good at addressing the feedback
                if self._is_good_for_feedback(tool.name, feedback):
                    confidence *= 1.5
                tool_scores.append((tool, confidence))
        
        if tool_scores:
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            selected = tool_scores[0][0]
            self.logger.info(f"Selected alternative tool: {selected.name}")
            return selected
        
        return None
    
    def _check_forced_usage(
        self, 
        observation: Dict[str, Any], 
        available_tools: List[Any],
        used_tools: Set[str]
    ) -> Optional[Any]:
        """Check if tool usage is forced"""
        # Explicit forcing
        if observation.get("must_use_tool"):
            tool_name = observation.get("tool_to_use")
            for tool in available_tools:
                if tool.name == tool_name and tool_name not in used_tools:
                    self.logger.info(f"Forced tool selection: {tool_name}")
                    return tool
        
        # Task-specific forcing (first attempt)
        if observation.get("attempt", 1) == 1:
            # ChartQA should use ChartMoE first
            if observation.get("is_visual_question") and observation.get("chartmoe_enabled"):
                for tool in available_tools:
                    if tool.name == "chartmoe" and "chartmoe" not in used_tools:
                        self.logger.info("ChartQA first attempt: forcing ChartMoE")
                        return tool
            
            # Geometry with MultiMath enabled
            if observation.get("is_geometry_task") and observation.get("multimath_enabled"):
                for tool in available_tools:
                    if tool.name == "multimath_server" and "multimath_server" not in used_tools:
                        self.logger.info("Geometry with MultiMath: forcing MultiMath Server")
                        return tool
        
        return None
    
    def _analyze_needed_capabilities(
        self, 
        observation: Dict[str, Any], 
        task_type: str
    ) -> List[str]:
        """Analyze what capabilities are needed for this task"""
        question = observation.get("question", "").lower()
        
        # Get base capabilities for task type
        base_capabilities = self.task_capabilities.get(task_type, [])
        needed = []
        
        # Filter capabilities based on question content
        for capability in base_capabilities:
            if self._is_capability_relevant(capability, question, observation):
                needed.append(capability)
        
        # Add additional capabilities based on specific patterns
        additional = self._detect_additional_capabilities(question, observation)
        needed.extend(additional)
        
        return list(set(needed))  # Remove duplicates
    
    def _is_capability_relevant(
        self, 
        capability: str, 
        question: str,
        observation: Dict[str, Any]
    ) -> bool:
        """Check if a capability is relevant to the question"""
        relevance_patterns = {
            "chart_analysis": ["trend", "pattern", "analyze", "compare"],
            "table_extraction": ["table", "value", "data", "number"],
            "text_recognition": ["text", "read", "written", "label"],
            "geometric_reasoning": ["angle", "triangle", "prove", "theorem"],
            "formal_proof": ["prove", "show that", "verify", "theorem"],
            "calculation": ["calculate", "compute", "find", "solve"],
            "medical_segmentation": ["region", "area", "segment", "boundary"],
            "organ_detection": ["organ", "brain", "lung", "heart", "liver"],
            "counting": ["how many", "count", "number of"],
            "object_localization": ["where", "locate", "position", "find"]
        }
        
        patterns = relevance_patterns.get(capability, [])
        return any(pattern in question for pattern in patterns)
    
    def _detect_additional_capabilities(
        self,
        question: str,
        observation: Dict[str, Any]
    ) -> List[str]:
        """Detect additional capabilities needed beyond task type defaults"""
        additional = []
        
        # Visual enhancement needed
        if any(word in question for word in ["unclear", "blurry", "small", "zoom"]):
            additional.append("visual_understanding")
        
        # Text extraction needed even for non-text tasks
        if "text" in question or "read" in question:
            additional.append("text_recognition")
        
        # Counting needed for various tasks
        if "how many" in question or "count" in question:
            additional.append("counting")
        
        return additional
    
    def _get_capable_tools(
        self,
        capabilities: List[str],
        available_tools: List[Any],
        used_tools: Set[str]
    ) -> List[Any]:
        """Get tools that can provide the needed capabilities"""
        capable_tool_names = set()
        
        # Find all tools that provide any of the needed capabilities
        for capability in capabilities:
            tool_names = self.capability_tools.get(capability, [])
            capable_tool_names.update(tool_names)
        
        # Filter to available and unused tools
        capable_tools = []
        for tool in available_tools:
            if tool.name in capable_tool_names and tool.name not in used_tools:
                capable_tools.append(tool)
        
        return capable_tools
    
    def _analyze_alternative_capabilities(
        self,
        observation: Dict[str, Any],
        feedback: str,
        task_type: str,
        used_tools: Set[str]
    ) -> List[str]:
        """Analyze what alternative capabilities might help based on feedback"""
        alternatives = []
        
        # For counting errors, try different extraction method
        if "count" in feedback or "number" in feedback:
            if "chartmoe" in used_tools:
                alternatives.append("text_recognition")  # Try OCR
            elif "easyocr" in used_tools:
                alternatives.append("chart_analysis")  # Try ChartMoE
        
        # For calculation errors, try alternative solver
        if "incorrect" in feedback and task_type == "geometry":
            if "multimath_server" in used_tools:
                alternatives.append("geometric_reasoning")  # Try SymPy
            elif "sympy_geometry" in used_tools:
                alternatives.append("calculation")  # Try MultiMath
        
        # For unclear visuals, try enhancement
        if "unclear" in feedback or "cannot see" in feedback:
            alternatives.append("visual_understanding")
        
        # For missed text, try OCR
        if "text" in feedback or "read" in feedback:
            alternatives.append("text_recognition")
        
        return alternatives
    
    def _is_good_for_feedback(self, tool_name: str, feedback: str) -> bool:
        """Check if tool is particularly good for addressing the feedback"""
        feedback_tool_affinity = {
            "easyocr": ["text", "read", "label", "written"],
            "deepeyes": ["unclear", "zoom", "detail", "small"],
            "chartmoe": ["data", "chart", "graph", "table"],
            "sam2": ["region", "segment", "area", "boundary"],
            "grounding_dino": ["locate", "find", "where", "position"]
        }
        
        affinity_keywords = feedback_tool_affinity.get(tool_name, [])
        return any(keyword in feedback for keyword in affinity_keywords)
    
    def _get_any_unused_capable_tool(
        self,
        observation: Dict[str, Any],
        available_tools: List[Any],
        used_tools: Set[str]
    ) -> Optional[Any]:
        """Get any unused tool that can handle the task"""
        candidates = []
        
        for tool in available_tools:
            if tool.name not in used_tools:
                can_handle, confidence = tool.can_handle(observation)
                if can_handle and confidence > 0.3:  # Minimum confidence threshold
                    candidates.append((tool, confidence))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None