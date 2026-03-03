#!/usr/bin/env python3
"""DiagramFormalizer tool handler for geometry CDL extraction"""

from typing import Dict, Any, Tuple, Optional
import json
import re
import logging
from ..base import BaseTool, ToolConfig, ToolResult

logger = logging.getLogger(__name__)


class DiagramFormalizerTool(BaseTool):
    """DiagramFormalizer tool handler for extracting CDL from geometric figures"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.supported_tasks = ['extract_cdl', 'analyze', 'formalize']
        
        # CDL parsing patterns
        self.cdl_patterns = {
            'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl|\nSolution is:|\Z)',
            'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl|\nSolution is:|\Z)',
        }
    
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if this tool can handle the task"""
        question = observation.get("question", "").lower()
        
        # Strong indicators for DiagramFormalizer
        geometry_indicators = [
            'geometric figure', 'diagram', 'construction',
            'formal', 'cdl', 'extract cdl', 'formalize'
        ]
        
        # Check for explicit DiagramFormalizer request
        if observation.get("diagram_formalizer_enabled"):
            return True, 0.95
        
        if observation.get("force_tool_use") and observation.get("tool_to_use") == "diagram_formalizer":
            return True, 1.0
        
        # Check for geometry problem that needs formalization
        score = 0.0
        
        # Check for CDL extraction request
        if 'extract cdl' in question or 'construction_cdl' in question or 'image_cdl' in question:
            score = 0.9
        
        # Check for formalization need
        elif any(indicator in question for indicator in geometry_indicators):
            score = 0.7
        
        # Check for complex geometry that benefits from formalization
        elif self._is_complex_geometry(observation):
            score = 0.6
        
        return score > 0, score
    
    def build_prompt(self, observation: Dict[str, Any]) -> str:
        """Build tool call prompt for DiagramFormalizer"""
        question = observation.get("question", "")
        original_question = observation.get("original_question", question)
        
        # Determine task type
        task = self._determine_task(question)
        
        # Build problem description for DiagramFormalizer
        if 'extract cdl' in question.lower() or task == 'extract_cdl':
            # Direct CDL extraction
            problem = "Look at the geometric figure in the image. Please describe the construction and measurements by predicting the construction_cdl and image_cdl."
        else:
            # Clean the question for geometry problem
            problem = self._clean_geometry_question(original_question)
        
        # Build tool call
        tool_call = {
            "tool": "diagram_formalizer",
            "parameters": {
                "task": task,
                "problem": problem
            }
        }
        
        logger.debug(f"DiagramFormalizer prompt: task={task}, problem='{problem[:100]}...'")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def process_result(self, raw_result: Any, observation: Dict[str, Any]) -> ToolResult:
        """Process DiagramFormalizer output"""
        if isinstance(raw_result, dict):
            # Extract CDL from the result
            cdl_data = {}
            success = False
            
            # Try different fields where CDL might be
            for field in ['formalized_output', 'raw_response', 'output', 'result']:
                if field in raw_result and raw_result[field]:
                    cdl_data = self._parse_cdl(str(raw_result[field]))
                    if cdl_data:
                        success = True
                        logger.info(f"Successfully extracted CDL from field: {field}")
                        break
            
            # Check if we got valid CDL
            if not cdl_data or not any(k in cdl_data for k in ['construction_cdl', 'image_cdl']):
                # Try to extract from the entire result
                cdl_data = self._parse_cdl(str(raw_result))
                success = bool(cdl_data)
            
            # Build structured CDL result
            result_data = {
                "cdl_data": cdl_data,
                "construction_cdl": cdl_data.get("construction_cdl", ""),
                "image_cdl": cdl_data.get("image_cdl", ""),
                "formalized_problem": self._extract_formalized_problem(cdl_data, observation),
                "constraints": self._extract_constraints(cdl_data),
                "known_values": self._extract_known_values(cdl_data)
            }
            
            # Add status information
            if success and cdl_data:
                status = "success"
                error = None
                # Check if we have partial success
                if not cdl_data.get("construction_cdl") or not cdl_data.get("image_cdl"):
                    result_data["partial_success"] = True
                    status = "partial_success"
            else:
                status = "failed"
                error = "Failed to extract valid CDL from DiagramFormalizer output"
                result_data["partial_success"] = False
            
            return ToolResult(
                success=success,
                data=result_data,
                error=error,
                metadata={
                    "status": status,
                    "has_construction": bool(cdl_data.get("construction_cdl")),
                    "has_image_cdl": bool(cdl_data.get("image_cdl")),
                    "raw_output_length": len(str(raw_result))
                }
            )
        
        return ToolResult(
            success=False,
            data=None,
            error="Invalid DiagramFormalizer result format"
        )
    
    def format_for_answer(self, result: ToolResult, observation: Dict[str, Any]) -> str:
        """Format DiagramFormalizer result for final answer generation"""
        if not result.success:
            return f"DiagramFormalizer failed: {result.error}"
        
        data = result.data
        cdl_data = data.get("cdl_data", {})
        
        # Build formatted output
        prompt_parts = []
        
        if cdl_data.get("construction_cdl"):
            prompt_parts.append(f"Construction CDL:\n{cdl_data['construction_cdl']}")
        
        if cdl_data.get("image_cdl"):
            prompt_parts.append(f"Image CDL:\n{cdl_data['image_cdl']}")
        
        if data.get("formalized_problem"):
            prompt_parts.append(f"Formalized Problem: {data['formalized_problem']}")
        
        if data.get("constraints"):
            prompt_parts.append(f"Constraints: {', '.join(data['constraints'])}")
        
        if data.get("known_values"):
            values_str = ", ".join([f"{k}={v}" for k, v in data['known_values'].items()])
            prompt_parts.append(f"Known Values: {values_str}")
        
        formatted_cdl = "\n\n".join(prompt_parts) if prompt_parts else "No CDL extracted"
        
        return f"""DiagramFormalizer extracted the following formal representation:

{formatted_cdl}

Based on this formal representation, the geometric problem can now be solved systematically.

For the question: "{observation.get('question', '')}"

Please provide your answer inside <answer> tags."""
    
    def _determine_task(self, question: str) -> str:
        """Determine the DiagramFormalizer task type"""
        question_lower = question.lower()
        
        if 'extract cdl' in question_lower or 'construction_cdl' in question_lower:
            return 'extract_cdl'
        elif 'formalize' in question_lower or 'formal' in question_lower:
            return 'formalize'
        else:
            return 'analyze'
    
    def _clean_geometry_question(self, question: str) -> str:
        """Clean geometry question for DiagramFormalizer"""
        # Remove tool-forcing instructions
        cleaned = re.sub(r'IMPORTANT:.*?(?:tool|MANDATORY:).*?\n', '', question, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'Step \d+:.*?\n', '', cleaned)
        cleaned = re.sub(r'MANDATORY:.*?\n', '', cleaned)
        
        # Extract just the geometry problem
        geometry_match = re.search(r'Geometry problem:\s*(.+?)(?:\n\n|$)', cleaned, re.DOTALL)
        if geometry_match:
            cleaned = geometry_match.group(1)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _parse_cdl(self, output: str) -> Dict[str, str]:
        """Parse CDL from DiagramFormalizer output"""
        results = {}
        
        for key, pattern in self.cdl_patterns.items():
            match = re.search(pattern, output, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
                logger.debug(f"Extracted {key}: {len(results[key])} chars")
        
        return results
    
    def _extract_formalized_problem(self, cdl_data: Dict[str, str], observation: Dict[str, Any]) -> str:
        """Extract formalized problem statement from CDL"""
        # Try to extract problem statement from image_cdl
        image_cdl = cdl_data.get("image_cdl", "")
        
        # Look for problem statement patterns
        problem_patterns = [
            r'Find\s+(.+?)(?:\.|$)',
            r'Calculate\s+(.+?)(?:\.|$)',
            r'Prove\s+(.+?)(?:\.|$)',
            r'Show\s+that\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in problem_patterns:
            match = re.search(pattern, image_cdl, re.IGNORECASE)
            if match:
                return f"Find {match.group(1)}"
        
        # Fallback to original question
        return observation.get("original_question", observation.get("question", ""))
    
    def _extract_constraints(self, cdl_data: Dict[str, str]) -> list:
        """Extract constraints from CDL"""
        constraints = []
        
        # Extract from construction_cdl
        construction = cdl_data.get("construction_cdl", "")
        
        # Look for common constraint patterns
        constraint_patterns = [
            r'(\w+)\s+(?:is\s+)?parallel\s+(?:to\s+)?(\w+)',
            r'(\w+)\s+(?:is\s+)?perpendicular\s+(?:to\s+)?(\w+)',
            r'(\w+)\s*=\s*(\w+)',  # Equal lengths/angles
            r'angle\s+(\w+)\s*=\s*(\d+)',
            r'(\w+)\s+is\s+(?:a\s+)?right\s+angle',
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, construction + " " + cdl_data.get("image_cdl", ""), re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    constraint = " ".join(match)
                else:
                    constraint = match
                constraints.append(constraint)
        
        return constraints[:10]  # Limit to 10 most important constraints
    
    def _extract_known_values(self, cdl_data: Dict[str, str]) -> Dict[str, Any]:
        """Extract known values from CDL"""
        known_values = {}
        
        # Combine CDL texts
        full_text = cdl_data.get("construction_cdl", "") + " " + cdl_data.get("image_cdl", "")
        
        # Look for value assignments
        value_patterns = [
            r'(\w+)\s*=\s*(\d+(?:\.\d+)?)',  # Simple assignment
            r'length\s+(?:of\s+)?(\w+)\s*=\s*(\d+(?:\.\d+)?)',
            r'angle\s+(\w+)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+(?:is|measures?)\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in value_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                var_name = match[0]
                value = match[1]
                try:
                    known_values[var_name] = float(value)
                except:
                    known_values[var_name] = value
        
        return known_values
    
    def _is_complex_geometry(self, observation: Dict[str, Any]) -> bool:
        """Check if this is a complex geometry problem that benefits from formalization"""
        question = observation.get("question", "").lower()
        
        # Indicators of complex geometry
        complex_indicators = [
            'prove', 'show that', 'verify',
            'congruent', 'similar', 'theorem',
            'given that', 'if and only if',
            'construct', 'bisector', 'circumcircle'
        ]
        
        return any(indicator in question for indicator in complex_indicators)
    
    def validate_input(self, observation: Dict[str, Any]) -> bool:
        """Validate that input has required fields for DiagramFormalizer"""
        # Must have an image for geometric figure
        if not observation.get("image_path"):
            logger.warning("No image path provided for DiagramFormalizer")
            return False
        
        return True
    
    def should_retry(self, result: ToolResult, attempt: int) -> bool:
        """Determine if should retry DiagramFormalizer"""
        # Retry if completely failed and under max attempts
        if not result.success and attempt < self.config.max_retries:
            return True
        
        # Retry if partial success on first attempt
        if result.data and result.data.get("partial_success") and attempt == 1:
            return True
        
        return False
    
    def get_fallback_strategy(self) -> Optional[str]:
        """Get fallback strategy if DiagramFormalizer fails"""
        # If DiagramFormalizer fails, can try:
        # 1. Direct solving with MultiMath (without CDL)
        # 2. Using SymPy Geometry for specific calculations
        return "multimath_direct"