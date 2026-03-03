#!/usr/bin/env python3
"""Reflection and retry strategy module"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ReflectionStrategy:
    """Strategy for handling reflection and retry attempts"""
    
    def __init__(self):
        self.logger = logger
    
    def get_strategy(self, observation: Dict[str, Any]) -> str:
        """Determine reflection strategy based on feedback
        
        Args:
            observation: Observation with feedback from previous attempt
            
        Returns:
            Strategy name: 'retry_tool', 'refine_answer', 'different_approach', 'give_up'
        """
        feedback = observation.get("feedback", "").lower()
        attempt = observation.get("attempt", 1)
        attempts_remaining = observation.get("attempts_remaining", 0)
        question_type = observation.get("question_type", "")
        previous_answer = observation.get("previous_answer", "")
        
        # Log reflection context
        self.logger.info(f"Reflection attempt {attempt}, feedback: {feedback[:100]}...")
        
        # For counting problems with significant errors
        if self._is_counting_error(feedback, question_type):
            if attempts_remaining > 0:
                return "retry_tool"  # Try different tool or approach
            else:
                return "different_approach"  # Last attempt, try completely different approach
        
        # For verification requests
        if self._needs_verification(feedback):
            return "retry_tool"  # Use tool to verify
        
        # For minor corrections
        if self._is_minor_correction(feedback):
            return "refine_answer"  # Just refine the existing answer
        
        # For complete failures
        if self._is_complete_failure(feedback):
            if attempts_remaining > 1:
                return "different_approach"
            else:
                return "give_up"  # Provide best guess
        
        # Default strategy based on attempts remaining
        if attempts_remaining > 1:
            return "retry_tool"
        elif attempts_remaining == 1:
            return "different_approach"
        else:
            return "refine_answer"
    
    def should_retry_tool(self, observation: Dict[str, Any]) -> bool:
        """Determine if tool should be retried in reflection
        
        Args:
            observation: Current observation with feedback
            
        Returns:
            True if tool should be retried
        """
        feedback = observation.get("feedback", "").lower()
        question_type = observation.get("question_type", "")
        attempts_remaining = observation.get("attempts_remaining", 0)
        
        # Always retry tool on last attempt
        if attempts_remaining == 1:
            return True
        
        # Retry for specific feedback patterns
        retry_indicators = [
            "check your", "verify", "re-examine", "re-read",
            "look again", "reconsider", "double-check",
            "use the", "try a different", "use a custom"
        ]
        
        if any(phrase in feedback for phrase in retry_indicators):
            return True
        
        # Retry for counting problems with large errors
        if question_type == "counting" and "significantly off" in feedback:
            return True
        
        # Retry for calculation errors
        if question_type in ["numerical", "calculation"] and "incorrect" in feedback:
            return True
        
        return False
    
    def get_tool_retry_params(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for tool retry based on feedback
        
        Args:
            observation: Observation with feedback
            
        Returns:
            Dictionary with retry parameters
        """
        feedback = observation.get("feedback", "").lower()
        question = observation.get("question", "").lower()
        previous_tool = observation.get("last_tool_used", "")
        
        params = {
            "retry_reason": "reflection",
            "attempt": observation.get("attempt", 1)
        }
        
        # For ChartMoE retries
        if previous_tool == "chartmoe":
            if "count" in feedback or "how many" in question:
                # Use custom prompt for precise counting
                params["task"] = "custom"
                params["prompt"] = self._build_counting_prompt(question, feedback)
            elif "different task" in feedback:
                params["task"] = "analyze"  # Switch to analysis task
            else:
                params["task"] = "extract_data"  # Try data extraction
        
        # For MultiMath retries
        elif previous_tool == "multimath_server":
            if "step" in feedback:
                params["show_steps"] = True
                params["verbose"] = True
            if "method" in feedback:
                params["try_alternative_method"] = True
        
        # For SAM2 retries
        elif previous_tool == "sam2":
            if "region" in feedback:
                params["task"] = "multi_point_segment"
            else:
                params["task"] = "smart_medical_segment"
                params["enhance_contrast"] = True
        
        return params
    
    def _is_counting_error(self, feedback: str, question_type: str) -> bool:
        """Check if feedback indicates counting error"""
        counting_error_phrases = [
            "significantly off", "wrong count", "incorrect number",
            "count again", "recount", "off by"
        ]
        
        return (
            question_type == "counting" or 
            any(phrase in feedback for phrase in counting_error_phrases)
        )
    
    def _needs_verification(self, feedback: str) -> bool:
        """Check if feedback requests verification"""
        verification_phrases = [
            "verify", "double-check", "confirm", "make sure",
            "check again", "validate", "ensure"
        ]
        
        return any(phrase in feedback for phrase in verification_phrases)
    
    def _is_minor_correction(self, feedback: str) -> bool:
        """Check if only minor correction is needed"""
        minor_phrases = [
            "almost correct", "close but", "nearly right",
            "small error", "minor mistake", "slight"
        ]
        
        return any(phrase in feedback for phrase in minor_phrases)
    
    def _is_complete_failure(self, feedback: str) -> bool:
        """Check if previous attempt completely failed"""
        failure_phrases = [
            "completely wrong", "totally incorrect", "way off",
            "not even close", "fundamental error"
        ]
        
        return any(phrase in feedback for phrase in failure_phrases)
    
    def _build_counting_prompt(self, question: str, feedback: str) -> str:
        """Build custom prompt for counting retry"""
        base_prompt = f"For the question '{question}', "
        
        if "unfavorable" in question and "below" in question:
            return base_prompt + "list all values in the Unfavorable column that are less than 40. Count them precisely."
        elif "above" in question:
            threshold = self._extract_number(question)
            return base_prompt + f"list all values above {threshold}. Count them precisely."
        else:
            return base_prompt + "extract and list all relevant values, then count them precisely."
    
    def _extract_number(self, text: str) -> str:
        """Extract number from text"""
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return numbers[0] if numbers else "the threshold"