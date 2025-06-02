
#!/usr/bin/env python3
"""VLM Agents module"""

from .base import BaseAgent, AgentConfig
from .vlm_agent import VLMAgent
from .utils import (
    load_image_safely,
    parse_vlm_response,
    extract_choice_letter,
    clean_response,
    format_prompt_with_choices,
    calculate_confidence_score
)

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentConfig',
    
    # Concrete implementations
    'VLMAgent',
    
    # Utility functions
    'load_image_safely',
    'parse_vlm_response',
    'extract_choice_letter',
    'clean_response',
    'format_prompt_with_choices',
    'calculate_confidence_score',
]
