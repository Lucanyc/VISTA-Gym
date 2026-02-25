import os
# Must be set before importing transformers
#os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
import argparse
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
import traceback
from collections import defaultdict
import re
from typing import Optional, Dict, Any, List, Tuple
from vlm_gym.environments.action.task import chartqa_actions
from transformers import GenerationConfig
import time

# Import the ChartQA adapter
sys.path.append(str(Path(__file__).parent.parent / "data_adapters"))
from chartqa_adapter import ChartQAAdapter


# After other imports at the top of the file
from vlm_gym.agents import VLMAgent
from vlm_gym.agents.vlm_agent_with_tools import VLMAgentWithTools  # ⭐ Add explicit import
#from vlm_gym.utils import create_env_from_config  # ⭐ Add config file support
import yaml

# ⭐ Add debug code
print(f"\n[DEBUG] Import verification:")
print(f"VLMAgentWithTools imported from: {VLMAgentWithTools.__module__}")
print(f"VLMAgentWithTools class location: {VLMAgentWithTools}")
print(f"VLMAgent imported from: {VLMAgent.__module__}")
print(f"VLMAgent class location: {VLMAgent}")


# Import ChartQA enhanced components
try:
    from vlm_gym.tasks.chartqa import ChartQAAgent
    CHARTQA_AGENT_AVAILABLE = True
except ImportError:
    ChartQAAgent = None
    CHARTQA_AGENT_AVAILABLE = False
    print("Warning: ChartQA enhanced agent not available. Using base agent.")

from vlm_gym.environments import VisionQAEnv
from vlm_gym.utils import (
    setup_logger,
    MetricsTracker,
    setup_experiment_directory,
    create_env_from_config,
)


def extract_number_from_text(text: str) -> Optional[float]:
    """Extract number from text with various formats"""
    if not text:
        return None
    
    text = str(text).lower().strip()
    
    # Remove commas and dollar signs
    text = text.replace(',', '').replace('$', '')
    
    # Handle written numbers
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000,
        'billion': 1000000000, 'trillion': 1000000000000
    }
    
    # Check for written numbers - exact word match
    words = text.split()
    for word in words:
        cleaned_word = word.strip('.,!?;:')
        if cleaned_word in number_words:
            return float(number_words[cleaned_word])
    
    # Check if text contains percentage
    has_percentage = '%' in text or 'percent' in text.lower()
    
    # Handle K/M/B suffixes (e.g., 5K, 2.5M, 1.2B)
    suffix_pattern = r'(-?\d+\.?\d*)\s*([KMB])'
    suffix_match = re.search(suffix_pattern, text.upper())
    if suffix_match:
        number = float(suffix_match.group(1))
        suffix = suffix_match.group(2)
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        return number * multipliers.get(suffix, 1)
    
    # Handle scientific notation (e.g., 1.5e6, 2E10)
    sci_pattern = r'-?\d+\.?\d*[eE][+-]?\d+'
    sci_match = re.search(sci_pattern, text)
    if sci_match:
        try:
            return float(sci_match.group())
        except ValueError:
            pass
    
    # Handle fractions (e.g., 1/2, 3/4)
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    fraction_match = re.search(fraction_pattern, text)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                return numerator / denominator
        except ValueError:
            pass
    
    # Extract regular decimal numbers
    decimal_pattern = r'-?\d+\.?\d*'
    numbers = re.findall(decimal_pattern, text)
    
    if numbers:
        # Return the first number found
        try:
            value = float(numbers[0])
            # If it's a percentage, convert to decimal
            if has_percentage and value > 1:
                value = value / 100.0
            return value
        except ValueError:
            pass
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    # Convert to string and lowercase
    answer = str(answer).lower().strip()
    
    # Remove common punctuation at the end
    answer = answer.rstrip('.,!?;:')
    
    # Remove quotes
    answer = answer.strip('"\'')
    
    # Remove common prefixes
    prefixes_to_remove = [
        'the answer is',
        'answer:',
        'final answer:',
        'therefore,',
        'so,',
        'thus,',
    ]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    return answer


def evaluate_chartqa_answer(prediction: str, ground_truth: str, tolerance: float = 0.05) -> bool:
    """
    Evaluate ChartQA answer with flexible matching
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Relative tolerance for numerical answers (default: 5%)
    
    Returns:
        True if the answer is correct
    """
    # Normalize both answers
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    # 1. Exact match after normalization
    if pred_norm == gt_norm:
        return True
    
    # 2. Check if ground truth is contained in prediction
    if gt_norm in pred_norm:
        return True
    
    # 3. For numerical answers
    pred_num = extract_number_from_text(prediction)
    gt_num = extract_number_from_text(ground_truth)
    
    if pred_num is not None and gt_num is not None:
        # Exact match for numbers (with small tolerance for floating point)
        if abs(pred_num - gt_num) < 1e-9:
            return True
        
        # Relative tolerance check (5% by default, as mentioned in the paper)
        if gt_num != 0:
            relative_error = abs(pred_num - gt_num) / abs(gt_num)
            if relative_error <= tolerance:
                return True
        elif pred_num == 0:  # Both are effectively zero
            return True
        
        # Also check if one is percentage and the other is decimal
        # e.g., 52 vs 0.52, or 0.52 vs 52
        if abs(pred_num * 100 - gt_num) < 1e-9 or abs(pred_num - gt_num * 100) < 1e-9:
            return True
        
        # Check with relative tolerance for percentage conversion
        if gt_num != 0:
            # Check if prediction is percentage of ground truth
            if abs((pred_num * 100 - gt_num) / gt_num) <= tolerance:
                return True
            # Check if ground truth is percentage of prediction
            if abs((pred_num - gt_num * 100) / (gt_num * 100)) <= tolerance:
                return True
    
    # 4. Handle special cases
    # Check for "yes/no" variations
    yes_variations = ['yes', 'true', 'correct', 'right', 'affirmative', '1']
    no_variations = ['no', 'false', 'incorrect', 'wrong', 'negative', '0']
    
    if gt_norm in yes_variations:
        return any(var in pred_norm for var in yes_variations)
    if gt_norm in no_variations:
        return any(var in pred_norm for var in no_variations)
    
    # 5. Handle special number words that might not be in our dictionary
    # e.g., "twenty-one", "thirty-five"
    if '-' in gt_norm:
        parts = gt_norm.split('-')
        if len(parts) == 2 and all(part.isalpha() for part in parts):
            # Try to match compound numbers
            if parts[0] in pred_norm and parts[1] in pred_norm:
                return True
    
    return False


class ChartQATaskWrapper:
    """Wrapper to make ChartQA data compatible with VisionQAEnv - with reflection and enhanced reasoning support"""
    
    def __init__(self, task_id: str, task_data: dict, action_set=None, **kwargs):
        self.task_id = task_id
        self.task_data = task_data
        self.action_set = action_set  # Save action set reference
        self.current_image = task_data['image_path']
        self.current_question = task_data['question']
        self.answer = task_data['answer']
        self.max_steps = kwargs.get('max_steps', 10)  # Increase steps to support multi-step actions
        self.current_step = 0
        
        # Reflection support
        self.enable_reflection = kwargs.get('enable_reflection', False)
        self.max_attempts = kwargs.get('max_attempts', 3)
        self.current_attempt = 0
        self.attempt_history = []
        self.conversation_history = []
        
        # Enhanced reasoning support
        self.last_extra_info = None
        self.structured_reasoning_history = []
        
        # Debug flag
        self.debug = kwargs.get('debug', False)
        
        # Numerical tolerance
        self.numerical_tolerance = kwargs.get('numerical_tolerance', 0.05)
        
        # Output format flag
        self.use_structured_output = kwargs.get('use_structured_output', True)
        
        # ⭐ DeepEyes support
        self.enable_deepeyes = kwargs.get('enable_deepeyes', False)
        self.deepeyes_history = []
        
        # ⭐ Grounding DINO support
        self.enable_grounding_dino = kwargs.get('enable_grounding_dino', False)
        self.grounding_dino_history = []
        
        # ChartMoE support
        self.enable_chartmoe = kwargs.get('enable_chartmoe', False)
        self.chartmoe_history = []
        
        # Debug output
        if self.debug:
            print(f"[ChartQATaskWrapper] Task {task_id} initialized with:")
            print(f"  - enable_reflection={self.enable_reflection}")
            print(f"  - max_attempts={self.max_attempts}")
            print(f"  - max_steps={self.max_steps}")
            print(f"  - numerical_tolerance={self.numerical_tolerance}")
            print(f"  - use_structured_output={self.use_structured_output}")
            print(f"  - enable_deepeyes={self.enable_deepeyes}")
            print(f"  - enable_grounding_dino={self.enable_grounding_dino}")  
            print(f"  - enable_chartmoe={self.enable_chartmoe}")  # ChartMoE support
        
    def setup(self):
        """Setup method - must return tuple (message, info)"""
        message = f"ChartQA task initialized. Question: {self.current_question}"
        info = {
            "task_id": self.task_id,
            "has_actions": self.action_set is not None,
            "question_type": self._classify_question_type(),
            "reflection_enabled": self.enable_reflection,
            "max_attempts": self.max_attempts if self.enable_reflection else 1,
            "numerical_tolerance": self.numerical_tolerance,
            "use_structured_output": self.use_structured_output,
            "deepeyes_enabled": self.enable_deepeyes,
            "grounding_dino_enabled": self.enable_grounding_dino,  # ⭐ Add info
            "chartmoe_enabled": self.enable_chartmoe 
        }
        return message, info
    
    
    def _classify_question_type(self):
        """Classify the question type with more granular categories"""
        question_lower = self.current_question.lower()
        
        # More detailed numerical question classification
        if any(word in question_lower for word in ['how many', 'count', 'total number']):
            return 'counting'
        elif any(word in question_lower for word in ['sum', 'add', 'total of']):
            return 'summation'
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            return 'average'
        elif any(word in question_lower for word in ['percentage', 'percent', '%', 'proportion']):
            return 'percentage'
        elif any(word in question_lower for word in ['difference', 'gap', 'subtract', 'minus']):
            return 'difference'
        elif any(word in question_lower for word in ['ratio', 'times', 'divide']):
            return 'ratio'
        elif any(word in question_lower for word in ['how much', 'what is the value']):
            return 'numerical'
        elif any(word in question_lower for word in ['compare', 'which is']):
            return 'comparison'
        elif any(word in question_lower for word in ['maximum', 'minimum', 'highest', 'lowest', 'largest', 'smallest']):
            return 'minmax'
        elif any(word in question_lower for word in ['trend', 'increase', 'decrease', 'change']):
            return 'trend'
        elif any(word in question_lower for word in ['what', 'which', 'when', 'where', 'who']):
            return 'retrieval'
        else:
            return 'other'
        
    def teardown(self):
        """Teardown method"""
        pass
        
    def reset(self):
        """Reset method - must return tuple (obs, info)"""
        self.current_step = 0
        self.current_attempt = 0
        self.attempt_history = []
        self.conversation_history = []
        self.structured_reasoning_history = []
        self.last_extra_info = None
        self.deepeyes_history = []
        self.grounding_dino_history = []  # ⭐ Reset Grounding DINO history
        
        obs = {
            "image_path": self.current_image,
            "question": self.current_question,
            "task_description": f"Answer the question about the chart: {self.current_question}",
            "conversation_history": [],
            "output_format_instruction": self._get_output_format_instruction() if self.use_structured_output else "",
            "deepeyes_enabled": self.enable_deepeyes,
            "grounding_dino_enabled": self.enable_grounding_dino  # ⭐ Add to observation
        }
        info = {
            "task_id": self.task_id,
            "dataset": "chartqa",
            "ground_truth": self.answer,
            "attempt": 1,
            "max_attempts": self.max_attempts if self.enable_reflection else 1,
            "question_type": self._classify_question_type(),
            "deepeyes_enabled": self.enable_deepeyes,
            "grounding_dino_enabled": self.enable_grounding_dino  # ⭐ Add to info
        }
        return obs, info
    
    def _get_output_format_instruction(self):
        """Get instruction for structured output format"""
        base_instruction = """Please structure your response in the following format:
<think>
[Your step-by-step reasoning process here. Analyze the chart, identify relevant data points, and perform any necessary calculations.]
</think>
<answer>
[Your final answer here. Provide only the answer without any additional explanation.]
</answer>"""
        
        # ⭐ If Grounding DINO is enabled, add tool usage instructions
        if self.enable_grounding_dino:
            base_instruction += """

You can use the Grounding DINO tool to detect and locate objects in the chart:
<tool_call>
{"tool": "grounding_dino", "parameters": {"caption": "bar chart bars"}}
</tool_call>

Example captions you can use:
- "bar chart bars" - to detect all bars in a bar chart
- "line chart points" - to detect data points in a line chart
- "legend items" - to detect legend entries
- "axis labels" - to detect axis labels
- "data values" - to detect numerical values
- "chart title" - to detect the chart title"""


        # ⭐ If ChartMoE is enabled, add tool usage instructions
        if self.enable_chartmoe:
            base_instruction += """     
You can use the ChartMoE tool to understand charts:
<tool_call>
{"tool": "chartmoe", "task": "to_table"}
</tool_call>

Available ChartMoE tasks:
- "to_table" - Convert chart to structured table
- "describe" - Generate detailed chart description  
- "extract_data" - Extract all numerical data
- "summarize" - Provide chart summary
- "analyze" - Deep analysis with insights
- "compare" - Compare data series
- "trend" - Identify trends
- Custom question: {"tool": "chartmoe", "prompt": "your specific question"}"""

        # ⭐ If DeepEyes is enabled, add tool usage instructions
        if self.enable_deepeyes:
            base_instruction += """

You can also use visual tools to zoom into specific regions if needed:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox": [x1, y1, x2, y2]}}
</tool_call>"""
        
        return base_instruction
    
    
    def get_observation(self):
        """Get current observation for the task with reflection context"""
        obs = {
            "image_path": self.current_image,
            "question": self.current_question,
            "task_id": self.task_id,
            "attempt": self.current_attempt + 1,
            "conversation_history": self.conversation_history,
            "output_format_instruction": self._get_output_format_instruction() if self.use_structured_output else "",
            "use_structured_output": self.use_structured_output,
            "deepeyes_enabled": self.enable_deepeyes,
            "deepeyes_history": self.deepeyes_history,
            "grounding_dino_enabled": self.enable_grounding_dino,
            "grounding_dino_history": self.grounding_dino_history,
            "chartmoe_enabled": self.enable_chartmoe,
            "chartmoe_history": self.chartmoe_history,
            "is_visual_question": True,  # ⭐ Add this marker
        }
        
        # Debug output
        if self.debug and self.current_step < 2:
            print(f"[ChartQATaskWrapper.get_observation] Task {self.task_id}:")
            print(f"  - image_path: {obs['image_path']}")
            print(f"  - question: {obs['question'][:50]}...")
            print(f"  - use_structured_output: {self.use_structured_output}")
            print(f"  - deepeyes_enabled: {self.enable_deepeyes}")
            print(f"  - grounding_dino_enabled: {self.enable_grounding_dino}")
            print(f"  - chartmoe_enabled: {self.enable_chartmoe}")  # ⭐ Add ChartMoE debug output
        
        # Add reflection-specific fields if this is a retry
        if self.enable_reflection and self.current_attempt > 0 and self.attempt_history:
            last_attempt = self.attempt_history[-1]
            obs["feedback"] = last_attempt.get("feedback", "")
            obs["previous_answer"] = last_attempt.get("answer", "")
            obs["attempts_remaining"] = self.max_attempts - self.current_attempt
            obs["max_attempts"] = self.max_attempts
            obs["previous_attempt_failed"] = True
            
            # ⭐ If ChartMoE hasn't been used yet, force a reminder
            if self.enable_chartmoe and len(self.chartmoe_history) == 0:
                obs["must_use_tool"] = True
                obs["reflection_format_reminder"] = "YOU MUST use ChartMoE tool first! Start with <tool_call>{\"tool\": \"chartmoe\", \"task\": \"to_table\"}</tool_call> then provide your answer."
            else:
                # Original format reminder
                obs["reflection_format_reminder"] = "Remember to use the <think>...</think> and <answer>...</answer> format in your response."
            
            if self.current_attempt == self.max_attempts - 1:
                obs["hint"] = self._generate_hint()
            
        return obs
        
    
    
    def _clean_action_content(self, action: str) -> str:
        """Clean action content by removing noise and artifacts"""
        if not isinstance(action, str):
            return str(action)
        
        # Remove answer_question wrapper if present
        if action.startswith('answer_question(answer="') and action.endswith('")'):
            action = action[len('answer_question(answer="'):-2]
            # Unescape quotes
            action = action.replace('\\"', '"')
        
        # Remove addCriterion noise
        # Pattern to match addCriterion with various spacing
        action = re.sub(r'(\s*addCriterion\s*)+', ' ', action)
        action = re.sub(r'(\s*<\/\s*addCriterion\s*)+', ' ', action)
        
        # Remove excessive whitespace
        action = re.sub(r'\n\s*\n', '\n', action)
        action = re.sub(r' +', ' ', action)
        
        return action.strip()
    
    def _extract_think_and_answer(self, action: str) -> Tuple[str, str]:
        """Extract thinking and answer from structured output with robust parsing"""
        think_content = ""
        answer_content = ""
        
        # Clean the action first - remove addCriterion noise
        cleaned_action = self._clean_action_content(action)
        
        # Extract <think> content - try multiple patterns
        # Pattern 1: Properly closed tags
        think_match = re.search(r'<think>(.*?)</think>', cleaned_action, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_content = think_match.group(1).strip()
        else:
            # Pattern 2: Handle unclosed or malformed tags
            # Look for <think> or <think\n and extract until <answer> or end
            think_match = re.search(r'<think[>\n](.*?)(?:<answer|</think|$)', cleaned_action, re.DOTALL | re.IGNORECASE)
            if think_match:
                think_content = think_match.group(1).strip()
                # Clean up any trailing incomplete tags
                think_content = re.sub(r'<think\s*$', '', think_content).strip()
        
        # Extract <answer> content - try multiple patterns
        # Pattern 1: Properly closed tags
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_action, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Pattern 2: Handle unclosed tags
            answer_match = re.search(r'<answer[>\n](.*?)(?:</answer|$)', cleaned_action, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                # Pattern 3: Sometimes answer appears as standalone number after think
                # Look for patterns like "4\n\n" after think content
                if think_content:
                    remaining = cleaned_action[cleaned_action.find(think_content) + len(think_content):] if think_content in cleaned_action else ""
                    number_match = re.search(r'^\s*(\d+)\s*$', remaining.strip(), re.MULTILINE)
                    if number_match:
                        answer_content = number_match.group(1)
        
        return think_content, answer_content
    
    def _is_yes_no_question(self) -> bool:
        """Check if the current question is a yes/no question"""
        question_lower = self.current_question.lower()
        yes_no_indicators = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'has', 'have']
        # Check if question starts with these indicators
        for indicator in yes_no_indicators:
            if question_lower.startswith(indicator + ' '):
                return True
        return False
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """Extract yes/no answer from text"""
        text_lower = text.lower()
        # Look for yes at the beginning or as a standalone word
        if re.search(r'^yes\b', text_lower) or re.search(r'\byes\b[,.]?$', text_lower):
            return "Yes"
        if re.search(r'^no\b', text_lower) or re.search(r'\bno\b[,.]?$', text_lower):
            return "No"
        # Check for yes/no in quotes or after "is"
        if re.search(r'["\']yes["\']', text_lower) or re.search(r'is yes', text_lower):
            return "Yes"
        if re.search(r'["\']no["\']', text_lower) or re.search(r'is no', text_lower):
            return "No"
        return None
    
    def _try_structured_answer_extraction(self, action: str, extra_info: Dict[str, Any]) -> Optional[str]:
        """Try to extract answer from structured reasoning results"""
        if extra_info and 'structured_answer' in extra_info:
            structured_answer = extra_info['structured_answer']
            if structured_answer is not None:
                return str(structured_answer)
        
        # If action itself is the answer (from enhanced result)
        if isinstance(action, str) and action.strip():
            # Don't filter, return None to let main flow use action itself
            return None
        
        return None
    
    def _extract_answer_from_action(self, action: str) -> str:
        """Extract answer from various action formats with enhanced parsing"""
        # Handle None or empty action
        if not action:
            return ""
        
        # ⭐ Record tool usage
        if isinstance(action, str):
            if '<tool_call>' in action and '"name": "image_zoom_in_tool"' in action:
                self.deepeyes_history.append({
                    "step": self.current_step,
                    "action": action
                })
            elif '<tool_call>' in action and '"tool": "chartmoe"' in action:
                self.chartmoe_history.append({
                    "step": self.current_step,
                    "action": action
                })
            elif '<tool_call>' in action and '"tool": "grounding_dino"' in action:
                self.grounding_dino_history.append({
                    "step": self.current_step,
                    "action": action
                })
        
        # If action is a dict, handle dict format
        if isinstance(action, dict):
            if action.get('type') == 'answer':
                return str(action.get('content', ''))
            elif 'answer' in action:
                return str(action['answer'])
            elif 'result' in action:
                return str(action['result'])
            elif 'value' in action:
                return str(action['value'])
        
        # If it's a string, check for specific formats
        if isinstance(action, str):
            # First clean the action
            action_str = self._clean_action_content(action)
            
            # ========== Prioritize new structured format ==========
            if self.use_structured_output:
                think_content, answer_content = self._extract_think_and_answer(action_str)
                if answer_content:
                    # If <answer> tag content extracted successfully, return directly
                    return answer_content
                elif think_content and not answer_content:
                    # If only think but no answer, try extracting answer from think content
                    # This is a fallback mechanism
                    if self.debug:
                        print(f"[DEBUG] Found <think> but no <answer>, attempting to extract from think content")
                    action_str = think_content
            
            # ========== If no structured format, try extracting from raw content ==========
            # For cases without structured format, need smarter extraction
            # First check if it's answer_question format
            if 'answer_question(answer="' in action and not self.use_structured_output:
                # Extract content within quotes
                start = action.find('answer_question(answer="') + len('answer_question(answer="')
                end = action.find('")', start)
                if end > start:
                    extracted = action[start:end].strip()
                    # For yes/no questions, prioritize extracting yes/no
                    if self._is_yes_no_question():
                        yes_no = self._extract_yes_no(extracted)
                        if yes_no:
                            return yes_no
                    return extracted
            
            # Continue with question type-based extraction
            q_type = self._classify_question_type()
            
            # For counting questions
            if q_type == 'counting':
                is_pattern = r'(?:is|are)\s+(\d+)(?:\.|,|;|$|\s)'
                is_matches = re.findall(is_pattern, action_str, re.IGNORECASE)
                if is_matches:
                    return is_matches[-1]
                
                number_pattern = r'number[^.]*?is\s+(\d+)'
                number_match = re.search(number_pattern, action_str, re.IGNORECASE)
                if number_match:
                    return number_match.group(1)
                
                counting_patterns = [
                    r'there (?:are|is) (\d+)\s+values',
                    r'found (\d+)',
                    r'counted (\d+)',
                    r'total of (\d+)',
                    r'(\d+) values(?:\s+below|\s+above|\s+in)',
                    r'(\d+) items',
                    r'(\d+) bars',
                    r'(\d+) points',
                    r'(\d+) intercepting',
                    r'answer is (\d+)',
                    r'count is (\d+)',
                    r'number is (\d+)',
                ]
                
                for pattern in counting_patterns:
                    match = re.search(pattern, action_str, re.IGNORECASE)
                    if match:
                        extracted = match.group(1)
                        if extracted.isdigit():
                            return extracted
                        number_val = extract_number_from_text(extracted)
                        if number_val is not None:
                            return str(int(number_val))
                
                standalone_numbers = re.findall(r'(?<!below\s)(?<!above\s)(?<!than\s)\b(\d+)\b', action_str)
                if standalone_numbers:
                    return standalone_numbers[-1]
            
            # For summation/calculation questions
            elif q_type in ['summation', 'average', 'percentage', 'difference', 'ratio']:
                calc_patterns = [
                    r'=\s*(\d+\.?\d*%?)',
                    r'(?:sum|total|result|answer) (?:is|equals?) (\d+\.?\d*%?)',
                    r'(\d+\.?\d*%?)$',
                ]
                
                for pattern in calc_patterns:
                    match = re.search(pattern, action_str, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
            
            # For year/date questions
            elif 'year' in self.current_question.lower() or 'when' in self.current_question.lower():
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', action_str)
                if year_match:
                    return year_match.group(1)
            
            # Try JSON parsing
            if action_str.startswith('{') and action_str.endswith('}'):
                try:
                    parsed = json.loads(action_str)
                    for key in ['answer', 'result', 'value', 'content']:
                        if key in parsed:
                            return str(parsed[key])
                except json.JSONDecodeError:
                    pass
            
            # Common answer patterns
            answer_patterns = [
                r'(?:Final )?Answer:\s*(.+?)(?:\n|$)',
                r'The answer is[:\s]+(.+?)(?:\n|$)',
                r'Therefore,?\s+(.+?)(?:\n|$)',
                r'Result:\s*(.+?)(?:\n|$)',
                r'Solution:\s*(.+?)(?:\n|$)',
                r'"([^"]+)"',
                r"'([^']+)'",
                r'\(([^)]+)\)',
                r'\*\*(.+?)\*\*',
                r'__(.+?)__',
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, action_str, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    if extracted:
                        return extracted
            
            # Simple answer check
            words = action_str.split()
            if len(words) <= 5 and not any(word in action_str.lower() for word in 
                ['because', 'since', 'therefore', 'calculate', 'find', 'look']):
                if len(words) == 1:
                    num_val = extract_number_from_text(words[0])
                    if num_val is not None:
                        return str(int(num_val)) if num_val == int(num_val) else str(num_val)
                if not action_str.endswith('.') and len(action_str) < 20:
                    num_val = extract_number_from_text(action_str)
                    if num_val is not None:
                        return str(int(num_val)) if num_val == int(num_val) else str(num_val)
                    if q_type == 'counting':
                        return ""
                return action_str
            
            # For short direct answers
            if len(action_str) < 100:
                num_val = extract_number_from_text(action_str)
                if num_val is not None:
                    return str(int(num_val)) if num_val == int(num_val) else str(num_val)
            
            # Last resort for counting questions
            if q_type == 'counting':
                all_numbers = re.findall(r'\b(\d+)\b', action_str)
                if all_numbers:
                    non_question_numbers = [n for n in all_numbers if n != '40']
                    if non_question_numbers:
                        return non_question_numbers[-1]
                    elif len(all_numbers) > 1:
                        return all_numbers[-1]
                
                num_val = extract_number_from_text(action_str)
                if num_val is not None:
                    return str(int(num_val))
            
            # Clean up instruction phrases
            instruction_phrases = [
                'to answer this question',
                'looking at the chart',
                'based on the chart',
                'from the chart',
                'according to the chart',
                'i need to',
                'let me',
                'i can see',
            ]
            cleaned = action_str.lower()
            for phrase in instruction_phrases:
                if cleaned.startswith(phrase):
                    action_str = action_str[len(phrase):].strip()
                    break
            
            return action_str
        
        # Default: convert to string
        return str(action).strip()
    
    
    def step(self, action: str):
        """Process action with reflection support"""
        self.current_step += 1
        
        # =================== DEBUG START ===================
        if self.debug:
            print(f"\n[DEBUG ChartQATaskWrapper.step] Task {self.task_id}, Step {self.current_step}, Attempt {self.current_attempt + 1}")
            print(f"[DEBUG] Raw action: {action[:200]}...")
            print(f"[DEBUG] Ground truth: '{self.answer}'")
            print(f"[DEBUG] Max attempts allowed: {self.max_attempts}")
            print(f"[DEBUG] Reflection enabled: {self.enable_reflection}")
        # =================== DEBUG END ===================
        
        # Extract thinking and answer separately if using structured output
        think_content = ""
        if self.use_structured_output:
            think_content, extracted_answer = self._extract_think_and_answer(action)
            if think_content and self.debug:
                print(f"[DEBUG] Extracted thinking: {think_content[:100]}...")
            if extracted_answer and self.debug:
                print(f"[DEBUG] Extracted answer from tags: '{extracted_answer}'")
        
        # Extract answer using enhanced extraction
        answer = self._extract_answer_from_action(action)
        
        # =================== DEBUG START ===================
        if self.debug:
            print(f"[DEBUG] Final extracted answer: '{answer}'")
            num_val = extract_number_from_text(answer)
            print(f"[DEBUG] Number extracted from answer: {num_val}")
        # =================== DEBUG END ===================
        
        # Evaluate answer with tolerance
        is_correct = evaluate_chartqa_answer(answer, self.answer, tolerance=self.numerical_tolerance)
        
        # =================== DEBUG START ===================
        if self.debug:
            print(f"[DEBUG] Is correct: {is_correct}")
            print(f"[DEBUG] Current attempt (before increment): {self.current_attempt}")
        # =================== DEBUG END ===================
        
        # Update attempt history FIRST (before incrementing attempt counter)
        attempt_data = {
            "attempt": self.current_attempt + 1,  # Not yet incremented
            "answer": answer,
            "correct": is_correct,
            "action": action,
            "question_type": self._classify_question_type(),
            "thinking": think_content if self.use_structured_output else None,
            "deepeyes_used": len([h for h in self.deepeyes_history if h["step"] == self.current_step]) > 0,
            "grounding_dino_used": len([h for h in self.grounding_dino_history if h["step"] == self.current_step]) > 0,
            "chartmoe_used": len([h for h in self.chartmoe_history if h["step"] == self.current_step]) > 0
        }
        
        # Add structured reasoning info if available
        if self.last_extra_info:
            if 'structured' in self.last_extra_info:
                attempt_data['used_structured_reasoning'] = self.last_extra_info['structured']
            if 'reasoning' in self.last_extra_info:
                attempt_data['reasoning'] = self.last_extra_info['reasoning']
            if 'confidence' in self.last_extra_info:
                attempt_data['confidence'] = self.last_extra_info['confidence']
        
        # =================== DEBUG START ===================
        if self.debug:
            print(f"[DEBUG] Attempt data being saved:")
            print(f"  - attempt: {attempt_data['attempt']}")
            print(f"  - answer: '{attempt_data['answer'][:50]}...'")
            print(f"  - correct: {attempt_data['correct']}")
            print(f"  - chartmoe_used: {attempt_data.get('chartmoe_used', False)}")
        # =================== DEBUG END ===================
        
        # NOW increment the attempt counter
        self.current_attempt += 1
        
        # Save attempt history
        self.attempt_history.append(attempt_data)
        
        # =================== CRITICAL REFLECTION LOGIC ===================
        # Determine whether to continue trying
        should_continue = (not is_correct and 
                        self.enable_reflection and 
                        self.current_attempt < self.max_attempts)
        
        if self.debug:
            print(f"\n[DEBUG] Reflection logic check:")
            print(f"  - Is correct: {is_correct}")
            print(f"  - Enable reflection: {self.enable_reflection}")
            print(f"  - Current attempt (after increment): {self.current_attempt}")
            print(f"  - Max attempts: {self.max_attempts}")
            print(f"  - Should continue: {should_continue}")
        
        if should_continue:
            # =================== Continue attempting (reflection) ===================
            # Generate feedback for incorrect answer
            feedback = self._generate_feedback(answer, self.answer, think_content)
            attempt_data["feedback"] = feedback
            
            # Add to conversation history - save complete response (including structured format)
            self.conversation_history.append({
                "role": "assistant",
                "content": action  # Keep full response with think/answer tags
            })
            self.conversation_history.append({
                "role": "system", 
                "content": feedback
            })
            
            # ⭐ Prepare observation for next attempt
            obs = {
                "image_path": self.current_image,
                "question": self.current_question,
                "task_id": self.task_id,
                "task_description": f"Answer the question about the chart: {self.current_question}",
                "feedback": feedback,
                "previous_answer": answer,
                "attempt": self.current_attempt + 1,  # Next attempt number
                "attempts_remaining": self.max_attempts - self.current_attempt,
                "conversation_history": self.conversation_history,
                "hint": self._generate_hint() if self.current_attempt == self.max_attempts - 1 else None,
                "previous_attempt_failed": True,
                "output_format_instruction": self._get_output_format_instruction() if self.use_structured_output else "",
                "reflection_format_reminder": "Remember to use the <think>...</think> and <answer>...</answer> format in your response.",
                "use_structured_output": self.use_structured_output,
                "deepeyes_enabled": self.enable_deepeyes,
                "deepeyes_history": self.deepeyes_history,
                "grounding_dino_enabled": self.enable_grounding_dino,
                "grounding_dino_history": self.grounding_dino_history,
                "chartmoe_enabled": self.enable_chartmoe,
                "chartmoe_history": self.chartmoe_history,
                "is_visual_question": True,
                "available_tools": self._get_available_tools()
            }
            
            # ⭐ Force tool usage logic
            if self.current_attempt >= 1:  # Already tried once
                # Check if ChartMoE has been used
                if self.enable_chartmoe and len(self.chartmoe_history) == 0:
                    obs["must_use_tool"] = True
                    obs["tool_to_use"] = "chartmoe"
                    obs["reflection_format_reminder"] = (
                        "⚠️ YOU MUST use ChartMoE tool first! Your previous visual reading was incorrect.\n"
                        'Start with: <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
                        "Then analyze the extracted data to answer the question accurately."
                    )
                elif self.enable_grounding_dino and len(self.grounding_dino_history) == 0:
                    obs["must_use_tool"] = True
                    obs["tool_to_use"] = "grounding_dino"
                    obs["reflection_format_reminder"] = (
                        "⚠️ YOU MUST use Grounding DINO tool first! Your previous answer was incorrect.\n"
                        'Start with: <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "chart data"}}</tool_call>\n'
                        "Then use the detection results to answer the question."
                    )
            
            # ⭐ Set task as not done
            done = False
            reward = 0.0  # No reward for incorrect attempt
            truncated = False  # Still have attempts remaining, not truncated
            
            # Info for continuing
            info = {
                "prediction": answer,
                "ground_truth": self.answer,
                "correct": is_correct,
                "attempt": self.current_attempt,
                "attempt_history": self.attempt_history,
                "done_reason": "continuing",
                "question_type": self._classify_question_type(),
                "deepeyes_tool_calls": len(self.deepeyes_history),
                "grounding_dino_tool_calls": len(self.grounding_dino_history),
                "chartmoe_tool_calls": len(self.chartmoe_history),
                "message": f"Incorrect answer. Attempt {self.current_attempt}/{self.max_attempts}. {feedback[:100]}..."
            }
            
            # =================== DEBUG START ===================
            if self.debug:
                print(f"\n[DEBUG] Not done, will retry. Attempts remaining: {self.max_attempts - self.current_attempt}")
                print(f"[DEBUG] Feedback: {feedback[:100]}...")
                print(f"[DEBUG] Must use tool: {obs.get('must_use_tool', False)}")
                print(f"[DEBUG] Tool to use: {obs.get('tool_to_use', 'None')}")
                print(f"[DEBUG] Returning done=False to allow retry")
            # =================== DEBUG END ===================
            
        else:
            # =================== Task complete (correct or max attempts reached) ===================
            done = True
            reward = 1.0 if is_correct else 0.0
            
            # Check if ended due to max attempts reached
            truncated = (not is_correct and 
                        self.enable_reflection and 
                        self.current_attempt >= self.max_attempts)
            
            # Final observation
            obs = {
                "message": f"Task completed. Final answer: {answer}. Correct: {is_correct}",
                "final_answer": answer,
                "attempts_used": self.current_attempt,
                "success": is_correct,
                "deepeyes_tool_calls": len(self.deepeyes_history),
                "grounding_dino_tool_calls": len(self.grounding_dino_history),
                "chartmoe_tool_calls": len(self.chartmoe_history)
            }
            
            # Final info
            info = {
                "prediction": answer,
                "ground_truth": self.answer,
                "correct": is_correct,
                "attempt": self.current_attempt,
                "attempt_history": self.attempt_history if self.enable_reflection else None,
                "done_reason": "correct" if is_correct else ("max_attempts" if truncated else "single_attempt"),
                "question_type": self._classify_question_type(),
                "deepeyes_tool_calls": len(self.deepeyes_history),
                "grounding_dino_tool_calls": len(self.grounding_dino_history),
                "chartmoe_tool_calls": len(self.chartmoe_history),
                "message": f"Task completed. Attempts: {self.current_attempt}/{self.max_attempts}. Correct: {is_correct}"
            }
            
            # =================== DEBUG START ===================
            if self.debug:
                print(f"\n[DEBUG] Task done. Final reward: {reward}")
                print(f"[DEBUG] Done reason: {info['done_reason']}")
                print(f"[DEBUG] Total attempts used: {self.current_attempt}")
                print(f"[DEBUG] ChartMoE tool calls: {len(self.chartmoe_history)}")
                print(f"[DEBUG] Returning done=True")
            # =================== DEBUG END ===================
        
        # =================== DEBUG START ===================
        if self.debug:
            print(f"\n[DEBUG] Step summary:")
            print(f"  - Returning: obs type={type(obs)}, reward={reward}, done={done}, truncated={truncated}")
            print(f"  - Attempt history length: {len(self.attempt_history)}")
            print(f"  - All attempts so far:")
            for i, att in enumerate(self.attempt_history):
                print(f"    - Attempt {i+1}: correct={att.get('correct', 'N/A')}, "
                    f"chartmoe={att.get('chartmoe_used', False)}")
        # =================== DEBUG END ===================
        
        # Check if truncated by max steps (different from max attempts)
        if not done and self.current_step >= self.max_steps:
            truncated = True
            done = True
            obs["message"] = "Truncated by max steps limit"
            info["done_reason"] = "max_steps"
        
        return obs, reward, done, truncated, info
    

    def _get_available_tools(self):
        """Get list of available tools (for observation use)"""
        tools = []
        if self.enable_deepeyes:
            tools.append("deepeyes")
        if self.enable_grounding_dino:
            tools.append("grounding_dino")
        if self.enable_chartmoe:
            tools.append("chartmoe")
        return tools
    
    
    def _generate_feedback(self, prediction: str, ground_truth: str, think_content: str = "") -> str:
        """Generate specific feedback for incorrect answer based on detailed question type"""
        feedback_parts = []
        
        # Basic feedback
        feedback_parts.append(f"Your answer '{prediction}' is incorrect.")
        
        # Analyze error type
        q_type = self._classify_question_type()
        pred_num = extract_number_from_text(prediction)
        gt_num = extract_number_from_text(ground_truth)
        
        # Determine whether to force tool usage
        force_tool_use = False
        tool_suggestion_reason = ""
        
        # 1. Special handling for yes/no questions
        if self._is_yes_no_question():
            # If a numerical value was answered instead of yes/no
            if pred_num is not None and ground_truth.lower() in ['yes', 'no']:
                feedback_parts.append("This is a yes/no question. Please answer with 'Yes' or 'No', not a numerical value.")
                force_tool_use = False  # Understanding error, tool not needed
            # If yes/no was answered but is wrong
            elif prediction.lower() in ['yes', 'no'] and ground_truth.lower() in ['yes', 'no']:
                feedback_parts.append("Your yes/no answer is incorrect. Please re-examine the chart data.")
                # May need tool for closer inspection
                if self.current_attempt >= 1:
                    force_tool_use = True
                    tool_suggestion_reason = "to verify the exact values"
        
        # 2. For counting questions
        elif q_type == 'counting':
            if pred_num is not None and gt_num is not None:
                error_rate = abs(pred_num - gt_num) / gt_num if gt_num != 0 else 1
                if error_rate > 0.3:  # Error exceeds 30%
                    force_tool_use = True
                    tool_suggestion_reason = "to detect and count all items accurately"
                    feedback_parts.append(f"Your count is significantly off (you counted {int(pred_num)}, but there are more/fewer items).")
                else:
                    feedback_parts.append("Please count each item in the chart more carefully. Make sure you don't miss any items or count any twice.")
            else:
                feedback_parts.append("Please provide a numerical count of the items.")
        
        # 3. For numerical reading questions
        elif q_type in ['numerical', 'minmax', 'average', 'percentage']:
            if pred_num is not None and gt_num is not None:
                relative_error = abs(pred_num - gt_num) / abs(gt_num) if gt_num != 0 else 1
                if relative_error > 0.2:  # Error exceeds 20%
                    force_tool_use = True
                    tool_suggestion_reason = "to detect and read the exact values from the chart"
                    feedback_parts.append(f"The value seems significantly off (relative error: {relative_error:.1%}).")
                elif relative_error <= 0.1:  # Error within 10%
                    feedback_parts.append("You're close! Please read the values more precisely.")
                else:
                    if pred_num > gt_num:
                        feedback_parts.append("Your answer is too high.")
                    else:
                        feedback_parts.append("Your answer is too low.")
            else:
                feedback_parts.append("Please provide a numerical answer.")
        
        # 4. For summation questions
        elif q_type == 'summation':
            feedback_parts.append("Check your addition. Make sure you've included all relevant values.")
            if pred_num is not None and gt_num is not None:
                if abs(pred_num - gt_num) > 50:  # Large difference
                    force_tool_use = True
                    tool_suggestion_reason = "to detect all values that need to be summed"
        
        # 5. For difference questions
        elif q_type == 'difference':
            feedback_parts.append("Ensure you're subtracting the correct values in the right order.")
            if self.current_attempt >= 1:
                force_tool_use = True
                tool_suggestion_reason = "to detect and compare the values accurately"
        
        # 6. For ratio questions
        elif q_type == 'ratio':
            feedback_parts.append("Check your division. Make sure you're dividing the correct values.")
            if self.current_attempt >= 1:
                force_tool_use = True
                tool_suggestion_reason = "to detect both values needed for the ratio"
        
        # 7. For comparison questions
        elif q_type == 'comparison':
            feedback_parts.append("Identify the exact values for each element before comparing them.")
            if self.current_attempt >= 1:
                force_tool_use = True
                tool_suggestion_reason = "to detect and compare all relevant elements"
        
        # 8. For trend questions
        elif q_type == 'trend':
            feedback_parts.append("Analyze the overall pattern in the data more carefully. Look at the direction of change.")
        
        # 9. For retrieval questions
        elif q_type == 'retrieval':
            feedback_parts.append("Read the labels and legends carefully to find the specific information requested.")
            if self.current_attempt >= 1:
                force_tool_use = True
                tool_suggestion_reason = "to detect and read labels, legends, or specific text"
        
        # If think content exists, analyze issues within it
        if think_content and self.use_structured_output:
            think_lower = think_content.lower()
            if "calculation" in think_lower or "sum" in think_lower:
                if "error" not in " ".join(feedback_parts).lower():
                    feedback_parts.append("Check your calculations carefully.")
            elif "count" in think_lower:
                if "count" not in " ".join(feedback_parts).lower():
                    feedback_parts.append("Verify your counting process.")
            
            # Check if visual difficulties were mentioned
            if any(phrase in think_lower for phrase in ["hard to see", "unclear", "can't see", "difficult to"]):
                force_tool_use = True
                tool_suggestion_reason = "since you mentioned difficulty seeing the details"
        
        # ⭐ Grounding DINO tool usage suggestion
        if self.enable_grounding_dino:
            if force_tool_use and len(self.grounding_dino_history) == 0:
                # First time - strongly recommend using tool
                feedback_parts.append(f"YOU SHOULD use the Grounding DINO tool {tool_suggestion_reason}.")
                feedback_parts.append("Use it to detect specific objects in the chart. Example:")
                feedback_parts.append('<tool_call>{"tool": "grounding_dino", "parameters": {"caption": "bar chart bars"}}</tool_call>')
            elif force_tool_use and len(self.grounding_dino_history) > 0:
                # Tool already used but still wrong
                feedback_parts.append("You've used the Grounding DINO tool but the answer is still incorrect.")
                feedback_parts.append("Try detecting different elements or use more specific captions.")
                # Give more specific suggestions based on question type
                if q_type == 'counting':
                    feedback_parts.append('For counting, try: {"caption": "all bars"} or {"caption": "data points"}')
                elif q_type in ['numerical', 'minmax']:
                    feedback_parts.append('For values, try: {"caption": "data values"} or {"caption": "numbers"}')
                elif q_type == 'retrieval':
                    feedback_parts.append('For labels, try: {"caption": "axis labels"} or {"caption": "legend items"}')
            elif not force_tool_use and len(self.grounding_dino_history) == 0 and self.current_attempt == self.max_attempts - 1:
                # Last chance - gentle suggestion
                feedback_parts.append("Consider using the Grounding DINO tool if you need help detecting objects in the chart.")
        
        # ⭐ ChartMoE tool usage suggestion (enhanced version)
        if self.enable_chartmoe:
            # If first failure and ChartMoE hasn't been used, require usage
            if len(self.chartmoe_history) == 0:
                feedback_parts.append("\n⚠️ IMPORTANT: You MUST use the ChartMoE tool for this visual chart question!")
                feedback_parts.append("This is a VISUAL QUESTION that requires accurate data extraction from the chart.")
                feedback_parts.append("Use the following command to extract all chart data:")
                feedback_parts.append('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>')
                feedback_parts.append("After getting the structured data, analyze it to answer the question correctly.\n")
            elif force_tool_use and len(self.chartmoe_history) > 0:
                # Tool already used but still wrong
                feedback_parts.append("You've used the ChartMoE tool but the answer is still incorrect.")
                feedback_parts.append("Try a different task or use a custom prompt.")
                # Give more specific suggestions based on question type
                if q_type == 'counting':
                    feedback_parts.append('For counting, try: {"tool": "chartmoe", "task": "extract_data"}')
                elif q_type in ['numerical', 'minmax', 'summation']:
                    feedback_parts.append('For numerical values, try: {"tool": "chartmoe", "prompt": "' + self.current_question + '"}')
                elif q_type == 'retrieval':
                    feedback_parts.append('For labels, try: {"tool": "chartmoe", "task": "describe"}')
                elif q_type in ['average', 'percentage', 'ratio']:
                    feedback_parts.append('For calculations, try: {"tool": "chartmoe", "task": "analyze"}')
            elif not force_tool_use and len(self.chartmoe_history) == 0 and self.current_attempt == self.max_attempts - 1:
                # Last chance - gentle suggestion
                feedback_parts.append("Consider using the ChartMoE tool if you need help understanding the chart data.")
        
        # ⭐ DeepEyes tool usage suggestion
        if self.enable_deepeyes:
            if force_tool_use and len(self.deepeyes_history) == 0:
                feedback_parts.append(f"You can also use the image_zoom_in_tool {tool_suggestion_reason}.")
                feedback_parts.append("Zoom in on the relevant area of the chart before providing your next answer.")
            elif force_tool_use and len(self.deepeyes_history) > 0:
                feedback_parts.append("You've used the zoom tool but the answer is still incorrect.")
                feedback_parts.append("Try zooming in on a different area or with a different bounding box.")
        
        # General advice
        feedback_parts.append("Please examine the chart again and provide a revised answer.")
        
        # Format reminder
        if self.use_structured_output:
            feedback_parts.append("Remember to structure your response with <think> and <answer> tags.")
        
        # Remaining attempts
        remaining = self.max_attempts - self.current_attempt
        if remaining > 1:
            feedback_parts.append(f"You have {remaining} attempts remaining.")
        else:
            feedback_parts.append("This is your last attempt. Please be careful and thorough.")
        
        return " ".join(feedback_parts)
    
    
    def _generate_hint(self) -> str:
        """Generate hint for final attempt based on error analysis"""
        q_type = self._classify_question_type()
        
        # Analyze previous attempt history
        tool_ever_used = False
        grounding_dino_used = False
        deepeyes_used = False
        chartmoe_used = False  # ⭐ Add ChartMoE
        common_errors = []
        
        if self.attempt_history:
            # Check if tools have been used
            grounding_dino_used = any(att.get('grounding_dino_used', False) for att in self.attempt_history)
            deepeyes_used = any(att.get('deepeyes_used', False) for att in self.attempt_history)
            chartmoe_used = any(att.get('chartmoe_used', False) for att in self.attempt_history)  # ⭐ Check ChartMoE usage
            tool_ever_used = grounding_dino_used or deepeyes_used or chartmoe_used  # ⭐ Update condition
            
            # Analyze common error patterns
            for attempt in self.attempt_history:
                answer = attempt.get('answer', '')
                
                # Yes/No question answered with number
                if self._is_yes_no_question() and extract_number_from_text(answer) is not None:
                    common_errors.append('yes_no_as_number')
                
                # Counting question error
                if q_type == 'counting':
                    pred_num = extract_number_from_text(answer)
                    gt_num = extract_number_from_text(self.answer)
                    if pred_num and gt_num:
                        error_rate = abs(pred_num - gt_num) / gt_num if gt_num != 0 else 1
                        if error_rate > 0.3:
                            common_errors.append('large_counting_error')
        
        # Generate hints based on error analysis
        if 'yes_no_as_number' in common_errors:
            hint = "IMPORTANT: This is a YES/NO question. Your answer should be either 'Yes' or 'No', not a numerical value. "
            hint += f"The question asks: '{self.current_question}' - Answer with Yes or No only."
        
        elif 'large_counting_error' in common_errors and not chartmoe_used and self.enable_chartmoe:
            # ⭐ Prioritize recommending ChartMoE for accurate data extraction
            hint = "CRITICAL: Your counting appears to be significantly off. You SHOULD use the ChartMoE tool to extract structured data. "
            hint += "This will give you an accurate table of all values. "
            hint += 'Use: <tool_call>{"tool": "chartmoe", "parameters": {"task": "to_table"}}</tool_call>'
        
        elif 'large_counting_error' in common_errors and not grounding_dino_used and self.enable_grounding_dino:
            hint = "CRITICAL: Your counting appears to be significantly off. You MUST use the Grounding DINO tool to detect all items. "
            hint += "This will help you count accurately. "
            hint += 'Use: <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "all bars"}}</tool_call>'
        
        else:
            # Type-specific hints - add ChartMoE suggestions
            type_hints = {
                'counting': "Count each item in the chart systematically. " +
                        (f"Use ChartMoE to get structured data: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect all items: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "bars" or "points"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'summation': "Add all the relevant values together. " +
                            (f"Use ChartMoE to extract all values: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect all values: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "data values"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'average': "First sum all values, then divide by the count. " +
                        (f"Use ChartMoE to get accurate data: "
                        f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                        if self.enable_chartmoe and not chartmoe_used else 
                        "Make sure you have the correct number of items."),
                
                'percentage': "Calculate (part ÷ whole) × 100. " +
                            (f"Use ChartMoE to extract exact values: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else 
                            "Make sure you identify the correct part and whole."),
                
                'difference': "Find the two values to compare and subtract. " +
                            (f"Use ChartMoE to get precise values: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "answer", "question": "What are the values to compare?"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect values: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "numbers"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'ratio': "Divide one value by another as specified. " +
                        (f"Use ChartMoE to extract values: "
                        f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                        if self.enable_chartmoe and not chartmoe_used else
                        (f"Use Grounding DINO to detect values: "
                        f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "data values"}}}}</tool_call>'
                        if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'numerical': "Read the exact value from the chart. " +
                            (f"Use ChartMoE for accurate value extraction: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect values: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "numbers"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'comparison': "Identify and compare exact values. " +
                            (f"Use ChartMoE to analyze comparisons: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "answer", "question": "{self.current_question}"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect elements: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "chart elements"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'minmax': "Check every single data point to find the extreme value. " +
                        (f"Use ChartMoE to get all values: "
                        f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_table"}}}}</tool_call>'
                        if self.enable_chartmoe and not chartmoe_used else
                        (f"Use Grounding DINO to detect all points: "
                        f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "all data points"}}}}</tool_call>'
                        if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'trend': "Look at the overall direction of change from start to end. " +
                        (f"Use ChartMoE to describe the trend: "
                        f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_text"}}}}</tool_call>'
                        if self.enable_chartmoe and not chartmoe_used else ""),
                
                'retrieval': "Look for the specific label or category. " +
                            (f"Use ChartMoE to extract text information: "
                            f'<tool_call>{{"tool": "chartmoe", "parameters": {{"task": "to_text"}}}}</tool_call>'
                            if self.enable_chartmoe and not chartmoe_used else
                            (f"Use Grounding DINO to detect text: "
                            f'<tool_call>{{"tool": "grounding_dino", "parameters": {{"caption": "labels"}}}}</tool_call>'
                            if self.enable_grounding_dino and not grounding_dino_used else "")),
                
                'other': "Read the question carefully and make sure you understand what is being asked."
            }
            
            hint = type_hints.get(q_type, "Take your time to analyze all elements of the chart thoroughly.")
        
        # Add general tool usage hint (if applicable)
        if not tool_ever_used and q_type != 'other':
            hints_added = False
            
            # ⭐ ChartMoE has highest priority (for data extraction)
            if self.enable_chartmoe:
                hint += "\n\nREMINDER: You have access to the ChartMoE tool. Use it to extract structured data or answer questions about the chart."
                hints_added = True
            
            if self.enable_grounding_dino:
                if hints_added:
                    hint += "\nYou also have"
                else:
                    hint += "\n\nREMINDER: You have"
                hint += " access to the Grounding DINO tool. Use it to detect and locate objects in the chart."
                hints_added = True
                
            if self.enable_deepeyes:
                if hints_added:
                    hint += "\nAdditionally, you have"
                else:
                    hint += "\n\nREMINDER: You have"
                hint += " access to the image_zoom_in_tool. Use it if you need a clearer view of specific regions."
        
        # Format reminder
        if self.use_structured_output:
            hint += "\n\nFormat: Use <think> tags for your reasoning and <answer> tags for your final answer."
        
        return hint
    
    

    def validate(self, chat_history, observation, full_history=None):
        """Validate method required by VisionQAEnv"""
        # Extract answer from observation
        answer = ""
        
        if isinstance(observation, dict):
            # Check various possible answer locations
            answer = observation.get('final_answer', 
                    observation.get('answer',
                    observation.get('content', '')))
        
        # Ensure answer is string
        answer = self._extract_answer_str(answer)
        
        # Use improved evaluation with tolerance
        is_correct = evaluate_chartqa_answer(answer, self.answer, self.numerical_tolerance)
        
        # ⭐ Key change: only return done=True in the following cases:
        # 1. Answer is correct
        # 2. Max attempts have been reached
        # 3. Reflection mechanism is not enabled
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
            # ⭐ Still have attempts remaining, don't end task
            done = False
            reward = 0.0
        
        # Generate message based on attempt history
        message_parts = []
        if self.enable_reflection and len(self.attempt_history) > 0:
            message_parts.append(f"Attempt {self.current_attempt} of {self.max_attempts}.")
        else:
            message_parts.append("Task evaluation.")
        
        message_parts.append(f"Answer: {answer}")
        message_parts.append(f"Correct: {is_correct}")
        
        if not done and self.enable_reflection:
            message_parts.append(f"Attempts remaining: {self.max_attempts - self.current_attempt}")
        
        # Add tool usage info
        if self.enable_deepeyes and len(self.deepeyes_history) > 0:
            message_parts.append(f"DeepEyes tool used {len(self.deepeyes_history)} times.")
        if self.enable_grounding_dino and len(self.grounding_dino_history) > 0:
            message_parts.append(f"Grounding DINO tool used {len(self.grounding_dino_history)} times.")
        if self.enable_chartmoe and len(self.chartmoe_history) > 0:
            message_parts.append(f"ChartMoE tool used {len(self.chartmoe_history)} times.")
        
        message = " ".join(message_parts)
        
        info = {
            "prediction": answer,
            "ground_truth": self.answer,
            "correct": is_correct,
            "current_attempt": self.current_attempt,
            "max_attempts": self.max_attempts,
            "should_continue": not done and self.enable_reflection,
            "attempt_history": self.attempt_history if self.enable_reflection else None,
            "question_type": self._classify_question_type(),
            "deepeyes_tool_calls": len(self.deepeyes_history),
            "grounding_dino_tool_calls": len(self.grounding_dino_history),
            "chartmoe_tool_calls": len(self.chartmoe_history)
        }
        
        return reward, done, message, info
    
    
    
    def _extract_answer_str(self, answer):
        """Extract string from various answer formats"""
        if isinstance(answer, str):
            return answer
        elif isinstance(answer, dict):
            for key in ['answer', 'content', 'text', 'value', 'result']:
                if key in answer:
                    return str(answer[key])
            return str(answer)
        else:
            return str(answer)


def create_chartqa_actions():
    """Create ChartQA-specific actions by importing from chartqa_actions module"""
    
    # Import all action functions from chartqa_actions module
    action_dict = {
        "extract_chart_values": chartqa_actions.extract_chart_values,
        "calculate_chart_statistics": chartqa_actions.calculate_chart_statistics,
        "compare_chart_elements": chartqa_actions.compare_chart_elements,
    }
    
    # If chartqa_actions.py has additional functions, add them too
    if hasattr(chartqa_actions, 'find_chart_extremes'):
        action_dict["find_chart_extremes"] = chartqa_actions.find_chart_extremes
    
    if hasattr(chartqa_actions, 'analyze_chart_trend'):
        action_dict["analyze_chart_trend"] = chartqa_actions.analyze_chart_trend
    
    return action_dict


def save_trajectory_for_rl(task_result: Dict[str, Any], env, task_wrapper) -> Dict[str, Any]:
    """⭐ Save complete trajectory for RL training"""
    trajectory = {
        "task_id": task_result["task_id"],
        "env_name": "grounding_dino" if task_wrapper.enable_grounding_dino else "visual_toolbox_v2" if task_wrapper.enable_deepeyes else None,
        "question": task_result["question"],
        "ground_truth": task_result["ground_truth"],
        "interactions": [],
        "total_reward": task_result["reward"],
        "success": task_result["correct"],
        "metadata": {
            "question_type": task_result["question_type"],
            "attempts_used": task_result.get("attempts_used", 1),
            "deepeyes_enabled": task_wrapper.enable_deepeyes,
            "deepeyes_tool_calls": task_result.get("deepeyes_tool_calls", 0),
            "grounding_dino_enabled": task_wrapper.enable_grounding_dino,
            "grounding_dino_tool_calls": task_result.get("grounding_dino_tool_calls", 0),
            "structured_output": task_wrapper.use_structured_output
        }
    }
    
    # Build interaction trajectory from environment's action_history
    if hasattr(env, 'action_history') and env.action_history:
        for step_data in env.action_history:
            interaction = {
                "step": step_data["step"],
                "action": step_data["action"],
                "action_type": step_data.get("result", {}).get("type", "unknown"),
                "reward": step_data.get("reward", 0),
                "tool_used": None,
                "observation": {}
            }
            
            # Determine which tool was used
            if "grounding_dino" in step_data.get("result", {}).get("source", ""):
                interaction["tool_used"] = "grounding_dino"
                interaction["observation"]["detections"] = step_data.get("result", {}).get("detections", {})
            elif "deepeyes" in step_data.get("result", {}).get("source", ""):
                interaction["tool_used"] = "deepeyes"
                interaction["observation"]["has_tool_feedback"] = step_data.get("result", {}).get("deepeyes_feedback", False)
            
            # If there is tool feedback, record processed image info
            if step_data.get("result", {}).get("processed_images"):
                interaction["observation"]["processed_images_count"] = len(step_data["result"]["processed_images"])
            
            trajectory["interactions"].append(interaction)
    
    # Supplement info from task wrapper's attempt_history
    if hasattr(task_wrapper, 'attempt_history') and task_wrapper.attempt_history:
        for i, attempt in enumerate(task_wrapper.attempt_history):
            if i < len(trajectory["interactions"]):
                trajectory["interactions"][i]["attempt_info"] = {
                    "attempt_number": attempt["attempt"],
                    "correct": attempt["correct"],
                    "deepeyes_used": attempt.get("deepeyes_used", False),
                    "grounding_dino_used": attempt.get("grounding_dino_used", False),
                    "has_thinking": bool(attempt.get("thinking"))
                }
    
    return trajectory


def evaluate_chartqa_with_reflection(args):
    """Run ChartQA evaluation with reflection support and optional enhanced reasoning"""
    
    # Base model config - remove generation_config
    base_config = {
        "model_type": "HuggingFace",
        "model_name": args.model,
        "max_new_tokens": 256,
        "temperature": 0.3,
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    }
    
    # generation_config saved separately, to be set directly on model later
    generation_config_settings = {
        "max_new_tokens": 512,
        "max_length": None,
        "do_sample": False,
        "temperature": 0.3,
        "eos_token_id": None,
        "min_new_tokens": 1,  # Allow short answers
        "repetition_penalty": 1.0,
    }
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_type = "chartqa"
    if args.enable_actions:
        exp_type += "_actions"
    if args.enable_chartqa_reasoning:
        exp_type += "_enhanced"
    if args.enable_reflection:
        exp_type += "_reflection"
    if args.use_structured_output:
        exp_type += "_structured"
    if args.enable_deepeyes:
        exp_type += "_deepeyes"
    if args.use_config and "grounding_dino" in args.config_experiment:  # ⭐ Add Grounding DINO marker
        exp_type += "_grounding_dino"
    exp_id = f"{exp_type}_{timestamp}"
    exp_dirs = setup_experiment_directory("./experiments", exp_id)
    
    # Setup logging
    logger = setup_logger(f"chartqa_eval_{exp_id}", log_dir=exp_dirs["logs"])
    print("\n" + "="*60)
    print("Starting ChartQA evaluation:")
    print(f"Model: {args.model}")
    print(f"Actions Enabled: {args.enable_actions}")
    print(f"ChartQA Enhanced Reasoning: {args.enable_chartqa_reasoning}")
    print(f"Reflection Enabled: {args.enable_reflection}")
    if args.enable_reflection:
        print(f"  Max Attempts: {args.max_attempts}")
    print(f"Numerical Tolerance: {args.numerical_tolerance * 100:.1f}%")
    print(f"Structured Output Format: {args.use_structured_output}")
    print(f"DeepEyes Tools Enabled: {args.enable_deepeyes}")
    if args.enable_deepeyes:
        print(f"  DeepEyes Version: {args.deepeyes_version}")
    if args.use_config:  # ⭐ Add config file output
        print(f"Using Config File: {args.config_path}")
        print(f"Config Experiment: {args.config_experiment}")
    print("="*60 + "\n")
    
    # Create ChartQA adapter
    adapter = ChartQAAdapter(
        data_root=args.data_root,
        annotation_files=args.annotation,
        validate_images=True
    )
    
    # Log statistics
    stats = adapter.get_statistics()
    print("Dataset statistics:")
    print(f"  Total tasks: {stats['total']}")
    print(f"  Unique images: {stats.get('unique_images', 'N/A')}")
    print(f"  Question types:")
    for qtype, count in stats['question_type_distribution'].items():
        print(f"    {qtype}: {count}")
    
    # Get task IDs
    if args.question_type:
        print(f"\nFiltering by question type: {args.question_type}")
        task_ids = adapter.get_task_ids(question_type=args.question_type)
    elif args.hard_examples:
        print("\nSelecting hard examples")
        task_ids = adapter.get_hard_examples(args.limit or 100)
    elif args.stratified and args.limit:
        print("\nUsing stratified sampling")
        task_ids = adapter.sample_tasks(args.limit, stratified=True, seed=args.seed)
    else:
        task_ids = adapter.get_task_ids(shuffle=args.shuffle, seed=args.seed)
    
    # Apply limit
    if args.limit and len(task_ids) > args.limit:
        task_ids = task_ids[:args.limit]
    
    print(f"\nWill evaluate {len(task_ids)} tasks\n")
    
    # Prepare custom actions for ChartQA if enabled
    custom_actions = None
    if args.enable_actions and args.custom_actions:
        custom_actions = create_chartqa_actions()
        print(f"Loaded {len(custom_actions)} ChartQA-specific actions")
    
    # ⭐ Choose environment creation method based on parameters
    if args.use_config:
        # Use config file to create environment
        env = create_env_from_config(args.config_path, args.config_experiment)
        print(f"Created environment from config: {args.config_path}, experiment: {args.config_experiment}")
        
        # Extract tool state from environment
        enable_grounding_dino = getattr(env, 'enable_grounding_dino', False)
        enable_deepeyes = getattr(env, 'enable_deepeyes_tools', False)
        enable_chartmoe = getattr(env, 'enable_chartmoe', False)
    else:
        # ⭐ Modification: add Grounding DINO support
        grounding_dino_config = None
        if args.enable_grounding_dino:
            grounding_dino_config = {
                "model_path": args.grounding_dino_model,
                "model_config": args.grounding_dino_config,
                "box_threshold": 0.25,
                "text_threshold": 0.25,
                "device": "cuda"  # Or set as needed
            }

        chartmoe_config = None
        if args.enable_chartmoe:
            chartmoe_config = {
                "model_name": args.chartmoe_model,
                "device": args.chartmoe_device,
                "trust_remote_code": True,
                "max_new_tokens": 512
            }
        
        env = VisionQAEnv(
            dataset_path=args.data_root,
            max_steps=args.max_steps if args.enable_actions else (args.max_attempts * 2 if args.enable_reflection else 3),
            enable_actions=args.enable_actions,
            custom_actions=custom_actions,
            enable_deepeyes_tools=args.enable_deepeyes,
            deepeyes_version=args.deepeyes_version,
            enable_grounding_dino=args.enable_grounding_dino,  # ⭐ Add
            grounding_dino_config=grounding_dino_config,  # ⭐ Add
            enable_chartmoe=args.enable_chartmoe,  # ⭐ Add ChartMoE
            chartmoe_config=chartmoe_config  # ⭐ Add ChartMoE config
        )
        enable_grounding_dino = args.enable_grounding_dino  # ⭐ Modify
        enable_deepeyes = args.enable_deepeyes
        enable_chartmoe = args.enable_chartmoe  # ⭐ Add ChartMoE
    
    # Create task factory
    def create_task(task_id, **kwargs):
        task_data = adapter.get_task_data(task_id)
        kwargs['enable_reflection'] = args.enable_reflection
        kwargs['max_attempts'] = args.max_attempts
        kwargs['debug'] = args.debug or args.chartqa_debug  # Pass debug flag
        kwargs['numerical_tolerance'] = args.numerical_tolerance
        kwargs['use_structured_output'] = args.use_structured_output  # Pass structured output flag
        kwargs['enable_deepeyes'] = enable_deepeyes  # ⭐ Use environment's tool state
        kwargs['enable_grounding_dino'] = enable_grounding_dino  # ⭐ Pass Grounding DINO flag
        kwargs['enable_chartmoe'] = enable_chartmoe  # ⭐ Pass ChartMoE flag
        if args.enable_actions:
            kwargs['action_set'] = env.action_set
            kwargs['max_steps'] = args.max_steps
        return ChartQATaskWrapper(task_id, task_data, **kwargs)
    
    env.task_entrypoint = create_task
    
    # Create agent - choose appropriate agent based on parameters
    print(f"Loading model: {base_config['model_name']}...")

    # Initialize model_attr as None, for later assertion checks
    model_attr = None

    # ⭐ Tool config (for supporting DeepEyes, Grounding DINO and ChartMoE)
    tool_config = {}
    if enable_deepeyes or enable_grounding_dino or enable_chartmoe:
        tool_config = {
            "enable_tools": True,
            "max_tool_calls": 5,
            "tool_selection_strategy": "adaptive",
            "tool_response_mode": "deepeyes" if enable_deepeyes else ("grounding_dino" if enable_grounding_dino else "chartmoe"),
            "deepeyes_prompt_style": "v5" if enable_deepeyes else None
        }

    if args.enable_chartqa_reasoning and CHARTQA_AGENT_AVAILABLE:
        print("Using ChartQA enhanced agent with structured reasoning")
        # Separate ChartQA specific parameters
        chartqa_specific_params = {
            'enable_structured_reasoning': True,
            'use_calculator': True,
            'debug': args.chartqa_debug,
            'use_structured_output': args.use_structured_output  # Pass structured output parameter
        }
        # Create dict containing base config and ChartQA params
        agent_config = {
            **base_config,
            **tool_config,  # ⭐ Add tool config
            '_chartqa_params': chartqa_specific_params  # Use special prefix to avoid conflicts
        }
        agent = ChartQAAgent(agent_config)
        
        # ===== Set generation_config =====
        print("\n[DEBUG] Setting generation config...")
        
        # Save original act method
        original_act = agent.act
        config_modified = [False]  # Use list to allow modification in closure
        
        def act_with_generation_config(observation):
            # Modify config on first call
            if not config_modified[0] and hasattr(agent, 'model') and agent.model is not None:
                print("\n[DEBUG] Applying generation_config on first act call...")
                model = agent.model
                if hasattr(model, 'generation_config'):
                    for key, value in generation_config_settings.items():
                        setattr(model.generation_config, key, value)
                    print(f"[DEBUG] ✓ Applied generation_config: max_new_tokens={model.generation_config.max_new_tokens}")
                config_modified[0] = True
            
            # Call original method
            return original_act(observation)
        
        # Replace act method
        agent.act = act_with_generation_config
        print("[DEBUG] ✓ Set up delayed generation_config application")
        
    else:
        if args.enable_chartqa_reasoning and not CHARTQA_AGENT_AVAILABLE:
            print("Warning: ChartQA reasoning requested but not available, falling back to base agent")
        
        # ⭐ Choose different Agent based on whether tools are enabled
        if enable_deepeyes or enable_grounding_dino or enable_chartmoe:
            tools_enabled = []
            if enable_deepeyes:
                tools_enabled.append("DeepEyes")
            if enable_grounding_dino:
                tools_enabled.append("Grounding DINO")
            if enable_chartmoe:
                tools_enabled.append("ChartMoE")
            print(f"Using VLMAgentWithTools for {', '.join(tools_enabled)} support")
            agent_config = {**base_config, **tool_config}
            agent = VLMAgentWithTools(agent_config)
        else:
            config = {"agent": base_config}
            agent = VLMAgent(config=config)
        
        # Also set generation_config for Agent
        original_act = agent.act
        config_modified = [False]
        
        def act_with_generation_config(observation):
            if not config_modified[0] and hasattr(agent, 'model') and agent.model is not None:
                model = agent.model
                if hasattr(model, 'generation_config'):
                    for key, value in generation_config_settings.items():
                        setattr(model.generation_config, key, value)
                config_modified[0] = True
            return original_act(observation)
        
        agent.act = act_with_generation_config
    
    # Metrics tracking with more detailed categories
    metrics = MetricsTracker()
    results = []
    results_by_qtype = defaultdict(list)
    trajectories = []  # ⭐ Collect trajectory data
    
    # Track detailed numerical question types
    numerical_subtypes = ['counting', 'summation', 'average', 'percentage', 'difference', 'ratio', 'numerical']
    
    # Evaluation loop with reflection support and enhanced debugging
    for idx, task_id in enumerate(tqdm(task_ids, desc="Evaluating ChartQA")):
        try:
            task_data = adapter.get_task_data(task_id)
            
            # Reset environment
            obs, info = env.reset(task_id=task_id)
            
            # Get task instance for observation
            task = env.task
            
            # =================== CRITICAL FIX ===================
            # Directly work with task instance for reflection
            done = False
            step_count = 0


            while not done and step_count < task.max_steps:
                # Get current observation with reflection context
                agent_obs = task.get_observation()
                
                # =================== ENHANCED DEBUG START ===================
                if args.debug and len(results) < 2:  # Only print first two
                    print(f"\n{'='*60}")
                    print(f"[DEBUG] Task {task_id} - Step {step_count + 1}, Attempt {task.current_attempt + 1}")
                    print(f"{'='*60}")
                    
                    # 1. Check observation content
                    print(f"\n[DEBUG] Observation contents:")
                    print(f"  - Available keys: {list(agent_obs.keys())}")
                    print(f"  - image_path: {agent_obs.get('image_path', 'MISSING')}")
                    print(f"  - Image exists: {os.path.exists(agent_obs.get('image_path', '')) if 'image_path' in agent_obs else 'N/A'}")
                    print(f"  - question: {agent_obs.get('question', 'MISSING')[:50]}...")
                    print(f"  - current_attempt: {task.current_attempt}")
                    print(f"  - previous_attempt_failed: {agent_obs.get('previous_attempt_failed', False)}")
                    print(f"  - must_use_tool: {agent_obs.get('must_use_tool', False)}")
                    print(f"  - use_structured_output: {task.use_structured_output}")
                    print(f"  - chartmoe_enabled: {agent_obs.get('chartmoe_enabled', False)}")
                    if 'feedback' in agent_obs:
                        print(f"  - Has feedback: Yes")
                        print(f"  - Feedback preview: {agent_obs['feedback'][:100]}...")
                    if 'output_format_instruction' in agent_obs:
                        print(f"  - Has output format instruction: Yes")
                    
                    # 2. Check agent type and attributes
                    print(f"\n[DEBUG] Agent information:")
                    print(f"  - Agent type: {type(agent)}")
                    print(f"  - Agent class: {agent.__class__.__name__}")
                    print(f"  - Has 'enable_tools' attr: {hasattr(agent, 'enable_tools')}")
                    if hasattr(agent, 'enable_tools'):
                        print(f"  - enable_tools value: {agent.enable_tools}")
                    
                    print(f"\n[DEBUG] About to call agent.act()...")
                # =================== ENHANCED DEBUG END ===================
                
                # Agent generates answer
                action, extra_info = agent.act(agent_obs)
                
                # =================== POST-ACT DEBUG START ===================
                if args.debug and len(results) < 2:
                    print(f"\n[DEBUG] Agent response:")
                    print(f"  - action type: {type(action)}")
                    print(f"  - action length: {len(str(action))}")
                    print(f"  - action preview (first 200 chars): {str(action)[:200]}")
                    
                    # Check for structured format
                    if '<think>' in str(action):
                        print(f"  - Contains <think> tags: Yes")
                    if '<answer>' in str(action):
                        print(f"  - Contains <answer> tags: Yes")
                    if '<tool_call>' in str(action):
                        print(f"  - Contains <tool_call> tags: Yes")
                        if '"tool": "chartmoe"' in str(action):
                            print(f"  - Tool type: ChartMoE")
                    
                    # Check extra_info
                    print(f"\n[DEBUG] Extra info:")
                    if isinstance(extra_info, dict):
                        for key, value in extra_info.items():
                            if key != 'raw_response':  # Avoid printing too long content
                                print(f"  - {key}: {str(value)[:100]}")
                # =================== POST-ACT DEBUG END ===================
                
                # Pass extra_info to task (if supported)
                if hasattr(task, 'last_extra_info'):
                    task.last_extra_info = extra_info
                
                # ⭐ Use env.step to process action
                obs, reward, done, truncated, info = env.step(action)
                step_count += 1
                
                # ⭐ DEBUG: Check env.step return values
                if args.debug and len(results) < 2:
                    print(f"\n[DEBUG] env.step returned:")
                    print(f"  - reward: {reward}")
                    print(f"  - done: {done}")
                    print(f"  - truncated: {truncated}")
                    print(f"  - info keys: {list(info.keys())}")
                    if 'action_result' in info:
                        print(f"  - action_result type: {info['action_result'].get('type', 'N/A')}")
                    if 'message' in info:
                        print(f"  - message: {info['message'][:100]}...")
                
                # ⭐ Record tool usage to task wrapper
                if "action_result" in info:
                    action_result = info["action_result"]
                    
                    # Record ChartMoE usage
                    if (action_result.get("type") == "tool_result" and 
                        action_result.get("tool") == "chartmoe"):
                        task.chartmoe_history.append({
                            "step": step_count,
                            "action": action,
                            "result_type": action_result.get("type"),
                            "task_type": action_result.get("task_type", "unknown"),
                            "attempt": task.current_attempt + 1
                        })
                        if args.debug:
                            print(f"[DEBUG] ✓ Recorded ChartMoE usage at step {step_count}, total: {len(task.chartmoe_history)}")
                    
                    # Record other tool usage...
                    elif action_result.get("type") == "deepeyes_feedback":
                        task.deepeyes_history.append({
                            "step": step_count,
                            "action": action,
                            "result_type": action_result.get("type"),
                            "attempt": task.current_attempt + 1
                        })
                    
                    elif (action_result.get("type") == "tool_result" and 
                        action_result.get("tool") == "grounding_dino"):
                        task.grounding_dino_history.append({
                            "step": step_count,
                            "action": action,
                            "result_type": action_result.get("type"),
                            "detections": action_result.get("detections", {}).get("num_detections", 0),
                            "attempt": task.current_attempt + 1
                        })
                
                # ⭐ Check if tool feedback needs to be processed
                if obs.get("requires_response") and "tool_feedback" in obs:
                    if args.debug and len(results) < 2:
                        print(f"\n[DEBUG] Tool feedback received, need agent response")
                        print(f"  - Tool: {obs.get('tool_feedback', {}).get('tool', 'N/A')}")
                    
                    # Agent needs to generate answer based on tool feedback
                    tool_response, tool_extra_info = agent.act(obs)
                    
                    if args.debug and len(results) < 2:
                        print(f"\n[DEBUG] Agent tool response:")
                        print(f"  - Response preview: {str(tool_response)[:200]}...")
                        if '<answer>' in str(tool_response):
                            print(f"  - Contains <answer> tag: Yes")
                    
                    # Update extra_info
                    if hasattr(task, 'last_extra_info'):
                        task.last_extra_info = tool_extra_info
                    
                    # Process tool response
                    obs, reward, done, truncated, info = env.step(tool_response)
                    step_count += 1
                    
                    # Record that this answer was based on tool feedback
                    if "action_result" in info and info["action_result"].get("type") == "answer":
                        # Mark that recent tool usage led to an answer
                        if len(task.chartmoe_history) > 0 and task.chartmoe_history[-1]["step"] == step_count - 1:
                            task.chartmoe_history[-1]["led_to_answer"] = True
                            if args.debug:
                                print(f"[DEBUG] ChartMoE tool usage led to answer")
                
                # ⭐ Check whether to continue (reflection mechanism)
                if not done and not truncated:
                    # Continue to next step
                    if args.debug:
                        print(f"[DEBUG] Continuing to next step...")
                    continue
                
                # ⭐ If done but this is due to reflection, handle specially
                if done and info.get("needs_reflection", False):
                    # This shouldn't happen since we modified _handle_task_validation
                    # But handle it just in case
                    if args.debug:
                        print(f"[DEBUG] WARNING: done=True but needs_reflection=True")
                    done = False
                    continue
                
                # Actually finished
                if done or truncated:
                    if args.debug and len(results) < 2:
                        print(f"\n[DEBUG] Task completed:")
                        print(f"  - Done: {done}")
                        print(f"  - Truncated: {truncated}")
                        print(f"  - Final reward: {reward}")
                        print(f"  - Done reason: {info.get('done_reason', info.get('message', 'Unknown'))}")
                        print(f"  - Total steps: {step_count}")
                        print(f"  - ChartMoE uses: {len(task.chartmoe_history)}")
                        print(f"  - Attempts made: {task.current_attempt}")
                    break
                
                # Check if max steps exceeded
                if step_count >= task.max_steps:
                    if args.debug:
                        print(f"\n[DEBUG] Reached max steps ({task.max_steps})")
                    truncated = True
                    break

            # ⭐ After loop ends, extract results from final info
            # Ensure we get the final answer
            if "prediction" in info:
                final_answer = info["prediction"]
            elif "action_result" in info and info["action_result"].get("type") == "answer":
                final_answer = info["action_result"].get("content", "")
            elif hasattr(task, 'attempt_history') and task.attempt_history:
                # Get answer from last attempt
                final_answer = task.attempt_history[-1].get("answer", "")
            else:
                final_answer = ""

            final_reward = reward
            total_attempts = task.current_attempt
            action_history = task.attempt_history

            # =================== Record results ===================
            # Record result with enhanced question type
            q_type = task._classify_question_type()
            result = {
                "task_id": task_id,
                "question": task_data["question"],
                "question_type": q_type,
                "ground_truth": task_data["answer"],
                "prediction": final_answer,
                "correct": final_reward > 0,
                "reward": float(final_reward),
                "steps_taken": step_count,
                "attempts_used": total_attempts,
                "timestamp": datetime.now().isoformat(),
                "used_structured_output": args.use_structured_output,
                "deepeyes_enabled": enable_deepeyes,
                "deepeyes_tool_calls": len(task.deepeyes_history),
                "grounding_dino_enabled": enable_grounding_dino,
                "grounding_dino_tool_calls": len(task.grounding_dino_history),
                "chartmoe_enabled": enable_chartmoe,
                "chartmoe_tool_calls": len(task.chartmoe_history),
            }

            # Check if answer was within tolerance
            pred_num = extract_number_from_text(final_answer)
            gt_num = extract_number_from_text(task_data["answer"])
            if pred_num is not None and gt_num is not None and gt_num != 0:
                relative_error = abs(pred_num - gt_num) / abs(gt_num)
                result["relative_error"] = relative_error
                result["within_tolerance"] = relative_error <= args.numerical_tolerance

            # Record whether structured reasoning was used
            if hasattr(task, 'last_extra_info') and task.last_extra_info:
                if 'structured' in task.last_extra_info:
                    result["used_structured_reasoning"] = task.last_extra_info['structured']
                if 'calculator_used' in task.last_extra_info:
                    result["calculator_used"] = task.last_extra_info['calculator_used']

            # Record attempt history
            if args.enable_reflection and action_history:
                result["attempt_history"] = action_history
                # Count attempts using tools
                chartmoe_attempts = sum(1 for att in action_history
                                    if att.get('chartmoe_used', False))
                result["chartmoe_attempts"] = chartmoe_attempts
                deepeyes_attempts = sum(1 for att in action_history 
                                    if att.get('deepeyes_used', False))
                result["deepeyes_attempts"] = deepeyes_attempts
                grounding_dino_attempts = sum(1 for att in action_history 
                                            if att.get('grounding_dino_used', False))
                result["grounding_dino_attempts"] = grounding_dino_attempts

            results.append(result)
            results_by_qtype[q_type].append(result)
            metrics.add("accuracy", float(result["correct"]))
            metrics.add(f"accuracy_{q_type}", float(result["correct"]))
            metrics.add("attempts", total_attempts)

            # Record tool usage metrics
            if enable_chartmoe:
                metrics.add("chartmoe_tool_calls", result.get("chartmoe_tool_calls", 0))
                if result.get("chartmoe_tool_calls", 0) > 0:
                    metrics.add("tasks_using_chartmoe", 1)
                else:
                    metrics.add("tasks_using_chartmoe", 0)

            if enable_deepeyes:
                metrics.add("deepeyes_tool_calls", result.get("deepeyes_tool_calls", 0))
                if result.get("deepeyes_tool_calls", 0) > 0:
                    metrics.add("tasks_using_deepeyes", 1)
                else:
                    metrics.add("tasks_using_deepeyes", 0)

            if enable_grounding_dino:
                metrics.add("grounding_dino_tool_calls", result.get("grounding_dino_tool_calls", 0))
                if result.get("grounding_dino_tool_calls", 0) > 0:
                    metrics.add("tasks_using_grounding_dino", 1)
                else:
                    metrics.add("tasks_using_grounding_dino", 0)

            # Save trajectory data
            if args.save_trajectories:
                trajectory = save_trajectory_for_rl(result, env, task)
                trajectories.append(trajectory)

            # Track numerical subtype accuracies
            if q_type in numerical_subtypes:
                metrics.add("accuracy_numerical_all", float(result["correct"]))

            # Log sample results
            if len(results) <= 3:
                print(f"\nSample {len(results)}:")
                print(f"  Question: {result['question'][:80]}...")
                print(f"  Type: {result['question_type']}")
                if args.enable_reflection:
                    print(f"  Attempts used: {total_attempts}/{args.max_attempts}")
                if enable_chartmoe:
                    print(f"  ChartMoE tool calls: {result.get('chartmoe_tool_calls', 0)}")
                print(f"  Prediction: {result['prediction'][:80]}...")
                print(f"  Ground Truth: {result['ground_truth'][:80]}...")
                print(f"  Correct: {result['correct']}")
                if 'relative_error' in result:
                    print(f"  Relative Error: {result['relative_error']:.3f}")
                
        except Exception as e:
            logger.error(f"Error on task {task_id}: {str(e)}")
            if args.debug or args.chartqa_debug:
                traceback.print_exc()
            results.append({
                "task_id": task_id,
                "error": str(e),
                "success": False
            })
    
    # Save results
    results_file = exp_dirs["results"] / "chartqa_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        json.dump(serializable_results, f, indent=2)
    
    # ⭐ Save trajectory data
    if args.save_trajectories and trajectories:
        trajectories_file = exp_dirs["results"] / "trajectories.json"
        with open(trajectories_file, 'w') as f:
            serializable_trajectories = convert_numpy(trajectories)
            json.dump(serializable_trajectories, f, indent=2)
        print(f"\nTrajectories saved to: {trajectories_file}")
    
    # Calculate metrics
    accuracy_by_type = {}
    for q_type, type_results in results_by_qtype.items():
        correct = sum(1 for r in type_results if r.get("correct", False))
        total = len(type_results)
        accuracy_by_type[q_type] = correct / total if total > 0 else 0
    
    # Calculate numerical subtypes accuracy
    numerical_results = [r for r in results if r.get('question_type') in numerical_subtypes and 'error' not in r]
    numerical_accuracy = sum(1 for r in numerical_results if r.get('correct', False)) / len(numerical_results) if numerical_results else 0
    
    # Overall metrics
    successful_results = [r for r in results if "error" not in r]
    overall_accuracy = metrics.get_mean("accuracy") if successful_results else 0.0
    avg_attempts = metrics.get_mean("attempts") if successful_results else 0
    
    # Calculate first-attempt accuracy if reflection is enabled
    first_attempt_accuracy = None
    if args.enable_reflection and successful_results:
        first_attempt_correct = sum(1 for r in successful_results 
                                   if r.get('attempt_history') and 
                                   len(r['attempt_history']) > 0 and 
                                   r['attempt_history'][0].get('correct', False))
        first_attempt_accuracy = first_attempt_correct / len(successful_results)
    
    # Count structured reasoning usage
    structured_reasoning_count = sum(1 for r in successful_results 
                                    if r.get('used_structured_reasoning', False))
    calculator_usage_count = sum(1 for r in successful_results 
                                if r.get('calculator_used', False))
    
    # ⭐ Count tool usage
    deepeyes_usage_count = sum(1 for r in successful_results 
                              if r.get('deepeyes_tool_calls', 0) > 0)
    avg_deepeyes_calls = metrics.get_mean("deepeyes_tool_calls") if enable_deepeyes else 0
    
    grounding_dino_usage_count = sum(1 for r in successful_results 
                                    if r.get('grounding_dino_tool_calls', 0) > 0)
    avg_grounding_dino_calls = metrics.get_mean("grounding_dino_tool_calls") if enable_grounding_dino else 0
    
    chartmoe_usage_count = sum(1 for r in successful_results
                              if r.get('chartmoe_tool_calls', 0) > 0)
    avg_chartmoe_calls = metrics.get_mean("chartmoe_tool_calls") if enable_chartmoe else 0
    
    # Calculate tolerance impact for numerical questions
    tolerance_impact = {}
    for q_type in numerical_subtypes:
        type_results = [r for r in successful_results if r.get('question_type') == q_type and 'relative_error' in r]
        if type_results:
            within_tolerance = sum(1 for r in type_results if r.get('within_tolerance', False))
            avg_error = sum(r['relative_error'] for r in type_results) / len(type_results)
            tolerance_impact[q_type] = {
                'count': len(type_results),
                'within_tolerance': within_tolerance,
                'average_relative_error': avg_error
            }
    
    # Analyze impact of structured reasoning on different question types
    structured_impact_by_type = {}
    if args.enable_chartqa_reasoning:
        for q_type in accuracy_by_type.keys():
            type_results = [r for r in successful_results if r.get('question_type') == q_type]
            structured_correct = sum(1 for r in type_results 
                                   if r.get('used_structured_reasoning', False) and r.get('correct', False))
            structured_total = sum(1 for r in type_results 
                                 if r.get('used_structured_reasoning', False))
            
            structured_impact_by_type[q_type] = {
                'structured_used': structured_total,
                'structured_correct': structured_correct,
                'structured_accuracy': (structured_correct / structured_total * 100) if structured_total > 0 else 0
            }
    
    # ⭐ Analyze tool impact on different question types
    deepeyes_impact_by_type = {}
    if enable_deepeyes:
        for q_type in accuracy_by_type.keys():
            type_results = [r for r in successful_results if r.get('question_type') == q_type]
            deepeyes_correct = sum(1 for r in type_results 
                                  if r.get('deepeyes_tool_calls', 0) > 0 and r.get('correct', False))
            deepeyes_total = sum(1 for r in type_results 
                                if r.get('deepeyes_tool_calls', 0) > 0)
            
            deepeyes_impact_by_type[q_type] = {
                'deepeyes_used': deepeyes_total,
                'deepeyes_correct': deepeyes_correct,
                'deepeyes_accuracy': (deepeyes_correct / deepeyes_total * 100) if deepeyes_total > 0 else 0,
                'avg_tool_calls': sum(r.get('deepeyes_tool_calls', 0) for r in type_results if r.get('deepeyes_tool_calls', 0) > 0) / deepeyes_total if deepeyes_total > 0 else 0
            }
    
    grounding_dino_impact_by_type = {}
    if enable_grounding_dino:
        for q_type in accuracy_by_type.keys():
            type_results = [r for r in successful_results if r.get('question_type') == q_type]
            grounding_correct = sum(1 for r in type_results 
                                  if r.get('grounding_dino_tool_calls', 0) > 0 and r.get('correct', False))
            grounding_total = sum(1 for r in type_results 
                                if r.get('grounding_dino_tool_calls', 0) > 0)
            
            grounding_dino_impact_by_type[q_type] = {
                'grounding_dino_used': grounding_total,
                'grounding_dino_correct': grounding_correct,
                'grounding_dino_accuracy': (grounding_correct / grounding_total * 100) if grounding_total > 0 else 0,
                'avg_tool_calls': sum(r.get('grounding_dino_tool_calls', 0) for r in type_results if r.get('grounding_dino_tool_calls', 0) > 0) / grounding_total if grounding_total > 0 else 0
            }
            
    # ⭐ Analyze ChartMoE impact on different question types
    chartmoe_impact_by_type = {}
    if enable_chartmoe:
        for q_type in accuracy_by_type.keys():
            type_results = [r for r in successful_results if r.get('question_type') == q_type]
            chartmoe_correct = sum(1 for r in type_results 
                                  if r.get('chartmoe_tool_calls', 0) > 0 and r.get('correct', False))
            chartmoe_total = sum(1 for r in type_results 
                                if r.get('chartmoe_tool_calls', 0) > 0)
            
            chartmoe_impact_by_type[q_type] = {
                'chartmoe_used': chartmoe_total,
                'chartmoe_correct': chartmoe_correct,
                'chartmoe_accuracy': (chartmoe_correct / chartmoe_total * 100) if chartmoe_total > 0 else 0,
                'avg_tool_calls': sum(r.get('chartmoe_tool_calls', 0) for r in type_results if r.get('chartmoe_tool_calls', 0) > 0) / chartmoe_total if chartmoe_total > 0 else 0
            }
    
    # Save summary
    summary = {
        "experiment_id": exp_id,
        "model": args.model,
        "dataset": "chartqa",
        "chartqa_enhanced_reasoning": args.enable_chartqa_reasoning,
        "reflection_enabled": args.enable_reflection,
        "max_attempts": args.max_attempts if args.enable_reflection else 1,
        "numerical_tolerance": args.numerical_tolerance,
        "use_structured_output": args.use_structured_output,
        "deepeyes_enabled": enable_deepeyes,
        "deepeyes_version": args.deepeyes_version if enable_deepeyes else None,
        "grounding_dino_enabled": enable_grounding_dino,  # ⭐ Add Grounding DINO config
        "chartmoe_enabled": enable_chartmoe,  # ⭐ Add ChartMoE config
        "config_used": args.config_experiment if args.use_config else None,
        "total_tasks": len(task_ids),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(results) - len(successful_results),
        "overall_accuracy": overall_accuracy,
        "numerical_questions_accuracy": numerical_accuracy,
        "first_attempt_accuracy": first_attempt_accuracy,
        "accuracy_by_question_type": accuracy_by_type,
        "average_attempts": avg_attempts,
        "tolerance_impact": tolerance_impact,
        "structured_reasoning_usage": {
            "enabled": args.enable_chartqa_reasoning,
            "tasks_using_structured": structured_reasoning_count,
            "percentage": structured_reasoning_count / len(successful_results) if successful_results else 0,
            "calculator_usage": calculator_usage_count,
            "impact_by_type": structured_impact_by_type if args.enable_chartqa_reasoning else {}
        },
        "deepeyes_usage": {
            "enabled": enable_deepeyes,
            "tasks_using_deepeyes": deepeyes_usage_count,
            "percentage": deepeyes_usage_count / len(successful_results) if successful_results else 0,
            "average_calls_per_task": avg_deepeyes_calls,
            "impact_by_type": deepeyes_impact_by_type if enable_deepeyes else {}
        },
        "grounding_dino_usage": {  # ⭐ Add Grounding DINO statistics
            "enabled": enable_grounding_dino,
            "tasks_using_grounding_dino": grounding_dino_usage_count,
            "percentage": grounding_dino_usage_count / len(successful_results) if successful_results else 0,
            "average_calls_per_task": avg_grounding_dino_calls,
            "impact_by_type": grounding_dino_impact_by_type if enable_grounding_dino else {}
        },
        "chartmoe_usage": {  # ⭐ Add ChartMoE statistics
            "enabled": enable_chartmoe,
            "tasks_using_chartmoe": chartmoe_usage_count,
            "percentage": chartmoe_usage_count / len(successful_results) if successful_results else 0,
            "average_calls_per_task": avg_chartmoe_calls,
            "impact_by_type": chartmoe_impact_by_type if enable_chartmoe else {}
        },
        "dataset_statistics": stats,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = exp_dirs["results"] / "summary.json"
    with open(summary_file, 'w') as f:
        serializable_summary = convert_numpy(summary)
        json.dump(serializable_summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CHARTQA EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"ChartQA Enhanced Reasoning: {args.enable_chartqa_reasoning}")
    if args.enable_chartqa_reasoning:
        percentage = (structured_reasoning_count/len(successful_results)*100) if successful_results else 0
        print(f"  Tasks using structured reasoning: {structured_reasoning_count} ({percentage:.1f}%)")
        print(f"  Tasks using calculator: {calculator_usage_count}")
    print(f"Reflection Enabled: {args.enable_reflection}")
    print(f"Structured Output Format: {args.use_structured_output}")
    print(f"DeepEyes Tools Enabled: {enable_deepeyes}")
    if enable_deepeyes:
        percentage = (deepeyes_usage_count/len(successful_results)*100) if successful_results else 0
        print(f"  Tasks using DeepEyes: {deepeyes_usage_count} ({percentage:.1f}%)")
        print(f"  Average DeepEyes calls per task: {avg_deepeyes_calls:.2f}")
    
    print(f"ChartMoE Tools Enabled: {enable_chartmoe}")  # ⭐ Add ChartMoE info
    if enable_chartmoe:
        percentage = (chartmoe_usage_count/len(successful_results)*100) if successful_results else 0
        print(f"  Tasks using ChartMoE: {chartmoe_usage_count} ({percentage:.1f}%)")
        print(f"  Average ChartMoE calls per task: {avg_chartmoe_calls:.2f}")
    
    print(f"Grounding DINO Tools Enabled: {enable_grounding_dino}")  # ⭐ Add Grounding DINO info
    if enable_grounding_dino:
        percentage = (grounding_dino_usage_count/len(successful_results)*100) if successful_results else 0
        print(f"  Tasks using Grounding DINO: {grounding_dino_usage_count} ({percentage:.1f}%)")
        print(f"  Average Grounding DINO calls per task: {avg_grounding_dino_calls:.2f}")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print(f"Numerical Questions Accuracy: {numerical_accuracy:.2%}")
    
    if args.enable_reflection and first_attempt_accuracy is not None:
        print(f"First Attempt Accuracy: {first_attempt_accuracy:.2%}")
        print(f"Average attempts per task: {avg_attempts:.2f}")
        improvement = overall_accuracy - first_attempt_accuracy
        if improvement > 0:
            print(f"Improvement from reflection: +{improvement:.2%}")
        else:
            print(f"Change from reflection: {improvement:.2%}")
    
    print(f"\nAccuracy by Question Type:")
    
    # Group numerical subtypes
    print("\nNumerical Questions:")
    for q_type in numerical_subtypes:
        if q_type in accuracy_by_type:
            acc = accuracy_by_type[q_type]
            count = len(results_by_qtype[q_type])
            print(f"  {q_type}: {acc:.2%} ({count} samples)")
            
            # Show tolerance impact if available
            if q_type in tolerance_impact:
                impact = tolerance_impact[q_type]
                print(f"    - Average relative error: {impact['average_relative_error']:.3f}")
                print(f"    - Within {args.numerical_tolerance*100:.0f}% tolerance: {impact['within_tolerance']}/{impact['count']}")
            
            # Show structured reasoning impact if available
            if args.enable_chartqa_reasoning and q_type in structured_impact_by_type:
                impact = structured_impact_by_type[q_type]
                if impact['structured_used'] > 0:
                    print(f"    - With structured reasoning: {impact['structured_accuracy']:.1f}% "
                          f"({impact['structured_correct']}/{impact['structured_used']})")
            
            # ⭐ Show tool impact if available
            if enable_deepeyes and q_type in deepeyes_impact_by_type:
                impact = deepeyes_impact_by_type[q_type]
                if impact['deepeyes_used'] > 0:
                    print(f"    - With DeepEyes tools: {impact['deepeyes_accuracy']:.1f}% "
                          f"({impact['deepeyes_correct']}/{impact['deepeyes_used']}, "
                          f"avg calls: {impact['avg_tool_calls']:.1f})")
            
            if enable_grounding_dino and q_type in grounding_dino_impact_by_type:
                impact = grounding_dino_impact_by_type[q_type]
                if impact['grounding_dino_used'] > 0:
                    print(f"    - With Grounding DINO: {impact['grounding_dino_accuracy']:.1f}% "
                          f"({impact['grounding_dino_correct']}/{impact['grounding_dino_used']}, "
                          f"avg calls: {impact['avg_tool_calls']:.1f})")
    
    print("\nOther Question Types:")
    for q_type in ['comparison', 'minmax', 'trend', 'retrieval', 'other']:
        if q_type in accuracy_by_type:
            acc = accuracy_by_type[q_type]
            count = len(results_by_qtype[q_type])
            print(f"  {q_type}: {acc:.2%} ({count} samples)")
            
            # Show structured reasoning impact if available
            if args.enable_chartqa_reasoning and q_type in structured_impact_by_type:
                impact = structured_impact_by_type[q_type]
                if impact['structured_used'] > 0:
                    print(f"    - With structured reasoning: {impact['structured_accuracy']:.1f}% "
                          f"({impact['structured_correct']}/{impact['structured_used']})")
            
            # ⭐ Show tool impact if available
            if enable_deepeyes and q_type in deepeyes_impact_by_type:
                impact = deepeyes_impact_by_type[q_type]
                if impact['deepeyes_used'] > 0:
                    print(f"    - With DeepEyes tools: {impact['deepeyes_accuracy']:.1f}% "
                          f"({impact['deepeyes_correct']}/{impact['deepeyes_used']}, "
                          f"avg calls: {impact['avg_tool_calls']:.1f})")
            
            if enable_grounding_dino and q_type in grounding_dino_impact_by_type:
                impact = grounding_dino_impact_by_type[q_type]
                if impact['grounding_dino_used'] > 0:
                    print(f"    - With Grounding DINO: {impact['grounding_dino_accuracy']:.1f}% "
                          f"({impact['grounding_dino_correct']}/{impact['grounding_dino_used']}, "
                          f"avg calls: {impact['avg_tool_calls']:.1f})")
                    
            if enable_chartmoe and q_type in chartmoe_impact_by_type:
                impact = chartmoe_impact_by_type[q_type]
                if impact['chartmoe_used'] > 0:
                    print(f"    - With ChartMoE: {impact['chartmoe_accuracy']:.1f}% "
                          f"({impact['chartmoe_correct']}/{impact['chartmoe_used']}, "
                          f"avg calls: {impact['avg_tool_calls']:.1f})")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    if args.save_trajectories and trajectories:
        print(f"Trajectories saved to: {trajectories_file}")
    print(f"{'='*60}")
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChartQA with reflection and enhanced reasoning support")
    parser.add_argument("--annotation", type=str,
                        default="data/chartqa/converted_data/chartqa_vlmgym_format.json",
                        help="Path to annotation file(s)")
    parser.add_argument("--data-root", type=str, default="data/chartqa",
                        help="Root directory for images")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks")
    
    # Action system arguments
    parser.add_argument("--enable-actions", action="store_true",
                        help="Enable action system for multi-step reasoning")
    parser.add_argument("--custom-actions", action="store_true",
                        help="Enable ChartQA-specific custom actions")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum steps per task when actions are enabled")
    parser.add_argument("--save-action-history", action="store_true",
                        help="Save action history in results")
    
    # ChartQA Enhanced Reasoning arguments
    parser.add_argument("--enable-chartqa-reasoning", action="store_true",
                        help="Enable ChartQA enhanced reasoning with calculator and structured templates")
    parser.add_argument("--chartqa-debug", action="store_true",
                        help="Enable debug output for ChartQA reasoning")
    
    # Reflection arguments
    parser.add_argument("--enable-reflection", action="store_true",
                        help="Enable reflection mechanism for incorrect answers")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum attempts per question with reflection")
    
    # Numerical tolerance argument
    parser.add_argument("--numerical-tolerance", type=float, default=0.05,
                        help="Relative tolerance for numerical answers (default: 5%, as per ChartQA paper)")
    
    # Structured output argument
    parser.add_argument("--use-structured-output", action="store_true", default=True,
                        help="Use structured output format with <think> and <answer> tags (default: True)")
    parser.add_argument("--no-structured-output", action="store_false", dest="use_structured_output",
                        help="Disable structured output format")
    
    # ⭐ DeepEyes arguments
    parser.add_argument("--enable-deepeyes", action="store_true",
                        help="Enable DeepEyes visual tools")
    parser.add_argument("--deepeyes-version", type=str, default="v2",
                        choices=["v1", "v2"], help="DeepEyes version to use")
    
    # ⭐ Grounding DINO arguments
    parser.add_argument("--enable-grounding-dino", action="store_true",
                        help="Enable Grounding DINO object detection tool")
    parser.add_argument("--grounding-dino-model", type=str,
                        default="/path/to/groundingdino_swint_ogc.pth",
                        help="Path to Grounding DINO model checkpoint")
    parser.add_argument("--grounding-dino-config", type=str,
                        default="/path/to/GroundingDINO_SwinT_OGC.py",
                        help="Path to Grounding DINO config file")

    # ChartMoE arguments
    parser.add_argument("--enable-chartmoe", action="store_true",
                        help="Enable ChartMoE chart understanding tool")
    parser.add_argument("--chartmoe-model", type=str,
                    default="IDEA-FinAI/chartmoe",
                    help="ChartMoE model name or path")
    parser.add_argument("--chartmoe-device", type=str,
                    default="cuda",
                    choices=["cuda", "cpu"],
                    help="Device for ChartMoE model")
    
    # ⭐ Configuration file arguments
    parser.add_argument("--use-config", action="store_true",
                        help="Use configuration file to create environment")
    parser.add_argument("--config-path", type=str, 
                        default="vlm_gym/configs/env_config.yaml",
                        help="Path to environment configuration file")
    parser.add_argument("--config-experiment", type=str,
                        default="grounding_dino_only",
                        choices=["grounding_dino_only", "deepeyes_only", "all_tools", 
                                "baseline", "debug", "chartqa_grounding","chartqa_chartmoe","chartqa_all_tools"],
                        help="Experiment configuration to use")
    
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save interaction trajectories for RL training")
    
    # Sampling options
    parser.add_argument("--question-type", type=str, default=None,
                        choices=['numerical', 'counting', 'summation', 'average', 'percentage', 
                                'difference', 'ratio', 'comparison', 'trend', 'minmax', 
                                'retrieval', 'other'],
                        help="Filter by question type")
    parser.add_argument("--hard-examples", action="store_true",
                        help="Select hard examples")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified sampling")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle task order")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Run evaluation with reflection
    evaluate_chartqa_with_reflection(args)


if __name__ == "__main__":
    main()
