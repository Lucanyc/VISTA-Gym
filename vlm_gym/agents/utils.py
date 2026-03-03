

#!/usr/bin/env python3
"""Utility functions for VLM agents"""

import os
import re
import math
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def load_image_safely(image_path: str, default_size: Tuple[int, int] = (336, 336)) -> Image.Image:
    """Safely load an image with multiple fallback options
    
    Args:
        image_path: Path to the image file
        default_size: Size for placeholder image if loading fails
        
    Returns:
        PIL Image object (either loaded image or placeholder)
    """
    # List of possible path variations to try
    possible_paths = [
        image_path,  # Original path
        os.path.abspath(image_path),  # Absolute path
        os.path.join('data', image_path),  # In data directory
        os.path.join('../', image_path),  # Parent directory
        os.path.join('../../', image_path),  # Two levels up
    ]
    
    # Add paths from environment variable if set
    if 'VLM_DATA_DIR' in os.environ:
        data_dir = os.environ['VLM_DATA_DIR']
        possible_paths.extend([
            os.path.join(data_dir, image_path),
            os.path.join(data_dir, os.path.basename(image_path))
        ])
    
    # Common dataset paths
    dataset_dirs = [
        'llava_cot_images',
        'data/llava_cot_images',
        '../llava_cot_images',
        'mulberry_images',
        'data/mulberry_images'
    ]
    
    # If the path contains a known dataset directory, try variations
    for dataset_dir in dataset_dirs:
        if dataset_dir in image_path:
            # Try replacing the dataset directory part
            base_name = image_path.split(dataset_dir)[-1].lstrip('/')
            for dir_variant in dataset_dirs:
                possible_paths.append(os.path.join(dir_variant, base_name))
    
    # Try to load from each path
    for path in possible_paths:
        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert('RGB')
                logger.debug(f"Successfully loaded image from: {path}")
                return img
        except Exception as e:
            logger.debug(f"Failed to load from {path}: {e}")
            continue
    
    # If all attempts fail, create a placeholder
    logger.warning(f"Could not load image: {image_path}, using placeholder")
    placeholder = Image.new('RGB', default_size, color='gray')
    
    # Draw a simple pattern to indicate it's a placeholder
    from PIL import ImageDraw
    draw = ImageDraw.Draw(placeholder)
    draw.text((10, 10), "IMAGE NOT FOUND", fill='white')
    draw.rectangle([5, 5, default_size[0]-5, default_size[1]-5], outline='white', width=2)
    
    return placeholder


def parse_vlm_response(response: str, choices: Optional[List[str]] = None) -> str:
    """Parse VLM response to extract the answer
    
    Args:
        response: Raw response from VLM
        choices: List of multiple choice options (if applicable)
        
    Returns:
        Parsed answer - either the selected choice or cleaned response
    """
    if not response:
        return choices[0] if choices else ""
    
    response = response.strip()
    
    if choices:
        # Multiple choice question - try to extract answer
        answer_letter = extract_choice_letter(response)
        
        if answer_letter:
            # Map letter to choice
            idx = ord(answer_letter.upper()) - ord('A')
            if 0 <= idx < len(choices):
                logger.debug(f"Extracted choice {answer_letter} -> {choices[idx]}")
                return choices[idx]
        
        # Fallback: check if any choice text appears in response
        response_lower = response.lower()
        for i, choice in enumerate(choices):
            if choice.lower() in response_lower:
                logger.debug(f"Found choice text match: {choice}")
                return choice
        
        # If we can't extract an answer, default to first choice
        logger.warning(f"Could not parse choice from response: {response[:100]}...")
        return choices[0]
    
    else:
        # Open-ended question - clean up response
        return clean_response(response)


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract multiple choice answer letter from text
    
    Args:
        text: Response text
        
    Returns:
        Letter (A, B, C, D, etc.) or None
    """
    # Common patterns for expressing choices
    patterns = [
        r'^([A-Z])[\.\)]\s',           # "A. " or "A) " at start
        r'^([A-Z])$',                  # Just the letter
        r'answer is ([A-Z])',          # "answer is X"
        r'correct answer is ([A-Z])',   # "correct answer is X"
        r'choose ([A-Z])',             # "choose X"
        r'select ([A-Z])',             # "select X"
        r'option ([A-Z])',             # "option X"
        r'\(([A-Z])\)',               # "(X)"
        r'^\s*([A-Z])\s*[\:\.\)]\s*', # "X:" or "X." with whitespace
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # Look for standalone letters in the text
    words = text.split()
    for word in words[:5]:  # Check first 5 words
        cleaned = word.strip('.,;:()[]{}')
        if len(cleaned) == 1 and cleaned.upper() in 'ABCDEFGHIJ':
            return cleaned.upper()
    
    return None


def clean_response(text: str, max_length: int = 200) -> str:
    """Clean up VLM response text
    
    Args:
        text: Raw response text
        max_length: Maximum length for response
        
    Returns:
        Cleaned text
    """
    # Remove common prefixes that don't add information
    prefixes_to_remove = [
        "Based on the image,",
        "Looking at the image,",
        "The image shows",
        "In the image,",
        "According to the image,",
        "From the image,",
        "I can see that",
        "The answer is:",
        "Answer:",
    ]
    
    text = text.strip()
    text_lower = text.lower()
    
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
    
    # Remove excess whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    if len(text) > max_length:
        # Try to cut at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        if sentences and len(sentences[0]) <= max_length:
            text = sentences[0].strip() + '.'
        else:
            # Cut at word boundary
            words = text[:max_length].split()
            text = ' '.join(words[:-1]) + '...'
    
    return text


def format_prompt_with_choices(question: str, choices: List[str], language: str = None) -> str:
    """Format a multiple choice question prompt
    
    Args:
        question: The question text
        choices: List of answer choices
        language: Language for prompts (auto-detect if None)
        
    Returns:
        Formatted prompt
    """
    # Auto-detect language if not specified
    if language is None:
        # Simple heuristic: if question contains Chinese characters
        import re
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', question)
        language = 'zh' if chinese_chars else 'en'
    
    # Ensure question ends with proper punctuation
    question = question.strip()
    if not question.endswith(('?', '.', '!', '？', '。', '！', ')', '）')):
        question += '？' if language == 'zh' else '?'
    
    # Format choices
    formatted_choices = "\n".join([
        f"{chr(65+i)}. {choice}" 
        for i, choice in enumerate(choices)
    ])
    
    # Build prompt based on language
    if language == 'zh':
        # 中文版本 - 不要重复选项
        prompt = f"{question}"  # 只返回问题，不添加额外的英文提示
    else:
        # 英文版本（保持原样）
        prompt = f"""{question}

Choose from the following options:
{formatted_choices}

Please select the correct answer by stating the letter (A, B, C, etc.) of your choice."""
    
    return prompt


def calculate_confidence_score(logprobs: List[float]) -> float:
    """Calculate confidence score from log probabilities
    
    Args:
        logprobs: List of log probabilities for each token
        
    Returns:
        Confidence score between 0 and 1
    """
    if not logprobs:
        return 0.5  # Neutral confidence if no logprobs
    
    # Calculate average log probability
    avg_logprob = sum(logprobs) / len(logprobs)
    
    # Convert to probability space
    # More negative logprobs = lower confidence
    # Typical range: -5 (very low) to 0 (very high)
    
    # Sigmoid-like transformation to map to [0, 1]
    # Center around -2.5 (moderate confidence)
    confidence = 1 / (1 + math.exp(-(avg_logprob + 2.5)))
    
    return max(0.0, min(1.0, confidence))


def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """Get information about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'width': image.width,
        'height': image.height,
        'aspect_ratio': image.width / image.height if image.height > 0 else 0
    }


def validate_observation(observation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate that observation has required fields
    
    Args:
        observation: Observation dictionary
        
    Returns:
        is_valid: Whether observation is valid
        error_msg: Error message if invalid
    """
    required_fields = ['image_path', 'question']
    
    for field in required_fields:
        if field not in observation:
            return False, f"Missing required field: {field}"
    
    if not isinstance(observation['question'], str):
        return False, "Question must be a string"
    
    if not observation['question'].strip():
        return False, "Question cannot be empty"
    
    return True, None


# ==================== Reflection Prompt Templates and Functions ====================
# 以下是 reflection+multi-turn 相关功能

# Basic reflection template
REFLECTION_PROMPT_TEMPLATE = """
Previous Attempt Analysis:
- Your answer: {previous_answer}
- Evaluation: Incorrect

{feedback}

Reflection Guidelines:
1. Carefully re-examine the chart/image
2. Consider what might have been missed in your previous attempt
3. {specific_guidance}

Attempt {attempt} of {max_attempts}
{hint_section}

Now, please provide your revised answer to the question:
{question}
"""

# More detailed template for complex reasoning
DETAILED_REFLECTION_TEMPLATE = """
📊 Reflection on Previous Attempt

Previous Response: {previous_answer}
Status: ❌ Incorrect

Feedback from System:
{feedback}

Key Points to Consider:
• {point1}
• {point2}
• {point3}

{hint_section}

Current Progress: Attempt {attempt}/{max_attempts}
{urgency_message}

Please carefully analyze the image again and answer:
{question}
"""

# Concise template for simple corrections
CONCISE_REFLECTION_TEMPLATE = """
Your answer "{previous_answer}" was incorrect.
{feedback}
{hint_section}
Please try again (Attempt {attempt}/{max_attempts}): {question}
"""


def format_reflection_prompt(observation: Dict[str, Any], 
                           template_style: str = "standard") -> str:
    """
    Format prompt for reflection attempts with various styles
    
    Args:
        observation: Dictionary containing reflection context
        template_style: One of "standard", "detailed", "concise"
    
    Returns:
        Formatted reflection prompt
    """
    
    # Extract observation data
    previous_answer = observation.get('previous_answer', '')
    feedback = observation.get('feedback', '')
    hint = observation.get('hint', '')
    attempt = observation.get('attempt', 2)
    max_attempts = observation.get('max_attempts', 3)
    question = observation.get('question', '')
    attempts_remaining = observation.get('attempts_remaining', max_attempts - attempt + 1)
    
    # Determine specific guidance based on feedback
    specific_guidance = _get_specific_guidance(feedback)
    
    # Build hint section
    hint_section = f"\n💡 Helpful hint: {hint}" if hint else ""
    
    # Urgency message for last attempt
    urgency_message = ""
    if attempts_remaining == 1:
        urgency_message = "⚠️ This is your FINAL attempt. Please be extra careful!"
    elif attempts_remaining == 2:
        urgency_message = "You have one more attempt after this if needed."
    
    # Select template based on style
    if template_style == "detailed":
        points = _extract_key_points(feedback, observation)
        return DETAILED_REFLECTION_TEMPLATE.format(
            previous_answer=previous_answer,
            feedback=feedback,
            point1=points[0],
            point2=points[1],
            point3=points[2],
            hint_section=hint_section,
            attempt=attempt,
            max_attempts=max_attempts,
            urgency_message=urgency_message,
            question=question
        )
    elif template_style == "concise":
        return CONCISE_REFLECTION_TEMPLATE.format(
            previous_answer=previous_answer,
            feedback=feedback,
            hint_section=hint_section,
            attempt=attempt,
            max_attempts=max_attempts,
            question=question
        )
    else:
        # Standard template
        return REFLECTION_PROMPT_TEMPLATE.format(
            previous_answer=previous_answer,
            feedback=feedback,
            specific_guidance=specific_guidance,
            attempt=attempt,
            max_attempts=max_attempts,
            hint_section=hint_section,
            question=question
        )


def _get_specific_guidance(feedback: str) -> str:
    """Generate specific guidance based on feedback content"""
    feedback_lower = feedback.lower()
    
    if 'too high' in feedback_lower:
        return "Pay attention to the scale and ensure you're reading values correctly"
    elif 'too low' in feedback_lower:
        return "Make sure you haven't missed any items in your count"
    elif 'count' in feedback_lower or 'number' in feedback_lower:
        return "Count systematically from left to right or top to bottom"
    elif 'compare' in feedback_lower or 'difference' in feedback_lower:
        return "Identify exact values for each element before calculating"
    elif 'maximum' in feedback_lower or 'minimum' in feedback_lower:
        return "Check every single data point to find the true extreme"
    elif 'trend' in feedback_lower:
        return "Analyze the overall pattern from start to end"
    elif 'color' in feedback_lower or 'legend' in feedback_lower:
        return "Pay close attention to the legend and color coding"
    else:
        return "Double-check all relevant data points before answering"


def _extract_key_points(feedback: str, observation: Dict[str, Any]) -> List[str]:
    """Extract key points for detailed reflection template"""
    points = []
    feedback_lower = feedback.lower()
    question = observation.get('question', '').lower()
    
    # Point 1: Error type
    if 'too high' in feedback_lower or 'too low' in feedback_lower:
        points.append("Check if you're misreading the scale or units")
    elif 'incorrect' in feedback_lower:
        points.append("Verify you understood the question correctly")
    else:
        points.append("Review your interpretation of the data")
    
    # Point 2: Question-specific guidance
    if 'how many' in question:
        points.append("Count each item individually and systematically")
    elif 'which' in question or 'what' in question:
        points.append("Read all labels and legends carefully")
    elif 'compare' in question:
        points.append("Ensure you're comparing the right elements")
    else:
        points.append("Focus on the specific data requested")
    
    # Point 3: General improvement
    if observation.get('attempt', 2) >= 3:
        points.append("Take a deep breath and approach the problem fresh")
    else:
        points.append("Look for details you might have overlooked")
    
    return points