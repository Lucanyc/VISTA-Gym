#!/usr/bin/env python3
"""Response formatting utilities for VLM agents"""

from typing import Dict, Any, Optional, List
import re
import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formatter for ensuring consistent response formats"""
    
    def __init__(self):
        self.logger = logger
    
    def ensure_answer_format(self, response: str, observation: Dict[str, Any]) -> str:
        """Ensure response has proper answer format
        
        Args:
            response: Raw response from agent
            observation: Original observation for context
            
        Returns:
            Properly formatted response with <answer> tags
        """
        # If it's already properly formatted, return as is
        if self._has_proper_format(response):
            return response
        
        # If it's a tool call, don't modify
        if self._is_tool_call(response):
            return response
        
        # Extract the answer from the response
        answer = self.extract_answer(response, observation)
        
        # Check if response has think tags
        if "<think>" in response and "</think>" in response:
            # Preserve thinking, add answer tag
            return response + f"\n<answer>{answer}</answer>"
        
        # Build complete formatted response
        return self.format_with_tags(response, answer)
    
    def extract_answer(self, response: str, observation: Dict[str, Any]) -> str:
        """Extract the actual answer from response content
        
        Args:
            response: Raw response text
            observation: Context for understanding question type
            
        Returns:
            Extracted answer
        """
        question = observation.get("question", "").lower()
        question_type = observation.get("question_type", "")
        
        # Try to extract from answer tags first
        answer = self._extract_from_tags(response)
        if answer:
            return answer
        
        # Extract based on question type
        if question_type == "counting" or any(kw in question for kw in ["how many", "count", "number", "total"]):
            return self._extract_number(response)
        
        elif question_type == "yes_no" or self._is_yes_no_question(question):
            return self._extract_yes_no(response)
        
        elif question_type == "numerical" or any(kw in question for kw in ["value", "how much", "sum", "difference"]):
            return self._extract_numerical(response)
        
        elif observation.get("choices"):
            return self._extract_choice(response, observation.get("choices", []))
        
        elif question_type == "percentage":
            return self._extract_percentage(response)
        
        elif question_type in ["minmax", "comparison"]:
            return self._extract_comparison(response)
        
        else:
            # Default: extract the most relevant sentence
            return self._extract_default_answer(response)
    
    def format_with_tags(self, reasoning: str, answer: str) -> str:
        """Format response with proper think and answer tags
        
        Args:
            reasoning: The reasoning/thinking part
            answer: The final answer
            
        Returns:
            Formatted response with tags
        """
        # Clean up reasoning if needed
        reasoning = reasoning.strip()
        
        # Remove any existing answer from reasoning
        if "<answer>" in reasoning:
            reasoning = reasoning.split("<answer>")[0].strip()
        
        return f"""<think>
{reasoning}
</think>
<answer>{answer}</answer>"""
    
    def format_tool_response(self, tool_name: str, tool_result: Any, observation: Dict[str, Any]) -> str:
        """Format response based on tool output
        
        Args:
            tool_name: Name of the tool used
            tool_result: Result from the tool
            observation: Original observation
            
        Returns:
            Formatted prompt for generating final answer
        """
        question = observation.get("question", "")
        
        # Tool-specific formatting
        if tool_name == "chartmoe":
            return self._format_chartmoe_response(tool_result, question)
        elif tool_name == "multimath_server":
            return self._format_multimath_response(tool_result, question)
        elif tool_name == "sam2":
            return self._format_sam2_response(tool_result, question)
        elif tool_name == "sympy_geometry":
            return self._format_sympy_response(tool_result, question)
        elif tool_name == "easyocr":
            return self._format_easyocr_response(tool_result, question)
        elif tool_name == "grounding_dino":
            return self._format_grounding_dino_response(tool_result, question)
        else:
            return self._format_generic_tool_response(tool_name, tool_result, question)
    
    def extract_thinking(self, response: str) -> Optional[str]:
        """Extract thinking/reasoning from <think> tags
        
        Args:
            response: Response text that may contain think tags
            
        Returns:
            Extracted thinking content or None
        """
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def split_thinking_and_answer(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Split response into thinking and answer parts
        
        Args:
            response: Complete response with potential think/answer tags
            
        Returns:
            Tuple of (thinking, answer) - either may be None
        """
        thinking = self.extract_thinking(response)
        answer = self._extract_from_tags(response)
        
        # If no tags found, try to infer
        if not thinking and not answer:
            # The whole response might be the answer
            answer = response.strip()
        
        return thinking, answer
    
    def parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse a structured response into components
        
        Args:
            response: Raw response that may contain various tags
            
        Returns:
            Dictionary with parsed components:
            - thinking: Extracted thinking process
            - answer: Extracted answer
            - raw: Original response
            - has_structure: Whether response had proper tags
        """
        result = {
            "raw": response,
            "thinking": None,
            "answer": None,
            "has_structure": False
        }
        
        # Check for tool call first
        if self._is_tool_call(response):
            result["is_tool_call"] = True
            return result
        
        # Extract thinking
        thinking = self.extract_thinking(response)
        if thinking:
            result["thinking"] = thinking
            result["has_structure"] = True
        
        # Extract answer
        answer = self._extract_from_tags(response)
        if answer:
            result["answer"] = answer
            result["has_structure"] = True
        
        # If no structured answer but has thinking, extract answer from remaining text
        if thinking and not answer:
            # Remove think tags and extract answer from remainder
            remainder = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            if remainder:
                result["answer"] = remainder
        
        # If nothing structured found, treat whole response as answer
        if not result["has_structure"]:
            result["answer"] = response.strip()
        
        return result
    
    def combine_thinking_and_answer(self, thinking: str, answer: str) -> str:
        """Combine thinking and answer into properly formatted response
        
        Args:
            thinking: The reasoning/thinking process
            answer: The final answer
            
        Returns:
            Formatted response with both components
        """
        return f"""<think>
{thinking.strip()}
</think>
<answer>{answer.strip()}</answer>"""
    
    def parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse a structured response into components
        
        Args:
            response: Raw response that may contain various tags
            
        Returns:
            Dictionary with parsed components:
            - thinking: Extracted thinking process
            - answer: Extracted answer
            - raw: Original response
            - has_structure: Whether response had proper tags
        """
        result = {
            "raw": response,
            "thinking": None,
            "answer": None,
            "has_structure": False
        }
        
        # Check for tool call first
        if self._is_tool_call(response):
            result["is_tool_call"] = True
            return result
        
        # Extract thinking
        thinking = self.extract_thinking(response)
        if thinking:
            result["thinking"] = thinking
            result["has_structure"] = True
        
        # Extract answer
        answer = self._extract_from_tags(response)
        if answer:
            result["answer"] = answer
            result["has_structure"] = True
        
        # If no structured answer but has thinking, extract answer from remaining text
        if thinking and not answer:
            # Remove think tags and extract answer from remainder
            remainder = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            if remainder:
                result["answer"] = remainder
        
        # If nothing structured found, treat whole response as answer
        if not result["has_structure"]:
            result["answer"] = response.strip()
        
        return result
    
    def combine_thinking_and_answer(self, thinking: str, answer: str) -> str:
        """Combine thinking and answer into properly formatted response
        
        Args:
            thinking: The reasoning/thinking process
            answer: The final answer
            
        Returns:
            Formatted response with both components
        """
        return f"""<think>
{thinking.strip()}
</think>
<answer>{answer.strip()}</answer>"""
        """Extract thinking/reasoning from <think> tags
        
        Args:
            response: Response text that may contain think tags
            
        Returns:
            Extracted thinking content or None
        """
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def split_thinking_and_answer(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Split response into thinking and answer parts
        
        Args:
            response: Complete response with potential think/answer tags
            
        Returns:
            Tuple of (thinking, answer) - either may be None
        """
        thinking = self.extract_thinking(response)
        answer = self._extract_from_tags(response)
        
        # If no tags found, try to infer
        if not thinking and not answer:
            # The whole response might be the answer
            answer = response.strip()
        
        return thinking, answer
    
    def _has_proper_format(self, response: str) -> bool:
        """Check if response already has proper format"""
        return "<answer>" in response and "</answer>" in response
    
    def _is_tool_call(self, response: str) -> bool:
        """Check if response is a tool call"""
        return "<tool_call>" in response or response.strip().startswith('{"tool":')
    
    def _extract_from_tags(self, response: str) -> Optional[str]:
        """Extract answer from <answer> tags"""
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_number(self, response: str) -> str:
        """Extract number answer from response"""
        # Look for patterns like "is 5", "are 10", "total: 42"
        patterns = [
            r'\b(?:is|are|was|were|equals?|total[s]?|count[s]?)\s*:?\s*(\d+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:items?|objects?|elements?|values?|entries?)\b',
            r'(?:answer|result)\s*:?\s*(\d+(?:\.\d+)?)',
            r'^(\d+(?:\.\d+)?)$',  # Just a number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Fallback: find all numbers and return the last one
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers:
            return numbers[-1]
        
        return "0"
    
    def _extract_yes_no(self, response: str) -> str:
        """Extract Yes/No answer from response"""
        response_lower = response.lower()
        
        # Direct patterns
        if re.search(r'\b(?:yes|correct|true|affirmative)\b', response_lower):
            return "Yes"
        elif re.search(r'\b(?:no|incorrect|false|negative)\b', response_lower):
            return "No"
        
        # Check for answers in sentences
        if "the answer is yes" in response_lower:
            return "Yes"
        elif "the answer is no" in response_lower:
            return "No"
        
        # Default to No if uncertain
        return "No"
    
    def _extract_numerical(self, response: str) -> str:
        """Extract numerical value from response"""
        # Similar to number extraction but may include units
        patterns = [
            r'(?:value|answer|result)\s*:?\s*([\d,]+(?:\.\d+)?)',
            r'\b(?:is|equals?|=)\s*([\d,]+(?:\.\d+)?)',
            r'^([\d,]+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
            if match:
                # Remove commas from number
                return match.group(1).replace(',', '')
        
        # Find last number in response
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return "0"
    
    def _extract_choice(self, response: str, choices: List[str]) -> str:
        """Extract multiple choice answer"""
        # Look for letter patterns
        patterns = [
            r'^([A-Z])[\.\)]\s',
            r'^([A-Z])$',
            r'answer is ([A-Z])',
            r'correct answer is ([A-Z])',
            r'choose ([A-Z])',
            r'\(([A-Z])\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.IGNORECASE | re.MULTILINE)
            if match:
                letter = match.group(1).upper()
                idx = ord(letter) - ord('A')
                if 0 <= idx < len(choices):
                    return choices[idx]
        
        # Check if choice text appears in response
        response_lower = response.lower()
        for choice in choices:
            if choice.lower() in response_lower:
                return choice
        
        # Default to first choice
        return choices[0] if choices else ""
    
    def _extract_percentage(self, response: str) -> str:
        """Extract percentage from response"""
        # Look for percentage patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*percent',
            r'percentage\s*:?\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return f"{match.group(1)}%"
        
        # Try to find a number and add %
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers:
            # Assume numbers under 100 are percentages
            for num in reversed(numbers):
                if float(num) <= 100:
                    return f"{num}%"
        
        return "0%"
    
    def _extract_comparison(self, response: str) -> str:
        """Extract comparison or min/max answer"""
        # Extract the key comparative statement
        patterns = [
            r'(?:maximum|highest|largest)\s+(?:is|was|:)\s*(.+?)(?:\.|$)',
            r'(?:minimum|lowest|smallest)\s+(?:is|was|:)\s*(.+?)(?:\.|$)',
            r'(.+?)\s+(?:is|was|has)\s+(?:the\s+)?(?:maximum|highest|largest|greater)',
            r'(.+?)\s+(?:is|was|has)\s+(?:the\s+)?(?:minimum|lowest|smallest|less)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return first noun phrase after comparative word
        return self._extract_default_answer(response)
    
    def _extract_default_answer(self, response: str) -> str:
        """Extract default answer when type is unknown"""
        # Clean and get the most relevant sentence
        sentences = re.split(r'[.!?]+', response.strip())
        
        # Filter out meta-sentences
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip meta-commentary
            lower = sentence.lower()
            if any(phrase in lower for phrase in [
                "based on", "looking at", "according to", 
                "the image shows", "i can see"
            ]):
                continue
            
            relevant_sentences.append(sentence)
        
        # Return the last relevant sentence
        if relevant_sentences:
            return relevant_sentences[-1]
        
        # Fallback: return first 100 chars
        return response.strip()[:100]
    
    def _is_yes_no_question(self, question: str) -> bool:
        """Check if question expects yes/no answer"""
        yes_no_patterns = [
            r'^(?:is|are|was|were|does|do|did|can|could|will|would|should)\b',
            r'\b(?:yes\s+or\s+no|true\s+or\s+false)\b',
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in yes_no_patterns)
    
    # Tool-specific formatting methods
    def _format_chartmoe_response(self, result: Dict[str, Any], question: str) -> str:
        """Format ChartMoE tool response"""
        output = result.get("output", "")
        task_type = result.get("task_type", "")
        
        return f"""Based on ChartMoE {task_type} analysis:

{output}

Now answer: "{question}"

Provide your answer inside <answer> tags."""
    
    def _format_multimath_response(self, result: Dict[str, Any], question: str) -> str:
        """Format MultiMath Server response"""
        answer = result.get("answer", "")
        confidence = result.get("confidence", 0)
        steps = result.get("steps", [])
        
        prompt = f"""MultiMath Server Solution:
Answer: {answer}
Confidence: {confidence:.1%}
"""
        
        if steps:
            prompt += "\nSteps:\n"
            for i, step in enumerate(steps[:3], 1):
                prompt += f"{i}. {step}\n"
        
        prompt += f"""
For the question: "{question}"

The answer is: <answer>{answer}</answer>"""
        
        return prompt
    
    def _format_sam2_response(self, result: Dict[str, Any], question: str) -> str:
        """Format SAM2 segmentation response"""
        results = result.get("results", [])
        
        if results:
            best = max(results, key=lambda x: x.get('score', 0))
            coverage = best.get('coverage_percent', 0)
            
            return f"""SAM2 Segmentation Analysis:
- Coverage: {coverage:.1f}% of image
- Confidence: {best.get('score', 0):.3f}
- Number of segments: {len(results)}

Based on this analysis, answer: "{question}"

Provide your answer inside <answer> tags."""
        
        return f"""SAM2 segmentation complete.

Answer the question: "{question}"

Provide your answer inside <answer> tags."""
    
    def _format_sympy_response(self, result: Dict[str, Any], question: str) -> str:
        """Format SymPy Geometry response"""
        function = result.get("function", "")
        
        if function == "triangle_angle":
            angle = result.get("angle_degrees", "N/A")
            return f"""SymPy Calculation:
Angle = {angle}°

For: "{question}"

<answer>{angle}°</answer>"""
        
        elif function == "triangle_area":
            area = result.get("area", "N/A")
            return f"""SymPy Calculation:
Area = {area}

For: "{question}"

<answer>{area}</answer>"""
        
        else:
            # Generic format
            return f"""SymPy Geometry Result:
{result}

Answer: "{question}"

Provide your answer inside <answer> tags."""
    
    def _format_easyocr_response(self, result: Dict[str, Any], question: str) -> str:
        """Format EasyOCR response"""
        texts = result.get("texts", [])
        
        if texts:
            text_list = "\n".join([f"- {t}" for t in texts[:10]])
            return f"""Text extracted by EasyOCR:
{text_list}

Based on this text, answer: "{question}"

Provide your answer inside <answer> tags."""
        
        return f"""No text detected in image.

Answer: "{question}"

Provide your answer inside <answer> tags."""
    
    def _format_grounding_dino_response(self, result: Dict[str, Any], question: str) -> str:
        """Format Grounding DINO detection response"""
        detections = result.get("detections", [])
        
        if detections:
            count = len(detections)
            objects = ", ".join([d.get("label", "object") for d in detections[:5]])
            
            return f"""Grounding DINO detected {count} objects:
{objects}

Based on these detections, answer: "{question}"

Provide your answer inside <answer> tags."""
        
        return f"""No objects detected.

Answer: "{question}"

Provide your answer inside <answer> tags."""
    
    def _format_generic_tool_response(self, tool_name: str, result: Any, question: str) -> str:
        """Format generic tool response"""
        return f"""Tool {tool_name} result:
{str(result)[:500]}

Based on this, answer: "{question}"

Provide your answer inside <answer> tags."""