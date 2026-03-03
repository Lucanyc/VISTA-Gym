"""
OlympiadBench Task Wrapper for VLMGym

Handles Olympic-level mathematics competition problems with:
- Multiple answer types (numeric, expression, equation, interval, tuple)
- Error tolerance for numerical answers
- SymPy-based symbolic expression validation
- Multi-answer support
- Context support for progressive problems
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict
import random
import re

from .base import BaseAdapter

logger = logging.getLogger(__name__)

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logging.warning("SymPy not available. Symbolic expression validation will be limited.")

from .vision_qa_task import VisionQATask

logger = logging.getLogger(__name__)


class OlympiadBenchTask(VisionQATask):
    """
    OlympiadBench specific task wrapper
    
    Handles Olympic-level mathematics and physics competition problems with:
    - Complex answer types (numeric, expression, equation, interval, tuple)
    - Numerical error tolerance
    - Multi-solution support
    - Symbolic expression validation
    - Progressive problem support (context)
    """
    
    # Default error tolerance as per OlympiadBench paper
    DEFAULT_NUMERIC_TOLERANCE = 1e-8
    DEFAULT_PHYSICS_TOLERANCE = 1e-3  # More lenient for physics if not specified
    
    @classmethod
    def get_task_id(cls) -> str:
        """Get task type ID"""
        return "vlm-gym.olympiadbench"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """Set up OlympiadBench specific task"""
        # Call parent setup
        task_goal, task_info = super().setup()
        
        # Extract OlympiadBench specific metadata
        self.subfield = self.metadata.get('subfield', 'Unknown')
        self.answer_type = self.metadata.get('answer_type', 'Numerical')
        self.problem_type = self.metadata.get('problem_type', 'OE')
        self.is_multiple_answer = self.metadata.get('is_multiple_answer', False)
        self.error_tolerance = self.metadata.get('error')  # May be None
        self.answer_unit = self.metadata.get('unit')  # May be None
        self.solutions = self.metadata.get('solution', [])  # Reference solutions
        self.final_answers = self.metadata.get('final_answer', [])  # Expected final answers
        self.context = self.metadata.get('context', '')  # For progressive physics problems
        self.language = self.metadata.get('language', 'english')
        
        # Update task info
        task_info.update({
            "dataset": "olympiadbench",
            "subfield": self.subfield,
            "answer_type": self.answer_type,
            "problem_type": self.problem_type,
            "is_multiple_answer": self.is_multiple_answer,
            "has_error_tolerance": self.error_tolerance is not None,
            "has_unit": self.answer_unit is not None,
            "num_solutions": len(self.solutions) if isinstance(self.solutions, list) else 1,
            "has_context": bool(self.context),
            "language": self.language
        })
        
        # Build enhanced task goal with OlympiadBench prompt template
        enhanced_goal = self._build_olympiad_prompt()
        
        return enhanced_goal, task_info
    
    def _build_olympiad_prompt(self) -> str:
        """Build prompt following OlympiadBench template exactly"""
        # Determine subject from subfield
        subject_map = {
            'Geometry': 'Mathematics',
            'Algebra': 'Mathematics', 
            'Number Theory': 'Mathematics',
            'Combinatorics': 'Mathematics',
            'Mechanics': 'Physics',
            'Electromagnetism': 'Physics',
            'Thermodynamics': 'Physics',
            'Optics': 'Physics',
            'Modern Physics': 'Physics'
        }
        subject = subject_map.get(self.subfield, 'Mathematics')
        
        # Convert answer_type to lowercase
        ans_type = self.answer_type.lower()
        
        # Check language
        is_english = self.language.lower() == 'english'
        
        prompt_parts = []
        
        # 1) Opening statement
        if is_english:
            prompt_parts.append(f"The following is a question from an International {subject} competition.")
        else:
            prompt_parts.append(f"以下是中国{subject}竞赛中的解答题。")
        prompt_parts.append("")
        
        # 2) Context (if exists, e.g., for progressive physics problems)
        if self.context:
            prompt_parts.append(self.context)
            prompt_parts.append("")
        
        # 3) Question
        prompt_parts.append(self.question)
        prompt_parts.append("")
        
        # 4) Answer type specification
        if not self.is_multiple_answer:
            if is_english:
                prompt_parts.append(f"The answer of the question should be {ans_type}.")
            else:
                prompt_parts.append(f"答案类型为{ans_type}。")
        else:
            if is_english:
                prompt_parts.append(f"The question has multiple answers, each of them should be {ans_type}.")
            else:
                prompt_parts.append(f"题目有多个答案，答案类型均为{ans_type}。")
        prompt_parts.append("")
        
        # 5) Instructions
        if is_english:
            prompt_parts.append("Please calculate the answer according to the given requirements and the information provided.")
            prompt_parts.append("Please use LaTeX format to represent the variables and formulas used in the solution process and results.")
            prompt_parts.append("")
            
            # Final answer format
            if not self.is_multiple_answer:
                prompt_parts.append("Please end your solution with \"So the final answer is \\boxed{answer}.\"")
            else:
                prompt_parts.append("Please end your solution with \"So the final answer is \\boxed{multiple answers connected with commas}.\"")
            prompt_parts.append("and give the result explicitly.")
        else:
            prompt_parts.append("请根据题目的要求和所提供的信息计算得出答案。")
            prompt_parts.append("解答过程和结果中使用的变量和公式请使用LaTeX格式表示。")
            prompt_parts.append("")
            
            # Final answer format
            if not self.is_multiple_answer:
                prompt_parts.append("请在最后以\"所以最终答案是\\boxed{答案}。\"")
            else:
                prompt_parts.append("请在最后以\"所以最终答案是\\boxed{用英文逗号连接的多个答案}。\"")
            prompt_parts.append("显式给出结果。")
        
        return "\n".join(prompt_parts)
    
    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract answer from \\boxed{...} format"""
        if not text:
            return None
        
        # Look for \boxed{...} pattern
        # Handle nested braces by counting brace levels
        boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(boxed_pattern, text, re.DOTALL)
        
        if matches:
            # Return the last match (most likely the final answer)
            return matches[-1].strip()
        
        # Also check for Chinese version
        chinese_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}。'
        matches = re.findall(chinese_pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return None
    
    def _parse_multiple_answers(self, boxed_content: str) -> List[str]:
        """Parse multiple answers from comma-separated string"""
        if not boxed_content:
            return []
        
        # Split by comma, but be careful with commas inside expressions
        # Simple approach: split by comma at top level
        answers = []
        current = ""
        paren_depth = 0
        brace_depth = 0
        
        for char in boxed_content:
            if char == '(' : paren_depth += 1
            elif char == ')': paren_depth -= 1
            elif char == '{': brace_depth += 1
            elif char == '}': brace_depth -= 1
            elif char == ',' and paren_depth == 0 and brace_depth == 0:
                answers.append(current.strip())
                current = ""
                continue
            
            current += char
        
        if current.strip():
            answers.append(current.strip())
        
        return answers
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text, handling various formats"""
        if not text:
            return None
        
        text = text.strip()
        
        # Remove common units if present
        if self.answer_unit:
            text = text.replace(self.answer_unit, '').strip()
        
        # Try direct conversion
        try:
            return float(text)
        except ValueError:
            pass
        
        # Try to extract number from text
        # Handle scientific notation
        sci_pattern = r'(-?\d+\.?\d*)\s*[×x\\times]\s*10\^?\{?(-?\d+)\}?'
        sci_match = re.search(sci_pattern, text)
        if sci_match:
            mantissa = float(sci_match.group(1))
            exponent = float(sci_match.group(2))
            return mantissa * (10 ** exponent)
        
        # Handle fractions
        frac_pattern = r'\\frac\{(-?\d+)\}\{(\d+)\}'
        frac_match = re.search(frac_pattern, text)
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            if denominator != 0:
                return numerator / denominator
        
        # Simple fraction a/b
        simple_frac = r'(-?\d+)\s*/\s*(\d+)'
        match = re.search(simple_frac, text)
        if match:
            num = float(match.group(1))
            den = float(match.group(2))
            if den != 0:
                return num / den
        
        # Extract first number
        number_pattern = r'-?\d+\.?\d*'
        match = re.search(number_pattern, text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass
        
        return None
    
    def _extract_interval_tuple_items(self, text: str) -> List[str]:
        """Extract items from interval or tuple notation"""
        # Remove outer parentheses, brackets
        text = text.strip()
        text = re.sub(r'^[\[\(\{]', '', text)
        text = re.sub(r'[\]\)\}]$', '', text)
        
        # Split by comma, considering nested structures
        items = []
        current = ""
        depth = 0
        
        for char in text:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 0:
                items.append(current.strip())
                current = ""
                continue
            current += char
        
        if current.strip():
            items.append(current.strip())
        
        return items
    
    def _judge_answer(self, pred: str, gold: str) -> bool:
        """
        Judge answer correctness following OlympiadBench official logic
        
        Args:
            pred: Predicted answer
            gold: Gold answer
            
        Returns:
            bool: Whether the answer is correct
        """
        answer_type = self.answer_type.lower()
        
        # 1) Numeric type
        if answer_type == "numeric" or answer_type == "numerical":
            try:
                pred_val = self._extract_numeric_value(pred)
                gold_val = self._extract_numeric_value(gold)
                
                if pred_val is None or gold_val is None:
                    # Fallback to direct float conversion
                    pred_val = float(pred)
                    gold_val = float(gold)
                
                # Use absolute error with tolerance
                tolerance = float(self.error_tolerance) if self.error_tolerance else self.DEFAULT_NUMERIC_TOLERANCE
                return abs(pred_val - gold_val) <= tolerance
                
            except (ValueError, TypeError):
                return False
        
        # 2) Expression / Equation types
        elif answer_type in ["expression", "equation"]:
            if not SYMPY_AVAILABLE:
                # Fallback to string comparison
                return pred.strip() == gold.strip()
            
            try:
                pred_expr = sp.sympify(pred)
                gold_expr = sp.sympify(gold)
                return sp.simplify(pred_expr - gold_expr) == 0
            except Exception:
                # If SymPy fails, try string comparison
                return pred.strip() == gold.strip()
        
        # 3) Interval / Tuple types
        elif answer_type in ["interval", "tuple"]:
            if not SYMPY_AVAILABLE:
                return pred.strip() == gold.strip()
            
            try:
                # Extract items from interval/tuple notation
                pred_items = self._extract_interval_tuple_items(pred)
                gold_items = self._extract_interval_tuple_items(gold)
                
                if len(pred_items) != len(gold_items):
                    return False
                
                # Compare each item
                for p, g in zip(pred_items, gold_items):
                    try:
                        # Try numeric comparison first
                        p_val = float(p)
                        g_val = float(g)
                        tolerance = float(self.error_tolerance) if self.error_tolerance else self.DEFAULT_NUMERIC_TOLERANCE
                        if abs(p_val - g_val) > tolerance:
                            return False
                    except ValueError:
                        # Fall back to symbolic comparison
                        p_expr = sp.sympify(p)
                        g_expr = sp.sympify(g)
                        if sp.simplify(p_expr - g_expr) != 0:
                            return False
                
                return True
                
            except Exception:
                return pred.strip() == gold.strip()
        
        # Default: string comparison
        else:
            return pred.strip() == gold.strip()
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        Check if the answer is correct based on OlympiadBench rules
        
        Args:
            action: User's answer (raw model output)
            
        Returns:
            success: Whether the answer is correct
            feedback: Feedback message
        """
        if not action:
            return False, "No answer provided"
        
        # Extract boxed answer
        boxed_answer = self._extract_boxed_answer(str(action))
        
        if not boxed_answer:
            return False, "Could not find answer. Please use \\boxed{answer} format."
        
        # Get expected answer(s)
        if self.final_answers:
            expected_answers = self.final_answers if isinstance(self.final_answers, list) else [self.final_answers]
        else:
            expected_answers = [self.answer] if self.answer else []
        
        if not expected_answers:
            return False, "No correct answer available for validation"
        
        # Handle multiple answers
        if self.is_multiple_answer:
            user_answers = self._parse_multiple_answers(boxed_answer)
            
            if len(user_answers) != len(expected_answers):
                return False, f"Expected {len(expected_answers)} answers, got {len(user_answers)}"
            
            # Check each answer
            all_correct = True
            feedback_parts = []
            
            for i, (user_ans, expected_ans) in enumerate(zip(user_answers, expected_answers)):
                is_correct = self._judge_answer(user_ans, str(expected_ans))
                if not is_correct:
                    all_correct = False
                    feedback_parts.append(f"Answer {i+1}: Incorrect")
                else:
                    feedback_parts.append(f"Answer {i+1}: Correct")
            
            if all_correct:
                return True, "All answers correct!"
            else:
                return False, "; ".join(feedback_parts)
        
        else:
            # Single answer
            is_correct = self._judge_answer(boxed_answer, str(expected_answers[0]))
            if is_correct:
                return True, "Correct!"
            else:
                return False, f"Incorrect. Expected {expected_answers[0]}, got {boxed_answer}"
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Validate OlympiadBench task execution
        """
        # Call parent validation
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # Add OlympiadBench specific information
        info.update({
            "subfield": self.subfield,
            "answer_type": self.answer_type,
            "is_multiple_answer": self.is_multiple_answer,
            "has_error_tolerance": self.error_tolerance is not None,
            "error_tolerance": self.error_tolerance,
            "has_unit": self.answer_unit is not None,
            "unit": self.answer_unit,
            "has_context": bool(self.context),
            "language": self.language
        })
        
        # Extract boxed answer if available
        if info.get("answer_provided"):
            boxed = self._extract_boxed_answer(str(info["answer_provided"]))
            if boxed:
                info["boxed_answer"] = boxed
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get OlympiadBench specific metrics"""
        metrics = super().get_metrics()
        
        metrics.update({
            "subfield": self.subfield,
            "answer_type": self.answer_type,
            "problem_type": self.problem_type,
            "is_multiple_answer": self.is_multiple_answer,
            "has_error_tolerance": self.error_tolerance is not None,
            "has_unit": self.answer_unit is not None,
            "num_reference_solutions": len(self.solutions) if isinstance(self.solutions, list) else 0,
            "has_context": bool(self.context),
            "language": self.language
        })
        
        return metrics
    
    def get_tool_recommendations(self) -> List[str]:
        """Get recommended tools based on problem type"""
        recommendations = []
        
        # Geometry problems benefit from DiagramFormalizer
        if self.subfield == 'Geometry':
            recommendations.append('diagram_formalizer')
            recommendations.append('grounding_dino')  # For detecting geometric elements
        
        # Physics problems might have diagrams
        elif self.subfield in ['Mechanics', 'Electromagnetism', 'Optics']:
            recommendations.append('deepeyes')  # For zooming into diagrams
            recommendations.append('grounding_dino')  # For detecting objects
        
        # Problems with charts/graphs
        if any(word in self.question.lower() for word in ['graph', 'chart', 'plot', 'diagram']):
            recommendations.append('chartmoe')
        
        return recommendations