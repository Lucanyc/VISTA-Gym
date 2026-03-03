# /data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/task/geos.py

from typing import Tuple, Dict, Any, List, Optional
import re
import math
import numpy as np


from .vision_qa_task import VisionQATask


class GEOSTask(VisionQATask):
    """
    GEOS (Geometry) dataset specific task.

    This class handles geometry problem-solving tasks, which involve
    understanding a geometric diagram and a textual problem to find a
    numerical or symbolic answer. It provides specialized setup,
    instructions, and answer-checking logic for geometry.
    """

    @classmethod
    def get_task_id(cls) -> str:
        """Get the unique identifier for the GEOSTask."""
        return "vlm-gym.geos"

    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """
        Set up the GEOS-specific task environment.

        This enhances the base task by adding geometry-specific context,
        instructions, and metadata.
        """
        # Call the parent setup method to get the base goal and info
        task_goal, task_info = super().setup()

        # Analyze the geometry problem to add specific metadata
        self.concept = self._get_geometric_concept()
        self.requires_calculation = self._requires_calculation()

        task_info["geometric_concept"] = self.concept
        task_info["requires_calculation"] = self.requires_calculation
        task_info["dataset"] = "geos"

        # Enhance the task goal with specific instructions for solving geometry problems
        enhanced_goal = self._enhance_task_goal(task_goal)

        return enhanced_goal, task_info

    def _enhance_task_goal(self, base_goal: str) -> str:
        """Adds GEOS-specific instructions to the base task goal."""
        enhanced_parts = [
            base_goal,
            "\n**This is a geometry problem-solving task.**",
            "\n**Instructions to solve the problem:**",
            "1. **Analyze the Diagram and Text**: Carefully examine the geometric figure and read the problem description to understand the given information and what is being asked.",
            "2. **Identify Shapes and Properties**: Recognize the geometric shapes (e.g., circles, triangles) and their properties.",
            "3. **Apply Geometric Theorems**: Use relevant geometric theorems and formulas (e.g., Pythagorean theorem, circle properties, area formulas) to formulate a solution.",
            "4. **Calculate and Finalize**: Perform the necessary calculations to find the final answer. Your answer should be a specific value or expression.",
        ]

        if self.concept != 'unknown':
            enhanced_parts.append(f"\n**Hint**: This problem likely involves properties of a **{self.concept}**.")

        enhanced_parts.append("\n**Answer Format**: Provide the final numerical answer or mathematical expression (e.g., '8' or '2*sqrt(5)').")

        return "\n".join(enhanced_parts)

    def _get_geometric_concept(self) -> str:
        """Classify the geometry question based on keywords."""
        if not self.question:
            return 'unknown'
            
        q_lower = self.question.lower()
        if any(word in q_lower for word in ['circle', 'radius', 'diameter', 'chord', 'tangent', 'arc']):
            return 'circle'
        if any(word in q_lower for word in ['triangle', 'hypotenuse', 'pythagorean', 'isosceles', 'equilateral']):
            return 'triangle'
        if any(word in q_lower for word in ['angle', 'degree', 'radian', 'perpendicular', 'parallel']):
            return 'angle_relation'
        if any(word in q_lower for word in ['area', 'surface']):
            return 'area'
        if any(word in q_lower for word in ['length', 'perimeter', 'distance', 'side', 'find ae']):
            return 'length_perimeter'
        if any(word in q_lower for word in ['square', 'rectangle', 'quadrilateral', 'rhombus', 'trapezoid']):
            return 'quadrilateral'
        if any(word in q_lower for word in ['volume', 'cube', 'sphere', 'cylinder', 'cone']):
            return '3d_shape'
        return 'unknown'
        
    def _requires_calculation(self) -> bool:
        """Determine if the question requires calculation. For GEOS, this is almost always true."""
        if not self.question:
            return False
        # Keywords that strongly imply calculation
        calc_keywords = [
            'find', 'calculate', 'what is the', 'value', 'length', 'area',
            'volume', 'perimeter', 'angle', 'distance'
        ]
        q_lower = self.question.lower()
        return any(key in q_lower for key in calc_keywords)

    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        Check the correctness of the answer for a GEOS task.

        This method can handle both plain numerical answers and mathematical
        expressions involving functions like sqrt.
        """
        if action is None or str(action).strip() == "":
            return False, "No answer provided."

        user_answer_str = str(action).strip()
        correct_answer_str = str(self.answer).strip()
        
        # Attempt to evaluate answers as mathematical expressions
        try:
            user_value = self._evaluate_math_expr(user_answer_str)
            correct_value = self._evaluate_math_expr(correct_answer_str)

            # If both are valid numbers, compare them with a tolerance
            if user_value is not None and correct_value is not None:
                if math.isclose(user_value, correct_value, rel_tol=1e-4, abs_tol=1e-4):
                    return True, f"Correct! The calculated value is approximately {correct_value:.4f}."
                else:
                    return False, f"Incorrect. Your calculated value {user_value:.4f} is not close to the correct value of {correct_value:.4f}."
        
        except (SyntaxError, TypeError, NameError):
            # This block will be reached if expressions are not valid, e.g., contain unknown symbols
            pass

        # Fallback to case-insensitive string comparison if evaluation fails or is not applicable
        if user_answer_str.lower() == correct_answer_str.lower():
            return True, f"Correct! The answer is '{correct_answer_str}'."

        return False, f"Incorrect. The correct answer is '{correct_answer_str}'."

    def _evaluate_math_expr(self, expr: str) -> Optional[float]:
        """
        Safely evaluate a string containing a mathematical expression.
        Handles numbers, basic operators, and common math functions.
        """
        if expr is None:
            return None
            
        # Security: A dictionary of safe functions and constants
        safe_dict = {
            'sqrt': math.sqrt,
            'pi': math.pi,
            'cos': math.cos,
            'sin': math.sin,
            'tan': math.tan,
            'acos': math.acos,
            'asin': math.asin,
            'atan': math.atan,
            'e': math.e,
            'log': math.log,
            'log10': math.log10,
            # numpy can also be used for more functions if needed
            'np': np 
        }

        # Normalize expression for evaluation
        # Replace `^` with `**` for exponentiation
        expr = expr.lower().replace('^', '**')

        try:
            # Evaluate the expression within the safe context
            # __builtins__ is restricted to prevent calling unsafe functions
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return float(result)
        except (SyntaxError, TypeError, NameError, KeyError, ValueError, ZeroDivisionError):
            # If eval fails, it's not a simple mathematical expression
            # or it's malformed. Return None.
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get GEOS-specific metrics for analysis."""
        metrics = super().get_metrics()
        metrics.update({
            "geometric_concept": self.concept,
            "requires_calculation": self.requires_calculation,
        })
        return metrics