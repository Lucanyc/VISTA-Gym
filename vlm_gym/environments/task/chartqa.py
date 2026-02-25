# vlm_gym/environments/task/chartqa_task.py

from typing import Tuple, Dict, Any, List, Optional
import re

from .vision_qa_task import VisionQATask


class ChartQATask(VisionQATask):
    """
    ChartQA specific task
    
    Handles chart-related visual question answering tasks, including:
    - Data extraction
    - Trend analysis
    - Numerical calculation
    - Chart comparison
    """
    
    @classmethod
    def get_task_id(cls) -> str:
        """Get task type ID"""
        return "vlm-gym.chart-qa"
    
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """Set up ChartQA specific task"""
        # Call parent setup
        task_goal, task_info = super().setup()
        
        # Add ChartQA specific processing
        task_info["chart_type"] = self._detect_chart_type()
        task_info["requires_calculation"] = self._requires_calculation()
        task_info["data_extraction_needed"] = self._needs_data_extraction()
        task_info["dataset"] = "chartqa"
        
        # Modify task goal to include chart-specific guidance
        enhanced_goal = task_goal
        
        if task_info["requires_calculation"]:
            enhanced_goal += "\n\nNote: This question may require numerical calculations based on the chart data."
        
        if task_info["data_extraction_needed"]:
            enhanced_goal += "\nPlease carefully extract the relevant data points from the chart before answering."
        
        # Add ChartQA specific hints
        enhanced_goal += "\n\nWhen analyzing the chart, please:"
        enhanced_goal += "\n- Identify the chart type and axes"
        enhanced_goal += "\n- Read values accurately from the chart"
        enhanced_goal += "\n- Show your reasoning if calculations are needed"
        
        return enhanced_goal, task_info
    
    def _detect_chart_type(self) -> str:
        """Detect chart type"""
        question_lower = self.question.lower() if self.question else ""
        
        # Infer chart type based on question content
        chart_keywords = {
            "bar_chart": ["bar", "column", "bars"],
            "line_chart": ["line", "trend", "over time", "timeline"],
            "pie_chart": ["pie", "percentage", "portion", "share", "distribution"],
            "scatter_plot": ["scatter", "correlation", "relationship"],
            "area_chart": ["area", "cumulative"],
            "histogram": ["histogram", "frequency", "bins"]
        }
        
        for chart_type, keywords in chart_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return chart_type
        
        # Infer based on question patterns
        if "compare" in question_lower or "difference" in question_lower:
            return "comparative_chart"
        elif "total" in question_lower or "sum" in question_lower:
            return "aggregated_chart"
        
        return "unknown"
    
    def _requires_calculation(self) -> bool:
        """Determine if calculation is required"""
        if not self.question:
            return False
        
        calculation_keywords = [
            "calculate", "compute", "sum", "total", "average", "mean",
            "difference", "ratio", "percentage", "increase", "decrease",
            "compare", "how many", "how much", "what is the",
            "growth", "decline", "change", "rate"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in calculation_keywords)
    
    def _needs_data_extraction(self) -> bool:
        """Determine if data extraction is needed"""
        if not self.question:
            return False
        
        extraction_keywords = [
            "value", "data", "point", "highest", "lowest", "maximum",
            "minimum", "specific", "exact", "what is", "which", "find",
            "identify", "locate", "read"
        ]
        
        question_lower = self.question.lower()
        return any(keyword in question_lower for keyword in extraction_keywords)
    
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        Check ChartQA answer
        
        For numerical answers, allow a certain margin of error
        """
        if not action:
            return False, "No answer provided"
        
        # Clean answer format
        action = str(action).strip()
        
        # First try parent class check
        success, feedback = super().check_success(action)
        
        # If parent check fails, try ChartQA specific check
        if not success and self.answer:
            # Check numerical answer (allow margin of error)
            user_value = self._extract_number(action)
            correct_value = self._extract_number(str(self.answer))
            
            if user_value is not None and correct_value is not None:
                # Calculate relative error
                if correct_value == 0:
                    # Absolute error check
                    if abs(user_value - correct_value) <= 0.1:
                        return True, f"Correct! (exact match for zero value)"
                else:
                    relative_error = abs(user_value - correct_value) / abs(correct_value)
                    if relative_error <= 0.05:  # Allow 5% margin of error
                        return True, f"Correct! (within acceptable range: {correct_value:.2f} ± 5%)"
                    elif relative_error <= 0.1:  # Partial credit for 10% error
                        return True, f"Acceptable answer (within 10% of {correct_value:.2f})"
                
                return False, f"Incorrect. Expected approximately {correct_value:.2f}, got {user_value:.2f}"
            
            # Check unit conversion
            if self._check_unit_conversion(action, str(self.answer)):
                return True, "Correct! (with unit conversion)"
        
        return success, feedback
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text"""
        if not text:
            return None
        
        # Clean text
        text = text.lower()
        
        # Remove common interfering characters
        text = text.replace(',', '').replace('%', '').replace('$', '')
        
        # Handle percentage
        is_percentage = '%' in text or 'percent' in text
        
        # Handle units
        unit_multipliers = {
            'k': 1000, 'thousand': 1000, 'thousands': 1000,
            'm': 1000000, 'million': 1000000, 'millions': 1000000,
            'b': 1000000000, 'billion': 1000000000, 'billions': 1000000000
        }
        
        multiplier = 1
        for unit, value in unit_multipliers.items():
            if unit in text:
                multiplier = value
                text = text.replace(unit, '')
        
        # Find number pattern
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                # Return the first number found
                number = float(matches[0])
                if is_percentage and number > 1:
                    number = number / 100
                return number * multiplier
            except ValueError:
                pass
        
        return None
    
    def _check_unit_conversion(self, user_answer: str, correct_answer: str) -> bool:
        """Check if the difference is caused by unit conversion"""
        user_value = self._extract_number(user_answer)
        correct_value = self._extract_number(correct_answer)
        
        if user_value is None or correct_value is None:
            return False
        
        # Check common unit conversion ratios
        common_ratios = [1000, 100, 12, 60, 24, 365]  # K/M, %, months/year, min/hour, hour/day, day/year
        
        for ratio in common_ratios:
            if abs(user_value * ratio - correct_value) < 0.01 * correct_value:
                return True
            if abs(user_value / ratio - correct_value) < 0.01 * correct_value:
                return True
        
        return False
    
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Validate ChartQA task execution
        """
        # Call parent validation
        reward, done, message, info = super().validate(
            chat_history, observation, full_history
        )
        
        # Add ChartQA specific information
        info["chart_type"] = self._detect_chart_type()
        info["required_calculation"] = self._requires_calculation()
        info["data_extraction"] = self._needs_data_extraction()
        
        # If numerical answer, add extracted numerical value info
        if info.get("answer_provided"):
            extracted_value = self._extract_number(str(info["answer_provided"]))
            if extracted_value is not None:
                info["extracted_numerical_value"] = extracted_value
        
        return reward, done, message, info
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ChartQA specific metrics"""
        metrics = super().get_metrics()
        
        metrics.update({
            "chart_type": self._detect_chart_type(),
            "requires_calculation": self._requires_calculation(),
            "needs_data_extraction": self._needs_data_extraction(),
            "is_numerical_answer": self._extract_number(str(self.answer)) is not None if self.answer else False,
            "question_complexity": self._assess_complexity()
        })
        
        return metrics
    
    def _assess_complexity(self) -> str:
        """Assess question complexity"""
        if not self.question:
            return "unknown"
        
        question_lower = self.question.lower()
        
        # Complex question features
        complex_features = [
            "compare", "calculate", "trend", "correlation",
            "percentage change", "growth rate", "difference between"
        ]
        
        # Simple question features
        simple_features = [
            "what is", "which", "highest", "lowest", "maximum", "minimum"
        ]
        
        complex_count = sum(1 for f in complex_features if f in question_lower)
        simple_count = sum(1 for f in simple_features if f in question_lower)
        
        if complex_count >= 2:
            return "high"
        elif complex_count >= 1 or (self._requires_calculation() and self._needs_data_extraction()):
            return "medium"
        elif simple_count >= 1:
            return "low"
        else:
            return "medium"
