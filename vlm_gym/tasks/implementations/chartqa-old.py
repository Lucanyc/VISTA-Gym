#!/usr/bin/env python3
"""ChartQA task implementation"""

import re
from typing import Dict, Any, Optional
from ..base.task_wrapper import BaseTaskWrapper
from ..base.registry import register_task
from ..components.evaluators.chartqa import ChartQAEvaluator


@register_task("chartqa")
class ChartQATask(BaseTaskWrapper):
    """ChartQA task wrapper"""
    
    task_type = "chartqa"
    
    def _post_init(self, config: Dict[str, Any]):
        """Additional initialization for ChartQA"""
        # Numerical tolerance
        self.numerical_tolerance = config.get('numerical_tolerance', 0.05)
        
        # Last extra info (for structured reasoning)
        self.last_extra_info = None
        self.structured_reasoning_history = []
    
    def get_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get ChartQA-specific components"""
        return {
            "evaluator": ChartQAEvaluator(tolerance=config.get('numerical_tolerance', 0.05)),
            "extractor": self,  # Use self for now, can be separated later
            "classifier": self,  # Use self for now
            "feedback": self,   # Use self for now
            "formatter": self    # Use self for now
        }
    
    def get_task_description(self) -> str:
        """Get task description"""
        return f"Answer the question about the chart: {self.current_question}"
    
    def get_output_format_instruction(self) -> str:
        """Get output format instruction"""
        base_instruction = """Please structure your response in the following format:
<think>
[Your step-by-step reasoning process here. Analyze the chart, identify relevant data points, and perform any necessary calculations.]
</think>
<answer>
[Your final answer here. Provide only the answer without any additional explanation.]
</answer>"""
        
        # Add tool instructions if enabled
        if self.enable_chartmoe:
            base_instruction += """

IMPORTANT: You MUST use the ChartMoE tool for this chart question.

You must use the ChartMoE tool to understand charts:
<tool_call>
{"tool": "chartmoe", "task": "to_table"}
</tool_call>

Available ChartMoE tasks:
- "to_table" - Convert chart to structured table
- "describe" - Generate detailed chart description  
- "extract_data" - Extract all numerical data
- "summarize" - Provide chart summary
- "analyze" - Deep analysis with insights
- Custom question: {"tool": "chartmoe", "prompt": "your specific question"}"""
        
        if self.enable_grounding_dino:
            base_instruction += """

You can use the Grounding DINO tool to detect and locate objects in the chart:
<tool_call>
{"tool": "grounding_dino", "parameters": {"caption": "bar chart bars"}}
</tool_call>"""
        
        if self.enable_deepeyes:
            base_instruction += """

You can also use visual tools to zoom into specific regions if needed:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox": [x1, y1, x2, y2]}}
</tool_call>"""
        
        return base_instruction
    
    def classify_question(self) -> str:
        """Classify the question type"""
        question_lower = self.current_question.lower()
        
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
        elif any(word in question_lower for word in ['value', 'what is the', 'how much']):
            return 'numerical'
        elif any(word in question_lower for word in ['compare', 'which is']):
            return 'comparison'
        elif any(word in question_lower for word in ['maximum', 'minimum', 'highest', 'lowest', 'largest', 'smallest']):
            return 'minmax'
        elif any(word in question_lower for word in ['trend', 'increase', 'decrease', 'change']):
            return 'trend'
        elif any(word in question_lower for word in ['yes', 'no', 'is', 'are', 'does', 'do']):
            return 'yes_no'
        elif any(word in question_lower for word in ['what', 'which', 'when', 'where', 'who']):
            return 'retrieval'
        else:
            return 'other'
    
    def extract_answer(self, action: str) -> str:
        """Extract answer from action"""
        if not action:
            return ""
        
        # Handle dictionary action
        if isinstance(action, dict):
            if action.get('type') == 'answer':
                return str(action.get('content', ''))
            elif 'answer' in action:
                return str(action['answer'])
            elif 'result' in action:
                return str(action['result'])
            elif 'value' in action:
                return str(action['value'])
        
        # String action
        if isinstance(action, str):
            action_str = self._clean_action_content(action)
            
            # Extract from structured format
            if self.use_structured_output:
                think_content, answer_content = self._extract_think_and_answer(action_str)
                if answer_content:
                    return answer_content
                elif think_content:
                    action_str = think_content
            
            # Extract from answer_question format
            if 'answer_question(answer="' in action:
                start = action.find('answer_question(answer="') + len('answer_question(answer="')
                end = action.find('")', start)
                if end > start:
                    return action[start:end].strip()
            
            # Question type specific extraction
            q_type = self.classify_question()
            
            if q_type == 'counting':
                # Look for numbers in specific patterns
                patterns = [
                    r'(?:is|are)\s+(\d+)(?:\.|,|;|$|\s)',
                    r'number[^.]*?is\s+(\d+)',
                    r'there (?:are|is) (\d+)',
                    r'found (\d+)',
                    r'counted (\d+)',
                    r'total of (\d+)',
                    r'(\d+) values',
                    r'answer is (\d+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, action_str, re.IGNORECASE)
                    if match:
                        return match.group(1)
                
                # Last resort - find any number
                numbers = re.findall(r'\b(\d+)\b', action_str)
                if numbers:
                    return numbers[-1]
            
            # Try to extract final answer patterns
            answer_patterns = [
                r'(?:Final )?Answer:\s*(.+?)(?:\n|$)',
                r'The answer is[:\s]+(.+?)(?:\n|$)',
                r'"([^"]+)"',
                r"'([^']+)'"
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, action_str, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # Short direct answer
            if len(action_str) < 20 and not action_str.endswith('.'):
                return action_str
            
            return action_str.strip()
        
        return str(action).strip()
    
    def evaluate_answer(self, answer: str) -> bool:
        """Evaluate if answer is correct"""
        return self.components["evaluator"].evaluate(answer, self.answer)
    
    def generate_feedback(self, answer: str) -> str:
        """Generate feedback for incorrect answer"""
        feedback_parts = []
        feedback_parts.append(f"Your answer '{answer}' is incorrect.")
        
        q_type = self.classify_question()
        
        # Analyze error type
        pred_num = self.components["evaluator"]._extract_number(answer)
        gt_num = self.components["evaluator"]._extract_number(self.answer)
        
        if q_type == 'counting' and pred_num is not None and gt_num is not None:
            error_rate = abs(pred_num - gt_num) / gt_num if gt_num != 0 else 1
            if error_rate > 0.3:
                feedback_parts.append(f"Your count is significantly off.")
        elif q_type in ['numerical', 'minmax', 'average', 'percentage']:
            if pred_num is not None and gt_num is not None:
                if pred_num > gt_num:
                    feedback_parts.append("Your answer is too high.")
                else:
                    feedback_parts.append("Your answer is too low.")
        
        # Tool suggestions
        if self.enable_chartmoe and len(self.chartmoe_history) == 0:
            feedback_parts.append("\n⚠️ You MUST use the ChartMoE tool!")
            feedback_parts.append('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>')
        elif self.enable_chartmoe and len(self.chartmoe_history) > 0:
            feedback_parts.append("Try a different ChartMoE task or custom prompt.")
        
        feedback_parts.append("Please examine the chart again and provide a revised answer.")
        
        if self.use_structured_output:
            feedback_parts.append("Remember to use <think> and <answer> tags.")
        
        remaining = self.max_attempts - self.current_attempt
        if remaining > 1:
            feedback_parts.append(f"You have {remaining} attempts remaining.")
        else:
            feedback_parts.append("This is your last attempt.")
        
        return " ".join(feedback_parts)
    
    def generate_hint(self) -> str:
        """Generate hint for final attempt"""
        q_type = self.classify_question()
        
        hints = {
            'counting': "Count each item systematically.",
            'summation': "Add all relevant values together.",
            'average': "Sum all values, then divide by count.",
            'percentage': "Calculate (part ÷ whole) × 100.",
            'difference': "Find the two values and subtract.",
            'ratio': "Divide one value by another.",
            'numerical': "Read the exact value from the chart.",
            'comparison': "Compare the exact values.",
            'minmax': "Check every data point.",
            'trend': "Look at the overall direction.",
            'retrieval': "Look for the specific label.",
            'other': "Read the question carefully."
        }
        
        hint = hints.get(q_type, "Analyze the chart thoroughly.")
        
        # Add tool hints
        if self.enable_chartmoe and len(self.chartmoe_history) == 0:
            hint += "\n\nUse ChartMoE to extract data: "
            hint += '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>'
        
        if self.use_structured_output:
            hint += "\n\nFormat: Use <think> for reasoning and <answer> for final answer."
        
        return hint
    
    def should_force_tool(self) -> Optional[str]:
        """Check if should force tool usage in reflection"""
        # Force ChartMoE if not used yet
        if self.current_attempt >= 1:
            if self.enable_chartmoe and len(self.chartmoe_history) == 0:
                return "chartmoe"
            elif self.enable_grounding_dino and len(self.grounding_dino_history) == 0:
                return "grounding_dino"
        return None
    
    def get_tool_force_reminder(self, tool_name: str) -> str:
        """Get reminder for forced tool usage"""
        if tool_name == "chartmoe":
            return (
                "⚠️ YOU MUST use ChartMoE tool first!\n"
                'Start with: <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n'
                "Then analyze the extracted data to answer the question."
            )
        elif tool_name == "grounding_dino":
            return (
                "⚠️ YOU MUST use Grounding DINO tool first!\n"
                'Start with: <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "chart data"}}</tool_call>\n'
                "Then use the detection results to answer."
            )
        return super().get_tool_force_reminder(tool_name)
    
    # Helper methods
    
    def _clean_action_content(self, action: str) -> str:
        """Clean action content"""
        if not isinstance(action, str):
            return str(action)
        
        # Remove answer_question wrapper
        if action.startswith('answer_question(answer="') and action.endswith('")'):
            action = action[len('answer_question(answer="'):-2]
            action = action.replace('\\"', '"')
        
        # Remove noise
        action = re.sub(r'(\s*addCriterion\s*)+', ' ', action)
        action = re.sub(r'(\s*<\/\s*addCriterion\s*)+', ' ', action)
        action = re.sub(r'\n\s*\n', '\n', action)
        action = re.sub(r' +', ' ', action)
        
        return action.strip()
    
    def _extract_think_and_answer(self, action: str) -> tuple:
        """Extract thinking and answer from structured output"""
        think_content = ""
        answer_content = ""
        
        cleaned_action = self._clean_action_content(action)
        
        # Extract <think> content
        think_match = re.search(r'<think>(.*?)</think>', cleaned_action, re.DOTALL | re.IGNORECASE)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # Extract <answer> content
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_action, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        
        return think_content, answer_content