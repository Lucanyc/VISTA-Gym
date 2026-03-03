#!/usr/bin/env python3
"""ChartQA task implementation"""

import re
from typing import Dict, Any, Optional, List
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
    
    def get_expected_format_tags(self) -> Dict[str, List[str]]:
        """ChartQA uses think/answer format exclusively
        
        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        return {
            "reasoning": ["<think>"],  # Only accept <think> tags
            "answer": ["<answer>"]
        }
    
    def get_task_description(self) -> str:
        """Get task description"""
        return f"Answer the question about the chart: {self.current_question}"
    


    def get_output_format_instruction(self) -> str:
        """Get output format instruction"""
        if self.enable_chartmoe:
            base_instruction = """⚠️ CRITICAL REQUIREMENT: You MUST use ChartMoE tool. Direct answers are FORBIDDEN!

    MANDATORY WORKFLOW:
    1. YOU MUST FIRST call ChartMoE to extract chart data
    2. YOU CANNOT answer without using the tool
    3. Direct visual reading is NOT ALLOWED

    Step 1 - REQUIRED tool call:
    <tool_call>
    {"tool": "chartmoe", "task": "to_table"}
    </tool_call>

    Step 2 - ONLY AFTER receiving tool response:
    <think>
    [Analyze the extracted data from ChartMoE]
    </think>
    <answer>
    [Your answer based on ChartMoE data]
    </answer>

    Available ChartMoE tasks:
    - "to_table" - Extract chart as table (RECOMMENDED)
    - "extract_data" - Extract numerical values
    - "describe" - Get chart description
    - Custom: {"tool": "chartmoe", "prompt": "specific question"}

    WARNING: Answering without ChartMoE will be marked as INVALID!
    DO NOT skip the tool call!
    DO NOT answer based on visual observation alone!"""
        else:
            base_instruction = """Please structure your response:
    <think>
    [Your reasoning process]
    </think>
    <answer>
    [Your final answer]
    </answer>"""
        
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
            
            # Extract from structured format (prioritize <answer> tags)
            if self.use_structured_output:
                think_content, answer_content = self._extract_think_and_answer(action_str)
                if answer_content:
                    return answer_content
                # Don't fall back to think content for answer
            
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
        
        # CRITICAL: Check ChartMoE tool usage first - this is MANDATORY
        if self.enable_chartmoe and len(self.chartmoe_history) == 0:
            # Tool not used - this is a critical violation
            feedback_parts.append("CRITICAL VIOLATION: You did NOT use the ChartMoE tool!")
            feedback_parts.append("Direct answers without tool usage are STRICTLY FORBIDDEN!")
            feedback_parts.append("")
            feedback_parts.append("MANDATORY REQUIREMENT:")
            feedback_parts.append("• You MUST call ChartMoE FIRST - NO EXCEPTIONS")
            feedback_parts.append("• Visual reading alone is NOT ACCEPTABLE")
            feedback_parts.append("• This is NOT optional - it is REQUIRED")
            feedback_parts.append("")
            feedback_parts.append("EXECUTE THIS IMMEDIATELY:")
            feedback_parts.append('<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>')
            feedback_parts.append("")
            feedback_parts.append("After receiving the tool response, then provide your answer using:")
            feedback_parts.append("<think>[Analysis of ChartMoE data]</think>")
            feedback_parts.append("<answer>[Your answer based on tool data]</answer>")
            feedback_parts.append("")
            feedback_parts.append("⚠️ Your answer is INVALID without ChartMoE data!")
            feedback_parts.append(f"You have {self.max_attempts - self.current_attempt} attempts remaining.")
            return " ".join(feedback_parts)
        
        # If tool was used but answer is still incorrect
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
        
        # If ChartMoE was used but answer is wrong, suggest different tasks
        if self.enable_chartmoe and len(self.chartmoe_history) > 0:
            feedback_parts.append("\nYou used ChartMoE, but the answer is still incorrect.")
            feedback_parts.append("Try a different ChartMoE task:")
            
            # Suggest different ChartMoE tasks based on question type
            if q_type in ['counting', 'summation']:
                feedback_parts.append('• <tool_call>{"tool": "chartmoe", "task": "extract_data"}</tool_call>')
                feedback_parts.append("  This will extract all numerical values.")
            elif q_type in ['comparison', 'minmax']:
                feedback_parts.append('• <tool_call>{"tool": "chartmoe", "task": "compare"}</tool_call>')
                feedback_parts.append("  This will compare data series.")
            elif q_type == 'trend':
                feedback_parts.append('• <tool_call>{"tool": "chartmoe", "task": "trend"}</tool_call>')
                feedback_parts.append("  This will identify trends in the data.")
            elif q_type in ['yes_no', 'numerical']:
                feedback_parts.append('• <tool_call>{"tool": "chartmoe", "prompt": "' + self.current_question + '"}</tool_call>')
                feedback_parts.append("  This will directly answer your specific question.")
            else:
                feedback_parts.append('• <tool_call>{"tool": "chartmoe", "task": "analyze"}</tool_call>')
                feedback_parts.append("  This will provide deep analysis of the chart.")
        
        feedback_parts.append("\nPlease analyze the chart data again and provide a revised answer.")
        
        if self.use_structured_output:
            feedback_parts.append("Remember to use <think> for your reasoning and <answer> for your final answer.")
        
        remaining = self.max_attempts - self.current_attempt
        if remaining > 1:
            feedback_parts.append(f"\nYou have {remaining} attempts remaining.")
        else:
            feedback_parts.append("\n⚠️ This is your LAST attempt. Please be careful and thorough.")
            if self.enable_chartmoe and len(self.chartmoe_history) > 0:
                feedback_parts.append("Consider using a custom prompt with the exact question:")
                feedback_parts.append('<tool_call>{"tool": "chartmoe", "prompt": "' + self.current_question + '"}</tool_call>')
        
        return " ".join(feedback_parts)
    
    
    
    
    def generate_hint(self) -> str:
        """Generate hint for final attempt"""
        q_type = self.classify_question()
        
        hints = {
            'counting': "Count each item in the chart systematically. Look at each bar/point individually.",
            'summation': "Add all relevant values together. Make sure you include all data points.",
            'average': "Sum all values, then divide by the count. Don't forget any values.",
            'percentage': "Calculate (part ÷ whole) × 100. Identify the correct part and whole.",
            'difference': "Find the two values and subtract. Make sure you identify the right values.",
            'ratio': "Divide one value by another. Check which value should be numerator.",
            'numerical': "Read the exact value from the chart. Look at the axis labels carefully.",
            'comparison': "Compare the exact values. Identify all relevant data points.",
            'minmax': "Check every data point to find the extreme value.",
            'trend': "Look at the overall direction of change over time.",
            'yes_no': "Check if the statement is true or false based on the chart data.",
            'retrieval': "Look for the specific label or category mentioned in the question.",
            'other': "Read the question carefully and identify what information is needed."
        }
        
        hint = hints.get(q_type, "Analyze the chart thoroughly.")
        
        # Add ChartMoE tool hint if not used
        if self.enable_chartmoe and len(self.chartmoe_history) == 0:
            hint += "\n\n💡 HINT: Use ChartMoE to extract structured data: "
            hint += '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>'
            hint += "\nThis will give you all the data in an easy-to-read table format."
        
        if self.use_structured_output:
            hint += "\n\n📝 Format reminder: Use <think> for your reasoning process and <answer> for your final answer only."
        
        return hint
    
    def should_force_tool(self) -> Optional[str]:
        """Check if should force tool usage in reflection"""
        # 从第一次就强制使用工具
        if self.enable_chartmoe and len(self.chartmoe_history) == 0:
            return "chartmoe"
        return None
    
    def get_tool_force_reminder(self, tool_name: str) -> str:
        """Get reminder for forced tool usage"""
        if tool_name == "chartmoe":
            return (
                "MANDATORY REQUIREMENT VIOLATED!\n"
                "You MUST use ChartMoE - NO EXCEPTIONS!\n"
                "Direct answers are FORBIDDEN!\n\n"
                "EXECUTE IMMEDIATELY:\n"
                '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>\n\n'
                "You CANNOT answer without tool data.\n"
                "This is a STRICT REQUIREMENT, not a suggestion!"
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