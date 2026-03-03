#!/usr/bin/env python3
"""ChartMoE tool handler for chart understanding and analysis"""

from typing import Dict, Any, Tuple, Optional
import json
import re
import logging
from ..base import BaseTool, ToolConfig, ToolResult

logger = logging.getLogger(__name__)


class ChartMoETool(BaseTool):
    """ChartMoE tool handler for chart analysis"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.supported_tasks = [
            'to_table', 'describe', 'extract_data', 'summarize', 
            'analyze', 'compare', 'trend', 'to_text', 'answer'
        ]
    
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if ChartMoE can handle this task"""
        # ChartQA tasks should always use ChartMoE if available
        if observation.get("is_visual_question") and observation.get("chartmoe_enabled"):
            return True, 0.95
        
        # Check for explicit ChartMoE request
        if observation.get("must_use_tool") and observation.get("tool_to_use") == "chartmoe":
            return True, 1.0
        
        question = observation.get("question", "").lower()
        
        # Chart-related keywords
        chart_keywords = [
            'chart', 'graph', 'plot', 'bar', 'line', 'pie',
            'axis', 'legend', 'data', 'trend', 'value', 'table'
        ]
        
        # Count matching keywords
        keyword_count = sum(1 for kw in chart_keywords if kw in question)
        
        if keyword_count == 0:
            return False, 0.0
        
        # Calculate confidence based on keywords
        confidence = min(keyword_count * 0.2, 0.9)
        
        # Boost confidence for specific question types
        question_type = observation.get("question_type", "")
        if question_type in ['counting', 'numerical', 'summation', 'percentage', 'comparison']:
            confidence = max(confidence, 0.8)
        
        return confidence > 0, confidence
    
    def build_prompt(self, observation: Dict[str, Any]) -> str:
        """Build ChartMoE tool call prompt"""
        question = observation.get("question", "")
        question_type = observation.get("question_type", "")
        
        # Determine best task based on question
        task = self._select_task(question, question_type, observation)
        
        # Build tool call
        if task == "answer":
            # Custom question mode
            tool_call = {
                "tool": "chartmoe",
                "task": "answer",
                "prompt": question
            }
        else:
            # Standard task mode
            tool_call = {
                "tool": "chartmoe",
                "task": task
            }
        
        logger.debug(f"ChartMoE task selected: {task}")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def process_result(self, raw_result: Any, observation: Dict[str, Any]) -> ToolResult:
        """Process ChartMoE output - 修正版本，正确处理tool_feedback格式"""
        if isinstance(raw_result, dict):
            # 兼容不同的结果格式
            # tool_feedback格式: {"tool": "chartmoe", "task_type": "to_table", "output": "...", ...}
            # 或者直接的结果格式: {"task_type": "...", "output": "...", "success": ...}
            
            task_type = raw_result.get("task_type", "unknown")
            
            # 获取输出内容 - 支持多种字段名
            output = raw_result.get("output", "") or raw_result.get("processed_output", "") or raw_result.get("result", "")
            
            # 判断成功的逻辑：
            # 1. 如果有explicit的success字段，使用它
            # 2. 否则，如果有非空的output，认为成功
            if "success" in raw_result:
                success = raw_result.get("success", False)
            else:
                success = bool(output and len(output.strip()) > 0)
            
            # Parse output based on task type
            data = {
                "task_type": task_type,
                "raw_output": output
            }
            
            if task_type == "to_table" and output:
                # Parse table data
                data["table_data"] = self._parse_table(output)
                data["has_structured_data"] = True
            
            elif task_type == "extract_data" and output:
                # Parse extracted data
                data["extracted_values"] = self._extract_values(output)
                data["has_numerical_data"] = True
            
            elif task_type == "analyze" and output:
                # Parse analysis
                data["analysis"] = output
                data["insights"] = self._extract_insights(output)
            
            else:
                # Generic output
                data["content"] = output
            
            return ToolResult(
                success=success,
                data=data,
                metadata={
                    "task_type": task_type,
                    "output_length": len(output) if output else 0
                }
            )
        
        return ToolResult(
            success=False,
            data=None,
            error="Invalid ChartMoE result format"
        )
    
    def format_for_answer(self, result: ToolResult, observation: Dict[str, Any]) -> str:
        """Format ChartMoE result for final answer generation - 增强版本，更好地引导CoT"""
        
        # 错误处理
        if not result.success or not result.data:
            error_msg = result.error if result.error else "ChartMoE tool did not return valid data"
            return f"""The ChartMoE tool encountered an issue: {error_msg}

Please analyze the chart visually and answer the question:
"{observation.get('question', '')}"

You MUST provide your answer inside <answer> tags."""
        
        data = result.data
        task_type = data.get("task_type", "unknown")
        raw_output = data.get("raw_output", "")
        
        # 确保有输出
        if not raw_output or len(raw_output.strip()) == 0:
            return f"""ChartMoE did not extract any data from the chart.

Please analyze the chart visually and answer the question:
"{observation.get('question', '')}"

You MUST provide your answer inside <answer> tags."""
        
        # 基于任务类型格式化输出
        if task_type == "to_table":
            return self._format_table_result_enhanced(data, observation)
        
        elif task_type == "extract_data" and data.get("extracted_values"):
            return self._format_extraction_result_enhanced(data, observation)
        
        elif task_type == "analyze":
            return self._format_analysis_result_enhanced(data, observation)
        
        else:
            # 通用格式，但更结构化，引导CoT推理
            return self._format_generic_result(data, observation)
    
    def _format_table_result_enhanced(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Enhanced format for table results - 引导chain of thought推理"""
        raw_output = data.get("raw_output", "")
        question = observation.get("question", "")
        table_data = data.get("table_data", {})
        
        # 构建引导CoT的prompt
        prompt = f"""Based on the extracted table data from ChartMoE:

{raw_output}

Now answer the question: "{question}"

Instructions:
1. Use the exact values from the table above
2. For counting questions, count the relevant entries
3. For comparison questions, compare the values directly
4. For yes/no questions, verify against the table data

"""
        
        # 如果有数值，添加统计信息帮助推理
        if table_data.get("values"):
            values = table_data["values"]
            if len(values) > 0:
                prompt += f"""
Helpful statistics from the table:
- Total numerical values: {len(values)}
- Sum: {sum(values):.2f}
- Average: {sum(values)/len(values):.2f}
- Min: {min(values):.2f}, Max: {max(values):.2f}

"""
        
        prompt += """Please think step by step:
1. First, identify the relevant data from the table
2. Then, perform any necessary calculations or comparisons
3. Finally, provide your answer

You MUST output your answer inside <answer> tags.
For example:
- For numbers: <answer>42</answer>
- For yes/no: <answer>Yes</answer>
- For text: <answer>The highest value is X</answer>"""
        
        return prompt
    
    def _format_extraction_result_enhanced(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Enhanced format for extraction results"""
        values = data.get("extracted_values", [])
        raw_output = data.get("raw_output", "")
        question = observation.get("question", "")
        
        prompt = f"""Based on the data extracted by ChartMoE:

{raw_output}

"""
        
        if values:
            prompt += f"""Extracted numerical values: {values}

Statistics:
- Count: {len(values)}
- Sum: {sum(values):.2f}
- Average: {sum(values)/len(values):.2f}
- Min: {min(values):.2f}, Max: {max(values):.2f}

"""
        
        prompt += f"""Now answer the question: "{question}"

Please analyze the data step by step and provide your answer.

You MUST output your answer inside <answer> tags."""
        
        return prompt
    
    def _format_analysis_result_enhanced(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Enhanced format for analysis results"""
        analysis = data.get("analysis", "")
        insights = data.get("insights", [])
        question = observation.get("question", "")
        
        prompt = f"""Based on ChartMoE's analysis:

{analysis}
"""
        
        if insights:
            prompt += "\nKey insights:\n"
            for i, insight in enumerate(insights, 1):
                prompt += f"{i}. {insight}\n"
        
        prompt += f"""
Now answer the question: "{question}"

Use the analysis above to formulate your answer.

You MUST output your answer inside <answer> tags."""
        
        return prompt
    
    def _format_generic_result(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Generic format for other task types"""
        raw_output = data.get("raw_output", "")
        question = observation.get("question", "")
        task_type = data.get("task_type", "unknown")
        
        return f"""Based on the extracted chart data from ChartMoE ({task_type}):

{raw_output}

Now answer the question: "{question}"

Instructions:
1. Carefully analyze the data provided above
2. Show your reasoning step by step
3. Use exact values from the extracted data
4. For numerical questions, perform calculations as needed
5. For comparison questions, compare the relevant values
6. For yes/no questions, verify against the data

You MUST output your answer inside <answer> tags.
For example:
- For numbers: <answer>42</answer>
- For yes/no: <answer>Yes</answer>
- For text: <answer>The answer is X</answer>"""
    
    def _select_task(self, question: str, question_type: str, observation: Dict[str, Any]) -> str:
        """Select the best ChartMoE task for the question"""
        question_lower = question.lower()
        
        # For reflection attempts, try different tasks
        if observation.get("previous_attempt_failed"):
            # Get previous task from history
            chartmoe_history = observation.get("chartmoe_history", [])
            if chartmoe_history:
                last_task = chartmoe_history[-1].get("task_type", "")
                # Try a different task
                if last_task == "to_table":
                    return "extract_data"
                elif last_task == "extract_data":
                    return "analyze"
                elif last_task == "describe":
                    return "to_table"
            # If custom prompt suggested in feedback
            if "custom prompt" in observation.get("feedback", "").lower():
                return "answer"
        
        # Task selection based on question type
        if question_type == "counting":
            return "to_table"  # Table format best for counting
        
        elif question_type in ["numerical", "summation", "average", "percentage"]:
            return "to_table"  # Structured data for calculations
        
        elif question_type == "comparison":
            return "compare" if "compare" in self.supported_tasks else "to_table"
        
        elif question_type == "trend":
            return "trend" if "trend" in self.supported_tasks else "analyze"
        
        elif question_type in ["minmax", "difference", "ratio"]:
            return "extract_data"  # Extract specific values
        
        elif question_type == "retrieval":
            return "to_text"  # Text format for labels/names
        
        # Keyword-based selection
        elif any(kw in question_lower for kw in ["describe", "what do you see"]):
            return "describe"
        
        elif any(kw in question_lower for kw in ["summarize", "overview"]):
            return "summarize"
        
        elif any(kw in question_lower for kw in ["analyze", "insight"]):
            return "analyze"
        
        elif any(kw in question_lower for kw in ["trend", "pattern", "change"]):
            return "trend"
        
        # Default to table format
        return "to_table"
    
    def _parse_table(self, output: str) -> Dict[str, Any]:
        """Parse table output from ChartMoE"""
        lines = output.strip().split('\n')
        
        if not lines:
            return {}
        
        # Try to identify header row
        header_row = None
        data_rows = []
        
        for i, line in enumerate(lines):
            # Skip separator lines (like |---|---|)
            if all(c in '|-' for c in line.replace(' ', '')):
                continue
                
            if '|' in line:
                # Table row
                if header_row is None:
                    header_row = line
                else:
                    data_rows.append(line)
        
        # Parse into structured format
        table_data = {
            "headers": [],
            "rows": [],
            "values": []
        }
        
        if header_row:
            # Clean and split header
            parts = [p.strip() for p in header_row.split('|')]
            table_data["headers"] = [p for p in parts if p]
        
        for row in data_rows:
            # Clean and split row
            parts = [p.strip() for p in row.split('|')]
            cells = [p for p in parts if p]
            table_data["rows"].append(cells)
            
            # Extract numerical values
            for cell in cells:
                # Try to extract number
                numbers = re.findall(r'-?\d+\.?\d*', cell)
                for num in numbers:
                    try:
                        value = float(num)
                        table_data["values"].append(value)
                    except:
                        pass
        
        return table_data
    
    def _extract_values(self, output: str) -> list:
        """Extract numerical values from output"""
        values = []
        
        # Find all numbers in the output
        numbers = re.findall(r'-?\d+\.?\d*', output)
        for num in numbers:
            try:
                values.append(float(num))
            except:
                pass
        
        return values
    
    def _extract_insights(self, output: str) -> list:
        """Extract key insights from analysis output"""
        insights = []
        
        # Look for bullet points or numbered lists
        patterns = [
            r'[•\-\*]\s*(.+)',
            r'\d+\.\s*(.+)',
            r'(?:First|Second|Third|Finally)[,:]?\s*(.+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            insights.extend(matches)
        
        # If no structured insights, split by sentences
        if not insights:
            sentences = re.split(r'[.!?]+', output)
            insights = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
        
        return insights
    
    def should_retry(self, result: ToolResult, attempt: int) -> bool:
        """Determine if should retry ChartMoE"""
        # Retry if failed and under max attempts
        if not result.success and attempt < self.config.max_retries:
            return True
        
        # Retry if output is too short (might be truncated)
        if result.data and result.data.get("raw_output", ""):
            output_length = len(result.data["raw_output"])
            if output_length < 50 and attempt < 2:  # Very short output
                return True
        
        return False
    
    def get_fallback_strategy(self) -> Optional[str]:
        """Get fallback strategy if ChartMoE fails"""
        # Can try Grounding DINO or EasyOCR as fallback
        return "grounding_dino"