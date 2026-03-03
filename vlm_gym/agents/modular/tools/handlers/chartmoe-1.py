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
        """Process ChartMoE output"""
        if isinstance(raw_result, dict):
            task_type = raw_result.get("task_type", "unknown")
            output = raw_result.get("output", "")
            success = raw_result.get("success", False)
            
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
                    "output_length": len(output)
                }
            )
        
        return ToolResult(
            success=False,
            data=None,
            error="Invalid ChartMoE result format"
        )
    
    def format_for_answer(self, result: ToolResult, observation: Dict[str, Any]) -> str:
        """Format ChartMoE result for final answer generation"""
        if not result.success:
            return f"ChartMoE failed: {result.error}"
        
        data = result.data
        task_type = data.get("task_type", "unknown")
        
        # Build formatted prompt based on task type
        if task_type == "to_table" and data.get("table_data"):
            return self._format_table_result(data, observation)
        
        elif task_type == "extract_data" and data.get("extracted_values"):
            return self._format_extraction_result(data, observation)
        
        elif task_type == "analyze":
            return self._format_analysis_result(data, observation)
        
        else:
            # Generic format
            output = data.get("raw_output", "")
            return f"""ChartMoE {task_type} result:

{output}

Based on this analysis, answer: "{observation.get('question', '')}"

Provide your answer inside <answer> tags."""
    
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
            if '|' in line or '\t' in line:
                # Table row
                if header_row is None:
                    header_row = line
                else:
                    data_rows.append(line)
            elif i == 0 and not any(char.isdigit() for char in line):
                # First line might be header without separators
                header_row = line
        
        # Parse into structured format
        table_data = {
            "headers": [],
            "rows": [],
            "values": []
        }
        
        if header_row:
            # Clean and split header
            header_row = header_row.replace('|', '\t')
            table_data["headers"] = [h.strip() for h in header_row.split('\t') if h.strip()]
        
        for row in data_rows:
            # Clean and split row
            row = row.replace('|', '\t')
            cells = [c.strip() for c in row.split('\t') if c.strip()]
            table_data["rows"].append(cells)
            
            # Extract numerical values
            for cell in cells:
                # Try to extract number
                numbers = re.findall(r'-?\d+\.?\d*', cell)
                for num in numbers:
                    try:
                        table_data["values"].append(float(num))
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
    
    def _format_table_result(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Format table result for answer generation"""
        table_data = data.get("table_data", {})
        raw_output = data.get("raw_output", "")
        question = observation.get("question", "")
        
        prompt = f"""ChartMoE extracted the following table data:

{raw_output}

"""
        
        if table_data.get("values"):
            values = table_data["values"]
            prompt += f"Numerical values found: {values}\n"
            prompt += f"Count of values: {len(values)}\n"
            if values:
                prompt += f"Sum: {sum(values)}\n"
                prompt += f"Average: {sum(values)/len(values):.2f}\n"
                prompt += f"Min: {min(values)}, Max: {max(values)}\n"
        
        prompt += f"""
Now answer the question: "{question}"

Instructions:
1. Use the exact values from the table above
2. For counting questions, count the relevant entries
3. For calculations, use the provided numbers

You MUST output your answer inside <answer> tags."""
        
        return prompt
    
    def _format_extraction_result(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Format extraction result for answer generation"""
        values = data.get("extracted_values", [])
        raw_output = data.get("raw_output", "")
        question = observation.get("question", "")
        
        prompt = f"""ChartMoE extracted the following data:

{raw_output}

Extracted values: {values}
"""
        
        if values:
            prompt += f"""
Statistics:
- Count: {len(values)}
- Sum: {sum(values)}
- Average: {sum(values)/len(values):.2f}
- Min: {min(values)}, Max: {max(values)}
"""
        
        prompt += f"""
Based on this data, answer: "{question}"

You MUST output your answer inside <answer> tags."""
        
        return prompt
    
    def _format_analysis_result(self, data: Dict[str, Any], observation: Dict[str, Any]) -> str:
        """Format analysis result for answer generation"""
        analysis = data.get("analysis", "")
        insights = data.get("insights", [])
        question = observation.get("question", "")
        
        prompt = f"""ChartMoE analysis:

{analysis}
"""
        
        if insights:
            prompt += "\nKey insights:\n"
            for i, insight in enumerate(insights, 1):
                prompt += f"{i}. {insight}\n"
        
        prompt += f"""
Based on this analysis, answer: "{question}"

You MUST output your answer inside <answer> tags."""
        
        return prompt
    
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