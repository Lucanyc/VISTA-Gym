"""Structured reasoning templates for ChartQA"""
from typing import Dict

class ChartQATemplates:
    """Templates for different question types"""
    
    @staticmethod
    def comparison_template() -> Dict[str, str]:
        return {
            "steps": [
                "Identify the elements to compare from the chart",
                "Extract the exact numerical values",
                "Calculate the difference",
                "Return the final answer"
            ],
            "format": """
Step 1: Elements to compare
- Element A: {element_a}
- Element B: {element_b}
Step 2: Values
- Value A: {value_a}
- Value B: {value_b}
Step 3: Calculation
Difference = |{value_a} - {value_b}| = {result}
Answer: {result}
"""
        }
    
    @staticmethod
    def minmax_template() -> Dict[str, str]:
        return {
            "steps": [
                "Extract all values from the chart",
                "Compare all values",
                "Identify the minimum/maximum",
                "Return the answer with label"
            ],
            "format": """
Step 1: Extracted values
{values_list}
Step 2: Comparison
- Minimum: {min_label} = {min_value}
- Maximum: {max_label} = {max_value}
Answer: {answer}
"""
        }
    
    @staticmethod
    def numerical_template() -> Dict[str, str]:
        return {
            "steps": [
                "Identify what needs to be counted/calculated",
                "Extract relevant values",
                "Perform calculation if needed",
                "Return the final number"
            ],
            "format": """
Step 1: Target
{target}
Step 2: Values
{values}
Step 3: Calculation
{calculation}
Answer: {answer}
"""
        }
    
    @staticmethod
    def counting_template() -> Dict[str, str]:
        return {
            "steps": [
                "Identify what needs to be counted",
                "List all items that meet the criteria",
                "Count the items",
                "Return the count"
            ],
            "format": """
Step 1: Counting target
{target}
Step 2: Items found
{items}
Step 3: Count
Total count: {count}
Answer: {count}
"""
        }
    
    @staticmethod
    def trend_template() -> Dict[str, str]:
        return {
            "steps": [
                "Identify the time period",
                "Extract values at different time points",
                "Analyze the direction of change",
                "Describe the trend"
            ],
            "format": """
Step 1: Time period
{time_period}
Step 2: Values over time
{values}
Step 3: Analysis
{analysis}
Answer: {answer}
"""
        }
    
    @staticmethod
    def retrieval_template() -> Dict[str, str]:
        return {
            "steps": [
                "Identify what information is requested",
                "Locate the information in the chart",
                "Extract the specific value or label",
                "Return the answer"
            ],
            "format": """
Step 1: Information requested
{request}
Step 2: Location in chart
{location}
Step 3: Extracted information
{info}
Answer: {answer}
"""
        }