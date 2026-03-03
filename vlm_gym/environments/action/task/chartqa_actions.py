
# vlm_gym/environments/action/task/chartqa_actions.py
"""ChartQA specific actions"""
from typing import Dict, Any, List, Optional
import re

def extract_chart_values(image_path: str, chart_element: str = "bars") -> Dict[str, Any]:
    """
    Extract numerical values from chart elements
    
    Examples:
        extract_chart_values("/path/to/chart.png", "bars")
        extract_chart_values("/path/to/chart.png", "line_points")
        extract_chart_values("/path/to/chart.png", "pie_segments")
    """
    # TODO: 实现实际的图表数值提取
    # 这里返回模拟数据用于测试
    mock_data = {
        "bars": {
            "element_type": "bars",
            "values": [100, 150, 120, 180],
            "labels": ["Q1", "Q2", "Q3", "Q4"]
        },
        "line_points": {
            "element_type": "line_points",
            "values": [50, 75, 100, 125, 150],
            "labels": ["Jan", "Feb", "Mar", "Apr", "May"]
        },
        "pie_segments": {
            "element_type": "pie_segments",
            "values": [30, 25, 20, 15, 10],
            "labels": ["A", "B", "C", "D", "Other"]
        }
    }
    
    return mock_data.get(chart_element, {
        "element_type": chart_element,
        "values": [],
        "labels": [],
        "error": "Unknown chart element type"
    })

def calculate_chart_statistics(values: List[float], stat_type: str = "mean") -> float:
    """
    Calculate statistics from chart values
    
    Examples:
        calculate_chart_statistics([100, 150, 120], "mean")
        calculate_chart_statistics([100, 150, 120], "max")
        calculate_chart_statistics([100, 150, 120], "min")
        calculate_chart_statistics([100, 150, 120], "sum")
    """
    if not values:
        return 0.0
    
    if stat_type == "mean":
        return sum(values) / len(values)
    elif stat_type == "max":
        return max(values)
    elif stat_type == "min":
        return min(values)
    elif stat_type == "sum":
        return sum(values)
    elif stat_type == "median":
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")

def compare_chart_elements(value1: float, value2: float, comparison: str = "difference") -> Dict[str, Any]:
    """
    Compare two chart elements
    
    Examples:
        compare_chart_elements(100, 150, "difference")
        compare_chart_elements(100, 150, "ratio")
        compare_chart_elements(100, 150, "percentage_change")
    """
    result = {
        "value1": value1,
        "value2": value2,
        "comparison_type": comparison
    }
    
    if comparison == "difference":
        result["result"] = value2 - value1
        result["description"] = f"The difference is {value2 - value1}"
    elif comparison == "ratio":
        if value1 != 0:
            ratio = value2 / value1
            result["result"] = ratio
            result["description"] = f"The ratio is {ratio:.2f}"
        else:
            result["result"] = float('inf')
            result["description"] = "Cannot divide by zero"
    elif comparison == "percentage_change":
        if value1 != 0:
            pct_change = ((value2 - value1) / value1) * 100
            result["result"] = pct_change
            result["description"] = f"The percentage change is {pct_change:.1f}%"
        else:
            result["result"] = float('inf')
            result["description"] = "Cannot calculate percentage change from zero"
    else:
        result["error"] = f"Unknown comparison type: {comparison}"
    
    return result

def find_chart_extremes(values: List[float], labels: List[str]) -> Dict[str, Any]:
    """
    Find maximum and minimum values in chart data
    
    Examples:
        find_chart_extremes([100, 150, 120, 80], ["Q1", "Q2", "Q3", "Q4"])
    """
    if not values or not labels:
        return {"error": "Empty values or labels"}
    
    if len(values) != len(labels):
        return {"error": "Values and labels must have same length"}
    
    max_idx = values.index(max(values))
    min_idx = values.index(min(values))
    
    return {
        "maximum": {
            "value": values[max_idx],
            "label": labels[max_idx],
            "index": max_idx
        },
        "minimum": {
            "value": values[min_idx],
            "label": labels[min_idx],
            "index": min_idx
        },
        "range": max(values) - min(values)
    }

def analyze_chart_trend(values: List[float]) -> Dict[str, Any]:
    """
    Analyze trend in sequential chart data
    
    Examples:
        analyze_chart_trend([100, 120, 140, 160])
        analyze_chart_trend([100, 80, 60, 40])
    """
    if len(values) < 2:
        return {"error": "Need at least 2 values to analyze trend"}
    
    # Calculate differences
    differences = [values[i+1] - values[i] for i in range(len(values)-1)]
    
    # Determine overall trend
    if all(d > 0 for d in differences):
        overall_trend = "strictly_increasing"
    elif all(d < 0 for d in differences):
        overall_trend = "strictly_decreasing"
    elif all(d >= 0 for d in differences):
        overall_trend = "increasing"
    elif all(d <= 0 for d in differences):
        overall_trend = "decreasing"
    else:
        overall_trend = "fluctuating"
    
    # Calculate average change
    avg_change = sum(differences) / len(differences)
    
    # Calculate total change
    total_change = values[-1] - values[0]
    
    return {
        "overall_trend": overall_trend,
        "average_change": avg_change,
        "total_change": total_change,
        "percentage_change": (total_change / values[0] * 100) if values[0] != 0 else None,
        "differences": differences
    }