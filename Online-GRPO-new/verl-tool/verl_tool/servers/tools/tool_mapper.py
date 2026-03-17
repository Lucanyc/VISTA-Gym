import re
import json
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Dataset type enumeration"""
    MATHVISTA = "mathvista"
    GEOMETRY3K = "geometry3k"
    CHARTQA = "chartqa"
    GEOQA = "geoqa"
    MATHVERSE = "mathverse"
    GENERAL = "general"

class QuestionType(Enum):
    """Question type enumeration"""
    GEOMETRY = "geometry"
    CHART_DATA = "chart_data"
    CHART_ANALYSIS = "chart_analysis"
    CALCULATION = "calculation"
    OCR_NEEDED = "ocr_needed"
    OBJECT_DETECTION = "object_detection"
    MIXED = "mixed"

class ToolMapper:
    """Tool Mapper - Converts simple calls to full parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Keyword mappings
        self.keywords = {
            'geometry': ['triangle', 'angle', 'circle', 'square', 'rectangle', 'polygon', 
                        'perimeter', 'area', 'degree', 'hypotenuse', 'radius'],
            
            'chart': ['chart', 'graph', 'plot', 'bar', 'line', 'pie', 'histogram',
                     'data', 'trend'],
            
            'calculation': ['calculate', 'solve', 'compute', 'equation', 'evaluate',
                          'x =', 'y =', '+', '-', '*', '/', 
                          'derivative', 'integral', 'limit'],
            
            'ocr': ['text', 'read', 'written', 'words', 'letters', 'numbers',
                   'inscription', 'label'],
            
            'detection': ['detect', 'find', 'locate', 'identify', 'object', 'count'],
            
            'table': ['table', 'convert', 'tabular', 'spreadsheet'],
            
            'analysis': ['analyze', 'trend', 'pattern', 'compare', 'relationship',
                        'correlation'],
            
            'formalize': ['formalize', 'cdl', 'construction', 'diagram description']
        }
        
        # Dataset to question type mapping
        self.dataset_defaults = {
            DatasetType.GEOMETRY3K: QuestionType.GEOMETRY,
            DatasetType.CHARTQA: QuestionType.CHART_DATA,
            DatasetType.GEOQA: QuestionType.GEOMETRY,
            DatasetType.MATHVISTA: QuestionType.MIXED,
            DatasetType.MATHVERSE: QuestionType.MIXED,
        }
        
        # All available ChartMoE tasks
        self.chartmoe_all_tasks = ["to_table", "describe", "extract_data", "summarize", "analyze"]
    
    def map_tool_call(
        self, 
        tool_name: str,
        question: str = "",
        dataset_type: str = None,
        task_type: str = None,
        choices: List[str] = None,
        extra_info: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Map a simple tool call to full parameters
        
        Args:
            tool_name: Tool name
            question: Question text
            dataset_type: Dataset type
            task_type: Task type
            choices: Multiple choice options (already structured)
            extra_info: Additional information
            
        Returns:
            Full tool call parameters, or a list of multiple tasks (when exact match is not possible)
        """
        self.logger.debug(f"Mapping tool call: {tool_name}, question: {question[:50] if question else 'None'}...")
        
        # Infer question type
        q_type = self._infer_question_type(question, dataset_type, task_type)
        
        # Map based on tool name
        if tool_name == "chartmoe":
            return self._map_chartmoe(question, q_type)
        elif tool_name == "gllava":
            return self._map_gllava(question, choices)
        elif tool_name == "multimath":
            return self._map_multimath(question, choices)
        elif tool_name == "diagramformalizer":
            return self._map_diagramformalizer(question)
        elif tool_name == "groundingdino":
            return self._map_groundingdino(question)
        elif tool_name == "easyocr":
            return self._map_easyocr(question)
        else:
            # Default: return original call
            return {"tool": tool_name}
    
    def _infer_question_type(self, question: str, dataset_type: str = None, task_type: str = None) -> QuestionType:
        """Infer the question type"""
        if not question:
            return QuestionType.MIXED
            
        question_lower = question.lower()
        
        # Prioritize dataset type
        if dataset_type:
            try:
                ds_type = DatasetType(dataset_type.lower())
                if ds_type in self.dataset_defaults:
                    return self.dataset_defaults[ds_type]
            except:
                pass
        
        # Infer based on keywords
        scores = {
            QuestionType.GEOMETRY: self._count_keywords(question_lower, self.keywords['geometry']),
            QuestionType.CHART_DATA: self._count_keywords(question_lower, self.keywords['chart']),
            QuestionType.CALCULATION: self._count_keywords(question_lower, self.keywords['calculation']),
            QuestionType.OCR_NEEDED: self._count_keywords(question_lower, self.keywords['ocr']),
            QuestionType.OBJECT_DETECTION: self._count_keywords(question_lower, self.keywords['detection']),
        }
        
        # Special case: chart analysis
        if scores[QuestionType.CHART_DATA] > 0 and self._count_keywords(question_lower, self.keywords['analysis']) > 0:
            return QuestionType.CHART_ANALYSIS
        
        # Return the type with the highest score
        max_type = max(scores, key=scores.get)
        if scores[max_type] > 0:
            return max_type
        
        return QuestionType.MIXED
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count keyword occurrences"""
        count = 0
        for keyword in keywords:
            if keyword.lower() in text:
                count += 1
        return count
    
    def _map_chartmoe(self, question: str, q_type: QuestionType) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Map ChartMoE tool call"""
        if not question:
            # No question text, execute all tasks
            return self._create_chartmoe_all_tasks("")
            
        question_lower = question.lower()
        
        # Try exact match
        if self._count_keywords(question_lower, self.keywords['table']) > 0:
            self.logger.debug("ChartMoE: Matched 'table' task")
            return {"tool": "chartmoe", "task": "to_table"}
        
        if any(word in question_lower for word in ['highest', 'lowest', 'maximum', 'minimum', 
                                                    'max', 'min']):
            self.logger.debug("ChartMoE: Matched 'extract_data' task")
            return {"tool": "chartmoe", "task": "extract_data"}
        
        if self._count_keywords(question_lower, self.keywords['analysis']) > 0:
            self.logger.debug("ChartMoE: Matched 'analyze' task")
            return {"tool": "chartmoe", "task": "analyze"}
        
        if any(word in question_lower for word in ['describe', 'explain', 'what']):
            self.logger.debug("ChartMoE: Matched 'describe' task")
            return {"tool": "chartmoe", "task": "describe"}
        
        if any(word in question_lower for word in ['summary', 'summarize']):
            self.logger.debug("ChartMoE: Matched 'summarize' task")
            return {"tool": "chartmoe", "task": "summarize"}
        
        # No exact match, execute all tasks
        self.logger.info(f"ChartMoE: No exact match for question '{question[:50]}...', executing all tasks")
        return self._create_chartmoe_all_tasks(question)
    
    def _create_chartmoe_all_tasks(self, question: str) -> List[Dict[str, Any]]:
        """Create a list of all ChartMoE task calls"""
        tasks = []
        
        # Base tasks
        for task in self.chartmoe_all_tasks:
            task_dict = {"tool": "chartmoe", "task": task}
            tasks.append(task_dict)
        
        # If there is a question, add an answer task
        if question and "answer" not in self.chartmoe_all_tasks:
            tasks.append({
                "tool": "chartmoe", 
                "task": "answer",
                "question": question
            })
        
        self.logger.debug(f"Created {len(tasks)} ChartMoE tasks")
        return tasks
    
    def _map_gllava(self, question: str, choices: List[str] = None) -> Dict[str, Any]:
        """
        Map G-LLaVA tool call
        Directly pass question and choices, let the tool determine the question type
        """
        result = {
            "tool": "gllava",
            "task": "solve",
            "question": question if question else ""
        }
        
        # Directly pass choices (if available)
        if choices:
            result["choices"] = choices
            self.logger.debug(f"G-LLaVA: Multiple choice question with {len(choices)} options")
        else:
            self.logger.debug("G-LLaVA: Open-ended question")
        
        # Output format (optional optimization)
        if question and len(question) > 300:
            result["output_format"] = "brief"
        else:
            result["output_format"] = "detailed"
        
        return result
    
    def _map_multimath(self, question: str, choices: List[str] = None) -> Dict[str, Any]:
        """
        Map MultiMath tool call
        Directly pass question and choices, let the tool determine the question type
        """
        result = {
            "tool": "multimath",
            "task": "solve",
            "question": question if question else ""
        }
        
        # Directly pass choices (if available)
        if choices:
            result["choices"] = choices
            self.logger.debug(f"MultiMath: Multiple choice question with {len(choices)} options")
        else:
            self.logger.debug("MultiMath: Open-ended calculation")
        
        # Optional: optimize output format based on keywords
        if question:
            question_lower = question.lower()
            if any(word in question_lower for word in ['step', 'show work']):
                result["output_format"] = "step_by_step"
            elif any(word in question_lower for word in ['answer only', 'just answer']):
                result["output_format"] = "answer_only"
            else:
                result["output_format"] = "detailed"
        
        return result
    
    def _map_diagramformalizer(self, question: str) -> Dict[str, Any]:
        """Map DiagramFormalizer tool call"""
        result = {
            "tool": "diagramformalizer",
            "task": "formalize"
        }
        
        # Add specific prompt based on the question
        if question:
            if "construction" in question.lower():
                result["prompt"] = "Focus on the construction_cdl"
            elif "image" in question.lower():
                result["prompt"] = "Focus on the image_cdl"
        
        return result
    
    def _map_groundingdino(self, question: str) -> Dict[str, Any]:
        """Map GroundingDINO tool call"""
        query = "objects"  # Default: detect all
        
        if question:
            # Try to extract specific target from the question
            patterns = [
                r"detect\s+(\w+)",
                r"find\s+(\w+)",
                r"locate\s+(\w+)",
                r"count\s+(\w+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, question.lower())
                if match:
                    query = match.group(1)
                    break
        
        return {
            "tool": "groundingdino",
            "task": "detect",
            "query": query
        }
    
    def _map_easyocr(self, question: str) -> Dict[str, Any]:
        """Map EasyOCR tool call"""
        result = {
            "tool": "easyocr",
            "parameters": {
                "detect_text": True
            }
        }
        
        # Detect language requirements
        if question:
            question_lower = question.lower()
            if any(word in question_lower for word in ['chinese']):
                result["parameters"]["lang"] = ["ch_sim", "en"]
            elif any(word in question_lower for word in ['english']):
                result["parameters"]["lang"] = ["en"]
        
        return result
    
    def auto_select_tool(
        self,
        question: str,
        image_path: str = None,
        dataset_type: str = None,
        task_type: str = None
    ) -> List[str]:
        """
        Automatically select appropriate tools based on the question
        
        Returns:
            Recommended tool list (sorted by priority)
        """
        q_type = self._infer_question_type(question, dataset_type, task_type)
        tools = []
        
        # Recommend tools based on question type
        if q_type == QuestionType.GEOMETRY:
            tools = ["gllava", "diagramformalizer"]
        elif q_type in [QuestionType.CHART_DATA, QuestionType.CHART_ANALYSIS]:
            tools = ["chartmoe"]
        elif q_type == QuestionType.CALCULATION:
            tools = ["multimath"]
        elif q_type == QuestionType.OCR_NEEDED:
            tools = ["easyocr"]
        elif q_type == QuestionType.OBJECT_DETECTION:
            tools = ["groundingdino"]
        else:
            # Mixed type, determine based on keywords
            if question:
                question_lower = question.lower()
                if self._count_keywords(question_lower, self.keywords['chart']) > 0:
                    tools.append("chartmoe")
                if self._count_keywords(question_lower, self.keywords['geometry']) > 0:
                    tools.append("gllava")
                if self._count_keywords(question_lower, self.keywords['calculation']) > 0:
                    tools.append("multimath")
        
        # If no tools matched, use default strategy
        if not tools:
            if image_path:
                tools = ["chartmoe", "gllava"]
            else:
                tools = ["multimath"]
        
        self.logger.info(f"Auto-selected tools for question type {q_type}: {tools}")
        return tools


# Usage example
def example_usage():
    mapper = ToolMapper()
    
    print("Tool Mapper Examples\n" + "="*50)
    
    # Test cases
    test_cases = [
        {
            "description": "ChartMoE with unclear question",
            "input": {"tool": "chartmoe"},
            "question": "What can you tell me about this?",
        },
        {
            "description": "ChartMoE with specific task",
            "input": {"tool": "chartmoe"},
            "question": "Convert this chart to table format",
        },
        {
            "description": "G-LLaVA with choices",
            "input": {"tool": "gllava"},
            "question": "Find the area of the triangle",
            "choices": ["A: 10", "B: 20", "C: 30", "D: 40"]
        },
        {
            "description": "MultiMath without choices",
            "input": {"tool": "multimath"},
            "question": "Solve x^2 + 5x + 6 = 0"
        },
        {
            "description": "MultiMath with choices",
            "input": {"tool": "multimath"},
            "question": "What is 15% of 200?",
            "choices": ["A: 25", "B: 30", "C: 35", "D: 40"]
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['description']}")
        print(f"Input: {test['input']}")
        print(f"Question: {test.get('question', 'N/A')}")
        if test.get('choices'):
            print(f"Choices: {test['choices']}")
        
        result = mapper.map_tool_call(
            tool_name=test["input"]["tool"],
            question=test.get("question", ""),
            choices=test.get("choices")
        )
        
        if isinstance(result, list):
            print(f"Output: Multiple tasks ({len(result)} tasks)")
            for i, task in enumerate(result, 1):
                print(f"  Task {i}: {task}")
        else:
            print(f"Output: {json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 50)


if __name__ == "__main__":
    example_usage()