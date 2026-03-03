#!/usr/bin/env python3
"""Task type detection module"""

from typing import Dict, Any, Optional, List, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class TaskDetector:
    """Detector for identifying task types and requirements"""
    
    def __init__(self):
        self.logger = logger
        
        # Task type indicators
        self.task_indicators = {
            'chartqa': {
                'observation_keys': ['is_visual_question', 'chartmoe_enabled'],
                'question_patterns': ['chart', 'graph', 'plot', 'bar', 'line', 'pie'],
                'context_clues': ['axis', 'legend', 'data', 'trend']
            },
            'medical_vqa': {
                'observation_keys': ['is_medical_vqa'],
                'question_patterns': [
                    'brain', 'lung', 'heart', 'liver', 'kidney', 'organ',
                    'ct', 'mri', 'x-ray', 'scan', 'medical', 'diagnosis',
                    'disease', 'abnormal', 'normal', 'lesion', 'tumor',
                    'infarcted', 'cardiovascular', 'pulmonary', 'cerebral'
                ],
                'context_clues': ['patient', 'clinical', 'radiological']
            },
            'geometry': {
                'observation_keys': ['is_geometry_task'],
                'question_patterns': [
                    'angle', 'triangle', 'circle', 'square', 'rectangle',
                    'parallel', 'perpendicular', 'congruent', 'similar',
                    'theorem', 'prove', 'show that', 'verify',
                    'pythagorean', 'inscribed', 'circumference'
                ],
                'context_clues': ['given', 'find', 'calculate', 'proof']
            },
            'text_extraction': {
                'observation_keys': [],
                'question_patterns': [
                    'read', 'text', 'written', 'equation', 'formula',
                    'label', 'sign', 'caption', 'title', 'ocr'
                ],
                'context_clues': ['extract', 'recognize', 'transcribe']
            },
            'object_detection': {
                'observation_keys': [],
                'question_patterns': [
                    'detect', 'locate', 'find', 'identify', 'where',
                    'position', 'bounding box', 'count objects'
                ],
                'context_clues': ['objects', 'items', 'things']
            },
            'general_vqa': {
                'observation_keys': [],
                'question_patterns': [],
                'context_clues': []
            }
        }
    
    def detect(self, observation: Dict[str, Any]) -> str:
        """Detect the task type from observation
        
        Args:
            observation: The observation dictionary
            
        Returns:
            Task type string
        """
        # First check observation keys for explicit task indicators
        for task_type, indicators in self.task_indicators.items():
            for key in indicators['observation_keys']:
                if observation.get(key):
                    self.logger.info(f"Task detected from observation key: {task_type}")
                    return task_type
        
        # Then analyze the question
        question = observation.get("question", "").lower()
        
        # Score each task type based on pattern matching
        task_scores = {}
        
        for task_type, indicators in self.task_indicators.items():
            score = 0
            
            # Check question patterns
            for pattern in indicators['question_patterns']:
                if pattern in question:
                    score += 2
            
            # Check context clues
            for clue in indicators['context_clues']:
                if clue in question:
                    score += 1
            
            if score > 0:
                task_scores[task_type] = score
        
        # Special checks
        task_scores = self._apply_special_rules(observation, task_scores)
        
        # Return highest scoring task type
        if task_scores:
            best_task = max(task_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Task detected: {best_task} (scores: {task_scores})")
            return best_task
        
        # Default to general VQA
        return 'general_vqa'
    
    def _apply_special_rules(self, observation: Dict[str, Any], scores: Dict[str, int]) -> Dict[str, int]:
        """Apply special rules for task detection"""
        question = observation.get("question", "").lower()
        
        # If image path contains hints
        image_path = observation.get("image_path", "").lower()
        
        if 'chart' in image_path or 'graph' in image_path:
            scores['chartqa'] = scores.get('chartqa', 0) + 5
        
        if 'medical' in image_path or 'scan' in image_path:
            scores['medical_vqa'] = scores.get('medical_vqa', 0) + 5
        
        if 'geometry' in image_path or 'math' in image_path:
            scores['geometry'] = scores.get('geometry', 0) + 5
        
        # Check for mathematical symbols
        math_symbols = ['∠', '△', '∼', '≅', '⊥', '∥', '°']
        if any(symbol in observation.get("question", "") for symbol in math_symbols):
            scores['geometry'] = scores.get('geometry', 0) + 3
        
        # Check for explicit tool requirements
        if observation.get("chartmoe_enabled"):
            scores['chartqa'] = scores.get('chartqa', 0) + 10
        
        if observation.get("multimath_enabled"):
            scores['geometry'] = scores.get('geometry', 0) + 10
        
        return scores
    
    def get_required_tools(self, task_type: str) -> List[str]:
        """Get list of recommended tools for task type
        
        Args:
            task_type: The detected task type
            
        Returns:
            List of tool names
        """
        tool_recommendations = {
            'chartqa': ['chartmoe', 'easyocr', 'deepeyes'],
            'medical_vqa': ['sam2', 'grounding_dino', 'deepeyes'],
            'geometry': ['multimath_server', 'diagram_formalizer', 'sympy_geometry'],
            'text_extraction': ['easyocr', 'deepeyes'],
            'object_detection': ['grounding_dino', 'sam2'],
            'general_vqa': []  # Use tool selection strategy
        }
        
        return tool_recommendations.get(task_type, [])
    
    def get_task_priority(self, task_type: str) -> int:
        """Get priority level for task type
        
        Args:
            task_type: The task type
            
        Returns:
            Priority level (higher is more important)
        """
        priorities = {
            'geometry': 10,  # Highest priority - needs specialized tools
            'chartqa': 9,
            'medical_vqa': 8,
            'text_extraction': 5,
            'object_detection': 5,
            'general_vqa': 1
        }
        
        return priorities.get(task_type, 0)
    
    def should_force_tool_use(self, observation: Dict[str, Any], task_type: str) -> Tuple[bool, Optional[str]]:
        """Determine if tool use should be forced for this task
        
        Args:
            observation: The observation
            task_type: The detected task type
            
        Returns:
            Tuple of (should_force, tool_name)
        """
        # Check explicit forcing
        if observation.get("must_use_tool"):
            return True, observation.get("tool_to_use")
        
        if observation.get("force_tool_use"):
            # Determine which tool based on task type
            if task_type == 'chartqa':
                return True, "chartmoe"
            elif task_type == 'geometry':
                if observation.get("multimath_enabled"):
                    return True, "multimath_server"
                else:
                    return True, "sympy_geometry"
            elif task_type == 'medical_vqa':
                return True, "sam2"
        
        # Check implicit forcing based on task requirements
        if task_type == 'chartqa' and observation.get("chartmoe_enabled"):
            # First attempt should use ChartMoE
            if observation.get("attempt", 1) == 1:
                return True, "chartmoe"
        
        if task_type == 'geometry' and observation.get("multimath_enabled"):
            # Geometry with MultiMath enabled should use it
            return True, "multimath_server"
        
        return False, None
    
    def is_complex_task(self, observation: Dict[str, Any], task_type: str) -> bool:
        """Check if task requires complex multi-tool workflow
        
        Args:
            observation: The observation
            task_type: The detected task type
            
        Returns:
            True if complex workflow is needed
        """
        question = observation.get("question", "").lower()
        
        # Geometry with proof requirements
        if task_type == 'geometry':
            if any(word in question for word in ['prove', 'show that', 'verify', 'theorem']):
                return True
        
        # Medical with multiple requirements
        if task_type == 'medical_vqa':
            requirements = ['identify', 'locate', 'measure', 'compare']
            if sum(1 for req in requirements if req in question) >= 2:
                return True
        
        # ChartQA with complex analysis
        if task_type == 'chartqa':
            if any(word in question for word in ['trend', 'correlation', 'pattern', 'analyze']):
                return True
        
        return False