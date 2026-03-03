"""
AI2D Task Implementation for VLMGym

This module implements the AI2D task for science diagram understanding
with multiple choice questions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import re

from .vision_qa_task import VisionQATask

logger = logging.getLogger(__name__)


class AI2DTask(VisionQATask):
    """
    AI2D (AI2 Diagrams) task implementation for VLMGym
    
    This task involves understanding science diagrams and answering
    multiple-choice questions about them.
    """
    
    def __init__(self, task_id: str, adapter: Any, **kwargs):
        """
        Initialize AI2D task
        
        Args:
            task_id: Unique task identifier
            adapter: AI2DAdapter instance
            **kwargs: Additional arguments
        """
        super().__init__(task_id=task_id, adapter=adapter, **kwargs)
        
        # Task-specific attributes
        self.answer_letter = None
        self.answer_index = None
        self.science_domain = None
        self.question_category = None
        
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """
        Set up the AI2D task
        
        Returns:
            Tuple of (goal_description, task_info)
        """
        # Get task data from adapter
        self.task_data = self.adapter.get_task_data(self.task_id)
        
        # Set basic attributes
        self.question = self.task_data.get('question', '')
        self.choices = self.task_data.get('choices', [])
        self.answer = self.task_data.get('answer', '')
        self.image_path = self.task_data.get('image_path', '')
        
        # Get metadata
        metadata = self.task_data.get('metadata', {})
        self.answer_letter = metadata.get('answer_letter', '')
        self.answer_index = metadata.get('answer_index', '')
        
        # Try to convert answer_index to integer
        try:
            self.answer_index = int(self.answer_index)
        except:
            self.answer_index = -1
        
        # Determine science domain and question category
        self.science_domain = self._determine_science_domain()
        self.question_category = self._classify_question_category()
        
        # Build goal description
        goal = self._build_goal_description()
        
        # Build task info
        task_info = {
            'task_type': 'science_diagram_qa',
            'dataset': 'ai2d',
            'question': self.question,
            'choices': self.choices,
            'answer': self.answer,
            'answer_letter': self.answer_letter,
            'answer_index': self.answer_index,
            'image_path': self.image_path,
            'science_domain': self.science_domain,
            'question_category': self.question_category,
            'question_complexity': self._assess_question_complexity(),
            'num_choices': len(self.choices),
            'has_diagram': True,
            'diagram_type': self._determine_diagram_type()
        }
        
        return goal, task_info
    
    def _build_goal_description(self) -> str:
        """Build a natural language goal description"""
        goal = f"Analyze the science diagram and answer the following question:\n\n"
        goal += f"Question: {self.question}\n\n"
        
        if self.choices:
            goal += "Choices:\n"
            for i, choice in enumerate(self.choices):
                letter = chr(65 + i)  # A, B, C, D
                goal += f"{letter}. {choice}\n"
            goal += "\nSelect the correct answer (A, B, C, or D)."
        else:
            goal += "Provide the correct answer."
        
        return goal
    
    def _determine_science_domain(self) -> str:
        """Determine the science domain based on question content"""
        question_lower = self.question.lower()
        
        # Biology keywords
        biology_keywords = [
            'cell', 'organism', 'plant', 'animal', 'ecosystem', 'habitat',
            'species', 'living', 'life', 'biology', 'organ', 'tissue',
            'photosynthesis', 'respiration', 'dna', 'gene', 'evolution'
        ]
        
        # Physics keywords
        physics_keywords = [
            'force', 'motion', 'energy', 'wave', 'light', 'sound',
            'electricity', 'magnetic', 'gravity', 'mass', 'velocity',
            'acceleration', 'circuit', 'voltage', 'current', 'physics'
        ]
        
        # Chemistry keywords
        chemistry_keywords = [
            'chemical', 'element', 'compound', 'molecule', 'atom',
            'reaction', 'bond', 'solution', 'acid', 'base', 'metal',
            'periodic', 'chemistry'
        ]
        
        # Earth science keywords
        earth_keywords = [
            'earth', 'rock', 'mineral', 'weather', 'climate', 'ocean',
            'atmosphere', 'geology', 'volcano', 'earthquake', 'erosion',
            'fossil', 'sediment', 'planet', 'solar', 'moon'
        ]
        
        # Count keyword matches
        domain_scores = {
            'biology': sum(1 for kw in biology_keywords if kw in question_lower),
            'physics': sum(1 for kw in physics_keywords if kw in question_lower),
            'chemistry': sum(1 for kw in chemistry_keywords if kw in question_lower),
            'earth_science': sum(1 for kw in earth_keywords if kw in question_lower)
        }
        
        # Return domain with highest score
        max_domain = max(domain_scores.items(), key=lambda x: x[1])
        return max_domain[0] if max_domain[1] > 0 else 'general_science'
    
    def _classify_question_category(self) -> str:
        """Classify the type of question being asked"""
        question_lower = self.question.lower()
        
        # Identification questions
        if any(phrase in question_lower for phrase in [
            'which', 'what is', 'identify', 'name', 'select'
        ]):
            return 'identification'
        
        # Counting questions
        if any(phrase in question_lower for phrase in [
            'how many', 'count', 'number of', 'total'
        ]):
            return 'counting'
        
        # Comparison questions
        if any(phrase in question_lower for phrase in [
            'compare', 'difference', 'similar', 'same', 'different'
        ]):
            return 'comparison'
        
        # Spatial reasoning questions
        if any(phrase in question_lower for phrase in [
            'where', 'location', 'position', 'above', 'below', 
            'left', 'right', 'between', 'next to'
        ]):
            return 'spatial'
        
        # Process/sequence questions
        if any(phrase in question_lower for phrase in [
            'process', 'sequence', 'order', 'step', 'stage',
            'first', 'then', 'finally', 'cycle'
        ]):
            return 'process'
        
        # Reasoning questions
        if any(phrase in question_lower for phrase in [
            'why', 'explain', 'because', 'reason', 'cause'
        ]):
            return 'reasoning'
        
        # Labeling questions
        if any(phrase in question_lower for phrase in [
            'label', 'part', 'component', 'structure'
        ]):
            return 'labeling'
        
        return 'other'
    
    def _assess_question_complexity(self) -> str:
        """Assess the complexity level of the question"""
        question_words = self.question.split()
        question_length = len(question_words)
        
        # Check for complex reasoning indicators
        complex_indicators = [
            'explain', 'compare', 'analyze', 'evaluate', 'determine',
            'infer', 'predict', 'conclude', 'relationship', 'function'
        ]
        
        has_complex_reasoning = any(
            indicator in self.question.lower() 
            for indicator in complex_indicators
        )
        
        # Determine complexity
        if question_length < 10 and not has_complex_reasoning:
            return 'simple'
        elif question_length < 20 or has_complex_reasoning:
            return 'moderate'
        else:
            return 'complex'
    
    def _determine_diagram_type(self) -> str:
        """Determine the type of diagram based on content"""
        question_lower = self.question.lower()
        
        # Check for specific diagram types
        if any(word in question_lower for word in ['cycle', 'circular']):
            return 'cycle_diagram'
        elif any(word in question_lower for word in ['flow', 'process', 'sequence']):
            return 'flow_diagram'
        elif any(word in question_lower for word in ['label', 'anatomy', 'structure']):
            return 'labeled_diagram'
        elif any(word in question_lower for word in ['graph', 'chart', 'plot']):
            return 'graph'
        elif any(word in question_lower for word in ['map', 'location']):
            return 'map'
        elif any(word in question_lower for word in ['cross-section', 'cutaway']):
            return 'cross_section'
        else:
            return 'general_diagram'
    
    def check_success(self, model_output: str) -> Tuple[bool, str]:
        """
        Check if the model output contains the correct answer
        
        Args:
            model_output: The model's response
            
        Returns:
            Tuple of (success, feedback_message)
        """
        if not model_output:
            return False, "No answer provided"
        
        model_output = model_output.strip()
        
        # Special handling for single character inputs
        if len(model_output) == 1:
            # First check if this matches the answer content directly
            if self.answer and model_output.upper() == self.answer.upper():
                return True, f"Correct! The answer is {self.answer}"
            
            # Then check if it's a letter selection (A, B, C, D)
            if model_output.upper() in 'ABCD':
                if model_output.upper() == self.answer_letter:
                    return True, f"Correct! The answer is {self.answer_letter}: {self.answer}"
                else:
                    selected_index = ord(model_output.upper()) - ord('A')
                    if 0 <= selected_index < len(self.choices):
                        selected_choice = self.choices[selected_index]
                        return False, f"Incorrect. You selected {model_output.upper()}: {selected_choice}, but the correct answer is {self.answer_letter}: {self.answer}"
        
        # Check for content answer first (with word boundary for short answers)
        if self.answer:
            # For very short answers (1-2 chars), check with word boundary
            if len(self.answer) <= 2:
                # Direct match at start/end of string
                if model_output.upper() == self.answer.upper():
                    return True, f"Correct! The answer is {self.answer}"
                # Word boundary check
                pattern = r'\b' + re.escape(self.answer) + r'\b'
                if re.search(pattern, model_output, re.IGNORECASE):
                    return True, f"Correct! The answer is {self.answer}"
            else:
                # For longer answers, use substring matching
                if self.answer.lower() in model_output.lower():
                    return True, f"Correct! The answer is {self.answer}"
        
        # Check for letter answer in sentence (e.g., "The answer is B")
        letter_match = re.search(r'\b([A-D])\b', model_output.upper())
        if letter_match:
            selected_letter = letter_match.group(1)
            if selected_letter == self.answer_letter:
                return True, f"Correct! The answer is {self.answer_letter}: {self.answer}"
            else:
                selected_index = ord(selected_letter) - ord('A')
                if 0 <= selected_index < len(self.choices):
                    selected_choice = self.choices[selected_index]
                    return False, f"Incorrect. You selected {selected_letter}: {selected_choice}, but the correct answer is {self.answer_letter}: {self.answer}"
        
        # Check if any choice content is mentioned
        for i, choice in enumerate(self.choices):
            # Skip single-letter choices to avoid confusion
            if len(choice) > 1 and choice.lower() in model_output.lower():
                choice_letter = chr(65 + i)
                if choice_letter == self.answer_letter:
                    return True, f"Correct! The answer is {self.answer_letter}: {self.answer}"
                else:
                    return False, f"Incorrect. You selected {choice_letter}: {choice}, but the correct answer is {self.answer_letter}: {self.answer}"
        
        # For inputs that don't match any pattern
        if re.match(r'^[A-D]+$', model_output.upper()) and len(model_output) > 1:
            # Multiple letters like "ABCD"
            return False, f"Could not determine your answer. The correct answer is {self.answer_letter}: {self.answer}"
        
        return False, f"Could not determine your answer. The correct answer is {self.answer_letter}: {self.answer}"
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation for the VLM
        
        Returns:
            Dictionary containing observation data
        """
        observation = {
            'type': 'vqa',
            'task': 'science_diagram_qa',
            'vqa_info': {
                'image_path': self.image_path,
                'question': self.question,
                'choices': self.choices,
                'task_instruction': self._build_goal_description()
            }
        }
        
        return observation
    
    def validate(self, trajectory: List[Any], response: str) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Validate the model's response
        
        Args:
            trajectory: List of previous actions (unused for VQA)
            response: Model's response
            
        Returns:
            Tuple of (reward, done, feedback, validation_info)
        """
        success, feedback = self.check_success(response)
        
        # Reward: 1.0 for correct, 0.0 for incorrect
        reward = 1.0 if success else 0.0
        
        # VQA tasks are single-turn, so always done
        done = True
        
        # Validation info
        validation_info = {
            'success': success,
            'model_answer': response,
            'correct_answer': self.answer,
            'correct_letter': self.answer_letter,
            'question_type': self.question_category,
            'science_domain': self.science_domain
        }
        
        return reward, done, feedback, validation_info
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get task-specific metrics
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'task_id': self.task_id,
            'question_length': len(self.question.split()),
            'num_choices': len(self.choices),
            'question_category': self.question_category,
            'science_domain': self.science_domain,
            'question_complexity': self._assess_question_complexity(),
            'diagram_type': self._determine_diagram_type(),
            'has_spatial_reasoning': self.question_category == 'spatial',
            'requires_counting': self.question_category == 'counting',
            'requires_comparison': self.question_category == 'comparison',
            'is_process_question': self.question_category == 'process'
        }
        
        return metrics