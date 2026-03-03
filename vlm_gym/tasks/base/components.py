#!/usr/bin/env python3
"""Component interfaces for task wrappers"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import re


class Evaluator(ABC):
    """Base evaluator interface"""
    
    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """Evaluate if prediction matches ground truth"""
        pass


class AnswerExtractor(ABC):
    """Base answer extractor interface"""
    
    @abstractmethod
    def extract(self, action: str, extra_info: Optional[Dict[str, Any]] = None) -> str:
        """Extract answer from action"""
        pass


class QuestionClassifier(ABC):
    """Base question classifier interface"""
    
    @abstractmethod
    def classify(self, question: str) -> str:
        """Classify question type"""
        pass


class FeedbackGenerator(ABC):
    """Base feedback generator interface"""
    
    @abstractmethod
    def generate(self, prediction: str, ground_truth: str, 
                question: str, question_type: str,
                attempt: int, max_attempts: int) -> str:
        """Generate feedback for incorrect answer"""
        pass


class OutputFormatter(ABC):
    """Base output formatter interface"""
    
    @abstractmethod
    def format_instruction(self, question: str, tools_enabled: Dict[str, bool]) -> str:
        """Format output instruction"""
        pass


class HintGenerator(ABC):
    """Base hint generator interface"""
    
    @abstractmethod
    def generate(self, question: str, ground_truth: str, 
                attempt_history: list, tools_enabled: Dict[str, bool]) -> str:
        """Generate hint for final attempt"""
        pass


# Make sure all classes are exported
__all__ = [
    'Evaluator',
    'AnswerExtractor',
    'QuestionClassifier',
    'FeedbackGenerator',
    'OutputFormatter',
    'HintGenerator'
]