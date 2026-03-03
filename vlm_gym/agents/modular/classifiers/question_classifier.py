#!/usr/bin/env python3
"""Question type classification module"""

from typing import Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class QuestionClassifier:
    """Classifier for determining question types"""
    
    def __init__(self):
        self.logger = logger
        
        # Define patterns for each question type
        self.patterns = {
            'counting': {
                'keywords': ['how many', 'count', 'total number', 'number of'],
                'patterns': [r'\bhow many\b', r'\bcount\b', r'\btotal number\b']
            },
            'summation': {
                'keywords': ['sum', 'add', 'total of', 'combined'],
                'patterns': [r'\bsum\b', r'\badd\b', r'\btotal of\b']
            },
            'average': {
                'keywords': ['average', 'mean', 'avg'],
                'patterns': [r'\baverage\b', r'\bmean\b', r'\bavg\b']
            },
            'percentage': {
                'keywords': ['percentage', 'percent', '%', 'proportion'],
                'patterns': [r'\bpercentage\b', r'\bpercent\b', r'%', r'\bproportion\b']
            },
            'difference': {
                'keywords': ['difference', 'gap', 'subtract', 'minus', 'between'],
                'patterns': [r'\bdifference\b', r'\bgap\b', r'\bsubtract\b', r'\bminus\b']
            },
            'ratio': {
                'keywords': ['ratio', 'times', 'divide', 'fraction'],
                'patterns': [r'\bratio\b', r'\btimes\b', r'\bdivide\b', r'\bfraction\b']
            },
            'numerical': {
                'keywords': ['value', 'what is the', 'how much'],
                'patterns': [r'\bvalue\b', r'\bwhat is the\b', r'\bhow much\b']
            },
            'comparison': {
                'keywords': ['compare', 'which is', 'greater', 'less', 'more', 'fewer'],
                'patterns': [r'\bcompare\b', r'\bwhich is\b', r'\bgreater\b', r'\bless\b']
            },
            'minmax': {
                'keywords': ['maximum', 'minimum', 'highest', 'lowest', 'largest', 'smallest'],
                'patterns': [r'\bmaximum\b', r'\bminimum\b', r'\bhighest\b', r'\blowest\b']
            },
            'trend': {
                'keywords': ['trend', 'increase', 'decrease', 'change', 'growth', 'decline'],
                'patterns': [r'\btrend\b', r'\bincrease\b', r'\bdecrease\b', r'\bchange\b']
            },
            'yes_no': {
                'keywords': ['yes', 'no', 'is', 'are', 'does', 'do', 'can', 'will'],
                'patterns': [r'\bis\s+(?:it|there|this|that)\b', r'\bdoes?\b', r'\bcan\b', r'\bwill\b']
            },
            'retrieval': {
                'keywords': ['what', 'which', 'when', 'where', 'who', 'name'],
                'patterns': [r'^what\b', r'^which\b', r'^when\b', r'^where\b', r'^who\b']
            },
            'geometry': {
                'keywords': ['angle', 'triangle', 'circle', 'area', 'perimeter', 'prove', 'theorem'],
                'patterns': [r'\bangle\b', r'\btriangle\b', r'\bcircle\b', r'\barea\b', r'\btheorem\b']
            },
            'calculation': {
                'keywords': ['calculate', 'compute', 'solve', 'find'],
                'patterns': [r'\bcalculate\b', r'\bcompute\b', r'\bsolve\b', r'\bfind\b']
            }
        }
    
    def classify(self, question: str) -> str:
        """Classify the question type
        
        Args:
            question: The question text
            
        Returns:
            Question type string
        """
        if not question:
            return 'other'
        
        question_lower = question.lower()
        
        # Check each pattern in order of priority
        type_scores = {}
        
        for qtype, config in self.patterns.items():
            score = 0
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in question_lower:
                    score += 1
            
            # Check regex patterns
            for pattern in config['patterns']:
                if re.search(pattern, question_lower):
                    score += 2  # Patterns are more specific, so higher weight
            
            if score > 0:
                type_scores[qtype] = score
        
        # Special handling for certain combinations
        type_scores = self._adjust_scores_for_context(question_lower, type_scores)
        
        # Return the type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            self.logger.debug(f"Question classified as: {best_type} (scores: {type_scores})")
            return best_type
        
        # Default fallback
        return 'other'
    
    def _adjust_scores_for_context(self, question_lower: str, scores: Dict[str, int]) -> Dict[str, int]:
        """Adjust scores based on question context"""
        
        # If it's clearly a counting question, boost counting score
        if 'how many' in question_lower:
            scores['counting'] = scores.get('counting', 0) + 3
        
        # If it asks for a specific value, boost numerical
        if 'what is the' in question_lower and 'value' in question_lower:
            scores['numerical'] = scores.get('numerical', 0) + 2
        
        # Yes/no questions should take precedence if they start with specific patterns
        if re.match(r'^(is|are|does|do|can|will)\b', question_lower):
            scores['yes_no'] = scores.get('yes_no', 0) + 3
        
        # Geometry problems often combine with calculation
        if 'geometry' in scores and 'calculation' in scores:
            scores['geometry'] += scores['calculation']
            del scores['calculation']
        
        return scores
    
    def get_answer_format(self, question_type: str) -> str:
        """Get expected answer format for question type
        
        Args:
            question_type: The classified question type
            
        Returns:
            Format description string
        """
        formats = {
            'counting': 'integer',
            'summation': 'number',
            'average': 'decimal',
            'percentage': 'percentage',
            'difference': 'number',
            'ratio': 'decimal or fraction',
            'numerical': 'number',
            'comparison': 'comparative statement',
            'minmax': 'value with label',
            'trend': 'trend description',
            'yes_no': 'Yes or No',
            'retrieval': 'specific information',
            'geometry': 'geometric value or proof',
            'calculation': 'calculated result',
            'other': 'text'
        }
        
        return formats.get(question_type, 'text')
    
    def requires_precise_calculation(self, question_type: str) -> bool:
        """Check if question type requires precise calculation
        
        Args:
            question_type: The classified question type
            
        Returns:
            True if precise calculation is needed
        """
        precise_types = {
            'counting', 'summation', 'average', 'percentage',
            'difference', 'ratio', 'numerical', 'calculation',
            'geometry'
        }
        
        return question_type in precise_types