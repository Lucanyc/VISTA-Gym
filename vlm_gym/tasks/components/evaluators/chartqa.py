#!/usr/bin/env python3
"""ChartQA-specific evaluator components"""

import re
from typing import Optional
from ...base.components import Evaluator


class ChartQAEvaluator(Evaluator):
    """Evaluator for ChartQA tasks with numerical tolerance"""
    
    def __init__(self, tolerance: float = 0.05):
        """Initialize evaluator
        
        Args:
            tolerance: Relative tolerance for numerical answers (default 5%)
        """
        self.tolerance = tolerance
    
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """Evaluate ChartQA answer with flexible matching"""
        # Normalize both answers
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match after normalization
        if pred_norm == gt_norm:
            return True
        
        # Extract numbers
        pred_num = self._extract_number(prediction)
        gt_num = self._extract_number(ground_truth)
        
        # For non-numerical answers, check containment
        if pred_num is None or gt_num is None:
            if gt_norm in pred_norm:
                # Extra check for numbers
                if gt_num is not None:
                    pattern = r'\b' + re.escape(gt_norm) + r'\b'
                    if not re.search(pattern, pred_norm):
                        return False
                return True
        
        # For numerical answers
        if pred_num is not None and gt_num is not None:
            # Strict match if tolerance is 0
            if self.tolerance == 0:
                return pred_num == gt_num
            
            # Exact match for numbers
            if abs(pred_num - gt_num) < 1e-9:
                return True
            
            # Year-like values require exact integer match (no tolerance)
            # Years are 4-digit integers in range 1900-2100
            if self._is_year_value(gt_num):
                return int(pred_num) == int(gt_num)
            
            # Large integer identifiers (e.g., IDs, codes) require exact match
            # If both are integers and gt > 1000, use exact match to avoid
            # false positives from tolerance on large numbers
            if (gt_num == int(gt_num) and pred_num == int(pred_num) 
                and abs(gt_num) >= 1000 and self._looks_like_identifier(gt_norm)):
                return int(pred_num) == int(gt_num)
            
            # Relative tolerance check
            if self.tolerance > 0 and gt_num != 0:
                relative_error = abs(pred_num - gt_num) / abs(gt_num)
                if relative_error <= self.tolerance:
                    return True
            elif self.tolerance > 0 and pred_num == 0:
                return True
            
            # Check percentage conversion
            if abs(pred_num * 100 - gt_num) < 1e-9 or abs(pred_num - gt_num * 100) < 1e-9:
                return True
        
        # Handle yes/no variations
        yes_variations = ['yes', 'true', 'correct', 'right', 'affirmative', '1']
        no_variations = ['no', 'false', 'incorrect', 'wrong', 'negative', '0']
        
        if gt_norm in yes_variations:
            return any(var in pred_norm for var in yes_variations)
        if gt_norm in no_variations:
            return any(var in pred_norm for var in no_variations)
        
        return False
    
    def _is_year_value(self, num: float) -> bool:
        """Check if a number looks like a year value
        
        Years are 4-digit integers in the range 1900-2100.
        These should always use exact match, never tolerance.
        
        Args:
            num: The number to check
            
        Returns:
            True if the number looks like a year
        """
        if num != int(num):
            return False
        return 1900 <= int(num) <= 2100
    
    def _looks_like_identifier(self, text: str) -> bool:
        """Check if normalized text looks like an identifier/code rather than a quantity
        
        Args:
            text: Normalized answer text
            
        Returns:
            True if it looks like an identifier
        """
        # Pure number string without units or context = likely an identifier
        text = text.strip()
        if re.match(r'^-?\d+$', text):
            return True
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = str(answer).lower().strip()
        answer = answer.rstrip('.,!?;:')
        answer = answer.strip('"\'')
        
        prefixes_to_remove = [
            'the answer is', 'answer:', 'final answer:',
            'therefore,', 'so,', 'thus,'
        ]
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        return answer
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text"""
        if not text:
            return None
        
        text = str(text).lower().strip()
        text = text.replace(',', '').replace('$', '')
        
        # Handle written numbers
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
            'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'million': 1000000
        }
        
        words = text.split()
        for word in words:
            cleaned_word = word.strip('.,!?;:')
            if cleaned_word in number_words:
                return float(number_words[cleaned_word])
        
        # Check for percentage
        has_percentage = False
        if '%' in text:
            has_percentage = True
        elif 'percent' in text and 'percentage point' not in text:
            has_percentage = True
        
        # Handle K/M/B suffixes
        suffix_pattern = r'(-?\d+\.?\d*)\s*([KMB])'
        suffix_match = re.search(suffix_pattern, text.upper())
        if suffix_match:
            number = float(suffix_match.group(1))
            suffix = suffix_match.group(2)
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
            return number * multipliers.get(suffix, 1)
        
        # Extract regular decimal numbers
        decimal_pattern = r'-?\d+\.?\d*'
        numbers = re.findall(decimal_pattern, text)
        
        if numbers:
            try:
                value = float(numbers[0])
                if has_percentage and value > 1:
                    value = value / 100.0
                return value
            except ValueError:
                pass
        
        return None