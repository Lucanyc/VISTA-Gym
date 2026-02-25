"""ChartQA reasoning engine - Complete version with all question types"""
from typing import Dict, Any, Optional, List
from .tools import ChartQATools
from .templates import ChartQATemplates
import re

class ChartQAReasoner:
    """Main reasoning engine for ChartQA tasks"""
    
    def __init__(self):
        self.tools = ChartQATools()
        self.templates = ChartQATemplates()
        self.debug = False  # Can be set via configuration
    
    def classify_question(self, question: str) -> str:
        """Classify question type - comprehensive version"""
        question_lower = question.lower()
        
        # 1. Counting questions (how many)
        if any(phrase in question_lower for phrase in [
            'how many', 'count', 'total number', 'number of'
        ]) and not any(word in question_lower for word in ['sum', 'add', 'total of']):
            return 'counting'
        
        # 2. Summation questions
        elif any(phrase in question_lower for phrase in [
            'sum', 'add', 'total of', 'sum of', 'add up', 'combined'
        ]):
            return 'summation'
        
        # 3. Average questions
        elif any(word in question_lower for word in [
            'average', 'mean', 'avg'
        ]):
            return 'average'
        
        # 4. Percentage questions
        elif any(word in question_lower for word in [
            'percentage', 'percent', '%', 'proportion', 'share'
        ]):
            return 'percentage'
        
        # 5. Difference questions
        elif any(word in question_lower for word in [
            'difference', 'gap', 'subtract', 'minus', 'between'
        ]) and 'which' not in question_lower:
            return 'difference'
        
        # 6. Ratio questions
        elif any(word in question_lower for word in [
            'ratio', 'times', 'divide', 'divided by', 'per'
        ]):
            return 'ratio'
        
        # 7. Comparison questions
        elif any(phrase in question_lower for phrase in [
            'compare', 'which is greater', 'which is higher', 
            'which is lower', 'which is more', 'which is less',
            'vs', 'versus', 'bigger', 'smaller'
        ]):
            return 'comparison'
        
        # 8. Min/Max questions
        elif any(word in question_lower for word in [
            'maximum', 'minimum', 'highest', 'lowest', 
            'most', 'least', 'max', 'min', 'largest', 'smallest',
            'greatest', 'peak', 'bottom'
        ]):
            return 'minmax'
        
        # 9. Trend questions
        elif any(word in question_lower for word in [
            'trend', 'increase', 'decrease', 'change',
            'growth', 'decline', 'rise', 'fall', 'pattern'
        ]):
            return 'trend'
        
        # 10. Numerical value questions (general)
        elif any(phrase in question_lower for phrase in [
            'what is the value', 'how much', 'what value', 'what number'
        ]):
            return 'numerical'
        
        # 11. Retrieval questions (what, which, when, where, who)
        elif any(word in question_lower for word in [
            'what', 'which', 'when', 'where', 'who', 'whose'
        ]):
            return 'retrieval'
        
        # 12. Other
        else:
            return 'other'
    
    def reason(self, question: str, vlm_output: str, question_type: Optional[str] = None) -> Dict[str, Any]:
        """Apply structured reasoning based on question type"""
        
        if question_type is None:
            question_type = self.classify_question(question)
        
        # Call corresponding reasoning method based on question type
        reasoning_methods = {
            'counting': self._reason_counting,
            'summation': self._reason_summation,
            'average': self._reason_average,
            'percentage': self._reason_percentage,
            'difference': self._reason_difference,
            'ratio': self._reason_ratio,
            'comparison': self._reason_comparison,
            'minmax': self._reason_minmax,
            'trend': self._reason_trend,
            'numerical': self._reason_numerical,
            'retrieval': self._reason_retrieval,
            'other': self._reason_other
        }
        
        method = reasoning_methods.get(question_type, self._reason_other)
        return method(question, vlm_output)
    
    def _reason_counting(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle counting questions"""
        return self._handle_counting_question(question, vlm_output)
    
    def _reason_summation(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle summation questions - improved version"""
        if self.debug:
            print(f"[DEBUG] Handling summation question: {question[:50]}...")
        
        # 1. First try to find already calculated sum in VLM output
        sum_patterns = [
            r'(?:sum|total|add up to|adds up to)(?:\s+is)?\s*:?\s*(\d+(?:\.\d+)?)',
            r'(?:=|equals?)\s*(\d+(?:\.\d+)?)\s*(?:total|sum|in total)?',
            r'(\d+(?:\.\d+)?)\s*(?:total|sum|in total)',
        ]
        
        for pattern in sum_patterns:
            match = re.search(pattern, vlm_output, re.IGNORECASE)
            if match:
                answer = match.group(1)
                return {
                    'answer': answer,
                    'reasoning': f"Extracted calculated sum: {answer}",
                    'confidence': 0.9,
                    'structured': True,
                    'calculator_used': False
                }
        
        # 2. If not found, extract values and calculate
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # 3. Smart filtering - key improvement!
        filtered_values = []
        question_lower = question.lower()
        
        for v in values:
            val = v['value']
            label = v.get('label', '').lower()
            
            # Exclude years (unless question explicitly asks for sum of years)
            if 1900 <= val <= 2099 and 'year' not in question_lower:
                if self.debug:
                    print(f"[DEBUG] Filtering out year: {val}")
                continue
            
            # For "sum of X in year Y and Z" type questions,
            # only include relevant values, not the years themselves
            if 'in year' in question_lower or 'in the year' in question_lower:
                # Check if this value might be a year
                if 1900 <= val <= 2099:
                    continue
            
            filtered_values.append(v)
        
        if filtered_values:
            # Calculate sum
            total = sum(v['value'] for v in filtered_values)
            answer = str(int(total) if total.is_integer() else round(total, 2))
            
            # Format calculation process
            calculation = f"Sum = {' + '.join(str(v['value']) for v in filtered_values)} = {answer}"
            
            # Create reasoning description
            reasoning = f"Extracted values:\n"
            reasoning += "\n".join([f"- {v.get('label', 'Value')}: {v['value']}" for v in filtered_values])
            reasoning += f"\n{calculation}"
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'summation')
    
    def _reason_average(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle average questions"""
        if self.debug:
            print(f"[DEBUG] Handling average question: {question[:50]}...")
        
        # Extract all values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # Filter years
        filtered_values = []
        question_lower = question.lower()
        for v in values:
            val = v['value']
            if not (1900 <= val <= 2099 and 'year' not in question_lower):
                filtered_values.append(v)
        
        if filtered_values:
            # Calculate average
            avg = sum(v['value'] for v in filtered_values) / len(filtered_values)
            answer = str(round(avg, 2))
            
            # Format calculation process
            calculation = f"Average = ({' + '.join(str(v['value']) for v in filtered_values)}) / {len(filtered_values)} = {answer}"
            
            # Create reasoning description
            reasoning = f"Extracted values:\n"
            reasoning += "\n".join([f"- {v.get('label', 'Value')}: {v['value']}" for v in filtered_values])
            reasoning += f"\n{calculation}"
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'average')
    
    def _reason_percentage(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle percentage questions"""
        if self.debug:
            print(f"[DEBUG] Handling percentage question: {question[:50]}...")
        
        # Look for percentage values
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(percentage_pattern, vlm_output)
        
        if matches:
            # If percentages found, return the first one
            answer = f"{matches[0]}%"
            return {
                'answer': answer,
                'reasoning': f"Extracted percentage value: {answer}",
                'confidence': 0.9,
                'structured': True,
                'calculator_used': False
            }
        
        # If percentage needs to be calculated
        values = self.tools.extractor.extract_numbers(vlm_output)
        if len(values) >= 2:
            # Assume first is part, second is whole
            part = values[0]['value']
            whole = values[1]['value']
            if whole != 0:
                percentage = (part / whole) * 100
                answer = f"{round(percentage, 2)}%"
                
                calculation = f"Percentage = ({part} / {whole}) × 100 = {answer}"
                return {
                    'answer': answer,
                    'reasoning': calculation,
                    'confidence': 0.8,
                    'structured': True,
                    'calculator_used': True
                }
        
        return self._extract_simple_answer(vlm_output, 'percentage')
    
    def _reason_difference(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle difference questions - improved version"""
        if self.debug:
            print(f"[DEBUG] Handling difference question: {question[:50]}...")
        
        # Extract values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # Filter years
        filtered_values = []
        for v in values:
            val = v['value']
            if not (1900 <= val <= 2099 and 'year' not in v.get('label', '').lower()):
                filtered_values.append(v)
        
        if len(filtered_values) >= 2:
            val1, val2 = filtered_values[0]['value'], filtered_values[1]['value']
            
            # Check if question needs signed difference
            question_lower = question.lower()
            if 'how much more' in question_lower or 'how much higher' in question_lower:
                # Need positive difference
                diff = max(val1, val2) - min(val1, val2)
            elif 'how much less' in question_lower or 'how much lower' in question_lower:
                # Need absolute value of negative difference
                diff = max(val1, val2) - min(val1, val2)
            else:
                # Default absolute difference
                diff = abs(val1 - val2)
            
            answer = str(int(diff) if diff.is_integer() else round(diff, 2))
            
            calculation = f"Difference = |{val1} - {val2}| = {answer}"
            
            return {
                'answer': answer,
                'reasoning': calculation,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'difference')
    
    def _reason_ratio(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle ratio questions - improved version"""
        if self.debug:
            print(f"[DEBUG] Handling ratio question: {question[:50]}...")
        
        # Analyze question type
        question_lower = question.lower()
        
        # 1. Check for "A to B" format
        # e.g.: "What is the ratio of favorable to unfavorable?"
        ratio_pattern = r'ratio\s+of\s+(\w+)\s+to\s+(\w+)'
        match = re.search(ratio_pattern, question_lower)
        
        if match:
            item1, item2 = match.groups()
            # Find corresponding values in VLM output
            # This requires more intelligent extraction logic
            if self.debug:
                print(f"[DEBUG] Looking for ratio of {item1} to {item2}")
        
        # 2. Extract values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # Filter out obviously unreasonable values (like years)
        filtered_values = []
        for v in values:
            val = v['value']
            # Exclude years (1900-2099)
            if not (1900 <= val <= 2099 and 'year' not in v.get('label', '').lower()):
                filtered_values.append(v)
        
        if len(filtered_values) >= 2:
            val1, val2 = filtered_values[0]['value'], filtered_values[1]['value']
            
            # 3. Intelligently determine division order
            # If question contains "A to B", need to ensure correct order
            if 'to' in question_lower:
                # Try to understand which should be numerator
                # This may require more complex NLP processing
                pass
            
            # 4. Handle "times" type questions
            if 'times' in question_lower or 'how many times' in question_lower:
                # For "A is how many times B", usually expect larger value divided by smaller
                if val1 > val2:
                    ratio = val1 / val2
                else:
                    ratio = val2 / val1
            else:
                # Standard ratio calculation
                if val2 != 0:
                    ratio = val1 / val2
                else:
                    return self._extract_simple_answer(vlm_output, 'ratio')
            
            # 5. Format answer
            # If ratio is close to integer, return integer
            if abs(ratio - round(ratio)) < 0.01:
                answer = str(int(round(ratio)))
            else:
                answer = str(round(ratio, 2))
            
            calculation = f"Ratio = {val1} / {val2} = {answer}"
            
            return {
                'answer': answer,
                'reasoning': calculation,
                'confidence': 0.85,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'ratio')
    
    def _reason_comparison(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle comparison questions"""
        # 1. Extract values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        # 2. Calculate difference
        if len(values) >= 2:
            diff = self.tools.calculate_difference(
                values[0]['value'], 
                values[1]['value']
            )
            
            # 3. Format answer
            template = self.templates.comparison_template()
            reasoning = template['format'].format(
                element_a=values[0]['label'],
                element_b=values[1]['label'],
                value_a=values[0]['value'],
                value_b=values[1]['value'],
                result=diff
            )
            
            # 4. Decide answer format
            answer = str(int(diff) if diff.is_integer() else round(diff, 2))
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.9,
                'structured': True,
                'calculator_used': True
            }
        
        return self._extract_simple_answer(vlm_output, 'comparison')
    
    def _reason_minmax(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle min/max questions"""
        # 1. Extract all values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # 2. Find min/max value
            result = self.tools.find_min_max(values)
            
            # 3. Determine if question asks for min or max
            question_lower = question.lower()
            if any(word in question_lower for word in ['minimum', 'lowest', 'least', 'smallest']):
                target = result['min']
                answer_type = "Minimum"
            else:
                target = result['max']
                answer_type = "Maximum"
            
            # 4. Format reasoning process
            values_list = "\n".join([f"- {v['label']}: {v['value']}" for v in values])
            template = self.templates.minmax_template()
            reasoning = template['format'].format(
                values_list=values_list,
                min_label=result['min']['label'],
                min_value=result['min']['value'],
                max_label=result['max']['label'],
                max_value=result['max']['value'],
                answer=f"{target['label']} ({target['value']})"
            )
            
            # 5. If question only asks for value or label
            if 'value' in question_lower or 'number' in question_lower:
                answer = str(target['value'])
            else:
                answer = target['label']
            
            return {
                'answer': answer,
                'reasoning': reasoning,
                'confidence': 0.85,
                'structured': True,
                'structured_answer': target['value']  # Save numeric value for evaluation
            }
        
        return self._extract_simple_answer(vlm_output, 'minmax')
    
    def _reason_trend(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle trend questions"""
        if self.debug:
            print(f"[DEBUG] Handling trend question: {question[:50]}...")
        
        # Look for trend-related keywords
        trend_keywords = {
            'increasing': ['increasing', 'rising', 'growing', 'upward', 'going up'],
            'decreasing': ['decreasing', 'falling', 'declining', 'downward', 'going down'],
            'stable': ['stable', 'constant', 'unchanged', 'flat', 'steady']
        }
        
        vlm_lower = vlm_output.lower()
        
        for trend, keywords in trend_keywords.items():
            if any(keyword in vlm_lower for keyword in keywords):
                return {
                    'answer': trend,
                    'reasoning': f"Identified trend: {trend}",
                    'confidence': 0.8,
                    'structured': True,
                    'calculator_used': False
                }
        
        return self._extract_simple_answer(vlm_output, 'trend')
    
    def _reason_numerical(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle general numerical questions"""
        # If "how many" type, redirect to counting handler
        if 'how many' in question.lower():
            return self._handle_counting_question(question, vlm_output)
        
        # Extract values
        values = self.tools.extractor.extract_numbers(vlm_output)
        
        if values:
            # Usually return the first value found
            answer = str(values[0]['value'])
            
            return {
                'answer': answer,
                'reasoning': f"Extracted value: {answer}",
                'confidence': 0.8,
                'structured': True,
                'calculator_used': False
            }
        
        return self._extract_simple_answer(vlm_output, 'numerical')
    
    def _reason_retrieval(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle retrieval questions (what, which, when, where, who)"""
        if self.debug:
            print(f"[DEBUG] Handling retrieval question: {question[:50]}...")
        
        # For retrieval questions, answer is usually a label, name, or description
        # Try to extract a short answer from VLM output
        
        # Look for common answer patterns
        answer_patterns = [
            r'(?:The answer is|Answer:)\s*([^.!?\n]+)',
            r'(?:It is|It\'s|This is)\s*([^.!?\n]+)',
            r'(?:^|\n)([A-Z][^.!?\n]{5,50})(?:\.|!|\?|$)',  # Short sentence starting with capital letter
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, vlm_output, re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                return {
                    'answer': answer,
                    'reasoning': f"Retrieved information: {answer}",
                    'confidence': 0.7,
                    'structured': True,
                    'calculator_used': False
                }
        
        # If no specific pattern found, return first sentence of response
        first_sentence = vlm_output.split('.')[0].strip()
        if len(first_sentence) < 100:  # Ensure not too long
            return {
                'answer': first_sentence,
                'reasoning': "Extracted from response",
                'confidence': 0.5,
                'structured': True,
                'calculator_used': False
            }
        
        return self._extract_simple_answer(vlm_output, 'retrieval')
    
    def _reason_other(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Handle other types of questions"""
        return self._extract_simple_answer(vlm_output, 'other')
    
    def _extract_simple_answer(self, vlm_output: str, question_type: str) -> Dict[str, Any]:
        """Extract simple answer when structured reasoning fails"""
        # Try to extract final answer
        answer_patterns = [
            r'(?:The answer is|Answer:)\s*([^.!?\n]+)',
            r'(?:Therefore|Thus|So),?\s*([^.!?\n]+)',
            r'(?:^|\n)(\d+(?:\.\d+)?)',  # Standalone number
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, vlm_output, re.MULTILINE | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return {
                    'answer': answer,
                    'reasoning': f"Extracted from VLM output ({question_type})",
                    'confidence': 0.4,
                    'structured': False,
                    'calculator_used': False
                }
        
        # If still not found, return first 100 characters of raw output
        answer = vlm_output.strip()[:100]
        if len(vlm_output.strip()) > 100:
            answer += "..."
        
        return {
            'answer': answer,
            'confidence': 0.3,
            'structured': False,
            'calculator_used': False
        }
    
    def _handle_counting_question(self, question: str, vlm_output: str) -> Dict[str, Any]:
        """Special handling for counting questions - enhanced version"""
        import re
        
        # Debug output
        if self.debug:
            print(f"[DEBUG] Handling counting question: {question[:50]}...")
            print(f"[DEBUG] VLM output: {vlm_output[:200]}...")
        
        # Extended pattern list, ordered by priority
        patterns = [
            # Direct numeric answer (at beginning or standalone line)
            (r'^(\d{1,2})$', 0.95),  # 1-2 digit number at line start
            (r'^(\d{1,2})\.$', 0.95),  # 1-2 digit number followed by period
            
            # "X years/values/colors" etc. - limit to 1-2 digits
            (r'(\d{1,2})\s+(?:years?|values?|colors?|items?|points?|entries?|data\s*points?|countries?|bars?|lines?)', 0.9),
            (r'(?:compares?\s+data\s+for|uses?|has?|contains?|shows?|features?|represents?)\s+(\d{1,2})\s+(?:years?|values?|colors?|countries?)', 0.9),
            
            # "there are X" format - limit to 1-3 digits
            (r'there\s+(?:are|is)\s+(\d{1,3})(?:\s|$)', 0.9),
            (r'(?:found|counted|identified)\s+(\d{1,3})(?:\s|$)', 0.9),
            
            # "X values are/were" format
            (r'(\d{1,3})\s+(?:values?|points?|items?|countries?|bars?)\s+(?:are|were|is|was)', 0.85),
            
            # Various expressions containing numbers
            (r'(?:answer|total|count)(?:\s+is)?:?\s*(\d{1,3})(?:\s|$)', 0.85),
            (r'(?:exactly|total\s+of|in\s+total)\s+(\d{1,3})(?:\s|$)', 0.85),
            
            # Extract any standalone number (last resort) - but needs filtering
            (r'(?:^|\s)(\d+)(?:\s|$)', 0.5),
        ]
        
        # Number word mapping
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # Check number words first
        for word, num in word_to_num.items():
            if word in vlm_output.lower():
                if self.debug:
                    print(f"[DEBUG] Found number word '{word}': {num}")
                return {
                    'answer': str(num),
                    'reasoning': f"Number word '{word}' extracted from text",
                    'confidence': 0.9,
                    'structured': True,
                    'calculator_used': False
                }
        
        # Store all found candidate answers
        candidates = []
        
        # Try all patterns
        for i, (pattern, confidence_base) in enumerate(patterns):
            matches = re.findall(pattern, vlm_output, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    try:
                        num = int(match)
                        
                        # Filter out years (1900-2099) unless question is explicitly about years
                        if 1900 <= num <= 2099 and 'year' not in question.lower():
                            if self.debug:
                                print(f"[DEBUG] Filtered out year: {num}")
                            continue
                        
                        # For "how many" questions, numbers greater than 100 are less likely
                        if num > 100 and 'how many' in question.lower():
                            confidence = confidence_base * 0.5  # Lower confidence
                        else:
                            confidence = confidence_base
                        
                        candidates.append({
                            'value': num,
                            'pattern_index': i,
                            'confidence': confidence,
                            'pattern': pattern
                        })
                        
                    except ValueError:
                        continue
        
        # Select best candidate based on confidence and reasonableness
        if candidates:
            # Prioritize smaller numbers (more reasonable for counting questions)
            candidates.sort(key=lambda x: (
                -x['confidence'],  # Higher confidence first
                x['value'] if x['value'] < 50 else 1000 + x['value']  # Numbers below 50 first
            ))
            
            best_candidate = candidates[0]
            if self.debug:
                print(f"[DEBUG] Selected count: {best_candidate['value']} (confidence: {best_candidate['confidence']:.2f})")
            
            return {
                'answer': str(best_candidate['value']),
                'reasoning': f"Count extracted with pattern {best_candidate['pattern_index']+1}",
                'confidence': best_candidate['confidence'],
                'structured': True,
                'calculator_used': False
            }
        
        # If no clear count found, return unstructured result
        if self.debug:
            print(f"[DEBUG] Could not extract count, returning original output")
        
        return {
            'answer': vlm_output,
            'confidence': 0.3,
            'structured': False,
            'calculator_used': False
        }
