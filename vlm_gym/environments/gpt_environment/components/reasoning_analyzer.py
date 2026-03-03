# vlm_gym/environments/gpt_environment/components/reasoning_analyzer.py

"""
Reasoning Analyzer for GPT Environment
Analyzes student responses and reasoning quality for intelligent teaching decisions
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReasoningAnalysis:
    """Results of reasoning analysis"""
    # Core metrics
    is_correct: bool = False
    partially_correct: bool = False
    reasoning_quality: float = 0.0  # 0-1 score
    progress: float = 0.0  # Progress toward solution
    
    # Detailed analysis
    completed_steps: List[str] = field(default_factory=list)
    missing_steps: List[str] = field(default_factory=list)
    misconceptions: List[str] = field(default_factory=list)
    unclear_points: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    
    # Quality dimensions
    clarity_score: float = 0.0
    completeness_score: float = 0.0
    correctness_score: float = 0.0
    logical_flow_score: float = 0.0
    
    # Flags
    is_stuck: bool = False
    is_confused: bool = False
    is_making_progress: bool = False
    student_gave_up: bool = False
    is_invalid: bool = False
    
    # Additional metadata
    confidence_level: float = 0.5
    error_types: List[str] = field(default_factory=list)
    next_step_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_correct': self.is_correct,
            'partially_correct': self.partially_correct,
            'reasoning_quality': self.reasoning_quality,
            'progress': self.progress,
            'completed_steps': self.completed_steps,
            'missing_steps': self.missing_steps,
            'misconceptions': self.misconceptions,
            'unclear_points': self.unclear_points,
            'strengths': self.strengths,
            'clarity_score': self.clarity_score,
            'completeness_score': self.completeness_score,
            'correctness_score': self.correctness_score,
            'logical_flow_score': self.logical_flow_score,
            'is_stuck': self.is_stuck,
            'is_confused': self.is_confused,
            'is_making_progress': self.is_making_progress,
            'student_gave_up': self.student_gave_up,
            'is_invalid': self.is_invalid,
            'confidence_level': self.confidence_level,
            'error_types': self.error_types,
            'next_step_suggestions': self.next_step_suggestions
        }


class ReasoningAnalyzer:
    """
    Analyzes student reasoning quality and progress
    
    Key responsibilities:
    1. Evaluate reasoning correctness and quality
    2. Identify completed and missing steps
    3. Detect misconceptions and errors
    4. Assess progress toward solution
    5. Provide actionable feedback for teaching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reasoning analyzer
        
        Args:
            config: Configuration dictionary with:
                - strict_mode: Whether to use strict correctness checking
                - min_quality_threshold: Minimum quality score threshold
                - task_specific_analyzers: Custom analyzers for task types
        """
        config = config or {}
        self.strict_mode = config.get('strict_mode', False)
        self.min_quality_threshold = config.get('min_quality_threshold', 0.3)
        
        # Task-specific reasoning patterns
        self.task_patterns = self._initialize_task_patterns()
        
        # Common reasoning indicators
        self.reasoning_indicators = self._initialize_reasoning_indicators()
        
        # Error patterns
        self.error_patterns = self._initialize_error_patterns()
        
    def analyze(self, response: str, task: Any, context: List[Dict[str, str]], 
                expected_reasoning: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze student's response
        
        Args:
            response: Student's response text
            task: Task object with answer and metadata
            context: Recent dialogue context
            expected_reasoning: Expected reasoning steps (optional)
            
        Returns:
            Analysis results dictionary
        """
        # Extract task information
        task_info = self._extract_task_info(task)
        
        # Create analysis object
        analysis = ReasoningAnalysis()
        
        # Basic validation
        if not response or len(response.strip()) < 5:
            analysis.is_invalid = True
            analysis.reasoning_quality = 0.0
            return analysis.to_dict()
        
        # Check if student gave up
        if self._check_gave_up(response):
            analysis.student_gave_up = True
            analysis.reasoning_quality = 0.1
            return analysis.to_dict()
        
        # Analyze different aspects
        analysis.is_correct = self._check_correctness(response, task_info['answer'], task_info['type'])
        analysis.partially_correct = self._check_partial_correctness(response, task_info)
        
        # Analyze reasoning steps
        if expected_reasoning:
            self._analyze_reasoning_steps(response, expected_reasoning, analysis)
        else:
            self._analyze_general_reasoning(response, task_info, analysis)
        
        # Analyze quality dimensions
        self._analyze_clarity(response, analysis)
        self._analyze_completeness(response, task_info, context, analysis)
        self._analyze_logical_flow(response, context, analysis)
        
        # Detect issues
        self._detect_misconceptions(response, task_info, analysis)
        self._detect_unclear_points(response, analysis)
        self._identify_strengths(response, task_info, analysis)
        
        # Calculate overall scores
        self._calculate_progress(analysis, context)
        self._calculate_reasoning_quality(analysis)
        
        # Determine states
        self._determine_student_state(response, context, analysis)
        
        # Generate suggestions
        self._generate_next_step_suggestions(analysis, task_info)
        
        return analysis.to_dict()
    
    def _extract_task_info(self, task: Any) -> Dict[str, Any]:
        """Extract task information"""
        # Try to get unified info if available
        if hasattr(task, 'get_gpt_teacher_info'):
            return task.get_gpt_teacher_info()
        
        # Otherwise extract manually
        return {
            'type': self._infer_task_type(task),
            'answer': self._get_task_answer(task),
            'question': getattr(task, 'question', ''),
            'metadata': getattr(task, 'metadata', {})
        }
    
    def _infer_task_type(self, task: Any) -> str:
        """Infer task type from task object"""
        class_name = task.__class__.__name__.lower()
        for task_type in ['figureqa', 'chartqa', 'clevr', 'geometry3k', 'geoqa', 
                         'iconqa', 'scienceqa', 'mathvista', 'olympiadbench']:
            if task_type in class_name:
                return task_type
        return 'unknown'
    
    def _get_task_answer(self, task: Any) -> Any:
        """Get answer from task object"""
        for attr in ['answer', 'solution', 'correct_answer', 'ground_truth']:
            if hasattr(task, attr):
                return getattr(task, attr)
        return None
    
    def _check_gave_up(self, response: str) -> bool:
        """Check if student gave up"""
        give_up_phrases = [
            "i don't know", "i give up", "i'm not sure", "i can't solve",
            "too difficult", "i'm stuck", "no idea", "i don't understand"
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in give_up_phrases)
    
    def _check_correctness(self, response: str, correct_answer: Any, task_type: str) -> bool:
        """Check if the answer is correct"""
        if correct_answer is None:
            return False
        
        # Extract answer from response
        extracted_answer = self._extract_answer(response, task_type)
        
        # Convert to strings for comparison
        response_answer = str(extracted_answer).lower().strip()
        expected_answer = str(correct_answer).lower().strip()
        
        # Direct match
        if response_answer == expected_answer:
            return True
        
        # Task-specific checking
        if task_type in ['figureqa', 'clevr', 'iconqa']:
            # Yes/No questions
            if expected_answer in ['yes', 'true', '1']:
                return response_answer in ['yes', 'true', '1', 'correct']
            elif expected_answer in ['no', 'false', '0']:
                return response_answer in ['no', 'false', '0', 'incorrect']
        
        # Numeric comparison with tolerance
        try:
            response_num = float(response_answer)
            expected_num = float(expected_answer)
            return abs(response_num - expected_num) < 0.01
        except:
            pass
        
        # Substring matching for longer answers
        if len(expected_answer) > 10:
            return expected_answer in response_answer or response_answer in expected_answer
        
        return False
    
    def _extract_answer(self, response: str, task_type: str) -> str:
        """Extract answer from response based on task type"""
        # Common answer patterns
        patterns = [
            r"(?:the answer is|my answer is|therefore,?) (.+?)(?:\.|$)",
            r"(?:so|thus|hence) (.+?) is the answer",
            r"= (.+?)(?:\.|$)",
            r"(?:yes|no)\b",
            r"\b(\d+(?:\.\d+)?)\b"  # Numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower(), re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        # If no pattern matches, return last sentence
        sentences = re.split(r'[.!?]+', response)
        return sentences[-1].strip() if sentences else response
    
    def _check_partial_correctness(self, response: str, task_info: Dict[str, Any]) -> bool:
        """Check if the answer is partially correct"""
        # For multi-step problems, check if some steps are correct
        if 'steps' in task_info.get('metadata', {}):
            correct_steps = sum(1 for step in task_info['metadata']['steps'] 
                              if step.lower() in response.lower())
            return correct_steps > 0 and correct_steps < len(task_info['metadata']['steps'])
        
        # For numeric problems, check if close to answer
        try:
            answer = float(task_info['answer'])
            extracted = float(self._extract_answer(response, task_info['type']))
            relative_error = abs(extracted - answer) / abs(answer)
            return 0.01 < relative_error < 0.1  # Within 10% but not exact
        except:
            pass
        
        return False
    
    def _analyze_reasoning_steps(self, response: str, expected_steps: List[str], 
                               analysis: ReasoningAnalysis):
        """Analyze reasoning against expected steps"""
        response_lower = response.lower()
        
        for step in expected_steps:
            step_keywords = self._extract_keywords(step)
            
            # Check if step is mentioned
            if any(keyword in response_lower for keyword in step_keywords):
                analysis.completed_steps.append(step)
            else:
                analysis.missing_steps.append(step)
        
        # Calculate completeness
        if expected_steps:
            analysis.completeness_score = len(analysis.completed_steps) / len(expected_steps)
    
    def _analyze_general_reasoning(self, response: str, task_info: Dict[str, Any], 
                                 analysis: ReasoningAnalysis):
        """Analyze reasoning without explicit expected steps"""
        task_type = task_info['type']
        
        # Get task-specific patterns
        if task_type in self.task_patterns:
            patterns = self.task_patterns[task_type]
            
            for step_name, pattern_list in patterns.items():
                if any(re.search(pattern, response, re.IGNORECASE) for pattern in pattern_list):
                    analysis.completed_steps.append(step_name)
        
        # General reasoning indicators
        reasoning_found = []
        for indicator_type, patterns in self.reasoning_indicators.items():
            if any(re.search(pattern, response, re.IGNORECASE) for pattern in patterns):
                reasoning_found.append(indicator_type)
        
        # Estimate completeness based on reasoning indicators
        expected_indicators = 3  # Observation, analysis, conclusion
        analysis.completeness_score = min(1.0, len(reasoning_found) / expected_indicators)
    
    def _analyze_clarity(self, response: str, analysis: ReasoningAnalysis):
        """Analyze response clarity"""
        # Factors affecting clarity
        clarity_factors = {
            'structured': bool(re.search(r'(first|second|then|finally)', response.lower())),
            'complete_sentences': len(re.findall(r'[.!?]', response)) > 1,
            'proper_length': 50 < len(response) < 500,
            'uses_connectors': bool(re.search(r'(because|therefore|thus|so)', response.lower())),
            'no_contradictions': not bool(re.search(r'(but actually|wait|no,)', response.lower()))
        }
        
        # Calculate clarity score
        analysis.clarity_score = sum(clarity_factors.values()) / len(clarity_factors)
        
        # Identify unclear points
        if not clarity_factors['complete_sentences']:
            analysis.unclear_points.append("Incomplete sentences")
        if not clarity_factors['uses_connectors']:
            analysis.unclear_points.append("Missing logical connections")
        if not clarity_factors['no_contradictions']:
            analysis.unclear_points.append("Contains contradictions")
    
    def _analyze_completeness(self, response: str, task_info: Dict[str, Any], 
                            context: List[Dict[str, str]], analysis: ReasoningAnalysis):
        """Analyze response completeness"""
        # Check if all parts of question are addressed
        question = task_info.get('question', '')
        question_parts = self._identify_question_parts(question)
        
        addressed_parts = 0
        for part in question_parts:
            if any(keyword in response.lower() for keyword in self._extract_keywords(part)):
                addressed_parts += 1
        
        if question_parts:
            question_coverage = addressed_parts / len(question_parts)
        else:
            question_coverage = 1.0
        
        # Combine with step completeness
        analysis.completeness_score = (analysis.completeness_score + question_coverage) / 2
    
    def _analyze_logical_flow(self, response: str, context: List[Dict[str, str]], 
                            analysis: ReasoningAnalysis):
        """Analyze logical flow of reasoning"""
        # Check for logical connectors
        logical_connectors = [
            'therefore', 'thus', 'hence', 'because', 'since', 'as a result',
            'consequently', 'it follows that', 'this means', 'so'
        ]
        
        connector_count = sum(1 for conn in logical_connectors if conn in response.lower())
        
        # Check for step progression
        progression_markers = ['first', 'second', 'then', 'next', 'finally']
        has_progression = any(marker in response.lower() for marker in progression_markers)
        
        # Check for evidence-based reasoning
        evidence_phrases = ['based on', 'according to', 'from the', 'shows that', 'indicates']
        uses_evidence = any(phrase in response.lower() for phrase in evidence_phrases)
        
        # Calculate score
        flow_factors = {
            'has_connectors': connector_count > 0,
            'adequate_connectors': connector_count >= 2,
            'has_progression': has_progression,
            'uses_evidence': uses_evidence,
            'coherent_length': 100 < len(response) < 1000
        }
        
        analysis.logical_flow_score = sum(flow_factors.values()) / len(flow_factors)
    
    def _detect_misconceptions(self, response: str, task_info: Dict[str, Any], 
                             analysis: ReasoningAnalysis):
        """Detect common misconceptions"""
        task_type = task_info['type']
        
        # Task-specific misconceptions
        misconception_patterns = {
            'figureqa': [
                ('color_confusion', r'red.*blue|blue.*red', "Confused colors"),
                ('axis_misread', r'x.*axis.*y|y.*axis.*x', "Confused axes"),
                ('comparison_error', r'greater.*less|less.*greater', "Reversed comparison")
            ],
            'geometry': [
                ('angle_sum_error', r'angles.*sum.*180', "Incorrect angle sum"),
                ('formula_error', r'area.*perimeter|perimeter.*area', "Confused formulas"),
                ('unit_error', r'cm.*m|m.*cm', "Unit confusion")
            ],
            'clevr': [
                ('counting_error', r'count.*all|total.*objects', "Counting error"),
                ('attribute_error', r'color.*shape|shape.*color', "Attribute confusion"),
                ('spatial_error', r'left.*right|right.*left', "Spatial confusion")
            ]
        }
        
        # Check for task-specific misconceptions
        patterns = misconception_patterns.get(task_type, [])
        if task_type == 'geometry3k' or task_type == 'geoqa':
            patterns = misconception_patterns.get('geometry', [])
        
        for error_type, pattern, description in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                analysis.misconceptions.append(description)
                analysis.error_types.append(error_type)
    
    def _detect_unclear_points(self, response: str, analysis: ReasoningAnalysis):
        """Detect unclear or ambiguous points"""
        # Check for vague language
        vague_phrases = [
            'somehow', 'maybe', 'probably', 'i think', 'it seems',
            'kind of', 'sort of', 'not sure', 'possibly'
        ]
        
        for phrase in vague_phrases:
            if phrase in response.lower():
                analysis.unclear_points.append(f"Vague language: '{phrase}'")
        
        # Check for incomplete thoughts
        if response.count('...') > 1:
            analysis.unclear_points.append("Multiple incomplete thoughts")
        
        # Check for questions in response
        if '?' in response:
            analysis.unclear_points.append("Contains unanswered questions")
    
    def _identify_strengths(self, response: str, task_info: Dict[str, Any], 
                          analysis: ReasoningAnalysis):
        """Identify strengths in reasoning"""
        # Clear observation
        if re.search(r'(I can see|I observe|I notice)', response, re.IGNORECASE):
            analysis.strengths.append("Clear observation")
        
        # Systematic approach
        if re.search(r'(first.*then|step by step)', response, re.IGNORECASE):
            analysis.strengths.append("Systematic approach")
        
        # Evidence-based reasoning
        if re.search(r'(because|based on|according to)', response, re.IGNORECASE):
            analysis.strengths.append("Evidence-based reasoning")
        
        # Self-correction
        if re.search(r'(actually|wait|let me correct)', response, re.IGNORECASE):
            analysis.strengths.append("Self-correction")
        
        # Complete answer
        if analysis.is_correct or analysis.partially_correct:
            analysis.strengths.append("Reached valid conclusion")
    
    def _calculate_progress(self, analysis: ReasoningAnalysis, context: List[Dict[str, str]]):
        """Calculate progress toward solution"""
        # Base progress on completed steps
        if analysis.completed_steps:
            step_progress = len(analysis.completed_steps) / max(1, 
                len(analysis.completed_steps) + len(analysis.missing_steps))
        else:
            step_progress = 0.0
        
        # Adjust for correctness
        if analysis.is_correct:
            correctness_factor = 1.0
        elif analysis.partially_correct:
            correctness_factor = 0.7
        else:
            correctness_factor = 0.3
        
        # Consider quality factors
        quality_factor = (analysis.clarity_score + analysis.logical_flow_score) / 2
        
        # Calculate weighted progress
        analysis.progress = (step_progress * 0.5 + 
                           correctness_factor * 0.3 + 
                           quality_factor * 0.2)
        
        # Determine if making progress
        analysis.is_making_progress = analysis.progress > 0.3
    
    def _calculate_reasoning_quality(self, analysis: ReasoningAnalysis):
        """Calculate overall reasoning quality score"""
        # Weight different dimensions
        weights = {
            'clarity': 0.2,
            'completeness': 0.3,
            'correctness': 0.3,
            'logical_flow': 0.2
        }
        
        # Calculate weighted score
        analysis.reasoning_quality = (
            weights['clarity'] * analysis.clarity_score +
            weights['completeness'] * analysis.completeness_score +
            weights['correctness'] * (1.0 if analysis.is_correct else 0.5 if analysis.partially_correct else 0.0) +
            weights['logical_flow'] * analysis.logical_flow_score
        )
        
        # Adjust for major issues
        if analysis.student_gave_up:
            analysis.reasoning_quality *= 0.2
        if len(analysis.misconceptions) > 2:
            analysis.reasoning_quality *= 0.7
        if analysis.is_invalid:
            analysis.reasoning_quality = 0.0
    
    def _determine_student_state(self, response: str, context: List[Dict[str, str]], 
                               analysis: ReasoningAnalysis):
        """Determine student's current state"""
        # Check if stuck
        if len(context) > 3:
            recent_responses = [turn['content'] for turn in context[-3:] if turn['role'] == 'agent']
            if len(recent_responses) >= 2:
                # Check for repetition
                if recent_responses[-1].lower()[:50] == recent_responses[-2].lower()[:50]:
                    analysis.is_stuck = True
        
        # Check if confused
        confusion_indicators = ['confused', "don't understand", 'not sure what', 'lost']
        if any(indicator in response.lower() for indicator in confusion_indicators):
            analysis.is_confused = True
        
        # Multiple unclear points also indicate confusion
        if len(analysis.unclear_points) > 2:
            analysis.is_confused = True
        
        # Estimate confidence
        if analysis.is_correct and analysis.clarity_score > 0.7:
            analysis.confidence_level = 0.8
        elif analysis.is_confused or analysis.student_gave_up:
            analysis.confidence_level = 0.2
        else:
            analysis.confidence_level = 0.5 + (analysis.reasoning_quality * 0.3)
    
    def _generate_next_step_suggestions(self, analysis: ReasoningAnalysis, 
                                      task_info: Dict[str, Any]):
        """Generate suggestions for next steps"""
        if analysis.is_stuck:
            analysis.next_step_suggestions.append("Provide a hint about the next step")
        
        if analysis.is_confused:
            analysis.next_step_suggestions.append("Clarify the task requirements")
        
        if analysis.missing_steps:
            analysis.next_step_suggestions.append(f"Guide toward: {analysis.missing_steps[0]}")
        
        if analysis.misconceptions:
            analysis.next_step_suggestions.append(f"Address misconception: {analysis.misconceptions[0]}")
        
        if analysis.unclear_points:
            analysis.next_step_suggestions.append("Ask for clarification on unclear points")
        
        if analysis.is_making_progress and not analysis.is_correct:
            analysis.next_step_suggestions.append("Encourage and scaffold next step")
    
    def _initialize_task_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize task-specific reasoning patterns"""
        return {
            'figureqa': {
                'identify_chart': [r'chart|graph|plot|diagram'],
                'read_values': [r'value|number|amount|quantity'],
                'compare': [r'compare|greater|less|maximum|minimum'],
                'conclude': [r'therefore|answer is|conclude']
            },
            'chartqa': {
                'identify_type': [r'bar chart|line graph|pie chart|scatter plot'],
                'read_axes': [r'x.axis|y.axis|horizontal|vertical'],
                'extract_data': [r'value|data point|shows|indicates'],
                'calculate': [r'sum|total|average|difference']
            },
            'clevr': {
                'identify_objects': [r'objects?|shapes?|cubes?|spheres?'],
                'count': [r'count|how many|number of|total'],
                'filter': [r'red|blue|green|large|small|metal|rubber'],
                'spatial': [r'left|right|behind|front|between']
            },
            'geometry3k': {
                'identify_given': [r'given|know that|tells us'],
                'apply_theorem': [r'theorem|formula|equation|rule'],
                'calculate': [r'calculate|solve|find|compute'],
                'verify': [r'check|verify|confirm|makes sense']
            }
        }
    
    def _initialize_reasoning_indicators(self) -> Dict[str, List[str]]:
        """Initialize general reasoning indicators"""
        return {
            'observation': [
                r'I (?:see|observe|notice)',
                r'The (?:image|chart|diagram) shows',
                r'Looking at'
            ],
            'analysis': [
                r'This means',
                r'By comparing',
                r'Analyzing'
            ],
            'inference': [
                r'Therefore',
                r'This suggests',
                r'I can conclude'
            ],
            'evidence': [
                r'Based on',
                r'According to',
                r'The evidence shows'
            ]
        }
    
    def _initialize_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize common error patterns"""
        return {
            'calculation_error': [r'calculate|compute|solve', r'wrong|incorrect|error'],
            'reading_error': [r'read|see|observe', r'mistake|misread|wrong'],
            'logic_error': [r'therefore|thus|so', r'but|however|actually'],
            'comprehension_error': [r'understand|mean|interpret', r'confused|unclear']
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching"""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords
    
    def _identify_question_parts(self, question: str) -> List[str]:
        """Identify distinct parts of a question"""
        # Split by conjunctions and question words
        parts = re.split(r'[,;]|and|or|what|which|how', question.lower())
        return [part.strip() for part in parts if len(part.strip()) > 5]
    
    def assess_dialogue_quality(self, dialogue: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Assess overall dialogue quality for collection decision
        
        Args:
            dialogue: Complete dialogue history
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'reasoning_clarity': 0.0,
            'step_completeness': 0.0,
            'logical_flow': 0.0,
            'answer_quality': 0.0
        }
        
        # Analyze agent turns
        agent_turns = [turn for turn in dialogue if turn.get('role') == 'agent']
        if not agent_turns:
            return metrics
        
        # Average clarity across turns
        clarity_scores = []
        for turn in agent_turns:
            analysis = ReasoningAnalysis()
            self._analyze_clarity(turn.get('content', ''), analysis)
            clarity_scores.append(analysis.clarity_score)
        metrics['reasoning_clarity'] = np.mean(clarity_scores) if clarity_scores else 0.0
        
        # Check step progression
        all_steps = []
        for turn in agent_turns:
            if 'analysis' in turn.get('metadata', {}):
                steps = turn['metadata']['analysis'].get('completed_steps', [])
                all_steps.extend(steps)
        
        # Estimate completeness
        unique_steps = len(set(all_steps))
        expected_steps = 4  # Typical number of reasoning steps
        metrics['step_completeness'] = min(1.0, unique_steps / expected_steps)
        
        # Logical flow across dialogue
        flow_scores = []
        for i in range(1, len(agent_turns)):
            prev_turn = agent_turns[i-1].get('content', '')
            curr_turn = agent_turns[i].get('content', '')
            
            # Check for logical progression
            if any(conn in curr_turn.lower() for conn in ['therefore', 'so', 'based on']):
                flow_scores.append(1.0)
            elif any(ref in curr_turn.lower() for ref in ['as i mentioned', 'previously', 'earlier']):
                flow_scores.append(0.8)
            else:
                flow_scores.append(0.5)
        
        metrics['logical_flow'] = np.mean(flow_scores) if flow_scores else 0.5
        
        # Answer quality from final turn
        if agent_turns:
            last_turn = agent_turns[-1]
            if 'analysis' in last_turn.get('metadata', {}):
                analysis_data = last_turn['metadata']['analysis']
                if analysis_data.get('is_correct'):
                    metrics['answer_quality'] = 1.0
                elif analysis_data.get('partially_correct'):
                    metrics['answer_quality'] = 0.6
                else:
                    metrics['answer_quality'] = 0.3
        
        return metrics