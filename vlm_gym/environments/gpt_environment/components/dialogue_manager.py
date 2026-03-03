# vlm_gym/environments/gpt_environment/components/dialogue_manager.py

"""
Dialogue Manager for GPT Environment
Manages conversation history and context for reasoning path collection
"""

import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """Single turn in the dialogue"""
    turn_id: int
    role: str  # 'environment', 'agent', 'gpt_teacher'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    # Analysis results
    reasoning_type: Optional[str] = None  # 'observation', 'analysis', 'inference', 'conclusion'
    confidence: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'turn_id': self.turn_id,
            'role': self.role,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'reasoning_type': self.reasoning_type,
            'confidence': self.confidence,
            'errors': self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTurn':
        """Create DialogueTurn from dictionary"""
        return cls(
            turn_id=data['turn_id'],
            role=data['role'],
            content=data['content'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', time.time()),
            reasoning_type=data.get('reasoning_type'),
            confidence=data.get('confidence'),
            errors=data.get('errors', [])
        )


class DialogueManager:
    """
    Manages multi-turn dialogue and extracts reasoning chains
    
    Key responsibilities:
    1. Track conversation history
    2. Maintain context window
    3. Extract reasoning steps
    4. Identify key decision points
    5. Format for training data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dialogue manager
        
        Args:
            config: Configuration dictionary with:
                - context_window: Number of recent turns for context
                - max_turns: Maximum turns allowed per task
                - extract_reasoning: Whether to extract reasoning automatically
        """
        config = config or {}
        self.context_window = config.get('context_window', 10)
        self.max_turns = config.get('max_turns', 50)
        self.extract_reasoning = config.get('extract_reasoning', True)
        
        # Conversation history
        self.conversation_history: List[DialogueTurn] = []
        self.turn_count = 0
        
        # Task metadata
        self.task_id: Optional[str] = None
        self.task_type: Optional[str] = None
        self.task_start_time: Optional[float] = None
        
        # Reasoning chain tracking
        self.reasoning_steps: List[Dict[str, Any]] = []
        self.key_insights: List[str] = []
        self.decision_points: List[int] = []  # Turn IDs where key decisions were made
        
        # Quality indicators
        self.clarification_requests = 0
        self.hint_requests = 0
        self.backtracking_count = 0
        self.corrections_made = 0
        
    def reset(self, task_id: Optional[str] = None, task_type: Optional[str] = None):
        """Reset dialogue manager for new task"""
        self.clear()
        if task_id:
            self.start_new_task(task_id, task_type or 'unknown')
        
    def start_new_task(self, task_id: str, task_type: str):
        """Start tracking a new task"""
        self.clear()
        self.task_id = task_id
        self.task_type = task_type
        self.task_start_time = time.time()
        logger.info(f"Started new task: {task_id} (type: {task_type})")
        
    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> DialogueTurn:
        """
        Add a new turn to the dialogue
        
        Args:
            role: Speaker role ('environment', 'agent', 'gpt_teacher')
            content: The message content
            metadata: Additional metadata about the turn
            
        Returns:
            The created DialogueTurn object
        """
        if self.turn_count >= self.max_turns:
            logger.warning(f"Reached maximum turns ({self.max_turns}) for task {self.task_id}")
            
        turn = DialogueTurn(
            turn_id=self.turn_count,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # Analyze the turn for reasoning patterns if enabled
        if self.extract_reasoning:
            if role == 'agent':
                self._analyze_agent_turn(turn)
            elif role in ['gpt_teacher', 'environment']:
                self._analyze_teacher_turn(turn)
            
        self.conversation_history.append(turn)
        self.turn_count += 1
        
        # Check for special events
        self._check_special_events(turn)
        
        return turn
        
    def get_context(self, window_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation context formatted for use
        
        Args:
            window_size: Number of recent turns to return
            
        Returns:
            List of dialogue turns as dictionaries
        """
        size = window_size or self.context_window
        recent_turns = self.conversation_history[-size:] if self.conversation_history else []
        
        return [
            {
                'role': turn.role,
                'content': turn.content,
                'turn_id': turn.turn_id
            }
            for turn in recent_turns
        ]
        
    def get_recent_context(self, n_turns: int) -> List[Dict[str, str]]:
        """Get last n turns of context"""
        return self.get_context(window_size=n_turns)
        
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Get complete conversation history as dictionaries"""
        return [turn.to_dict() for turn in self.conversation_history]
        
    def get_formatted_context(self) -> str:
        """Get context formatted as a single string for display"""
        context_turns = self.get_context()
        formatted_lines = []
        
        for turn in context_turns:
            role_display = {
                'environment': 'Teacher',
                'gpt_teacher': 'Teacher', 
                'agent': 'Student'
            }.get(turn['role'], turn['role'])
            
            formatted_lines.append(f"{role_display}: {turn['content']}")
            
        return "\n\n".join(formatted_lines)
        
    def get_last_environment_turn(self) -> Optional[str]:
        """Get the last turn from environment/teacher"""
        for turn in reversed(self.conversation_history):
            if turn.role in ['environment', 'gpt_teacher']:
                return turn.content
        return None
        
    def get_last_agent_turn(self) -> Optional[str]:
        """Get the last turn from agent"""
        for turn in reversed(self.conversation_history):
            if turn.role == 'agent':
                return turn.content
        return None
        
    def extract_reasoning_chain(self) -> List[Dict[str, Any]]:
        """
        Extract the reasoning chain from the dialogue
        
        Returns:
            List of reasoning steps with metadata
        """
        reasoning_chain = []
        
        for turn in self.conversation_history:
            if turn.role == 'agent' and turn.reasoning_type:
                step = {
                    'turn_id': turn.turn_id,
                    'type': turn.reasoning_type,
                    'content': turn.content,
                    'confidence': turn.confidence,
                    'errors': turn.errors,
                    'timestamp': turn.timestamp
                }
                
                # Extract key information based on reasoning type
                if turn.reasoning_type == 'observation':
                    step['observations'] = self._extract_observations(turn.content)
                elif turn.reasoning_type == 'analysis':
                    step['analysis_points'] = self._extract_analysis_points(turn.content)
                elif turn.reasoning_type == 'inference':
                    step['inferences'] = self._extract_inferences(turn.content)
                elif turn.reasoning_type == 'conclusion':
                    step['conclusion'] = self._extract_conclusion(turn.content)
                    
                reasoning_chain.append(step)
                
        return reasoning_chain
        
    def _analyze_agent_turn(self, turn: DialogueTurn):
        """Analyze agent's turn for reasoning patterns"""
        content_lower = turn.content.lower()
        
        # Enhanced reasoning type identification
        reasoning_patterns = {
            'observation': ['see', 'observe', 'notice', 'look', 'appears', 'shows', 'displays', 'contains'],
            'analysis': ['analyze', 'compare', 'examine', 'consider', 'evaluate', 'assess', 'study'],
            'inference': ['because', 'since', 'therefore', 'thus', 'hence', 'implies', 'suggests', 'means'],
            'conclusion': ['conclude', 'answer', 'final', 'result', 'solution', 'therefore the answer', 'so the answer']
        }
        
        # Score each type based on keyword presence
        type_scores = {}
        for rtype, keywords in reasoning_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                type_scores[rtype] = score
                
        # Assign the type with highest score
        if type_scores:
            turn.reasoning_type = max(type_scores, key=type_scores.get)
            if turn.reasoning_type == 'conclusion':
                self.decision_points.append(turn.turn_id)
        
        # Enhanced confidence estimation
        confidence_indicators = {
            0.9: ['definitely', 'clearly', 'obviously', 'certain', 'must be', 'has to be'],
            0.7: ['probably', 'likely', 'seems', 'appears', 'should be', 'would be'],
            0.5: ['maybe', 'possibly', 'might', 'could', 'perhaps', 'potentially'],
            0.3: ['unsure', 'confused', "don't know", 'not sure', 'uncertain', 'guess']
        }
        
        turn.confidence = 0.6  # Default confidence
        for conf_level, indicators in confidence_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                turn.confidence = conf_level
                break
                
        # Check for errors or confusion
        error_indicators = ['wrong', 'mistake', 'incorrect', 'error', 'oops', 'sorry']
        if any(indicator in content_lower for indicator in error_indicators):
            turn.errors.append('self_identified_error')
            self.corrections_made += 1
            
        # Check for questions
        if '?' in turn.content:
            self.clarification_requests += 1
            turn.metadata['contains_question'] = True
            
    def _analyze_teacher_turn(self, turn: DialogueTurn):
        """Analyze teacher's turn for teaching patterns"""
        content_lower = turn.content.lower()
        
        # Check for hints
        hint_indicators = ['hint', 'clue', 'consider', 'think about', 'try to', 'what if', 'remember']
        if any(indicator in content_lower for indicator in hint_indicators):
            self.hint_requests += 1
            turn.metadata['contains_hint'] = True
            
        # Check for corrections
        correction_indicators = ['actually', 'incorrect', 'not quite', 'try again', 'reconsider', 'but']
        if any(indicator in content_lower for indicator in correction_indicators):
            turn.metadata['contains_correction'] = True
            self.backtracking_count += 1
            
        # Check for encouragement
        encouragement_indicators = ['good', 'excellent', 'great', 'well done', 'correct', 'right']
        if any(indicator in content_lower for indicator in encouragement_indicators):
            turn.metadata['contains_encouragement'] = True
            
        # Identify teaching action type
        if 'action_type' in turn.metadata:
            turn.metadata['teaching_action'] = turn.metadata['action_type']
            
    def _check_special_events(self, turn: DialogueTurn):
        """Check for special events in the dialogue"""
        # Track key insights
        if turn.role == 'agent' and turn.confidence and turn.confidence > 0.8:
            insight_patterns = [
                r"I realize[d]? that",
                r"The key is",
                r"What matters is",
                r"The important thing is",
                r"I understand now",
                r"This means that"
            ]
            for pattern in insight_patterns:
                if re.search(pattern, turn.content, re.IGNORECASE):
                    self.key_insights.append(turn.content)
                    turn.metadata['is_key_insight'] = True
                    break
                    
    def _extract_observations(self, content: str) -> List[str]:
        """Extract observation statements"""
        observations = []
        
        # Enhanced pattern matching for observations
        patterns = [
            r"I (?:see|observe|notice|can see) (.+?)(?:\.|,|;|$)",
            r"There (?:is|are) (.+?)(?:\.|,|;|$)",
            r"The (.+?) (?:is|are|has|have|shows|contains) (.+?)(?:\.|,|;|$)",
            r"(?:Looking at|Examining|In) the (?:image|chart|diagram|figure), (.+?)(?:\.|,|;|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    observations.append(' '.join(match).strip())
                else:
                    observations.append(match.strip())
                    
        return observations
        
    def _extract_analysis_points(self, content: str) -> List[str]:
        """Extract analysis points"""
        points = []
        
        # Split by common analysis markers
        markers = ['First', 'Second', 'Third', 'Next', 'Then', 'Additionally', 
                   'Moreover', 'Furthermore', 'Also', 'Another']
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Check if sentence starts with a marker
            if any(sentence.startswith(marker) for marker in markers):
                points.append(sentence)
            # Check for analysis keywords
            elif any(word in sentence.lower() for word in ['compare', 'analyze', 'examine', 'consider']):
                points.append(sentence)
                
        return points if points else [content.strip()]  # Return full content if no points found
        
    def _extract_inferences(self, content: str) -> List[str]:
        """Extract inference statements"""
        inferences = []
        
        # Enhanced pattern matching for inferences
        patterns = [
            r"(?:Therefore|Thus|So|Hence|Consequently),? (.+?)(?:\.|$)",
            r"This (?:means|suggests|indicates|implies|shows) that (.+?)(?:\.|$)",
            r"(?:Because|Since|As) (.+?), (?:we can conclude that |it follows that )?(.+?)(?:\.|$)",
            r"Based on (?:this|these), (.+?)(?:\.|$)",
            r"From this,? (?:we can|I can) (?:see|infer|deduce) that (.+?)(?:\.|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    inferences.append(' '.join(match).strip())
                else:
                    inferences.append(match.strip())
                    
        return inferences
        
    def _extract_conclusion(self, content: str) -> str:
        """Extract the final conclusion"""
        # Look for explicit conclusion markers
        patterns = [
            r"(?:The answer is|My answer is|Therefore,? the answer is) (.+?)(?:\.|$)",
            r"(?:I conclude that|In conclusion,?) (.+?)(?:\.|$)",
            r"(?:Finally,|Ultimately,?) (.+?)(?:\.|$)",
            r"So,? (.+?) is the (?:answer|solution)(?:\.|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        # If no explicit marker, look for answer patterns
        # For yes/no questions
        if re.search(r'\b(?:yes|no)\b', content.lower()):
            return re.search(r'\b(yes|no)\b', content.lower()).group(1)
            
        # For numeric answers
        numeric_match = re.search(r'(?:is |= )(\d+(?:\.\d+)?)', content)
        if numeric_match:
            return numeric_match.group(1)
            
        # Default: take the last sentence
        sentences = re.split(r'[.!?]+', content)
        return sentences[-1].strip() if sentences else content
        
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dialogue for analysis
        
        Returns:
            Dictionary containing dialogue statistics and metadata
        """
        agent_turns = [t for t in self.conversation_history if t.role == 'agent']
        teacher_turns = [t for t in self.conversation_history if t.role in ['gpt_teacher', 'environment']]
        
        # Calculate average confidence
        confidences = [t.confidence for t in agent_turns if t.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'total_turns': self.turn_count,
            'agent_turns': len(agent_turns),
            'teacher_turns': len(teacher_turns),
            'clarification_requests': self.clarification_requests,
            'hint_requests': self.hint_requests,
            'backtracking_count': self.backtracking_count,
            'corrections_made': self.corrections_made,
            'key_insights_count': len(self.key_insights),
            'decision_points': self.decision_points,
            'duration': time.time() - self.task_start_time if self.task_start_time else 0,
            'reasoning_chain_length': len(self.extract_reasoning_chain()),
            'average_confidence': avg_confidence
        }
        
    def format_for_training(self) -> Dict[str, Any]:
        """
        Format the dialogue for training data
        
        Returns:
            Dictionary formatted for model training
        """
        summary = self.get_dialogue_summary()
        reasoning_chain = self.extract_reasoning_chain()
        
        return {
            'dialogue_id': f"{self.task_id}_{int(self.task_start_time or 0)}",
            'task_id': self.task_id,
            'task_type': self.task_type,
            'dialogue': [turn.to_dict() for turn in self.conversation_history],
            'reasoning_chain': reasoning_chain,
            'key_insights': self.key_insights,
            'summary': summary,
            'quality_indicators': {
                'has_clear_reasoning': len(reasoning_chain) >= 3,
                'has_conclusion': any(t.reasoning_type == 'conclusion' 
                                    for t in self.conversation_history),
                'required_hints': self.hint_requests > 0,
                'self_corrected': self.corrections_made > 0,
                'average_confidence': summary['average_confidence'],
                'reasoning_depth': len(reasoning_chain) / max(1, summary['agent_turns'])
            }
        }
        
    def export_state(self) -> Dict[str, Any]:
        """Export current state for checkpointing"""
        return {
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'turn_count': self.turn_count,
            'task_id': self.task_id,
            'task_type': self.task_type,
            'task_start_time': self.task_start_time,
            'reasoning_steps': self.reasoning_steps,
            'key_insights': self.key_insights,
            'decision_points': self.decision_points,
            'clarification_requests': self.clarification_requests,
            'hint_requests': self.hint_requests,
            'backtracking_count': self.backtracking_count,
            'corrections_made': self.corrections_made
        }
        
    def import_state(self, state: Dict[str, Any]):
        """Import state from checkpoint"""
        self.conversation_history = [
            DialogueTurn.from_dict(turn_dict) 
            for turn_dict in state.get('conversation_history', [])
        ]
        self.turn_count = state.get('turn_count', 0)
        self.task_id = state.get('task_id')
        self.task_type = state.get('task_type')
        self.task_start_time = state.get('task_start_time')
        self.reasoning_steps = state.get('reasoning_steps', [])
        self.key_insights = state.get('key_insights', [])
        self.decision_points = state.get('decision_points', [])
        self.clarification_requests = state.get('clarification_requests', 0)
        self.hint_requests = state.get('hint_requests', 0)
        self.backtracking_count = state.get('backtracking_count', 0)
        self.corrections_made = state.get('corrections_made', 0)
        
    def clear(self):
        """Clear the dialogue history"""
        self.conversation_history.clear()
        self.turn_count = 0
        self.reasoning_steps.clear()
        self.key_insights.clear()
        self.decision_points.clear()
        self.clarification_requests = 0
        self.hint_requests = 0
        self.backtracking_count = 0
        self.corrections_made = 0
        self.task_id = None
        self.task_type = None
        self.task_start_time = None
        
    def __len__(self) -> int:
        """Return the number of turns"""
        return self.turn_count
        
    def __repr__(self) -> str:
        """String representation"""
        return f"DialogueManager(task={self.task_id}, turns={self.turn_count})"