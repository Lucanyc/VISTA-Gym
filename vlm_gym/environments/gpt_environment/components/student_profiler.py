# vlm_gym/environments/gpt_environment/components/student_profiler.py

"""
Student Profiler for GPT Environment
Tracks and analyzes student (VLM agent) learning patterns and capabilities
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskPerformance:
    """Performance record for a single task"""
    task_id: str
    task_type: str
    success: bool
    turns_taken: int
    time_taken: float
    reasoning_quality: float
    hints_needed: int
    errors_made: List[str]
    timestamp: str
    difficulty: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillAssessment:
    """Assessment of specific skills"""
    visual_observation: float = 0.5
    logical_reasoning: float = 0.5
    calculation_accuracy: float = 0.5
    spatial_reasoning: float = 0.5
    chart_comprehension: float = 0.5
    pattern_recognition: float = 0.5
    attention_to_detail: float = 0.5
    problem_decomposition: float = 0.5
    
    def get_overall_skill(self) -> float:
        """Calculate overall skill level"""
        skills = [
            self.visual_observation,
            self.logical_reasoning,
            self.calculation_accuracy,
            self.spatial_reasoning,
            self.chart_comprehension,
            self.pattern_recognition,
            self.attention_to_detail,
            self.problem_decomposition
        ]
        return np.mean(skills)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LearningPattern:
    """Identified learning patterns"""
    learning_pace: str = "medium"  # slow, medium, fast
    preferred_strategy: str = "balanced"  # systematic, exploratory, balanced
    needs_visual_hints: bool = False
    self_corrects: bool = True
    asks_clarifications: bool = False
    systematic_approach: bool = True
    prone_to_overthinking: bool = False
    benefits_from_encouragement: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StudentProfiler:
    """
    Profiles student (VLM agent) capabilities and learning patterns
    
    Key responsibilities:
    1. Track task performance history
    2. Assess skills across dimensions
    3. Identify learning patterns
    4. Provide personalized recommendations
    5. Support adaptive teaching
    """
    
    def __init__(self, profile_path: Optional[str] = None):
        """
        Initialize student profiler
        
        Args:
            profile_path: Path to save/load profiles
        """
        self.profile_path = Path(profile_path) if profile_path else None
        
        # Student identification
        self.student_id = f"student_{int(time.time())}"
        self.created_at = datetime.now()
        
        # Performance history
        self.task_history: List[TaskPerformance] = []
        self.task_history_by_type: Dict[str, List[TaskPerformance]] = defaultdict(list)
        
        # Skill assessment
        self.skills = SkillAssessment()
        self.skill_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Learning patterns
        self.learning_patterns = LearningPattern()
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.common_errors: List[str] = []
        
        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_time = 0.0
        self.total_turns = 0
        
        # Recent performance window
        self.recent_window = 10
        self.recent_performances: deque = deque(maxlen=self.recent_window)
        
        # Load existing profile if available
        if self.profile_path and self.profile_path.exists():
            self.load_profile()
    
    def update(self, task_id: str, performance: Dict[str, Any], turn_count: int):
        """
        Update profile with task performance
        
        Args:
            task_id: Task identifier
            performance: Performance analysis from ReasoningAnalyzer
            turn_count: Number of turns taken
        """
        # Extract task info
        task_type = self._extract_task_type(task_id)
        
        # Create performance record
        task_perf = TaskPerformance(
            task_id=task_id,
            task_type=task_type,
            success=performance.get('is_correct', False),
            turns_taken=turn_count,
            time_taken=performance.get('time_taken', 0.0),
            reasoning_quality=performance.get('reasoning_quality', 0.0),
            hints_needed=performance.get('hints_requested', 0),
            errors_made=performance.get('error_types', []),
            timestamp=datetime.now().isoformat(),
            difficulty=performance.get('task_difficulty', 0.5)
        )
        
        # Add to history
        self.task_history.append(task_perf)
        self.task_history_by_type[task_type].append(task_perf)
        self.recent_performances.append(task_perf)
        
        # Update statistics
        self.total_tasks += 1
        if task_perf.success:
            self.successful_tasks += 1
        self.total_time += task_perf.time_taken
        self.total_turns += task_perf.turns_taken
        
        # Update skills based on performance
        self._update_skills(task_perf, performance)
        
        # Update error tracking
        self._update_error_tracking(task_perf)
        
        # Update learning patterns
        if self.total_tasks % 5 == 0:  # Update patterns every 5 tasks
            self._analyze_learning_patterns()
        
        # Log progress
        if self.total_tasks % 10 == 0:
            logger.info(f"Student {self.student_id}: {self.total_tasks} tasks, "
                       f"success rate: {self.get_success_rate():.2%}")
    
    def _extract_task_type(self, task_id: str) -> str:
        """Extract task type from task ID"""
        # Try to infer from task_id pattern
        task_types = ['figureqa', 'chartqa', 'clevr', 'geometry3k', 'geoqa', 
                     'iconqa', 'scienceqa', 'mathvista', 'olympiadbench']
        
        task_id_lower = task_id.lower()
        for task_type in task_types:
            if task_type in task_id_lower:
                return task_type
        
        return 'unknown'
    
    def _update_skills(self, task_perf: TaskPerformance, performance: Dict[str, Any]):
        """Update skill assessments based on performance"""
        # Visual observation
        if 'observation' in str(performance.get('completed_steps', [])):
            score = 0.8 if task_perf.success else 0.4
            self._update_skill('visual_observation', score)
        
        # Logical reasoning
        reasoning_score = performance.get('reasoning_quality', 0.5)
        self._update_skill('logical_reasoning', reasoning_score)
        
        # Calculation accuracy (for math-heavy tasks)
        if task_perf.task_type in ['geometry3k', 'mathvista', 'olympiadbench']:
            if 'calculation_error' not in task_perf.errors_made:
                self._update_skill('calculation_accuracy', 0.9 if task_perf.success else 0.6)
            else:
                self._update_skill('calculation_accuracy', 0.3)
        
        # Spatial reasoning (for spatial tasks)
        if task_perf.task_type in ['clevr', 'geometry3k', 'geoqa']:
            spatial_score = 0.8 if task_perf.success else 0.4
            if 'spatial_error' in task_perf.errors_made:
                spatial_score = 0.3
            self._update_skill('spatial_reasoning', spatial_score)
        
        # Chart comprehension
        if task_perf.task_type in ['figureqa', 'chartqa']:
            chart_score = performance.get('clarity_score', 0.5)
            if task_perf.success:
                chart_score = max(chart_score, 0.8)
            self._update_skill('chart_comprehension', chart_score)
        
        # Pattern recognition
        if performance.get('key_insights_count', 0) > 0:
            self._update_skill('pattern_recognition', 0.8)
        
        # Attention to detail
        if performance.get('completeness_score', 0) > 0.8:
            self._update_skill('attention_to_detail', 0.9)
        elif performance.get('unclear_points', []):
            self._update_skill('attention_to_detail', 0.4)
        
        # Problem decomposition
        if len(performance.get('completed_steps', [])) > 3:
            self._update_skill('problem_decomposition', 0.8)
    
    def _update_skill(self, skill_name: str, score: float):
        """Update a specific skill with exponential moving average"""
        # Add to history
        self.skill_history[skill_name].append(score)
        
        # Update skill assessment with weighted average
        current_value = getattr(self.skills, skill_name)
        alpha = 0.2  # Learning rate
        new_value = alpha * score + (1 - alpha) * current_value
        setattr(self.skills, skill_name, new_value)
    
    def _update_error_tracking(self, task_perf: TaskPerformance):
        """Update error tracking"""
        for error in task_perf.errors_made:
            self.error_counts[error] += 1
        
        # Update common errors list
        if self.total_tasks % 10 == 0:
            sorted_errors = sorted(self.error_counts.items(), 
                                 key=lambda x: x[1], reverse=True)
            self.common_errors = [error for error, _ in sorted_errors[:5]]
    
    def _analyze_learning_patterns(self):
        """Analyze recent performance to identify learning patterns"""
        if len(self.recent_performances) < 5:
            return
        
        recent = list(self.recent_performances)
        
        # Learning pace
        recent_success_rate = sum(1 for p in recent if p.success) / len(recent)
        avg_turns = np.mean([p.turns_taken for p in recent])
        
        if recent_success_rate > 0.8 and avg_turns < 5:
            self.learning_patterns.learning_pace = "fast"
        elif recent_success_rate < 0.4 or avg_turns > 10:
            self.learning_patterns.learning_pace = "slow"
        else:
            self.learning_patterns.learning_pace = "medium"
        
        # Preferred strategy
        reasoning_qualities = [p.reasoning_quality for p in recent]
        if np.mean(reasoning_qualities) > 0.7:
            self.learning_patterns.systematic_approach = True
            self.learning_patterns.preferred_strategy = "systematic"
        elif np.std(reasoning_qualities) > 0.3:
            self.learning_patterns.preferred_strategy = "exploratory"
        
        # Other patterns
        hints_needed = sum(p.hints_needed for p in recent)
        self.learning_patterns.needs_visual_hints = hints_needed > len(recent) * 0.5
        
        # Self-correction pattern
        error_patterns = [p.errors_made for p in recent]
        self.learning_patterns.self_corrects = any('self_correction' in errors 
                                                  for errors in error_patterns)
        
        # Overthinking pattern
        avg_recent_turns = np.mean([p.turns_taken for p in recent])
        self.learning_patterns.prone_to_overthinking = avg_recent_turns > 8
    
    def get_profile(self) -> Dict[str, Any]:
        """Get current student profile"""
        return {
            'student_id': self.student_id,
            'skill_level': self.skills.get_overall_skill(),
            'total_tasks': self.total_tasks,
            'success_rate': self.get_success_rate(),
            'skills': self.skills.to_dict(),
            'learning_patterns': self.learning_patterns.to_dict(),
            'common_errors': self.common_errors[:3],
            'recent_performance': self._get_recent_performance_summary()
        }
    
    def get_skill_level(self) -> float:
        """Get overall skill level (0-1)"""
        return self.skills.get_overall_skill()
    
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task history for task selection"""
        return [
            {
                'task_id': perf.task_id,
                'task_type': perf.task_type,
                'timestamp': perf.timestamp,
                'success': perf.success
            }
            for perf in self.task_history[-20:]  # Last 20 tasks
        ]
    
    def get_weak_areas(self) -> List[Tuple[str, float]]:
        """Identify weak skill areas"""
        skills_dict = self.skills.to_dict()
        weak_areas = [(skill, score) for skill, score in skills_dict.items() 
                     if score < 0.5]
        return sorted(weak_areas, key=lambda x: x[1])
    
    def get_strong_areas(self) -> List[Tuple[str, float]]:
        """Identify strong skill areas"""
        skills_dict = self.skills.to_dict()
        strong_areas = [(skill, score) for skill, score in skills_dict.items() 
                       if score > 0.7]
        return sorted(strong_areas, key=lambda x: x[1], reverse=True)
    
    def get_recommended_task_types(self) -> List[str]:
        """Recommend task types based on profile"""
        recommendations = []
        
        # Focus on weak areas
        if self.skills.chart_comprehension < 0.5:
            recommendations.extend(['figureqa', 'chartqa'])
        if self.skills.spatial_reasoning < 0.5:
            recommendations.extend(['clevr', 'geometry3k'])
        if self.skills.calculation_accuracy < 0.5:
            recommendations.extend(['mathvista', 'scienceqa'])
        
        # Avoid overwhelming with difficult tasks
        if self.learning_patterns.learning_pace == "slow":
            recommendations = [t for t in recommendations 
                             if t not in ['olympiadbench', 'geometry3k']]
        
        # Default recommendations if none specific
        if not recommendations:
            recommendations = ['figureqa', 'chartqa', 'clevr']
        
        return recommendations
    
    def get_teaching_recommendations(self) -> Dict[str, Any]:
        """Get personalized teaching recommendations"""
        return {
            'strategy': self._recommend_strategy(),
            'difficulty_adjustment': self._recommend_difficulty(),
            'focus_areas': self._recommend_focus_areas(),
            'teaching_pace': self._recommend_pace(),
            'encouragement_level': self._recommend_encouragement()
        }
    
    def _recommend_strategy(self) -> str:
        """Recommend teaching strategy"""
        if self.learning_patterns.systematic_approach:
            return "scaffolding"
        elif self.learning_patterns.preferred_strategy == "exploratory":
            return "guided_discovery"
        elif self.skills.get_overall_skill() < 0.4:
            return "direct"
        else:
            return "socratic"
    
    def _recommend_difficulty(self) -> float:
        """Recommend difficulty adjustment"""
        skill_level = self.skills.get_overall_skill()
        recent_success = self._get_recent_success_rate()
        
        if recent_success > 0.8 and skill_level > 0.6:
            return 0.1  # Increase difficulty
        elif recent_success < 0.3 or skill_level < 0.4:
            return -0.1  # Decrease difficulty
        else:
            return 0.0  # Maintain current difficulty
    
    def _recommend_focus_areas(self) -> List[str]:
        """Recommend areas to focus on"""
        focus_areas = []
        
        # Based on weak skills
        weak_areas = self.get_weak_areas()
        for skill, score in weak_areas[:2]:  # Top 2 weak areas
            if skill == "visual_observation":
                focus_areas.append("Careful image examination")
            elif skill == "logical_reasoning":
                focus_areas.append("Step-by-step reasoning")
            elif skill == "calculation_accuracy":
                focus_areas.append("Double-check calculations")
            elif skill == "spatial_reasoning":
                focus_areas.append("Spatial relationship analysis")
        
        # Based on common errors
        if "calculation_error" in self.common_errors:
            focus_areas.append("Arithmetic verification")
        if "reading_error" in self.common_errors:
            focus_areas.append("Careful data reading")
        
        return focus_areas
    
    def _recommend_pace(self) -> str:
        """Recommend teaching pace"""
        if self.learning_patterns.learning_pace == "fast":
            return "accelerated"
        elif self.learning_patterns.learning_pace == "slow":
            return "patient"
        else:
            return "moderate"
    
    def _recommend_encouragement(self) -> str:
        """Recommend encouragement level"""
        recent_success = self._get_recent_success_rate()
        
        if recent_success < 0.3:
            return "high"
        elif self.learning_patterns.benefits_from_encouragement:
            return "moderate"
        else:
            return "minimal"
    
    def _get_recent_performance_summary(self) -> Dict[str, Any]:
        """Summarize recent performance"""
        if not self.recent_performances:
            return {
                'success_rate': 0.0,
                'avg_turns': 0,
                'avg_quality': 0.0
            }
        
        recent = list(self.recent_performances)
        return {
            'success_rate': sum(1 for p in recent if p.success) / len(recent),
            'avg_turns': np.mean([p.turns_taken for p in recent]),
            'avg_quality': np.mean([p.reasoning_quality for p in recent])
        }
    
    def _get_recent_success_rate(self) -> float:
        """Get success rate for recent tasks"""
        if not self.recent_performances:
            return 0.0
        recent = list(self.recent_performances)
        return sum(1 for p in recent if p.success) / len(recent)
    
    def record_episode(self, episode_summary: Dict[str, Any]):
        """Record episode results (called after task completion)"""
        # This is called from GPTEnvironment's _finalize_episode
        # Extract relevant information for profiling
        performance = {
            'is_correct': episode_summary.get('success', False),
            'reasoning_quality': episode_summary.get('reasoning_quality', 0.0),
            'time_taken': episode_summary.get('duration', 0.0),
            'task_difficulty': episode_summary.get('task_difficulty', 0.5),
            'hints_requested': episode_summary.get('hint_requests', 0),
            'error_types': []  # Would need to extract from dialogue
        }
        
        self.update(
            task_id=episode_summary['task_id'],
            performance=performance,
            turn_count=episode_summary['total_turns']
        )
    
    def export_state(self) -> Dict[str, Any]:
        """Export profile state for saving"""
        return {
            'student_id': self.student_id,
            'created_at': self.created_at.isoformat(),
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'total_time': self.total_time,
            'total_turns': self.total_turns,
            'skills': self.skills.to_dict(),
            'learning_patterns': self.learning_patterns.to_dict(),
            'error_counts': dict(self.error_counts),
            'common_errors': self.common_errors,
            'task_history': [perf.to_dict() for perf in self.task_history[-100:]],  # Keep last 100
            'recent_performances': [perf.to_dict() for perf in self.recent_performances]
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import profile state from saved data"""
        self.student_id = state.get('student_id', self.student_id)
        self.created_at = datetime.fromisoformat(state.get('created_at', 
                                                          datetime.now().isoformat()))
        
        self.total_tasks = state.get('total_tasks', 0)
        self.successful_tasks = state.get('successful_tasks', 0)
        self.total_time = state.get('total_time', 0.0)
        self.total_turns = state.get('total_turns', 0)
        
        # Import skills
        skills_data = state.get('skills', {})
        for skill, value in skills_data.items():
            if hasattr(self.skills, skill):
                setattr(self.skills, skill, value)
        
        # Import learning patterns
        patterns_data = state.get('learning_patterns', {})
        for pattern, value in patterns_data.items():
            if hasattr(self.learning_patterns, pattern):
                setattr(self.learning_patterns, pattern, value)
        
        # Import error tracking
        self.error_counts = defaultdict(int, state.get('error_counts', {}))
        self.common_errors = state.get('common_errors', [])
        
        # Import task history
        self.task_history = []
        for task_data in state.get('task_history', []):
            self.task_history.append(TaskPerformance(**task_data))
        
        # Rebuild task history by type
        self.task_history_by_type.clear()
        for perf in self.task_history:
            self.task_history_by_type[perf.task_type].append(perf)
        
        # Import recent performances
        self.recent_performances.clear()
        for perf_data in state.get('recent_performances', []):
            self.recent_performances.append(TaskPerformance(**perf_data))
    
    def save_profile(self, path: Optional[str] = None):
        """Save profile to file"""
        save_path = Path(path) if path else self.profile_path
        if not save_path:
            logger.warning("No save path specified for profile")
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.export_state(), f, indent=2)
        
        logger.info(f"Saved student profile to {save_path}")
    
    def load_profile(self, path: Optional[str] = None):
        """Load profile from file"""
        load_path = Path(path) if path else self.profile_path
        if not load_path or not load_path.exists():
            logger.warning(f"Profile file not found: {load_path}")
            return
        
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            self.import_state(state)
            logger.info(f"Loaded student profile from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
    
    def reset(self):
        """Reset profile to initial state"""
        self.__init__(profile_path=str(self.profile_path) if self.profile_path else None)