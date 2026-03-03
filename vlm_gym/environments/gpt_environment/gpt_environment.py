# vlm_gym/environments/gpt_environment/gpt_environment.py

import time
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path
import gymnasium as gym
from ..base import BaseVLMEnv
from .components.gpt_teacher import GPTTeacher
from .components.dialogue_manager import DialogueManager
from .components.reasoning_analyzer import ReasoningAnalyzer
from .components.student_profiler import StudentProfiler
#from .components.task_selector import TaskSelector
from .components.strategy_selector import StrategySelector
from .collectors.reasoning_path_collector import ReasoningPathCollector
from .exceptions import GPTAPIError, TaskLoadError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Task information structure"""
    id: str
    type: str
    difficulty: float
    question: str
    answer: Any
    image_path: str
    metadata: Dict[str, Any]
    adapter_name: Optional[str] = None


class TaskAdapterInterface:
    """Unified task adapter interface"""
    def get_task_by_id(self, task_id: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_tasks_by_difficulty(self, difficulty: float, tolerance: float = 0.1) -> List[str]:
        raise NotImplementedError
    
    def get_task_types(self) -> List[str]:
        raise NotImplementedError
    
    def estimate_difficulty(self, task: Any) -> float:
        raise NotImplementedError


class GPTEnvironment(gym.Env):
    """
    GPT-driven intelligent teaching environment for VLM training.
    
    This environment uses GPT as a teacher to guide VLM agents through
    reasoning tasks, collecting high-quality reasoning paths in the process.
    """
    
    DEFAULT_CONFIG = {
        'max_turns_per_task': 10,
        'min_turns_per_task': 3,
        'checkpoint_dir': './checkpoints',
        'enable_checkpointing': True,
        'checkpoint_interval': 100,
        'reward_config': {
            'correct_answer': 1.0,
            'good_reasoning': 0.5,
            'partial_progress': 0.2,
            'incorrect': -0.1,
            'invalid_response': -0.2
        },
        'quality_threshold': 0.7,
        'fallback_strategy': 'direct_teaching',
        'max_retries': 3,
        'timeout': 30.0
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPT Environment with validation.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Validate and merge config
        self.config = self._validate_and_merge_config(config)
        
        # Initialize logging
        self._setup_logging()
        
        # Core components initialization
        try:
            self.gpt_teacher = GPTTeacher(self.config['gpt_config'])
            self.dialogue_manager = DialogueManager(self.config.get('dialogue_config', {}))
            self.reasoning_analyzer = ReasoningAnalyzer(self.config.get('analyzer_config', {}))
            self.student_profiler = StudentProfiler()
            #self.task_selector = TaskSelector(self.config['task_config'])
            self.strategy_selector = StrategySelector()
            self.reasoning_collector = ReasoningPathCollector(self.config['collection_config'])
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}")
        
        # Task adapters
        self.task_adapters = self._initialize_task_adapters()
        
        # Task management
        self.current_task: Optional[TaskInfo] = None
        self.current_strategy = None
        self.current_image: Optional[np.ndarray] = None
        self.task_start_time = None
        
        # Environment state
        self.turn_count = 0
        self.episode_reward = 0.0
        self.done = False
        self.info = {}
        self.episode_count = 0
        
        # Performance tracking
        self.current_performance = {
            'reasoning_quality': 0.0,
            'answer_correctness': 0.0,
            'progress_rate': 0.0
        }
        
        # Checkpointing
        if self.config['enable_checkpointing']:
            self._setup_checkpointing()
    
    def _validate_and_merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and merge with defaults"""
        # Check required configs
        required_configs = ['gpt_config', 'task_config', 'collection_config']
        for key in required_configs:
            if key not in config:
                raise ConfigurationError(f"Missing required config: {key}")
        
        # Validate GPT config
        if 'model' not in config['gpt_config']:
            raise ConfigurationError("GPT config must specify model")
        
        # Merge with defaults
        merged_config = self.DEFAULT_CONFIG.copy()
        merged_config.update(config)
        
        return merged_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_task_adapters(self) -> Dict[str, TaskAdapterInterface]:
        """Initialize task adapters for different task types"""
        adapters = {}
        
        task_config = self.config['task_config']
        for task_type in task_config.get('task_types', ['figureqa']):
            try:
                if task_type == 'figureqa':
                    # 修改导入路径
                    import sys
                    sys.path.append('/workspace/mathvista')
                    from data_adapters.figureqa_adapter import FigureQAAdapter
                    adapters[task_type] = FigureQAAdapter(
                        data_root=task_config.get('figureqa', {}).get('data_path', ''),
                        annotation_files=task_config.get('figureqa', {}).get('annotation_file', [])
                    )
                logger.info(f"Initialized adapter for {task_type}")
            except ImportError as e:
                logger.warning(f"Could not import adapter for {task_type}: {e}")
        
        if not adapters:
            raise ConfigurationError("No task adapters could be initialized")
        
        return adapters
    
    def _setup_checkpointing(self):
        """Setup checkpointing directory and state"""
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_episode = 0
    
    #def reset(self, task=None) -> Dict[str, Any]:
        """
        Reset environment and initialize a new task.
        
        Args:
            task: 外部传入的任务对象（可选）
            
        Returns:
            Initial observation containing task and initial prompt
        """
        #try:
            # Reset state
        #    self.turn_count = 0
        #    self.episode_reward = 0.0
        #    self.done = False
        #    self.info = {}
        #    self.task_start_time = time.time()
            
            # Clear dialogue history
        #    self.dialogue_manager.reset()
            
            # 1. Get student profile
        #    student_profile = self.student_profiler.get_profile()
            
            # 2. 使用外部传入的任务或已设置的任务
        #    if task is not None:
                # 将外部任务转换为 TaskInfo
        #        self.current_task = TaskInfo(
        #            id=task.task_id,
        #            type=task.type,
        #            difficulty=task._estimate_difficulty(),  # 直接调用方法，不检查
        #            question=task.question,
        #            answer=task.answer,
        #            image_path=task.image_path,
        #            metadata=task.metadata if hasattr(task, 'metadata') else {}
       #         )
       #     elif self.current_task is None:
       #         raise ValueError("No task provided and no current task set")
            
            # 3. Load task image
        #    self.current_image = self._load_task_image(self.current_task.image_path)
            
            # 4. Select teaching strategy
        #    self.current_strategy = self.strategy_selector.select_strategy(
        #        task_type=self.current_task.type,
        #        student_profile=student_profile,
        #        task_difficulty=self.current_task.difficulty
        #    )
            
            # 5. Generate initial prompt using GPT teacher
        #    initial_prompt = self._generate_initial_prompt_with_retry(
        #        task=self.current_task,
        #        strategy=self.current_strategy,
        #        student_level=student_profile.get('skill_level', 'intermediate')
        #    )
            
            # 6. Record initial turn
        #    self.dialogue_manager.add_turn(
        #        role='environment',
        #        content=initial_prompt,
        #        metadata={
        #            'task_id': self.current_task.id,
        #            'strategy': self.current_strategy.name,
        #            'turn_type': 'initial_prompt'
        #        }
        #    )
            
            # Return observation
        #    return self._create_task_specific_observation()
            
        #except Exception as e:
        #    logger.error(f"Error in reset: {e}")
        #    return self._create_error_observation(str(e))
    
    
    def reset(self, task=None) -> Dict[str, Any]:
        """Reset environment and initialize a new task."""
        try:
            # Reset state
            self.turn_count = 0
            self.episode_reward = 0.0
            self.done = False
            self.info = {}
            self.task_start_time = time.time()
            
            # Clear dialogue history
            self.dialogue_manager.reset()
            
            # 检查是否有当前任务
            if self.current_task is None:
                logger.error("No task assigned to environment")
                return self._create_error_observation("No task assigned")
            
            # 获取学生档案
            student_profile = self.student_profiler.get_profile()
            
            # 选择教学策略
            self.current_strategy = self.strategy_selector.select_strategy(
                task_type='figureqa',  # 暂时硬编码
                student_profile=student_profile,
                task_difficulty=0.5  # 默认难度
            )
            
            # 创建初始提示
            initial_prompt = f"Let's solve this problem: {self.current_task.question if hasattr(self.current_task, 'question') else 'Unknown question'}"
            
            # 记录初始回合
            self.dialogue_manager.add_turn(
                role='environment',
                content=initial_prompt,
                metadata={
                    'task_id': getattr(self.current_task, 'task_id', 'unknown'),
                    'strategy': self.current_strategy.name,
                    'turn_type': 'initial_prompt'
                }
            )
            
            # 返回观察，确保包含正确的 image_path
            return {
                'image_path': getattr(self.current_task, 'image_path', None),
                'task_type': 'figureqa',
                'dialogue': self.dialogue_manager.get_formatted_context(),
                'current_prompt': initial_prompt,
                'turn_count': self.turn_count
            }
            
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            return self._create_error_observation(str(e))
    
    
    
    
    def step(self, agent_action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process agent's response and generate next teaching action.
        
        Args:
            agent_action: The agent's response/answer
            
        Returns:
            observation: Next state observation
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information
        """
        try:
            # Validate input
            if not agent_action or not isinstance(agent_action, str):
                return self._handle_invalid_action()
            
            self.turn_count += 1
            
            # 1. Record agent's response
            self.dialogue_manager.add_turn(
                role='agent',
                content=agent_action,
                metadata={
                    'turn_number': self.turn_count,
                    'timestamp': time.time()
                }
            )
            
            # 2. Analyze agent's response
            analysis = self._analyze_response_with_retry(agent_action)
            
            # 3. Update student profile
            self.student_profiler.update(
                task_id=self.current_task.id,
                performance=analysis,
                turn_count=self.turn_count
            )
            
            # 4. Calculate immediate reward
            reward = self._calculate_reward(analysis)
            self.episode_reward += reward
            
            # 5. Decide next action
            next_action_type = self._decide_next_action(analysis)
            
            # 6. Generate appropriate response
            teacher_response = self._generate_teacher_response(next_action_type, analysis)
            
            # 7. Record teacher's response
            self.dialogue_manager.add_turn(
                role='environment',
                content=teacher_response,
                metadata={
                    'action_type': next_action_type,
                    'analysis': analysis
                }
            )
            
            # 8. Check termination conditions
            if not self.done:
                self.done = self._check_termination_conditions(analysis)
            
            # 9. Collect reasoning path if episode is done
            if self.done:
                self._finalize_episode(analysis)
                self.episode_count += 1
                
                # Checkpoint if needed
                if self._should_checkpoint():
                    self.save_checkpoint()
            
            # 10. Prepare info with metrics
            self.info = self._prepare_info_dict(analysis, next_action_type)
            
            return self._create_task_specific_observation(), reward, self.done, self.info
            
        except GPTAPIError as e:
            logger.error(f"GPT API error: {e}")
            return self._create_fallback_response(str(e))
        except Exception as e:
            logger.error(f"Unexpected error in step: {e}")
            self.done = True
            return self._create_error_response(str(e))

    #def _select_and_load_task(self, student_profile: Dict[str, Any]) -> TaskInfo:
        """Select and load task using adapters"""
        # Select task type and ID
        task_type, task_id = self.task_selector.select_task(
            student_profile=student_profile,
            available_types=list(self.task_adapters.keys()),
            previous_tasks=self.student_profiler.get_task_history()
        )
        
        # Load task from appropriate adapter
        adapter = self.task_adapters[task_type]
        task_data = adapter.get_task_by_id(task_id)
        
        # Convert to TaskInfo
        return TaskInfo(
            id=task_id,
            type=task_type,
            difficulty=adapter.estimate_difficulty(task_data),
            question=task_data['question'],
            answer=task_data['answer'],
            image_path=task_data['image_path'],
            metadata=task_data.get('metadata', {}),
            adapter_name=task_type
        )
    
    
    def load_task(self, task_type: str, task_id: str) -> TaskInfo:
        """Load task from adapter (without selection)"""
        # Load task from appropriate adapter
        adapter = self.task_adapters[task_type]
        task_data = adapter.get_task_by_id(task_id)
        
        # Convert to TaskInfo
        return TaskInfo(
            id=task_id,
            type=task_type,
            difficulty=adapter.estimate_difficulty(task_data),
            question=task_data['question'],
            answer=task_data['answer'],
            image_path=task_data['image_path'],
            metadata=task_data.get('metadata', {}),
            adapter_name=task_type
        )
    
    
    def _execute_action(self, action: str) -> Dict[str, Any]:
        """执行具体动作 - 在 GPT 环境中，动作就是学生的回答"""
        # 这个方法已经在 step() 中实现了主要逻辑
        # 这里只需要返回基本的执行结果
        return {
            "type": "student_response",
            "content": action,
            "success": True,
            "metadata": {
                "turn_number": self.turn_count,
                "timestamp": time.time()
            }
        }

    def _get_obs(self) -> Dict[str, Any]:
        """获取当前观察 - 调用已有的观察创建方法"""
        # 直接调用已经实现的方法
        return self._create_task_specific_observation()
    
    
    def _calculate_reward(self, analysis: Dict[str, Any]) -> float:
        """计算即时奖励 - GPT环境根据学生表现给出奖励"""
        reward_config = self.config.get('reward_config', {})
        
        # 基于分析结果计算奖励
        if analysis.get('is_correct', False):
            return reward_config.get('correct_answer', 1.0)
        elif analysis.get('reasoning_quality', 0) > 0.7:
            return reward_config.get('good_reasoning', 0.5)
        elif analysis.get('progress', 0) > 0.5:
            return reward_config.get('partial_progress', 0.2)
        else:
            return reward_config.get('incorrect', -0.1)
    
    
    def _decide_next_action(self, analysis: Dict[str, Any]) -> str:
        """决定下一个教学动作"""
        if analysis.get('is_correct', False):
            return 'conclude'
        elif self.turn_count >= self.config.get('max_turns_per_task', 10):
            return 'conclude'
        elif analysis.get('unclear_points', []):
            return 'ask_clarification'
        elif analysis.get('progress', 0) < 0.3:
            return 'provide_hint'
        else:
            return 'scaffold_next_step'
    
    
    def _check_termination_conditions(self, analysis: Dict[str, Any]) -> bool:
        """检查终止条件"""
        if analysis.get('is_correct', False):
            return True
        if self.turn_count >= self.config.get('max_turns_per_task', 10):
            return True
        return False
    
    
    
    def _load_task_image(self, image_path: str) -> np.ndarray:
        """Load task image with error handling"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = self.config.get('max_image_size', (1024, 1024))
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return self._get_placeholder_image()
    
    def _get_placeholder_image(self) -> np.ndarray:
        """Generate placeholder image for error cases"""
        placeholder_size = (224, 224, 3)
        placeholder = np.ones(placeholder_size, dtype=np.uint8) * 128
        return placeholder
    
    def _create_task_specific_observation(self) -> Dict[str, Any]:
        """Create observation with task-specific enhancements"""
        base_obs = {
            'image_path': self.current_task.image_path if self.current_task else None,  # 使用路径而不是数组
            'task_type': self.current_task.type if self.current_task else None,
            'dialogue': self.dialogue_manager.get_formatted_context(),
            'current_prompt': self.dialogue_manager.get_last_environment_turn(),
            'turn_count': self.turn_count
        }
        
        # Add task-specific information
        if self.current_task.type == 'spatial_reasoning':
            base_obs['spatial_hints'] = self._extract_spatial_elements()
        elif self.current_task.type == 'figureqa':
            base_obs['chart_type'] = self._identify_chart_type()
        elif self.current_task.type == 'chartqa':
            base_obs['data_visualization_type'] = self._identify_visualization_type()
        
        return base_obs
    
    def _extract_spatial_elements(self) -> Dict[str, Any]:
        """Extract spatial elements for spatial reasoning tasks"""
        return {
            'reference_objects': self.current_task.metadata.get('objects', []),
            'spatial_relations': self.current_task.metadata.get('relations', []),
            'grid_info': self.current_task.metadata.get('grid', None)
        }
    
    def _identify_chart_type(self) -> str:
        """Identify chart type for figure QA tasks"""
        return self.current_task.metadata.get('chart_type', 'unknown')
    
    def _identify_visualization_type(self) -> str:
        """Identify visualization type for chart QA tasks"""
        return self.current_task.metadata.get('viz_type', 'unknown')
    
    def _generate_initial_prompt_with_retry(self, task: TaskInfo, strategy: Any, 
                                          student_level: str) -> str:
        """Generate initial prompt with retry logic"""
        for attempt in range(self.config['max_retries']):
            try:
                return self.gpt_teacher.generate_initial_prompt(
                    task=task,
                    strategy=strategy,
                    student_level=student_level
                )
            except GPTAPIError as e:
                if attempt == self.config['max_retries'] - 1:
                    raise
                logger.warning(f"GPT API error on attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Fallback
        return self._get_fallback_initial_prompt(task)
    
    def _analyze_response_with_retry(self, agent_action: str) -> Dict[str, Any]:
        """Analyze response with retry logic"""
        for attempt in range(self.config['max_retries']):
            try:
                return self.reasoning_analyzer.analyze(
                    response=agent_action,
                    task=self.current_task,
                    context=self.dialogue_manager.get_context(),
                    expected_reasoning=self.current_strategy.get_expected_reasoning_steps()
                )
            except Exception as e:
                if attempt == self.config['max_retries'] - 1:
                    logger.error(f"Failed to analyze response after {attempt + 1} attempts")
                    return self._get_fallback_analysis()
                time.sleep(1)
    
    def _generate_teacher_response(self, action_type: str, analysis: Dict[str, Any]) -> str:
        """Generate teacher response with error handling"""
        try:
            if action_type == 'provide_hint':
                return self.gpt_teacher.generate_hint(
                    current_step=self.current_strategy.get_current_step(),
                    student_progress=analysis['progress'],
                    misconceptions=analysis.get('misconceptions', [])
                )
            
            elif action_type == 'ask_clarification':
                return self.gpt_teacher.ask_for_clarification(
                    unclear_points=analysis['unclear_points'],
                    context=self.dialogue_manager.get_recent_context(3)
                )
            
            elif action_type == 'provide_encouragement':
                return self.gpt_teacher.provide_encouragement(
                    progress=analysis['progress'],
                    strengths=analysis.get('strengths', [])
                )
            
            elif action_type == 'scaffold_next_step':
                return self.gpt_teacher.scaffold_next_step(
                    completed_steps=analysis['completed_steps'],
                    remaining_steps=self.current_strategy.get_remaining_steps()
                )
            
            elif action_type == 'conclude':
                self.done = True
                logger.info(f"[DEBUG] Passing to GPT - Correct answer: {self.current_task.answer}")
                logger.info(f"[DEBUG] Agent's last response: {self.dialogue_manager.get_last_agent_turn()}")
    
                return self.gpt_teacher.provide_final_feedback(
                    dialogue_history=self.dialogue_manager.get_full_history(),
                    final_answer=self.dialogue_manager.get_last_agent_turn(),
                    correct_answer=self.current_task.answer, #这里是传递正确答案的
                    reasoning_quality=analysis['reasoning_quality'],
                    original_question=self.current_task.question
                )

            else:  # continue dialogue
                return self.gpt_teacher.continue_dialogue(
                    current_state=analysis,
                    strategy=self.current_strategy
                )
                
        except Exception as e:
            logger.error(f"Error generating teacher response: {e}")
            return self._get_fallback_teacher_response(action_type)
    
    
    
    def _finalize_episode(self, final_analysis: Dict[str, Any]):
        """Finalize episode with enhanced tracking"""
        # Calculate episode duration
        episode_duration = time.time() - self.task_start_time
        
        # Prepare episode summary
        episode_summary = {
            'task_id': self.current_task.id,
            'task_type': self.current_task.type,
            'task_difficulty': self.current_task.difficulty,
            'total_turns': self.turn_count,
            'duration': episode_duration,
            'final_reward': self.episode_reward,
            'success': final_analysis.get('is_correct', False),
            'reasoning_quality': final_analysis.get('reasoning_quality', 0.0),
            'strategy_used': self.current_strategy.name
        }
        
        # Assess dialogue quality
        quality_metrics = self._assess_dialogue_quality_comprehensive()
        
        # Only collect reasoning path if answer is correct
        if final_analysis.get('is_correct', False):
            self.reasoning_collector.collect_path(
                dialogue=self.dialogue_manager.get_full_history(),
                task=self.current_task,
                outcome=episode_summary,
                quality_metrics=quality_metrics
            )
            logger.info(f"Collected reasoning path for task {self.current_task.id} - answer is correct")
        
        # Update strategy effectiveness
        strategy_effectiveness = {
            'strategy_name': self.current_strategy.name,
            'task_type': self.current_task.type,
            'success': final_analysis.get('is_correct', False),
            'turns_used': self.turn_count,
            'reasoning_quality': final_analysis.get('reasoning_quality', 0.0)
        }
        self.strategy_selector.update_effectiveness(strategy_effectiveness)
        
        # Update student profile with episode results
        self.student_profiler.record_episode(episode_summary)
    
    
    def _assess_dialogue_quality_comprehensive(self) -> Dict[str, float]:
        """Comprehensive dialogue quality assessment"""
        dialogue = self.dialogue_manager.get_full_history()
        
        # Extract multiple quality dimensions
        metrics = self.reasoning_analyzer.assess_dialogue_quality(dialogue)
        
        # Add additional metrics
        metrics['turn_efficiency'] = min(1.0, self.config['min_turns_per_task'] / self.turn_count)
        metrics['reasoning_depth'] = len(self.dialogue_manager.extract_reasoning_chain()) / self.turn_count
        
        # Calculate weighted overall score
        weights = {
            'reasoning_clarity': 0.25,
            'step_completeness': 0.25,
            'logical_flow': 0.20,
            'answer_quality': 0.20,
            'turn_efficiency': 0.05,
            'reasoning_depth': 0.05
        }
        
        metrics['overall'] = sum(
            metrics.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        
        return metrics
    
    def save_checkpoint(self):
        """Save current state for recovery"""
        checkpoint = {
            'episode_count': self.episode_count,
            'dialogue_history': self.dialogue_manager.export_state(),
            'student_profile': self.student_profiler.export_state(),
            'collected_paths': self.reasoning_collector.get_statistics(),
            'strategy_effectiveness': self.strategy_selector.export_effectiveness_data(),
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_ep{self.episode_count}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint at episode {self.episode_count}")
        self.last_checkpoint_episode = self.episode_count
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for recovery"""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.episode_count = checkpoint['episode_count']
            self.dialogue_manager.import_state(checkpoint['dialogue_history'])
            self.student_profiler.import_state(checkpoint['student_profile'])
            self.strategy_selector.import_effectiveness_data(checkpoint['strategy_effectiveness'])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _should_checkpoint(self) -> bool:
        """Determine if checkpoint should be saved"""
        if not self.config['enable_checkpointing']:
            return False
        
        episodes_since_checkpoint = self.episode_count - self.last_checkpoint_episode
        return episodes_since_checkpoint >= self.config['checkpoint_interval']
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for current episode"""
        reasoning_chain = self.dialogue_manager.extract_reasoning_chain()
        
        return {
            'episode_number': self.episode_count,
            'task_type': self.current_task.type if self.current_task else None,
            'task_difficulty': self.current_task.difficulty if self.current_task else None,
            'reasoning_steps_identified': len(reasoning_chain),
            'avg_turn_quality': self._calculate_avg_turn_quality(),
            'strategy_effectiveness': self.current_strategy.get_effectiveness_score() if self.current_strategy else 0.0,
            'student_engagement': self._estimate_engagement_level(),
            'collection_worthy': self._assess_dialogue_quality_comprehensive()['overall'] >= self.config['quality_threshold'],
            'turns_used': self.turn_count,
            'time_elapsed': time.time() - self.task_start_time if self.task_start_time else 0
        }
    
    def _calculate_avg_turn_quality(self) -> float:
        """Calculate average quality across all turns"""
        if self.turn_count == 0:
            return 0.0
        
        total_quality = sum(
            turn.get('metadata', {}).get('analysis', {}).get('quality_score', 0.0)
            for turn in self.dialogue_manager.get_full_history()
            if turn['role'] == 'agent'
        )
        
        return total_quality / max(1, len([t for t in self.dialogue_manager.get_full_history() if t['role'] == 'agent']))
    
    def _estimate_engagement_level(self) -> float:
        """Estimate student engagement based on response patterns"""
        agent_turns = [t for t in self.dialogue_manager.get_full_history() if t['role'] == 'agent']
        
        if not agent_turns:
            return 0.0
        
        # Factors: response length, diversity, question asking
        avg_length = np.mean([len(turn['content'].split()) for turn in agent_turns])
        length_score = min(1.0, avg_length / 50)  # Normalize to expected length
        
        # Check for questions
        question_ratio = sum(1 for t in agent_turns if '?' in t['content']) / len(agent_turns)
        
        # Response time consistency (if available)
        time_consistency = 1.0  # Placeholder
        
        engagement = (length_score * 0.5 + question_ratio * 0.3 + time_consistency * 0.2)
        return engagement
    
    def _prepare_info_dict(self, analysis: Dict[str, Any], next_action_type: str) -> Dict[str, Any]:
        """Prepare comprehensive info dictionary"""
        return {
            'turn_count': self.turn_count,
            'analysis': analysis,
            'next_action_type': next_action_type,
            'episode_reward': self.episode_reward,
            'reasoning_quality': analysis.get('reasoning_quality', 0.0),
            'episode_metrics': self.get_episode_metrics(),
            'strategy_info': {
                'current_strategy': self.current_strategy.name if self.current_strategy else None,
                'strategy_progress': self.current_strategy.get_progress() if self.current_strategy else 0.0
            }
        }
    
    # Error handling methods
    def _handle_invalid_action(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle invalid agent actions"""
        logger.warning("Received invalid action from agent")
        
        # Penalize invalid action
        reward = self.config['reward_config']['invalid_response']
        
        # Create error response
        self.dialogue_manager.add_turn(
            role='environment',
            content="I didn't understand your response. Could you please try again?",
            metadata={'error': 'invalid_action'}
        )
        
        obs = self._create_task_specific_observation()
        info = {'error': 'invalid_action', 'turn_count': self.turn_count}
        
        return obs, reward, False, info
    
    def _create_fallback_response(self, error_msg: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Create fallback response when GPT fails"""
        logger.warning(f"Creating fallback response due to: {error_msg}")
        
        # Use fallback teaching strategy
        fallback_response = self._get_fallback_teacher_response('continue')
        
        self.dialogue_manager.add_turn(
            role='environment',
            content=fallback_response,
            metadata={'fallback': True, 'error': error_msg}
        )
        
        obs = self._create_task_specific_observation()
        info = {'fallback_used': True, 'error': error_msg}
        
        return obs, 0.0, False, info
    
    def _create_error_response(self, error_msg: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Create error response for critical failures"""
        logger.error(f"Critical error: {error_msg}")
        
        obs = self._create_error_observation(error_msg)
        info = {'critical_error': True, 'error_message': error_msg}
        
        return obs, 0.0, True, info
    
    def _create_error_observation(self, error_msg: str) -> Dict[str, Any]:
        """Create observation for error states"""
        return {
            'image': self._get_placeholder_image(),
            'task_type': 'error',
            'dialogue': [{'role': 'system', 'content': f'Error: {error_msg}'}],
            'current_prompt': "An error occurred. Please reset the environment.",
            'turn_count': self.turn_count
        }
    
    def _get_fallback_initial_prompt(self, task: TaskInfo) -> str:
        """Fallback initial prompt when GPT fails"""
        return f"Let's solve this {task.type} problem. {task.question}"
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when analyzer fails"""
        return {
            'is_correct': False,
            'reasoning_quality': 0.0,
            'progress': 0.0,
            'unclear_points': [],
            'misconceptions': [],
            'completed_steps': []
        }
    
    def _get_fallback_teacher_response(self, action_type: str) -> str:
        """Fallback teacher responses"""
        fallback_responses = {
            'provide_hint': "Think about this step by step. What do you see in the image?",
            'ask_clarification': "Could you explain your reasoning more clearly?",
            'provide_encouragement': "You're on the right track! Keep going.",
            'scaffold_next_step': "Now, let's think about the next step.",
            'conclude': "Thank you for working on this problem.",
            'continue': "Let's continue. What else can you observe?"
        }
        
        return fallback_responses.get(action_type, fallback_responses['continue'])
    
    # Additional utility methods
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
        return {
            'episode_count': self.episode_count,
            'task_id': self.current_task.id if self.current_task else None,
            'turn_count': self.turn_count,
            'episode_reward': self.episode_reward,
            'current_strategy': self.current_strategy.name if self.current_strategy else None,
            'dialogue_length': len(self.dialogue_manager.get_full_history()),
            'collected_paths': self.reasoning_collector.get_statistics()['total_collected']
        }
    
    def render(self, mode: str = 'human'):
        """Render current environment state"""
        if mode == 'human':
            print(f"\n{'='*80}")
            print(f"Episode: {self.episode_count} | Task: {self.current_task.type if self.current_task else 'None'}")
            print(f"Turn: {self.turn_count}/{self.config['max_turns_per_task']}")
            print(f"Strategy: {self.current_strategy.name if self.current_strategy else 'None'}")
            print(f"Episode Reward: {self.episode_reward:.3f}")
            print(f"Collected Paths: {self.reasoning_collector.get_statistics()['total_collected']}")
            print(f"{'='*80}\n")
            
            # Print recent dialogue
            recent = self.dialogue_manager.get_recent_context(3)
            for turn in recent:
                role = "Teacher" if turn['role'] == 'environment' else "Student"
                print(f"{role}: {turn['content'][:200]}{'...' if len(turn['content']) > 200 else ''}\n")
    
    def close(self):
        """Cleanup resources"""
        # Save final checkpoint
        if self.config['enable_checkpointing']:
            self.save_checkpoint()
        
        # Close any open resources
        self.reasoning_collector.close()
        logger.info("GPTEnvironment closed successfully")