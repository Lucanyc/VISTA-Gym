import time
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging
from dataclasses import dataclass

from .chat import MultiModalChat
from .spaces import Unicode, ImageSpace, AnyDict
from .action.base import AbstractActionSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnvObservation:
    """Environment observation result"""
    type: str
    content: Any
    timestamp: float
    metadata: Dict[str, Any] = None

class BaseVLMEnv(gym.Env, ABC):
    """Base class for VLM environments"""
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        task_entrypoint: type,
        task_kwargs: Dict[str, Any] = None,
        action_set: AbstractActionSet = None,
        max_steps: int = 100,
        time_limit: float = 300.0,
    ):
        super().__init__()
        
        self.task_entrypoint = task_entrypoint
        self.task_kwargs = task_kwargs or {}
        self.max_steps = max_steps
        self.time_limit = time_limit
        
        # Initialize task and action set
        self.task = None
        self.action_set = action_set
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "image": ImageSpace(),
            "text": Unicode(),
            "chat_history": spaces.Sequence(
                spaces.Dict({
                    "role": Unicode(),
                    "content": AnyDict(),  # Supports multimodal content
                    "timestamp": spaces.Box(low=0, high=float('inf'), shape=()),
                })
            ),
            "task_info": AnyDict(),
        })
        
        # Define action space
        self.action_space = Unicode()
        
        # Environment state
        self.chat: MultiModalChat = None
        self.current_step = 0
        self.start_time = None
        self.history: List[EnvObservation] = []
        
    def reset(self, task_id: str = None, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        # Clean up old task
        if self.task:
            self.task.teardown()
            
        # Create new task
        self.task = self.task_entrypoint(
            task_id=task_id,
            **self.task_kwargs
        )
        
        # Initialize chat
        self.chat = MultiModalChat()
        
        # Set up task
        task_goal, task_info = self.task.setup()
        
        # Initialize environment state
        self.current_step = 0
        self.start_time = time.time()
        self.history = []
        
        # Get initial observation
        obs = self._get_obs()
        info = {
            "task_goal": task_goal,
            "task_info": task_info,
        }
        
        return obs, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute an action"""
        self.current_step += 1
        
        # Execute action
        try:
            result = self._execute_action(action)
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            result = {
                "type": "error",
                "content": str(e),
                "success": False
            }
        
        # Create observation
        obs = self._create_observation(result)
        self.history.append(obs)
        
        # Validate task state
        reward, done, info = self.task.validate(
            self.chat.messages,
            obs,
            self.history
        )
        
        # Check termination conditions
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            info["truncation_reason"] = "max_steps_reached"
        elif time.time() - self.start_time > self.time_limit:
            truncated = True
            info["truncation_reason"] = "time_limit_exceeded"
            
        return self._get_obs(), reward, done, truncated, info
    
    @abstractmethod
    def _execute_action(self, action: str) -> Dict[str, Any]:
        """Execute a specific action"""
        pass
    
    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """Get the current observation"""
        pass
    
    def _create_observation(self, result: Dict[str, Any]) -> EnvObservation:
        """Create an observation object"""
        return EnvObservation(
            type=result.get("type", "unknown"),
            content=result.get("content"),
            timestamp=time.time(),
            metadata=result.get("metadata", {})
        )
    
    def close(self):
        """Close the environment"""
        if self.task:
            self.task.teardown()
        if self.chat:
            self.chat.close()
