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
    """环境观察结果"""
    type: str
    content: Any
    timestamp: float
    metadata: Dict[str, Any] = None

class BaseVLMEnv(gym.Env, ABC):
    """VLM环境基础类"""
    
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
        
        # 初始化任务和动作集
        self.task = None
        self.action_set = action_set
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            "image": ImageSpace(),
            "text": Unicode(),
            "chat_history": spaces.Sequence(
                spaces.Dict({
                    "role": Unicode(),
                    "content": AnyDict(),  # 支持多模态内容
                    "timestamp": spaces.Box(low=0, high=float('inf'), shape=()),
                })
            ),
            "task_info": AnyDict(),
        })
        
        # 定义动作空间
        self.action_space = Unicode()
        
        # 环境状态
        self.chat: MultiModalChat = None
        self.current_step = 0
        self.start_time = None
        self.history: List[EnvObservation] = []
        
    def reset(self, task_id: str = None, **kwargs) -> Tuple[Dict, Dict]:
        """重置环境"""
        # 清理旧任务
        if self.task:
            self.task.teardown()
            
        # 创建新任务
        self.task = self.task_entrypoint(
            task_id=task_id,
            **self.task_kwargs
        )
        
        # 初始化聊天
        self.chat = MultiModalChat()
        
        # 设置任务
        task_goal, task_info = self.task.setup()
        
        # 初始化环境状态
        self.current_step = 0
        self.start_time = time.time()
        self.history = []
        
        # 获取初始观察
        obs = self._get_obs()
        info = {
            "task_goal": task_goal,
            "task_info": task_info,
        }
        
        return obs, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作"""
        self.current_step += 1
        
        # 执行动作
        try:
            result = self._execute_action(action)
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            result = {
                "type": "error",
                "content": str(e),
                "success": False
            }
        
        # 创建观察
        obs = self._create_observation(result)
        self.history.append(obs)
        
        # 验证任务状态
        reward, done, info = self.task.validate(
            self.chat.messages,
            obs,
            self.history
        )
        
        # 检查终止条件
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
        """执行具体动作"""
        pass
    
    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """获取当前观察"""
        pass
    
    def _create_observation(self, result: Dict[str, Any]) -> EnvObservation:
        """创建观察对象"""
        return EnvObservation(
            type=result.get("type", "unknown"),
            content=result.get("content"),
            timestamp=time.time(),
            metadata=result.get("metadata", {})
        )
    
    def close(self):
        """关闭环境"""
        if self.task:
            self.task.teardown()
        if self.chat:
            self.chat.close()