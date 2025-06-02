from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List

class AbstractVLMTask(ABC):
    """VLM任务抽象基类"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    @abstractmethod
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """task setting
        
        Returns:
            task_goal: description of task goal
            task_info: related information of goal
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task excution
        
        Args:
            chat_history: chat history
            last_observation: last observation
            full_history: full history
            
        Returns:
            reward: reward value
            done: whether finished
            info: additional information
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """get current information"""
        pass
    
    def teardown(self):
        """clear up the task resource（optional）"""
        pass
