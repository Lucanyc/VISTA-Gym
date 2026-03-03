# vlm_gym/environments/task/base.py
"""
VLM-Gym 任务基类定义
定义所有VLM任务的抽象基类和通用接口
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional

class BaseTask(ABC):
    """任务基类 - 统一的VLM任务接口"""
    
    def __init__(self, task_id: str, adapter: 'BaseAdapter', action_set: Optional['VLMActionSet'] = None, **kwargs):
        """
        初始化任务
        
        Args:
            task_id: 任务唯一标识符
            adapter: 数据适配器实例
            action_set: 可选的动作集实例
            **kwargs: 额外的任务参数
        """
        self.task_id = task_id
        self.adapter = adapter
        self.action_set = action_set  # 保存动作集引用
        self.task_data = adapter.get_task_data(task_id)
        self.kwargs = kwargs
        self._setup_complete = False
        
    @classmethod
    def get_task_id(cls) -> str:
        """
        获取任务类型ID
        
        Returns:
            任务类型标识符
        """
        # 获取类名并移除末尾的'Task'
        class_name = cls.__name__.replace("Task", "")
        # 转换CamelCase为连字符格式
        formatted_name = "".join(
            ["-" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("-")
        return f"vlm-gym.{formatted_name}"
    
    @abstractmethod
    def setup(self) -> Tuple[str, Dict[str, Any]]:
        """
        设置任务
        
        在环境重置时调用，用于初始化任务状态
        
        Returns:
            task_goal: 任务目标描述
            task_info: 任务相关信息
        """
        pass
    
    @abstractmethod
    def check_success(self, action: Any) -> Tuple[bool, str]:
        """
        检查动作是否成功完成任务
        
        Args:
            action: 执行的动作
            
        Returns:
            success: 是否成功
            feedback: 反馈信息
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        chat_history: List[Dict],
        observation: Any,
        full_history: Optional[List[Any]] = None
    ) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        验证任务执行情况
        
        Args:
            chat_history: 聊天历史
            observation: 当前观察
            full_history: 完整执行历史
            
        Returns:
            reward: 奖励值
            done: 是否完成
            message: 反馈消息
            info: 额外信息
        """
        pass
    
    def get_observation(self) -> Dict[str, Any]:
        """
        获取当前观察
        
        Returns:
            观察字典
        """
        obs = {
            "type": "task_observation",
            "task_id": self.task_id,
            "task_data": self.task_data,
            "info": self.get_info()
        }
        return obs
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取任务当前信息
        
        Returns:
            任务信息字典
        """
        info = {
            "task_id": self.task_id,
            "task_type": self.get_task_id(),
            "setup_complete": self._setup_complete
        }
        
        # 如果有动作集，添加动作信息
        if self.action_set:
            info["actions_available"] = self.action_set.list_actions()
        
        return info
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取任务指标（可选实现）
        
        Returns:
            指标字典
        """
        return {}
    
    def teardown(self) -> None:
        """
        清理任务资源（可选实现）
        
        在任务结束时调用，用于释放资源
        """
        pass


class BaseAdapter(ABC):
    """数据适配器基类"""
    
    @abstractmethod
    def get_task_data(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务数据
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务数据字典
        """
        pass
    
    @abstractmethod
    def get_task_ids(self, **kwargs) -> List[str]:
        """
        获取可用的任务ID列表
        
        Args:
            **kwargs: 过滤条件
            
        Returns:
            任务ID列表
        """
        pass
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息（可选实现）
        
        Returns:
            统计信息字典
        """
        return {}


# 任务注册相关的异常
class TaskNotFoundError(Exception):
    """任务未找到异常"""
    pass


class TaskRegistrationError(Exception):
    """任务注册错误"""
    pass