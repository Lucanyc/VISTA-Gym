from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class ToolConfig:
    """工具配置基类"""
    name: str
    enabled: bool = True
    max_retries: int = 3
    timeout: float = 30.0
    priority: int = 0  # 用于工具选择优先级
    
class ToolResult:
    """工具执行结果的统一格式"""
    def __init__(self, success: bool, data: Any, error: Optional[str] = None, 
                 metadata: Optional[Dict] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }

class BaseTool(ABC):
    """所有工具的基类"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.name = config.name
        self.call_history = []
        
    @abstractmethod
    def can_handle(self, observation: Dict[str, Any]) -> Tuple[bool, float]:
        """
        判断是否能处理该任务
        返回: (能否处理, 置信度分数0-1)
        """
        pass
    
    @abstractmethod
    def build_prompt(self, observation: Dict[str, Any]) -> str:
        """构建工具调用提示"""
        pass
    
    @abstractmethod
    def process_result(self, raw_result: Any, observation: Dict[str, Any]) -> ToolResult:
        """处理工具返回的原始结果"""
        pass
    
    @abstractmethod
    def format_for_answer(self, result: ToolResult, observation: Dict[str, Any]) -> str:
        """将工具结果格式化为最终答案的提示"""
        pass
    
    def validate_input(self, observation: Dict[str, Any]) -> bool:
        """验证输入是否符合工具要求"""
        return True
    
    def should_retry(self, result: ToolResult, attempt: int) -> bool:
        """判断是否应该重试"""
        return not result.success and attempt < self.config.max_retries
    
    def get_fallback_strategy(self) -> Optional[str]:
        """获取失败时的后备策略"""
        return None
