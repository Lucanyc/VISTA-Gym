from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AbstractActionSet(ABC):
    """抽象动作集基类"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.actions = {}
    
    @abstractmethod
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """返回动作空间的文本描述"""
        pass
    
    @abstractmethod
    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """解析动作字符串"""
        pass
    
    def validate_action(self, action_str: str) -> bool:
        """验证动作是否有效"""
        try:
            self.parse_action(action_str)
            return True
        except:
            return False