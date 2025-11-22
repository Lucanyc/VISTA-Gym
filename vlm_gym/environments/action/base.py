from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AbstractActionSet(ABC):
    """"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.actions = {}
    
    @abstractmethod
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """return description"""
        pass
    
    @abstractmethod
    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """parse string"""
        pass
    
    def validate_action(self, action_str: str) -> bool:
        """validate the effectiveness of action"""
        try:
            self.parse_action(action_str)
            return True
        except:
            return False
