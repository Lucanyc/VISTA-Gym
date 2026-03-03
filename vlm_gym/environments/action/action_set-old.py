# vlm_gym/environments/action/action_set.py
"""
VLM动作集管理器
类似于MedAgentGym中的action_set.py (BasicActionSet)
"""
from typing import Dict, Any, Optional, List, Callable
import inspect
from .base import AbstractActionSet
from . import vlm_actions
from .parser import ActionParser

class VLMActionSet(AbstractActionSet):
    """VLM动作集管理器 - 负责注册、管理和执行动作"""
    
    def __init__(self, custom_actions: Optional[Dict[str, Callable]] = None, strict: bool = False):
        super().__init__(strict=strict)
        
        # 初始化解析器
        self.parser = ActionParser()
        
        # 自动从vlm_actions模块导入所有动作
        self.action_set = {}
        self._load_core_actions()
        
        # 添加自定义动作
        if custom_actions:
            for name, func in custom_actions.items():
                self._register_action(name, func)
    
    def _load_core_actions(self):
        """从vlm_actions模块加载所有核心动作"""
        # 获取vlm_actions模块中的所有函数
        for name, func in inspect.getmembers(vlm_actions, inspect.isfunction):
            # 跳过私有函数
            if not name.startswith('_'):
                self._register_action(name, func)
    
    def _register_action(self, name: str, func: Callable):
        """注册一个动作"""
        # 解析函数信息
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        # 提取描述和示例
        description, examples = self._parse_docstring(docstring)
        
        # 注册动作
        self.action_set[name] = {
            "function": func,
            "signature": str(signature),
            "description": description,
            "examples": examples,
            "params": list(signature.parameters.keys())
        }
    
    def _parse_docstring(self, docstring: str) -> tuple[str, List[str]]:
        """解析docstring提取描述和示例"""
        if not docstring:
            return "", []
        
        lines = docstring.strip().split('\n')
        description = []
        examples = []
        in_examples = False
        
        for line in lines:
            line = line.strip()
            if line.lower() == "examples:":
                in_examples = True
                continue
            
            if in_examples:
                if line and not line.startswith(' '):
                    # 如果遇到非缩进的非空行，结束示例部分
                    break
                if line:
                    examples.append(line.strip())
            else:
                if line and not line.lower().startswith("examples:"):
                    description.append(line)
        
        return ' '.join(description), examples
    
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        """返回动作空间的文本描述"""
        description = f"{len(self.action_set)} different types of actions are available.\n\n"
        
        # 按字母顺序排序动作
        for action_name in sorted(self.action_set.keys()):
            action_info = self.action_set[action_name]
            
            # 动作签名
            description += f"{action_name}{action_info['signature']}\n"
            
            # 动作描述
            if with_long_description and action_info['description']:
                description += f"    Description: {action_info['description']}\n"
            
            # 动作示例
            if with_examples and action_info['examples']:
                description += "    Examples:\n"
                for example in action_info['examples']:
                    description += f"        {example}\n"
            
            description += "\n"
        
        return description
    
    def parse_action(self, action_str: str) -> Dict[str, Any]:
        """解析动作字符串"""
        action_name, params = self.parser.parse(action_str)
        
        # 验证动作是否存在
        if action_name not in self.action_set:
            if self.strict:
                raise ValueError(f"Unknown action: {action_name}. Available actions: {list(self.action_set.keys())}")
            else:
                return {"action": action_name, "params": params, "valid": False}
        
        return {"action": action_name, "params": params, "valid": True}
    
    def execute_action(self, action_str: str) -> Dict[str, Any]:
        """执行动作并返回结果"""
        try:
            # 解析动作
            parsed = self.parse_action(action_str)
            
            if not parsed.get("valid", True):
                return {
                    "type": "action_execution",
                    "status": "FAILED",
                    "error": f"Unknown action: {parsed['action']}",
                    "available_actions": list(self.action_set.keys())
                }
            
            action_name = parsed["action"]
            params = parsed["params"]
            
            # 获取动作函数
            action_func = self.action_set[action_name]["function"]
            
            # 执行动作
            result = action_func(**params)
            
            return {
                "type": "action_execution",
                "status": "SUCCESS",
                "action": action_name,
                "result": result
            }
            
        except Exception as e:
            error_msg = str(e)
            if "missing" in error_msg and "required positional argument" in error_msg:
                # 提供更友好的参数错误信息
                required_params = self.action_set[action_name]["params"]
                error_msg = f"Missing required parameters. Expected: {required_params}"
            
            return {
                "type": "action_execution",
                "status": "FAILED",
                "action": action_str,
                "error": error_msg
            }
    
    def validate_action(self, action_str: str) -> bool:
        """验证动作是否有效"""
        try:
            parsed = self.parse_action(action_str)
            return parsed.get("valid", False)
        except:
            return False
    
    def list_actions(self) -> List[str]:
        """列出所有可用的动作"""
        return list(self.action_set.keys())
    
    def get_action_info(self, action_name: str) -> Dict[str, Any]:
        """获取特定动作的详细信息"""
        if action_name not in self.action_set:
            raise ValueError(f"Unknown action: {action_name}")
        
        return self.action_set[action_name].copy()
    
    def example_action(self, action_name: str) -> List[str]:
        """返回动作的示例"""
        if action_name not in self.action_set:
            raise ValueError(f"Unknown action: {action_name}")
        
        return self.action_set[action_name]["examples"]