# tools/registry.py
from typing import Dict, List, Optional, Type, Any, Tuple
import logging
from .base import BaseTool, ToolConfig


class ToolRegistry:
    """工具注册表 - 单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tools = {}
            cls._instance.tool_classes = {}
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def register_tool_class(self, name: str, tool_class: Type[BaseTool]):
        """注册工具类"""
        self.tool_classes[name] = tool_class
        self.logger.info(f"Registered tool class: {name}")
    
    def create_tool(self, name: str, config: ToolConfig) -> Optional[BaseTool]:
        """创建工具实例"""
        if name not in self.tool_classes:
            self.logger.error(f"Tool class not found: {name}")
            return None
            
        tool_class = self.tool_classes[name]
        tool = tool_class(config)
        self.tools[name] = tool
        return tool
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[BaseTool]:
        """获取所有可用工具"""
        return [tool for tool in self.tools.values() if tool.config.enabled]
    
    def rank_tools_for_task(self, observation: Dict[str, Any]) -> List[Tuple[BaseTool, float]]:
        """为任务排序工具（按适合度）"""
        rankings = []
        for tool in self.get_available_tools():
            can_handle, confidence = tool.can_handle(observation)
            if can_handle:
                # 结合置信度和优先级计算最终分数
                score = confidence * 0.7 + (tool.config.priority / 10) * 0.3
                rankings.append((tool, score))
        
        # 按分数降序排序
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
