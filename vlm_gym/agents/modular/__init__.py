#!/usr/bin/env python3
"""
Modular VLM Agent with Tools Package
基于任务需求的工具选择架构
"""

import logging
from typing import Type, Dict, Any

# 导出主要类
from .core.agent import VLMAgentWithTools
from .tools.registry import ToolRegistry
from .tools.base import BaseTool, ToolConfig, ToolResult
from .strategies.tool_selection import ToolSelector
from .strategies.reflection import ReflectionStrategy

# 设置日志
logger = logging.getLogger(__name__)

# 全局工具注册表实例
_registry = ToolRegistry()

# 工具类映射（延迟导入）
TOOL_CLASSES: Dict[str, str] = {
    "chartmoe": ".tools.handlers.chartmoe:ChartMoETool",
    "grounding_dino": ".tools.handlers.grounding_dino:GroundingDinoTool",
    "deepeyes": ".tools.handlers.deepeyes:DeepEyesTool",
    "sam2": ".tools.handlers.sam2:SAM2Tool",
    "sympy_geometry": ".tools.handlers.sympy_geometry:SymPyGeometryTool",
    "multimath_server": ".tools.handlers.multimath:MultiMathTool",
    "easyocr": ".tools.handlers.easyocr:EasyOCRTool",
    "diagram_formalizer": ".tools.handlers.diagram_formalizer:DiagramFormalizerTool"
}


def _lazy_import_tool_class(module_path: str) -> Type[BaseTool]:
    """延迟导入工具类"""
    module_name, class_name = module_path.split(':')
    
    # 处理相对导入
    if module_name.startswith('.'):
        from importlib import import_module
        # 获取当前包名
        package_name = __name__
        module = import_module(module_name, package=package_name)
    else:
        from importlib import import_module
        module = import_module(module_name)
    
    return getattr(module, class_name)


def register_all_tools():
    """注册所有可用的工具类"""
    registered = []
    failed = []
    
    for tool_name, module_path in TOOL_CLASSES.items():
        try:
            tool_class = _lazy_import_tool_class(module_path)
            _registry.register_tool_class(tool_name, tool_class)
            registered.append(tool_name)
            logger.debug(f"Registered tool class: {tool_name}")
        except Exception as e:
            failed.append(tool_name)
            logger.warning(f"Failed to register tool {tool_name}: {e}")
    
    logger.info(f"Tool registration complete. Registered: {registered}, Failed: {failed}")
    return registered, failed


def get_registry() -> ToolRegistry:
    """获取全局工具注册表实例"""
    return _registry


def create_agent(config: Dict[str, Any]) -> VLMAgentWithTools:
    """创建配置好的Agent实例
    
    Args:
        config: Agent配置字典
        
    Returns:
        配置好的VLMAgentWithTools实例
    """
    # 确保工具已注册
    if not _registry.tools:
        register_all_tools()
    
    # 创建Agent
    agent = VLMAgentWithTools(config)
    
    # 如果环境提供了工具实例，设置它们
    # 这是为了兼容原有的环境工具管理器
    if "tool_instances" in config:
        tool_instances = config["tool_instances"]
        for tool_name, tool_instance in tool_instances.items():
            if hasattr(agent, f"{tool_name}_tool"):
                setattr(agent, f"{tool_name}_tool", tool_instance)
                logger.debug(f"Set external tool instance: {tool_name}")
    
    return agent


# 自动注册已实现的工具类
# 只注册已经实现的ChartMoE
try:
    from .tools.handlers.chartmoe import ChartMoETool
    _registry.register_tool_class("chartmoe", ChartMoETool)
    logger.info("Auto-registered ChartMoE tool")
except ImportError as e:
    logger.warning(f"Could not auto-register ChartMoE: {e}")


# 导出的公共接口
__all__ = [
    # 主要类
    "VLMAgentWithTools",
    "ToolRegistry",
    "BaseTool",
    "ToolConfig",
    "ToolResult",
    "ToolSelector",
    "ReflectionStrategy",
    
    # 函数
    "register_all_tools",
    "get_registry",
    "create_agent"
]


# 版本信息
__version__ = "0.1.0"
__author__ = "VLM Gym Team"