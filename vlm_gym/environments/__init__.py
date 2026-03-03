
# vlm_gym/environments/__init__.py
"""
VLM Gym Environments Module

This module provides environments for vision-language model tasks.
It includes a registration system for easy environment creation and management.
"""

import logging

# 导入基础类
from .base import BaseVLMEnv
from .vision_qa_env import VisionQAEnv

# 导入注册系统
from .registration import (
    # 核心函数
    register_env,
    register_task,
    register_adapter,
    make,
    quick_make,
    list_envs,
    get_env_spec,
    init_registry,
    
    # 管理函数
    update_env,
    unregister_env,
    export_registry,
    
    # 基类
    BaseAdapter,
    BaseTask,
    
    # 配置类
    EnvConfig,
    TaskConfig,
    EnvSpec,
)

# 设置日志
logger = logging.getLogger(__name__)

# 自动初始化注册表
try:
    init_registry()
    logger.info("VLM Gym registry initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize VLM Gym registry: {e}")
    logger.info("You can manually initialize with init_registry()")

# 导出的公共API
__all__ = [
    # 环境类
    'BaseVLMEnv',
    'VisionQAEnv',
    
    # 注册系统 - 核心功能
    'register_env',
    'register_task', 
    'register_adapter',
    'make',
    'quick_make',
    'list_envs',
    'get_env_spec',
    'init_registry',
    
    # 注册系统 - 管理功能
    'update_env',
    'unregister_env',
    'export_registry',
    
    # 基类
    'BaseAdapter',
    'BaseTask',
    
    # 配置类
    'EnvConfig',
    'TaskConfig',
    'EnvSpec',
]

# 提供便捷的顶层访问
def create_env(env_id: str, **kwargs):
    """
    便捷函数：创建环境
    
    这是make()的别名，提供更直观的API
    
    Args:
        env_id: 环境ID
        **kwargs: 环境配置参数
        
    Returns:
        环境实例
        
    Example:
        >>> env = create_env("chartqa", max_steps=5)
    """
    return make(env_id, **kwargs)


def available_envs():
    """
    便捷函数：打印所有可用的环境
    
    Example:
        >>> available_envs()
        ChartQA (chartqa-v1): Chart Question Answering tasks
        ScienceQA (scienceqa-v1): Science Question Answering with diagrams
    """
    envs = list_envs()
    if not envs:
        print("No environments registered.")
        return
    
    print("Available environments:")
    for env in envs:
        print(f"  {env['id']}: {env['description']}")
        if env['tags']:
            print(f"    Tags: {', '.join(env['tags'])}")


# 版本信息
__version__ = "0.1.0"