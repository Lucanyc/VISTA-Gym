# vlm_gym/environments/__init__.py
"""
VLM Gym Environments Module
This module provides environments for vision-language model tasks.
It includes a registration system for easy environment creation and management.
"""
import logging

# Import base classes
from .base import BaseVLMEnv
from .vision_qa_env import VisionQAEnv

# Import registration system
from .registration import (
    # Core functions
    register_env,
    register_task,
    register_adapter,
    make,
    quick_make,
    list_envs,
    get_env_spec,
    init_registry,
    
    # Management functions
    update_env,
    unregister_env,
    export_registry,
    
    # Base classes
    BaseAdapter,
    BaseTask,
    
    # Configuration classes
    EnvConfig,
    TaskConfig,
    EnvSpec,
)

# Set up logging
logger = logging.getLogger(__name__)

# Auto-initialize registry
try:
    init_registry()
    logger.info("VLM Gym registry initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize VLM Gym registry: {e}")
    logger.info("You can manually initialize with init_registry()")

# Exported public API
__all__ = [
    # Environment classes
    'BaseVLMEnv',
    'VisionQAEnv',
    
    # Registration system - Core functions
    'register_env',
    'register_task', 
    'register_adapter',
    'make',
    'quick_make',
    'list_envs',
    'get_env_spec',
    'init_registry',
    
    # Registration system - Management functions
    'update_env',
    'unregister_env',
    'export_registry',
    
    # Base classes
    'BaseAdapter',
    'BaseTask',
    
    # Configuration classes
    'EnvConfig',
    'TaskConfig',
    'EnvSpec',
]

# Provide convenient top-level access
def create_env(env_id: str, **kwargs):
    """
    Convenience function: Create an environment
    
    This is an alias for make(), providing a more intuitive API
    
    Args:
        env_id: Environment ID
        **kwargs: Environment configuration parameters
        
    Returns:
        Environment instance
        
    Example:
        >>> env = create_env("chartqa", max_steps=5)
    """
    return make(env_id, **kwargs)

def available_envs():
    """
    Convenience function: Print all available environments
    
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

# Version info
__version__ = "0.1.0"
