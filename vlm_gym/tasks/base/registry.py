#!/usr/bin/env python3
"""Task registry for managing task types"""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry for task wrapper classes"""
    
    _instance = None
    _registry: Dict[str, Type] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, task_type: str, task_class: Type):
        """Register a task wrapper class
        
        Args:
            task_type: Task type identifier (e.g., 'chartqa', 'scienceqa')
            task_class: Task wrapper class
        """
        if task_type in cls._registry:
            logger.warning(f"Overwriting existing task type: {task_type}")
        
        cls._registry[task_type] = task_class
        logger.info(f"Registered task type: {task_type}")
    
    @classmethod
    def get_task_class(cls, task_type: str) -> Optional[Type]:
        """Get task wrapper class by type
        
        Args:
            task_type: Task type identifier
            
        Returns:
            Task wrapper class or None
        """
        return cls._registry.get(task_type)
    
    @classmethod
    def create_task(cls, task_type: str, task_id: str, 
                   task_data: Dict[str, Any], **kwargs):
        """Create task wrapper instance
        
        Args:
            task_type: Task type identifier
            task_id: Unique task ID
            task_data: Task data (question, answer, image_path, etc.)
            **kwargs: Additional configuration
            
        Returns:
            Task wrapper instance
            
        Raises:
            ValueError: If task type not registered
        """
        task_class = cls.get_task_class(task_type)
        if task_class is None:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return task_class(task_id, task_data, **kwargs)
    
    @classmethod
    def list_registered(cls) -> list:
        """List all registered task types"""
        return list(cls._registry.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registrations (mainly for testing)"""
        cls._registry.clear()


def register_task(task_type: str):
    """Decorator to register task wrapper class
    
    Usage:
        @register_task("chartqa")
        class ChartQATask(BaseTaskWrapper):
            ...
    """
    def decorator(cls):
        TaskRegistry.register(task_type, cls)
        return cls
    return decorator


# Convenience functions
def get_task_class(task_type: str) -> Optional[Type]:
    """Get task class by type"""
    return TaskRegistry.get_task_class(task_type)


def create_task(task_type: str, task_id: str, 
               task_data: Dict[str, Any], **kwargs):
    """Create task instance"""
    return TaskRegistry.create_task(task_type, task_id, task_data, **kwargs)