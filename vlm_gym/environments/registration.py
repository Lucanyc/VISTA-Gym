# vlm_gym/environments/registration.py
"""
VLM Gym Environment Registration System
Supports registration and management of various vision-language tasks
"""
from __future__ import annotations
from typing import Dict, Any, Callable, Type, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import gymnasium as gym
import logging
import os
import importlib
import json

# Import task base classes
from vlm_gym.environments.task.base import BaseTask, BaseAdapter

logger = logging.getLogger(__name__)

# Global registries
_ENV_REGISTRY: Dict[str, 'EnvSpec'] = {}
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}
_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


@dataclass
class TaskConfig:
    """Task configuration"""
    task_type: str
    adapter_type: str
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Process environment variables in configuration"""
        self.adapter_config = self._expand_env_vars(self.adapter_config)
        self.task_kwargs = self._expand_env_vars(self.task_kwargs)
    
    def _expand_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in configuration"""
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = os.path.expandvars(value)
            elif isinstance(value, dict):
                result[key] = self._expand_env_vars(value)
            else:
                result[key] = value
        return result
    

@dataclass
class EnvConfig:
    """Environment configuration"""
    env_id: str
    env_class: str = "VisionQAEnv"
    dataset_path: str = "${WORKSPACE:-/workspace}/data" 
    max_steps: int = 3
    time_limit: Optional[float] = None
    reward_config: Dict[str, Any] = field(default_factory=dict)
    observation_space_config: Dict[str, Any] = field(default_factory=dict)
    action_space_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Replace environment variables"""
        self.dataset_path = os.path.expandvars(self.dataset_path)
        # Process other configurations that may contain paths
        for key in ['reward_config', 'observation_space_config', 'action_space_config']:
            setattr(self, key, self._expand_env_vars(getattr(self, key)))
    
    def _expand_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in configuration"""
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = os.path.expandvars(value)
            elif isinstance(value, dict):
                result[key] = self._expand_env_vars(value)
            else:
                result[key] = value
        return result
    

@dataclass
class EnvSpec:
    """Environment specification"""
    env_id: str
    env_class: Type
    task_config: TaskConfig
    env_config: EnvConfig
    version: str = "v1"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


def _get_env_class(env_class: Union[str, Type]) -> Type:
    """Get environment class, supports dynamic import"""
    if not isinstance(env_class, str):
        return env_class
    
    # Predefined environment class mapping
    env_class_map = {
        "VisionQAEnv": "vlm_gym.environments.vision_qa_env.VisionQAEnv",
        "MultiModalEnv": "vlm_gym.environments.multimodal_env.MultiModalEnv",
        "InteractiveVLMEnv": "vlm_gym.environments.interactive_vlm_env.InteractiveVLMEnv",
        "VLMContainerEnv": "vlm_gym.environments.env.VLMContainerEnv",
        # More environment class mappings can be added here
    }
    
    # First try to get from the mapping
    if env_class in env_class_map:
        module_path = env_class_map[env_class]
    else:
        # If not in the mapping, use the fully qualified class name directly
        module_path = env_class
    
    try:
        # Dynamic import
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # If dynamic import fails, try importing from the current module
        if env_class == "VisionQAEnv":
            try:
                from .vision_qa_env import VisionQAEnv
                return VisionQAEnv
            except ImportError:
                pass
        raise ValueError(f"Cannot import environment class: {env_class}") from e


def register_env(
    env_id: str,
    env_class: Union[str, Type],
    task_type: str,
    adapter_type: str,
    adapter_config: Dict[str, Any] = None,
    env_config: Dict[str, Any] = None,
    task_kwargs: Dict[str, Any] = None,
    version: str = "v1",
    description: str = "",
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
    make_kwargs: Dict[str, Any] = None,
    force: bool = False,
) -> None:
    """
    Register a VLM environment
    
    Args:
        env_id: Environment ID, e.g. "chartqa", "scienceqa", "vizwiz"
        env_class: Environment class or class name
        task_type: Task type, e.g. "vision_qa_task", "vqa_task"
        adapter_type: Adapter type, e.g. "chartqa_adapter"
        adapter_config: Adapter configuration
        env_config: Environment configuration
        task_kwargs: Additional task parameters
        version: Version number
        description: Environment description
        tags: Tag list, e.g. ["vision", "qa", "chart"]
        metadata: Additional metadata
        make_kwargs: Extra arguments for gym.register
        force: Whether to forcefully overwrite an existing registration
    """
    # Get environment class
    env_class = _get_env_class(env_class)
    
    # Create configuration
    task_config = TaskConfig(
        task_type=task_type,
        adapter_type=adapter_type,
        adapter_config=adapter_config or {},
        task_kwargs=task_kwargs or {}
    )
    
    env_cfg = EnvConfig(
        env_id=env_id,
        env_class=env_class.__name__,
        **(env_config or {})
    )
    
    # Create environment specification
    spec = EnvSpec(
        env_id=env_id,
        env_class=env_class,
        task_config=task_config,
        env_config=env_cfg,
        version=version,
        description=description,
        tags=tags or [],
        metadata=metadata or {}
    )
    
    # Check if already exists
    full_id = f"{env_id}-{version}"
    if full_id in _ENV_REGISTRY and not force:
        raise ValueError(f"Environment {full_id} already registered. Use force=True to overwrite.")
    
    # Register in the internal registry
    _ENV_REGISTRY[full_id] = spec
    _ENV_REGISTRY[env_id] = spec  # Also register without version for convenient lookup
    
    # Create environment factory function
    def _make_env(**kwargs):
        return _create_env_from_spec(spec, **kwargs)
    
    # Register with Gymnasium
    gym_id = f"vlm-gym/{full_id}"
    if gym_id in gym.registry:
        if force:
            logger.warning(f"Environment {gym_id} already registered, overwriting...")
            del gym.registry[gym_id]
        else:
            raise ValueError(f"Gym environment {gym_id} already registered. Use force=True to overwrite.")
        
    gym.register(
        id=gym_id,
        entry_point=_make_env,
        **(make_kwargs or {})
    )
    
    logger.info(f"Registered environment: {gym_id}")


def update_env(env_id: str, **kwargs) -> None:
    """
    Update a registered environment's configuration
    
    Args:
        env_id: Environment ID
        **kwargs: Configuration items to update
    """
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    
    spec = _ENV_REGISTRY[env_id]
    
    # Update configuration
    for key, value in kwargs.items():
        if key == 'env_config':
            for k, v in value.items():
                setattr(spec.env_config, k, v)
        elif key == 'task_config':
            for k, v in value.items():
                setattr(spec.task_config, k, v)
        elif key == 'adapter_config':
            spec.task_config.adapter_config.update(value)
        elif key == 'task_kwargs':
            spec.task_config.task_kwargs.update(value)
        elif hasattr(spec, key):
            setattr(spec, key, value)
    
    logger.info(f"Updated environment: {env_id}")


def unregister_env(env_id: str) -> None:
    """
    Unregister an environment
    
    Args:
        env_id: Environment ID
    """
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    
    spec = _ENV_REGISTRY[env_id]
    full_id = f"{env_id}-{spec.version}"
    
    # Remove from internal registry
    if full_id in _ENV_REGISTRY:
        del _ENV_REGISTRY[full_id]
    if env_id in _ENV_REGISTRY:
        del _ENV_REGISTRY[env_id]
    
    # Unregister from Gymnasium
    gym_id = f"vlm-gym/{full_id}"
    if gym_id in gym.registry:
        del gym.registry[gym_id]
    
    logger.info(f"Unregistered environment: {env_id}")


def register_task(task_name: str, task_class: Type[BaseTask]) -> None:
    """
    Register a task class
    
    Args:
        task_name: Task name
        task_class: Task class (must inherit from BaseTask)
    """
    # Use local import to avoid circular dependencies
    from vlm_gym.environments.task.base import BaseTask
    
    if not issubclass(task_class, BaseTask):
        raise ValueError(f"Task class must inherit from BaseTask, got {task_class}")
    
    _TASK_REGISTRY[task_name] = task_class
    logger.info(f"Registered task: {task_name}")


def register_adapter(adapter_name: str, adapter_class: Type[BaseAdapter]) -> None:
    """
    Register an adapter class
    
    Args:
        adapter_name: Adapter name
        adapter_class: Adapter class (must inherit from BaseAdapter)
    """
    # Use local import to avoid circular dependencies
    from vlm_gym.environments.task.base import BaseAdapter
    
    if not issubclass(adapter_class, BaseAdapter):
        raise ValueError(f"Adapter class must inherit from BaseAdapter, got {adapter_class}")
    
    _ADAPTER_REGISTRY[adapter_name] = adapter_class
    logger.info(f"Registered adapter: {adapter_name}")


def _create_env_from_spec(spec: EnvSpec, **kwargs) -> Any:
    """Create an environment from a specification"""
    # Get task class
    task_class = _TASK_REGISTRY.get(spec.task_config.task_type)
    if not task_class:
        raise ValueError(f"Task type not registered: {spec.task_config.task_type}")
    
    # Get adapter class
    adapter_class = _ADAPTER_REGISTRY.get(spec.task_config.adapter_type)
    if not adapter_class:
        raise ValueError(f"Adapter type not registered: {spec.task_config.adapter_type}")
    
    # Create adapter
    adapter_config = spec.task_config.adapter_config.copy()
    adapter_config.update(kwargs.get('adapter_config', {}))
    try:
        adapter = adapter_class(**adapter_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create adapter {spec.task_config.adapter_type}: {e}") from e
    
    # Create environment
    env_config = asdict(spec.env_config)
    env_config.pop('env_id', None)  # env_id does not need to be passed to the environment
    env_config.update(kwargs.get('env_config', {}))
    
    try:
        env = spec.env_class(**env_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create environment {spec.env_class.__name__}: {e}") from e
    
    # Set task entry point
    task_kwargs = spec.task_config.task_kwargs.copy()
    task_kwargs.update(kwargs.get('task_kwargs', {}))
    
    def task_entrypoint(task_id, **kw):
        kw.update(task_kwargs)
        return task_class(task_id=task_id, adapter=adapter, **kw)
    
    env.task_entrypoint = task_entrypoint
    env._adapter = adapter  # Save adapter reference
    env._spec = spec  # Save spec reference
    
    return env


def make(env_id: str, **kwargs) -> Any:
    """
    Create an environment (without relying on Gymnasium)
    
    Args:
        env_id: Environment ID
        **kwargs: Parameters to override default configuration
            - adapter_config: Adapter configuration overrides
            - env_config: Environment configuration overrides
            - task_kwargs: Task parameter overrides
            
    Returns:
        The created environment instance
    """
    spec = _ENV_REGISTRY.get(env_id)
    if not spec:
        available = list(_ENV_REGISTRY.keys())
        raise ValueError(f"Environment not registered: {env_id}. Available: {available}")
    
    return _create_env_from_spec(spec, **kwargs)


def list_envs(tags: List[str] = None, include_versions: bool = False) -> List[Dict[str, Any]]:
    """
    List all registered environments
    
    Args:
        tags: Filter tags
        include_versions: Whether to include version information
        
    Returns:
        List of environment information
    """
    envs = []
    seen = set()
    
    for env_id, spec in _ENV_REGISTRY.items():
        # Decide display logic based on whether versions are included
        if not include_versions and '-' not in env_id:
            continue
        if include_versions and '-' in env_id:
            continue
            
        if env_id in seen:
            continue
        seen.add(env_id)
        
        if tags:
            if not any(tag in spec.tags for tag in tags):
                continue
                
        envs.append({
            'id': env_id,
            'description': spec.description,
            'tags': spec.tags,
            'version': spec.version,
            'task_type': spec.task_config.task_type,
            'adapter_type': spec.task_config.adapter_type,
            'metadata': spec.metadata
        })
    
    return sorted(envs, key=lambda x: x['id'])


def get_env_spec(env_id: str) -> EnvSpec:
    """Get environment specification"""
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    return _ENV_REGISTRY[env_id]


def export_registry(filepath: str) -> None:
    """Export the registry to a file"""
    data = {
        'environments': {},
        'tasks': list(_TASK_REGISTRY.keys()),
        'adapters': list(_ADAPTER_REGISTRY.keys())
    }
    
    for env_id, spec in _ENV_REGISTRY.items():
        if '-' in env_id:  # Only export full IDs with version
            data['environments'][env_id] = {
                'env_class': spec.env_class.__name__,
                'task_type': spec.task_config.task_type,
                'adapter_type': spec.task_config.adapter_type,
                'description': spec.description,
                'tags': spec.tags,
                'version': spec.version
            }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported registry to {filepath}")


# ===== Register concrete implementations =====

def register_builtin_adapters():
    """Register built-in adapters"""

    # try:
    #     from vlm_gym.data_adapters.chartqa_adapter import ChartQAAdapter
    #     register_adapter("chartqa_adapter", ChartQAAdapter)
    # except ImportError as e:
    #     logger.warning(f"ChartQAAdapter not found: {e}")
    
    # Temporarily empty, since you are using an external chartqa_adapter.py
    pass


def register_builtin_tasks():
    """Register built-in tasks"""
    # Only register tasks that actually exist
    try:
        from vlm_gym.environments.task.vision_qa_task import VisionQATask
        register_task("vision_qa_task", VisionQATask)
        register_task("vqa_task", VisionQATask)  # Alias
        logger.info("Registered VisionQATask")
    except ImportError as e:
        logger.warning(f"VisionQATask not found: {e}")
    
    # If you have other task classes, add them here
    # try:
    #     from vlm_gym.environments.task.chart_qa_task import ChartQATask
    #     register_task("chart_qa_task", ChartQATask)
    # except ImportError as e:
    #     logger.warning(f"ChartQATask not found: {e}")


def register_builtin_envs():
    """Register built-in environments"""
    # Temporarily commented out since there are no corresponding adapters and tasks
    # You can modify this according to your actual setup
    
    # # Use environment variables to set default paths
    # default_data_root = os.environ.get('VLM_DATA_ROOT', '/workspace/data')
    
    #
    # # ChartQA
    # register_env(
    #     env_id="chartqa",
    #     env_class="VisionQAEnv",
    #     task_type="vision_qa_task",  # Use an actually existing task
    #     adapter_type="chartqa_adapter",  # Use an actually existing adapter
    #     adapter_config={
    #         "data_root": default_data_root,
    #     },
    #     env_config={
    #         "dataset_path": default_data_root,
    #         "max_steps": 3
    #     },
    #     description="Chart Question Answering tasks",
    #     tags=["vision", "qa", "chart", "reasoning"],
    #     metadata={"difficulty": "medium", "dataset_size": "10k"}
    # )
    
    pass


def init_registry(skip_builtin: bool = False):
    """
    Initialize the registry
    
    Args:
        skip_builtin: Whether to skip registration of built-in components
    """
    if not skip_builtin:
        register_builtin_adapters()
        register_builtin_tasks()
        register_builtin_envs()
        logger.info("Registry initialized with builtin components")
    else:
        logger.info("Registry initialized without builtin components")


# Convenience function
def quick_make(env_id: str, data_root: str = None, max_steps: int = None, **kwargs) -> Any:
    """
    Convenience function for quickly creating an environment
    
    Args:
        env_id: Environment ID
        data_root: Data root directory
        max_steps: Maximum number of steps
        **kwargs: Other parameters
        
    Returns:
        Environment instance
    """
    config_overrides = {}
    
    if data_root:
        config_overrides['adapter_config'] = {'data_root': data_root}
        config_overrides['env_config'] = {'dataset_path': data_root}
    
    if max_steps is not None:
        if 'env_config' not in config_overrides:
            config_overrides['env_config'] = {}
        config_overrides['env_config']['max_steps'] = max_steps
    
    config_overrides.update(kwargs)
    
    return make(env_id, **config_overrides)


if __name__ == "__main__":
    # Initialize
    init_registry()
    
    # List all environments
    print("Available environments:")
    envs = list_envs()
    if envs:
        for env in envs:
            print(f"  - {env['id']}: {env['description']}")
            print(f"    Tags: {env['tags']}")
            print(f"    Task: {env['task_type']} with {env['adapter_type']}")
    else:
        print("  No environments registered yet.")
    
    # If there are registered environments, you can test creation
    # if envs:
    #     env_id = envs[0]['id']
    #     print(f"\nTesting environment creation for: {env_id}")
    #     env = make(env_id)
    #     print(f"Created env: {type(env).__name__}")
