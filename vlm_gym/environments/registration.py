
# vlm_gym/environments/registration.py
"""
VLM Gym 环境注册系统
支持多种视觉-语言任务的注册和管理
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

# 导入任务基类
from vlm_gym.environments.task.base import BaseTask, BaseAdapter

logger = logging.getLogger(__name__)

# 全局注册表
_ENV_REGISTRY: Dict[str, 'EnvSpec'] = {}
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}
_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


@dataclass
class TaskConfig:
    """任务配置"""
    task_type: str
    adapter_type: str
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """处理配置中的环境变量"""
        self.adapter_config = self._expand_env_vars(self.adapter_config)
        self.task_kwargs = self._expand_env_vars(self.task_kwargs)
    
    def _expand_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归展开配置中的环境变量"""
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
    """环境配置"""
    env_id: str
    env_class: str = "VisionQAEnv"
    dataset_path: str = "${WORKSPACE:-/workspace}/data" 
    max_steps: int = 3
    time_limit: Optional[float] = None
    reward_config: Dict[str, Any] = field(default_factory=dict)
    observation_space_config: Dict[str, Any] = field(default_factory=dict)
    action_space_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """替换环境变量"""
        self.dataset_path = os.path.expandvars(self.dataset_path)
        # 处理其他可能包含路径的配置
        for key in ['reward_config', 'observation_space_config', 'action_space_config']:
            setattr(self, key, self._expand_env_vars(getattr(self, key)))
    
    def _expand_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归展开配置中的环境变量"""
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
    """环境规范"""
    env_id: str
    env_class: Type
    task_config: TaskConfig
    env_config: EnvConfig
    version: str = "v1"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


def _get_env_class(env_class: Union[str, Type]) -> Type:
    """获取环境类，支持动态导入"""
    if not isinstance(env_class, str):
        return env_class
    
    # 预定义的环境类映射
    env_class_map = {
        "VisionQAEnv": "vlm_gym.environments.vision_qa_env.VisionQAEnv",
        "MultiModalEnv": "vlm_gym.environments.multimodal_env.MultiModalEnv",
        "InteractiveVLMEnv": "vlm_gym.environments.interactive_vlm_env.InteractiveVLMEnv",
        "VLMContainerEnv": "vlm_gym.environments.env.VLMContainerEnv",
        # 这里可以留着添加更多环境类的映射
    }
    
    # 首先尝试从映射中获取
    if env_class in env_class_map:
        module_path = env_class_map[env_class]
    else:
        # 如果不在映射中，则直接使用传入完整的类名
        module_path = env_class
    
    try:
        # 动态导入
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # 如果动态导入失败，尝试从当前模块导入
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
    注册VLM环境
    
    Args:
        env_id: 环境ID，如 "chartqa", "scienceqa", "vizwiz"
        env_class: 环境类或类名
        task_type: 任务类型，如 "vision_qa_task", "vqa_task"
        adapter_type: 适配器类型，如 "chartqa_adapter"
        adapter_config: 适配器配置
        env_config: 环境配置
        task_kwargs: 任务额外参数
        version: 版本号
        description: 环境描述
        tags: 标签列表，如 ["vision", "qa", "chart"]
        metadata: 额外的元数据
        make_kwargs: gym.register的额外参数
        force: 是否强制覆盖已存在的注册
    """
    # 获取环境类
    env_class = _get_env_class(env_class)
    
    # 创建配置
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
    
    # 创建环境规范
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
    
    # 检查是否已存在
    full_id = f"{env_id}-{version}"
    if full_id in _ENV_REGISTRY and not force:
        raise ValueError(f"Environment {full_id} already registered. Use force=True to overwrite.")
    
    # 注册到内部注册表
    _ENV_REGISTRY[full_id] = spec
    _ENV_REGISTRY[env_id] = spec  # 也注册不带版本的ID，方便查询
    
    # 创建 environment factory 函数
    def _make_env(**kwargs):
        return _create_env_from_spec(spec, **kwargs)
    
    # 注册到Gymnasium
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
    更新已注册的环境配置
    
    Args:
        env_id: 环境ID
        **kwargs: 要更新的配置项
    """
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    
    spec = _ENV_REGISTRY[env_id]
    
    # 更新配置
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
    注销环境
    
    Args:
        env_id: 环境ID
    """
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    
    spec = _ENV_REGISTRY[env_id]
    full_id = f"{env_id}-{spec.version}"
    
    # 从内部注册表删除
    if full_id in _ENV_REGISTRY:
        del _ENV_REGISTRY[full_id]
    if env_id in _ENV_REGISTRY:
        del _ENV_REGISTRY[env_id]
    
    # 从Gymnasium注销
    gym_id = f"vlm-gym/{full_id}"
    if gym_id in gym.registry:
        del gym.registry[gym_id]
    
    logger.info(f"Unregistered environment: {env_id}")


def register_task(task_name: str, task_class: Type[BaseTask]) -> None:
    """
    注册任务类
    
    Args:
        task_name: 任务名称
        task_class: 任务类（必须继承自BaseTask）
    """
    # 使用本地导入避免循环依赖
    from vlm_gym.environments.task.base import BaseTask
    
    if not issubclass(task_class, BaseTask):
        raise ValueError(f"Task class must inherit from BaseTask, got {task_class}")
    
    _TASK_REGISTRY[task_name] = task_class
    logger.info(f"Registered task: {task_name}")


def register_adapter(adapter_name: str, adapter_class: Type[BaseAdapter]) -> None:
    """
    注册适配器类
    
    Args:
        adapter_name: 适配器名称
        adapter_class: 适配器类（必须继承自BaseAdapter）
    """
    # 使用本地导入避免循环依赖
    from vlm_gym.environments.task.base import BaseAdapter
    
    if not issubclass(adapter_class, BaseAdapter):
        raise ValueError(f"Adapter class must inherit from BaseAdapter, got {adapter_class}")
    
    _ADAPTER_REGISTRY[adapter_name] = adapter_class
    logger.info(f"Registered adapter: {adapter_name}")


def _create_env_from_spec(spec: EnvSpec, **kwargs) -> Any:
    """从规范创建环境"""
    # 获取任务类
    task_class = _TASK_REGISTRY.get(spec.task_config.task_type)
    if not task_class:
        raise ValueError(f"Task type not registered: {spec.task_config.task_type}")
    
    # 获取适配器类
    adapter_class = _ADAPTER_REGISTRY.get(spec.task_config.adapter_type)
    if not adapter_class:
        raise ValueError(f"Adapter type not registered: {spec.task_config.adapter_type}")
    
    # 创建适配器
    adapter_config = spec.task_config.adapter_config.copy()
    adapter_config.update(kwargs.get('adapter_config', {}))
    try:
        adapter = adapter_class(**adapter_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create adapter {spec.task_config.adapter_type}: {e}") from e
    
    # 创建环境
    env_config = asdict(spec.env_config)
    env_config.pop('env_id', None) # env_id不需要传递给环境
    env_config.update(kwargs.get('env_config', {}))
    
    try:
        env = spec.env_class(**env_config)
    except Exception as e:
        raise RuntimeError(f"Failed to create environment {spec.env_class.__name__}: {e}") from e
    
    # 设置任务入口点
    task_kwargs = spec.task_config.task_kwargs.copy()
    task_kwargs.update(kwargs.get('task_kwargs', {}))
    
    def task_entrypoint(task_id, **kw):
        kw.update(task_kwargs)
        return task_class(task_id=task_id, adapter=adapter, **kw)
    
    env.task_entrypoint = task_entrypoint
    env._adapter = adapter  # 保存adapter引用
    env._spec = spec  # 保存spec引用
    
    return env


def make(env_id: str, **kwargs) -> Any:
    """
    创建环境（不依赖Gymnasium）
    
    Args:
        env_id: 环境ID
        **kwargs: 覆盖默认配置的参数
            - adapter_config: 适配器配置覆盖
            - env_config: 环境配置覆盖
            - task_kwargs: 任务参数覆盖
            
    Returns:
        创建的环境实例
    """
    spec = _ENV_REGISTRY.get(env_id)
    if not spec:
        available = list(_ENV_REGISTRY.keys())
        raise ValueError(f"Environment not registered: {env_id}. Available: {available}")
    
    return _create_env_from_spec(spec, **kwargs)


def list_envs(tags: List[str] = None, include_versions: bool = False) -> List[Dict[str, Any]]:
    """
    列出所有注册的环境
    
    Args:
        tags: 过滤标签
        include_versions: 是否包含版本信息
        
    Returns:
        环境信息列表
    """
    envs = []
    seen = set()
    
    for env_id, spec in _ENV_REGISTRY.items():
        # 根据是否包含版本决定显示逻辑
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
    """获取环境规范"""
    if env_id not in _ENV_REGISTRY:
        raise ValueError(f"Environment not registered: {env_id}")
    return _ENV_REGISTRY[env_id]


def export_registry(filepath: str) -> None:
    """导出注册表到文件"""
    data = {
        'environments': {},
        'tasks': list(_TASK_REGISTRY.keys()),
        'adapters': list(_ADAPTER_REGISTRY.keys())
    }
    
    for env_id, spec in _ENV_REGISTRY.items():
        if '-' in env_id:  # 只导出带版本的完整ID
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


# ===== 注册具体实现 =====

def register_builtin_adapters():
    """注册内置适配器"""

    # try:
    #     from vlm_gym.data_adapters.chartqa_adapter import ChartQAAdapter
    #     register_adapter("chartqa_adapter", ChartQAAdapter)
    # except ImportError as e:
    #     logger.warning(f"ChartQAAdapter not found: {e}")
    
    # 暂时为空，因为您使用的是外部的 chartqa_adapter.py
    
    try:
        from data_adapters.olympiadbench_adapter import OlympiadBenchAdapter
        register_adapter("olympiadbench_adapter", OlympiadBenchAdapter)
        logger.info("Registered OlympiadBenchAdapter")
    except ImportError as e:
        logger.warning(f"OlympiadBenchAdapter not found: {e}")
    
    pass


def register_builtin_tasks():
    """注册内置任务"""
    # 只注册实际存在的任务
    try:
        from vlm_gym.environments.task.vision_qa_task import VisionQATask
        register_task("vision_qa_task", VisionQATask)
        register_task("vqa_task", VisionQATask)  # 别名
        logger.info("Registered VisionQATask")
    except ImportError as e:
        logger.warning(f"VisionQATask not found: {e}")
        
    
    try:
        from vlm_gym.environments.task.olympiadbench import OlympiadBenchTask
        register_task("olympiadbench_task", OlympiadBenchTask)
        logger.info("Registered OlympiadBenchTask")
    except ImportError as e:
        logger.warning(f"OlympiadBenchTask not found: {e}")
    
    # 如果您有其他任务类，在这里添加
    # try:
    #     from vlm_gym.environments.task.chart_qa_task import ChartQATask
    #     register_task("chart_qa_task", ChartQATask)
    # except ImportError as e:
    #     logger.warning(f"ChartQATask not found: {e}")


def register_builtin_envs():
    """注册内置环境"""
    # 暂时注释掉，因为没有对应的 adapter 和 task
    # 您可以根据实际情况修改
    
    # # 使用环境变量设置默认路径
    # default_data_root = os.environ.get('VLM_DATA_ROOT', '/workspace/data')
    
    #
    # # ChartQA
    # register_env(
    #     env_id="chartqa",
    #     env_class="VisionQAEnv",
    #     task_type="vision_qa_task",  # 使用实际存在的任务
    #     adapter_type="chartqa_adapter",  # 使用实际存在的适配器
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
    
    default_data_root = os.environ.get('VLM_DATA_ROOT', '/workspace/data')
    
    # OlympiadBench
    register_env(
        env_id="olympiadbench",
        env_class="VisionQAEnv",
        task_type="olympiadbench_task",
        adapter_type="olympiadbench_adapter",
        adapter_config={
            "data_root": "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/olympiadbench/OlympiadBench_Dataset",
            "annotation_files": "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/olympiadbench/OlympiadBench_Dataset/combined_dataset/olympiadbench_format.json",
            "validate_images": True
        },
        env_config={
            "dataset_path": "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/data/olympiadbench/OlympiadBench_Dataset",
            "max_steps": 5,  # 给更多步数处理复杂问题
            "enable_grounding_dino": True,
            "enable_chartmoe": True,
            "enable_diagram_formalizer": True,
            "enable_deepeyes_tools": True
        },
        description="Olympic-level mathematics competition problems",
        tags=["vision", "qa", "math", "olympiad", "geometry", "algebra", "physics"],
        metadata={
            "difficulty": "very_high", 
            "dataset_size": "8900+",
            "answer_types": ["numeric", "expression", "equation", "interval", "tuple"]
        }
    )
    
    
    pass


def init_registry(skip_builtin: bool = False):
    """
    初始化注册表
    
    Args:
        skip_builtin: 是否跳过内置组件的注册
    """
    if not skip_builtin:
        register_builtin_adapters()
        register_builtin_tasks()
        register_builtin_envs()
        logger.info("Registry initialized with builtin components")
    else:
        logger.info("Registry initialized without builtin components")


# 便捷函数
def quick_make(env_id: str, data_root: str = None, max_steps: int = None, **kwargs) -> Any:
    """
    快速创建环境的便捷函数
    
    Args:
        env_id: 环境ID
        data_root: 数据根目录
        max_steps: 最大步数
        **kwargs: 其他参数
        
    Returns:
        环境实例
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
    # 初始化
    init_registry()
    
    # 列出所有环境
    print("Available environments:")
    envs = list_envs()
    if envs:
        for env in envs:
            print(f"  - {env['id']}: {env['description']}")
            print(f"    Tags: {env['tags']}")
            print(f"    Task: {env['task_type']} with {env['adapter_type']}")
    else:
        print("  No environments registered yet.")
    
    # 如果有注册的环境，可以测试创建
    # if envs:
    #     env_id = envs[0]['id']
    #     print(f"\nTesting environment creation for: {env_id}")
    #     env = make(env_id)
    #     print(f"Created env: {type(env).__name__}")