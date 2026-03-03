# vlm_gym/utils/config_loader.py

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from typing import List, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_env_config(config_path: str, experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """
    加载环境配置，支持基础配置和实验特定配置
    
    Args:
        config_path: 配置文件路径
        experiment_name: 可选的实验名称（用于加载特定实验配置）
    
    Returns:
        环境配置字典
    """
    config = load_config(config_path)
    
    # 获取基础环境配置
    base_config = config.get('environment', {}).copy()
    
    # 如果指定了实验名称，合并实验特定的配置
    if experiment_name and 'experiments' in config:
        if experiment_name in config['experiments']:
            exp_config = config['experiments'][experiment_name]
            # 深度合并配置
            base_config = merge_configs(base_config, exp_config)
        else:
            available_experiments = list(config['experiments'].keys())
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available_experiments}"
            )
    
    # 处理路径（确保是绝对路径）
    base_config = resolve_paths(base_config)
    
    return base_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
    
    Returns:
        合并后的配置
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并字典
            result[key] = merge_configs(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    
    return result


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析配置中的路径，将相对路径转换为绝对路径
    
    Args:
        config: 配置字典
    
    Returns:
        处理后的配置
    """
    # 需要处理的路径键
    path_keys = ['dataset_path', 'model_path', 'model_config', 'checkpoint_path']
    
    def _resolve_recursive(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in path_keys and isinstance(v, str):
                    # 处理路径
                    path = Path(v)
                    if not path.is_absolute():
                        # 相对路径 - 可以相对于项目根目录或当前工作目录
                        d[k] = str(path.resolve())
                elif isinstance(v, dict):
                    _resolve_recursive(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            _resolve_recursive(item)
    
    result = config.copy()
    _resolve_recursive(result)
    return result


def create_env_from_config(
    config_path: str, 
    experiment_name: Optional[str] = None,
    **kwargs
) -> 'VisionQAEnv':
    """从配置文件创建环境实例"""
    from vlm_gym.environments.vision_qa_env import VisionQAEnv
    
    # 加载配置
    env_config = load_env_config(config_path, experiment_name)
    
    # 应用额外的参数
    env_config.update(kwargs)
    
    # ⭐ 添加调试输出
    print(f"\n[DEBUG create_env_from_config]")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - enable_grounding_dino: {env_config.get('enable_grounding_dino')}")
    print(f"  - grounding_dino_config: {env_config.get('grounding_dino_config')}")
    
    # 创建环境
    return VisionQAEnv(**env_config)



def list_experiments(config_path: str) -> List[str]:
    """
    列出配置文件中定义的所有实验
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        实验名称列表
    """
    config = load_config(config_path)
    return list(config.get('experiments', {}).keys())


# 便捷函数
def get_default_env_config_path() -> Path:
    """获取默认的环境配置文件路径"""
    from vlm_gym import configs
    return Path(configs.__file__).parent / "env_config.yaml"