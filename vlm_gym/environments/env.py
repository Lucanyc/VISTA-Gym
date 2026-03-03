
"""
VLM-Gym 容器化环境实现

提供在Docker容器中执行VLM任务的完整环境，包括：
- 容器生命周期管理
- GPU资源分配
- 任务执行和监控
- 与注册系统的集成
- 错误处理和恢复
"""

from __future__ import annotations

import os
import json
import time
import logging
import tempfile
import subprocess
import shlex
import re
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

import docker
import docker.errors
import docker.models.containers
import gymnasium as gym

from .base import BaseVLMEnv
from .utils import (
    PROCESS_DONE_MARKER_END,
    PROCESS_DONE_MARKER_START,
    NoOutputTimeoutError,
    copy_anything_from_container,
    copy_anything_to_container,
    copy_file_to_container,
    get_container,
    image_exists,
    read_with_timeout,
    read_with_timeout_pid,
)
from .registration import get_env_spec
from .task.base import BaseTask, BaseAdapter
from .spaces import Unicode

logger = logging.getLogger(__name__)

# 默认超时设置
DEFAULT_TIMEOUT = 300  # 5分钟
DEFAULT_NO_OUTPUT_TIMEOUT = 60  # 1分钟
AGENT_SHORT_ACTION_TIMEOUT = 25  # 短命令超时
AGENT_LONG_ACTION_TIMEOUT = 3600  # 训练命令超时


@dataclass
class ContainerConfig:
    """容器配置"""
    image_name: str = "vlm_gym:latest"
    container_name: Optional[str] = None
    gpu_devices: List[str] = field(default_factory=lambda: ["0"])
    memory_limit: str = "32g"
    cpu_limit: Optional[float] = None
    shm_size: str = "32g"
    network_mode: str = "bridge"
    persistent: bool = False
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    working_dir: str = "/workspace"
    user: Optional[str] = None
    privileged: bool = False


@dataclass
class VLMTaskConfig:
    """VLM任务配置"""
    task_id: str
    task_type: str
    env_id: str
    model_path: Optional[str] = None
    model_cache_dir: Optional[str] = None
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    no_output_timeout: float = DEFAULT_NO_OUTPUT_TIMEOUT
    enable_gpu_monitor: bool = True
    max_steps: int = 100


class VLMContainerEnv(BaseVLMEnv):
    """
    VLM容器化环境
    
    在Docker容器中运行VLM任务，提供完整的环境隔离和资源管理。
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        env_id: str,
        container_image: str = "vlm_gym:latest",
        gpu_devices: Optional[List[str]] = None,
        memory_limit: str = "32g",
        timeout: float = DEFAULT_TIMEOUT,
        no_output_timeout: float = DEFAULT_NO_OUTPUT_TIMEOUT,
        persistent_container: bool = False,
        enable_monitoring: bool = True,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        初始化容器化环境
        
        Args:
            env_id: 环境ID（从注册表获取）
            container_image: Docker镜像
            gpu_devices: GPU设备列表
            memory_limit: 内存限制
            timeout: 任务超时时间
            no_output_timeout: 无输出超时时间
            persistent_container: 是否使用持久容器
            enable_monitoring: 是否启用资源监控
            max_steps: 最大步数
            render_mode: 渲染模式
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        # 环境配置
        self.env_id = env_id
        self.container_image = container_image
        self.gpu_devices = gpu_devices or ["0"]
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.no_output_timeout = no_output_timeout
        self.persistent_container = persistent_container
        self.enable_monitoring = enable_monitoring
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # 获取环境规范
        try:
            self.env_spec = get_env_spec(env_id)
            self.task_class = self._get_task_class()
            self.adapter_class = self._get_adapter_class()
        except ValueError:
            logger.warning(f"Environment {env_id} not found in registry")
            self.env_spec = None
            self.task_class = None
            self.adapter_class = None
        
        # 容器相关
        self.container_type = "docker"
        self.container: Optional[subprocess.Popen] = None
        self.container_obj: Optional[docker.models.containers.Container] = None
        self.container_config: Optional[ContainerConfig] = None
        self.docker_client: Optional[docker.DockerClient] = None
        
        # 任务相关
        self.current_task: Optional[BaseTask] = None
        self.current_task_id: Optional[str] = None
        self.current_step: int = 0
        self.adapter: Optional[BaseAdapter] = None
        
        # 工作空间
        self.workspace_dir = Path("/home/agent")
        self.task_workspace = self.workspace_dir / "workspace"
        self.memory_path = self.workspace_dir / "memory.json"
        
        # 空间定义
        self.action_space = Unicode(min_length=0, max_length=10000)
        self.observation_space = Unicode(min_length=0, max_length=100000)
        
        # 初始化Docker客户端
        self._init_docker_client()
        
        # 任务脚本模板
        self.task_script_template = self._load_task_script_template()
    
    def _init_docker_client(self):
        """初始化Docker客户端"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except docker.errors.DockerException as e:
            if "connection" in str(e).lower():
                raise RuntimeError(
                    "Docker daemon is not running. Please start Docker and try again."
                ) from e
            raise RuntimeError(f"Failed to connect to Docker: {e}") from e
    
    def _get_task_class(self):
        """获取任务类"""
        if not self.env_spec:
            return None
        from .registration import _TASK_REGISTRY
        return _TASK_REGISTRY.get(self.env_spec.task_config.task_type)
    
    def _get_adapter_class(self):
        """获取适配器类"""
        if not self.env_spec:
            return None
        from .registration import _ADAPTER_REGISTRY
        return _ADAPTER_REGISTRY.get(self.env_spec.task_config.adapter_type)
    
    def _load_task_script_template(self) -> str:
        """加载任务脚本模板"""
        # 尝试加载外部脚本，如果不存在则使用内联版本
        script_path = Path(__file__).parent.parent.parent / "scripts" / "container_task_executor.py"
        if script_path.exists():
            with open(script_path, 'r') as f:
                return f.read()
        
        # 内联版本作为后备
        return '''#!/usr/bin/env python3
"""VLM Task Executor Script (Inline Version)"""
import sys
import json
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 加载任务配置
    with open('/workspace/task_config.json', 'r') as f:
        task_config = json.load(f)
    
    logger.info(f"Starting task: {task_config['task_id']}")
    logger.info(f"Task type: {task_config.get('task_type', 'unknown')}")
    
    # Parse action
    action_data = json.loads(task_config['action']) if isinstance(task_config['action'], str) else task_config['action']
    
    # Import and create agent
    from vlm_gym.agents import VLMAgent
    
    agent_config = {
        "agent": {
            "model_type": "HuggingFace",
            "model_name": action_data.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "max_new_tokens": 512,
            "temperature": 0.3,
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "trust_remote_code": True
        }
    }
    
    agent = VLMAgent(config=agent_config)
    
    # Prepare observation
    observation = {
        "image_path": action_data.get("image_path"),
        "question": action_data.get("question"),
        "choices": action_data.get("choices")
    }
    
    # Execute
    response, extra_info = agent.act(observation)
    
    # Save result
    result = {
        "task_id": task_config['task_id'],
        "status": "success",
        "prediction": response,
        "extra_info": extra_info
    }
    
    output_dir = Path('/workspace/outputs')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info("Task completed successfully")
    
except Exception as e:
    logger.error(f"Task failed: {str(e)}")
    logger.error(traceback.format_exc())
    
    result = {
        "task_id": task_config.get('task_id', 'unknown'),
        "status": "error",
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    
    output_dir = Path('/workspace/outputs')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    sys.exit(1)
'''
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            task_id: 任务ID
            options: 额外选项
            
        Returns:
            (observation, info)元组
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_task_id = task_id or f"task_{int(time.time())}"
        
        # 清理之前的容器（如果不是持久的）
        if self.container_obj and not self.persistent_container:
            self._cleanup_container()
        
        # 创建容器配置
        self.container_config = ContainerConfig(
            image_name=self.container_image,
            container_name=f"vlm_{self.current_task_id}" if not self.persistent_container else "vlm_persistent",
            gpu_devices=self.gpu_devices,
            memory_limit=self.memory_limit,
            persistent=self.persistent_container,
            environment={
                'PYTHONPATH': '/workspace',
                'PYTHONUNBUFFERED': '1',
                'CUDA_VISIBLE_DEVICES': ','.join(self.gpu_devices),
                'VLM_TASK_ID': self.current_task_id,
            },
            volumes=self._prepare_volumes()
        )
        
        # 初始化容器
        if not self.container_obj or not self.persistent_container:
            self._init_container()
        
        # 设置工作空间
        self._setup_workspace()
        
        # 初始化适配器和任务
        if self.adapter_class and self.task_class:
            adapter_config = self.env_spec.task_config.adapter_config.copy()
            self.adapter = self.adapter_class(**adapter_config)
            
            task_kwargs = self.env_spec.task_config.task_kwargs.copy()
            self.current_task = self.task_class(
                task_id=self.current_task_id,
                adapter=self.adapter,
                **task_kwargs
            )
            
            # 设置任务
            task_goal, task_info = self.current_task.setup()
        else:
            task_goal = "No task configured"
            task_info = {}
        
        # 构建观察
        observation = {
            'task_id': self.current_task_id,
            'task_goal': task_goal,
            'container_status': 'ready',
            'gpu_devices': self.gpu_devices
        }
        
        info = {
            'container_name': self.container_config.container_name,
            'task_info': task_info
        }
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: 智能体的动作
            
        Returns:
            (observation, reward, terminated, truncated, info)元组
        """
        if not self.container_obj:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.current_step += 1
        
        # 检查是否超过最大步数
        truncated = self.current_step >= self.max_steps
        
        # 准备任务配置
        task_config_data = {
            'task_id': self.current_task_id,
            'task_type': self.env_spec.task_config.task_type if self.env_spec else 'unknown',
            'env_id': self.env_id,
            'action': action,
            'step': self.current_step
        }
        
        # 执行动作
        try:
            # 写入任务配置
            self._write_to_container('/workspace/task_config.json', json.dumps(task_config_data))
            
            # 执行任务脚本
            output = self._communicate(
                'python /workspace/task_executor.py',
                timeout_duration=self.timeout,
                no_output_timeout_duration=self.no_output_timeout
            )
            
            # 获取结果
            result = self._retrieve_results()
            
            # 检查任务成功
            if self.current_task and result:
                # 从结果中获取预测
                prediction = result.get("prediction", "")
                
                # 使用任务的check_success方法
                success, feedback = self.current_task.check_success(prediction)
                reward = 1.0 if success else 0.0
                terminated = True  # VQA任务通常一步完成
            else:
                reward = 0.0
                terminated = False
                feedback = "No result retrieved"
                success = False
            
            # 构建观察
            observation = {
                'output': output[-1000:] if output else '',
                'feedback': feedback,
                'step': self.current_step
            }
            
            # 获取资源统计
            stats = self._get_container_stats() if self.enable_monitoring else {}
            
            info = {
                'result': result,
                'stats': stats,
                'full_output': output if len(output) < 10000 else output[-10000:],
                'success': success
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            observation = {
                'error': str(e),
                'step': self.current_step
            }
            reward = 0.0
            terminated = True
            truncated = False
            info = {'error': str(e), 'success': False}
        
        return observation, reward, terminated, truncated, info
    
    def _init_container(self):
        """初始化容器"""
        if self.container_obj:
            return
        
        # 检查镜像是否存在
        if not image_exists(self.container_config.image_name):
            raise RuntimeError(f"Image {self.container_config.image_name} not found")
        
        # 获取容器
        self.container, self.parent_pids = get_container(
            self.container_config.container_name,
            self.container_config.image_name,
            container_type=self.container_type,
            persistent=self.container_config.persistent,
            devices=self.gpu_devices
        )
        
        # 获取容器对象
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                self.container_obj = self.docker_client.containers.get(
                    self.container_config.container_name
                )
                break
            except docker.errors.NotFound:
                logger.debug("Waiting for container...")
                time.sleep(1)
        else:
            raise RuntimeError("Failed to get container object")
        
        logger.info(f"Container {self.container_config.container_name} initialized")
        
        # 写入任务执行脚本
        self._write_to_container('/workspace/task_executor.py', self.task_script_template)
    
    def _setup_workspace(self):
        """设置工作空间"""
        self._communicate(f"mkdir -p {self.task_workspace}")
        self._communicate(f"mkdir -p {self.task_workspace}/outputs")
        
        # 复制数据文件（如果有）
        if self.current_task and hasattr(self.current_task, 'task_data'):
            task_data = self.current_task.task_data
            if 'image_path' in task_data and os.path.exists(task_data['image_path']):
                copy_file_to_container(
                    self.container_obj,
                    self.container_type,
                    task_data['image_path'],
                    f"{self.task_workspace}/image.jpg"
                )
    
    def _prepare_volumes(self) -> List[Dict[str, Any]]:
        """准备挂载卷"""
        volumes = []
        
        # 模型缓存目录
        model_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        volumes.append({
            'source': model_cache,
            'target': '/root/.cache/huggingface',
            'type': 'bind',
            'read_only': False
        })
        
        # 数据目录
        if hasattr(self, 'env_spec') and self.env_spec:
            data_root = self.env_spec.env_config.dataset_path
            if os.path.exists(data_root):
                volumes.append({
                    'source': data_root,
                    'target': '/workspace/data',
                    'type': 'bind',
                    'read_only': True
                })
        
        return volumes
    
    def _communicate(
        self,
        input: str,
        timeout_duration: float = AGENT_SHORT_ACTION_TIMEOUT,
        no_output_timeout_duration: Optional[float] = None
    ) -> str:
        """与容器通信"""
        if no_output_timeout_duration is None:
            no_output_timeout_duration = timeout_duration
        
        if not self.container:
            raise RuntimeError("Container not initialized")
        
        # 添加命令结束标记
        command_suffix = (
            f'EXITSTATUS="$?"; sleep 0.01; echo {PROCESS_DONE_MARKER_START}$EXITSTATUS{PROCESS_DONE_MARKER_END}\n'
        )
        
        try:
            cmd = input if input.endswith("\n") else input + "\n"
            cmd += command_suffix
            os.write(self.container.stdin.fileno(), cmd.encode())
            time.sleep(0.1)
            self.container.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("Failed to communicate with container")
        
        try:
            buffer, exit_code = read_with_timeout(
                self.container, timeout_duration, no_output_timeout_duration
            )
        except NoOutputTimeoutError as e:
            raise NoOutputTimeoutError(
                f"No output for {no_output_timeout_duration}s",
                buffer if 'buffer' in locals() else ""
            )
        except TimeoutError as e:
            raise TimeoutError(
                f"Command timed out after {timeout_duration}s",
                buffer if 'buffer' in locals() else ""
            )
        
        return buffer
    
    def _write_to_container(self, container_path: str, content: str):
        """写入文件到容器"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            copy_file_to_container(
                self.container_obj,
                self.container_type,
                tmp.name,
                container_path
            )
            os.unlink(tmp.name)
    
    def _retrieve_results(self) -> Optional[Dict[str, Any]]:
        """从容器获取结果"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name
            
            copy_anything_from_container(
                self.container_obj,
                self.container_type,
                tmp_path,
                '/workspace/outputs/result.json'
            )
            
            with open(tmp_path, 'r') as f:
                result = json.load(f)
            
            os.unlink(tmp_path)
            return result
            
        except Exception as e:
            logger.warning(f"Failed to retrieve results: {e}")
            return None
    
    def _get_container_stats(self) -> Dict[str, Any]:
        """获取容器统计信息"""
        if not self.container_obj:
            return {}
        
        try:
            stats = self.container_obj.stats(stream=False)
            
            # CPU使用率
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
            
            # 内存使用
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "memory_limit_mb": memory_limit / (1024 * 1024),
                "memory_percent": memory_percent
            }
        except Exception as e:
            logger.warning(f"Failed to get container stats: {e}")
            return {}
    
    def _cleanup_container(self):
        """清理容器"""
        if self.container_obj:
            try:
                self.container_obj.stop(timeout=10)
                if not self.persistent_container:
                    self.container_obj.remove(force=True)
                logger.info(f"Container {self.container_obj.name} cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")
        
        self.container = None
        self.container_obj = None
    
    def close(self):
        """关闭环境"""
        self._cleanup_container()
        if self.docker_client:
            self.docker_client.close()
        super().close()
    
    def render(self) -> Optional[Any]:
        """渲染环境状态"""
        if self.render_mode == "human":
            if self.container_obj:
                stats = self._get_container_stats()
                print(f"\n=== Container Status ===")
                print(f"Name: {self.container_obj.name}")
                print(f"Status: {self.container_obj.status}")
                print(f"CPU: {stats.get('cpu_percent', 0):.1f}%")
                print(f"Memory: {stats.get('memory_usage_mb', 0):.1f}MB / {stats.get('memory_limit_mb', 0):.1f}MB")
                print(f"Step: {self.current_step}/{self.max_steps}")
                print("=" * 20)
        
        return None


# 注册容器化环境的便捷函数
def register_containerized_env(
    env_id: str,
    base_env_id: str,
    container_image: str = "vlm_gym:latest",
    **kwargs
):
    """
    注册容器化环境
    
    Args:
        env_id: 新环境ID
        base_env_id: 基础环境ID
        container_image: Docker镜像
        **kwargs: 其他配置
    """
    from .registration import register_env, get_env_spec
    
    # 获取基础环境规范
    base_spec = get_env_spec(base_env_id)
    
    # 注册容器化版本
    register_env(
        env_id=f"{env_id}-container",
        env_class=VLMContainerEnv,
        task_type=base_spec.task_config.task_type,
        adapter_type=base_spec.task_config.adapter_type,
        adapter_config=base_spec.task_config.adapter_config,
        env_config={
            'env_id': base_env_id,
            'container_image': container_image,
            **kwargs
        },
        description=f"Containerized version of {base_spec.description}",
        tags=base_spec.tags + ["containerized"],
        version="v1"
    )


if __name__ == "__main__":
    # 使用示例
    from .registration import init_registry, make
    
    # 初始化注册表
    init_registry()
    
    # 注册容器化版本
    register_containerized_env(
        "chartqa",
        "chartqa",
        container_image="vlm_gym:latest",
        gpu_devices=["0"],
        memory_limit="16g"
    )
    
    # 创建环境
    env = make("chartqa-container")
    
    # 运行任务
    obs, info = env.reset(task_id="sample_0")
    print(f"Reset observation: {obs}")
    
    # 执行动作
    action = "analyze_chart"
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: reward={reward}, terminated={terminated}")
    
    # 清理
    env.close()