"""
VLM-Gym Containerized Environment Implementation

Provides a complete environment for executing VLM tasks in Docker containers, including:
- Container lifecycle management
- GPU resource allocation
- Task execution and monitoring
- Integration with the registration system
- Error handling and recovery
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

# Default timeout settings
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_NO_OUTPUT_TIMEOUT = 60  # 1 minute
AGENT_SHORT_ACTION_TIMEOUT = 25  # Short command timeout
AGENT_LONG_ACTION_TIMEOUT = 3600  # Training command timeout


@dataclass
class ContainerConfig:
    """Container configuration"""
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
    """VLM task configuration"""
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
    VLM Containerized Environment
    
    Runs VLM tasks in Docker containers, providing complete environment isolation
    and resource management.
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
        Initialize the containerized environment
        
        Args:
            env_id: Environment ID (retrieved from the registry)
            container_image: Docker image
            gpu_devices: List of GPU devices
            memory_limit: Memory limit
            timeout: Task timeout duration
            no_output_timeout: No-output timeout duration
            persistent_container: Whether to use a persistent container
            enable_monitoring: Whether to enable resource monitoring
            max_steps: Maximum number of steps
            render_mode: Render mode
            **kwargs: Arguments passed to the parent class
        """
        super().__init__(**kwargs)
        
        # Environment configuration
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
        
        # Get environment specification
        try:
            self.env_spec = get_env_spec(env_id)
            self.task_class = self._get_task_class()
            self.adapter_class = self._get_adapter_class()
        except ValueError:
            logger.warning(f"Environment {env_id} not found in registry")
            self.env_spec = None
            self.task_class = None
            self.adapter_class = None
        
        # Container-related
        self.container_type = "docker"
        self.container: Optional[subprocess.Popen] = None
        self.container_obj: Optional[docker.models.containers.Container] = None
        self.container_config: Optional[ContainerConfig] = None
        self.docker_client: Optional[docker.DockerClient] = None
        
        # Task-related
        self.current_task: Optional[BaseTask] = None
        self.current_task_id: Optional[str] = None
        self.current_step: int = 0
        self.adapter: Optional[BaseAdapter] = None
        
        # Workspace
        self.workspace_dir = Path("/home/agent")
        self.task_workspace = self.workspace_dir / "workspace"
        self.memory_path = self.workspace_dir / "memory.json"
        
        # Space definitions
        self.action_space = Unicode(min_length=0, max_length=10000)
        self.observation_space = Unicode(min_length=0, max_length=100000)
        
        # Initialize Docker client
        self._init_docker_client()
        
        # Task script template
        self.task_script_template = self._load_task_script_template()
    
    def _init_docker_client(self):
        """Initialize Docker client"""
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
        """Get the task class"""
        if not self.env_spec:
            return None
        from .registration import _TASK_REGISTRY
        return _TASK_REGISTRY.get(self.env_spec.task_config.task_type)
    
    def _get_adapter_class(self):
        """Get the adapter class"""
        if not self.env_spec:
            return None
        from .registration import _ADAPTER_REGISTRY
        return _ADAPTER_REGISTRY.get(self.env_spec.task_config.adapter_type)
    
    def _load_task_script_template(self) -> str:
        """Load the task script template"""
        # Try to load an external script; fall back to the inline version if not found
        script_path = Path(__file__).parent.parent.parent / "scripts" / "container_task_executor.py"
        if script_path.exists():
            with open(script_path, 'r') as f:
                return f.read()
        
        # Inline version as fallback
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
    # Load task configuration
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
        Reset the environment
        
        Args:
            seed: Random seed
            task_id: Task ID
            options: Additional options
            
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_task_id = task_id or f"task_{int(time.time())}"
        
        # Clean up the previous container (if not persistent)
        if self.container_obj and not self.persistent_container:
            self._cleanup_container()
        
        # Create container configuration
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
        
        # Initialize container
        if not self.container_obj or not self.persistent_container:
            self._init_container()
        
        # Set up workspace
        self._setup_workspace()
        
        # Initialize adapter and task
        if self.adapter_class and self.task_class:
            adapter_config = self.env_spec.task_config.adapter_config.copy()
            self.adapter = self.adapter_class(**adapter_config)
            
            task_kwargs = self.env_spec.task_config.task_kwargs.copy()
            self.current_task = self.task_class(
                task_id=self.current_task_id,
                adapter=self.adapter,
                **task_kwargs
            )
            
            # Set up task
            task_goal, task_info = self.current_task.setup()
        else:
            task_goal = "No task configured"
            task_info = {}
        
        # Build observation
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
        Execute one step
        
        Args:
            action: The agent's action
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if not self.container_obj:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.current_step += 1
        
        # Check if maximum steps exceeded
        truncated = self.current_step >= self.max_steps
        
        # Prepare task configuration
        task_config_data = {
            'task_id': self.current_task_id,
            'task_type': self.env_spec.task_config.task_type if self.env_spec else 'unknown',
            'env_id': self.env_id,
            'action': action,
            'step': self.current_step
        }
        
        # Execute action
        try:
            # Write task configuration
            self._write_to_container('/workspace/task_config.json', json.dumps(task_config_data))
            
            # Execute task script
            output = self._communicate(
                'python /workspace/task_executor.py',
                timeout_duration=self.timeout,
                no_output_timeout_duration=self.no_output_timeout
            )
            
            # Retrieve results
            result = self._retrieve_results()
            
            # Check task success
            if self.current_task and result:
                # Get prediction from results
                prediction = result.get("prediction", "")
                
                # Use the task's check_success method
                success, feedback = self.current_task.check_success(prediction)
                reward = 1.0 if success else 0.0
                terminated = True  # VQA tasks typically complete in one step
            else:
                reward = 0.0
                terminated = False
                feedback = "No result retrieved"
                success = False
            
            # Build observation
            observation = {
                'output': output[-1000:] if output else '',
                'feedback': feedback,
                'step': self.current_step
            }
            
            # Get resource statistics
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
        """Initialize the container"""
        if self.container_obj:
            return
        
        # Check if the image exists
        if not image_exists(self.container_config.image_name):
            raise RuntimeError(f"Image {self.container_config.image_name} not found")
        
        # Get container
        self.container, self.parent_pids = get_container(
            self.container_config.container_name,
            self.container_config.image_name,
            container_type=self.container_type,
            persistent=self.container_config.persistent,
            devices=self.gpu_devices
        )
        
        # Get container object
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
        
        # Write the task executor script
        self._write_to_container('/workspace/task_executor.py', self.task_script_template)
    
    def _setup_workspace(self):
        """Set up the workspace"""
        self._communicate(f"mkdir -p {self.task_workspace}")
        self._communicate(f"mkdir -p {self.task_workspace}/outputs")
        
        # Copy data files (if any)
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
        """Prepare mount volumes"""
        volumes = []
        
        # Model cache directory
        model_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        volumes.append({
            'source': model_cache,
            'target': '/root/.cache/huggingface',
            'type': 'bind',
            'read_only': False
        })
        
        # Data directory
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
        """Communicate with the container"""
        if no_output_timeout_duration is None:
            no_output_timeout_duration = timeout_duration
        
        if not self.container:
            raise RuntimeError("Container not initialized")
        
        # Add command end marker
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
        """Write a file to the container"""
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
        """Retrieve results from the container"""
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
        """Get container statistics"""
        if not self.container_obj:
            return {}
        
        try:
            stats = self.container_obj.stats(stream=False)
            
            # CPU usage
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
            
            # Memory usage
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
        """Clean up the container"""
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
        """Close the environment"""
        self._cleanup_container()
        if self.docker_client:
            self.docker_client.close()
        super().close()
    
    def render(self) -> Optional[Any]:
        """Render the environment state"""
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


# Convenience function for registering containerized environments
def register_containerized_env(
    env_id: str,
    base_env_id: str,
    container_image: str = "vlm_gym:latest",
    **kwargs
):
    """
    Register a containerized environment
    
    Args:
        env_id: New environment ID
        base_env_id: Base environment ID
        container_image: Docker image
        **kwargs: Other configuration options
    """
    from .registration import register_env, get_env_spec
    
    # Get the base environment specification
    base_spec = get_env_spec(base_env_id)
    
    # Register the containerized version
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
    # Usage example
    from .registration import init_registry, make
    
    # Initialize registry
    init_registry()
    
    # Register containerized version
    register_containerized_env(
        "chartqa",
        "chartqa",
        container_image="vlm_gym:latest",
        gpu_devices=["0"],
        memory_limit="16g"
    )
    
    # Create environment
    env = make("chartqa-container")
    
    # Run task
    obs, info = env.reset(task_id="sample_0")
    print(f"Reset observation: {obs}")
    
    # Execute action
    action = "analyze_chart"
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result: reward={reward}, terminated={terminated}")
    
    # Clean up
    env.close()
