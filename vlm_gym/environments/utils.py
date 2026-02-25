# vlm_gym/environments/container_utils.py
"""
VLM Gym Container Management Utilities

Provides functionality for containerized task execution, including:
- Docker container lifecycle management
- File and data transfer
- Process monitoring and timeout handling
- GPU resource management
- VLM-specific optimizations (large file handling, model caching, etc.)
"""

from __future__ import annotations

import os
import time
import json
import shlex
import docker
import logging
import tempfile
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from subprocess import PIPE, STDOUT
from concurrent.futures import ThreadPoolExecutor
import threading

import docker.errors
import docker.types
from docker.models.containers import Container

logger = logging.getLogger(__name__)


DOCKER_START_UP_DELAY = float(os.getenv("VLM_GYM_DOCKER_STARTUP_DELAY", "2"))
DEFAULT_TIMEOUT = 300  
DEFAULT_NO_OUTPUT_TIMEOUT = 60  
PROCESS_DONE_MARKER = "<<<VLM_TASK_COMPLETE>>>"


class ContainerError(Exception):

    pass


class TimeoutError(ContainerError):

    pass


class NoOutputTimeoutError(TimeoutError):

    pass


@dataclass
class ContainerConfig:

    image_name: str
    container_name: str
    gpu_devices: List[str] = field(default_factory=lambda: ["0"])
    memory_limit: Optional[str] = None  # e.g., "16g"
    cpu_limit: Optional[float] = None   # e.g., 4.0
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

    task_id: str
    task_type: str
    model_path: Optional[str] = None
    model_cache_dir: Optional[str] = None
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    no_output_timeout: float = DEFAULT_NO_OUTPUT_TIMEOUT
    enable_gpu_monitor: bool = True
    

class ContainerManager:

    
    def __init__(self, container_type: str = "docker"):
        """
        Args:
            container_type: Container type, "docker" or "podman"
        """
        self.container_type = container_type
        self.client = None
        self._init_client()
        self.active_containers: Dict[str, Container] = {}
        self._lock = threading.Lock()
        
    def _init_client(self):

        try:
            self.client = docker.from_env()

            self.client.ping()
        except docker.errors.DockerException as e:
            if any(x in str(e).lower() for x in ["connection aborted", "connection refused"]):
                raise ContainerError(
                    "Docker daemon is not running. Please start Docker and try again."
                ) from e
            raise ContainerError(f"Failed to connect to Docker: {e}") from e
    
    def create_vlm_container(
        self,
        config: ContainerConfig,
        task_config: VLMTaskConfig
    ) -> Container:
        """
        Create a VLM task container
        
        Args:
            config: Container configuration
            task_config: VLM task configuration
            
        Returns:
            Container: Docker container object
        """

        # Check if image exists
        if not self.image_exists(config.image_name):
            raise ContainerError(f"Image {config.image_name} not found")
        
        # Build container parameters
        container_kwargs = {
            "image": config.image_name,
            "name": config.container_name,
            "working_dir": config.working_dir,
            "network_mode": config.network_mode,
            "shm_size": config.shm_size,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "auto_remove": not config.persistent,
        }
        
        # GPU configuration
        if config.gpu_devices and config.gpu_devices != ["cpu"]:
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=config.gpu_devices,
                    capabilities=[["gpu"]]
                )
            ]
            container_kwargs["device_requests"] = device_requests
        
        # Resource limits
        if config.memory_limit:
            container_kwargs["mem_limit"] = config.memory_limit
        if config.cpu_limit:
            container_kwargs["cpu_quota"] = int(config.cpu_limit * 100000)
            container_kwargs["cpu_period"] = 100000
        
        # Environment variables
        env = config.environment.copy()
        env.update({
            "VLM_TASK_ID": task_config.task_id,
            "VLM_TASK_TYPE": task_config.task_type,
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(config.gpu_devices),
        })
        container_kwargs["environment"] = env
        
        # Mount volumes
        mounts = self._prepare_mounts(config, task_config)
        if mounts:
            container_kwargs["mounts"] = mounts
        
        # Create container
        try:
            container = self.client.containers.create(**container_kwargs)
            with self._lock:
                self.active_containers[config.container_name] = container
            logger.info(f"Created container {config.container_name}")
            return container
        except Exception as e:
            raise ContainerError(f"Failed to create container: {e}") from e
    
    def _prepare_mounts(
        self,
        config: ContainerConfig,
        task_config: VLMTaskConfig
    ) -> List[docker.types.Mount]:

        mounts = []
        
        # Default mount points
        default_mounts = [
            # Model cache
            {
                "source": task_config.model_cache_dir or os.path.expanduser("~/.cache/huggingface"),
                "target": "/root/.cache/huggingface",
                "type": "bind",
                "read_only": False
            },
            # Data directory
            {
                "source": task_config.data_dir or "./data",
                "target": "/workspace/data",
                "type": "bind",
                "read_only": True
            },
            # Output directory
            {
                "source": task_config.output_dir or "./outputs",
                "target": "/workspace/outputs",
                "type": "bind",
                "read_only": False
            }
        ]
        
        # Merge user-specified volumes
        all_mounts = default_mounts + config.volumes
        
        # Create mount objects
        for mount_config in all_mounts:
            source_path = Path(mount_config["source"]).absolute()
            if not source_path.exists() and not mount_config.get("read_only", False):
                source_path.mkdir(parents=True, exist_ok=True)
            
            mount = docker.types.Mount(
                target=mount_config["target"],
                source=str(source_path),
                type=mount_config.get("type", "bind"),
                read_only=mount_config.get("read_only", False)
            )
            mounts.append(mount)
        
        return mounts
    
    def start_container(self, container: Container) -> subprocess.Popen:
        """
        Start the container and return an interactive process
        
        Args:
            container: Docker container object
            
        Returns:
            subprocess.Popen: Process for interacting with the container
        """
        # Start the container
        container.start()
        time.sleep(DOCKER_START_UP_DELAY)
        
        # Create an interactive process
        exec_cmd = [
            self.container_type,
            "exec",
            "-i",
            container.name,
            "/bin/bash",
            "-l"
        ]
        
        process = subprocess.Popen(
            exec_cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1
        )
        
        return process
    
    def execute_in_container(
        self,
        container: Container,
        command: Union[str, List[str]],
        timeout: float = DEFAULT_TIMEOUT,
        no_output_timeout: float = DEFAULT_NO_OUTPUT_TIMEOUT,
        stream: bool = False
    ) -> Union[str, Tuple[str, int]]:
        """
        Execute a command in the container
        
        Args:
            container: Docker container object
            command: Command to execute
            timeout: Total timeout duration
            no_output_timeout: No-output timeout duration
            stream: Whether to use streaming output
            
        Returns:
            If stream=False: (output, exit_code)
            If stream=True: Returns a generator
        """
        if isinstance(command, list):
            command = shlex.join(command)
        
        # Wrap command with completion marker
        wrapped_command = f"{command} && echo '{PROCESS_DONE_MARKER}:0' || echo '{PROCESS_DONE_MARKER}:$?'"
        
        try:
            if stream:
                return self._execute_stream(container, wrapped_command, timeout, no_output_timeout)
            else:
                return self._execute_wait(container, wrapped_command, timeout, no_output_timeout)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def _execute_wait(
        self,
        container: Container,
        command: str,
        timeout: float,
        no_output_timeout: float
    ) -> Tuple[str, int]:

        exec_result = container.exec_run(
            command,
            stdout=True,
            stderr=True,
            stream=True,
            demux=True
        )
        
        output_lines = []
        exit_code = -1
        start_time = time.time()
        last_output_time = start_time
        
        for stdout, stderr in exec_result.output:
            current_time = time.time()
            
            # Check total timeout
            if current_time - start_time > timeout:
                raise TimeoutError(f"Command timed out after {timeout}s")
            
            if current_time - last_output_time > no_output_timeout:
                raise NoOutputTimeoutError(
                    f"No output for {no_output_timeout}s",
                    "\n".join(output_lines)
                )
            
            # Process output
            if stdout:
                line = stdout.decode("utf-8", errors="replace")
                output_lines.append(line)
                last_output_time = current_time
                
                # Check for completion marker
                if PROCESS_DONE_MARKER in line:
                    exit_code = int(line.split(":")[-1])
                    break
            
            if stderr:
                line = stderr.decode("utf-8", errors="replace")
                output_lines.append(f"STDERR: {line}")
                last_output_time = current_time
        
        output = "".join(output_lines)
        # Remove completion marker
        output = output.replace(f"{PROCESS_DONE_MARKER}:{exit_code}", "").strip()
        
        return output, exit_code
    
    def _execute_stream(
        self,
        container: Container,
        command: str,
        timeout: float,
        no_output_timeout: float
    ):

        # Streaming execution
        exec_result = container.exec_run(
            command,
            stdout=True,
            stderr=True,
            stream=True,
            demux=True
        )
        
        start_time = time.time()
        last_output_time = start_time
        
        for stdout, stderr in exec_result.output:
            current_time = time.time()
            
            # Check timeouts
            if current_time - start_time > timeout:
                raise TimeoutError(f"Command timed out after {timeout}s")
            
            if current_time - last_output_time > no_output_timeout:
                raise NoOutputTimeoutError(f"No output for {no_output_timeout}s")
            
            if stdout:
                last_output_time = current_time
                yield stdout.decode("utf-8", errors="replace")
            
            if stderr:
                last_output_time = current_time
                yield f"STDERR: {stderr.decode('utf-8', errors='replace')}"
    
    def copy_to_container(
        self,
        container: Container,
        source: Union[str, bytes],
        target: str
    ):
        """
        Copy a file or data to the container
        
        Args:
            container: Docker container object
            source: Source file path or byte data
            target: Target path inside the container
        """
        if isinstance(source, bytes):
            # Handle byte data
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(source)
                tmp.flush()
                self._copy_file_to_container(container, tmp.name, target)
                os.unlink(tmp.name)
        elif isinstance(source, str):
            if os.path.exists(source):
                # Copy file
                self._copy_file_to_container(container, source, target)
            else:
                # Treat as text content
                self.copy_to_container(container, source.encode("utf-8"), target)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _copy_file_to_container(self, container: Container, host_path: str, container_path: str):
        """Copy a file to the container"""
        cmd = [
            self.container_type,
            "cp",
            host_path,
            f"{container.id}:{container_path}"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Copied {host_path} to {container.name}:{container_path}")
        except subprocess.CalledProcessError as e:
            raise ContainerError(
                f"Failed to copy file: {e.stderr.decode()}"
            ) from e
    
    def copy_from_container(
        self,
        container: Container,
        source: str,
        target: str
    ):
        """
        Copy a file from the container
        
        Args:
            container: Docker container object
            source: Source path inside the container
            target: Host target path
        """
        cmd = [
            self.container_type,
            "cp",
            f"{container.id}:{source}",
            target
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.debug(f"Copied {container.name}:{source} to {target}")
        except subprocess.CalledProcessError as e:
            raise ContainerError(
                f"Failed to copy file: {e.stderr.decode()}"
            ) from e
    
    def get_container_stats(self, container: Container) -> Dict[str, Any]:
        """
        Get container resource usage statistics
        
        Args:
            container: Docker container object
            
        Returns:
            Resource usage statistics
        """
        stats = container.stats(stream=False)
        
        # Parse CPU usage
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                   stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                      stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
        
        # Parse memory usage
        memory_usage = stats["memory_stats"]["usage"]
        memory_limit = stats["memory_stats"]["limit"]
        memory_percent = (memory_usage / memory_limit) * 100.0
        
        return {
            "cpu_percent": cpu_percent,
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "memory_limit_mb": memory_limit / (1024 * 1024),
            "memory_percent": memory_percent,
            "network_rx_mb": stats["networks"]["eth0"]["rx_bytes"] / (1024 * 1024)
                            if "networks" in stats and "eth0" in stats["networks"] else 0,
            "network_tx_mb": stats["networks"]["eth0"]["tx_bytes"] / (1024 * 1024)
                            if "networks" in stats and "eth0" in stats["networks"] else 0,
        }
    
    def monitor_gpu(self, container: Container) -> Dict[str, Any]:
        """
        Monitor container GPU usage
        
        Args:
            container: Docker container object
            
        Returns:
            GPU usage information
        """
        try:
            output, _ = self.execute_in_container(
                container,
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
                timeout=10
            )
            
            gpu_info = []
            for line in output.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu_info.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_used_mb": float(parts[2]),
                            "memory_total_mb": float(parts[3]),
                            "utilization_percent": float(parts[4])
                        })
            
            return {"gpus": gpu_info}
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {"gpus": []}
    
    def stop_container(self, container: Container, timeout: int = 10):
        """
        Stop the container
        
        Args:
            container: Docker container object
            timeout: Stop timeout duration
        """
        try:
            container.stop(timeout=timeout)
            logger.info(f"Stopped container {container.name}")
        except Exception as e:
            logger.warning(f"Failed to stop container gracefully: {e}")
            try:
                container.kill()
                logger.info(f"Killed container {container.name}")
            except Exception as e2:
                logger.error(f"Failed to kill container: {e2}")
        
        with self._lock:
            self.active_containers.pop(container.name, None)
    
    def cleanup_container(self, container: Container):
        """
        Clean up the container
        
        Args:
            container: Docker container object
        """
        self.stop_container(container)
        try:
            container.remove(force=True)
            logger.info(f"Removed container {container.name}")
        except Exception as e:
            logger.warning(f"Failed to remove container: {e}")
    
    def cleanup_all(self):
        """Clean up all active containers"""
        with self._lock:
            containers = list(self.active_containers.values())
        
        for container in containers:
            self.cleanup_container(container)
    
    def image_exists(self, image_name: str) -> bool:
        """Check if an image exists"""
        try:
            self.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception as e:
            logger.warning(f"Failed to check image: {e}")
            return False
    
    def pull_image(self, image_name: str, progress_callback: Optional[Callable] = None):
        """
        Pull a Docker image
        
        Args:
            image_name: Image name
            progress_callback: Progress callback function
        """
        try:
            logger.info(f"Pulling image {image_name}")
            for line in self.client.api.pull(image_name, stream=True, decode=True):
                if progress_callback:
                    progress_callback(line)
                if "status" in line:
                    logger.debug(f"Pull status: {line['status']}")
        except Exception as e:
            raise ContainerError(f"Failed to pull image: {e}") from e


class VLMContainerExecutor:
    """VLM Container Executor - High-level interface"""
    
    def __init__(self, manager: Optional[ContainerManager] = None):
        """
        Initialize the executor
        
        Args:
            manager: Container manager instance
        """
        self.manager = manager or ContainerManager()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def run_vlm_task(
        self,
        task_config: VLMTaskConfig,
        container_config: ContainerConfig,
        task_script: str,
        cleanup: bool = True
    ) -> Dict[str, Any]:
        """
        Run a VLM task
        
        Args:
            task_config: VLM task configuration
            container_config: Container configuration
            task_script: Task script to execute
            cleanup: Whether to clean up the container after completion
            
        Returns:
            Task execution result
        """
        container = None
        result = {
            "task_id": task_config.task_id,
            "status": "failed",
            "output": "",
            "exit_code": -1,
            "stats": {},
            "error": None
        }
        
        try:
            # Create container
            container = self.manager.create_vlm_container(container_config, task_config)
            
            # Start container
            self.manager.start_container(container)
            
            # Copy task script
            script_path = "/workspace/task_script.py"
            self.manager.copy_to_container(container, task_script, script_path)
            
            # Execute task
            logger.info(f"Executing task {task_config.task_id}")
            output, exit_code = self.manager.execute_in_container(
                container,
                f"python {script_path}",
                timeout=task_config.timeout,
                no_output_timeout=task_config.no_output_timeout
            )
            
            result["output"] = output
            result["exit_code"] = exit_code
            result["status"] = "success" if exit_code == 0 else "failed"
            
            # Get statistics
            result["stats"] = self.manager.get_container_stats(container)
            if task_config.enable_gpu_monitor:
                result["gpu_stats"] = self.manager.monitor_gpu(container)
            
            # Copy output files
            output_files = ["/workspace/outputs/result.json", "/workspace/outputs/log.txt"]
            for file_path in output_files:
                try:
                    local_path = f"{task_config.output_dir}/{task_config.task_id}_{Path(file_path).name}"
                    self.manager.copy_from_container(container, file_path, local_path)
                except Exception as e:
                    logger.debug(f"Failed to copy {file_path}: {e}")
            
        except TimeoutError as e:
            result["status"] = "timeout"
            result["error"] = str(e)
            logger.error(f"Task {task_config.task_id} timed out: {e}")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Task {task_config.task_id} failed: {e}")
        finally:
            if container and cleanup:
                self.manager.cleanup_container(container)
        
        return result
    
    def run_vlm_tasks_parallel(
        self,
        tasks: List[Tuple[VLMTaskConfig, ContainerConfig, str]],
        max_parallel: int = 4,
        cleanup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run multiple VLM tasks in parallel
        
        Args:
            tasks: Task list, each element is (task_config, container_config, task_script)
            max_parallel: Maximum number of parallel tasks
            cleanup: Whether to clean up containers
            
        Returns:
            List of task results
        """
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = []
            for task_config, container_config, task_script in tasks:
                future = executor.submit(
                    self.run_vlm_task,
                    task_config,
                    container_config,
                    task_script,
                    cleanup
                )
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    results.append({
                        "status": "error",
                        "error": str(e)
                    })
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.cleanup_all()
        self.executor.shutdown()


# Convenience functions
def create_vlm_container_config(
    image_name: str = "vlm_gym:latest",
    gpu_devices: List[str] = None,
    memory_limit: str = "32g",
    **kwargs
) -> ContainerConfig:
    """
    Convenience function for creating a VLM container configuration
    
    Args:
        image_name: Docker image name
        gpu_devices: List of GPU devices
        memory_limit: Memory limit
        **kwargs: Other configuration parameters
        
    Returns:
        ContainerConfig instance
    """
    config = ContainerConfig(
        image_name=image_name,
        container_name=f"vlm_task_{int(time.time())}",
        gpu_devices=gpu_devices or ["0"],
        memory_limit=memory_limit,
        shm_size=kwargs.pop("shm_size", "32g"),
        **kwargs
    )
    return config


def create_vlm_task_config(
    task_id: str,
    task_type: str,
    timeout: float = 300,
    **kwargs
) -> VLMTaskConfig:
    """
    Convenience function for creating a VLM task configuration
    
    Args:
        task_id: Task ID
        task_type: Task type
        timeout: Timeout duration
        **kwargs: Other configuration parameters
        
    Returns:
        VLMTaskConfig instance
    """
    config = VLMTaskConfig(
        task_id=task_id,
        task_type=task_type,
        timeout=timeout,
        **kwargs
    )
    return config


# Usage examples
if __name__ == "__main__":
    # Example 1: Basic usage
    manager = ContainerManager()
    
    # Configuration
    container_config = ContainerConfig(
        image_name="vlm_gym:latest",
        container_name="test_vlm_task",
        gpu_devices=["0"],
        memory_limit="16g",
        shm_size="32g"
    )
    
    task_config = VLMTaskConfig(
        task_id="test_001",
        task_type="chartqa",
        timeout=300,
        model_cache_dir="/home/models"
    )
    
    # Create and run container
    try:
        container = manager.create_vlm_container(container_config, task_config)
        process = manager.start_container(container)
        
        # Execute command
        output, exit_code = manager.execute_in_container(
            container,
            "python --version",
            timeout=10
        )
        print(f"Python version: {output}")
        
        # Get statistics
        stats = manager.get_container_stats(container)
        print(f"Container stats: {stats}")
        
    finally:
        manager.cleanup_all()
    
    # Example 2: Using the executor
    with VLMContainerExecutor() as executor:
        # Define task script
        task_script = """
import time
print("Starting VLM task...")
time.sleep(2)
print("Task completed!")
"""
        
        # Run task
        result = executor.run_vlm_task(
            task_config,
            container_config,
            task_script
        )
        
        print(f"Task result: {result}")
