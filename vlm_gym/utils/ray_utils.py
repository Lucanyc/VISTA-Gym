import ray
import time
from typing import Dict, Any, List, Callable, Optional
from ray.util import state
import logging

logger = logging.getLogger(__name__)

def init_ray(
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    memory: Optional[int] = None,
    object_store_memory: Optional[int] = None,
    dashboard_host: str = "0.0.0.0",
    **kwargs
) -> None:
    """Initialize Ray with specified resources"""
    if ray.is_initialized():
        logger.info("Ray is already initialized")
        return
        
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        object_store_memory=object_store_memory,
        dashboard_host=dashboard_host,
        **kwargs
    )
    logger.info(f"Ray initialized with {ray.available_resources()}")

@ray.remote
class RayVLMWorker:
    """Remote worker for VLM processing"""
    
    def __init__(self, worker_id: int, config: Dict[str, Any]):
        self.worker_id = worker_id
        self.config = config
        self.env = None
        self.agent = None
        
    def setup(self):
        """Setup environment and agent"""
        # Import here to avoid serialization issues
        from vlm_gym.environments import VisionQAEnv
        from vlm_gym.agents import VLMAgent
        
        self.env = VisionQAEnv(**self.config["env_kwargs"])
        self.agent = VLMAgent(**self.config["agent_kwargs"])
        
    def run_episode(self, task_id: str) -> Dict[str, Any]:
        """Run a single episode"""
        if self.env is None:
            self.setup()
            
        obs, info = self.env.reset(task_id=task_id)
        done = False
        episode_data = []
        
        while not done:
            action = self.agent.act(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            
            episode_data.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "info": info
            })
            
        return {
            "worker_id": self.worker_id,
            "task_id": task_id,
            "episode_data": episode_data,
            "total_reward": sum(step["reward"] for step in episode_data)
        }

def distributed_evaluation(
    task_ids: List[str],
    config: Dict[str, Any],
    num_workers: int = 4,
    timeout: int = 3600
) -> List[Dict[str, Any]]:
    """Run distributed evaluation using Ray"""
    
    # Create workers
    workers = [
        RayVLMWorker.remote(worker_id=i, config=config)
        for i in range(num_workers)
    ]
    
    # Distribute tasks
    task_futures = []
    for i, task_id in enumerate(task_ids):
        worker = workers[i % num_workers]
        future = worker.run_episode.remote(task_id)
        task_futures.append(future)
        
    # Collect results with timeout
    results = []
    start_time = time.time()
    
    while task_futures and time.time() - start_time < timeout:
        ready, not_ready = ray.wait(task_futures, timeout=1)
        
        for future in ready:
            try:
                result = ray.get(future)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed with error: {e}")
                
        task_futures = not_ready
        
    # Cancel remaining tasks
    for future in task_futures:
        ray.cancel(future)
        
    return results

@ray.remote
def process_batch(
    batch_data: List[Any],
    process_fn: Callable,
    **kwargs
) -> List[Any]:
    """Process a batch of data remotely"""
    results = []
    for item in batch_data:
        try:
            result = process_fn(item, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            results.append(None)
    return results

def parallel_process(
    data: List[Any],
    process_fn: Callable,
    batch_size: int = 10,
    num_workers: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """Process data in parallel using Ray"""
    
    if num_workers is None:
        num_workers = ray.available_resources().get("CPU", 1)
        
    # Split data into batches
    batches = [
        data[i:i + batch_size]
        for i in range(0, len(data), batch_size)
    ]
    
    # Process batches in parallel
    futures = [
        process_batch.remote(batch, process_fn, **kwargs)
        for batch in batches
    ]
    
    # Collect results
    results = []
    for future in futures:
        batch_results = ray.get(future)
        results.extend(batch_results)
        
    return results

def monitor_ray_tasks(interval: int = 5) -> None:
    """Monitor Ray tasks and print statistics"""
    while True:
        tasks = state.list_tasks()
        
        # Count task states
        state_counts = {}
        for task in tasks:
            state_name = task.state
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
            
        # Print statistics
        print(f"\nRay Task Statistics ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print("-" * 40)
        for state_name, count in state_counts.items():
            print(f"{state_name}: {count}")
            
        time.sleep(interval)
