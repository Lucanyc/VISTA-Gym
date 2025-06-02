import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager
import signal
import ray
from ray.util import state

logger = logging.getLogger(__name__)

def parse_and_truncate_error(error_msg: str, max_length: int = 1000) -> str:
    """
    Parse and truncate error messages to ensure complete but not too long output
    """
    error_msg = error_msg.replace('^', '')
    
    # Extract the most important parts of the error
    lines = error_msg.split('\n')
    
    # Keep the error type and message
    important_lines = []
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in ['Error', 'Exception', 'Traceback']):
            important_lines.extend(lines[i:i+3])
            
    if important_lines:
        error_msg = '\n'.join(important_lines)
        
    # Truncate if still too long
    if len(error_msg) > max_length:
        error_msg = error_msg[:max_length-3] + "..."
        
    return error_msg

class TimeoutException(Exception):
    """Custom exception for timeout handling"""
    pass

@contextmanager
def timeout_manager(seconds: int):
    """Context manager for handling timeouts"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def safe_execute(func: Callable, *args, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Safely execute a function with timeout and error handling"""
    start_time = time.time()
    
    try:
        if timeout:
            with timeout_manager(timeout):
                result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time,
            "error": None
        }
        
    except TimeoutException as e:
        return {
            "success": False,
            "result": None,
            "execution_time": time.time() - start_time,
            "error": str(e),
            "error_type": "timeout"
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "result": None,
            "execution_time": time.time() - start_time,
            "error": parse_and_truncate_error(error_trace),
            "error_type": type(e).__name__
        }

# Ray utilities (if using Ray for distributed processing)
if ray.is_initialized():
    
    @ray.remote
    def run_exp_remote(exp_arg, *dependencies, avg_step_timeout=60):
        """Run experiment remotely with Ray"""
        return exp_arg.run()
    
    def get_elapsed_time(task_ref: ray.ObjectRef) -> Optional[float]:
        """Get elapsed time for a Ray task"""
        task_id = task_ref.task_id().hex()
        task_info = state.get_task(task_id, address="auto")
        
        if task_info and task_info.start_time_ms is not None:
            start_time_s = task_info.start_time_ms / 1000.0
            current_time_s = time.time()
            return current_time_s - start_time_s
        return None
    
    def poll_for_timeout(
        tasks: Dict[str, ray.ObjectRef],
        timeout: float,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Cancel Ray tasks that exceed timeout"""
        task_list = list(tasks.values())
        task_ids = list(tasks.keys())
        
        logger.warning(f"Tasks exceeding {timeout} seconds will be cancelled.")
        
        while True:
            ready, not_ready = ray.wait(
                task_list,
                num_returns=len(task_list),
                timeout=poll_interval
            )
            
            # Check timeouts for running tasks
            for task in not_ready:
                elapsed_time = get_elapsed_time(task)
                if elapsed_time and elapsed_time > timeout:
                    task_id = task.task_id().hex()
                    logger.warning(f"Task {task_id} exceeded timeout ({elapsed_time:.1f}s > {timeout}s)")
                    
                    if elapsed_time < timeout + 60:
                        ray.cancel(task, force=False)
                    else:
                        ray.cancel(task, force=True)
                        
            # All tasks completed
            if len(ready) == len(task_list):
                results = []
                for task in ready:
                    try:
                        result = ray.get(task)
                    except Exception as e:
                        result = {"error": str(e), "success": False}
                    results.append(result)
                    
                return {task_id: result for task_id, result in zip(task_ids, results)}

def validate_environment_state(env) -> Dict[str, Any]:
    """Validate the current state of an environment"""
    checks = {
        "env_created": env is not None,
        "observation_space_valid": False,
        "action_space_valid": False,
        "can_reset": False,
        "can_step": False,
    }
    
    if env:
        try:
            checks["observation_space_valid"] = hasattr(env, 'observation_space')
            checks["action_space_valid"] = hasattr(env, 'action_space')
            
            # Try reset
            obs, info = env.reset()
            checks["can_reset"] = True
            
            # Try a dummy step
            if hasattr(env.action_space, 'sample'):
                action = env.action_space.sample()
            else:
                action = "dummy_action"
                
            obs, reward, done, truncated, info = env.step(action)
            checks["can_step"] = True
            
        except Exception as e:
            logger.error(f"Environment validation error: {e}")
            
    return checks

def create_episode_summary(episode_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of an episode"""
    summary = {
        "total_steps": len(episode_data),
        "total_reward": sum(step.get("reward", 0) for step in episode_data),
        "success": False,
        "truncated": False,
        "final_observation": None,
        "action_distribution": {},
    }
    
    # Analyze actions
    action_counts = {}
    for step in episode_data:
        action = step.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        
    summary["action_distribution"] = action_counts
    
    # Check final status
    if episode_data:
        final_step = episode_data[-1]
        summary["success"] = final_step.get("done", False) and not final_step.get("truncated", False)
        summary["truncated"] = final_step.get("truncated", False)
        summary["final_observation"] = final_step.get("observation")
        
    return summary
