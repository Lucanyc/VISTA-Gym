import yaml
import json
import jsonlines
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import pickle

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Handle includes
    if 'include' in config:
        base_dir = config_path.parent
        for include_path in config['include']:
            include_config = load_config(base_dir / include_path)
            config = merge_configs(config, include_config)
            
    return config

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Recursively merge two configuration dictionaries"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def save_conversation_history(
    conversation_history: List[Dict[str, Any]],
    save_path: Union[str, Path],
    format: str = "json"
) -> None:
    """Save conversation history in various formats"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(save_path, 'w') as f:
            json.dump(conversation_history, f, indent=2, default=str)
    elif format == "jsonl":
        with jsonlines.open(save_path, 'w') as writer:
            for item in conversation_history:
                writer.write(item)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_conversation_history(
    load_path: Union[str, Path],
    format: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load conversation history from file"""
    load_path = Path(load_path)
    
    if format is None:
        # Infer format from extension
        format = load_path.suffix[1:] if load_path.suffix else "json"
        
    if format == "json":
        with open(load_path, 'r') as f:
            return json.load(f)
    elif format in ["jsonl", "jsonlines"]:
        with jsonlines.open(load_path, 'r') as reader:
            return list(reader)
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_experiment_id(config: Dict[str, Any], prefix: str = "exp") -> str:
    """Create a unique experiment ID based on configuration"""
    # Create a hash of the config
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{prefix}_{timestamp}_{config_hash}"

def setup_experiment_directory(
    base_dir: Union[str, Path],
    experiment_id: str
) -> Dict[str, Path]:
    """Setup directory structure for an experiment"""
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_id
    
    # Create directories
    dirs = {
        "root": exp_dir,
        "logs": exp_dir / "logs",
        "checkpoints": exp_dir / "checkpoints",
        "results": exp_dir / "results",
        "configs": exp_dir / "configs",
        "visualizations": exp_dir / "visualizations",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def save_checkpoint(
    data: Any,
    checkpoint_dir: Union[str, Path],
    name: str,
    format: str = "pickle"
) -> Path:
    """Save a checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.{format}"
    filepath = checkpoint_dir / filename
    
    if format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == "json":
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    return filepath

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    format: Optional[str] = None
) -> Any:
    """Load a checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    if format is None:
        format = checkpoint_path.suffix[1:]
        
    if format == "pickle":
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    elif format == "json":
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_git_info() -> Dict[str, str]:
    """Get current git information"""
    import subprocess
    
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('ascii').strip()
        
        # Check if there are uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii')
        dirty = len(status) > 0
        
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False
        }
