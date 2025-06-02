import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Iterator
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class VLMDataset(Dataset):
    """Base dataset class for VLM tasks"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        transform=None,
        max_samples: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path.parent / "images"
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
            
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                return json.load(f)
        elif self.data_path.suffix == '.jsonl':
            with jsonlines.open(self.data_path, 'r') as reader:
                return list(reader)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item"""
        item = self.data[idx].copy()
        
        # Load image if path is provided
        if 'image_path' in item:
            image_path = self.image_dir / item['image_path']
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            item['image'] = image
            
        return item

def create_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader with custom collate function for VLM data"""
    
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for mixed data types"""
        collated = {}
        
        # Get all keys
        keys = set()
        for item in batch:
            keys.update(item.keys())
            
        # Collate each key
        for key in keys:
            values = [item.get(key) for item in batch if key in item]
            
            if not values:
                continue
                
            # Handle different types
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                collated[key] = torch.tensor(np.stack(values))
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values
                
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )

def split_dataset(
    data: List[Dict[str, Any]],
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Split dataset into train/val/test sets"""
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
        data = [data[i] for i in indices]
        
    # Calculate split sizes
    n_total = len(data)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])
    
    # Split data
    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:]
    }
    
    return splits

def export_to_dpo_format(
    episodes: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "trl"
) -> None:
    """Export episodes to DPO training format"""
    output_path = Path(output_path)
    dpo_data = []
    
    for episode in episodes:
        # Extract conversation and preferences
        conversation = episode.get("conversation", [])
        preference = episode.get("preference", {})
        
        if format == "trl":
            # TRL format: {"prompt": ..., "chosen": ..., "rejected": ...}
            item = {
                "prompt": conversation[0]["content"] if conversation else "",
                "chosen": preference.get("chosen", ""),
                "rejected": preference.get("rejected", "")
            }
        elif format == "openrlhf":
            # OpenRLHF format
            item = {
                "conversations": conversation,
                "chosen": preference.get("chosen", ""),
                "rejected": preference.get("rejected", ""),
                "metadata": episode.get("metadata", {})
            }
        else:
            raise ValueError(f"Unknown format: {format}")
            
        dpo_data.append(item)
        
    # Save data
    with jsonlines.open(output_path, 'w') as writer:
        for item in dpo_data:
            writer.write(item)
            
def load_episode_data(
    episode_dir: Union[str, Path],
    pattern: str = "*.json"
) -> Iterator[Dict[str, Any]]:
    """Load episode data from directory"""
    episode_dir = Path(episode_dir)
    
    for file_path in episode_dir.glob(pattern):
        with open(file_path, 'r') as f:
            yield json.load(f)

def create_dataset_statistics(
    dataset: Union[List[Dict], Dataset]
) -> Dict[str, Any]:
    """Generate statistics for a dataset"""
    stats = {
        "total_samples": len(dataset),
        "fields": {},
        "distributions": {}
    }
    
    # Analyze fields
    if isinstance(dataset, list):
        all_keys = set()
        for item in dataset:
            all_keys.update(item.keys())
            
        stats["fields"] = {key: 0 for key in all_keys}
        
        # Count occurrences
        for item in dataset:
            for key in item:
                stats["fields"][key] += 1
                
    # Compute distributions for categorical fields
    categorical_fields = ["task_type", "difficulty", "question_type"]
    for field in categorical_fields:
        values = []
        for item in dataset:
            if field in item:
                values.append(item[field])
                
        if values:
            unique, counts = np.unique(values, return_counts=True)
            stats["distributions"][field] = dict(zip(unique.tolist(), counts.tolist()))
            
    return stats
