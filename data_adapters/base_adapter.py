
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path

class BaseDataAdapter(ABC):

    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        
    @abstractmethod
    def get_task_ids(self) -> List[str]:
        """get all ids for all tasks"""
        pass
        
    @abstractmethod
    def get_task_data(self, task_id: str) -> Dict[str, Any]:
        """get data for single task"""
        pass
        
    @abstractmethod
    def _load_annotations(self) -> Dict[str, Any]:
        """load annotation data"""
        pass
