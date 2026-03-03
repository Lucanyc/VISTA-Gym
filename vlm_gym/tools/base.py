# vlm_gym/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

class BaseVisionTool(ABC):
    """视觉工具基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """处理图像并返回结果"""
        pass