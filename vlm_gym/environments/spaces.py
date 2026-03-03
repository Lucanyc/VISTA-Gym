
from typing import Any, Tuple, Optional
import numpy as np
from gymnasium.spaces import Space, Box
from PIL import Image

class Unicode(Space):
    """Unicode字符串空间"""
    
    def __init__(self, max_length: Optional[int] = None):
        self.max_length = max_length
        super().__init__()
        
    def sample(self) -> str:
        """采样一个字符串"""
        length = np.random.randint(1, self.max_length or 100)
        return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), length))
    
    def contains(self, x: Any) -> bool:
        """检查是否包含"""
        return isinstance(x, str) and (self.max_length is None or len(x) <= self.max_length)
    
    def __repr__(self) -> str:
        return f"Unicode(max_length={self.max_length})"


class ImageSpace(Box):
    """图像空间，继承自Box"""
    
    def __init__(
        self,
        shape: Optional[Tuple[int, int, int]] = (224, 224, 3),
        dtype: np.dtype = np.uint8
    ):

        if shape is None:
            shape = (224, 224, 3)
        
        # 调用Box的构造函数
        super().__init__(
            low=0,
            high=255,
            shape=shape,
            dtype=dtype
        )
    
    def contains(self, x: Any) -> bool:
        """检查是否是有效图像"""
        # 支持numpy数组和PIL图像
        if isinstance(x, Image.Image):
            return True
        if isinstance(x, np.ndarray):
            return super().contains(x)
        return False
    
    def __repr__(self) -> str:
        return f"ImageSpace(shape={self.shape}, dtype={self.dtype})"

class AnyDict(Space):
    """任意字典空间"""
    
    def __init__(self):
        super().__init__()
    
    def sample(self) -> dict:
        """采样一个字典"""
        return {}
    
    def contains(self, x: Any) -> bool:
        """检查是否是字典"""
        return isinstance(x, dict)
    
    def __repr__(self) -> str:
        return "AnyDict()"

class Float(Box):
    """浮点数空间"""
    
    def __init__(self):
        super().__init__(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
    
    def __repr__(self) -> str:
        return "Float()"

class Integer(Box):
    """整数空间"""
    
    def __init__(self):
        super().__init__(low=-np.inf, high=np.inf, shape=(), dtype=np.int64)
    
    def __repr__(self) -> str:
        return "Integer()"

class Anything(Space):
    """任意类型空间"""
    
    def contains(self, x: Any) -> bool:
        return True
    
    def sample(self) -> Any:
        return None
    
    def __repr__(self) -> str:
        return "Anything()"
