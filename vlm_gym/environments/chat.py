from typing import List, Dict, Any, Literal, Union
import time
import numpy as np
from PIL import Image

class MultiModalChat:
    """支持多模态的聊天管理器"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        
    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: Union[str, Dict[str, Any]]
    ):
        """添加消息
        
        Args:
            role: 消息角色
            content: 消息内容，可以是字符串或包含多模态内容的字典
                    如 {"text": "...", "image": PIL.Image}
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        # 标准化内容格式
        if isinstance(content, str):
            message["content"] = {"text": content}
            
        self.messages.append(message)
        
    def get_history(self, 
                    include_images: bool = True,
                    max_messages: int = None) -> List[Dict[str, Any]]:
        """获取聊天历史
        
        Args:
            include_images: 是否包含图像
            max_messages: 最大消息数量
        """
        messages = self.messages.copy()
        
        if max_messages:
            messages = messages[-max_messages:]
            
        if not include_images:
            # 移除图像内容
            processed_messages = []
            for msg in messages:
                new_msg = msg.copy()
                if isinstance(msg["content"], dict) and "image" in msg["content"]:
                    new_msg["content"] = {
                        "text": msg["content"].get("text", ""),
                        "has_image": True
                    }
                processed_messages.append(new_msg)
            return processed_messages
            
        return messages
    
    def get_last_message(self) -> Dict[str, Any]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def clear(self):
        """清空聊天历史"""
        self.messages = []
        
    def close(self):
        """关闭聊天"""
        self.clear()