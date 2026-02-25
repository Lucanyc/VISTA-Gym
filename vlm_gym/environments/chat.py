from typing import List, Dict, Any, Literal, Union
import time
import numpy as np
from PIL import Image

class MultiModalChat:
    """Multimodal chat manager"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        
    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: Union[str, Dict[str, Any]]
    ):
        """Add a message
        
        Args:
            role: Message role
            content: Message content, can be a string or a dict with multimodal content
                    e.g. {"text": "...", "image": PIL.Image}
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        # Normalize content format
        if isinstance(content, str):
            message["content"] = {"text": content}
            
        self.messages.append(message)
        
    def get_history(self, 
                    include_images: bool = True,
                    max_messages: int = None) -> List[Dict[str, Any]]:
        """Get chat history
        
        Args:
            include_images: Whether to include images
            max_messages: Maximum number of messages
        """
        messages = self.messages.copy()
        
        if max_messages:
            messages = messages[-max_messages:]
            
        if not include_images:
            # Remove image content
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
        """Get the last message"""
        return self.messages[-1] if self.messages else None
    
    def clear(self):
        """Clear chat history"""
        self.messages = []
        
    def close(self):
        """Close the chat"""
        self.clear()
