import sys
import os
import json
import re
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from math import ceil, floor

# 添加DeepEyes路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'DeepEyes'))

from vlm_gym.environments.tools.base import ToolBase


class DeepEyesTool(ToolBase):
    """
    DeepEyes工具集成到VLMGym
    支持image_zoom_in_tool和image_rotate_tool
    """
    name = "deepeyes"
    
    def __init__(self, _name=None, _desc='', _params={}, **kwargs):
        # 初始化父类
        super().__init__(
            name=self.name,
            description="DeepEyes visual reasoning tools with zoom and rotate capabilities",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action string containing tool calls in XML format"
                    }
                },
                "required": ["action"]
            }
        )
        
        self.config = kwargs.get('config', {})
        
        # 工具状态
        self.current_image = None
        self.original_image = None
        self.width = None
        self.height = None
        
        # 提示模板
        self.user_prompt = (
            "Here is the cropped/rotated image returned after you calling the function.\n"
            "If the images provided above are sufficient to answer the user's question, "
            "please put your final answer within <answer></answer>. "
            "Otherwise you can continue to call tools within <tool_call></tool_call>."
        )
        
        print("[DeepEyes] Tool initialized")
    
    def reset(self, image: Image.Image):
        """重置工具状态，准备处理新图像"""
        # 处理输入
        if isinstance(image, str):
            image = Image.open(image)
        
        # 保存原始图像
        self.original_image = image.convert("RGB")
        self.current_image = self.original_image.copy()
        
        # 记录图像尺寸
        self.width = self.original_image.width
        self.height = self.original_image.height
        
        print(f"[DeepEyes] Reset with image size: {self.width}x{self.height}")
    
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行DeepEyes工具调用"""
        if self.current_image is None:
            return {
                "error": "No image loaded. Please call reset() first.",
                "error_type": "NoImageError"
            }
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
                action_string = params.get("action", action_string)
            except json.JSONDecodeError:
                # 如果不是JSON，直接使用原始字符串
                pass
        
        # 检查是否包含答案
        answer = self._extract_answer(action_string)
        if answer:
            return {
                "processed_output": answer,
                "success": True,
                "tool_used": "answer_extraction",
                "final_answer": answer
            }
        
        # 提取工具调用
        tool_call = self._extract_tool_call(action_string)
        if not tool_call:
            return {
                "error": "No valid tool call found in action string",
                "error_type": "NoToolCallError",
                "action_string": action_string
            }
        
        try:
            # 解析工具调用JSON
            tool_data = json.loads(tool_call.strip())
            tool_name = tool_data.get("name")
            args = tool_data.get("arguments", {})
            
            # 执行具体工具
            if tool_name == "image_zoom_in_tool":
                result = self._execute_zoom_in(args)
            elif tool_name == "image_rotate_tool":
                result = self._execute_rotate(args)
            else:
                return {
                    "error": f"Unknown tool name: {tool_name}",
                    "error_type": "UnknownToolError"
                }
            
            # 添加工具使用信息
            result["tool_used"] = tool_name
            result["success"] = True
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid tool call JSON format: {str(e)}",
                "error_type": "JSONDecodeError",
                "tool_call": tool_call
            }
        except Exception as e:
            import traceback
            return {
                "error": f"Tool execution failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    def _extract_answer(self, action_string: str) -> Optional[str]:
        """提取<answer>标签中的内容"""
        matches = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    def _extract_tool_call(self, action_string: str) -> Optional[str]:
        """提取<tool_call>标签中的内容"""
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return matches[-1] if matches else None
    
    def _execute_zoom_in(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像缩放"""
        # 获取边界框
        bbox = args.get("bbox_2d", args.get("bbox"))
        if not bbox:
            return {
                "error": "Missing required argument: bbox_2d or bbox",
                "error_type": "MissingArgumentError"
            }
        
        # 验证并调整边界框
        bbox = self._maybe_resize_bbox(*bbox)
        if not bbox:
            return {
                "error": "Invalid bounding box",
                "error_type": "InvalidBBoxError",
                "original_bbox": args.get("bbox_2d", args.get("bbox"))
            }
        
        # 裁剪图像
        self.current_image = self.current_image.crop(bbox)
        
        # 返回结果
        return {
            "processed_output": self.user_prompt,
            "image": self.current_image,
            "bbox_used": bbox,
            "new_size": [self.current_image.width, self.current_image.height]
        }
    
    def _execute_rotate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像旋转"""
        angle = args.get("angle")
        if angle is None:
            return {
                "error": "Missing required argument: angle",
                "error_type": "MissingArgumentError"
            }
        
        # 旋转图像
        self.current_image = self.current_image.rotate(angle, expand=True)
        
        # 返回结果
        return {
            "processed_output": self.user_prompt,
            "image": self.current_image,
            "rotation_angle": angle,
            "new_size": [self.current_image.width, self.current_image.height]
        }
    
    def _validate_bbox(self, left: int, top: int, right: int, bottom: int) -> bool:
        """验证边界框的有效性"""
        try:
            assert left < right and bottom > top, f'Invalid shape: {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"Aspect ratio error"
            assert min(height, width) > 30, f"Box too small: {height=}, {width=}"
            return True
        except Exception as e:
            print(f"[DeepEyes] BBox validation error: {e}")
            return False
    
    def _maybe_resize_bbox(self, left: int, top: int, right: int, bottom: int) -> Optional[Tuple[int, int, int, int]]:
        """调整边界框以确保有效性"""
        # 确保在图像范围内
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        
        if not self._validate_bbox(left, top, right, bottom):
            return None
        
        height = bottom - top
        width = right - left
        
        # 如果太小，进行缩放
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            
            # 再次确保在图像范围内
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(self.width, new_right)
            new_bottom = min(self.height, new_bottom)
            
            if not self._validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            
            return (new_left, new_top, new_right, new_bottom)
        
        return (left, top, right, bottom)
    
    def get_current_image(self) -> Optional[Image.Image]:
        """获取当前处理的图像"""
        return self.current_image
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": "DeepEyes visual reasoning tools",
            "capabilities": [
                "图像缩放（zoom in）",
                "图像旋转（rotate）",
                "视觉推理（visual reasoning）"
            ],
            "tools": {
                "image_zoom_in_tool": {
                    "description": "Zoom into a specific region of the image",
                    "parameters": {
                        "bbox_2d": {
                            "type": "array",
                            "description": "Bounding box [left, top, right, bottom]",
                            "required": True
                        }
                    }
                },
                "image_rotate_tool": {
                    "description": "Rotate the image by a specified angle",
                    "parameters": {
                        "angle": {
                            "type": "number",
                            "description": "Rotation angle in degrees",
                            "required": True
                        }
                    }
                }
            },
            "input_format": {
                "tool_call": "XML format: <tool_call>{json}</tool_call>",
                "answer": "XML format: <answer>final answer</answer>"
            }
        }