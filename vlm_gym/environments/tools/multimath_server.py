import os
import json
import requests
import base64
from typing import Dict, Any, Optional
from PIL import Image
import io
import logging

from vlm_gym.environments.tools.base import ToolBase

logger = logging.getLogger(__name__)

class MultiMathRemoteTool(ToolBase):
    """Remote MultiMath tool via API"""
    
    name = "multimath_server"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name=self.name)
        
        # 工具描述
        self.description = "Remote multimodal math problem solver using MultiMath-7B API"
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["solve", "analyze", "explain"],
                    "description": "Task type"
                },
                "question": {
                    "type": "string",
                    "description": "Math question or problem statement"
                },
                "image": {
                    "type": "string",
                    "description": "Base64 encoded image or image path (optional)"
                },
                "problem_type": {
                    "type": "string",
                    "enum": ["geometry", "algebra", "word_problem", "graph", "table", "general"],
                    "default": "general",
                    "description": "Type of math problem"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["answer_only", "with_steps", "detailed"],
                    "default": "answer_only",
                    "description": "Output format preference"
                }
            },
            "required": ["task", "question"]
        }
        
        self.config = config or {}
        self.api_url = self.config.get("api_url", "http://localhost:8001")
        self.timeout = self.config.get("timeout", 60)
        self.current_image = None
        
    def reset(self, image=None):
        """重置工具状态"""
        self.current_image = image
        logger.debug("MultiMath remote tool reset")
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行远程 API 调用"""
        try:
            # 解析参数
            if isinstance(params, str):
                params = json.loads(params)
            
            question = params.get("question", "")
            if not question:
                return {
                    "error": "No question provided",
                    "error_type": "InvalidInput",
                    "success": False
                }
            
            # 准备请求数据
            request_data = {
                "question": question,
                "task": params.get("task", "solve"),
                "output_format": params.get("output_format", "answer_only"),
                "problem_type": params.get("problem_type", "general")
            }

            # 处理图像
            image_path = params.get("image")
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode()
                    request_data["image_base64"] = image_base64
                except Exception as e:
                    logger.error(f"Failed to encode image: {e}")
            elif self.current_image:
                # 使用当前上下文图像
                try:
                    buffer = io.BytesIO()
                    self.current_image.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    request_data["image_base64"] = image_base64
                except Exception as e:
                    logger.error(f"Failed to encode current image: {e}")
            
            # 调用 API
            logger.info(f"Calling MultiMath API at {self.api_url}/solve")
            response = requests.post(
                f"{self.api_url}/solve",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return {
                        "success": True,
                        "answer": result.get("answer", ""),
                        "steps": result.get("steps"),
                        "method": "MultiMath-7B (Remote API)",
                        "has_image": "image_base64" in request_data
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "error_type": "APIError"
                    }
            else:
                return {
                    "success": False,
                    "error": f"API returned status code: {response.status_code}",
                    "error_type": "HTTPError"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"API request timed out after {self.timeout} seconds",
                "error_type": "Timeout"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Could not connect to API at {self.api_url}",
                "error_type": "ConnectionError"
            }
        except Exception as e:
            logger.error(f"Remote MultiMath execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def set_current_image(self, image):
        """设置当前图像（由环境调用）"""
        self.current_image = image