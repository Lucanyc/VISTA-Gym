"""
G-LLaVA Tool for VLMGym
"""
import torch
import json
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import os
import re
from vlm_gym.environments.tools.base import ToolBase

# 设置使用Mock版本还是真实模型
USE_MOCK = True  # 设置为False时使用真实模型

logger = logging.getLogger(__name__)


class GLLaVATool(ToolBase):
    """G-LLaVA tool for geometry problem solving"""
    
    name = "gllava"
    
    def __init__(self, config=None):
        """Initialize G-LLaVA tool"""
        super().__init__(
            name=self.name,
            description="G-LLaVA-13B model for visual reasoning and geometry problem solving",
            parameters={
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string", 
                        "description": "The geometry problem to solve"
                    },
                    "task": {
                        "type": "string",
                        "description": "Task type: 'solve' or 'analyze'",
                        "enum": ["solve", "analyze"],
                        "default": "solve"
                    }
                },
                "required": ["problem"]
            }
        )
        
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._model_loaded = False
        self.current_image = None
        self.current_image_path = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设备配置
        self.device = self._get_device()
        
        # Mock模式标志
        self.use_mock = USE_MOCK
        if self.use_mock:
            self.logger.info("[G-LLaVA] Running in MOCK mode - no model will be loaded")
        else:
            self.logger.info(f"[G-LLaVA] Initialized for real model on device: {self.device}")
    
    def _get_device(self):
        """获取设备配置"""
        if torch.cuda.is_available():
            # 如果有多个GPU，使用第二个
            if torch.cuda.device_count() > 1:
                return torch.device("cuda:1")
            else:
                return torch.device("cuda:0")
        return torch.device("cpu")
    
    def reset(self, image=None, image_path=None):
        """重置工具并保存图像"""
        if image is not None:
            # 确保是PIL Image
            if not isinstance(image, Image.Image):
                try:
                    if hasattr(image, 'numpy'):
                        # tensor转numpy
                        image_array = image.cpu().numpy() if hasattr(image, 'cpu') else image.numpy()
                        if len(image_array.shape) == 4:
                            image_array = image_array[0]
                        if image_array.shape[0] in [3, 4]:
                            image_array = np.transpose(image_array, (1, 2, 0))
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        image = Image.fromarray(image_array)
                    elif isinstance(image, np.ndarray):
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image.astype(np.uint8))
                except Exception as e:
                    self.logger.error(f"Failed to convert image: {e}")
                    return
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.current_image = image
            self.current_image_path = image_path
            self.logger.info(f"[G-LLaVA] Image loaded: size={image.size}, mode={image.mode}")
        else:
            self.current_image = None
            self.current_image_path = image_path
            self.logger.warning("[G-LLaVA] No image provided!")
    
    def execute(self, params_str):
        """执行工具"""
        try:
            # 解析参数
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
            problem = params.get('problem', params.get('query', ''))
            task = params.get('task', 'solve')
            
            self.logger.info(f"[G-LLaVA] Execute: problem='{problem[:50]}...', task={task}")
            
            # 检查图像
            if self.current_image is None:
                return self._provide_fallback(problem, task, "No image provided")
            
            # Mock模式
            if self.use_mock:
                return self._mock_execute(problem, task)
            
            # 真实模型执行
            return self._real_execute(problem, task)
            
        except Exception as e:
            self.logger.error(f"G-LLaVA execution failed: {str(e)}")
            return self._provide_fallback(
                params.get('problem', ''), 
                params.get('task', 'solve'),
                f"Error: {str(e)}"
            )
    
    def _mock_execute(self, problem: str, task: str) -> Dict[str, Any]:
        """Mock执行，返回模拟结果"""
        self.logger.info("[G-LLaVA MOCK] Generating mock response")
        
        # 分析问题类型
        problem_lower = problem.lower()
        
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', problem)
        
        # 生成mock响应
        if "angle" in problem_lower or "degree" in problem_lower:
            steps = [
                "Step 1: Analyzing the geometric diagram to identify all angles",
                "Step 2: Identifying the given angle measurements in the figure",
                "Step 3: Applying angle relationships (e.g., sum of angles in triangle = 180°)",
                "Step 4: Setting up equations based on the angle constraints",
                "Step 5: Solving for the unknown angle"
            ]
            
            # Mock计算
            if "triangle" in problem_lower and len(numbers) >= 2:
                known_angles = [float(n) for n in numbers if float(n) < 180]
                if len(known_angles) == 2:
                    answer = 180 - sum(known_angles)
                    final_answer = str(int(answer) if answer.is_integer() else answer)
                    steps.append(f"Step 6: 180° - {known_angles[0]}° - {known_angles[1]}° = {final_answer}°")
                else:
                    final_answer = "55"  # Mock答案
            else:
                final_answer = "45"  # Mock答案
                
        elif "length" in problem_lower or "side" in problem_lower:
            steps = [
                "Step 1: Examining the geometric figure to identify all sides",
                "Step 2: Looking for congruent segments or similar triangles",
                "Step 3: Applying the Pythagorean theorem where applicable",
                "Step 4: Using properties of special geometric figures",
                "Step 5: Calculating the unknown length"
            ]
            final_answer = "12"  # Mock答案
            
        elif "area" in problem_lower:
            steps = [
                "Step 1: Identifying the shape in the diagram",
                "Step 2: Determining the appropriate area formula",
                "Step 3: Finding all necessary measurements from the figure",
                "Step 4: Applying the area formula",
                "Step 5: Computing the final area"
            ]
            final_answer = "24"  # Mock答案
            
        else:
            steps = [
                "Step 1: Analyzing the geometric diagram",
                "Step 2: Identifying given measurements and constraints",
                "Step 3: Applying relevant geometric theorems",
                "Step 4: Setting up equations",
                "Step 5: Solving for the unknown"
            ]
            final_answer = "15"  # Mock答案
        
        response_text = "\n".join(steps)
        if final_answer:
            response_text += f"\n\nTherefore, the answer is {final_answer}."
        
        return {
            "success": True,
            "task_type": task,
            "formalized_output": f"[MOCK] Analyzing geometry problem: {problem}",
            "solution": final_answer,
            "steps": steps,
            "raw_response": response_text,
            "method": "mock",
            "warning": "This is a mock response for testing"
        }
    
    def _real_execute(self, problem: str, task: str) -> Dict[str, Any]:
        """真实模型执行"""
        # TODO: 实现真实模型加载和推理
        # 这里是真实模型的实现占位符
        return self._provide_fallback(problem, task, "Real model not implemented yet")
    
    def _provide_fallback(self, problem: str, task: str, reason: str) -> Dict[str, Any]:
        """提供备用响应"""
        return {
            "success": False,
            "task_type": task,
            "formalized_output": f"Unable to process: {reason}",
            "solution": "",
            "steps": [
                "Unable to analyze the geometry problem due to technical limitations",
                f"Reason: {reason}"
            ],
            "raw_response": f"Fallback response: {reason}",
            "method": "fallback",
            "error": reason
        }
    
    def _extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        # 尝试各种模式提取答案
        patterns = [
            r'answer is[:\s]+(\d+\.?\d*)',
            r'equals?[:\s]+(\d+\.?\d*)',
            r'=\s*(\d+\.?\d*)',
            r'Therefore,?\s+.*?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 提取最后一个合理的数字
        numbers = re.findall(r'\b\d+\.?\d*\b', response)
        for num in reversed(numbers):
            try:
                val = float(num)
                if 0 < val < 1000:
                    return num
            except:
                pass
        
        return ""


# 注册工具的便捷函数
def create_gllava_tool(config=None):
    """创建G-LLaVA工具实例"""
    return GLLaVATool(config)