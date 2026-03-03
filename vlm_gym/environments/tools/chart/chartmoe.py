# vlm_gym/environments/tools/chart/chartmoe.py

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import json
import logging

from transformers import AutoModel, AutoProcessor
from ..base import ToolBase

logger = logging.getLogger(__name__)


class ChartMoETool(ToolBase):
    name = "chartmoe"
    """
    ChartMoE工具 - 多模态图表理解专家
    支持：Chart-to-Table、Chart-to-Text、Chart QA
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 设置工具属性
        self.name = "chartmoe"
        self.description = "多模态图表理解工具，支持图表转表格、图表描述生成、图表问答"
        
        # 传递参数给父类
        parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["to_table", "to_text", "to_caption", "answer", "custom"],
                    "description": "任务类型"
                },
                "prompt": {
                    "type": "string",
                    "description": "自定义prompt（当task为custom时必需）"
                },
                "question": {
                    "type": "string", 
                    "description": "当task为answer时，需要回答的问题"
                }
            },
            "required": []
        }
        
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=parameters
        )
        
        # 配置
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        self.model_name = self.config.get('model_name', 'IDEA-FinAI/chartmoe')
        self.model_path = self.config.get('model_path', None)  # 支持本地路径
        self.trust_remote_code = self.config.get('trust_remote_code', True)
        self.max_new_tokens = self.config.get('max_new_tokens', 512)
        
        # 预定义的prompt模板
        self.prompt_templates = {
            "to_table": "Convert this chart to a table format with clear rows and columns.",
            "to_text": "Describe this chart in detail, including all data points and trends.",
            "to_caption": "Generate a concise caption for this chart.",
            "answer": "Based on the chart, {question}"
        }
        
        # 模型延迟加载
        self.model = None
        self.processor = None
        self.current_image = None
        
        logger.info(f"ChartMoE tool initialized with config: {self.config}")
        
    def _load_model(self):
        """延迟加载模型 - 简化版本"""
        if self.model is None:
            model_id = self.model_path or self.model_name
            logger.info(f"Loading ChartMoE model: {model_id}")
            
            try:
                # 直接使用AutoModel加载，trust_remote_code会自动处理自定义类
                self.model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map="auto" if self.device == 'cuda' else None
                )
                
                # 加载processor
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=self.trust_remote_code
                )
                
                logger.info("ChartMoE model and processor loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load ChartMoE model: {e}")
                raise RuntimeError(
                    f"Could not load ChartMoE model '{model_id}'. "
                    f"Error: {str(e)}\n"
                    "Please ensure:\n"
                    "1. PyTorch >= 2.6 is installed\n"
                    "2. transformers >= 4.42 is installed\n"
                    "3. Internet connection is available for downloading model files"
                )
    
    def reset(self, image: Image.Image):
        """重置工具状态，准备处理新图像"""
        self._load_model()  # 确保模型已加载
        
        # 保存图像
        self.current_image = image.convert("RGB")
        logger.debug(f"ChartMoE reset with image size: {self.current_image.size}")
        
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行ChartMoE处理"""
        if self.current_image is None:
            return {
                "error": "No image loaded. Please call reset() first.",
                "error_type": "NoImageError"
            }
        
        # 解析参数
        if isinstance(action_string, str):
            try:
                params = json.loads(action_string)
            except json.JSONDecodeError:
                # 如果不是JSON，作为prompt直接使用
                params = {"task": "custom", "prompt": action_string}
        elif isinstance(action_string, dict):
            params = action_string
        else:
            params = {"task": "custom", "prompt": str(action_string)}
            
        # 获取任务类型和prompt
        task_type = params.get("task", "custom")
        custom_prompt = params.get("prompt", "")
        question = params.get("question", "")
        
        logger.debug(f"Executing ChartMoE - task: {task_type}, question: {question[:50] if question else 'N/A'}...")
        
        # 构建最终prompt
        if task_type in self.prompt_templates:
            if task_type == "answer" and question:
                prompt = self.prompt_templates["answer"].format(question=question)
            else:
                prompt = self.prompt_templates[task_type]
        elif custom_prompt:
            prompt = custom_prompt
        else:
            return {
                "error": "Either 'task' or 'prompt' parameter is required",
                "error_type": "MissingParameterError"
            }
        
        try:
            # 执行推理
            logger.debug(f"Running inference with prompt: {prompt}")
            
            # 使用processor准备输入
            inputs = self.processor(
                text=prompt,
                images=self.current_image,
                return_tensors="pt"
            )
            
            # 移动到正确的设备
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1
                )
            
            # 解码输出
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 移除输入prompt（如果包含在输出中）
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.debug(f"Generated text length: {len(generated_text)}")
            
            # 根据任务类型后处理
            processed_result = self._post_process(generated_text, task_type)
            
            return {
                "task_type": task_type,
                "prompt": prompt,
                "raw_output": generated_text,
                "processed_output": processed_result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"ChartMoE execution failed: {e}")
            return {
                "error": f"ChartMoE execution failed: {str(e)}",
                "error_type": type(e).__name__,
                "task_type": task_type,
                "prompt": prompt
            }
    
    def _post_process(self, text: str, task_type: str) -> Any:
        """后处理输出结果"""
        text = text.strip()
        
        if task_type == "to_table":
            # 尝试解析表格格式
            lines = text.split('\n')
            table_data = []
            
            for line in lines:
                if not line.strip():
                    continue
                    
                # 支持多种分隔符
                if '\t' in line:
                    row = [cell.strip() for cell in line.split('\t')]
                elif '|' in line and not line.strip().startswith('|---'):
                    # Markdown表格格式
                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                elif ',' in line:
                    row = [cell.strip() for cell in line.split(',')]
                else:
                    # 空格分隔
                    row = line.split()
                
                if row and any(cell for cell in row):
                    table_data.append(row)
            
            # 尝试识别表头
            has_header = False
            if table_data and len(table_data) > 1:
                # 简单启发式：第一行包含非数字内容
                first_row = table_data[0]
                if any(not self._is_numeric(cell) for cell in first_row):
                    has_header = True
            
            return {
                "format": "table",
                "data": table_data,
                "rows": len(table_data),
                "columns": len(table_data[0]) if table_data else 0,
                "has_header": has_header,
                "raw_text": text
            }
        
        elif task_type in ["to_text", "to_caption"]:
            return {
                "format": "text",
                "content": text,
                "length": len(text)
            }
        
        elif task_type == "answer":
            # 尝试提取简洁答案
            answer = text
            
            # 如果答案很长，尝试提取第一句或关键信息
            if len(text) > 100:
                # 查找第一个句号
                first_sentence_end = text.find('.')
                if first_sentence_end > 0:
                    answer = text[:first_sentence_end + 1]
            
            return {
                "format": "answer",
                "answer": answer.strip(),
                "full_response": text
            }
        
        else:
            # 自定义任务
            return {
                "format": "custom",
                "content": text
            }
    
    def _is_numeric(self, value: str) -> bool:
        """检查字符串是否为数字"""
        try:
            float(value.replace(',', '').replace('%', ''))
            return True
        except ValueError:
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": ["chart-to-table", "chart-to-text", "chart-qa", "chart-understanding"],
            "parameters": self.parameters,
            "supported_tasks": list(self.prompt_templates.keys()),
            "model": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "torch_version": torch.__version__,
            "examples": [
                {
                    "description": "将图表转换为表格",
                    "params": {"task": "to_table"}
                },
                {
                    "description": "生成图表描述",
                    "params": {"task": "to_text"}
                },
                {
                    "description": "回答图表相关问题",
                    "params": {"task": "answer", "question": "What is the highest value in the chart?"}
                },
                {
                    "description": "自定义prompt",
                    "params": {"prompt": "Extract all numerical values from this chart"}
                }
            ]
        }