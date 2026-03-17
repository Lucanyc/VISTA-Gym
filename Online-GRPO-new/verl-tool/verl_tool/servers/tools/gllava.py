"""
G-LLaVA Tool for VERL Framework
Geometry problem solver using G-LLaVA model with Ray Actor
"""

"""
G-LLaVA Tool for VERL Framework
Geometry problem solver using G-LLaVA model with Ray Actor
"""

import sys
import types

# 在gllava环境中创建假的qwen2_5_vl模块以避免导入错误
if 'transformers.models.qwen2_5_vl' not in sys.modules:
    # 创建假模块
    fake_qwen = types.ModuleType('transformers.models.qwen2_5_vl')
    fake_processing = types.ModuleType('transformers.models.qwen2_5_vl.processing_qwen2_5_vl')
    
    # 创建假的Processor类
    class FakeProcessor:
        pass
    
    fake_processing.Qwen2_5_VLProcessor = FakeProcessor
    
    # 注册到sys.modules
    sys.modules['transformers.models.qwen2_5_vl'] = fake_qwen
    sys.modules['transformers.models.qwen2_5_vl.processing_qwen2_5_vl'] = fake_processing


import json
import re
import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import traceback

import ray
import torch
from PIL import Image

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Ray Actor - 模型的唯一持有者
# ==============================================================================
@ray.remote(num_gpus=1)
class GLLaVAActor:
    """Ray Actor that holds the G-LLaVA model"""
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing G-LLaVA Actor on PID: {os.getpid()}")
        
        # 默认模型路径
        if model_path is None:
            model_path = "/data/models/G-LLaVA-13B"
        
        self.model_path = model_path
        
        try:
            # 添加G-LLaVA到系统路径
            gllava_base = "/home/meng/model/G-LLaVA"
            if gllava_base not in sys.path:
                sys.path.insert(0, gllava_base)
            
            # 处理AutoConfig注册冲突
            from transformers import AutoConfig
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            
            # 先尝试移除已存在的llava配置
            if hasattr(CONFIG_MAPPING, '_extra_content'):
                if 'llava' in CONFIG_MAPPING._extra_content:
                    del CONFIG_MAPPING._extra_content['llava']
            
            # 或者使用monkey patch来忽略重复注册错误
            original_register = AutoConfig.register
            def patched_register(model_type, config, exist_ok=True):
                try:
                    original_register(model_type, config, exist_ok=True)
                except ValueError as e:
                    if "already used by a Transformers config" in str(e):
                        self.logger.warning(f"Config already registered: {model_type}, skipping...")
                    else:
                        raise
            AutoConfig.register = patched_register
            
            # 现在安全地导入G-LLaVA模块
            from gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from gllava.conversation import conv_templates, SeparatorStyle
            from gllava.model.builder import load_pretrained_model
            from gllava.utils import disable_torch_init
            from gllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
            
            # 恢复原始register函数
            AutoConfig.register = original_register
            
            # 保存为实例变量
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
            self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
            self.conv_templates = conv_templates
            self.SeparatorStyle = SeparatorStyle
            self.tokenizer_image_token = tokenizer_image_token
            self.get_model_name_from_path = get_model_name_from_path
            self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
            self.process_images = process_images
            
            # 禁用torch初始化加速
            disable_torch_init()
            
            # 加载模型
            self.logger.info(f"Loading G-LLaVA model from: {model_path}")
            model_name = get_model_name_from_path(model_path)
            
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, 
                None,  # model_base
                model_name,
                load_8bit=False,
                load_4bit=False,
                device_map="auto"
            )
            
            # 设置对话模式
            self.conv_mode = "llava_v1"
            
            self.model_name = model_name
            self.device = next(self.model.parameters()).device
            
            self.logger.info(f"✓ G-LLaVA model loaded successfully")
            self.logger.info(f"  Model name: {model_name}")
            self.logger.info(f"  Context length: {self.context_len}")
            self.logger.info(f"  Conv mode: {self.conv_mode}")
            self.logger.info(f"  Device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize G-LLaVA: {e}")
            self.model = None
            self.tokenizer = None
            import traceback
            traceback.print_exc()
            raise


    
    def solve_geometry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """解决几何问题 - 增强版支持多种参数"""
        try:
            if not self.model:
                return {
                    "success": False,
                    "error": "G-LLaVA model is not available"
                }
            
            # 获取参数
            image_path = params.get('image_path')
            image_array = params.get('image_array')
            question = params.get('question', '')
            choices = params.get('choices', [])
            output_format = params.get('output_format', 'detailed')
            problem_type = params.get('problem_type', 'general')
            
            self.logger.debug(f"Processing geometry problem: {question[:100]}...")
            if choices:
                self.logger.debug(f"Choices provided: {choices}")
            
            # 1. 加载图像
            if image_array is not None:
                image = Image.fromarray(image_array.astype('uint8'), 'RGB')
                self.logger.debug("Using provided image array")
            elif image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.logger.debug(f"Loaded image from: {image_path}")
            else:
                return {
                    "success": False,
                    "error": f"Invalid image source"
                }
            
            # 2. 构建问题（支持选择题格式）
            formatted_question = question
            if choices and not any(marker in question for marker in ['Choices:', 'A:', 'B:']):
                formatted_question += "\n\nChoices:"
                for i, choice in enumerate(choices):
                    formatted_question += f"\n{chr(65+i)}: {choice}"
            
            # 3. 处理图像
            image_tensor = self.process_images([image], self.image_processor, self.model.config)
            
            if type(image_tensor) is list:
                image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            
            # 4. 创建对话
            conv = self.conv_templates[self.conv_mode].copy()
            
            # 构建输入
            if self.model.config.mm_use_im_start_end:
                formatted_input = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + formatted_question
            else:
                formatted_input = self.DEFAULT_IMAGE_TOKEN + '\n' + formatted_question
            
            conv.append_message(conv.roles[0], formatted_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # 5. Tokenize
            input_ids = self.tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)
            input_ids = input_ids.to(self.device)
            
            # 6. 设置停止条件
            stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            # 7. 生成响应
            max_tokens = 512 if output_format == 'detailed' else 256
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95,
                    num_beams=1,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            
            # 8. 解码输出
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], 
                skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            # 9. 提取答案
            final_answer, extraction_method = self._extract_answer_enhanced(outputs, question, choices)
            
            self.logger.debug(f"Generated response: {outputs[:200]}...")
            self.logger.debug(f"Extracted answer: {final_answer} (method: {extraction_method})")
            
            return {
                "success": True,
                "full_response": outputs,
                "answer": final_answer,
                "final_answer": final_answer,
                "extraction_method": extraction_method,
                "model": "G-LLaVA-13B",
                "method": "Geometric reasoning with G-LLaVA",
                "has_image": True,
                "response_length": len(outputs)
            }
            
        except Exception as e:
            self.logger.error(f"Geometry solving failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Solving failed: {str(e)}",
                "final_answer": None,
                "full_response": ""
            }
    
    def _extract_answer_enhanced(self, text: str, question: str, choices: List[str] = None) -> Tuple[str, str]:
        """增强的答案提取 - 支持选择题和普通题"""
        
        text = text.strip()
        
        if not text:
            return "No answer", "empty_response"
        
        # 1. 如果是选择题
        if choices:
            # 选择题答案模式
            choice_patterns = [
                r"(?:the\s+)?answer\s+is\s+(?:option\s+)?([A-Da-d])\b",
                r"Answer:\s*([A-Da-d])\b",
                r"(?:Therefore|Thus|Hence|So),?\s+(?:the\s+)?answer\s+is\s+([A-Da-d])\b",
                r"(?:correct\s+answer\s+is\s+)?([A-Da-d])\b",
                r"(?:option|选项)\s*([A-Da-d])\s*(?:is|为|是)\s*(?:correct|正确)",
            ]
            
            for pattern in choice_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    answer = matches[-1].upper()
                    max_valid = chr(65 + len(choices) - 1)
                    if answer <= max_valid:
                        return answer, "choice_pattern"
            
            # 查找文本末尾的单独字母
            last_lines = '\n'.join(text.split('\n')[-3:])
            letter_matches = re.findall(r'\b([A-Da-d])\b', last_lines)
            if letter_matches:
                answer = letter_matches[-1].upper()
                max_valid = chr(65 + len(choices) - 1)
                if answer <= max_valid:
                    return answer, "fallback_letter"
        
        # 2. 查找数值答案（角度）
        degree_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:°|degrees?)',
            r'angle\s+is\s+(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*(?:°|degrees?)?'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return f"{matches[-1]}°", "degree_pattern"
        
        # 3. 查找面积/长度等数值答案
        numeric_patterns = [
            r"(?:area|perimeter|length|value)\s+is\s+(\d+(?:\.\d+)?)",
            r"(?:equals?|is)\s+(\d+(?:\.\d+)?)",
            r"=\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1], "numeric_pattern"
        
        # 4. 提取第一个数字作为备选
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[0], "first_number"
        
        # 5. 返回文本片段
        return text[:50].strip(), "text_snippet"
    
    def health_check(self) -> str:
        """健康检查"""
        if self.model is not None:
            return f"healthy: model loaded from {self.model_path}"
        else:
            return "unhealthy: model not loaded"

# ==============================================================================
# 2. G-LLaVATool - 符合verl-tool框架
# ==============================================================================
@register_tool
class GLLaVATool(BaseTool):
    tool_type = "gllava"
    
    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 获取模型路径
        self.model_path = model_path or os.environ.get(
            'GLLAVA_MODEL_PATH',
            "/data/models/G-LLaVA-13B"
        )
        
        self.logger.info(f"Using G-LLaVA model path: {self.model_path}")
        
        # 连接到Ray Actor
        self.logger.info("Connecting to G-LLaVA Actor...")
        try:
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = GLLaVAActor.options(
                name="GLLaVAActor",
                get_if_exists=True,
                num_gpus=num_gpus,
                max_concurrency=2
            ).remote(model_path=self.model_path)
            
            # 健康检查
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=500)
            
            if "healthy" in health_status:
                self.logger.info(f"✓ G-LLaVA Actor is {health_status}")
            else:
                self.logger.error(f"G-LLaVA Actor health check failed: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to G-LLaVA Actor: {e}")
            self.actor = None
    
    def get_usage_inst(self):
        """返回详细的工具使用说明"""
        return ('G-LLaVA: Geometry problem solver. '
                'Use <tool_call>{"tool": "gllava", "task": "solve", '
                '"question": "problem text", "choices": ["A", "B", "C", "D"], '
                '"output_format": "detailed|brief"}</tool_call>')
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数 - 增强版支持完整参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证工具名
            if params.get('tool') != 'gllava':
                return {}, False
            
            # 支持嵌套格式（像MultiMath）
            if 'parameters' in params and isinstance(params['parameters'], dict):
                inner_params = params['parameters']
            else:
                inner_params = params
            
            # 提取所有参数
            parsed = {
                'task': inner_params.get('task', 'solve'),
                'question': inner_params.get('question', '') or inner_params.get('prompt', ''),
                'choices': inner_params.get('choices', []),
                'problem_type': inner_params.get('problem_type', 'general'),
                'output_format': inner_params.get('output_format', 'detailed')
            }
            
            # 验证必要参数
            if not parsed['question']:
                self.logger.warning("No question found in params")
                return {}, False
            
            self.logger.debug(f"Parsed parameters: question={parsed['question'][:50]}..., "
                            f"choices={len(parsed['choices'])} options, "
                            f"output_format={parsed['output_format']}")
            
            return parsed, True
            
        except Exception as e:
            self.logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def _get_image_path(self, extra_field: Dict[str, Any], env: Dict[str, Any]) -> Optional[str]:
        """从extra_field或环境中获取图片路径 - 支持verl-tool格式"""
        
        # 优先处理verl-tool格式的extra_info（包含images列表）
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[G-LLaVA] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.debug(f"[G-LLaVA] Found image path: {img_path}")
                        return img_path
                    else:
                        self.logger.warning(f"[G-LLaVA] Image path does not exist: {img_path}")
        
        # 向后兼容：原有的image字段处理
        for field_name in ['image', 'image_path']:
            if extra_field and field_name in extra_field:
                img_path = extra_field[field_name]
                if isinstance(img_path, str) and os.path.exists(img_path):
                    self.logger.debug(f"[G-LLaVA] Found image in extra_field['{field_name}']: {img_path}")
                    return img_path
        
        # 从环境中获取
        if 'current_image_path' in env:
            img_path = env['current_image_path']
            if os.path.exists(img_path):
                self.logger.debug(f"[G-LLaVA] Using cached image from environment: {img_path}")
                return img_path
        
        self.logger.debug("[G-LLaVA] No image found in extra_field or environment")
        return None
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行几何问题求解 - 优化版"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Error: Invalid action format. Expected <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, False
        
        try:
            # 调试日志
            self.logger.debug(f"[G-LLaVA] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[G-LLaVA] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图片路径
            image_path = self._get_image_path(extra_field, env)
            
            if not image_path:
                observation = "Error: No image provided for geometry problem"
                self.logger.error(f"[G-LLaVA] {observation}")
                self.logger.debug(f"[G-LLaVA] extra_field content: {extra_field}")
            elif self.actor is None:
                observation = "Error: G-LLaVA Actor is not available"
                self.logger.error(observation)
            else:
                # 准备参数（使用从action解析的参数）
                solve_params = {
                    'image_path': image_path,
                    'question': parsed_params['question'],
                    'choices': parsed_params['choices'],
                    'problem_type': parsed_params['problem_type'],
                    'output_format': parsed_params['output_format']
                }
                
                self.logger.debug(f"[G-LLaVA] Solving geometry problem with image: {image_path}")
                self.logger.debug(f"[G-LLaVA] Question: {solve_params['question'][:100]}...")
                
                # 调用Actor求解
                result_ref = self.actor.solve_geometry.remote(solve_params)
                try:
                    result = ray.get(result_ref, timeout=120)
                    
                    if result.get("success"):
                        # 构建详细的观察结果
                        observation_parts = []
                        
                        # 完整响应
                        if result.get('full_response'):
                            observation_parts.append("Mathematical Solution:")
                            observation_parts.append("="*50)
                            observation_parts.append(result['full_response'])
                            observation_parts.append("="*50)
                        
                        # 最终答案
                        final_answer = result.get('final_answer') or result.get('answer', 'No answer')
                        observation_parts.append(f"\nFinal Answer: {final_answer}")
                        
                        # 提取方法（调试用）
                        if result.get('extraction_method'):
                            observation_parts.append(f"(Extraction method: {result['extraction_method']})")
                        
                        observation = "\n".join(observation_parts)
                        self.logger.info(f"[G-LLaVA] Successfully solved, answer: {final_answer}")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        observation = f"Error: {error_msg}"
                        self.logger.error(f"[G-LLaVA] Solving failed: {error_msg}")
                        
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: G-LLaVA solving timed out (120s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"G-LLaVA tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any,
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境状态"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)
        
        # 缓存图片路径
        image_path = self._get_image_path(extra_field, env)
        if image_path and 'current_image_path' not in env:
            env['current_image_path'] = image_path