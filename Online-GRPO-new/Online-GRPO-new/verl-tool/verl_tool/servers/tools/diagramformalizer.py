# verl_tool/servers/tools/diagramformalizer.py
import json
import re
import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import time
import base64
from io import BytesIO

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import traceback

from .base import BaseTool, register_tool

# 设置日志
logger = logging.getLogger(__name__)

# 默认 prompt
DEFAULT_PROMPT = 'Based on the image, first describe what you see in the figure, then predict the construction_cdl and image_cdl and calibrate it.'

# ==============================================================================
# 1. 定义 Ray Actor - 模型的唯一持有者
# ==============================================================================
@ray.remote(num_gpus=1)
class DiagramFormalizerActor:
    """Ray Actor that holds the DiagramFormalizer model.
    
    IMPORTANT: max_concurrency must be 1. PyTorch model.generate() is NOT
    thread-safe — concurrent calls corrupt KV cache and position_ids,
    causing CUDA device-side assert errors that permanently break the GPU.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing DiagramFormalizerActor on PID: {os.getpid()}")
        
        try:
            # 设置设备
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch.set_default_device(self.device)
            self.logger.info(f"Using device: {self.device}")
            
            # 加载模型和 tokenizer
            self.logger.info("Loading DiagramFormalizer model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                'NaughtyDog97/DiagramFormalizer',
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto',
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                'NaughtyDog97/DiagramFormalizer',
                use_fast=True,
                padding_side="right",
                trust_remote_code=True
            )
            
            self.logger.info(f"✓ DiagramFormalizer model loaded successfully")
            self.logger.info(f"  vocab_size={self.model.config.vocab_size}, "
                           f"embedding={self.model.get_input_embeddings().weight.shape}, "
                           f"tokenizer_len={len(self.tokenizer)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DiagramFormalizer: {e}")
            raise
    
    def formalize_diagram(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """形式化几何图形"""
        try:
            image_data = params.get('image_data')
            image_path = params.get('image_path')
            # 修复：当 prompt 为 None 时正确使用默认值
            prompt = params.get('prompt') or DEFAULT_PROMPT
            max_new_tokens = params.get('max_new_tokens', 3500)
            
            results = {
                'construction_cdl': None,
                'image_cdl': None,
                'description': None,
                'full_response': None,
                'success': False
            }
            
            # 加载图像
            if image_data:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            elif image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                results['error'] = "No valid image provided"
                return results
            
            self.logger.debug(f"Processing image: {image_path}, size={image.size}")
            
            # 直接调用生成，不使用 func_timeout
            # func_timeout 通过线程异常中断 CUDA 操作会永久损坏 GPU 上下文
            try:
                response = self._generate_formalization(image, prompt, max_new_tokens)
                
                if response:
                    results['full_response'] = response
                    results['success'] = True
                    
                    # 解析 CDL
                    cdl_info = self._parse_cdl(response)
                    results['construction_cdl'] = cdl_info.get('construction_cdl')
                    results['image_cdl'] = cdl_info.get('image_cdl')
                    
                    # 提取描述（第一句话）
                    desc_match = re.search(r'^(.*?(?:construction_cdl|The construction_cdl))', response, re.DOTALL)
                    if desc_match:
                        results['description'] = desc_match.group(1).strip()
                    
            except Exception as e:
                results['error'] = f"Generation failed: {str(e)}"
                self.logger.error(f"Generation error: {e}", exc_info=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Formalization failed: {e}", exc_info=True)
            return {"error": f"Formalization failed: {str(e)}"}
    
    def _generate_formalization(self, image, prompt, max_new_tokens=3500):
        """生成形式化描述
        
        NOTE: 不使用 func_timeout。生成长度通过 max_new_tokens 控制。
        
        IMPORTANT: model.generate() 中必须显式传 temperature=None, top_p=None,
        top_k=None, repetition_penalty=None。否则模型的默认 generation_config
        中的 repetition_penalty 会触发 RepetitionPenaltyLogitsProcessor，该
        processor 会对 input_ids 做 scatter/gather 操作。而 input_ids 中包含
        -200（图像占位符），这个负数索引会导致 CUDA index out of bounds 错误。
        """
        # 构建输入
        text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        
        # Tokenize
        input_ids = self._tokenizer_image_token(text, self.tokenizer, -200, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(self.device)
        
        # 处理图像
        image_tensor = self.model.process_images([image], self.model.config)
        image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.device)
        
        # 生成
        # 必须显式设置 temperature/top_p/top_k/repetition_penalty 为 None，
        # 覆盖模型默认的 generation_config，避免 logits processor 对含有
        # -200 的 input_ids 进行非法索引操作
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=None,
                use_cache=True
            )[0]
        
        # 解码
        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response
    
    def _tokenizer_image_token(self, prompt, tokenizer, image_token_index, return_tensors=None):
        """处理包含图像标记的文本"""
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    
    def _parse_cdl(self, input_string):
        """解析 CDL 信息"""
        patterns = {
            'construction_cdl': r'(?:The )?(?:calibrate )?construction_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
            'image_cdl': r'(?:The )?(?:calibrate )?image_cdl(?: is)?:\n(.*?)(?=\n(?:The )?(?:calibrate )?\w+_cdl is:|\n(?:The )?(?:calibrate )?\w+_cdl:|\nSolution is:|\Z)',
        }
        
        results = {}
        for key, pattern in patterns.items():
            pattern = pattern.replace("(?:calibrate )?", "(?:calibrate )")
            match = re.search(pattern, input_string, re.DOTALL)
            if match:
                results[key] = match.group(1).strip()
            else:
                pattern = pattern.replace("(?:calibrate )", "(?:calibrate )?")
                match = re.search(pattern, input_string, re.DOTALL)
                if match:
                    results[key] = match.group(1).strip()
        
        return results
    
    def health_check(self) -> str:
        """健康检查"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                return f"healthy: model loaded, device={self.device}"
            else:
                return "unhealthy: model not loaded"
        except:
            return "unhealthy: error checking status"

# ==============================================================================
# 2. DiagramFormalizerTool - 修改以支持verl-tool格式
# ==============================================================================
@register_tool
class DiagramFormalizerTool(BaseTool):
    tool_type = "diagramformalizer"
    
    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("Connecting to DiagramFormalizer Actor...")
        try:
            # 创建或连接到 Ray Actor
            # CRITICAL: max_concurrency=1，不能大于1！
            # PyTorch model.generate() 不是线程安全的，并发调用会导致
            # KV cache / position_ids 状态冲突，触发 CUDA device-side assert，
            # 永久破坏该 GPU 上下文。
            self.actor = DiagramFormalizerActor.options(
                name="DiagramFormalizerActor",
                get_if_exists=True,
                num_gpus=1,
                max_concurrency=1  # 必须为1！并发会导致CUDA错误
            ).remote()
            
            # 健康检查
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=30)
            
            self.logger.info(f"✓ DiagramFormalizer Actor status: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to DiagramFormalizer Actor: {e}")
            self.actor = None
    
    def get_usage_inst(self):
        """返回工具使用说明"""
        return 'DiagramFormalizer: Formalize geometry diagrams to CDL format. Use <tool_call>{"tool": "diagramformalizer", "task": "formalize"}</tool_call>'
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            if params.get('tool') != 'diagramformalizer':
                return {}, False
            
            parsed_params = {
                'task': params.get('task', 'formalize'),
                'prompt': params.get('prompt'),
                'time_limit': params.get('time_limit', 60),
                'max_new_tokens': params.get('max_new_tokens', 3500),
                'raw_params': params
            }
            
            return parsed_params, True
            
        except Exception as e:
            logger.error(f"Error parsing action: {e}")
            return {}, False
    
    def _get_image_path(self, extra_field: Dict[str, Any]) -> Optional[str]:
        """从extra_field中获取图片路径 - 支持verl-tool格式"""
        
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[DiagramFormalizer] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.debug(f"[DiagramFormalizer] Found image path: {img_path}")
                        return img_path
                    else:
                        self.logger.warning(f"[DiagramFormalizer] Image path does not exist: {img_path}")
                else:
                    self.logger.warning(f"[DiagramFormalizer] Invalid image path format: {img_path}")
        
        if extra_field and 'image' in extra_field:
            img_path = extra_field['image']
            if isinstance(img_path, str) and os.path.exists(img_path):
                self.logger.debug(f"[DiagramFormalizer] Found image in extra_field['image']: {img_path}")
                return img_path
        
        if extra_field and 'image_path' in extra_field:
            img_path = extra_field['image_path']
            if isinstance(img_path, str) and os.path.exists(img_path):
                self.logger.debug(f"[DiagramFormalizer] Found image in extra_field['image_path']: {img_path}")
                return img_path
        
        self.logger.warning("[DiagramFormalizer] No image found in extra_field")
        return None
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行图形形式化"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            self.update_env(trajectory_id, env, action, False, extra_field, "")
            self.save_env(trajectory_id, env)
            return "", False, False
        
        try:
            self.logger.debug(f"[DiagramFormalizer] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[DiagramFormalizer] Processing task_id: {extra_field.get('qid')}")
            
            image_path = self._get_image_path(extra_field)
            
            if not image_path:
                observation = "Error: No image provided for diagram formalization"
                self.logger.error(f"[DiagramFormalizer] {observation}")
                self.logger.debug(f"[DiagramFormalizer] extra_field content: {extra_field}")
            elif self.actor is None:
                observation = "Error: DiagramFormalizer Actor is not available"
            else:
                formalize_params = {
                    'image_path': image_path,
                    'image_data': None,
                    'prompt': parsed_params.get('prompt'),
                    'max_new_tokens': parsed_params.get('max_new_tokens', 3500),
                }
                
                self.logger.debug(f"[DiagramFormalizer] Processing image from: {image_path}")
                
                # 调用 Actor 进行形式化
                # 超时由 ray.get 的 timeout 控制，不会破坏 CUDA 上下文
                result_ref = self.actor.formalize_diagram.remote(formalize_params)
                try:
                    result = ray.get(result_ref, timeout=parsed_params['time_limit'] + 10)
                    
                    if "error" in result:
                        observation = f"Error: {result['error']}"
                        self.logger.warning(f"[DiagramFormalizer] {observation}")
                    else:
                        observation = self._format_result(result)
                        qid = extra_field.get('qid', 'unknown') if extra_field else 'unknown'
                        self.logger.info(f"[DiagramFormalizer] ✓ Formalized {qid} ({image_path})")
                        self.logger.info(f"[DiagramFormalizer] CDL Result:\n{observation[:500]}")
                        
                except ray.exceptions.GetTimeoutError:
                    observation = f"Error: DiagramFormalizer timed out ({parsed_params['time_limit']}s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"DiagramFormalizer tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """格式化结果"""
        lines = []
        
        if result.get('full_response'):
            lines.append("Full Response:")
            lines.append("=" * 50)
            lines.append(result['full_response'])
            lines.append("=" * 50)
            lines.append("")
        
        if result.get('description'):
            lines.append("Description:")
            lines.append(result['description'])
            lines.append("")
        
        if result.get('construction_cdl'):
            lines.append("Construction CDL:")
            lines.append(result['construction_cdl'])
            lines.append("")
        
        if result.get('image_cdl'):
            lines.append("Image CDL:")
            lines.append(result['image_cdl'])
            lines.append("")
        
        if result.get('success'):
            lines.append("Status: Successfully formalized")
        else:
            lines.append("Status: Formalization failed")
        
        return "\n".join(lines)
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any,
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境状态"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)