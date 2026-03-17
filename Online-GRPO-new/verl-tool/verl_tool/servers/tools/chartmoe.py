"""
ChartMoE Tool for VERL Framework 
Modified to support verl-tool extra_info format with enhanced consistency
"""

import sys
sys.path.insert(0, "/data/cache/models--IDEA-FinAI--chartmoe/snapshots/951f46f96e307fa91c68ba9f318b30e6f9ce4405")

import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

import ray
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTool, register_tool

# --- 移除强制离线设置，允许自动下载依赖 ---
os.environ["HF_TORCH_LOAD_DISABLE_SAFE_CHECK"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 定义 Ray Actor - 模型的唯一持有者
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class ChartMoEActor:
    """一个有状态的Ray Actor,从本地路径加载模型"""
    def __init__(self, model_path: str = "/data/cache/models--IDEA-FinAI--chartmoe/snapshots/951f46f96e307fa91c68ba9f318b30e6f9ce4405"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing ChartMoEActor on PID: {os.getpid()}, Device: {self.device}")
        
        # 验证路径
        if not os.path.exists(model_path):
            self.logger.error(f"Model path does not exist: {model_path}")
            self.model = None
            self.tokenizer = None
            return
            
        # 验证必要文件
        required_files = ["config.json", "tokenizer.model", "model.safetensors.index.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                self.logger.warning(f"Missing file: {file}")

        # 加载模型和分词器
        self.model_path = model_path
        self.model, self.tokenizer = self._load_model(model_path)

        # 图像预处理
        self.image_size = 490
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        if self.model:
            self.logger.info("✓ ChartMoE Actor initialized successfully.")
        else:
            self.logger.error("❌ ChartMoE Actor failed to initialize model.")

    def _load_model(self, model_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """从指定的本地路径加载模型，允许自动下载依赖"""
        self.logger.info(f"Loading model from LOCAL path: {model_path}")
        self.logger.info("Note: CLIP dependencies will be downloaded automatically if needed")
        
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            
            # 加载模型，local_files_only=False 允许下载缺失的依赖
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map,
                local_files_only=False
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False
            )
            
            self.logger.info(f"✓ Model loaded successfully from: {model_path}")
            
            # 打印模型信息
            model_size = sum(p.numel() for p in model.parameters()) / 1e9
            self.logger.info(f"Model size: {model_size:.2f}B parameters")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load ChartMoE model: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def inference(self, image_data: np.ndarray, params: Dict[str, Any]) -> str:
        """执行模型推理 - 增强版支持更多参数"""
        self.logger.info(f"[ChartMoE] Received image: shape={image_data.shape}, params={params}")
        
        if not self.model:
            return "Error: ChartMoE model is not available in the Actor."
        
        try:
            image = Image.fromarray(image_data.astype('uint8'), 'RGB')
            
            # 获取参数
            task = params.get('task', 'to_table')
            prompt = params.get('prompt', '')
            question = params.get('question', '')
            output_format = params.get('output_format', 'detailed')
            
            # 预定义的任务提示
            task_prompts = {
                'to_table': 'Convert this chart to a table format with clear rows and columns.',
                'describe': 'Describe this chart in detail.',
                'extract_data': 'Extract all numerical data values and labels from this chart.',
                'summarize': 'Provide a brief summary of what this chart shows.',
                'analyze': 'Analyze this chart and provide key insights and patterns.',
                'answer': '',  # 使用自定义问题
            }
            
            # 确定最终的查询内容
            if task == 'answer':
                # Q&A任务优先使用question，其次是prompt
                query = question if question else prompt
                if not query:
                    query = "What does this chart show?"
            elif prompt:
                # 如果有自定义prompt，使用它
                query = prompt
            else:
                # 使用预定义的任务提示
                query = task_prompts.get(task, "Analyze this chart.")
            
            # 根据output_format调整提示
            if output_format == 'brief' and task != 'answer':
                query += " Be concise."
            elif output_format == 'detailed' and task != 'answer':
                query += " Provide a comprehensive response."

            formatted_query = f"<ImageHere>{query}"
            
            # 处理图像
            image_tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(next(self.model.parameters()).device)
            if torch.cuda.is_available():
                image_tensor = image_tensor.half()

            self.logger.debug(f"Image tensor shape: {image_tensor.shape}")
            self.logger.debug(f"Query: {formatted_query}")
            
            # 生成响应
            response = self.model.chat(
                tokenizer=self.tokenizer, 
                query=formatted_query, 
                image=image_tensor,
                history=[], 
                max_new_tokens=800 if output_format == 'detailed' else 400,
                do_sample=False, 
                use_cache=True
            )
            
            result = response[0] if isinstance(response, tuple) else response
            
            self.logger.debug(f"Generated response (first 200 chars): {result[:200]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}", exc_info=True)
            return f"Error: Inference failed: {str(e)}"

    def health_check(self) -> str:
        """健康检查"""
        if self.model is not None:
            return f"healthy: model loaded from {self.model_path}"
        else:
            return "unhealthy: model not loaded"

# ==============================================================================
# 2. ChartMoETool - 增强一致性
# ==============================================================================
@register_tool
class ChartMoETool(BaseTool):
    tool_type = "chartmoe"
    
    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 获取模型路径
        self.model_path = model_path or os.environ.get(
            'CHARTMOE_MODEL_PATH',
            "/data/cache/models--IDEA-FinAI--chartmoe/snapshots/951f46f96e307fa91c68ba9f318b30e6f9ce4405"
        )
        
        self.logger.info(f"Using model path: {self.model_path}")
        
        # 验证路径
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model path does not exist: {self.model_path}")
        
        self.logger.info("Connecting to ChartMoE Actor...")
        try:
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = ChartMoEActor.options(
                name="ChartMoEActor", 
                get_if_exists=True, 
                num_gpus=num_gpus, 
                max_concurrency=3
            ).remote(model_path=self.model_path)
            
            self.logger.info("Performing health check...")
            health_status = ray.get(self.actor.health_check.remote(), timeout=700)
            
            if "healthy" in health_status:
                self.logger.info(f"✓ ChartMoE Actor is {health_status}")
            else:
                self.logger.error(f"ChartMoE Actor health check failed: {health_status}")
                
        except Exception as e:
            self.logger.error(f"Failed to create or connect to ChartMoE Actor: {e}")
            self.actor = None

    def get_usage_inst(self):
        """返回详细的工具使用说明"""
        return ('ChartMoE: Chart analysis tool. '
                'Use <tool_call>{"tool": "chartmoe", "task": "to_table|describe|extract_data|summarize|analyze|answer", '
                '"prompt": "optional custom prompt", "question": "for Q&A tasks", "output_format": "detailed|brief"}</tool_call>')

    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析工具调用参数 - 增强版支持嵌套格式"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches: 
            return {}, False
            
        try:
            params = json.loads(matches[0].strip())
            
            # 验证工具名
            if params.get('tool') != 'chartmoe': 
                return {}, False
            
            # 支持嵌套格式（像MultiMath）
            if 'parameters' in params and isinstance(params['parameters'], dict):
                inner_params = params['parameters']
            else:
                inner_params = params
            
            # 提取所有可能的参数
            parsed = {
                'task': inner_params.get('task', 'to_table'),
                'prompt': inner_params.get('prompt', ''),
                'question': inner_params.get('question', ''),
                'output_format': inner_params.get('output_format', 'detailed'),
            }
            
            # 如果有question但没有prompt，使用question作为prompt
            if parsed['question'] and not parsed['prompt']:
                parsed['prompt'] = parsed['question']
            
            # 验证task是否有效
            valid_tasks = ['to_table', 'describe', 'extract_data', 'summarize', 'analyze', 'answer']
            if parsed['task'] not in valid_tasks:
                self.logger.warning(f"Invalid task: {parsed['task']}, defaulting to 'analyze'")
                parsed['task'] = 'analyze'
            
            self.logger.debug(f"Parsed parameters: task={parsed['task']}, "
                            f"prompt={parsed['prompt'][:50] if parsed['prompt'] else 'None'}..., "
                            f"output_format={parsed['output_format']}")
            
            return parsed, True
            
        except Exception as e:
            self.logger.error(f"Error parsing action: {e}")
            return {}, False

    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行ChartMoE工具调用 - 优化版"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Error: Invalid action format. Expected <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, False  # 注意：无效时返回False, False
        
        try:
            # 调试日志：查看收到的extra_field
            self.logger.debug(f"[ChartMoE] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[ChartMoE] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图像
            image = self._get_image(trajectory_id, extra_field, env)
            
            if image is None:
                observation = "Error: No image provided for chart analysis"
                self.logger.error(f"[ChartMoE] {observation}")
                self.logger.debug(f"[ChartMoE] extra_field content: {extra_field}")
            elif self.actor is None:
                observation = "Error: ChartMoE Actor is not available"
                self.logger.error(observation)
            else:
                # 使用parsed_params中的参数
                self.logger.debug(f"[ChartMoE] Processing with task: {parsed_params['task']}, "
                                f"prompt: {parsed_params.get('prompt', '')[:50]}...")
                
                image_array = np.array(image.convert('RGB'))
                
                # 传递完整的parsed_params到Actor
                result_ref = self.actor.inference.remote(image_array, parsed_params)
                try:
                    result = ray.get(result_ref, timeout=900)
                    observation = self._format_output(result, parsed_params['task'])
                    self.logger.debug(f"[ChartMoE] Successfully analyzed chart")
                except ray.exceptions.GetTimeoutError:
                    observation = "Error: ChartMoE inference timed out (900s)"
                    self.logger.error(observation)
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"ChartMoE tool conduct_action failed: {e}", exc_info=True)
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True

    def _get_image(self, trajectory_id: str, extra_field: Dict[str, Any], env: Dict[str, Any]) -> Optional[Image.Image]:
        """从extra_field或环境中获取图像 - 支持verl-tool格式"""
        
        # 优先处理verl-tool格式的extra_info（包含images列表）
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[ChartMoE] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                # 获取第一个图片路径
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.debug(f"[ChartMoE] Loading image from path: {img_path}")
                        return Image.open(img_path).convert('RGB')
                    else:
                        self.logger.warning(f"[ChartMoE] Image path does not exist: {img_path}")
                else:
                    self.logger.warning(f"[ChartMoE] Invalid image path format: {img_path}")
        
        # 向后兼容：原有的image字段处理
        if extra_field and 'image' in extra_field and extra_field['image']:
            img_data = extra_field['image']
            if isinstance(img_data, Image.Image): 
                self.logger.debug("[ChartMoE] Using PIL Image from extra_field['image']")
                return img_data
            if isinstance(img_data, str) and os.path.exists(img_data): 
                self.logger.debug(f"[ChartMoE] Loading image from extra_field['image']: {img_data}")
                return Image.open(img_data).convert('RGB')
        
        # 向后兼容：原有的image_path字段处理
        if extra_field and 'image_path' in extra_field and extra_field['image_path']:
            path_info = extra_field['image_path']
            if isinstance(path_info, list) and path_info:
                img_path = path_info[0].get("image_url") if isinstance(path_info[0], dict) else path_info[0]
                if img_path and os.path.exists(img_path): 
                    self.logger.debug(f"[ChartMoE] Loading image from extra_field['image_path']: {img_path}")
                    return Image.open(img_path).convert('RGB')
            elif isinstance(path_info, str) and os.path.exists(path_info):
                self.logger.debug(f"[ChartMoE] Loading image from extra_field['image_path']: {path_info}")
                return Image.open(path_info).convert('RGB')
        
        # 从环境缓存中获取
        if 'current_image' in env: 
            self.logger.debug("[ChartMoE] Using cached image from environment")
            return env['current_image']
        
        self.logger.warning("[ChartMoE] No image found in any expected location")
        return None

    def _format_output(self, result: str, task: str) -> str:
        """格式化输出结果"""
        if result.startswith("Error:"): 
            return result
        
        # 为表格任务添加格式化
        if task == 'to_table' and '|' not in result:
            # 尝试将文本转换为表格格式
            lines = result.strip().split('\n')
            if len(lines) > 1:
                formatted_lines = []
                for line in lines:
                    if line.strip():
                        # 如果行中有制表符或逗号，尝试分割
                        if '\t' in line:
                            cells = line.split('\t')
                            formatted_lines.append('| ' + ' | '.join(cells) + ' |')
                        elif ',' in line:
                            cells = line.split(',')
                            formatted_lines.append('| ' + ' | '.join(cells) + ' |')
                        else:
                            formatted_lines.append(f"| {line.strip()} |")
                
                # 添加表头分隔符（如果看起来像表格）
                if len(formatted_lines) > 1:
                    # 检查第一行是否可能是表头
                    first_cells = formatted_lines[0].count('|') - 1
                    if first_cells > 1:
                        separator = '|' + '---|' * first_cells
                        formatted_lines.insert(1, separator)
                
                return '\n'.join(formatted_lines)
        
        return result

    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any, 
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境，包括保存当前图像"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)
        
        # 缓存图像以供后续使用
        image_to_cache = self._get_image(trajectory_id, extra_field, env)
        if image_to_cache and 'current_image' not in env:
            env['current_image'] = image_to_cache