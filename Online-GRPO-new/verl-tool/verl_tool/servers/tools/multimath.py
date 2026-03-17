import sys
sys.path.insert(0, "/data/models/MultiMath")

import json
import re
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from PIL import Image
import torch
from pathlib import Path

import ray
from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Ray Actor - 修复了所有问题的版本
# ==============================================================================
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class MultiMathActor:
    """Ray Actor that holds the MultiMath model locally"""
    
    def __init__(self, model_path: str = "/data/models/MultiMath"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Initializing MultiMathActor on PID: {os.getpid()}, Device: {self.device}")
        
        # 初始化
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_path = model_path
        self.model_name = None
        self.context_len = None
        
        # 验证路径
        if not os.path.exists(model_path):
            self.logger.error(f"Model path does not exist: {model_path}")
            return
        
        # 加载模型
        try:
            sys.path.insert(0, model_path)
            
            # 直接导入LLaVA模块
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            
            self.logger.info(f"Loading model from: {model_path}")
            
            # 直接使用已知的模型路径
            model_path_full = os.path.join(model_path, "checkpoints", "multimath-7b-llava-v1.5")
            
            if not os.path.exists(model_path_full):
                self.logger.error(f"Model not found at: {model_path_full}")
                return
            
            self.logger.info(f"Found model at: {model_path_full}")
            
            # 加载模型
            self.model_name = get_model_name_from_path(model_path_full)
            self.logger.info(f"Model name detected: {self.model_name}")
            
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path_full,
                model_base=None,
                model_name=self.model_name,
                load_8bit=False,
                load_4bit=False,
                device=self.device
            )
            
            # 保存必要的函数和常量
            self.process_images = process_images
            self.tokenizer_image_token = tokenizer_image_token
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.conv_templates = conv_templates
            
            # 设置pad_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.logger.info("✓ Model loaded successfully")
            self.logger.info(f"  - Model type: {type(self.model).__name__}")
            self.logger.info(f"  - Model name: {self.model_name}")
            self.logger.info(f"  - Tokenizer type: {type(self.tokenizer).__name__}")
            self.logger.info(f"  - Context length: {self.context_len}")
            self.logger.info(f"  - Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    def solve_math(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """解决数学问题 - 修复了所有设备和tokenization问题"""
        
        # 检查模型
        if self.model is None or self.tokenizer is None:
            return {
                "success": False,
                "error": "Model or tokenizer not loaded"
            }
        
        try:
            # 提取参数
            question = params.get('question', '')
            choices = params.get('choices', [])
            problem_type = params.get('problem_type', 'general')
            output_format = params.get('output_format', 'detailed')
            image_path = params.get('image_path', None)
            image_array = params.get('image_array', None)
            
            if not question:
                return {"success": False, "error": "No question provided"}
            
            self.logger.info(f"Processing question: {question[:100]}...")
            self.logger.info(f"Problem type: {problem_type}, Output format: {output_format}")
            if choices:
                self.logger.info(f"Choices: {choices}")
            if image_path:
                self.logger.info(f"Using image from: {image_path}")
            
            # 1. 准备图像
            image = None
            image_tensor = None
            
            if image_array is not None:
                image = Image.fromarray(image_array.astype('uint8'), 'RGB')
                self.logger.info(f"Using provided image array")
            elif image_path and os.path.exists(image_path):
                image = Image.open(image_path)
                # 确保图像是RGB格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                self.logger.info(f"Loaded image from path: {image_path}, size: {image.size}")
            
            # 处理图像并移到正确设备
            if image is not None:
                self.logger.debug("Processing image...")
                image_tensor = self.process_images([image], self.image_processor, self.model.config)
                if self.device == 'cuda':
                    model_dtype = next(self.model.parameters()).dtype
                    image_tensor = image_tensor.to(self.model.device, dtype=model_dtype)
                else:
                    image_tensor = image_tensor.to('cpu')
                self.logger.debug(f"Image tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
            
            # 2. 构建prompt
            if choices:
                # 选择题格式
                prompt = question
                if not any(marker in question.lower() for marker in ['choose', 'select', 'which', '选择']):
                    prompt += "\n\nPlease choose from the following options:"
                for i, choice in enumerate(choices):
                    prompt += f"\n{chr(65+i)}: {choice}"
                prompt += "\n\nProvide your answer as a single letter (A, B, C, or D)."
            else:
                # 普通问题格式
                if output_format == "step_by_step":
                    prompt = f"Question: {question}\nSolve step by step.\nSolution:"
                elif output_format == "answer_only":
                    prompt = f"Question: {question}\nAnswer:"
                else:  # detailed
                    prompt = f"Question: {question}\n\nPlease solve this problem step by step, and give your final answer.\n\nSolution:"
            
            # 添加图像标记
            if image is not None:
                if self.model.config.mm_use_im_start_end:
                    prompt = self.DEFAULT_IMAGE_TOKEN + '\n' + prompt
                else:
                    prompt = self.DEFAULT_IMAGE_TOKEN + prompt
                self.logger.debug("Added image token to prompt")
            
            self.logger.debug(f"Constructed prompt (first 200 chars): {prompt[:200]}...")
            
            # 3. 使用对话模板
            self.logger.debug("Creating conversation template...")
            conv_mode = "llava_v1"
            conv = self.conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_with_template = conv.get_prompt()
            
            self.logger.debug(f"Final prompt with template (first 300 chars):\n{prompt_with_template[:300]}...")
            
            # 4. Tokenization - 修复了None问题
            self.logger.debug("Tokenizing...")
            input_ids = None
            
            try:
                input_ids = self.tokenizer_image_token(
                    prompt_with_template, 
                    self.tokenizer, 
                    self.IMAGE_TOKEN_INDEX, 
                    return_tensors='pt'
                )
                
                if input_ids is not None:
                    input_ids = input_ids.unsqueeze(0)
                    self.logger.debug("tokenizer_image_token succeeded")
                else:
                    self.logger.warning("tokenizer_image_token returned None")
            except Exception as e:
                self.logger.warning(f"tokenizer_image_token failed: {e}")
            
            # 备用tokenization方法
            if input_ids is None:
                self.logger.info("Using fallback tokenization...")
                try:
                    input_ids = self.tokenizer.encode(
                        prompt_with_template, 
                        return_tensors='pt',
                        add_special_tokens=True
                    ).unsqueeze(0)
                    self.logger.debug("Fallback tokenization succeeded")
                except Exception as e:
                    self.logger.error(f"Fallback tokenization also failed: {e}")
                    return {
                        "success": False,
                        "error": f"All tokenization methods failed: {e}",
                        "final_answer": None,
                        "full_response": ""
                    }
            
            # 确保input_ids在正确的设备上
            if self.device == 'cuda':
                input_ids = input_ids.to(self.model.device)
            else:
                input_ids = input_ids.to('cpu')
            
            self.logger.debug(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")
            
            # 创建attention_mask并确保在同一设备
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            if self.device == 'cuda':
                attention_mask = attention_mask.to(self.model.device)
            
            self.logger.debug(f"Attention mask shape: {attention_mask.shape}, device: {attention_mask.device}")
            
            # 5. 生成答案
            self.logger.info("Starting generation...")
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,  # 作为第一个位置参数
                    attention_mask=attention_mask,
                    images=image_tensor if image is not None else None,
                    do_sample=False,
                    max_new_tokens=2000 if output_format == "detailed" else 500,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            self.logger.debug(f"Output IDs shape: {output_ids.shape}")
            self.logger.debug(f"Token difference: {output_ids.shape[1] - input_ids.shape[1]}")
            
            # 6. 解码输出
            self.logger.debug("Decoding full output...")
            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            self.logger.debug(f"Full decoded length: {len(full_output)} characters")
            
            # 从完整输出中提取响应部分
            generated_text = ""
            
            # 方法1：查找ASSISTANT标记
            if "ASSISTANT:" in full_output:
                generated_text = full_output.split("ASSISTANT:")[-1].strip()
                self.logger.info(f"Extracted from ASSISTANT tag: {len(generated_text)} characters")
            # 方法2：查找Solution标记
            elif "Solution:" in full_output:
                parts = full_output.split("Solution:")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                    self.logger.info(f"Extracted from Solution tag: {len(generated_text)} characters")
            # 方法3：标准解码
            elif output_ids.shape[1] > input_ids.shape[1]:
                generated_text = self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                self.logger.info(f"Used standard decoding: {len(generated_text)} characters")
            # 方法4：使用完整输出
            else:
                generated_text = full_output
                self.logger.warning("Using full output as response")
            
            self.logger.info(f"Generated text length: {len(generated_text)} chars")
            self.logger.info(f"Generated text preview: {generated_text[:200]}...")
            
            # 7. 提取答案
            final_answer, extraction_method = self._extract_answer_enhanced(
                generated_text, question, choices
            )
            
            self.logger.info(f"Extracted answer: {final_answer} (method: {extraction_method})")
            
            # 8. 构建完整响应
            return {
                "success": True,
                "answer": final_answer,
                "final_answer": final_answer,
                "full_response": generated_text,
                "steps": generated_text,
                "solution": generated_text[:500],
                "extraction_method": extraction_method,
                "response_length": len(generated_text),
                "has_image": image is not None,
                "method": "MultiMath-7B (Ray Actor)"
            }
                
        except Exception as e:
            self.logger.error(f"solve_math error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False, 
                "error": str(e),
                "final_answer": None,
                "full_response": ""
            }
    
    def _extract_answer_enhanced(self, text: str, question: str, choices: List[str] = None) -> Tuple[str, str]:
        """增强的答案提取"""
        
        text = text.strip()
        
        if not text:
            return "No answer", "empty_response"
        
        self.logger.info(f"Extracting answer from response (length: {len(text)})")
        
        # 1. 查找选择题答案（如果有选项）
        if choices:
            choice_patterns = [
                # 最常见的格式
                r"(?:the\s+)?correct\s+answer\s+is\s+([A-Da-d])\b",
                r"(?:the\s+)?answer\s+is\s+([A-Da-d])\b",
                r"Answer\s*[:：]\s*([A-Da-d])\b",
                # Therefore/Thus/So引导
                r"(?:Therefore|Thus|Hence|So),?\s+(?:the\s+)?answer\s+is\s+([A-Da-d])\b",
                # 选项相关
                r"(?:option|选项)\s*([A-Da-d])\s*(?:is|为|是)\s*(?:correct|正确)",
                # 句末答案
                r"(?:is|are)\s+([A-Da-d])\.$",
                r"\b([A-Da-d])\s*\.$",
                # 独立字母
                r"^([A-Da-d])\s*$",
                r"\n([A-Da-d])\s*$",
            ]
            
            for pattern in choice_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    answer = matches[-1].upper()
                    max_valid = chr(65 + len(choices) - 1)
                    if answer <= max_valid:
                        self.logger.info(f"Choice pattern matched: '{pattern[:30]}...' -> Answer: '{answer}'")
                        return answer, "choice_pattern"
        
        # 2. 查找boxed格式答案
        boxed_patterns = [
            r"\\boxed\{([^}]+)\}",
            r"\$\\boxed\{([^}]+)\}\$",
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                answer = matches[-1].strip()
                answer = answer.replace('°', '').replace('度', '').strip()
                self.logger.info(f"Boxed pattern matched -> Answer: '{answer}'")
                return answer, "boxed_pattern"
        
        # 3. 查找数值答案
        numeric_patterns = [
            r"(?:final\s+)?answer\s*[:：]\s*([0-9.,\-\+]+)",
            r"=\s*([0-9.,\-\+]+)(?:\s|$|\.)",
            r"(?:equals?|is)\s+([0-9.,\-\+]+)",
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                answer = re.sub(r'[,\s]', '', answer)
                self.logger.info(f"Numeric pattern matched -> Answer: '{answer}'")
                return answer, "numeric_pattern"
        
        # 4. 如果是选择题但没找到字母答案，尝试在最后几行找
        if choices:
            last_lines = '\n'.join(text.split('\n')[-3:])
            letter_matches = re.findall(r'\b([A-Da-d])\b', last_lines)
            if letter_matches:
                answer = letter_matches[-1].upper()
                max_valid = chr(65 + len(choices) - 1)
                if answer <= max_valid:
                    self.logger.info(f"Fallback: found letter in last lines -> Answer: '{answer}'")
                    return answer, "fallback_letter"
        
        # 5. 提取第一个数字作为备选
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[0], "first_number"
        
        # 6. 返回文本的前50个字符
        self.logger.warning("No answer pattern matched")
        return text[:50].strip(), "text_snippet"
    
    def health_check(self) -> str:
        """健康检查"""
        status = []
        if self.model is not None:
            status.append("model loaded")
        if self.tokenizer is not None:
            status.append("tokenizer loaded")
        if self.image_processor is not None:
            status.append("processor loaded")
        
        if len(status) == 3:
            return f"healthy: all components loaded (model: {self.model_name})"
        elif status:
            return f"partial: {', '.join(status)}"
        else:
            return "unhealthy: no components loaded"

# ==============================================================================
# 2. MultiMathTool - 客户端
# ==============================================================================
@register_tool
class MultiMathTool(BaseTool):
    tool_type = "multimath"
    
    def __init__(self, num_workers=1, model_path=None, **kwargs):
        super().__init__(num_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model_path = model_path or "/data/models/MultiMath"
        
        self.logger.info(f"Using model path: {self.model_path}")
        
        # 创建Actor
        try:
            num_gpus = 1 if torch.cuda.is_available() else 0
            self.actor = MultiMathActor.options(
                name="MultiMathActor",
                get_if_exists=True,
                num_gpus=num_gpus,
                max_concurrency=3
            ).remote(model_path=self.model_path)
            
            # 健康检查
            health_status = ray.get(self.actor.health_check.remote(), timeout=60)
            self.logger.info(f"MultiMath Actor status: {health_status}")
            
        except Exception as e:
            self.logger.error(f"Failed to create Actor: {e}")
            self.actor = None
    
    def get_usage_inst(self):
        return 'MultiMath: Use <tool_call>{"tool": "multimath", "task": "solve", "question": "problem", "choices": ["A", "B", "C", "D"]}</tool_call>'
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """解析action - 支持完整参数"""
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, action, re.DOTALL)
        
        if not matches:
            return {}, False
        
        try:
            params = json.loads(matches[0].strip())
            
            # 验证工具名
            if params.get('tool') not in ['multimath', 'multimath_server']:
                return {}, False
            
            # 处理嵌套格式和简单格式
            if 'parameters' in params and isinstance(params['parameters'], dict):
                inner_params = params['parameters']
            else:
                inner_params = params
            
            # 提取所有必要参数
            parsed = {}
            
            # 问题文本（必需）
            question = (inner_params.get('question') or 
                       inner_params.get('prompt') or 
                       inner_params.get('problem', ''))
            
            if not question:
                self.logger.warning(f"No question found in params: {params}")
                return {}, False
            
            parsed['question'] = question
            
            # 选择题选项（可选）
            if 'choices' in inner_params:
                parsed['choices'] = inner_params['choices']
            
            # 问题类型（可选）
            if 'problem_type' in inner_params:
                parsed['problem_type'] = inner_params['problem_type']
            elif 'type' in inner_params:
                parsed['problem_type'] = inner_params['type']
            
            # 输出格式（可选）
            if 'output_format' in inner_params:
                parsed['output_format'] = inner_params['output_format']
            elif 'format' in inner_params:
                parsed['output_format'] = inner_params['format']
            
            # 图片路径（可选）
            if 'image_path' in inner_params:
                parsed['image_path'] = inner_params['image_path']
            elif 'image_ref' in inner_params:
                parsed['image_path'] = inner_params['image_ref']
            
            self.logger.debug(f"Parsed parameters: question={parsed.get('question', '')[:50]}..., "
                            f"choices={parsed.get('choices', [])}, "
                            f"problem_type={parsed.get('problem_type', 'general')}")
            
            return parsed, True
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return {}, False
    
    def _get_image_path(self, extra_field: Dict[str, Any], env: Dict[str, Any]) -> Optional[str]:
        """从extra_field或环境中获取图片路径"""
        
        # 优先处理verl-tool格式的extra_info（包含images列表）
        if extra_field and 'images' in extra_field:
            images_list = extra_field['images']
            self.logger.debug(f"[MultiMath] Found 'images' in extra_field: {images_list}")
            
            if images_list:
                img_path = images_list[0] if isinstance(images_list, list) else images_list
                
                if img_path and isinstance(img_path, str):
                    if os.path.exists(img_path):
                        self.logger.debug(f"[MultiMath] Found image path: {img_path}")
                        return img_path
                    else:
                        self.logger.warning(f"[MultiMath] Image path does not exist: {img_path}")
        
        # 向后兼容：原有的image字段处理
        for field_name in ['image', 'image_path']:
            if extra_field and field_name in extra_field:
                img_path = extra_field[field_name]
                if isinstance(img_path, str) and os.path.exists(img_path):
                    self.logger.debug(f"[MultiMath] Found image in extra_field['{field_name}']: {img_path}")
                    return img_path
        
        self.logger.debug("[MultiMath] No image found in extra_field")
        return None
    
    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]) -> Tuple[str, bool, bool]:
        """执行action - 返回完整的推理过程"""
        parsed_params, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "Error: Invalid action format. Expected <tool_call>{...}</tool_call>"
            self.update_env(trajectory_id, env, action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return f"\n```output\n{observation}\n```\n", False, False
        
        try:
            # 调试日志
            self.logger.debug(f"[MultiMath] Received extra_field keys: {list(extra_field.keys()) if extra_field else 'None'}")
            if extra_field and 'qid' in extra_field:
                self.logger.debug(f"[MultiMath] Processing task_id: {extra_field.get('qid')}")
            
            # 获取图片路径（如果有）
            image_path = self._get_image_path(extra_field, env)
            if image_path and 'image_path' not in parsed_params:
                parsed_params['image_path'] = image_path
                self.logger.debug(f"[MultiMath] Will use image: {image_path}")
            
            if self.actor is None:
                observation = "Error: Actor not available"
            else:
                # 调用Actor求解
                result = ray.get(self.actor.solve_math.remote(parsed_params), timeout=200)
                
                if result.get("success"):
                    # 构建详细的观察结果（包含完整推理）
                    observation_parts = []
                    
                    # 如果有完整响应，优先使用
                    if result.get('full_response'):
                        observation_parts.append("Mathematical Solution:")
                        observation_parts.append("="*50)
                        observation_parts.append(result['full_response'])
                        observation_parts.append("="*50)
                    elif result.get('solution'):
                        observation_parts.append("Solution:")
                        observation_parts.append(result['solution'])
                    
                    # 添加最终答案
                    final_answer = result.get('final_answer') or result.get('answer', 'No answer')
                    observation_parts.append(f"\nFinal Answer: {final_answer}")
                    
                    # 添加提取方法（用于调试）
                    if result.get('extraction_method'):
                        observation_parts.append(f"(Extraction method: {result['extraction_method']})")
                    
                    observation = "\n".join(observation_parts)
                    
                    self.logger.info(f"Success! Final answer: {final_answer}")
                else:
                    observation = f"Error: {result.get('error', 'Unknown error')}"
                    self.logger.error(f"Solve failed: {observation}")
            
            self.update_env(trajectory_id, env, parsed_params, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            
            # 返回格式化的输出
            return f"\n```output\n{observation}\n```\n", False, True
            
        except Exception as e:
            self.logger.error(f"conduct_action error: {e}")
            import traceback
            traceback.print_exc()
            observation = f"Error: {str(e)}"
            return f"\n```output\n{observation}\n```\n", False, True
    
    def update_env(self, trajectory_id: str, env: Dict[str, Any], action: Any,
                   is_valid: bool, extra_field: Dict[str, Any], observation: str, **kwargs):
        """更新环境"""
        super().update_env(trajectory_id, env, action, is_valid, extra_field, observation, **kwargs)