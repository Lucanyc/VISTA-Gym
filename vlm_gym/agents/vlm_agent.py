"""VLM Agent implementation with reflection support and structured output"""
#基础VLM类，提供基本的视觉-语言模型功能

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

from .base import BaseAgent
from .utils import (
    load_image_safely,
    validate_observation,
    format_prompt_with_choices,
    format_reflection_prompt, 
)

logger = logging.getLogger(__name__)


class VLMAgent(BaseAgent):
    """Vision-Language Model agent for multimodal tasks with reflection support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
        
        # Handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            # It's a dict
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            # It's an object, try to access __dict__
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            # Fallback - assume config itself contains the agent settings
            agent_config = {}
            
        self.reflection_template_style = agent_config.get("reflection_template_style", "standard")
    
    
    def load_model(self):
        """Load model and processor if not already loaded"""
        if self._loaded:
            return
        # Handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
            
        model_name = agent_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        logger.info(f"Loading VLM model: {model_name}")
        # Processor & model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=agent_config.get("trust_remote_code", True)
        )
        
        # Model loading parameters
        model_kwargs = {
            "trust_remote_code": agent_config.get("trust_remote_code", True),
            "device_map": agent_config.get("device_map", "auto"),
        }
        
        # Add torch dtype if specified
        if "torch_dtype" in agent_config:
            dtype_str = agent_config["torch_dtype"]
            if dtype_str == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype_str == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # ===== 修复设备检测问题 =====
        # 当使用 device_map="auto" 时，模型可能分布在多个设备上
        # 需要更智能的方式检测主设备
        
        # 先检查CUDA是否可用
        if torch.cuda.is_available():
            # 尝试从模型的第一个参数获取设备
            try:
                first_param_device = next(self.model.parameters()).device
                if first_param_device.type != 'cpu':
                    self.device = first_param_device
                    print(f"[DEBUG] Using device from model parameters: {self.device}")
                else:
                    # 如果参数在CPU上，但CUDA可用，强制使用CUDA
                    self.device = torch.device("cuda:0")
                    print(f"[DEBUG] Model params on CPU, but CUDA available. Using: {self.device}")
            except StopIteration:
                # 没有参数，使用CUDA
                self.device = torch.device("cuda:0")
                print(f"[DEBUG] No model parameters found, using: {self.device}")
        else:
            # CUDA不可用，使用CPU
            self.device = torch.device("cpu")
            print(f"[DEBUG] CUDA not available, using: {self.device}")
        
        # 验证设备
        print(f"[DEBUG] Final device selection: {self.device}")
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
            print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
        
        self._loaded = True
        logger.info("Model loaded successfully")
    
    
    def act(self, observation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Generate action based on observation with reflection support
        
        Args:
            observation: Dictionary containing image_path, question, and optionally
                        feedback, previous_answer, conversation_history for reflection
            
        Returns:
            Tuple of (action_string, extra_info_dict)
        """
        # Ensure model is loaded
        self.load_model()
        try:
            # ===== DEBUG: 输入信息 =====
            print(f"\n[DEBUG act] Starting generation")
            print(f"  - Question: {observation.get('question', 'N/A')[:100]}...")
            print(f"  - Has reflection: {observation.get('previous_attempt_failed', False)}")
            print(f"  - Attempt: {observation.get('attempt', 1)}")
            
            # Prepare input with reflection support
            inputs = self._prepare_input(observation)
            
            # ===== DEBUG: 生成配置 =====
            print(f"\n[DEBUG act] Generation config:")
            if hasattr(self.model, 'generation_config'):
                print(f"  - max_new_tokens: {self.model.generation_config.max_new_tokens}")
                print(f"  - temperature: {self.model.generation_config.temperature}")
                print(f"  - do_sample: {self.model.generation_config.do_sample}")
            else:
                print(f"  - No generation_config found")
            
            # Generate response
            print(f"\n[DEBUG act] Calling _generate_response...")
            outputs = self._generate_response(inputs)
            
            # ===== DEBUG: 生成结果 =====
            print(f"\n[DEBUG act] Generation complete:")
            print(f"  - Output type: {type(outputs)}")
            print(f"  - Output length: {len(outputs) if isinstance(outputs, str) else 'N/A'}")
            if isinstance(outputs, str):
                print(f"  - First 200 chars: {outputs[:200]}")
                print(f"  - Last 100 chars: {outputs[-100:]}")
                # 检查是否包含特殊格式
                if "answer_question(" in outputs:
                    print(f"  - Contains answer_question format: Yes")
                if "<tool_call>" in outputs:
                    print(f"  - Contains tool_call format: Yes")
                if "<reasoning>" in outputs or "<answer>" in outputs:
                    print(f"  - Contains structured format: Yes")
            
            # Parse response
            response = self._parse_response(outputs, observation)
            
            # ===== DEBUG: 解析结果 =====
            print(f"\n[DEBUG act] Parsed response:")
            print(f"  - Response type: {type(response)}")
            print(f"  - Response: {response[:100] if isinstance(response, str) else response}")
            
            # Prepare extra info
            extra_info = {
                "raw_response": outputs,
                "has_reflection": inputs.get('_has_reflection', False),
                "attempt": observation.get('attempt', 1)
            }
            
            print(f"[VLMAgent.act] DEBUG END\n")
            return response, extra_info
            
        except Exception as e:
            logger.error(f"Error in act method: {str(e)}")
            print(f"\n[DEBUG act] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
    def _prepare_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the VLM model with reflection support
        
        Args:
            observation: Dictionary with image_path, question, choices, 
                        and optionally: feedback, previous_answer, conversation_history
            
        Returns:
            Dictionary with prepared inputs for model
        """
        # Basic validation
        is_valid, error_msg = validate_observation(observation)
        if not is_valid:
            raise ValueError(f"Invalid observation: {error_msg}")
        
        # Extract components
        image_path = observation['image_path']
        question = observation['question']
        choices = observation.get('choices', None)
        
        # Load image
        image = load_image_safely(image_path)
        # 确保最终是 RGB PIL.Image
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            # 如 utils 返回 ndarray / tensor，重新加载为 PIL
            image = Image.open(image_path).convert("RGB")
        
        # Check for reflection context
        has_reflection = 'feedback' in observation or 'previous_answer' in observation
        
        # Build prompt based on context
        if has_reflection:
            # Reflection prompt - build special prompt for retry attempts
            prompt = self._build_reflection_prompt(observation)
        else:
            # Standard prompt
            prompt = (
                format_prompt_with_choices(question, choices)
                if choices
                else question
            )
        
        # ========== 新增：添加结构化输出格式指令 ==========
        output_format_instruction = observation.get('output_format_instruction', '')
        if output_format_instruction:
            # 将格式指令添加到 prompt
            prompt = f"{prompt}\n\n{'='*60}\nOUTPUT FORMAT REQUIREMENTS:\n{'='*60}\n{output_format_instruction}"
        
        # Build chat messages (single‑turn or multi‑turn)
        if observation.get("conversation_history"):
            # Multi-turn conversation with history
            messages = self._build_conversation_messages(
                observation['conversation_history'], 
                image, 
                prompt
            )
        else:
            # Single-turn or first attempt
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # ===== 添加调试代码 =====
        print(f"\n[DEBUG] Chat template output check:")
        print(f"  - Contains <|vision_start|>: {'<|vision_start|>' in text}")
        print(f"  - Text preview: {text[:1000]}...")

        # Processor
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )
        
        # 标记是否包含反思
        inputs["_has_reflection"] = has_reflection
        
        # ===== 处理 pixel_values 维度（修复后的版本）=====
        pv = inputs.get("pixel_values")
        if pv is not None:
            if pv.ndim == 2:
                # Qwen2.5-VL patch-grid format: [num_patches, hidden_dim]
                print(f"[DEBUG] ✓ Qwen2.5-VL pixel_values 2D: {pv.shape}")
            elif pv.ndim == 3:
                # 3D tensor [channels, height, width]，添加 batch dimension
                inputs["pixel_values"] = pv.unsqueeze(0)
                print(f"[DEBUG] Added batch dim: {pv.shape} → {inputs['pixel_values'].shape}")
            elif pv.ndim == 4:
                # Standard 4D format [batch, channels, height, width]
                print(f"[DEBUG] ✓ Standard 4D pixel_values: {pv.shape}")
            else:
                # 其他维度
                print(f"[DEBUG] Unusual pixel_values dims: {pv.shape}")
        
        if inputs is not None:
            print(f"  - Keys: {list(inputs.keys())}")
            # 检查每个值
            for k, v in inputs.items():
                if v is None:
                    print(f"  - {k}: None")
                elif hasattr(v, 'shape'):
                    print(f"  - {k}: shape={v.shape}, dtype={v.dtype if hasattr(v, 'dtype') else 'N/A'}")
                else:
                    print(f"  - {k}: type={type(v)}")
        
        # 检查 input_ids
        if 'input_ids' in inputs:
            max_id = inputs['input_ids'].max().item()
            min_id = inputs['input_ids'].min().item()
            
            # 检查词汇表大小
            if hasattr(self.processor, 'tokenizer'):
                vocab_size = len(self.processor.tokenizer)
                print(f"[DEBUG _prepare_input] Tokenizer vocab size: {vocab_size}")
                if max_id >= vocab_size:
                    print(f"[DEBUG _prepare_input] ⚠️ WARNING: Max token ID {max_id} >= vocab size {vocab_size}")
        
        # Move to device
        try:
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    processed_inputs[k] = v.to(self.device)
                else:
                    processed_inputs[k] = v
            inputs = processed_inputs
        except RuntimeError as e:
            raise

        return inputs
    
    def _build_reflection_prompt(self, observation: Dict[str, Any]) -> str:
        """Build prompt for reflection attempts using utils templates"""
        
        # 使用 utils.py 中的高级模板功能
        # 根据不同情况选择合适的模板风格
        attempt = observation.get('attempt', 2)
        previous_answer = observation.get('previous_answer', '')
        feedback = observation.get('feedback', '')
        
        # 动态选择模板风格
        if attempt >= 3:
            # 最后一次尝试使用详细模板，提供更多指导
            template_style = "detailed"
        elif len(previous_answer) < 10 and 'numerical' in feedback.lower():
            # 如果之前的答案很短且是数值类问题，使用简洁模板
            template_style = "concise"
        else:
            # 使用配置的默认模板风格
            template_style = self.reflection_template_style
        
        # 调用 utils.py 中的 format_reflection_prompt
        reflection_prompt = format_reflection_prompt(observation, template_style)
        
        # ========== 新增：确保反思提示也包含格式要求 ==========
        reflection_format_reminder = observation.get('reflection_format_reminder', '')
        if reflection_format_reminder:
            reflection_prompt = f"{reflection_prompt}\n\n{reflection_format_reminder}"
        elif observation.get('output_format_instruction'):
            # 如果没有特定的反思格式提醒，使用通用格式指令
            reflection_prompt = f"{reflection_prompt}\n\n{observation['output_format_instruction']}"
        
        return reflection_prompt
    
    def _build_conversation_messages(self, conversation_history: List[Dict], 
                                    image: Any, 
                                    current_prompt: str) -> List[Dict]:
        """Build messages for multi-turn conversation with reflection context"""
        
        messages = []
        
        # First message should include the image
        first_user_message = True
        
        for turn in conversation_history:
            if turn['role'] == 'user':
                if first_user_message:
                    # First user message includes image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": turn['content']}
                        ]
                    })
                    first_user_message = False
                else:
                    # Subsequent user messages are text only
                    messages.append({
                        "role": "user",
                        "content": turn['content']
                    })
            else:
                # System or assistant messages
                messages.append({
                    "role": turn['role'],
                    "content": turn['content']
                })
        
        # Add current prompt as the latest user message
        if first_user_message:
            # If no conversation history, include image with current prompt
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": current_prompt}
                ]
            })
        else:
            # Otherwise, current prompt is text only
            messages.append({
                "role": "user",
                "content": current_prompt
            })
        
        return messages
    
    
    def _generate_response(self, inputs: Dict[str, Any]) -> str:
        """Generate response using the model"""
        
        # ===== DEBUG: 输入信息 =====
        print(f"\n[DEBUG _generate_response] Starting generation")
        print(f"  - Input keys: {list(inputs.keys())}")
        print(f"  - Has reflection: {inputs.get('_has_reflection', False)}")
        
        # Extract model inputs (remove our custom fields)
        model_inputs = {k: v for k, v in inputs.items() 
                    if not k.startswith('_')}
        
        # 检查必要的输入
        if 'input_ids' not in model_inputs:
            print(f"[_generate_response] WARNING: 'input_ids' not found in model_inputs!")
        else:
            # ===== DEBUG: 输入形状 =====
            print(f"  - input_ids shape: {model_inputs['input_ids'].shape}")
            print(f"  - input length: {model_inputs['input_ids'].shape[1]} tokens")
        
        # 检查是否有图像输入
        if 'pixel_values' in model_inputs:
            print(f"  - Image input detected")
            print(f"  - pixel_values shape: {model_inputs['pixel_values'].shape}")
        elif 'images' in model_inputs:
            print(f"  - images found in inputs")
        else:
            print(f"  - No image inputs found")
        
        # Get generation config - handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
        
        # ===== 修复1: 增加默认max_new_tokens =====
        generation_config = {
            "max_new_tokens": agent_config.get("max_new_tokens", 1024),  # 默认1024
            "temperature": agent_config.get("temperature", 0.1),
            "do_sample": agent_config.get("temperature", 0.1) > 0,
            "repetition_penalty": agent_config.get("repetition_penalty", 1.05),
        }
        
        # ===== DEBUG: 生成配置 =====
        print(f"\n[DEBUG _generate_response] Generation config:")
        for key, value in generation_config.items():
            print(f"  - {key}: {value}")
        
        # ===== 修复2: 修正反思模式的逻辑 =====
        if inputs.get('_has_reflection', False):
            original_max = generation_config['max_new_tokens']
            # 使用 max 而不是 min，确保增加而不是减少
            generation_config['max_new_tokens'] = max(original_max * 2, 2048)
            # 但也要有上限
            generation_config['max_new_tokens'] = min(generation_config['max_new_tokens'], 4096)
            print(f"  - Adjusted max_new_tokens for reflection: {original_max} -> {generation_config['max_new_tokens']}")
        
        # 确保所有输入在同一设备上
        model_device = next(self.model.parameters()).device
        print(f"  - Model device: {model_device}")
        
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                if model_inputs[key].device != model_device:
                    print(f"  - Moving {key} to {model_device}")
                    model_inputs[key] = model_inputs[key].to(model_device)
        
        # Generate
        try:
            with torch.no_grad():
                # ===== 修复3: 添加更多生成参数 =====
                generation_config.update({
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,  # 使用KV缓存加速
                    "top_p": 0.95,  # 添加top_p
                    "top_k": 50,    # 添加top_k
                    "early_stopping": False,  # 不要提前停止
                })
                
                print(f"\n[DEBUG _generate_response] Calling model.generate...")
                import time
                start_time = time.time()
                
                outputs = self.model.generate(
                    **model_inputs,
                    **generation_config
                )
                
                generation_time = time.time() - start_time
                print(f"[DEBUG _generate_response] Generation completed in {generation_time:.2f}s")
                
        except Exception as e:
            print(f"[_generate_response] ERROR in model.generate(): {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # ===== DEBUG: 生成输出 =====
        print(f"\n[DEBUG _generate_response] Generation output:")
        print(f"  - outputs shape: {outputs.shape}")
        print(f"  - total length: {outputs.shape[1]} tokens")
        
        # Decode
        try:
            input_length = model_inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            
            print(f"  - input length: {input_length} tokens")
            print(f"  - generated length: {len(generated_ids)} tokens")
            
            # 检查是否可能被截断
            if len(generated_ids) >= generation_config['max_new_tokens'] - 5:
                print(f"  - ⚠️  WARNING: Output might be truncated!")
                print(f"  - Generated {len(generated_ids)} tokens, max was {generation_config['max_new_tokens']}")
            
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # ===== DEBUG: 解码结果 =====
            print(f"\n[DEBUG _generate_response] Decoded response:")
            print(f"  - Response length: {len(response)} characters")
            print(f"  - First 300 chars: {response[:300]}")
            if len(response) > 300:
                print(f"  - Last 200 chars: {response[-200:]}")
            
            # 检查特殊格式
            if "answer_question(" in response:
                print(f"  - Contains answer_question format: Yes")
                # 尝试提取答案内容
                import re
                match = re.search(r'answer_question\(answer="([^"]*)"', response)
                if match:
                    answer_content = match.group(1)
                    print(f"  - Answer content length: {len(answer_content)} chars")
                    if len(answer_content) > 100:
                        print(f"  - Answer preview: {answer_content[:100]}...")
            
            if "<tool_call>" in response:
                print(f"  - Contains tool_call format: Yes")
            
            if "<reasoning>" in response or "<answer>" in response:
                print(f"  - Contains structured format: Yes")
            
            if response.endswith(("...", "。。。", "...")):
                print(f"  - ⚠️  WARNING: Response ends with ellipsis, might be incomplete")
            
        except Exception as e:
            print(f"[_generate_response] ERROR in decoding: {type(e).__name__}: {str(e)}")
            raise
        
        return response
    
    
    def _parse_response(self, response: str, observation: Dict[str, Any]) -> str:
        """Parse response and format as action
        
        支持的格式：
        1. <tool_call>...</tool_call> - 工具调用格式
        2. ## REASONING: / ## FINAL_ANSWER: - Markdown格式
        3. <think>...</think> / <answer>...</answer> - XML格式
        4. 普通文本 - 根据use_structured_output决定是否包装
        """
        response = response.strip()
        
        # ========== 关键修改：优先检查工具调用格式 ==========
        # 如果响应包含<tool_call>标签，直接返回，不做任何处理
        if '<tool_call>' in response:
            print(f"[DEBUG _parse_response] Tool call detected, returning as-is")
            return response
        
        # 检查是否看起来像工具调用JSON（但缺少标签）
        if response.startswith('{') and '"tool"' in response and '"parameters"' in response:
            try:
                import json
                obj = json.loads(response)
                if "tool" in obj and "parameters" in obj:
                    # 添加标签包装
                    wrapped = f"<tool_call>\n{response}\n</tool_call>"
                    print(f"[DEBUG _parse_response] JSON tool call detected, wrapping with tags")
                    return wrapped
            except:
                pass  # 不是有效的JSON，继续其他处理
        
        # Check if using structured output
        use_structured = observation.get('use_structured_output', False)
        
        # If response contains Markdown format markers, return as is
        if '## REASONING:' in response or '## FINAL_ANSWER:' in response:
            return response
        
        # For backward compatibility with XML format
        if '<think>' in response or '<answer>' in response:
            # Handle any XML format cleanup if needed
            if response.startswith('answer_question(answer="') and response.endswith('")'):
                inner = response[24:-2]
                inner = inner.replace('\\"', '"')
                return inner
            return response
        
        # If not structured, add answer_question wrapper
        # 但是要排除工具调用相关的响应
        if not use_structured and not response.startswith("answer_question("):
            # 再次检查是否可能是工具调用相关内容
            if '"tool"' in response or 'tool_call' in response.lower():
                print(f"[DEBUG _parse_response] Possible tool-related content, not wrapping")
                return response
            
            escaped_response = response.replace('"', '\\"')
            return f'answer_question(answer="{escaped_response}")'
        
        return response
        
    def reset(self):
        """Reset agent state"""
        # VLMAgent is stateless, so nothing to reset
        pass