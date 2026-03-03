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
        
        # Set device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== 添加调试打印 =====
        #print(f"\n{'='*60}")
        #print(f"[DEBUG] Model Configuration Info:")
        #print(f"{'='*60}")
        
        # 打印模型配置信息
        #if hasattr(self.model, 'config'):
        #    config = self.model.config
            
            # 最大序列长度相关
        #    if hasattr(config, 'max_position_embeddings'):
        #        print(f"  - max_position_embeddings: {config.max_position_embeddings}")
        #    if hasattr(config, 'max_length'):
        #        print(f"  - max_length: {config.max_length}")
        #    if hasattr(config, 'model_max_length'):
        #        print(f"  - model_max_length: {config.model_max_length}")
            
            # 词汇表大小
        #    if hasattr(config, 'vocab_size'):
        #        print(f"  - vocab_size: {config.vocab_size}")
            
            # 隐藏层大小
        #    if hasattr(config, 'hidden_size'):
        #        print(f"  - hidden_size: {config.hidden_size}")
            
            # 其他可能相关的配置
        #    if hasattr(config, 'max_window_layers'):
        #        print(f"  - max_window_layers: {config.max_window_layers}")
        #    if hasattr(config, 'use_sliding_window'):
        #        print(f"  - use_sliding_window: {config.use_sliding_window}")
        
        # 打印 processor/tokenizer 信息
        #if hasattr(self.processor, 'tokenizer'):
        #    tokenizer = self.processor.tokenizer
        #    print(f"\n  Tokenizer info:")
        #    print(f"  - vocab_size: {len(tokenizer)}")
        #    if hasattr(tokenizer, 'model_max_length'):
        #        print(f"  - model_max_length: {tokenizer.model_max_length}")
        #    if hasattr(tokenizer, 'max_length'):
        #        print(f"  - max_length: {tokenizer.max_length}")
            
            # 打印特殊token
        #    if hasattr(tokenizer, 'pad_token_id'):
        #        print(f"  - pad_token_id: {tokenizer.pad_token_id}")
        #    if hasattr(tokenizer, 'eos_token_id'):
        #        print(f"  - eos_token_id: {tokenizer.eos_token_id}")
        
        # 打印设备信息
        #print(f"\n  Device info:")
        #print(f"  - device: {self.device}")
        #print(f"  - device_map: {agent_config.get('device_map', 'auto')}")
        #print(f"  - torch_dtype: {agent_config.get('torch_dtype', 'default')}")
        
        #print(f"{'='*60}\n")
        # ===== 调试打印结束 =====
        
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
        self.load_model()  # 修正：从 _load_model() 改为 load_model()
        try:
            # ===== DEBUG: 输入信息 =====
            print(f"\n[DEBUG act] Starting generation")
            print(f"  - Question: {observation.get('question', 'N/A')[:100]}...")
            print(f"  - Has reflection: {observation.get('previous_attempt_failed', False)}")
            print(f"  - Attempt: {observation.get('attempt', 1)}")
            # Prepare input with reflection support
            inputs = self._prepare_input(observation)
            
            # ===== 验证 inputs =====
            #print(f"\n[VERIFY] After _prepare_input in act:")
            #print(f"  - inputs is None: {inputs is None}")
            #print(f"  - inputs type: {type(inputs)}")
            # =====
            
            # Generate response
            outputs = self._generate_response(inputs)
            # Parse response
            response = self._parse_response(outputs, observation)
            # Prepare extra info
            extra_info = {
                "raw_response": outputs,
                "has_reflection": inputs.get('_has_reflection', False),
                "attempt": observation.get('attempt', 1)
            }
            
            return response, extra_info
            
        except Exception as e:
            logger.error(f"Error in act method: {str(e)}")
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
        
        # ===== 检查图像 =====
        #print(f"\n[VERIFY] Image check:")
        #print(f"  - Type: {type(image)}")
        #print(f"  - Mode: {image.mode if hasattr(image, 'mode') else 'N/A'}")
        #print(f"  - Size: {image.size if hasattr(image, 'size') else 'N/A'}")
        # =====
        
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
            #prompt = f"{prompt}\n\n{output_format_instruction}\n\nPlease solve the question above following this format."
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

        #图像占位符为 Qwen-VL 格式
        #text = text.replace('<|vision_start|><|image_pad|><|vision_end|>', '<|image|>')
        # ===== 添加调试代码 =====
        print(f"\n[DEBUG] Chat template output check:")
        print(f"  - Contains <|vision_start|>: {'<|vision_start|>' in text}")
        print(f"  - Text preview: {text[:3000]}...")
        # ========================

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
                # 不再 reshape，也不再报警
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
        
        # image_grid_thw 通常由 processor 返回；如缺失可在此检查
        if "image_grid_thw" not in inputs and pv is not None:
            print("[DEBUG] image_grid_thw missing — normal for Qwen2.5-VL")
        # =====

        # ===== 立即检查 processor 输出 =====
        #print(f"\n[VERIFY] Processor output check:")
        #print(f"  - inputs type: {type(inputs)}")
        #print(f"  - inputs is None: {inputs is None}")
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
        # =====
        
        # ===== 添加详细调试 =====
        #print(f"\n[DEBUG _prepare_input] BEFORE moving to device:")
        #print(f"[DEBUG _prepare_input] Input keys: {list(inputs.keys())}")
        
        # 检查 input_ids
        if 'input_ids' in inputs:
            #print(f"[DEBUG _prepare_input] input_ids shape: {inputs['input_ids'].shape}")
            #print(f"[DEBUG _prepare_input] input_ids dtype: {inputs['input_ids'].dtype}")
            max_id = inputs['input_ids'].max().item()
            min_id = inputs['input_ids'].min().item()
            #print(f"[DEBUG _prepare_input] Token ID range: {min_id} to {max_id}")
            
            # 检查词汇表大小
            if hasattr(self.processor, 'tokenizer'):
                vocab_size = len(self.processor.tokenizer)
                print(f"[DEBUG _prepare_input] Tokenizer vocab size: {vocab_size}")
                if max_id >= vocab_size:
                    print(f"[DEBUG _prepare_input] ⚠️ WARNING: Max token ID {max_id} >= vocab size {vocab_size}")
                    # 找出哪些位置有问题
                    problem_positions = (inputs['input_ids'] >= vocab_size).nonzero()
                    print(f"[DEBUG _prepare_input] Problem positions: {problem_positions}")
                    if len(problem_positions) > 0:
                        for pos in problem_positions[:5]:  # 只打印前5个
                            token_id = inputs['input_ids'][pos[0], pos[1]].item()
                            print(f"  Position {pos.tolist()}: token_id = {token_id}")
        
        # 打印文本长度
        #print(f"[DEBUG _prepare_input] Text length: {len(text)} characters")
        #print(f"[DEBUG _prepare_input] Text preview (last 200 chars): ...{text[-200:]}")
        # ===== 调试结束 =====
        
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
            #print(f"\n[DEBUG _prepare_input] ERROR during .to(device):")
            #print(f"  Device: {self.device}")
            #print(f"  Error: {e}")
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
    
    #def _generate_response(self, inputs: Dict[str, Any]) -> str:
        """Generate response using the model"""
        # Extract model inputs (remove our custom fields)
        model_inputs = {k: v for k, v in inputs.items() 
                       if not k.startswith('_')}
        
        # Get generation config - handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
            
        generation_config = {
            "max_new_tokens": 512,
            "temperature": agent_config.get("temperature", 0.3),
            "do_sample": agent_config.get("temperature", 0.3) > 0,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                **generation_config
            )
        
        # Decode
        generated_ids = outputs[0][model_inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response
    
    
    def _generate_response(self, inputs: Dict[str, Any]) -> str:
        """Generate response using the model"""
        
        # Extract model inputs (remove our custom fields)
        model_inputs = {k: v for k, v in inputs.items() 
                    if not k.startswith('_')}
        
        # 检查必要的输入
        if 'input_ids' not in model_inputs:
            print(f"[_generate_response] WARNING: 'input_ids' not found in model_inputs!")
        
        # 检查是否有图像输入（保留）
        #if 'pixel_values' in model_inputs:
            #print(f"[_generate_response] Image input detected")
            #print(f"[_generate_response] pixel_values shape: {model_inputs['pixel_values'].shape}")
        #elif 'images' in model_inputs:
            #print(f"[_generate_response] images found in inputs")
        #else:
            #print(f"[_generate_response] No image inputs found")
        
        # Get generation config - handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
           
        generation_config = {
            "max_new_tokens": agent_config.get("max_new_tokens", 512),
            "temperature": agent_config.get("temperature", 0.1),
            "do_sample": agent_config.get("temperature", 0.1) > 0,
            "repetition_penalty": agent_config.get("repetition_penalty", 1.15),
        }
        
        # 确保所有输入在同一设备上
        model_device = next(self.model.parameters()).device
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                if model_inputs[key].device != model_device:
                    model_inputs[key] = model_inputs[key].to(model_device)
        
        # Generate
        try:
            with torch.no_grad():
                # 添加必要的生成参数
                generation_config.update({
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,  # 使用KV缓存加速
                })
                
                outputs = self.model.generate(
                    **model_inputs,
                    **generation_config
                )
                
        except Exception as e:
            print(f"[_generate_response] ERROR in model.generate(): {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Decode
        try:
            generated_ids = outputs[0][model_inputs['input_ids'].shape[1]:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            #print(f"[_generate_response] ERROR in decoding: {type(e).__name__}: {str(e)}")
            raise
        
        return response
    
    
    def _parse_response(self, response: str, observation: Dict[str, Any]) -> str:
        """Parse response and format as action"""
        response = response.strip()
        
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
        if not use_structured and not response.startswith("answer_question("):
            escaped_response = response.replace('"', '\\"')
            return f'answer_question(answer="{escaped_response}")'
        
        return response
        
    def reset(self):
        """Reset agent state"""
        # VLMAgent is stateless, so nothing to reset
        pass