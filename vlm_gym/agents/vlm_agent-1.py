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
        
        # ===== 添加调试打印 =====
        print(f"\n[VLMAgent.__init__] Initializing VLMAgent")
        print(f"[VLMAgent.__init__] Config type: {type(config)}")
        print(f"[VLMAgent.__init__] Config keys: {list(config.keys()) if hasattr(config, 'keys') else 'Not a dict'}")
        
        # Handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            # It's a dict
            agent_config = self.config.get("agent", {})
            print(f"[VLMAgent.__init__] Found 'agent' key in config")
            print(f"[VLMAgent.__init__] agent_config keys: {list(agent_config.keys())}")
            if 'max_new_tokens' in agent_config:
                print(f"[VLMAgent.__init__] max_new_tokens in agent_config: {agent_config['max_new_tokens']}")
        elif hasattr(self.config, '__dict__'):
            # It's an object, try to access __dict__
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
            print(f"[VLMAgent.__init__] Config is object with __dict__")
        else:
            # Fallback - assume config itself contains the agent settings
            agent_config = {}
            print(f"[VLMAgent.__init__] Using config directly as agent_config")
            
        # ===== 直接检查config中的max_new_tokens =====
        if hasattr(config, 'get') and 'max_new_tokens' in config:
            print(f"[VLMAgent.__init__] max_new_tokens found directly in config: {config['max_new_tokens']}")
        # =====
            
        self.reflection_template_style = agent_config.get("reflection_template_style", "standard")
    
    
    def load_model(self):
        """Load model and processor if not already loaded"""
        if self._loaded:
            return
            
        print(f"\n[VLMAgent.load_model] Starting model load")
        
        # Handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
            # ===== 如果agent中没有但config顶层有，复制过来 =====
            if 'max_new_tokens' not in agent_config and 'max_new_tokens' in self.config:
                agent_config['max_new_tokens'] = self.config['max_new_tokens']
                print(f"[VLMAgent.load_model] Copied max_new_tokens from top-level config: {self.config['max_new_tokens']}")
            # =====
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
            
        model_name = agent_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        print(f"[VLMAgent.load_model] agent_config keys: {list(agent_config.keys())}")
        print(f"[VLMAgent.load_model] max_new_tokens in agent_config: {agent_config.get('max_new_tokens', 'NOT FOUND')}")
        
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

        # ===== 检查特殊tokens和addCriterion =====
        print(f"\n[VLMAgent.load_model] Checking tokenizer for special tokens")
        if hasattr(self.processor, 'tokenizer'):
            tokenizer = self.processor.tokenizer
            print(f"[VLMAgent.load_model] Tokenizer type: {type(tokenizer)}")
            print(f"[VLMAgent.load_model] Special tokens map: {tokenizer.special_tokens_map}")
            print(f"[VLMAgent.load_model] EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
            print(f"[VLMAgent.load_model] PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
            
            # 检查词汇表中是否有addCriterion
            vocab = tokenizer.get_vocab()
            if "addCriterion" in vocab:
                print(f"[VLMAgent.load_model] ⚠️ Found 'addCriterion' in vocab! Token ID: {vocab['addCriterion']}")
            else:
                print(f"[VLMAgent.load_model] 'addCriterion' not found in vocab")
                
            # 检查其他可疑token
            suspicious_tokens = ['addCriterion', '自动生成', 'matchCondition', '!***', 'BEGIN_OF']
            for token in suspicious_tokens:
                if token in vocab:
                    print(f"[VLMAgent.load_model] ⚠️ Found suspicious token '{token}' with ID: {vocab[token]}")
        # =====
        
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
            # Prepare input with reflection support
            inputs = self._prepare_input(observation)
            
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
        print(f"  - Text preview: {text[:300]}...")
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
        
        # ===== 添加详细调试 =====
        print(f"[DEBUG _prepare_input] Tokenizer vocab size: {len(self.processor.tokenizer)}")
        # =====
        
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
        
        print(f"\n[VLMAgent._generate_response] Starting generation")
        
        # Extract model inputs (remove our custom fields)
        model_inputs = {k: v for k, v in inputs.items() 
                    if not k.startswith('_')}
        
        # 检查必要的输入
        if 'input_ids' not in model_inputs:
            print(f"[_generate_response] WARNING: 'input_ids' not found in model_inputs!")
        else:
            print(f"[_generate_response] input_ids shape: {model_inputs['input_ids'].shape}")
            print(f"[_generate_response] input_ids length: {model_inputs['input_ids'].shape[1]}")
        
        # Get generation config - handle both dict and AgentConfig object
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
            # ===== 如果agent中没有但config顶层有，使用顶层的 =====
            if 'max_new_tokens' not in agent_config and 'max_new_tokens' in self.config:
                agent_config = self.config.copy()
                print(f"[_generate_response] Using top-level config as agent_config")
            # =====
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
        
        # ===== 打印agent_config内容 =====
        print(f"[_generate_response] agent_config keys: {list(agent_config.keys())}")
        print(f"[_generate_response] agent_config max_new_tokens: {agent_config.get('max_new_tokens', 'NOT FOUND')}")
        # =====
        
        generation_config = {
            "max_new_tokens": agent_config.get("max_new_tokens", 512),
            "temperature": agent_config.get("temperature", 0.1),
            "do_sample": agent_config.get("temperature", 0.1) > 0,
            "repetition_penalty": agent_config.get("repetition_penalty", 1.15),
        }
        
        # ===== 打印最终的generation_config =====
        print(f"[_generate_response] Initial generation_config: {generation_config}")
        # =====
        
        # 确保所有输入在同一设备上
        model_device = next(self.model.parameters()).device
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                if model_inputs[key].device != model_device:
                    model_inputs[key] = model_inputs[key].to(model_device)

        # Generate
        try:
            with torch.no_grad():
                # ⭐ 修改：优先使用模型已有的 generation_config 设置
                if hasattr(self.model, 'generation_config'):
                    model_gen_config = self.model.generation_config
                    
                    # ⭐ 从模型的 generation_config 复制已设置的值
                    for key in ['eos_token_id', 'min_new_tokens', 'min_length', 'bad_words_ids', 
                                'forced_eos_token_id', 'suppress_tokens']:
                        if hasattr(model_gen_config, key):
                            value = getattr(model_gen_config, key)
                            
                            # ⭐ 特殊处理 eos_token_id
                            if key == 'eos_token_id':
                                if value is not None:
                                    # 接受列表或整数形式的 eos_token_id
                                    generation_config[key] = value
                                    print(f"[_generate_response] Using model's {key}: {value}")
                            
                            # ⭐ 专门检查 bad_words_ids
                            elif key == 'bad_words_ids':
                                print(f"[_generate_response] Checking bad_words_ids:")
                                print(f"  - hasattr(model_gen_config, 'bad_words_ids'): {hasattr(model_gen_config, 'bad_words_ids')}")
                                print(f"  - value is None: {value is None}")
                                if value is not None:
                                    print(f"  - value type: {type(value)}")
                                    print(f"  - value length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                                    if value:
                                        print(f"  - First 3 items: {value[:3] if len(value) >= 3 else value}")
                                        # 设置到 generation_config
                                        generation_config[key] = value
                                        print(f"  - ✓ Set bad_words_ids to generation_config with {len(value)} tokens")
                                    else:
                                        print(f"  - ⚠️ value is empty list/None")
                                else:
                                    print(f"  - ⚠️ bad_words_ids value is None, not setting")
                            
                            # 其他 key 的处理
                            elif value is not None:
                                generation_config[key] = value
                                print(f"[_generate_response] Using model's {key}: {value}")
                    
                    # 只设置必要的默认值
                    generation_config.update({
                        "pad_token_id": self.processor.tokenizer.pad_token_id,
                        "use_cache": True,
                    })
                    
                    # ⭐ 如果 eos_token_id 还没有被设置，使用默认值
                    if 'eos_token_id' not in generation_config:
                        generation_config['eos_token_id'] = self.processor.tokenizer.eos_token_id
                        print(f"[_generate_response] Using default eos_token_id: {generation_config['eos_token_id']}")
                else:
                    # 如果模型没有 generation_config，使用默认设置
                    generation_config.update({
                        "pad_token_id": self.processor.tokenizer.pad_token_id,
                        "eos_token_id": self.processor.tokenizer.eos_token_id,
                        "use_cache": True,
                    })
                
                # ===== 打印生成前的信息 =====
                print(f"[_generate_response] Generation config before generate:")
                print(f"  - max_new_tokens: {generation_config['max_new_tokens']}")
                print(f"  - temperature: {generation_config['temperature']}")
                print(f"  - eos_token_id: {generation_config.get('eos_token_id', 'None')}")
                print(f"  - pad_token_id: {generation_config['pad_token_id']}")
                print(f"  - min_new_tokens: {generation_config.get('min_new_tokens', 'Not set')}")
                print(f"  - bad_words_ids: {len(generation_config.get('bad_words_ids', [])) if generation_config.get('bad_words_ids') else 'Not set'}")
                # =====
                
                outputs = self.model.generate(
                    **model_inputs,
                    **generation_config
                )
                
                # ===== 打印生成后的信息 =====
                print(f"[_generate_response] Generation complete")
                print(f"  - Output shape: {outputs.shape}")
                print(f"  - Generated tokens: {outputs.shape[1] - model_inputs['input_ids'].shape[1]}")
                # =====
                
        except Exception as e:
            print(f"[_generate_response] ERROR in model.generate(): {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Decode
        try:
            generated_ids = outputs[0][model_inputs['input_ids'].shape[1]:]
            
            # ===== 检查生成的token =====
            print(f"[_generate_response] Generated {len(generated_ids)} tokens")
            if len(generated_ids) > 0:
                print(f"  - First 10 tokens: {generated_ids[:10].tolist()}")
                print(f"  - Last 10 tokens: {generated_ids[-10:].tolist()}")
                
                # 检查是否有特殊token
                if hasattr(self.processor, 'tokenizer'):
                    tokenizer = self.processor.tokenizer
                    # 解码最后几个token看看
                    last_tokens_text = [tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids[-10:]]
                    print(f"  - Last 10 tokens decoded: {last_tokens_text}")
            # =====
            
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # ===== 打印解码后的响应 =====
            print(f"[_generate_response] Decoded response length: {len(response)} chars")
            print(f"[_generate_response] Response preview (first 200): {response[:200]}")
            print(f"[_generate_response] Response preview (last 200): {response[-200:]}")
            # =====
            
        except Exception as e:
            print(f"[_generate_response] ERROR in decoding: {type(e).__name__}: {str(e)}")
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