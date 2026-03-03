"""VLM Agent implementation with reflection support and structured output"""
#基础VLM类，提供基本的视觉-语言模型功能

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

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
        
        # Handle configuration
        if hasattr(self.config, 'get'):
            agent_config = self.config.get("agent", {})
        elif hasattr(self.config, '__dict__'):
            config_dict = self.config.__dict__
            agent_config = config_dict.get("agent", config_dict)
        else:
            agent_config = {}
            
        model_name = agent_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        logger.info(f"Loading VLM model: {model_name}")
        
        # Load processor
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
        
        # ⭐ 关键修改：根据模型名称选择正确的模型类
        if "72B" in model_name:
            # 72B模型使用不同的类
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            # 7B模型使用原来的类
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        # Set device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        # Validate observation
        is_valid, error_msg = validate_observation(observation)
        if not is_valid:
            raise ValueError(f"Invalid observation: {error_msg}")
        
        # Extract components
        image_path = observation['image_path']
        question = observation['question']
        choices = observation.get('choices', None)
        
        # Load image
        image = load_image_safely(image_path)
        
        # Check for reflection context
        has_reflection = 'feedback' in observation or 'previous_answer' in observation
        
        # Build prompt based on context
        if has_reflection:
            # Reflection prompt - build special prompt for retry attempts
            prompt = self._build_reflection_prompt(observation)
        else:
            # Standard prompt
            if choices:
                prompt = format_prompt_with_choices(question, choices)
            else:
                prompt = question
        
        # ========== 新增：添加结构化输出格式指令 ==========
        output_format_instruction = observation.get('output_format_instruction', '')
        if output_format_instruction:
            # 将格式指令添加到 prompt
            prompt = f"{prompt}\n\n{output_format_instruction}"
        
        # Build messages with potential conversation history
        if 'conversation_history' in observation and observation['conversation_history']:
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
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        # Store additional info for response parsing
        inputs['_observation'] = observation
        inputs['_messages'] = messages
        inputs['_has_reflection'] = has_reflection
        
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
        if 'pixel_values' in model_inputs:
            print(f"[_generate_response] Image input detected")
            print(f"[_generate_response] pixel_values shape: {model_inputs['pixel_values'].shape}")
        elif 'images' in model_inputs:
            print(f"[_generate_response] images found in inputs")
        else:
            print(f"[_generate_response] No image inputs found")
        
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
            "temperature": agent_config.get("temperature", 0.3),
            "do_sample": agent_config.get("temperature", 0.3) > 0,
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
            print(f"[_generate_response] ERROR in decoding: {type(e).__name__}: {str(e)}")
            raise
        
        return response
    
    
    
    def _parse_response(self, response: str, observation: Dict[str, Any]) -> str:
        """Parse response and format as action"""
        # Clean up response
        response = response.strip()
        
        # ========== 新增：如果使用结构化输出，不要修改响应内容 ==========
        # 保留原始响应，让 ChartQATaskWrapper 去解析 <think> 和 <answer> 标签
        use_structured = observation.get('use_structured_output', False)
        if use_structured and ('<think>' in response or '<answer>' in response):
            # 如果响应包含结构化标签，保持原样
            # 但仍然需要包装在 answer_question 格式中
            escaped_response = response.replace('"', '\\"')
            return f'answer_question(answer="{escaped_response}")'
        
        # If response is already in action format, return as is
        if response.startswith("answer_question("):
            return response
        
        # Otherwise, format as answer action
        # Escape quotes in the response
        escaped_response = response.replace('"', '\\"')
        return f'answer_question(answer="{escaped_response}")'
    
    def reset(self):
        """Reset agent state"""
        # VLMAgent is stateless, so nothing to reset
        pass