"""VLM Agent implementation with reflection support and structured output"""
#基础VLM类，提供基本的视觉-语言模型功能

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer

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
        self.tokenizer = None  # 添加 tokenizer 属性
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
        
        # ========== 特殊处理 InternVL3 ==========
        if "InternVL3" in model_name:
            # InternVL3 使用 AutoModel 和 AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Model loading parameters for InternVL3
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": agent_config.get("device_map", "auto"),
            }
            
            # Add torch dtype if specified
            if "torch_dtype" in agent_config:
                dtype_str = agent_config["torch_dtype"]
                if dtype_str == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif dtype_str == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16  # InternVL3 推荐使用 bfloat16
            
            self.model = AutoModel.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # 设置为评估模式
            self.model.eval()
            
            # InternVL3 使用 tokenizer 作为 processor
            self.processor = self.tokenizer
            
        else:
            # 原有的加载逻辑（Qwen 等模型）
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
        """Prepare input for the VLM model with reflection support"""
        # 获取模型名称以判断类型
        model_name = self.config.get("model_name", "") if hasattr(self.config, 'get') else \
                     getattr(self.config, 'model_name', "")
        
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
            prompt = self._build_reflection_prompt(observation)
        else:
            if choices:
                prompt = format_prompt_with_choices(question, choices)
            else:
                prompt = question
        
        # 添加结构化输出格式指令
        output_format_instruction = observation.get('output_format_instruction', '')
        if output_format_instruction:
            prompt = f"{prompt}\n\n{output_format_instruction}"
        
        # Build messages
        if 'conversation_history' in observation and observation['conversation_history']:
            messages = self._build_conversation_messages(
                observation['conversation_history'], 
                image, 
                prompt
            )
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
        
        # ========== InternVL3 特殊处理 ==========
        if "InternVL3" in model_name:
            # InternVL3 直接使用 messages 和 image
            inputs = {
                '_messages': messages,
                '_image': image,
                '_observation': observation,
                '_has_reflection': has_reflection
            }
        else:
            # 原有的处理逻辑（Qwen 等）
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
            
            # Store additional info
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
    
    
    def _generate_response(self, inputs: Dict[str, Any]) -> str:
        """Generate response using the model"""
        # 获取模型名称
        model_name = self.config.get("model_name", "") if hasattr(self.config, 'get') else \
                    getattr(self.config, 'model_name', "")
        
        # ========== InternVL3 特殊处理 ==========
        if "InternVL3" in model_name:
            messages = inputs['_messages']
            image = inputs['_image']  # PIL Image
            
            try:
                # 从消息中提取查询文本
                query = ""
                last_user_message = None
                for msg in reversed(messages):
                    if msg['role'] == 'user':
                        last_user_message = msg
                        break
                
                if last_user_message and isinstance(last_user_message['content'], list):
                    for item in last_user_message['content']:
                        if item.get('type') == 'text':
                            query = item.get('text', '')
                            break
                elif last_user_message and isinstance(last_user_message['content'], str):
                    query = last_user_message['content']
                else:
                    # 从observation获取问题
                    query = inputs.get('_observation', {}).get('question', '')
                
                # ===== 使用InternVL3官方的图像处理方法 =====
                # 定义必要的常量和函数（从官方文档复制）
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                
                def build_transform(input_size):
                    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD)
                    ])
                    return transform
                
                def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
                    best_ratio_diff = float('inf')
                    best_ratio = (1, 1)
                    area = width * height
                    for ratio in target_ratios:
                        target_aspect_ratio = ratio[0] / ratio[1]
                        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                        if ratio_diff < best_ratio_diff:
                            best_ratio_diff = ratio_diff
                            best_ratio = ratio
                        elif ratio_diff == best_ratio_diff:
                            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                                best_ratio = ratio
                    return best_ratio
                
                def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
                    orig_width, orig_height = image.size
                    aspect_ratio = orig_width / orig_height
                    
                    # calculate the existing image aspect ratio
                    target_ratios = set(
                        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                        i * j <= max_num and i * j >= min_num)
                    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
                    
                    # find the closest aspect ratio to the target
                    target_aspect_ratio = find_closest_aspect_ratio(
                        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
                    
                    # calculate the target width and height
                    target_width = image_size * target_aspect_ratio[0]
                    target_height = image_size * target_aspect_ratio[1]
                    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
                    
                    # resize the image
                    resized_img = image.resize((target_width, target_height))
                    processed_images = []
                    for i in range(blocks):
                        box = (
                            (i % (target_width // image_size)) * image_size,
                            (i // (target_width // image_size)) * image_size,
                            ((i % (target_width // image_size)) + 1) * image_size,
                            ((i // (target_width // image_size)) + 1) * image_size
                        )
                        # split the image
                        split_img = resized_img.crop(box)
                        processed_images.append(split_img)
                    assert len(processed_images) == blocks
                    if use_thumbnail and len(processed_images) != 1:
                        thumbnail_img = image.resize((image_size, image_size))
                        processed_images.append(thumbnail_img)
                    return processed_images
                
                # 处理图像
                input_size = 448
                max_num = 12
                
                # 确保image是PIL Image格式
                if not hasattr(image, 'size'):
                    from PIL import Image as PILImage
                    if isinstance(image, torch.Tensor):
                        # 如果是tensor，转换为PIL Image
                        image = T.ToPILImage()(image)
                    elif isinstance(image, str):
                        # 如果是路径，加载图像
                        image = PILImage.open(image).convert('RGB')
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                
                # 使用dynamic_preprocess处理图像
                transform = build_transform(input_size=input_size)
                images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(img) for img in images]
                pixel_values = torch.stack(pixel_values)
                
                # 转换为正确的dtype和device
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
                
                # 使用官方的chat方法
                generation_config = {
                    'max_new_tokens': 512,
                    'do_sample': False,
                    'temperature': 0.3,
                }
                
                # 格式化问题 - InternVL3期望问题中有<image>标记
                if '<image>' not in query:
                    query = '<image>\n' + query
                
                # 调用模型的chat方法
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=query,
                    generation_config=generation_config,
                    history=None,
                    return_history=False
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error with InternVL3 generation: {e}")
                import traceback
                traceback.print_exc()
                
                # 备用方案：尝试纯文本对话
                try:
                    logger.warning("Trying pure text conversation without image")
                    generation_config = {
                        'max_new_tokens': 256,
                        'do_sample': False
                    }
                    
                    response = self.model.chat(
                        tokenizer=self.tokenizer,
                        pixel_values=None,  # 不使用图像
                        question=query,
                        generation_config=generation_config,
                        history=None,
                        return_history=False
                    )
                    
                    return response
                    
                except Exception as e2:
                    logger.error(f"Pure text generation also failed: {e2}")
                    raise e
        
        else:
            # 原有的生成逻辑（Qwen 等）
            # Extract model inputs (remove our custom fields)
            model_inputs = {k: v for k, v in inputs.items() 
                        if not k.startswith('_')}
            
            # Get generation config
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