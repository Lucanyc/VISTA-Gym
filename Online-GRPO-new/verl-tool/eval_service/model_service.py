print("!!! LOADING UPDATED MODEL_SERVICE.PY WITH PROPER IMAGE HANDLING !!!")
import sys
sys.path.insert(0, '/mnt/nfs/meng/Online-GRPO-new/verl-tool')
from verl_tool.patches import qwen25vl_no_trunc_no_video

import time
import uuid
import hashlib
import aiohttp
import requests
import regex as re
import os
import torch
from typing import Dict, Any, List, Tuple, Optional
from config import ModelConfig, ToolConfig
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import asyncio
import json
from PIL import Image

# Control character sanitizer
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def sanitize_request(obj: Any) -> Any:
    """Recursively sanitize request objects"""
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj

class ModelService:
    """Direct inference service using Transformers with proper image handling"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """Initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.session = None
        self.device = None
        
    def fix_tool_call_format(self, tool_call_content: str, user_question: str = None) -> str:
        """Fix tool call format to match expected standards"""
        try:
            # Parse JSON
            obj = json.loads(tool_call_content)
            
            # Standardize tool name to lowercase
            if "tool" in obj:
                tool_name = obj["tool"].lower()
            else:
                # If no tool field, try to infer from content
                content_lower = tool_call_content.lower()
                if "chart" in content_lower:
                    tool_name = "chartmoe"
                else:
                    return tool_call_content  # Can't fix, return original
            
            # Fix based on tool type - chartmoe特别处理
            if tool_name == "chartmoe":
                # Standard format: {"tool": "chartmoe", "task": "to_table"}
                fixed_obj = {"tool": "chartmoe", "task": "to_table"}
                
                # If there's a parameters field with task, extract it
                if "parameters" in obj and isinstance(obj["parameters"], dict):
                    task = obj["parameters"].get("task", "to_table")
                    # Standardize task value
                    if isinstance(task, str):
                        task_lower = task.lower().replace(" ", "_")
                        # 各种可能的变体都转换为to_table
                        if "table" in task_lower or "to_table" in task_lower:
                            task = "to_table"
                    fixed_obj["task"] = task
                elif "task" in obj:
                    # Task is at top level
                    task = str(obj["task"]).lower().replace(" ", "_")
                    if "table" in task or "to_table" in task:
                        task = "to_table"
                    fixed_obj["task"] = task
                
                return json.dumps(fixed_obj, ensure_ascii=False)
            
            # 其他工具保持原样
            return tool_call_content
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract tool name and return default format
            content_lower = tool_call_content.lower()
            
            if "chartmoe" in content_lower or "chart" in content_lower:
                return json.dumps({"tool": "chartmoe", "task": "to_table"})
            
            # If nothing matches, return original
            return tool_call_content
        
        except Exception as e:
            print(f"Error fixing tool call format: {e}")
            return tool_call_content
        
    def load_model(self):
        """Load the model using Transformers directly"""
        print(f"Loading Model using Transformers: {self.model_config.model}...")
        
        # Determine device allocation
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
        num_gpus = len(gpu_ids)
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model,
            trust_remote_code=True
        )
        
        # Load processor for handling images
        self.processor = AutoProcessor.from_pretrained(
            self.model_config.model,
            trust_remote_code=True
        )
        
        # Configure device map for multi-GPU
        if num_gpus > 1:
            device_map = "auto"
            self.device = None
        else:
            device_map = None
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
        
        # Load Qwen2.5-VL model specifically
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        
        if device_map is None and self.device:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set up mtrl_sep if needed
        if self.tool_config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            self.tool_config.mtrl_sep = "\n" + self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        print(f"✅ Model loaded successfully on {device_map if device_map else self.device}")
    
    def load_images_from_extra_fields(self, extra_fields: Optional[List[Dict]]) -> List[Image.Image]:
        """Load images from extra_fields"""
        images = []
        
        if not extra_fields:
            return images
        
        # Extract first extra_field (single sample)
        extra = extra_fields[0] if isinstance(extra_fields, list) else extra_fields
        
        # Try different possible keys for images
        image_paths = []
        if 'images' in extra:
            img_data = extra['images']
            if isinstance(img_data, list):
                for item in img_data:
                    if isinstance(item, dict) and 'image' in item:
                        image_paths.append(item['image'])
                    elif isinstance(item, str):
                        image_paths.append(item)
            elif isinstance(img_data, str):
                image_paths.append(img_data)
        elif 'image' in extra:
            if isinstance(extra['image'], str):
                image_paths.append(extra['image'])
        
        # Load images
        for path in image_paths:
            try:
                if os.path.exists(path):
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                    print(f"Loaded image: {path}")
                else:
                    print(f"Image not found: {path}")
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        
        return images
    
    def call_tool_server(self, trajectory_ids: List[str], actions: List[str], 
                        finish: List[bool], extra_fields: Optional[List[Dict]] = None,
                        **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Query the tool server for observations"""
        server_url = self.tool_config.tool_server_url
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            "is_last_step": finish,  # 添加is_last_step
            **kwargs
        }
        
        # 修复extra_fields格式 - 重要！
        if extra_fields:
            fixed_extra_fields = []
            for extra in (extra_fields if isinstance(extra_fields, list) else [extra_fields]):
                fixed_extra = {}
                
                # 处理images字段，提取路径为字符串列表
                if 'images' in extra:
                    img_data = extra['images']
                    image_paths = []
                    
                    if isinstance(img_data, list):
                        for item in img_data:
                            if isinstance(item, dict) and 'image' in item:
                                # 从字典中提取路径
                                image_paths.append(item['image'])
                            elif isinstance(item, str):
                                # 已经是字符串路径
                                image_paths.append(item)
                    elif isinstance(img_data, str):
                        image_paths.append(img_data)
                    
                    # 设置修复后的images列表（仅包含字符串路径）
                    fixed_extra['images'] = image_paths
                
                # 复制其他字段
                for key, value in extra.items():
                    if key != 'images':
                        fixed_extra[key] = value
                
                fixed_extra_fields.append(fixed_extra)
            
            data["extra_fields"] = fixed_extra_fields
            print(f"Fixed extra_fields format: images are now {type(fixed_extra_fields[0].get('images', [None])[0]) if fixed_extra_fields and fixed_extra_fields[0].get('images') else 'None'}")
       
        try:
            data = sanitize_request(data)
            print(f"Sending to tool server: {json.dumps(data, indent=2)}")
            response = requests.post(server_url, json=data)
            print(f"Tool server response status: {response.status_code}")
            print(f"Tool server response: {response.text[:500]}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error: {str(e)}" for _ in trajectory_ids],
                "dones": [True for _ in trajectory_ids],
                "valids": [False for _ in trajectory_ids]
            }
    
    async def call_tool_server_async(self, trajectory_ids: List[str], actions: List[str],
                                    finish: List[bool], extra_fields: Optional[List[Dict]] = None,
                                    **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Async version of tool server call with extra_fields support"""
        # 使用同步版本的逻辑
        return self.call_tool_server(trajectory_ids, actions, finish, extra_fields, **kwargs)
    
    def process_observations(self, next_obs: List[str], dones: List[bool], 
                            valid_action: List[bool], finishs: List[bool]) -> List[str]:
        """Process observations"""
        next_obs = [obs if not done else "" for obs, done in zip(next_obs, dones)]
        
        # Tokenize and truncate if needed
        if self.tool_config.truncate_obs_side == 'left':
            next_obs_ids = self.tokenizer(
                next_obs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
                padding_side='left',
            )['input_ids']
            if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                next_obs_ids = next_obs_ids[:, -self.tool_config.max_obs_length:]
        else:
            next_obs_ids = self.tokenizer(
                next_obs,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
                padding_side='right',
            )['input_ids']
            if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                next_obs_ids = next_obs_ids[:, :self.tool_config.max_obs_length]
        
        next_obs = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)
        
        # Apply mtrl formatting if enabled
        if self.tool_config.enable_mtrl:
            mtrl_sep = self.tool_config.mtrl_sep
            processed_next_obs = []
            for i in range(len(next_obs)):
                if finishs[i] or dones[i]:
                    processed_next_obs.append("")
                elif valid_action[i]:
                    processed_next_obs.append(mtrl_sep.format(obs=next_obs[i]))
                else:
                    processed_next_obs.append(mtrl_sep.format(
                        obs="Your action is not valid, please check the format and try again." + next_obs[i]
                    ))
            next_obs = processed_next_obs
        
        return next_obs
    
    #async def generate_text(self, prompt: str, sampling_params: dict, 
    #                      images: Optional[List[Image.Image]] = None) -> str:
    #    """Generate text using the model with optional image inputs"""
        
    #    if not isinstance(prompt, str):
    #        if isinstance(prompt, list):
    #            prompt = ' '.join(str(p) for p in prompt)
    #        else:
    #            prompt = str(prompt)
        
        # 确保prompt不为空
    #    if not prompt:
    #        prompt = ""
        
    #    if images and self.processor:
    #        inputs = self.processor(
    #            text=prompt,
    #            images=images,
    #            return_tensors="pt",
    #            padding=True
    #        )
    #        print(f"Processing with {len(images)} image(s)")
    #        print(f"Input shape: {inputs['input_ids'].shape}")
    #        print(f"First 50 tokens: {inputs['input_ids'][0][:50].tolist()}")
            
            # 检查是否包含图片token（用于调试）
    #        special_tokens = [151655, 151656, 151657]  # Qwen2-VL的图片相关token
    #        for token_id in special_tokens:
    #            if token_id in inputs['input_ids'][0].tolist():
    #                print(f"Found special token {token_id}")
    #    else:
            # Text-only input - 也要确保prompt是字符串
    #        try:
    #            inputs = self.tokenizer(prompt, return_tensors="pt")
    #        except Exception as e:
    #            print(f"Tokenizer error with prompt type {type(prompt)}: {e}")
    #            print(f"Prompt content: {prompt[:500] if prompt else 'Empty'}")
    #            raise

    #    if images and self.processor:
    #        inputs = self.processor(
    #            text=prompt,
    #            images=images,
    #            return_tensors="pt",
    #            padding=True
    #        )
    #        print(f"Processing with {len(images)} image(s)")
    #        print(f"Input shape: {inputs['input_ids'].shape}")
    #        print(f"First 50 tokens: {inputs['input_ids'][0][:50].tolist()}")
            
            # 检查是否包含图片token（用于调试）
    #        special_tokens = [151655, 151656, 151657]  # Qwen2-VL的图片相关token
    #        for token_id in special_tokens:
    #            if token_id in inputs['input_ids'][0].tolist():
    #                print(f"Found special token {token_id}")
    #    else:
            # Text-only input
    #        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Ensure inputs are on the correct device
    #    if self.device:
    #        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    #    elif hasattr(self.model, 'device'):
    #        model_device = next(self.model.parameters()).device
    #        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Prepare generation kwargs
    #    stop_tokens = sampling_params.get("stop", [])
    #    filtered_stop_tokens = [tok for tok in stop_tokens if tok not in self.tool_config.action_stop_tokens]
        
    #    gen_kwargs = {
    #        "max_new_tokens": sampling_params.get("max_tokens", 512),
    #        "temperature": sampling_params.get("temperature", 1.0),
    #        "top_p": sampling_params.get("top_p", 1.0),
    #        "do_sample": sampling_params.get("temperature", 1.0) > 0,
    #        "pad_token_id": self.tokenizer.pad_token_id,
    #        "eos_token_id": self.tokenizer.eos_token_id,
    #    }
        
        # Add filtered stop tokens
    #    if filtered_stop_tokens:
    #        stop_token_ids = []
    #        for token in filtered_stop_tokens:
    #            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
    #            if token_ids:
    #                stop_token_ids.extend(token_ids)
    #        if stop_token_ids:
    #            gen_kwargs["eos_token_id"] = stop_token_ids
        
        # Generate
    #    with torch.no_grad():
    #        output_ids = await asyncio.get_event_loop().run_in_executor(
    #           None,
    #            lambda: self.model.generate(**inputs, **gen_kwargs)
    #        )
        
        # Decode output (remove input tokens)
    #    input_len = inputs.get('input_ids').shape[1]
    #    output_text = self.tokenizer.decode(
    #        output_ids[0][input_len:],
    #        skip_special_tokens=False
    #    )
        
    #    return output_text
    

    async def generate_text(self, prompt: str, sampling_params: dict, 
                      images: Optional[List[Image.Image]] = None) -> str:
        """Generate text using the model with optional image inputs"""
        
        # 确保prompt是字符串类型
        if not isinstance(prompt, str):
            print(f"WARNING: prompt type is {type(prompt)}, converting to str")
            if isinstance(prompt, list):
                prompt = ' '.join(str(p) for p in prompt)
            else:
                prompt = str(prompt)
        
        # 确保prompt不为空
        if not prompt:
            prompt = ""
        
        # 明确判断：只有真正有图片时才用processor
        if images is not None and len(images) > 0 and self.processor:
            print(f"Processing with {len(images)} image(s)")
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True
            )
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"First 50 tokens: {inputs['input_ids'][0][:50].tolist()}")
            
            # 检查是否包含图片token（用于调试）
            special_tokens = [151655, 151656, 151657]  # Qwen2-VL的图片相关token
            for token_id in special_tokens:
                if token_id in inputs['input_ids'][0].tolist():
                    print(f"Found special token {token_id}")
        else:
            # Text-only input - 使用tokenizer
            print(f"Text-only generation (no images)")
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            except Exception as e:
                print(f"Tokenizer error with prompt type {type(prompt)}: {e}")
                print(f"Prompt content: {prompt[:500] if prompt else 'Empty'}")
                raise
        
        # Ensure inputs are on the correct device
        if self.device:
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        elif hasattr(self.model, 'device'):
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Prepare generation kwargs
        stop_tokens = sampling_params.get("stop", [])
        filtered_stop_tokens = [tok for tok in stop_tokens if tok not in self.tool_config.action_stop_tokens]
        
        gen_kwargs = {
            "max_new_tokens": sampling_params.get("max_tokens", 512),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "do_sample": sampling_params.get("temperature", 1.0) > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add filtered stop tokens
        if filtered_stop_tokens:
            stop_token_ids = []
            for token in filtered_stop_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    stop_token_ids.extend(token_ids)
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = stop_token_ids
        
        # Generate
        with torch.no_grad():
            output_ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate(**inputs, **gen_kwargs)
            )
        
        # Decode output (remove input tokens)
        input_len = inputs.get('input_ids').shape[1]
        output_text = self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=False
        )
        
        return output_text


    async def generate_with_tools(self, prompt: str, sampling_params: dict, 
                                 images: Optional[List[Image.Image]] = None,
                                 extra_fields: Optional[List[Dict]] = None,
                                 user_question: str = None) -> Tuple[List[str], List[str]]:
        """Generate text with tool calls and image support"""
        
        context = prompt
        final_response = ""
        

        # 或者使用基于内容的哈希（生产环境）
        traj_id = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        finish_reason = "stop"
        
        for action_step in range(self.tool_config.max_turns + 1):
            print(f"\n=== Turn {action_step}/{self.tool_config.max_turns} ===")
            
            # Last turn: remove action stop tokens
            current_sampling_params = sampling_params.copy()
            if action_step == self.tool_config.max_turns:
                current_sampling_params["stop"] = [tok for tok in sampling_params.get("stop", []) 
                                                  if tok not in self.tool_config.action_stop_tokens]
            
            # Only pass images on first turn
            current_images = images if action_step == 0 else None
            response = await self.generate_text(context, current_sampling_params, current_images)
            print(f"Generated response: {response[:200]}...")
            
            # Check if this is a tool call
            has_tool_call = any(token in response for token in self.tool_config.action_stop_tokens)
            
            # Determine if we should finish
            if action_step >= self.tool_config.max_turns:
                finish = True
                print("Reached max turns, finishing")
            elif has_tool_call:
                finish = False
                print(f"Found tool call marker, will call tool server")
                
                if self.tool_config.enable_mtrl and self.tool_config.turn_end_token:
                    response += self.tool_config.turn_end_token
            else:
                if not response or response.strip().endswith(('<|im_end|>', '<|endoftext|>')):
                    finish = True
                    print("Natural end detected")
                else:
                    finish = action_step >= self.tool_config.max_turns - 1
                    print(f"No tool call, finish={finish}")
            
            # Call tool server if tool call detected and not finished
            if has_tool_call and not finish:
                # Extract clean tool call
                import re
                tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
                if tool_match:
                    tool_content = tool_match.group(1).strip()
                    
                    # 修复工具调用格式
                    fixed_tool_content = self.fix_tool_call_format(tool_content, user_question)
                    
                    clean_tool_call = f"<tool_call>{fixed_tool_content}</tool_call>"
                    
                    print(f"Original tool call: {tool_content[:100]}...")
                    print(f"Fixed tool call: {fixed_tool_content[:100]}...")
                    print(f"Calling tool server with fixed action: {clean_tool_call[:100]}...")
                    
                    # Pass extra_fields to tool server
                    tool_result = await self.call_tool_server_async(
                        [traj_id], [clean_tool_call], [False], extra_fields
                    )
                else:
                    print(f"Calling tool server with original action: {response[:100]}...")
                    tool_result = await self.call_tool_server_async(
                        [traj_id], [response], [False], extra_fields
                    )
                
                print(f"Tool server returned: valid={tool_result['valids'][0]}, done={tool_result['dones'][0]}")
                
                #observation = self.process_observations(
                #    tool_result["observations"],
                #    tool_result["dones"],
                #    tool_result["valids"],
                #    [False]
                #)[0]
                

                # 处理可能的字典格式（当工具调用无效时）
                observations = tool_result["observations"]
                fixed_observations = []
                for obs in observations:
                    if isinstance(obs, dict):
                        # 工具调用无效时返回的错误信息
                        if 'invalid_reason' in obs:
                            obs_str = obs.get('obs', '')
                            if obs.get('invalid_reason'):
                                obs_str += obs['invalid_reason']
                            if obs.get('available_tools'):
                                obs_str += '\n' + obs['available_tools']
                        else:
                            obs_str = json.dumps(obs)
                    else:
                        obs_str = str(obs) if obs else ""
                    fixed_observations.append(obs_str)

                # 使用修复后的observations
                observation = self.process_observations(
                    fixed_observations,  # 注意这里改成了fixed_observations
                    tool_result["dones"],
                    tool_result["valids"],
                    [False]
                )[0]
                print(f"Processed observation: {observation[:200]}...")
                
                # Add to context for next turn
                context += response + observation
                final_response += response + observation
                
                if tool_result["dones"][0]:
                    print("Tool server indicates done")
                    break
            else:
                context += response
                final_response += response
                
                if finish:
                    print("Finishing generation")
                    break
        
        print(f"=== Final response length: {len(final_response)} ===\n")
        return [final_response], [finish_reason]
    
    async def chat_completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat completion request with proper image support"""
        if "messages" not in body or not body["messages"]:
            raise ValueError("No messages found in the request.")
        
        # Extract extra_fields and load images
        extra_fields = body.get("extra_fields", None)
        images = self.load_images_from_extra_fields(extra_fields) if extra_fields else []
        
        messages = body['messages']
        
        # Extract user question for tool call fixing
        user_question = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                user_question = msg['content']
                if isinstance(user_question, list):
                    # Extract text from structured content
                    for item in user_question:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            user_question = item['text']
                            break
                break
        
        # 如果有图片，需要将messages转换为Qwen2.5-VL需要的结构化格式
        if images:
            # 找到最后一条用户消息并转换格式
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]['role'] == 'user':
                    original_content = messages[i]['content']
                    
                    # 构建新的content列表（Qwen2.5-VL需要的格式）
                    new_content = []
                    
                    # 处理原始内容，分离图片占位符和文本
                    if isinstance(original_content, str):
                        # 移除所有<image>标记，保留纯文本
                        text_content = original_content.replace('<image>', '').strip()
                        
                        # 先添加所有图片
                        for _ in range(len(images)):
                            new_content.append({"type": "image"})
                        
                        # 再添加文本
                        if text_content:
                            new_content.append({"type": "text", "text": text_content})
                    elif isinstance(original_content, list):
                        # 如果已经是列表格式，保持不变
                        new_content = original_content
                    else:
                        # 其他格式，转为文本
                        new_content = [{"type": "text", "text": str(original_content)}]
                    
                    # 更新消息格式
                    messages[i] = {
                        "role": messages[i]['role'],
                        "content": new_content
                    }
                    print(f"Converted message to structured format with {len(images)} images")
                    break
        
        # 使用processor处理messages
        try:
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            print("Using processor.apply_chat_template")
        except Exception as e:
            print(f"processor.apply_chat_template failed: {e}")
            # 如果processor不支持，回退到tokenizer
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            print("Using tokenizer.apply_chat_template (fallback)")
        
        # 调试信息
        print(f"Number of images loaded: {len(images)}")
        if images:
            print(f"Prompt preview: {prompt[:500]}...")
        
        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }
        
        # Generate with images
        responses, finish_reasons = await self.generate_with_tools(
            prompt, sampling_params, images=images, extra_fields=extra_fields,
            user_question=user_question
        )
        
        # Calculate tokens (approximate)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(responses[0]))
        
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": responses[0],
                },
                "finish_reason": finish_reasons[0]
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    
    def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Sync wrapper for chat completions"""
        return asyncio.run(self.chat_completions_async(body))
    
    async def completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process completion request"""
        prompt = body['prompt']
        extra_fields = body.get("extra_fields", None)
        images = self.load_images_from_extra_fields(extra_fields) if extra_fields else []
        
        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }
        
        responses, finish_reasons = await self.generate_with_tools(
            prompt, sampling_params, images=images, extra_fields=extra_fields
        )
        
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(responses[0]))
        
        return {
            "id": f"cmpl-{str(uuid.uuid4())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": [{
                "index": 0,
                "text": responses[0],
                "finish_reason": finish_reasons[0]
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    
    def completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Sync wrapper for completions"""
        return asyncio.run(self.completions_async(body))
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def __del__(self):
        """Destructor"""
        try:
            asyncio.run(self.close())
        except:
            pass