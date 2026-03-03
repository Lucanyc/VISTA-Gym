# 修改 /data/wang/meng/GYM-Work/vlm_gym-tool-usage-geometry/vlm_gym/environments/tools/geometry_tools/diagram_formalizer.py

"""
DiagramFormalizer Tool for VLMGym
"""
import torch
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import os
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from vlm_gym.environments.tools.base import ToolBase

# 全局模型实例
_model_instance = None
_tokenizer_instance = None


def get_model_and_tokenizer():
    global _model_instance, _tokenizer_instance
    
    if _model_instance is None:
        print("首次加载 DiagramFormalizer 模型...")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        model_path = "/workspace/geometry/model/diagramformalizer"
        
        try:
            # 打印GPU信息
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"检测到 {gpu_count} 个GPU")
                for i in range(gpu_count):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 将DiagramFormalizer加载到第二个GPU
            if torch.cuda.device_count() > 1:
                device_map = {"": 1}  # 使用cuda:1
                print("将 DiagramFormalizer 加载到 GPU 1")
            else:
                device_map = "auto"
                print("使用自动设备映射")
            
            _model_instance = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device_map,
                low_cpu_mem_usage=True
            )
            
            _tokenizer_instance = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                padding_side="right",
                trust_remote_code=True
            )
            
            # 设置为评估模式
            _model_instance.eval()
            
            print("✓ DiagramFormalizer 模型加载完成")
            
            # ===== 添加模型配置检查 =====
            print("\n[模型配置检查]")
            if hasattr(_model_instance, 'config'):
                config = _model_instance.config
                
                # 打印重要配置属性
                important_attrs = ['model_type', 'architectures', 'vocab_size', 
                                 'hidden_size', 'num_hidden_layers']
                print("基本配置：")
                for attr_name in important_attrs:
                    if hasattr(config, attr_name):
                        print(f"  - {attr_name}: {getattr(config, attr_name)}")
                
                # 特别检查图像相关配置
                image_attrs = ['image_token_index', 'image_token_id', 'vision_config', 
                             'image_seq_length', 'image_processor', 'vision_tower',
                             'mm_vision_tower', 'mm_projector_type', 'image_aspect_ratio']
                print("\n图像相关配置：")
                found_image_config = False
                for attr in image_attrs:
                    if hasattr(config, attr):
                        value = getattr(config, attr)
                        print(f"  - config.{attr}: {value}")
                        found_image_config = True
                
                if not found_image_config:
                    print("  - 未找到图像相关配置")
                
                # 检查是否有 process_images 方法
                print("\n模型方法检查：")
                if hasattr(_model_instance, 'process_images'):
                    print("  - ✓ 模型有 process_images 方法")
                else:
                    print("  - ✗ 模型没有 process_images 方法")
                
                if hasattr(_model_instance, 'get_vision_tower'):
                    print("  - ✓ 模型有 get_vision_tower 方法")
                else:
                    print("  - ✗ 模型没有 get_vision_tower 方法")
            # ===== 配置检查结束 =====
            
            # 打印模型所在设备
            if hasattr(_model_instance, 'device'):
                print(f"\n模型设备: {_model_instance.device}")
            else:
                print(f"\n模型第一个参数设备: {next(_model_instance.parameters()).device}")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    
    return _model_instance, _tokenizer_instance


class DiagramFormalizerTool(ToolBase):
    name = "diagram_formalizer"
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self.current_image = None
        self.current_image_path = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 使用第二个GPU（cuda:1）
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device("cuda:1")
            print(f"[DiagramFormalizer] 使用 GPU 1 (实际是物理GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[1]})")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("[DiagramFormalizer] 只有一个GPU可用，使用 GPU 0")
        else:
            self.device = torch.device("cpu")
            print("[DiagramFormalizer] 没有GPU可用，使用CPU")
    
    def _load_model(self):
        if not self._model_loaded:
            try:
                # 清理内存
                self._cleanup_memory()
                
                self.model, self.tokenizer = get_model_and_tokenizer()
                self._model_loaded = True
                
                # ===== 检查 tokenizer 配置 =====
                print("\n[Tokenizer 配置检查]")
                print(f"  - Tokenizer 类型: {type(self.tokenizer).__name__}")
                print(f"  - 词汇表大小: {len(self.tokenizer)}")
                
                # 检查特殊 tokens
                special_tokens = ['bos', 'eos', 'unk', 'pad']
                for token_name in special_tokens:
                    token_id = getattr(self.tokenizer, f"{token_name}_token_id", None)
                    if token_id is not None:
                        print(f"  - {token_name}_token_id: {token_id}")
                
                # 再次清理
                self._cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"模型加载失败: {e}")
                raise e
    
    def _cleanup_memory(self):
        """清理GPU/CPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def reset(self, image=None, image_path=None):
        """重置工具并保存图像"""
        # 清理内存
        self._cleanup_memory()
        
        if image is not None:
            # 确保是PIL Image
            if not isinstance(image, Image.Image):
                print(f"[DEBUG DiagramFormalizer.reset] Converting from {type(image)} to PIL Image")
                try:
                    if hasattr(image, 'numpy'):
                        # 如果是tensor，先转换为numpy
                        image_array = image.cpu().numpy() if hasattr(image, 'cpu') else image.numpy()
                        # 检查数组形状并调整
                        if len(image_array.shape) == 4:  # batch dimension
                            image_array = image_array[0]
                        if image_array.shape[0] in [3, 4]:  # channels first
                            image_array = np.transpose(image_array, (1, 2, 0))
                        # 确保值在正确范围
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        else:
                            image_array = image_array.astype(np.uint8)
                        image = Image.fromarray(image_array)
                    elif isinstance(image, np.ndarray):
                        # 直接是numpy array
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image.astype(np.uint8))
                    else:
                        # 尝试直接转换
                        image = Image.fromarray(image)
                except Exception as e:
                    print(f"[DEBUG DiagramFormalizer.reset] Failed to convert image: {e}")
                    self.current_image = None
                    self.logger.error(f"Failed to convert image to PIL format: {e}")
                    return
            
            # 转换为RGB格式
            original_mode = image.mode
            if image.mode == 'RGBA':
                # RGBA需要特殊处理，创建白色背景
                print(f"[DEBUG DiagramFormalizer.reset] Converting RGBA to RGB with white background")
                background = Image.new('RGB', image.size, (255, 255, 255))
                # 使用alpha通道作为mask
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode == 'L':
                # 灰度图转RGB
                print(f"[DEBUG DiagramFormalizer.reset] Converting grayscale to RGB")
                image = image.convert('RGB')
            elif image.mode == 'P':
                # 调色板模式转RGB
                print(f"[DEBUG DiagramFormalizer.reset] Converting palette mode to RGB")
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                # 其他格式直接转换
                print(f"[DEBUG DiagramFormalizer.reset] Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            self.current_image = image
            self.current_image_path = image_path
            
            print(f"[DEBUG DiagramFormalizer.reset] ✓ Image loaded successfully")
            print(f"  - Original mode: {original_mode}")
            print(f"  - Final mode: {image.mode}")
            print(f"  - Image size: {image.size}")
            
        else:
            self.current_image = None
            self.current_image_path = image_path
            print(f"[DEBUG DiagramFormalizer.reset] ❌ NO IMAGE PROVIDED!")
            self.logger.error("CRITICAL: No image provided to DiagramFormalizer!")
    
    def execute(self, params_str):
        try:
            # 清理内存
            self._cleanup_memory()
            
            self._load_model()
            
            # 解析参数
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
            problem = params.get('problem', params.get('query', ''))
            task = params.get('task', 'solve')
            
            print(f"\n{'='*60}")
            print(f"[DEBUG DiagramFormalizer.execute] START")
            print(f"  - Problem: {problem}")
            print(f"  - Task: {task}")
            print(f"  - Has image: {self.current_image is not None}")
            print(f"  - DiagramFormalizer设备: {self.device}")
            
            # 检查图像
            if self.current_image is None:
                error_msg = "CRITICAL: No image provided! Cannot solve geometry problems without the diagram!"
                print(f"  ❌ {error_msg}")
                print(f"{'='*60}\n")
                return self._provide_enhanced_fallback(problem, task)
            
            # 确保图像是RGB格式的PIL Image
            if self.current_image.mode != 'RGB':
                print(f"  - Converting image from {self.current_image.mode} to RGB")
                if self.current_image.mode == 'RGBA':
                    background = Image.new('RGB', self.current_image.size, (255, 255, 255))
                    background.paste(self.current_image, mask=self.current_image.split()[3])
                    self.current_image = background
                else:
                    self.current_image = self.current_image.convert('RGB')
            
            print(f"  - Image size: {self.current_image.size}")
            print(f"  - Image mode: {self.current_image.mode}")
            
            # ===== 核心修改：构建不包含 <image> 的 prompt =====
            base_prompt = (
                f"Based on the provided geometric diagram, solve this problem step by step: {problem}\n\n"
                "Show all work and calculations. Identify shapes, apply theorems, and calculate the answer."
                if task == "solve"
                else f"Based on the diagram, formalize this geometry problem into mathematical notation: {problem}"
            )
            
            # 简化的文本提示，不包含 <image> 标记
            text = (
                "<|im_start|>system\n"
                "You are an expert geometry tutor. Analyze geometric diagrams and solve problems step by step.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{base_prompt}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            try:
                # 直接使用 tokenizer，不使用 tokenizer_image_token
                print(f"  - Tokenizing text (without image token)...")
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                
                # 处理图像
                with torch.no_grad():
                    # 获取模型实际所在的设备
                    model_device = next(self.model.parameters()).device
                    print(f"  - 模型参数设备: {model_device}")
                    
                    # 检查模型是否有 process_images 方法
                    if hasattr(self.model, 'process_images'):
                        print(f"  - Processing image with model.process_images")
                        image_tensor = self.model.process_images([self.current_image], self.model.config)
                    else:
                        print(f"  - WARNING: Model does not have process_images method")
                        # 如果没有，尝试其他方法或使用 fallback
                        return self._provide_enhanced_fallback(problem, task)
                    
                    # 将输入移到模型所在的设备
                    input_ids = input_ids.to(model_device)
                    image_tensor = image_tensor.to(dtype=self.model.dtype, device=model_device)
                    
                    print(f"  - Input shape: {input_ids.shape}, device: {input_ids.device}")
                    print(f"  - Image tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
                
                # 清理内存
                self._cleanup_memory()
                
                print(f"  - Generating response...")
                
                # 生成响应
                response = self._generate_with_compatibility(input_ids, image_tensor)
                
                print(f"  - Response generated (length: {len(response)})")
                print(f"  - Response preview: {response[:200]}...")
                print(f"{'='*60}\n")
                
                # 解析响应
                solution_info = self._extract_solution_info(response, problem)
                
                return {
                    "success": True,
                    "task_type": task,
                    "formalized_output": response,
                    "solution": solution_info.get("final_answer", ""),
                    "steps": solution_info.get("steps", []),
                    "raw_response": response
                }
                
            except Exception as e:
                error_str = str(e)
                print(f"  - Error during processing: {type(e).__name__}: {error_str}")
                
                if "out of memory" in error_str.lower():
                    print(f"  - GPU内存不足，使用fallback")
                    self._cleanup_memory()
                    torch.cuda.synchronize()
                elif "device-side assert triggered" in error_str:
                    print(f"  - CUDA设备断言错误，使用fallback")
                else:
                    import traceback
                    traceback.print_exc()
                
                return self._provide_enhanced_fallback(problem, task)
                
        except Exception as e:
            self.logger.error(f"DiagramFormalizer execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            print(f"\n[DEBUG] DiagramFormalizer FAILED:")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - Error message: {str(e)}")
            print(f"{'='*60}\n")
            
            return self._provide_fallback_response(params_str)
        
        finally:
            # 清理内存
            self._cleanup_memory()
    
    def _generate_with_compatibility(self, input_ids, image_tensor):
        """生成响应，处理兼容性问题"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                attempt += 1
                print(f"  - 生成尝试 {attempt}/{max_attempts}")
                
                # 清理内存
                self._cleanup_memory()
                
                # 获取词汇表大小
                vocab_size = len(self.tokenizer)
                
                # 获取安全的token ID
                original_eos_id = self.tokenizer.eos_token_id
                original_pad_id = self.tokenizer.pad_token_id
                
                # 使用安全的token ID进行生成
                safe_eos_id = min(original_eos_id, vocab_size - 1) if original_eos_id is not None else 2
                safe_pad_id = min(original_pad_id, vocab_size - 1) if original_pad_id is not None else safe_eos_id
                
                print(f"  - Vocab size: {vocab_size}")
                print(f"  - Safe tokens: eos={safe_eos_id}, pad={safe_pad_id}")
                
                with torch.inference_mode():
                    # 基本生成参数
                    gen_kwargs = {
                        "input_ids": input_ids,
                        "images": image_tensor,
                        "do_sample": False,
                        "max_new_tokens": 300 if attempt == 1 else 200,
                        "eos_token_id": safe_eos_id,
                        "pad_token_id": safe_pad_id,
                        "use_cache": False,
                        "num_beams": 1,
                    }
                    
                    output_ids = self.model.generate(**gen_kwargs)[0]
                    
                # 解码响应
                response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                if len(response) > 10:
                    return response
                    
            except Exception as e:
                print(f"  - 生成错误: {e}")
                if attempt == max_attempts:
                    raise e
                    
        return "Failed to generate response after multiple attempts."
    
    def _extract_solution_info(self, response: str, problem: str) -> Dict[str, Any]:
        """从响应中提取解决方案信息"""
        info = {
            "steps": [],
            "final_answer": ""
        }
        
        # 分割成行
        lines = response.strip().split('\n')
        
        # 提取步骤
        current_step = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是步骤标记
            if re.match(r'^\d+\.', line) or re.match(r'^Step \d+:', line, re.I):
                if current_step:
                    info["steps"].append(current_step)
                current_step = line
            else:
                if current_step:
                    current_step += " " + line
                else:
                    info["steps"].append(line)
        
        if current_step:
            info["steps"].append(current_step)
        
        # 提取最终答案
        answer_patterns = [
            r'(?:final answer|answer|therefore|thus)[\s:]+.*?(\d+\.?\d*)',
            r'[xy]\s*=\s*(\d+\.?\d*)',
            r'(?:equals?|is)\s+(\d+\.?\d*)',
            r'∠\s*\w+\s*=\s*(\d+\.?\d*)',
            r'measure.*?(?:is|equals?)\s*(\d+\.?\d*)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                info["final_answer"] = matches[-1]
                break
        
        # 如果没找到，尝试提取最后出现的合理数字
        if not info["final_answer"]:
            numbers = re.findall(r'\b\d+\.?\d*\b', response)
            reasonable_numbers = []
            for num in numbers:
                try:
                    val = float(num)
                    if 0 < val < 1000 and not re.match(r'^\d\.$', num):
                        reasonable_numbers.append(num)
                except:
                    pass
            
            if reasonable_numbers:
                info["final_answer"] = reasonable_numbers[-1]
        
        print(f"[DEBUG] Extracted from DiagramFormalizer:")
        print(f"  - Final answer: {info['final_answer']}")
        print(f"  - Number of steps: {len(info['steps'])}")
        
        return info
    
    def _provide_enhanced_fallback(self, problem: str, task: str) -> Dict[str, Any]:
        """提供增强的备用响应（基于问题文本）"""
        # 识别问题类型
        problem_lower = problem.lower()
        problem_type = "general"
        
        if any(word in problem_lower for word in ['angle', 'degree', '°', '∠']):
            problem_type = "angle"
        elif any(word in problem_lower for word in ['length', 'distance', 'side']):
            problem_type = "length"
        elif any(word in problem_lower for word in ['area', 'square']):
            problem_type = "area"
        elif any(word in problem_lower for word in ['perimeter', 'circumference']):
            problem_type = "perimeter"
        
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', problem)
        
        # 生成步骤
        steps = []
        final_answer = ""
        
        if problem_type == "angle":
            steps = [
                "Step 1: Identify all angles in the geometric figure",
                "Step 2: Note the given angle measurements",
                "Step 3: Apply angle relationships (sum of angles in triangle = 180°)",
                "Step 4: Set up an equation to solve for the unknown angle",
                "Step 5: Calculate the final answer"
            ]
            # 如果是三角形角度问题，尝试计算
            if "triangle" in problem_lower and len(numbers) >= 2:
                known_angles = [float(n) for n in numbers if float(n) < 180]
                if len(known_angles) == 2:
                    answer = 180 - sum(known_angles)
                    final_answer = str(int(answer) if answer.is_integer() else answer)
                    steps.append(f"Step 6: 180° - {known_angles[0]}° - {known_angles[1]}° = {final_answer}°")
        
        elif problem_type == "length":
            steps = [
                "Step 1: Identify the shapes and their properties",
                "Step 2: Look for congruent sides or similar triangles",
                "Step 3: Apply Pythagorean theorem if applicable",
                "Step 4: Use properties of special shapes",
                "Step 5: Calculate the unknown length"
            ]
        
        elif problem_type == "area":
            steps = [
                "Step 1: Identify the shape(s) in the figure",
                "Step 2: Determine the appropriate area formula",
                "Step 3: Find all necessary measurements",
                "Step 4: Apply the area formula",
                "Step 5: Calculate the final area"
            ]
        
        else:
            steps = [
                "Step 1: Examine the geometric figure carefully",
                "Step 2: Identify all given measurements and relationships",
                "Step 3: Apply relevant geometric theorems",
                "Step 4: Set up equations based on the constraints",
                "Step 5: Solve for the unknown variable",
                "Step 6: Verify the solution"
            ]
        
        return {
            "success": True,
            "task_type": task,
            "formalized_output": f"Analyzing {problem_type} problem: {problem}",
            "solution": final_answer,
            "steps": steps,
            "raw_response": f"Based on the geometric principles for {problem_type} problems, please examine the diagram and apply the relevant theorems.",
            "method": "enhanced_fallback",
            "problem_type": problem_type,
            "extracted_numbers": numbers,
            "warning": "Using rule-based approach due to technical limitations"
        }
    
    def _provide_fallback_response(self, params_str) -> Dict[str, Any]:
        """提供备用响应"""
        try:
            params = json.loads(params_str) if isinstance(params_str, str) else params_str
            problem = params.get('problem', params.get('query', ''))
            
            steps = [
                "Examine the geometric figure carefully",
                "Identify all given measurements and relationships",
                "Apply relevant geometric theorems",
                "Set up equations based on the constraints",
                "Solve for the unknown variable",
                "Verify the solution"
            ]
            
            return {
                "success": True,
                "task_type": "solve",
                "formalized_output": f"Due to technical limitations, please solve step by step:\n{problem}",
                "solution": "",
                "steps": steps,
                "raw_response": f"Fallback response for: {problem}",
                "fallback": True
            }
        except:
            return {
                "success": False,
                "error": "Unable to process the geometry problem",
                "error_type": "fallback_error"
            }