# vlm_gym/environments/tools/multimath.py

import os
import sys
import json
import re
import time
import base64
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from PIL import Image
import io
import torch
from pathlib import Path

from vlm_gym.environments.tools.base import ToolBase

# 设置日志
logger = logging.getLogger(__name__)

# MultiMath路径设置 - 自动检测
possible_bases = [
    "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/MultiMath",
    "/workspace/vlm_gym/environments/tools/MultiMath",
    os.path.join(os.path.dirname(__file__), "MultiMath")
]

MULTIMATH_BASE = None
for base in possible_bases:
    if os.path.exists(base):
        MULTIMATH_BASE = base
        break

if not MULTIMATH_BASE:
    logger.error("MultiMath not found in any expected location!")
    logger.error(f"Tried: {possible_bases}")
else:
    logger.info(f"Using MultiMath at: {MULTIMATH_BASE}")
    # Add MultiMath to Python path
    sys.path.insert(0, MULTIMATH_BASE)

# Import MultiMath modules
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    logger.info("Successfully imported MultiMath modules")
except ImportError as e:
    logger.error(f"Failed to import MultiMath modules: {e}")
    logger.error("Make sure you have installed MultiMath dependencies with: pip install -e .[train]")


class MultiMathTool(ToolBase):
    """MultiMath工具 - 多模态数学问题求解器"""
    
    name = "multimath"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name=self.name)
        
        # 工具描述
        self.description = "Multimodal math problem solver using MultiMath-7B"
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["solve", "analyze", "explain"],
                    "description": "Task type"
                },
                "question": {
                    "type": "string",
                    "description": "Math question or problem statement"
                },
                "image": {
                    "type": "string",
                    "description": "Base64 encoded image or image path (optional)"
                },
                "problem_type": {
                    "type": "string",
                    "enum": ["geometry", "algebra", "word_problem", "graph", "table", "general"],
                    "default": "general",
                    "description": "Type of math problem"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["answer_only", "with_steps", "detailed"],
                    "default": "answer_only",
                    "description": "Output format preference"
                }
            },
            "required": ["task", "question"]
        }
        
        self.config = config or {}
        self.debug = self.config.get("debug", False)
        
        # MultiMath配置
        self.model_path = self.config.get("model_path", "./checkpoints/multimath-7b-llava-v1.5")
        self.model_base = self.config.get("model_base", None)
        self.temperature = self.config.get("temperature", 0.0)
        self.max_new_tokens = self.config.get("max_new_tokens", 512)
        
        # 模型缓存
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_name = None
        
        # 当前问题上下文
        self.current_image = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
            
        logger.info(f"Loading MultiMath model from {self.model_path}")
        
        try:
            # 加载模型
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=self.model_base,
                model_name=get_model_name_from_path(self.model_path),
                load_8bit=False,
                load_4bit=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            self.model_name = get_model_name_from_path(self.model_path)
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load MultiMath model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def reset(self, image=None):
        """重置工具状态"""
        self.current_image = image
        logger.debug("MultiMath tool reset")
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行数学问题求解"""
        try:
            # 确保模型已加载
            self._load_model()
            
            # 解析参数
            if isinstance(params, str):
                params = json.loads(params)
            
            task = params.get("task", "solve")
            question = params.get("question", "")
            
            if not question:
                return {
                    "error": "No question provided",
                    "error_type": "InvalidInput",
                    "success": False
                }
            
            # 获取图像
            image = self._get_image(params)
            
            # 根据任务类型执行
            if task == "solve":
                return self._solve_problem(question, image, params)
            elif task == "analyze":
                return self._analyze_problem(question, image, params)
            elif task == "explain":
                return self._explain_solution(question, image, params)
            else:
                return {
                    "error": f"Unknown task: {task}",
                    "error_type": "InvalidTask",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"MultiMath execution error: {str(e)}")
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            }
    
    def _get_image(self, params: Dict[str, Any]) -> Optional[Image.Image]:
        """获取图像输入"""
        # 优先使用参数中的图像
        if "image" in params:
            image_data = params["image"]
            
            # 如果是base64编码
            if isinstance(image_data, str) and image_data.startswith("data:image"):
                try:
                    # 提取base64数据
                    base64_data = image_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    
            # 如果是文件路径
            elif isinstance(image_data, str) and os.path.exists(image_data):
                try:
                    return Image.open(image_data)
                except Exception as e:
                    logger.error(f"Failed to load image from path: {e}")
        
        # 使用当前上下文图像
        return self.current_image
    
    def _solve_problem(self, question: str, image: Optional[Image.Image], params: Dict[str, Any]) -> Dict[str, Any]:
        """求解数学问题"""
        try:
            # 准备输入
            problem_type = params.get("problem_type", "general")
            output_format = params.get("output_format", "answer_only")
            
            # 构建提示词
            prompt = self._build_prompt(question, problem_type, output_format)
            
            # 生成响应
            response = self._generate_response(prompt, image)
            
            # 解析答案
            answer, steps = self._parse_response(response, output_format)
            
            result = {
                "success": True,
                "answer": answer,
                "method": "MultiMath-7B",
                "problem_type": problem_type,
                "has_image": image is not None
            }
            
            if steps and output_format != "answer_only":
                result["steps"] = steps
                
            if output_format == "detailed":
                result["full_response"] = response
                
            return result
            
        except Exception as e:
            logger.error(f"Problem solving error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SolveError"
            }
    
    def _analyze_problem(self, question: str, image: Optional[Image.Image], params: Dict[str, Any]) -> Dict[str, Any]:
        """分析数学问题结构"""
        try:
            # 构建分析提示词
            prompt = f"""Analyze this math problem and identify:
1. Problem type (geometry, algebra, word problem, etc.)
2. Key concepts involved
3. Given information
4. What needs to be found
5. Suggested solution approach

Problem: {question}"""
            
            # 生成分析
            response = self._generate_response(prompt, image)
            
            # 解析分析结果
            analysis = self._parse_analysis(response)
            
            return {
                "success": True,
                "analysis": analysis,
                "has_image": image is not None,
                "raw_analysis": response
            }
            
        except Exception as e:
            logger.error(f"Problem analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "AnalysisError"
            }
    
    def _explain_solution(self, question: str, image: Optional[Image.Image], params: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细的解题说明"""
        try:
            # 构建解释提示词
            prompt = f"""Solve this math problem step by step with detailed explanations:

Problem: {question}

Please:
1. First understand what the problem is asking
2. Identify all given information
3. Show each step clearly
4. Explain the reasoning behind each step
5. Verify the final answer"""
            
            # 生成解释
            response = self._generate_response(prompt, image)
            
            # 解析解释
            explanation = self._parse_explanation(response)
            
            return {
                "success": True,
                "explanation": explanation,
                "has_image": image is not None
            }
            
        except Exception as e:
            logger.error(f"Solution explanation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "ExplanationError"
            }
    
    def _build_prompt(self, question: str, problem_type: str, output_format: str) -> str:
        """构建提示词"""
        if output_format == "answer_only":
            return f"Question: {question}\nAnswer:"
        elif output_format == "with_steps":
            return f"Question: {question}\nSolve step by step.\nSolution:"
        else:  # detailed
            return f"""Question: {question}

Please provide a detailed solution including:
- Problem understanding
- Step-by-step solution
- Final answer
- Verification if applicable

Solution:"""
    
    def _generate_response(self, prompt: str, image: Optional[Image.Image]) -> str:
        """生成模型响应"""
        try:
            # 准备输入
            if image is not None:
                # 处理图像
                image_tensor = process_images([image], self.image_processor, self.model.config)
                if torch.cuda.is_available():
                    image_tensor = image_tensor.to(self.model.device)
                
                # 添加图像标记
                if self.model.config.mm_use_im_start_end:
                    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                else:
                    prompt = DEFAULT_IMAGE_TOKEN + prompt
            
            # 使用对话模板
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # 生成
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor if image is not None else None,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True
                )
            
            # 解码
            outputs = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return outputs
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            raise
    
    def _parse_response(self, response: str, output_format: str) -> Tuple[str, Optional[List[str]]]:
        """解析模型响应提取答案和步骤"""
        # 提取答案
        answer = self._extract_answer(response)
        
        # 提取步骤（如果需要）
        steps = None
        if output_format != "answer_only":
            steps = self._extract_steps(response)
        
        return answer, steps
    
    def _extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        # 查找常见的答案模式
        patterns = [
            r"(?:The answer is|Answer:|Therefore,?|Thus,?|So,?)\s*[\$]?([0-9.,\-\+\s\w\(\)\/\^\=]+)[\$]?",
            r"(?:=|equals?)\s*[\$]?([0-9.,\-\+\s\w\(\)\/\^\=]+)[\$]?(?:\.|$)",
            r"[\$]([0-9.,\-\+\s\w\(\)\/\^\=]+)[\$]",
            r"^([0-9.,\-\+\s\w\(\)\/\^\=]+)$"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                # 返回最后一个匹配（通常是最终答案）
                return matches[-1].strip()
        
        # 如果没有匹配，返回整个响应的最后一行非空内容
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return lines[-1] if lines else response.strip()
    
    def _extract_steps(self, response: str) -> List[str]:
        """从响应中提取解题步骤"""
        steps = []
        
        # 按行分割
        lines = response.split('\n')
        
        current_step = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_step:
                    steps.append(' '.join(current_step))
                    current_step = []
                continue
                
            # 检查是否是新步骤的开始
            if re.match(r'^(?:Step\s*\d+|First|Second|Third|Next|Then|Finally)', line, re.IGNORECASE):
                if current_step:
                    steps.append(' '.join(current_step))
                current_step = [line]
            else:
                current_step.append(line)
        
        if current_step:
            steps.append(' '.join(current_step))
        
        # 如果没有明确的步骤，就按句子分割
        if not steps:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return steps
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """解析问题分析结果"""
        analysis = {
            "problem_type": "unknown",
            "key_concepts": [],
            "given_info": [],
            "to_find": "",
            "approach": ""
        }
        
        # 简单的关键词匹配提取
        lines = response.lower().split('\n')
        
        for i, line in enumerate(lines):
            if 'type' in line and ('geometry' in line or 'algebra' in line or 'word' in line):
                if 'geometry' in line:
                    analysis["problem_type"] = "geometry"
                elif 'algebra' in line:
                    analysis["problem_type"] = "algebra"
                elif 'word' in line:
                    analysis["problem_type"] = "word_problem"
                    
            elif 'concept' in line or 'involve' in line:
                # 提取下一行作为概念
                if i + 1 < len(lines):
                    concepts = re.findall(r'[a-z]+', lines[i + 1])
                    analysis["key_concepts"] = concepts[:5]  # 限制数量
                    
            elif 'given' in line or 'know' in line:
                if i + 1 < len(lines):
                    analysis["given_info"].append(lines[i + 1].strip())
                    
            elif 'find' in line or 'calculate' in line:
                if i + 1 < len(lines):
                    analysis["to_find"] = lines[i + 1].strip()
                    
            elif 'approach' in line or 'solution' in line or 'method' in line:
                if i + 1 < len(lines):
                    analysis["approach"] = lines[i + 1].strip()
        
        return analysis
    
    def _parse_explanation(self, response: str) -> Dict[str, Any]:
        """解析详细解释"""
        sections = {
            "understanding": "",
            "given_information": [],
            "solution_steps": [],
            "final_answer": "",
            "verification": ""
        }
        
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 检查节标题
            line_lower = line.lower()
            if 'understand' in line_lower:
                current_section = "understanding"
            elif 'given' in line_lower or 'information' in line_lower:
                current_section = "given_information"
            elif 'step' in line_lower or 'solution' in line_lower:
                current_section = "solution_steps"
            elif 'answer' in line_lower:
                current_section = "final_answer"
            elif 'verif' in line_lower or 'check' in line_lower:
                current_section = "verification"
            else:
                # 添加内容到当前节
                if current_section:
                    if current_section in ["given_information", "solution_steps"]:
                        sections[current_section].append(line)
                    else:
                        if sections[current_section]:
                            sections[current_section] += " " + line
                        else:
                            sections[current_section] = line
        
        return sections
    
    def set_current_image(self, image: Union[str, Image.Image, None]):
        """设置当前图像（由环境调用）"""
        if isinstance(image, str):
            try:
                self.current_image = Image.open(image)
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                self.current_image = None
        elif isinstance(image, Image.Image):
            self.current_image = image
        else:
            self.current_image = None