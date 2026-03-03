from typing import Dict, Any, Optional, List, Tuple
from .vlm_agent import VLMAgent
import json
import re
import os
from PIL import Image
import tempfile
import random
import hashlib


class VLMAgentWithTools(VLMAgent):
    """
    VLM Agent with tool support - Modified for proper MultiMath integration
    Supports two-stage process for geometry problems:
    - Stage 1: Generate tool call
    - Stage 2: Generate answer based on tool result
    """
    
    # 定义工具索引映射
    TOOL_INDEX_MAP = {
        0: {
            "name": "deepeyes",
            "type": "visual_enhancement",
            "brief": "zoom and rotate images for better visual analysis",
            "requires_params": True
        },
        1: {
            "name": "diagram_formalizer",
            "type": "geometry_solver",
            "brief": "analyze and solve geometry problems formally",
            "requires_params": True
        },
        2: {
            "name": "grounding_dino",
            "type": "object_detection",
            "brief": "detect and locate objects in images with text descriptions",
            "requires_params": True
        },
        3: {
            "name": "chartmoe",
            "type": "chart_analyzer",
            "brief": "analyze charts, graphs and visualizations",
            "requires_params": True
        },
        4: {
            "name": "easyocr",
            "type": "text_extraction",
            "brief": "extract and recognize text from images with multi-language support",
            "requires_params": True
        },
        5: {
            "name": "sam2",
            "type": "image_segmentation",
            "brief": "segment images based on point, box or smart medical prompts",
            "requires_params": True
        },
        6: {
            "name": "sympy_geometry",
            "type": "geometry_calculator",
            "brief": "perform precise geometric calculations and proofs",
            "requires_params": True
        },
        7: {  
            "name": "multimath_server",
            "type": "math_solver",
            "brief": "solve mathematical problems using specialized models",
            "requires_params": True
        },
        8: {
            "name": "geometry_solver",
            "type": "combined_solver",
            "brief": "solve geometry problems using DF+MM pipeline",
            "requires_params": True
        }
    }

    def __init__(self, config: Dict[str, Any]):
        """初始化带工具的VLM Agent"""
        # 分离工具相关配置和基础配置
        tool_config_keys = [
            "enable_tools", 
            "max_tool_calls", 
            "tool_selection_strategy", 
            "tool_response_mode",
            "enable_grounding_dino",
            "enable_diagram_formalizer",
            "enable_chartmoe",
            "enable_deepeyes_tools",
            "enable_easyocr",
            "enable_sam2",
            "enable_sympy_geometry",
            "enable_multimath_server",
            "enable_geometry_solver",
            "grounding_dino_config",
            "diagram_formalizer_config",
            "chartmoe_config",
            "deepeyes_config",
            "easyocr_config",
            "sam2_config",
            "sympy_geometry_config",
            "multimath_server_config",
            "geometry_solver_config",
            "enable_tool_collaboration",
            "debug" 
        ]
        
        # 提取工具配置
        tool_config = {}
        base_config = {}
        
        for key, value in config.items():
            if key in tool_config_keys:
                tool_config[key] = value
            else:
                base_config[key] = value
        
        # 使用基础配置初始化父类
        super().__init__(base_config)
        
        # 保存 model_type（从父类继承或从配置中获取）
        self.model_type = base_config.get("model_type", "HuggingFace")
        
        # 打印调试信息
        print(f"\n[VLMAgentWithTools.__init__] Initializing VLM Agent with Tools")
        print(f"  - Base config keys: {list(base_config.keys())}")
        print(f"  - Tool config keys: {list(tool_config.keys())}")
        print(f"  - Model type: {self.model_type}")
        
        # 工具使用相关配置
        self.enable_tools = tool_config.get("enable_tools", True)
        self.max_tool_calls = tool_config.get("max_tool_calls", 3)
        self.tool_selection_strategy = tool_config.get("tool_selection_strategy", "auto")
        self.tool_response_mode = tool_config.get("tool_response_mode", "auto")
        self.debug = tool_config.get("debug", False)
        
        # 各工具启用状态
        self.enable_tool_collaboration = tool_config.get("enable_tool_collaboration", False)
        self.enable_grounding_dino = tool_config.get("enable_grounding_dino", False)
        self.enable_chartmoe = tool_config.get("enable_chartmoe", False)
        self.enable_diagram_formalizer = tool_config.get("enable_diagram_formalizer", False)
        self.enable_deepeyes_tools = tool_config.get("enable_deepeyes_tools", False)
        self.enable_easyocr = tool_config.get("enable_easyocr", False)
        self.enable_sam2 = tool_config.get("enable_sam2", False)
        self.enable_sympy_geometry = tool_config.get("enable_sympy_geometry", False)
        self.enable_multimath_server = tool_config.get("enable_multimath_server", False)
        self.enable_geometry_solver = tool_config.get("enable_geometry_solver", False)
        
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - enable_multimath_server: {self.enable_multimath_server}")
        
        # 跟踪工具使用历史
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}
        self.tool_context = {}
        
        # bbox相关状态
        self.current_bbox = None
        self.bbox_history = []
        
        # 任务跟踪
        self._last_task_id = None
        
        # 工具性能跟踪
        self.tool_performance = {
            idx: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "problem_types": {}
            }
            for idx in self.TOOL_INDEX_MAP.keys()
        }
        
        # 工具使用统计
        self.tool_use_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "problems_solved": 0,
            "collaboration_chains": 0,
            "reflection_tool_calls": 0
        }
        
        # 工具实例引用（将由环境设置）
        self.grounding_dino_tool = None
        self.diagram_formalizer_tool = None
        self.chartmoe_tool = None
        self.deepeyes_tool = None
        self.easyocr_tool = None
        self.sam2_tool = None
        self.sympy_geometry_tool = None
        self.multimath_server_tool = None
        self.geometry_solver_tool = None
        
        print(f"  - Tool instances initialized to None (will be set by environment)")
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        根据观察生成action - 修改版本以正确处理MultiMath调用
        返回 (action_string, extra_info_dict)
        
        关键：Stage 2 的检查必须在最前面，避免被误判为新任务
        """
        # 确保模型已加载
        self.load_model()
        
        # ============================================
        # 最最最优先：检查是否是 Stage 2（处理工具反馈）
        # 这必须在所有其他逻辑之前执行！
        # ============================================
        if observation.get("requires_response") and observation.get("tool_feedback"):
            print(f"\n[VLMAgentWithTools.act] Stage 2 - Processing tool feedback")
            print(f"  - Tool: {observation['tool_feedback'].get('tool', 'unknown')}")
            print(f"  - Success: {observation['tool_feedback'].get('success', False)}")
            print(f"  - Answer from tool: {observation['tool_feedback'].get('answer', 'N/A')}")
            
            # 直接处理工具反馈，不执行任何其他逻辑
            response, extra_info = self._handle_tool_feedback(observation)
            
            # 确保答案格式正确
            response = self._ensure_answer_format(response, observation)
            
            print(f"  - Final response: {response[:100]}...")
            print(f"[VLMAgentWithTools.act] Stage 2 END\n")
            return response, extra_info
        
        # ============================================
        # 以下是 Stage 1 和其他正常逻辑
        # ============================================
        
        print(f"\n[VLMAgentWithTools.act] DEBUG START")
        
        if self.debug:
            print(f"  - enable_multimath_server: {self.enable_multimath_server}")
            print(f"  - multimath_server_tool is None: {self.multimath_server_tool is None}")
        
        # 检查是否是几何任务
        is_geometry = observation.get("is_geometry_task", False) or self._is_geometry_problem(observation.get("question", ""))
        if is_geometry:
            print(f"  - ✓ Geometry task detected")
        
        # 检查是否是新任务（Stage 2已经在上面处理了，所以这里不会误判）
        if observation.get("episode_start", False) or self._is_new_task(observation):
            self.current_tool_calls = 0
            self.conversation_history = []
            self.tool_call_history = {}
            self.tool_context = {}
            self.current_bbox = None
            self.bbox_history = []
            print(f"  - Starting new task, reset tool history")
        
        # 获取可用工具列表
        available_tools = self._get_available_tools(observation)
        print(f"  - Available tools: {available_tools}")
        
        # ============================================
        # 处理 Stage 1：生成工具调用
        # ============================================
        
        # 检查是否需要生成工具调用
        if self._should_generate_tool_call(observation, available_tools):
            print(f"  - Tool needed for this task")
            
            # 特别处理 MultiMath Server
            if is_geometry and self.enable_multimath_server and 7 in available_tools:
                # 获取问题的唯一标识
                question = observation.get("question", "")
                original_question = observation.get("original_question", question)
                
                # 使用原始问题作为键
                question_key = hashlib.md5(original_question.encode()).hexdigest()
                
                if question_key not in self.tool_call_history or "multimath_server" not in self.tool_call_history.get(question_key, set()):
                    print(f"  - Geometry task: using MultiMath Server for solving")
                    
                    # 如果有 system_message，说明需要生成特定格式的 tool call
                    if "system_message" in observation and "tool_call" in observation.get("system_message", ""):
                        # 直接生成 tool call，不经过父类的格式化
                        response = self._generate_raw_response(observation)
                        
                        # 清理响应中的 markdown 代码块标记
                        response = self._clean_tool_call_response(response)
                    else:
                        # 构建标准的 tool call
                        response = self._build_multimath_server_prompt(observation)
                    
                    # 记录工具调用
                    if question_key not in self.tool_call_history:
                        self.tool_call_history[question_key] = set()
                    self.tool_call_history[question_key].add("multimath_server")
                    
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "tool_call_count": self.current_tool_calls,
                        "tool_used": "multimath_server",
                        "reason": "Geometry problem solving with MultiMath",
                        "first_call": True,
                        "stage": "tool_call_generation"
                    }
                    
                    print(f"[VLMAgentWithTools.act] DEBUG END\n")
                    return response, extra_info
        
        # 如果没有 system_message 或不需要工具，生成直接答案
        print(f"  - Generating direct answer without tools")
        response, extra_info = self._generate_direct_answer(observation)
        
        print(f"[VLMAgentWithTools.act] DEBUG END\n")
        return response, extra_info
    
    def _generate_raw_response(self, observation: Dict[str, Any]) -> str:
        """
        直接生成原始响应，绕过父类的格式化
        这是修复的核心 - 直接调用底层方法，避免 answer_question 包装
        """
        # 确保模型已加载
        self.load_model()
        
        # 获取关键信息
        image_path = observation.get("image_path", "")
        question = observation.get("question", "")
        system_message = observation.get("system_message", "")
        output_format = observation.get("output_format_instruction", "")
        
        # 构建完整的提示
        if system_message:
            # 如果有 system_message，它包含了完整的指令
            full_prompt = system_message
            if output_format:
                full_prompt += f"\n\n{output_format}"
        else:
            # 否则使用标准问题
            full_prompt = question
        
        # 构建一个新的 observation，只包含必要的字段
        modified_obs = {
            "question": full_prompt,
            "image_path": image_path if image_path else None
        }
        
        # 复制其他可能需要的字段
        for key in ["task_id", "episode_start", "attempt", "choices"]:
            if key in observation:
                modified_obs[key] = observation[key]
        
        try:
            # 关键修改：直接调用父类的底层方法，绕过 act() 和 _parse_response()
            # 步骤1：准备输入
            inputs = super()._prepare_input(modified_obs)
            
            # 步骤2：生成原始响应
            raw_response = super()._generate_response(inputs)
            
            # 返回原始响应，没有任何包装
            return raw_response
            
        except Exception as e:
            print(f"[ERROR] Failed to generate raw response: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，返回一个默认的 tool call
            return self._build_multimath_server_prompt(observation)
    
    def _clean_tool_call_response(self, response: str) -> str:
        """
        清理生成的 tool call 响应
        移除 markdown 代码块标记，提取 JSON 内容
        """
        # 如果响应已经是正确的 <tool_call> 格式，直接返回
        if response.startswith("<tool_call>") and response.endswith("</tool_call>"):
            return response
        
        # 清理 markdown 代码块
        if "```json" in response or "```" in response:
            # 提取 JSON 内容
            import re
            # 更灵活的模式匹配
            json_pattern = r'```(?:json)?\s*(.*?)\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                # 包装成 tool_call 格式
                return f"<tool_call>{json_content}</tool_call>"
        
        # 如果响应包含 JSON 对象
        if "{" in response and "}" in response:
            # 尝试提取 JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = response[start_idx:end_idx+1]
                # 验证是否是 tool call JSON
                if '"tool"' in json_content and '"parameters"' in json_content:
                    return f"<tool_call>{json_content}</tool_call>"
        
        # 如果无法清理，返回原始响应
        return response
    
    def _should_generate_tool_call(self, observation: Dict[str, Any], available_tools: List[int]) -> bool:
        """
        判断是否应该生成工具调用
        """
        # 如果没有可用工具，不生成
        if not available_tools:
            return False
        
        # 如果已经有工具反馈，不应该再生成新的工具调用
        if observation.get("tool_feedback"):
            return False
        
        # 如果明确要求不使用工具
        if observation.get("force_tool_use") == False:
            return False
        
        # 检查各种强制使用工具的标志
        if observation.get("force_tool_use") == True:
            return True
        
        if observation.get("tool_use_required"):
            return True
        
        # 修正：统一使用 multimath_server_enabled
        if observation.get("multimath_server_enabled") or observation.get("multimath_enabled"):
            if 7 in available_tools:  # MultiMath Server
                return True
        
        # 几何问题且有相关工具
        question = observation.get("question", "")
        if self._is_geometry_problem(question):
            if 7 in available_tools or 6 in available_tools or 8 in available_tools:
                return True
        
        return False
    
    def _get_available_tools(self, observation: Dict[str, Any]) -> List[int]:
        """获取当前可用的工具索引"""
        available = []
        
        # 基于实际启用的工具和工具实例是否存在
        if self.enable_deepeyes_tools and self.deepeyes_tool is not None:
            available.append(0)
        if self.enable_diagram_formalizer and self.diagram_formalizer_tool is not None:
            available.append(1)
        if self.enable_grounding_dino and self.grounding_dino_tool is not None:
            available.append(2)
        if self.enable_chartmoe and self.chartmoe_tool is not None:
            available.append(3)
        if self.enable_easyocr and self.easyocr_tool is not None:
            available.append(4)
        if self.enable_sam2 and self.sam2_tool is not None:
            available.append(5)
        if self.enable_sympy_geometry and self.sympy_geometry_tool is not None:
            available.append(6)
        if self.enable_multimath_server and self.multimath_server_tool is not None:
            available.append(7)
        if self.enable_geometry_solver and self.geometry_solver_tool is not None:
            available.append(8)
        
        # 对于启用了 multimath 的情况（统一字段名）
        if (observation.get("multimath_server_enabled") or observation.get("multimath_enabled")) and self.enable_multimath_server:
            if 7 not in available:
                available.append(7)
        
        if self.debug:
            print(f"[DEBUG _get_available_tools] Actually available tools: {available}")
            tool_names = [self.TOOL_INDEX_MAP[idx]["name"] for idx in available]
            print(f"[DEBUG _get_available_tools] Tool names: {tool_names}")
        
        return available
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理工具反馈 - 专门优化 MultiMath 反馈处理"""
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        print(f"\n[_handle_tool_feedback] Processing {tool_name} feedback")
        print(f"  - Success: {tool_feedback.get('success', False)}")
        print(f"  - Answer: {tool_feedback.get('answer', 'N/A')}")
        
        # 特殊处理 MultiMath Server 反馈
        if tool_name == "multimath_server":
            return self._handle_multimath_server_feedback_special(observation)
        
        # 其他工具的处理保持不变
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        
        # 如果有 system_message（Stage 2），使用原始生成
        if "system_message" in observation:
            response = self._generate_raw_response(enhanced_observation)
        else:
            # 使用父类方法但设置 use_structured_output 避免包装
            enhanced_observation["use_structured_output"] = True
            response, _ = super().act(enhanced_observation)
        
        extra_info = {
            "action_type": "tool_feedback_response",
            "tool_used": tool_name,
            "stage": "final_answer_with_tool"
        }
        
        return response, extra_info
    
    def _handle_multimath_server_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理 MultiMath Server 反馈 - 适配两阶段流程"""
        tool_feedback = observation.get("tool_feedback", {})
        success = tool_feedback.get("success", False)
        answer = tool_feedback.get("answer", "")
        method = tool_feedback.get("method", "")
        steps = tool_feedback.get("steps", [])
        
        print(f"  - MultiMath Server success: {success}")
        print(f"  - Answer: {answer}")
        print(f"  - Method: {method}")
        
        # 如果 observation 中有 system_message（Stage 2），直接使用它
        if "system_message" in observation:
            print(f"  - Using system_message for Stage 2")
            # 使用提供的 system_message 生成答案
            response = self._generate_raw_response(observation)
        else:
            # 否则构建默认的提示
            print(f"  - Building default prompt for Stage 2")
            enhanced_observation = observation.copy()
            enhanced_observation.pop("tool_feedback", None)
            enhanced_observation.pop("requires_response", None)
            
            # 构建提示
            question = observation.get("original_question", observation.get("question", ""))
            if success and answer:
                # 格式化步骤
                steps_text = ""
                if steps:
                    for i, step in enumerate(steps[:3], 1):
                        steps_text += f"Step {i}: {step}\n"
                
                prompt = f"""Based on MultiMath Server calculation:

Question: {question}
Method: {method}
Answer: {answer}

{steps_text if steps_text else ''}

Please provide the final answer in the format: <answer>{answer}</answer>"""
            else:
                prompt = f"""MultiMath Server could not solve this problem.

Question: {question}

Please analyze the problem and provide your best answer.
Output your answer in <answer> tags."""
            
            enhanced_observation["question"] = prompt
            enhanced_observation["use_structured_output"] = True  # 避免 answer_question 包装
            response, _ = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = {
            "action_type": "multimath_analysis",
            "tool_used": "multimath_server",
            "success": success,
            "method": method,
            "stage": "final_answer_with_multimath"
        }
        
        return response, extra_info
    
    def _build_multimath_server_prompt(self, observation: Dict[str, Any]) -> str:
        """构建 MultiMath Server 工具调用提示"""
        question = observation.get("question", "")
        original_question = observation.get("original_question", question)
        image_path = observation.get("image_path", "")
        
        # 清理问题
        clean_question = original_question
        
        # 构建工具调用
        tool_call = {
            "tool": "multimath_server",
            "parameters": {
                "task": "solve",
                "question": clean_question,
                "problem_type": "geometry",
                "output_format": "with_steps"
            }
        }
        
        # 如果有图像路径，添加到参数中
        if image_path:
            tool_call["parameters"]["image"] = image_path
        
        print(f"  - MultiMath Server solving: '{clean_question[:100]}...'")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """生成直接答案 - 正确处理 system_message"""
        print(f"  - Generating direct answer")
        
        # 如果有 system_message，使用原始生成
        if "system_message" in observation:
            response = self._generate_raw_response(observation)
        else:
            # 设置 use_structured_output 避免包装
            modified_obs = observation.copy()
            modified_obs["use_structured_output"] = True
            response, base_info = super().act(modified_obs)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = {
            "action_type": "direct_answer",
            "tool_call_count": 0,
            "tools_disabled": not self.enable_tools,
            "stage": "direct_answer"
        }
        
        return response, extra_info
    
    def _ensure_answer_format(self, response: str, observation: Dict[str, Any]) -> str:
        """确保响应包含正确的格式"""
        # 检查是否已有正确格式
        if "<answer>" in response and "</answer>" in response:
            return response
        
        # 如果是工具调用，不需要修改
        if "<tool_call>" in response:
            return response
        
        # 提取答案
        answer = self._extract_answer_from_content(response, observation)
        
        # 构建完整响应
        return f"<answer>{answer}</answer>"
    
    def _extract_answer_from_content(self, response: str, observation: Dict[str, Any]) -> str:
        """从响应内容中提取答案"""
        # 如果已经有 answer 标签，提取内容
        answer_match = re.search(r'<answer>\s*([^<]+?)\s*</answer>', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 尝试提取数字答案
        question = observation.get("question", "").lower()
        if any(kw in question for kw in ["how many", "count", "number", "total", "sum", "difference", "find"]):
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
            if numbers:
                return numbers[-1]
        
        # 提取 Yes/No 答案
        if any(kw in question for kw in ["is", "are", "does", "do", "can", "will"]):
            response_lower = response.lower()
            if "yes" in response_lower:
                return "Yes"
            elif "no" in response_lower:
                return "No"
        
        # 默认返回整个响应的精简版本
        return response.strip()[:100]
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """判断是否是新任务 - 修正版本"""
        # 如果有 tool_feedback，肯定不是新任务
        if observation.get("tool_feedback"):
            return False
        
        # 如果有 requires_response，也不是新任务
        if observation.get("requires_response"):
            return False
        
        current_task_id = observation.get("task_id")
        if hasattr(self, "_last_task_id") and self._last_task_id != current_task_id:
            self._last_task_id = current_task_id
            return True
        
        self._last_task_id = current_task_id
        
        if not self.conversation_history:
            return True
        
        current_question = observation.get("question")
        for hist in reversed(self.conversation_history):
            if hist["role"] == "observation" and isinstance(hist["content"], dict):
                last_question = hist["content"].get("question")
                return current_question != last_question
        
        return True
    
    def _is_geometry_problem(self, question: str) -> bool:
        """判断是否是几何问题"""
        geometry_keywords = [
            'angle', 'degree', '°', 'triangle', 'circle', 'square', 'rectangle',
            'perimeter', 'area', 'volume', 'parallel', 'perpendicular',
            'congruent', 'similar', 'polygon', 'radius', 'diameter',
            'hypotenuse', 'pythagorean', 'inscribed', 'circumference',
            'find', 'calculate', 'measure'
        ]
        return any(keyword in question.lower() for keyword in geometry_keywords)
    
    def reset(self):
        """重置 agent 状态"""
        super().reset()
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}
        self.tool_context = {}
        self.current_bbox = None
        self.bbox_history = []
        self._last_task_id = None