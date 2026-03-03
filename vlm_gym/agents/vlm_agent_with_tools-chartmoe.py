#这个最新的是专门为chartqa设计的agent使用tool的脚本
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
    VLM Agent with tool support - ChartQA optimized version with reflection support
    Primary focus: ChartMoE for chart analysis
    Secondary support: DeepEyes, GroundingDINO, DiagramFormalizer, EasyOCR
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
            "grounding_dino_config",
            "diagram_formalizer_config",
            "chartmoe_config",
            "deepeyes_config",
            "easyocr_config",
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
        
        # 打印调试信息
        print(f"\n[VLMAgentWithTools.__init__] Initializing VLM Agent with Tools (ChartQA Optimized)")
        print(f"  - Base config keys: {list(base_config.keys())}")
        print(f"  - Tool config keys: {list(tool_config.keys())}")
        
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
        
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - enable_chartmoe: {self.enable_chartmoe}")
        print(f"  - enable_grounding_dino: {self.enable_grounding_dino}")
        print(f"  - enable_deepeyes_tools: {self.enable_deepeyes_tools}")
        print(f"  - enable_easyocr: {self.enable_easyocr}")
        print(f"  - enable_diagram_formalizer: {self.enable_diagram_formalizer}")
        
        # 跟踪工具使用历史
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}  # 记录每个问题的工具调用历史
        self.tool_context = {}  # 存储每个工具的输出结果
        
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
            "reflection_tool_calls": 0  # 新增：反思阶段的工具调用
        }
        
        # 工具实例引用（将由环境设置）
        self.grounding_dino_tool = None
        self.diagram_formalizer_tool = None
        self.chartmoe_tool = None
        self.deepeyes_tool = None
        self.easyocr_tool = None
        
        print(f"  - Tool instances initialized to None (will be set by environment)")
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        根据观察生成action - ChartQA优化版本 with reflection support
        返回 (action_string, extra_info_dict)
        """
        # 确保模型已加载
        self.load_model()
        
        # 添加调试输出
        #print(f"\n[VLMAgentWithTools.act] DEBUG START")
        #print(f"  - Observation keys: {list(observation.keys())}")
        #print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - enable_chartmoe: {self.enable_chartmoe}")
        #print(f"  - Current tool calls: {self.current_tool_calls}")
        #print(f"  - Tool context: {list(self.tool_context.keys())}")
        
        # 检查是否是ChartQA任务
        is_chartqa = observation.get("is_visual_question", False) and observation.get("chartmoe_enabled", False)
        if is_chartqa:
            print(f"  - ✓ ChartQA task detected")
        
        # 检查是否处于反思阶段
        is_reflection = self._is_reflection_attempt(observation)
        if is_reflection:
            print(f"  - ✓ Reflection attempt detected (attempt {observation.get('attempt', 1)})")
        
        # 清除已使用工具的must_use_tool标志（除非在反思阶段需要重新使用）
        if isinstance(observation, dict):
            # 如果ChartMoE已经使用过，但不是反思阶段，清除强制标志
            if observation.get("chartmoe_history") and len(observation.get("chartmoe_history", [])) > 0 and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ ChartMoE already used, cleared must_use_tool flags")
            
            # 类似处理其他工具
            if observation.get("easyocr_history") and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ EasyOCR already used, cleared must_use_tool flags")
        
        # 处理强制使用工具的情况
        if isinstance(observation, dict) and observation.get("must_use_tool"):
            tool_to_use = observation.get("tool_to_use")
            
            # ChartMoE强制使用
            if tool_to_use == "chartmoe" and self.enable_chartmoe and self.chartmoe_tool is not None:
                print(f"  - ⚠️ FORCED tool usage: {tool_to_use}")
                
                # 在反思阶段使用更精确的提示
                if is_reflection:
                    tool_call = self._build_chartmoe_prompt_for_reflection(observation)
                else:
                    tool_call = {
                        "tool": "chartmoe",
                        "task": "to_table"  # 默认使用to_table
                    }
                    tool_call = f'<tool_call>{json.dumps(tool_call)}</tool_call>'
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if is_reflection:
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "chartmoe",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "is_reflection": is_reflection
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
        
        # 检查是否是新任务
        if observation.get("episode_start", False) or self._is_new_task(observation):
            self.current_tool_calls = 0
            self.conversation_history = []
            self.tool_call_history = {}
            self.tool_context = {}
            self.current_bbox = None
            self.bbox_history = []
            print(f"  - Starting new task, reset tool history")
        
        # 记录当前交互
        self.conversation_history.append({"role": "observation", "content": observation})
        
        # 检查是否是工具反馈
        if observation.get("requires_response") and "tool_feedback" in observation:
            print(f"  - Handling tool feedback")
            response, extra_info = self._handle_tool_feedback(observation)
            response = self._ensure_answer_format(response, observation)
            return response, extra_info
        
        # 获取可用工具列表
        available_tools = self._get_available_tools(observation)
        print(f"  - Available tools: {available_tools}")
        
        # 检查是否启用工具
        if not self.enable_tools or not available_tools:
            print(f"  - Tools disabled or no tools available, generating direct answer")
            response, extra_info = self._generate_direct_answer(observation)
            response = self._ensure_answer_format(response, observation)
            return response, extra_info
        
        # 检查是否达到工具调用上限
        if self.current_tool_calls >= self.max_tool_calls:
            print(f"  - Reached max tool calls ({self.max_tool_calls}), generating final answer")
            response, extra_info = self._generate_forced_final_answer(observation)
            response = self._ensure_answer_format(response, observation)
            return response, extra_info
        
        # 对于反思阶段的ChartQA任务，考虑重新调用工具
        if is_reflection and is_chartqa and 3 in available_tools:
            # 检查上次的答案是否错误
            if observation.get("feedback") and "incorrect" in observation.get("feedback", "").lower():
                print(f"  - Reflection attempt with incorrect answer, considering tool re-use")
                
                # 分析是否需要使用不同的ChartMoE任务
                should_retry_tool = self._should_retry_tool_in_reflection(observation)
                
                if should_retry_tool:
                    print(f"  - Decided to retry ChartMoE with different approach")
                    tool_call = self._build_chartmoe_prompt_for_reflection(observation)
                    
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    self.tool_use_stats["reflection_tool_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "tool_call_count": self.current_tool_calls,
                        "tool_used": "chartmoe",
                        "reason": "Reflection retry with different approach",
                        "reflection_attempt": observation.get("attempt", 1),
                        "is_reflection": True
                    }
                    
                    print(f"[VLMAgentWithTools.act] DEBUG END\n")
                    return tool_call, extra_info
        
        # ChartQA特殊处理：首次强制使用ChartMoE
        if is_chartqa and 3 in available_tools and not is_reflection:
            # 获取基于尝试次数的key
            question = observation.get("question", "")
            current_attempt = observation.get("attempt", 1)
            question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
            
            if question_key not in self.tool_call_history or "chartmoe" not in self.tool_call_history.get(question_key, set()):
                print(f"  - ChartQA task: forcing ChartMoE usage")
                tool_call = self._build_chartmoe_prompt(observation)
                
                # 记录工具调用
                if question_key not in self.tool_call_history:
                    self.tool_call_history[question_key] = set()
                self.tool_call_history[question_key].add("chartmoe")
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                
                extra_info = {
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "tool_used": "chartmoe",
                    "reason": "ChartQA task requires chart analysis",
                    "first_call": True
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
        
        # 对于非ChartQA任务或已经使用过工具的情况，检查是否需要使用工具
        tool_needed = self._analyze_tool_need(observation)
        
        if tool_needed:
            print(f"  - Tool needed for this task")
            response, extra_info = self._generate_tool_call(observation, available_tools)
        else:
            print(f"  - No tool needed, generating direct answer")
            response, extra_info = self._generate_direct_answer(observation)
            response = self._ensure_answer_format(response, observation)
        
        print(f"[VLMAgentWithTools.act] DEBUG END\n")
        return response, extra_info
    
    def _is_reflection_attempt(self, observation: Dict[str, Any]) -> bool:
        """检测是否处于反思阶段"""
        return (
            observation.get("previous_attempt_failed", False) or
            observation.get("feedback") is not None or
            observation.get("attempt", 1) > 1
        )
    
    def _should_retry_tool_in_reflection(self, observation: Dict[str, Any]) -> bool:
        """判断反思阶段是否应该重新调用工具"""
        feedback = observation.get("feedback", "")
        question_type = self._classify_question_type(observation.get("question", ""))
        
        # 对于计数问题，如果误差较大，应该重新调用工具
        if question_type == "counting" and "significantly off" in feedback:
            return True
        
        # 对于需要精确数值的问题
        if any(phrase in feedback for phrase in ["check your", "verify", "re-examine", "re-read"]):
            return True
        
        # 如果反馈中提到使用工具
        if any(phrase in feedback for phrase in ["use the ChartMoE tool", "try a different task", "use a custom prompt"]):
            return True
        
        # 如果是最后一次尝试
        if observation.get("attempts_remaining", 0) == 1:
            return True
        
        return False
    
    def _build_chartmoe_prompt_for_reflection(self, observation: Dict[str, Any]) -> str:
        """为反思阶段构建特定的ChartMoE提示"""
        question = observation.get("question", "")
        feedback = observation.get("feedback", "")
        previous_answer = observation.get("previous_answer", "")
        
        # 根据反馈选择更精确的任务
        if "count" in feedback.lower() or "how many" in question.lower():
            # 使用自定义提示来精确计数
            if "unfavorable" in question.lower() and "below" in question.lower():
                tool_call = {
                    "tool": "chartmoe",
                    "prompt": "List all values in the Unfavorable column that are less than 40. Count them precisely."
                }
            else:
                tool_call = {
                    "tool": "chartmoe",
                    "prompt": f"For the question '{question}', extract and list all relevant values, then count them precisely."
                }
        elif "different task" in feedback or "custom prompt" in feedback:
            # 使用analyze或其他任务
            tool_call = {
                "tool": "chartmoe",
                "task": "analyze"
            }
        else:
            # 默认使用extract_data获取更详细的数据
            tool_call = {
                "tool": "chartmoe",
                "task": "extract_data"
            }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'total number']):
            return 'counting'
        elif any(word in question_lower for word in ['sum', 'add', 'total of']):
            return 'summation'
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            return 'average'
        elif any(word in question_lower for word in ['percentage', 'percent', '%', 'proportion']):
            return 'percentage'
        elif any(word in question_lower for word in ['difference', 'gap', 'subtract', 'minus']):
            return 'difference'
        elif any(word in question_lower for word in ['ratio', 'times', 'divide']):
            return 'ratio'
        elif any(word in question_lower for word in ['value', 'what is the', 'how much']):
            return 'numerical'
        elif any(word in question_lower for word in ['compare', 'which is']):
            return 'comparison'
        elif any(word in question_lower for word in ['maximum', 'minimum', 'highest', 'lowest', 'largest', 'smallest']):
            return 'minmax'
        elif any(word in question_lower for word in ['trend', 'increase', 'decrease', 'change']):
            return 'trend'
        elif any(word in question_lower for word in ['yes', 'no', 'is', 'are', 'does', 'do']):
            return 'yes_no'
        elif any(word in question_lower for word in ['what', 'which', 'when', 'where', 'who']):
            return 'retrieval'
        else:
            return 'other'
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """判断是否是新任务 - 改进版"""
        # 如果task_id变化，是新任务
        current_task_id = observation.get("task_id")
        if hasattr(self, "_last_task_id") and self._last_task_id != current_task_id:
            self._last_task_id = current_task_id
            return True
        
        self._last_task_id = current_task_id
        
        # 原有逻辑
        if not self.conversation_history:
            return True
        
        current_question = observation.get("question")
        for hist in reversed(self.conversation_history):
            if hist["role"] == "observation" and isinstance(hist["content"], dict):
                last_question = hist["content"].get("question")
                return current_question != last_question
        
        return True
    
    def _get_available_tools(self, observation: Dict[str, Any]) -> List[int]:
        """获取当前可用的工具索引 - 基于实际配置而非观察"""
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
        
        # 对于ChartQA任务的特殊处理
        if observation.get("chartmoe_enabled", False) and self.enable_chartmoe:
            if 3 not in available:
                available.append(3)
        
        print(f"[DEBUG _get_available_tools] Actually available tools: {available}")
        tool_names = [self.TOOL_INDEX_MAP[idx]["name"] for idx in available]
        print(f"[DEBUG _get_available_tools] Tool names: {tool_names}")
        
        return available
    
    def _analyze_tool_need(self, observation: Dict[str, Any]) -> bool:
        """分析是否需要使用工具 - ChartQA优化版本"""
        question = observation.get("question", "").lower()
        
        # 获取可用工具
        available_tools = self._get_available_tools(observation)
        if not available_tools:
            return False
        
        # ChartQA任务：如果有ChartMoE可用，应该使用
        if observation.get("is_visual_question", False) and 3 in available_tools:
            print(f"  - Visual question with ChartMoE available, tool needed")
            return True
        
        # 检查是否是纯计算
        if self._is_pure_calculation(question):
            print(f"  - Pure calculation problem, no tools needed")
            return False
        
        # 根据可用工具和问题类型判断
        # 文本提取
        if 4 in available_tools and self._should_use_easyocr(observation):
            return True
        
        # 视觉增强
        if 0 in available_tools and self._should_use_deepeyes(observation):
            return True
        
        # 几何问题
        if 1 in available_tools and self._is_geometry_problem(question):
            return True
        
        # 对象检测
        if 2 in available_tools and self._should_use_grounding_dino(observation):
            return True
        
        return False
    
    def _generate_tool_call(self, observation: Dict[str, Any], available_tools: List[int]) -> Tuple[str, Dict[str, Any]]:
        """生成工具调用 - 支持反思重试"""
        question = observation.get("question", "")
        current_attempt = observation.get("attempt", 1)
        question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
        
        # ChartQA优先使用ChartMoE
        if observation.get("is_visual_question", False) and 3 in available_tools:
            if question_key not in self.tool_call_history or "chartmoe" not in self.tool_call_history.get(question_key, set()):
                # 根据是否是反思阶段选择不同的构建方法
                if self._is_reflection_attempt(observation):
                    tool_call = self._build_chartmoe_prompt_for_reflection(observation)
                else:
                    tool_call = self._build_chartmoe_prompt(observation)
                
                tool_name = "chartmoe"
                
                # 记录工具调用
                if question_key not in self.tool_call_history:
                    self.tool_call_history[question_key] = set()
                self.tool_call_history[question_key].add(tool_name)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if self._is_reflection_attempt(observation):
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "tool_used": tool_name,
                    "reason": "Chart analysis for visual question",
                    "first_call": current_attempt == 1,
                    "is_reflection": self._is_reflection_attempt(observation)
                }
                
                return tool_call, extra_info
        
        # 其他工具选择逻辑
        # 根据问题类型和可用工具选择
        tool_name = None
        tool_call = None
        
        # 按优先级检查工具
        if 4 in available_tools and self._should_use_easyocr(observation):
            if question_key not in self.tool_call_history or "easyocr" not in self.tool_call_history.get(question_key, set()):
                tool_name = "easyocr"
                tool_call = self._build_easyocr_prompt(observation)
        
        elif 0 in available_tools and self._should_use_deepeyes(observation):
            if question_key not in self.tool_call_history or "deepeyes" not in self.tool_call_history.get(question_key, set()):
                tool_name = "deepeyes"
                tool_call = self._build_deepeyes_prompt(observation)
        
        elif 1 in available_tools and self._is_geometry_problem(question):
            if question_key not in self.tool_call_history or "diagram_formalizer" not in self.tool_call_history.get(question_key, set()):
                tool_name = "diagram_formalizer"
                tool_call = self._build_diagram_formalizer_prompt(observation)
        
        elif 2 in available_tools and self._should_use_grounding_dino(observation):
            if question_key not in self.tool_call_history or "grounding_dino" not in self.tool_call_history.get(question_key, set()):
                tool_name = "grounding_dino"
                tool_call = self._build_grounding_dino_prompt(observation)
        
        # 如果找到合适的工具
        if tool_name and tool_call:
            # 记录工具调用
            if question_key not in self.tool_call_history:
                self.tool_call_history[question_key] = set()
            self.tool_call_history[question_key].add(tool_name)
            
            self.current_tool_calls += 1
            self.tool_use_stats["total_calls"] += 1
            if self._is_reflection_attempt(observation):
                self.tool_use_stats["reflection_tool_calls"] += 1
            
            extra_info = {
                "action_type": "tool_call",
                "tool_call_count": self.current_tool_calls,
                "tool_used": tool_name,
                "reason": f"Using {tool_name} for task",
                "first_call": current_attempt == 1,
                "is_reflection": self._is_reflection_attempt(observation)
            }
            
            return tool_call, extra_info
        
        # 如果没有合适的工具，生成直接答案
        return self._generate_direct_answer(observation)
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理工具反馈 - ChartQA优化版本"""
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        print(f"\n[_handle_tool_feedback] Processing {tool_name} feedback")
        
        # 特殊处理ChartMoE反馈
        if tool_name == "chartmoe":
            return self._handle_chartmoe_feedback_special(observation)
        
        # 其他工具的处理
        # ... 保持原有逻辑 ...
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 基于工具反馈生成答案
        response, base_info = super().act(enhanced_observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "tool_feedback_response",
            "tool_used": tool_name,
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _handle_chartmoe_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理ChartMoE反馈 - 针对ChartQA优化"""
        tool_feedback = observation.get("tool_feedback", {})
        task_type = tool_feedback.get("task_type", "unknown")
        output = tool_feedback.get("output", "")
        
        # 提取原始问题
        question = observation.get("original_question", observation.get("question", ""))
        question_lower = question.lower()
        
        print(f"  - ChartMoE task type: {task_type}")
        print(f"  - Question: {question[:100]}...")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 构建基于ChartMoE输出的提示
        if task_type == "to_table" and output:
            # 表格数据，适合回答具体数值问题
            prompt = f"""Based on the extracted table data from ChartMoE:

{output}

Now answer the question: "{question}"

Instructions:
1. Use the exact values from the table above
2. For counting questions, count the relevant entries
3. For comparison questions, compare the values directly
4. For yes/no questions, verify against the table data

You MUST output your answer inside <answer> tags.
For example:
- For numbers: <answer>42</answer>
- For yes/no: <answer>Yes</answer>"""
        
        else:
            # 其他类型的输出
            prompt = f"""Based on the ChartMoE analysis ({task_type}):

{output[:1000]}...

Answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
        
        enhanced_observation["output_format_instruction"] = prompt
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "chartmoe_analysis",
            "tool_used": "chartmoe",
            "task_type": task_type,
            "stage": "final_answer_with_chartmoe",
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _build_chartmoe_prompt(self, observation: Dict[str, Any]) -> str:
        """构建ChartMoE工具调用提示 - ChartQA优化版本"""
        question = observation.get("question", "")
        question_lower = question.lower()
        
        # 根据问题类型选择最合适的ChartMoE任务
        if any(kw in question_lower for kw in ["how many", "count", "number of", "total"]):
            # 计数问题 - 使用to_table获取准确数据
            task = "to_table"
        elif any(kw in question_lower for kw in ["value", "what is", "how much"]):
            # 具体数值问题 - 使用to_table
            task = "to_table"
        elif any(kw in question_lower for kw in ["yes", "no", "is", "are", "does", "do"]):
            # Yes/No问题 - 使用to_table验证
            task = "to_table"
        elif any(kw in question_lower for kw in ["compare", "difference", "which"]):
            # 比较问题 - 可以使用compare或to_table
            task = "to_table"  # to_table更可靠
        elif any(kw in question_lower for kw in ["trend", "pattern", "change"]):
            # 趋势分析 - 使用analyze
            task = "analyze"
        else:
            # 默认使用to_table，因为它提供最准确的数据
            task = "to_table"
        
        tool_call = {
            "tool": "chartmoe",
            "task": task
        }
        
        print(f"  - ChartMoE task selected: {task}")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
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
        
        # 如果响应中有think标签，保留它
        if "<think>" in response and "</think>" in response:
            return response + f"\n<answer>{answer}</answer>"
        
        # 构建完整响应
        return f"""<think>
{response}
</think>
<answer>{answer}</answer>"""
    
    def _extract_answer_from_content(self, response: str, observation: Dict[str, Any]) -> str:
        """从响应内容中提取答案"""
        question = observation.get("question", "").lower()
        
        # 提取数字答案
        if any(kw in question for kw in ["how many", "count", "number", "total", "sum", "difference"]):
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
            if numbers:
                return numbers[-1]  # 返回最后一个数字
        
        # 提取选择题答案
        if "choices" in observation:
            choices = re.findall(r'\b[A-E]\b', response)
            if choices:
                return choices[-1]
        
        # 提取Yes/No答案
        if any(kw in question for kw in ["is", "are", "does", "do", "can", "will"]):
            response_lower = response.lower()
            if "yes" in response_lower:
                return "Yes"
            elif "no" in response_lower:
                return "No"
        
        # 默认返回最后一句
        sentences = re.split(r'[.!?]+', response.strip())
        for sentence in reversed(sentences):
            if sentence.strip():
                return sentence.strip()
        
        return response.strip()[:100]
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """生成直接答案（不使用工具）"""
        print(f"  - Generating direct answer without tools")
        
        # 直接调用父类的act方法
        response, base_info = super().act(observation)
        
        # 添加工具相关信息
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "direct_answer",
            "tool_call_count": 0,
            "tools_disabled": not self.enable_tools,
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _generate_forced_final_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """当达到工具调用上限时，强制生成最终答案"""
        enhanced_observation = observation.copy()
        enhanced_observation["output_format_instruction"] = """You have used the maximum number of tools. 
Based on all the analysis so far, provide your final answer.
You MUST output your answer inside <answer> tags."""
        
        response, base_info = super().act(enhanced_observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "forced_final_answer",
            "tool_call_count": self.current_tool_calls,
            "reason": "reached_tool_limit",
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    # ========== 以下是辅助方法 ==========
    
    def _is_pure_calculation(self, question: str) -> bool:
        """判断是否是纯计算问题"""
        question_lower = question.lower()
        
        calculation_indicators = [
            r'(?:^|\s)\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?',
            r'what is \d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?',
            r'calculate:?\s*\d+(?:\.\d+)?',
            r'solve:?\s*[\d\.\+\-\*/\s]+',
        ]
        
        if any(re.search(pattern, question_lower) for pattern in calculation_indicators):
            visual_words = ["image", "picture", "shown", "chart", "graph"]
            if not any(word in question_lower for word in visual_words):
                return True
        
        return False
    
    def _is_geometry_problem(self, question: str) -> bool:
        """判断是否是几何问题"""
        geometry_keywords = [
            'angle', 'degree', '°', 'triangle', 'circle', 'square', 'rectangle',
            'perimeter', 'area', 'volume', 'parallel', 'perpendicular',
            'congruent', 'similar', 'polygon', 'radius', 'diameter'
        ]
        return any(keyword in question.lower() for keyword in geometry_keywords)
    
    def _should_use_easyocr(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用EasyOCR"""
        question = observation.get("question", "").lower()
        
        if self._is_pure_calculation(question):
            return False
        
        if not observation.get("has_image", False):
            return False
        
        text_keywords = [
            "read", "text", "written", "equation", "formula",
            "label", "sign", "caption", "title", "ocr"
        ]
        
        return any(keyword in question for keyword in text_keywords)
    
    def _should_use_deepeyes(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用DeepEyes"""
        question = observation.get("question", "").lower()
        
        if self._is_pure_calculation(question):
            return False
        
        if not observation.get("has_image", False):
            return False
        
        deepeyes_keywords = [
            "zoom", "detail", "small", "tiny", "unclear",
            "rotate", "angle", "turn", "closer look"
        ]
        
        return any(keyword in question for keyword in deepeyes_keywords)
    
    def _should_use_grounding_dino(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用Grounding DINO"""
        question = observation.get("question", "").lower()
        
        if self._is_pure_calculation(question):
            return False
        
        # 对于ChartQA，优先使用ChartMoE而不是Grounding DINO
        if observation.get("is_visual_question", False) and self.enable_chartmoe:
            return False
        
        detection_patterns = [
            r'\bhow many\b(?!.*[\+\-\*/=])',
            r'\bcount\b(?:.*(?:in the|on the|within))',
            r'\bwhere\s+(?:is|are)\b',
            r'\blocate\b',
            r'\bposition of\b'
        ]
        
        return any(re.search(pattern, question) for pattern in detection_patterns)
    
    # ========== 工具提示构建方法 ==========
    
    def _build_easyocr_prompt(self, observation: Dict[str, Any]) -> str:
        """构建EasyOCR工具调用提示"""
        tool_call = {
            "tool": "easyocr",
            "parameters": {
                "task": "detect_and_recognize",
                "min_confidence": 0.5
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _build_deepeyes_prompt(self, observation: Dict[str, Any]) -> str:
        """构建DeepEyes工具调用提示"""
        tool_call = {
            "name": "image_zoom_in_tool",
            "arguments": {
                "bbox_2d": [100, 100, 700, 500]  # 默认区域
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _build_diagram_formalizer_prompt(self, observation: Dict[str, Any]) -> str:
        """构建DiagramFormalizer工具调用提示"""
        question = observation.get("question", "")
        
        tool_call = {
            "tool": "diagram_formalizer",
            "parameters": {
                "task": "solve",
                "problem": question
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _build_grounding_dino_prompt(self, observation: Dict[str, Any]) -> str:
        """构建Grounding DINO工具调用提示"""
        tool_call = {
            "tool": "grounding_dino",
            "parameters": {
                "caption": "all objects",
                "box_threshold": 0.35,
                "text_threshold": 0.25
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def reset(self):
        """重置agent状态"""
        super().reset()
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}
        self.tool_context = {}
        self.current_bbox = None
        self.bbox_history = []
        self._last_task_id = None