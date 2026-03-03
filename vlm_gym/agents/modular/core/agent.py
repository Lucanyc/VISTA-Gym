#!/usr/bin/env python3
"""
Modular VLM Agent with Tools - Task-based tool selection
"""
from typing import Dict, Any, Optional, List, Tuple
import json
import logging
import hashlib
from vlm_gym.agents import VLMAgent
from vlm_gym.agents.modular.tools.registry import ToolRegistry
from vlm_gym.agents.modular.tools.base import ToolConfig, ToolResult, BaseTool
from vlm_gym.agents.modular.strategies.tool_selection import ToolSelector
from vlm_gym.agents.modular.strategies.reflection import ReflectionStrategy

logger = logging.getLogger(__name__)


class VLMAgentWithTools(VLMAgent):
    """
    模块化的VLM Agent with Tools
    核心改进：基于任务能力需求选择工具，而非固定优先级
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Agent"""
        # 分离工具配置和基础配置
        tool_config_keys = [
            "enable_tools", "max_tool_calls", "tool_selection_strategy",
            "enable_chartmoe", "enable_grounding_dino", "enable_deepeyes_tools",
            "enable_sam2", "enable_sympy_geometry", "enable_multimath_server",
            "enable_easyocr", "enable_diagram_formalizer", "enable_tool_collaboration",
            "debug"
        ]
        
        tool_config = {}
        base_config = {}
        
        for key, value in config.items():
            if key in tool_config_keys:
                tool_config[key] = value
            else:
                base_config[key] = value
        
        # 初始化父类
        super().__init__(base_config)
        
        # 日志
        logger.info("[VLMAgentWithTools] Initializing Modular VLM Agent with Tools")
        logger.debug(f"  - Base config keys: {list(base_config.keys())}")
        logger.debug(f"  - Tool config keys: {list(tool_config.keys())}")
        
        # 工具配置
        self.enable_tools = tool_config.get("enable_tools", True)
        self.max_tool_calls = tool_config.get("max_tool_calls", 3)
        self.debug = tool_config.get("debug", False)
        
        # 初始化策略模块
        self.tool_selector = ToolSelector()
        self.reflection_strategy = ReflectionStrategy()
        
        # 初始化工具注册表
        self.registry = ToolRegistry()
        
        # 注册启用的工具
        self._register_enabled_tools(tool_config)
        
        # 工具使用历史（兼容原有格式）
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}  # {question_key: set(tool_names)}
        self.tool_context = {}
        
        # 兼容原有的历史记录属性
        self.chartmoe_history = []
        self.grounding_dino_history = []
        self.deepeyes_history = []
        self.sam2_history = []
        self.sympy_geometry_history = []
        self.multimath_history = []
        self.easyocr_history = []
        
        # 任务跟踪
        self._last_task_id = None
        
        # 工具实例引用（兼容原有代码）
        self.chartmoe_tool = None
        self.grounding_dino_tool = None
        self.deepeyes_tool = None
        self.sam2_tool = None
        self.sympy_geometry_tool = None
        self.multimath_server_tool = None
        self.easyocr_tool = None
        self.diagram_formalizer_tool = None
        
        logger.info(f"  - Tools enabled: {self.enable_tools}")
        logger.info(f"  - Registered tools: {list(self.registry.tools.keys())}")
    
    def _register_enabled_tools(self, config: Dict[str, Any]):
        """注册启用的工具"""
        tool_mappings = {
            "enable_chartmoe": ("chartmoe", 10),
            "enable_grounding_dino": ("grounding_dino", 8),
            "enable_deepeyes_tools": ("deepeyes", 5),
            "enable_sam2": ("sam2", 7),
            "enable_sympy_geometry": ("sympy_geometry", 6),
            "enable_multimath_server": ("multimath_server", 9),
            "enable_easyocr": ("easyocr", 4),
            "enable_diagram_formalizer": ("diagram_formalizer", 7)
        }
        
        for config_key, (tool_name, priority) in tool_mappings.items():
            if config.get(config_key, False):
                # 创建工具配置
                tool_config = ToolConfig(
                    name=tool_name,
                    enabled=True,
                    priority=priority
                )
                
                # 尝试创建工具实例
                tool = self.registry.create_tool(tool_name, tool_config)
                if tool:
                    logger.info(f"  - Registered tool: {tool_name}")
                    # 设置兼容属性
                    setattr(self, f"{tool_name}_tool", tool)
                else:
                    logger.warning(f"  - Failed to register tool: {tool_name}")
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        根据观察生成action - 基于任务需求的工具选择
        返回 (action_string, extra_info_dict)
        """
        # 确保模型已加载
        self.load_model()
        
        if self.debug:
            logger.debug("[act] Starting action generation")
            logger.debug(f"  - Question: {observation.get('question', '')[:100]}...")
            logger.debug(f"  - Attempt: {observation.get('attempt', 1)}")
        
        # 检测任务类型
        task_type = self._detect_task_type(observation)
        observation["task_type"] = task_type
        
        # 检查是否是新任务
        if self._is_new_task(observation):
            self._reset_task_state()
        
        # 记录当前交互
        self.conversation_history.append({"role": "observation", "content": observation})
        
        # 处理工具反馈
        if observation.get("requires_response") and "tool_feedback" in observation:
            return self._handle_tool_feedback(observation)
        
        # 如果不启用工具，直接生成答案
        if not self.enable_tools:
            return self._generate_direct_answer(observation)
        
        # 检查是否达到工具调用上限
        if self.current_tool_calls >= self.max_tool_calls:
            logger.info("Reached max tool calls, generating final answer")
            return self._generate_forced_final_answer(observation)
        
        # 检查是否是反思阶段
        is_reflection = self._is_reflection_attempt(observation)
        
        # 基于任务需求选择工具
        selected_tool = self._select_tool_for_task(observation, is_reflection)
        
        if selected_tool:
            # 使用选定的工具
            return self._use_tool(selected_tool, observation, is_reflection)
        else:
            # 无需工具，生成直接答案
            return self._generate_direct_answer(observation)
    
    def _detect_task_type(self, observation: Dict[str, Any]) -> str:
        """检测任务类型"""
        # ChartQA任务
        if observation.get("is_visual_question") and observation.get("chartmoe_enabled"):
            return "chartqa"
        
        # 医学VQA任务
        if observation.get("is_medical_vqa") or self._is_medical_question(observation):
            return "medical_vqa"
        
        # 几何任务
        if observation.get("is_geometry_task") or self._is_geometry_problem(observation.get("question", "")):
            return "geometry"
        
        # 文本提取任务
        question = observation.get("question", "").lower()
        if any(kw in question for kw in ["read", "text", "written", "ocr"]):
            return "text_extraction"
        
        # 对象检测任务
        if any(kw in question for kw in ["locate", "find", "where", "position"]):
            return "object_detection"
        
        # 默认为通用VQA
        return "general_vqa"
    
    def _select_tool_for_task(self, observation: Dict[str, Any], is_reflection: bool) -> Optional[BaseTool]:
        """基于任务需求选择工具"""
        # 获取可用工具
        available_tools = self.registry.get_available_tools()
        
        if not available_tools:
            return None
        
        # 反思阶段的工具选择
        if is_reflection and self.reflection_strategy.should_retry_tool(observation):
            # 选择替代工具
            tool = self.tool_selector.select_alternative_tool(
                observation, available_tools, self.tool_call_history
            )
            if tool:
                logger.info(f"Reflection: selected alternative tool {tool.name}")
                return tool
        
        # 正常的工具选择
        tool = self.tool_selector.select_tool(
            observation, available_tools, self.tool_call_history
        )
        
        if tool:
            logger.info(f"Selected tool {tool.name} for task")
        
        return tool
    
    def _use_tool(self, tool: BaseTool, observation: Dict[str, Any], is_reflection: bool) -> Tuple[str, Dict[str, Any]]:
        """使用选定的工具"""
        # 获取问题键值用于历史记录
        question = observation.get("question", "")
        current_attempt = observation.get("attempt", 1)
        question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
        
        # 记录工具使用
        if question_key not in self.tool_call_history:
            self.tool_call_history[question_key] = set()
        self.tool_call_history[question_key].add(tool.name)
        
        # 如果是反思阶段，获取特殊参数
        if is_reflection:
            retry_params = self.reflection_strategy.get_tool_retry_params(observation)
            observation.update(retry_params)
        
        # 构建工具调用提示
        tool_call = tool.build_prompt(observation)
        
        # 更新计数器
        self.current_tool_calls += 1
        
        # 记录到相应的历史
        self._record_tool_usage(tool.name, tool_call)
        
        # 构建extra_info
        extra_info = {
            "action_type": "tool_call",
            "tool_used": tool.name,
            "tool_call_count": self.current_tool_calls,
            "is_reflection": is_reflection,
            "task_type": observation.get("task_type", "unknown"),
            "attempt": current_attempt
        }
        
        logger.info(f"Generated tool call for {tool.name}")
        
        return tool_call, extra_info
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理工具反馈"""
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        logger.info(f"Processing {tool_name} feedback")
        
        # 获取工具实例
        tool = self.registry.get_tool(tool_name)
        
        if not tool:
            logger.warning(f"Tool {tool_name} not found in registry")
            return self._generate_direct_answer(observation)
        
        # 处理工具结果
        result = tool.process_result(tool_feedback, observation)
        
        # 生成基于工具结果的答案提示
        formatted_prompt = tool.format_for_answer(result, observation)
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["output_format_instruction"] = formatted_prompt
        enhanced_observation["available_tools"] = []  # 不再使用工具
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        # 构建extra_info
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "tool_feedback_response",
            "tool_used": tool_name,
            "tool_success": result.success,
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """生成直接答案（不使用工具）"""
        logger.info("Generating direct answer without tools")
        
        # 调用父类的act方法
        response, base_info = super().act(observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        # 添加额外信息
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
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "forced_final_answer",
            "tool_call_count": self.current_tool_calls,
            "reason": "reached_tool_limit",
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _record_tool_usage(self, tool_name: str, tool_call: str):
        """记录工具使用到相应的历史"""
        history_entry = {
            "step": self.current_tool_calls,
            "action": tool_call
        }
        
        # 记录到对应的历史列表（兼容原有格式）
        if tool_name == "chartmoe":
            self.chartmoe_history.append(history_entry)
        elif tool_name == "grounding_dino":
            self.grounding_dino_history.append(history_entry)
        elif tool_name == "deepeyes":
            self.deepeyes_history.append(history_entry)
        elif tool_name == "sam2":
            self.sam2_history.append(history_entry)
        elif tool_name == "sympy_geometry":
            self.sympy_geometry_history.append(history_entry)
        elif tool_name == "multimath_server":
            self.multimath_history.append(history_entry)
        elif tool_name == "easyocr":
            self.easyocr_history.append(history_entry)
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """判断是否是新任务"""
        current_task_id = observation.get("task_id")
        if hasattr(self, "_last_task_id") and self._last_task_id != current_task_id:
            self._last_task_id = current_task_id
            return True
        
        self._last_task_id = current_task_id
        
        if not self.conversation_history:
            return True
        
        return observation.get("episode_start", False)
    
    def _reset_task_state(self):
        """重置任务状态"""
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}
        self.tool_context = {}
        
        # 清空各工具历史
        self.chartmoe_history = []
        self.grounding_dino_history = []
        self.deepeyes_history = []
        self.sam2_history = []
        self.sympy_geometry_history = []
        self.multimath_history = []
        self.easyocr_history = []
        
        logger.debug("Reset task state for new task")
    
    def _is_reflection_attempt(self, observation: Dict[str, Any]) -> bool:
        """检测是否处于反思阶段"""
        return (
            observation.get("previous_attempt_failed", False) or
            observation.get("feedback") is not None or
            observation.get("attempt", 1) > 1
        )
    
    def _is_medical_question(self, observation: Dict[str, Any]) -> bool:
        """判断是否是医学相关问题"""
        question = observation.get("question", "").lower()
        
        medical_keywords = [
            'brain', 'lung', 'heart', 'liver', 'kidney', 'organ',
            'ct', 'mri', 'x-ray', 'scan', 'medical', 'diagnosis',
            'disease', 'abnormal', 'normal', 'lesion', 'tumor'
        ]
        
        return any(keyword in question for keyword in medical_keywords)
    
    def _is_geometry_problem(self, question: str) -> bool:
        """判断是否是几何问题"""
        geometry_keywords = [
            'angle', 'degree', '°', 'triangle', 'circle', 'square',
            'perimeter', 'area', 'volume', 'parallel', 'perpendicular',
            'congruent', 'similar', 'polygon', 'radius', 'diameter',
            'pythagorean', 'theorem', 'proof'
        ]
        return any(keyword in question.lower() for keyword in geometry_keywords)
    
    def _ensure_answer_format(self, response: str, observation: Dict[str, Any]) -> str:
        """确保响应包含正确的格式"""
        # 检查是否已有正确格式
        if "<answer>" in response and "</answer>" in response:
            return response
        
        # 如果是工具调用，不需要修改
        if "<tool_call>" in response:
            return response
        
        # 尝试提取答案并格式化
        # 这里简化处理，实际可能需要更复杂的逻辑
        return f"""<think>
{response}
</think>
<answer>{response.strip()}</answer>"""
    
    def reset(self):
        """重置agent状态"""
        super().reset()
        self._reset_task_state()
        self._last_task_id = None