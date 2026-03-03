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
    Secondary support: DeepEyes, GroundingDINO, DiagramFormalizer, EasyOCR, SAM2, SymPy Geometry, MultiMath Server, GeometrySolver
    """
    
    # 定义工具索引映射 - 添加geometry_solver
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
        8: {  # 新增：geometry_solver
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
            "enable_geometry_solver",  # 新增
            "grounding_dino_config",
            "diagram_formalizer_config",
            "chartmoe_config",
            "deepeyes_config",
            "easyocr_config",
            "sam2_config",
            "sympy_geometry_config",
            "multimath_server_config",
            "geometry_solver_config",  # 新增
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
        self.enable_sam2 = tool_config.get("enable_sam2", False)
        self.enable_sympy_geometry = tool_config.get("enable_sympy_geometry", False)
        self.enable_multimath_server = tool_config.get("enable_multimath_server", False)
        self.enable_geometry_solver = tool_config.get("enable_geometry_solver", False)  # 新增
        
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - enable_chartmoe: {self.enable_chartmoe}")
        print(f"  - enable_grounding_dino: {self.enable_grounding_dino}")
        print(f"  - enable_deepeyes_tools: {self.enable_deepeyes_tools}")
        print(f"  - enable_easyocr: {self.enable_easyocr}")
        print(f"  - enable_sam2: {self.enable_sam2}")
        print(f"  - enable_sympy_geometry: {self.enable_sympy_geometry}")
        print(f"  - enable_diagram_formalizer: {self.enable_diagram_formalizer}")
        print(f"  - enable_multimath_server: {self.enable_multimath_server}")
        print(f"  - enable_geometry_solver: {self.enable_geometry_solver}")  # 新增
        
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
        self.sam2_tool = None
        self.sympy_geometry_tool = None
        self.multimath_server_tool = None
        self.geometry_solver_tool = None  # 新增
        
        print(f"  - Tool instances initialized to None (will be set by environment)")
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        根据观察生成action - ChartQA优化版本 with reflection support
        返回 (action_string, extra_info_dict)
        """
        # 确保模型已加载
        self.load_model()
        
        # 添加调试输出
        print(f"  - enable_chartmoe: {self.enable_chartmoe}")
        print(f"  - enable_sam2: {self.enable_sam2}")
        print(f"  - enable_sympy_geometry: {self.enable_sympy_geometry}")
        print(f"  - enable_multimath_server: {self.enable_multimath_server}")
        print(f"  - enable_geometry_solver: {self.enable_geometry_solver}")  # 新增
        
        # 检查是否是ChartQA任务
        is_chartqa = observation.get("is_visual_question", False) and observation.get("chartmoe_enabled", False)
        if is_chartqa:
            print(f"  - ✓ ChartQA task detected")
        
        # 检查是否是医学图像任务（VQA-RAD）
        is_medical = observation.get("is_medical_vqa", False) or self._is_medical_question(observation)
        if is_medical:
            print(f"  - ✓ Medical VQA task detected")
        
        # 检查是否是几何任务
        is_geometry = observation.get("is_geometry_task", False) or self._is_geometry_problem(observation.get("question", ""))
        if is_geometry:
            print(f"  - ✓ Geometry task detected")
        
        # 检查是否需要使用geometry_solver
        requires_geometry_solver = observation.get("requires_geometry_solver", False)
        if requires_geometry_solver:
            print(f"  - ✓ Requires geometry_solver tool")
        
        # 检查是否处于反思阶段
        is_reflection = self._is_reflection_attempt(observation)
        if is_reflection:
            print(f"  - ✓ Reflection attempt detected (attempt {observation.get('attempt', 1)})")
        
        # 清除已使用工具的must_use_tool标志（除非在反思阶段需要重新使用）
        if isinstance(observation, dict):
            # 如果geometry_solver已经使用过，但不是反思阶段，清除强制标志
            if observation.get("geometry_solver_history") and len(observation.get("geometry_solver_history", [])) > 0 and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ geometry_solver already used, cleared must_use_tool flags")
            
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
            
            # SAM2相关
            if observation.get("sam2_history") and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ SAM2 already used, cleared must_use_tool flags")
            
            # SymPy相关
            if observation.get("sympy_geometry_history") and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ SymPy Geometry already used, cleared must_use_tool flags")
            
            # MultiMath Server相关
            if observation.get("multimath_history") and not is_reflection:
                observation.pop("must_use_tool", None)
                observation.pop("tool_to_use", None)
                print(f"  - ✓ MultiMath Server already used, cleared must_use_tool flags")
        
        # 处理强制使用工具的情况
        if isinstance(observation, dict) and observation.get("must_use_tool"):
            tool_to_use = observation.get("tool_to_use")
            
            # geometry_solver强制使用
            if tool_to_use == "geometry_solver" and self.enable_geometry_solver and self.geometry_solver_tool is not None:
                print(f"  - ⚠️ FORCED tool usage: {tool_to_use}")
                
                tool_call = self._build_geometry_solver_prompt(observation)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if is_reflection:
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "geometry_solver",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "is_reflection": is_reflection
                }
                
                #print(f"[VLMAgentWithTools.act] DEBUG END\n")
                #return tool_call, extra_info
                if isinstance(response, str) and '<tool_call>' in response:
                    # 确保工具调用格式正确，没有被包装
                    if response.startswith('answer_question('):
                        print(f"[WARNING] Tool call incorrectly wrapped! Attempting to fix...")
                        # 尝试提取内容
                        import re
                        match = re.search(r'answer_question\(answer="(.+)"\)', response, re.DOTALL)
                        if match:
                            response = match.group(1).replace('\\"', '"')
                            print(f"[INFO] Fixed tool call format")
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return response, extra_info

            # MultiMath Server强制使用
            if tool_to_use == "multimath_server" and self.enable_multimath_server and self.multimath_server_tool is not None:
                print(f"  - ⚠️ FORCED tool usage: {tool_to_use}")
                
                tool_call = self._build_multimath_server_prompt(observation)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if is_reflection:
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "multimath_server",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "is_reflection": is_reflection
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
            
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
            
            # SAM2强制使用
            if tool_to_use == "sam2" and self.enable_sam2 and self.sam2_tool is not None:
                print(f"  - ⚠️ FORCED tool usage: {tool_to_use}")
                
                tool_call = self._build_sam2_prompt(observation)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if is_reflection:
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "sam2",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "is_reflection": is_reflection
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
            
            # SymPy强制使用
            if tool_to_use == "sympy_geometry" and self.enable_sympy_geometry and self.sympy_geometry_tool is not None:
                print(f"  - ⚠️ FORCED tool usage: {tool_to_use}")
                
                tool_call = self._build_sympy_geometry_prompt(observation)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                if is_reflection:
                    self.tool_use_stats["reflection_tool_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "sympy_geometry",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "is_reflection": is_reflection
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
        
        # 特殊处理：对于需要geometry_solver的任务
        if requires_geometry_solver and self.enable_geometry_solver and self.geometry_solver_tool is not None:
            question = observation.get("question", "")
            current_attempt = observation.get("attempt", 1)
            question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
            
            if question_key not in self.tool_call_history or "geometry_solver" not in self.tool_call_history.get(question_key, set()):
                print(f"  - Geometry task: using geometry_solver")
                tool_call = self._build_geometry_solver_prompt(observation)
                
                # 记录工具调用
                if question_key not in self.tool_call_history:
                    self.tool_call_history[question_key] = set()
                self.tool_call_history[question_key].add("geometry_solver")
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                
                extra_info = {
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "tool_used": "geometry_solver",
                    "reason": "Geometry problem solving with combined tool",
                    "first_call": True
                }
                
                print(f"[VLMAgentWithTools.act] DEBUG END\n")
                return tool_call, extra_info
        
        # 特殊处理：对于几何任务强制使用 MultiMath Server
        if is_geometry and observation.get("force_tool_use") and observation.get("multimath_enabled"):
            if self.enable_multimath_server and self.multimath_server_tool is not None:
                print(f"  - ⚠️ Geometry task with forced MultiMath Server usage")
                
                tool_call = self._build_multimath_server_prompt(observation)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                
                extra_info = {
                    "tool_forced": True,
                    "tool_used": "multimath_server",
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "stage": "forced_tool_call",
                    "reason": "Geometry task requires MultiMath Server",
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
        
        # 几何任务特殊处理：优先使用geometry_solver（如果启用）
        if is_geometry:
            question = observation.get("question", "")
            current_attempt = observation.get("attempt", 1)
            question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
            
            # 如果geometry_solver可用，优先使用
            if 8 in available_tools and self._should_use_geometry_solver(observation):
                if question_key not in self.tool_call_history or "geometry_solver" not in self.tool_call_history.get(question_key, set()):
                    print(f"  - Geometry task: using geometry_solver for solving")
                    tool_call = self._build_geometry_solver_prompt(observation)
                    
                    # 记录工具调用
                    if question_key not in self.tool_call_history:
                        self.tool_call_history[question_key] = set()
                    self.tool_call_history[question_key].add("geometry_solver")
                    
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "tool_call_count": self.current_tool_calls,
                        "tool_used": "geometry_solver",
                        "reason": "Geometry problem solving with combined tool",
                        "first_call": True
                    }
                    
                    print(f"[VLMAgentWithTools.act] DEBUG END\n")
                    return tool_call, extra_info
            # 如果 MultiMath Server 可用，使用它
            elif 7 in available_tools and self._should_use_multimath_server(observation):
                if question_key not in self.tool_call_history or "multimath_server" not in self.tool_call_history.get(question_key, set()):
                    print(f"  - Geometry task: using MultiMath Server for solving")
                    tool_call = self._build_multimath_server_prompt(observation)
                    
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
                        "first_call": True
                    }
                    
                    print(f"[VLMAgentWithTools.act] DEBUG END\n")
                    return tool_call, extra_info
            # 否则使用 SymPy
            elif 6 in available_tools and self._should_use_sympy_geometry(observation):
                if question_key not in self.tool_call_history or "sympy_geometry" not in self.tool_call_history.get(question_key, set()):
                    print(f"  - Geometry task: using SymPy for precise calculation")
                    tool_call = self._build_sympy_geometry_prompt(observation)
                    
                    # 记录工具调用
                    if question_key not in self.tool_call_history:
                        self.tool_call_history[question_key] = set()
                    self.tool_call_history[question_key].add("sympy_geometry")
                    
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "tool_call_count": self.current_tool_calls,
                        "tool_used": "sympy_geometry",
                        "reason": "Precise geometric calculation needed",
                        "first_call": True
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
        
        # 医学图像特殊处理：考虑使用SAM2
        if is_medical and 5 in available_tools and not is_reflection:
            question = observation.get("question", "")
            current_attempt = observation.get("attempt", 1)
            question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
            
            # 如果问题涉及定位、分割或特定区域
            if self._should_use_sam2(observation):
                if question_key not in self.tool_call_history or "sam2" not in self.tool_call_history.get(question_key, set()):
                    print(f"  - Medical VQA task: using SAM2 for segmentation")
                    tool_call = self._build_sam2_prompt(observation)
                    
                    # 记录工具调用
                    if question_key not in self.tool_call_history:
                        self.tool_call_history[question_key] = set()
                    self.tool_call_history[question_key].add("sam2")
                    
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "tool_call_count": self.current_tool_calls,
                        "tool_used": "sam2",
                        "reason": "Medical image segmentation needed",
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
    
    def _is_medical_question(self, observation: Dict[str, Any]) -> bool:
        """判断是否是医学相关问题"""
        question = observation.get("question", "").lower()
        
        medical_keywords = [
            'brain', 'lung', 'heart', 'liver', 'kidney', 'organ',
            'ct', 'mri', 'x-ray', 'scan', 'medical', 'diagnosis',
            'disease', 'abnormal', 'normal', 'lesion', 'tumor',
            'infarcted', 'cardiovascular', 'pulmonary', 'cerebral'
        ]
        
        return any(keyword in question for keyword in medical_keywords)
    
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
        if self.enable_sam2 and self.sam2_tool is not None:
            available.append(5)
        if self.enable_sympy_geometry and self.sympy_geometry_tool is not None:
            available.append(6)
        if self.enable_multimath_server and self.multimath_server_tool is not None:
            available.append(7)
        if self.enable_geometry_solver and self.geometry_solver_tool is not None:  # 新增
            available.append(8)
        
        # 对于ChartQA任务的特殊处理
        if observation.get("chartmoe_enabled", False) and self.enable_chartmoe:
            if 3 not in available:
                available.append(3)
        
        # 对于医学图像任务的特殊处理
        if observation.get("is_medical_vqa", False) and self.enable_sam2:
            if 5 not in available:
                available.append(5)
        
        # 对于几何任务的特殊处理
        if observation.get("is_geometry_task", False):
            if self.enable_geometry_solver and 8 not in available:
                available.append(8)
            elif self.enable_multimath_server and 7 not in available:
                available.append(7)
            elif self.enable_sympy_geometry and 6 not in available:
                available.append(6)
        
        # 对于启用了 multimath 的情况
        if observation.get("multimath_enabled", False) and self.enable_multimath_server:
            if 7 not in available:
                available.append(7)
        
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
        
        # 如果明确要求使用工具
        if observation.get("force_tool_use") or observation.get("tool_use_required"):
            return True
        
        # ChartQA任务：如果有ChartMoE可用，应该使用
        if observation.get("is_visual_question", False) and 3 in available_tools:
            print(f"  - Visual question with ChartMoE available, tool needed")
            return True
        
        # 医学图像任务：如果有SAM2可用且需要分割
        if observation.get("is_medical_vqa", False) and 5 in available_tools:
            if self._should_use_sam2(observation):
                print(f"  - Medical image with SAM2 available, tool needed")
                return True
        
        # 几何任务：如果有 geometry_solver、MultiMath Server 或 SymPy 可用
        if self._is_geometry_problem(question):
            if 8 in available_tools:  # geometry_solver
                print(f"  - Geometry problem with geometry_solver available, tool needed")
                return True
            elif 7 in available_tools:  # MultiMath Server
                print(f"  - Geometry problem with MultiMath Server available, tool needed")
                return True
            elif 6 in available_tools:  # SymPy
                print(f"  - Geometry problem with SymPy available, tool needed")
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
        
        # 对象检测
        if 2 in available_tools and self._should_use_grounding_dino(observation):
            return True
        
        return False
    
    def _generate_tool_call(self, observation: Dict[str, Any], available_tools: List[int]) -> Tuple[str, Dict[str, Any]]:
        """生成工具调用 - 支持反思重试"""
        question = observation.get("question", "")
        current_attempt = observation.get("attempt", 1)
        question_key = hashlib.md5(f"{question}_attempt_{current_attempt}".encode()).hexdigest()
        
        # 几何问题优先使用geometry_solver
        if self._is_geometry_problem(question) and 8 in available_tools:
            if question_key not in self.tool_call_history or "geometry_solver" not in self.tool_call_history.get(question_key, set()):
                tool_name = "geometry_solver"
                tool_call = self._build_geometry_solver_prompt(observation)
                
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
                    "reason": "Geometry problem solving with combined tool",
                    "first_call": current_attempt == 1,
                    "is_reflection": self._is_reflection_attempt(observation)
                }
                
                return tool_call, extra_info
        
        # 几何问题使用 MultiMath Server（如果geometry_solver不可用）
        if self._is_geometry_problem(question) and 7 in available_tools:
            if question_key not in self.tool_call_history or "multimath_server" not in self.tool_call_history.get(question_key, set()):
                tool_name = "multimath_server"
                tool_call = self._build_multimath_server_prompt(observation)
                
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
                    "reason": "Geometry problem solving with MultiMath",
                    "first_call": current_attempt == 1,
                    "is_reflection": self._is_reflection_attempt(observation)
                }
                
                return tool_call, extra_info
        
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
        
        # 医学图像优先使用SAM2
        if observation.get("is_medical_vqa", False) and 5 in available_tools:
            if self._should_use_sam2(observation) and (question_key not in self.tool_call_history or "sam2" not in self.tool_call_history.get(question_key, set())):
                tool_name = "sam2"
                tool_call = self._build_sam2_prompt(observation)
                
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
                    "reason": "Medical image segmentation",
                    "first_call": current_attempt == 1,
                    "is_reflection": self._is_reflection_attempt(observation)
                }
                
                return tool_call, extra_info
        
        # 其他工具选择逻辑
        # 根据问题类型和可用工具选择
        tool_name = None
        tool_call = None
        
        # 按优先级检查工具
        # 几何问题优先使用SymPy（如果MultiMath和geometry_solver都不可用）
        if 6 in available_tools and self._should_use_sympy_geometry(observation):
            if question_key not in self.tool_call_history or "sympy_geometry" not in self.tool_call_history.get(question_key, set()):
                tool_name = "sympy_geometry"
                tool_call = self._build_sympy_geometry_prompt(observation)
        
        elif 4 in available_tools and self._should_use_easyocr(observation):
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
        
        # 特殊处理geometry_solver反馈
        if tool_name == "geometry_solver":
            return self._handle_geometry_solver_feedback_special(observation)
        
        # 特殊处理MultiMath Server反馈
        if tool_name == "multimath_server":
            return self._handle_multimath_server_feedback_special(observation)
        
        # 特殊处理ChartMoE反馈
        if tool_name == "chartmoe":
            return self._handle_chartmoe_feedback_special(observation)
        
        # 特殊处理SAM2反馈
        if tool_name == "sam2":
            return self._handle_sam2_feedback_special(observation)
        
        # 特殊处理SymPy反馈
        if tool_name == "sympy_geometry":
            return self._handle_sympy_geometry_feedback_special(observation)
        
        # 其他工具的处理
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
    
    def _handle_geometry_solver_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理geometry_solver反馈"""
        tool_feedback = observation.get("tool_feedback", {})
        success = tool_feedback.get("success", False)
        final_answer = tool_feedback.get("final_answer", "")
        cdl = tool_feedback.get("cdl", {})
        mm_output = tool_feedback.get("multimath_output", {})
        
        # 提取原始问题
        question = observation.get("original_question", observation.get("question", ""))
        
        print(f"  - geometry_solver success: {success}")
        print(f"  - Final answer: {final_answer}")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 构建基于geometry_solver输出的提示
        if success and final_answer:
            prompt = f"""Based on the geometry_solver analysis:

The combined DiagramFormalizer and MultiMath solver has solved the problem.

CDL extracted: {cdl.get('image_cdl', 'N/A')[:200]}...
MultiMath solution method: {mm_output.get('method', 'N/A')}
Confidence: {mm_output.get('confidence', 0):.2%}

Final answer: {final_answer}

Now provide the answer to: "{question}"

You MUST output your answer inside <answer> tags.
The answer should be: <answer>{final_answer}</answer>"""
        else:
            # geometry_solver失败的情况
            error_msg = tool_feedback.get("error", "Unknown error")
            prompt = f"""The geometry_solver could not solve this problem.

Error: {error_msg}

Question: "{question}"

Please analyze the problem and provide your best answer based on the available information.

You MUST output your answer inside <answer> tags."""
        
        enhanced_observation["output_format_instruction"] = prompt
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "geometry_solver_analysis",
            "tool_used": "geometry_solver",
            "success": success,
            "stage": "final_answer_with_geometry_solver",
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _handle_multimath_server_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理MultiMath Server反馈"""
        tool_feedback = observation.get("tool_feedback", {})
        success = tool_feedback.get("success", False)
        answer = tool_feedback.get("answer", "")
        method = tool_feedback.get("method", "")
        steps = tool_feedback.get("steps", [])
        confidence = tool_feedback.get("confidence", 0)
        
        # 提取原始问题
        question = observation.get("original_question", observation.get("question", ""))
        
        print(f"  - MultiMath Server success: {success}")
        print(f"  - Answer: {answer}")
        print(f"  - Method: {method}")
        print(f"  - Confidence: {confidence}")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 构建基于MultiMath输出的提示
        if success and answer:
            # 根据置信度和步骤构建提示
            if steps:
                steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps[:5])])  # 最多显示5步
                prompt = f"""Based on the MultiMath Server analysis:

Method used: {method}
Confidence: {confidence:.2%}

Solution steps:
{steps_text}

Final answer: {answer}

Now provide the answer to: "{question}"

You MUST output your answer inside <answer> tags.
The answer should be: <answer>{answer}</answer>"""
            else:
                prompt = f"""Based on the MultiMath Server calculation:

Method: {method}
Answer: {answer}
Confidence: {confidence:.2%}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags.
The answer should be: <answer>{answer}</answer>"""
        else:
            # MultiMath失败的情况
            prompt = f"""The MultiMath Server could not solve this problem.

Question: "{question}"

Please analyze the problem and provide your best answer based on the available information.

You MUST output your answer inside <answer> tags."""
        
        enhanced_observation["output_format_instruction"] = prompt
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "multimath_analysis",
            "tool_used": "multimath_server",
            "success": success,
            "method": method,
            "confidence": confidence,
            "stage": "final_answer_with_multimath",
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
    
    def _handle_sam2_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理SAM2反馈 - 针对医学图像优化"""
        tool_feedback = observation.get("tool_feedback", {})
        task_type = tool_feedback.get("task", "unknown")
        results = tool_feedback.get("results", [])
        
        # 提取原始问题
        question = observation.get("original_question", observation.get("question", ""))
        question_lower = question.lower()
        
        print(f"  - SAM2 task type: {task_type}")
        print(f"  - Question: {question[:100]}...")
        print(f"  - Number of segmentation results: {len(results)}")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 构建基于SAM2输出的提示
        if results:
            # 提取关键信息
            best_mask = max(results, key=lambda x: x.get('score', 0))
            coverage = best_mask.get('coverage_percent', 0)
            bbox = best_mask.get('bbox', [])
            
            prompt = f"""Based on the SAM2 segmentation analysis:

Task: {task_type}
Best segmentation result:
- Coverage: {coverage:.1f}% of the image
- Bounding box: {bbox}
- Confidence score: {best_mask.get('score', 0):.3f}

Number of segmentation masks generated: {len(results)}

Now answer the question: "{question}"

Instructions:
1. Use the segmentation information to understand the image regions
2. For questions about specific organs or regions, consider the coverage percentage
3. For yes/no questions about abnormalities, larger coverage might indicate issues
4. Consider that normal organs typically have predictable coverage ranges

You MUST output your answer inside <answer> tags."""
        
        else:
            # 没有分割结果
            prompt = f"""SAM2 segmentation did not produce results for this image.

This might indicate:
- The image doesn't have clear regions to segment
- The medical structures are not well-defined
- The image quality might be an issue

Answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
        
        enhanced_observation["output_format_instruction"] = prompt
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "sam2_analysis",
            "tool_used": "sam2",
            "task_type": task_type,
            "stage": "final_answer_with_sam2",
            "segmentation_count": len(results),
            "best_coverage": coverage if results else 0,
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _handle_sympy_geometry_feedback_special(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """特殊处理SymPy几何反馈"""
        tool_feedback = observation.get("tool_feedback", {})
        function_name = tool_feedback.get("function", "unknown")
        result = tool_feedback.get("result", {})
        
        # 提取原始问题
        question = observation.get("original_question", observation.get("question", ""))
        
        print(f"  - SymPy function: {function_name}")
        print(f"  - Question: {question[:100]}...")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []
        
        # 构建基于SymPy输出的提示
        if result:
            # 根据不同的函数类型构建提示
            if function_name == "triangle_angle":
                angle = result.get("angle_degrees", "N/A")
                prompt = f"""Based on the SymPy geometric calculation:

The angle at the specified vertex is: {angle}°

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            elif function_name == "pythagorean":
                c_value = result.get("c", "N/A")
                prompt = f"""Based on the Pythagorean theorem calculation:

The hypotenuse (c) = {c_value}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            elif function_name == "polygon_area":
                area = result.get("area", "N/A")
                perimeter = result.get("perimeter", "N/A")
                prompt = f"""Based on the polygon calculation:

Area: {area}
Perimeter: {perimeter}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            elif function_name == "circle_from_points":
                center = result.get("center", [])
                radius = result.get("radius", "N/A")
                area = result.get("area", "N/A")
                circumference = result.get("circumference", "N/A")
                prompt = f"""Based on the circle calculation:

Center: ({center[0]}, {center[1]})
Radius: {radius}
Area: {area}
Circumference: {circumference}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            elif function_name == "triangle_area":
                area = result.get("area", "N/A")
                prompt = f"""Based on the triangle area calculation:

Area: {area}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            elif function_name == "inscribed_angle":
                inscribed = result.get("inscribed_angle", "N/A")
                central = result.get("central_angle", "N/A")
                prompt = f"""Based on the inscribed angle theorem:

Inscribed angle: {inscribed}°
Central angle: {central}°

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
            
            else:
                # 通用格式
                prompt = f"""Based on the SymPy geometric calculation ({function_name}):

{json.dumps(result, indent=2)}

Now answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
        
        else:
            prompt = f"""SymPy calculation did not produce results.

Answer the question: "{question}"

You MUST output your answer inside <answer> tags."""
        
        enhanced_observation["output_format_instruction"] = prompt
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "sympy_geometry_analysis",
            "tool_used": "sympy_geometry",
            "function_used": function_name,
            "stage": "final_answer_with_sympy",
            "has_reflection": self._is_reflection_attempt(observation),
            "attempt": observation.get("attempt", 1)
        })
        
        return response, extra_info
    
    def _build_geometry_solver_prompt(self, observation: Dict[str, Any]) -> str:
        """构建geometry_solver工具调用提示"""
        question = observation.get("question", "")
        image_path = observation.get("image_path", "")
        
        # 清理问题中的强制指令（如果有）
        if "MANDATORY INSTRUCTION" in question:
            # 提取原始问题
            original_question = observation.get("original_question", question)
            clean_question = original_question
        else:
            clean_question = question
        
        # 构建工具调用
        tool_call = {
            "tool": "geometry_solver",
            "parameters": {
                "image_path": image_path,
                "question": clean_question
            }
        }
        
        print(f"  - geometry_solver solving: '{clean_question[:100]}...'")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _build_multimath_server_prompt(self, observation: Dict[str, Any]) -> str:
        """构建MultiMath Server工具调用提示"""
        question = observation.get("question", "")
        original_question = observation.get("original_question", question)
        
        # 清理问题，去除强制使用工具的指令
        if "IMPORTANT: You MUST use the MultiMath Server tool" in question:
            # 提取原始的几何问题
            import re
            match = re.search(r'Geometry problem: (.+?)(?:\n\nStep 1:|$)', question, re.DOTALL)
            if match:
                clean_question = match.group(1).strip()
            else:
                clean_question = original_question
        else:
            clean_question = original_question
        
        # 构建工具调用
        tool_call = {
            "tool": "multimath_server",
            "parameters": {
                "task": "solve",
                "question": clean_question,
                "problem_type": "geometry",
                "output_format": "answer_only"
            }
        }
        
        print(f"  - MultiMath Server solving: '{clean_question[:100]}...'")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
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
    
    def _build_sam2_prompt(self, observation: Dict[str, Any]) -> str:
        """构建SAM2工具调用提示 - 医学图像优化版本"""
        question = observation.get("question", "")
        question_lower = question.lower()
        
        # 检测是否有先前的检测结果（如来自GroundingDINO）
        if "grounding_dino_results" in self.tool_context:
            # 使用检测框作为SAM2的输入
            detections = self.tool_context["grounding_dino_results"]
            if detections and len(detections) > 0:
                # 使用第一个检测框
                bbox = detections[0].get("bbox", None)
                if bbox:
                    tool_call = {
                        "tool": "sam2",
                        "task": "box_segment",
                        "box": bbox
                    }
                    print(f"  - SAM2 using GroundingDINO box: {bbox}")
                    json_str = json.dumps(tool_call, ensure_ascii=False)
                    return f'<tool_call>{json_str}</tool_call>'
        
        # 基于问题内容选择合适的SAM2任务
        if any(kw in question_lower for kw in ["brain", "cerebral", "head", "skull"]):
            # 脑部相关 - 使用智能医学分割
            task = "smart_medical_segment"
        elif any(kw in question_lower for kw in ["lung", "pulmonary", "chest", "thorax"]):
            # 肺部相关 - 使用智能医学分割
            task = "smart_medical_segment"
        elif any(kw in question_lower for kw in ["heart", "cardiac", "cardiovascular"]):
            # 心脏相关 - 使用智能医学分割
            task = "smart_medical_segment"
        elif any(kw in question_lower for kw in ["region", "area", "location", "where"]):
            # 需要定位特定区域 - 使用多点分割
            task = "multi_point_segment"
        else:
            # 默认使用智能医学分割
            task = "smart_medical_segment"
        
        tool_call = {
            "tool": "sam2",
            "task": task,
            "question": question  # 传递问题以便智能分析
        }
        
        print(f"  - SAM2 task selected: {task}")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    def _build_sympy_geometry_prompt(self, observation: Dict[str, Any]) -> str:
        """构建SymPy几何工具调用提示 - 引导Agent从图像中提取信息"""
        question = observation.get("question", "")
        question_lower = question.lower()
        
        # 构建引导Agent分析图像的提示
        base_prompt = """Analyze the geometry problem step by step:

1. First, examine the geometric figure in the image carefully
2. Identify all points/vertices and their labels (e.g., A, B, C, D, E)
3. Based on the figure and given information, establish a coordinate system
4. Determine reasonable coordinates for each point

Then call the SymPy geometry tool with the extracted information."""

        # 根据问题类型提供具体的示例和指导
        if 'angle' in question_lower:
            if 'triangle' in question_lower:
                specific_prompt = f"""
{base_prompt}

For this triangle angle problem:
- Set up coordinates for the triangle vertices
- Common approach: place one vertex at origin (0,0), another on x-axis
- Example: For triangle ABC, you might use A=(0,0), B=(4,0), C=(2,3)

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "triangle_angle", "args": {{"A": [x1, y1], "B": [x2, y2], "C": [x3, y3], "vertex": "A"}}}}}}
</tool_call>

Question: {question}"""
            else:
                # 角度但非三角形
                specific_prompt = f"""
{base_prompt}

For this angle problem:
- Identify the lines or segments forming the angle
- Set up coordinates for the endpoints

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "angle_between_lines", "args": {{"p1": [x1, y1], "p2": [x2, y2], "p3": [x3, y3], "p4": [x4, y4]}}}}}}
</tool_call>

Question: {question}"""

        elif 'pythagorean' in question_lower or 'right triangle' in question_lower:
            specific_prompt = f"""
{base_prompt}

For this Pythagorean theorem problem:
- Identify the right triangle in the image
- Note which sides are given and which need to be calculated
- Extract the numerical values for the known sides

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "pythagorean", "args": {{"a": 3, "b": 4}}}}}}
</tool_call>
(Include only the known sides in args)

Question: {question}"""

        elif 'area' in question_lower:
            if 'triangle' in question_lower:
                specific_prompt = f"""
{base_prompt}

For this triangle area problem:
- Identify the three vertices of the triangle
- Set up a coordinate system and assign coordinates to each vertex
- Consider any given measurements (side lengths, heights, etc.)

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "triangle_area", "args": {{"A": [0, 0], "B": [6, 0], "C": [3, 4]}}}}}}
</tool_call>

Question: {question}"""
            else:
                # 多边形面积
                specific_prompt = f"""
{base_prompt}

For this polygon area problem:
- Identify all vertices of the polygon in order
- Set up coordinates for each vertex

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "polygon_area", "args": {{"vertices": [[0, 0], [4, 0], [4, 3], [0, 3]]}}}}}}
</tool_call>

Question: {question}"""

        elif 'circle' in question_lower:
            specific_prompt = f"""
{base_prompt}

For this circle problem:
- Identify key points on the circle or related to it
- If finding a circle through points, identify three non-collinear points

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "circle_from_points", "args": {{"p1": [0, 0], "p2": [3, 0], "p3": [0, 4]}}}}}}
</tool_call>

Question: {question}"""

        elif 'distance' in question_lower and 'point' in question_lower and 'line' in question_lower:
            specific_prompt = f"""
{base_prompt}

For this point-to-line distance problem:
- Identify the point and the line
- Set up coordinates for the point and two points on the line

After analyzing the image, make a tool call like:
<tool_call>
{{"tool": "sympy_geometry", "parameters": {{"function": "distance_point_to_line", "args": {{"point": [2, 3], "line_p1": [0, 0], "line_p2": [4, 0]}}}}}}
</tool_call>

Question: {question}"""

        else:
            # 默认情况：分析图形并选择合适的函数
            specific_prompt = f"""
{base_prompt}

Based on the geometric figure and question:
1. Identify what type of calculation is needed
2. Extract relevant points and their coordinates
3. Choose the appropriate SymPy function:
- triangle_angle: for triangle angles
- triangle_area: for triangle area
- pythagorean: for right triangle calculations
- polygon_area: for polygon areas
- circle_from_points: for circle problems
- angle_between_lines: for angle between lines

Make an appropriate tool call with the extracted coordinates.

Question: {question}"""

        # 不直接返回工具调用，而是返回提示让Agent生成
        return specific_prompt
    
    def _should_use_geometry_solver(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用geometry_solver"""
        # 如果明确要求使用
        if observation.get("requires_geometry_solver", False):
            return True
        
        # 如果是几何问题且geometry_solver可用
        question = observation.get("question", "").lower()
        if self._is_geometry_problem(question):
            return True
        
        return False
    
    def _should_use_multimath_server(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用MultiMath Server"""
        question = observation.get("question", "").lower()
        
        # 如果明确要求使用
        if observation.get("force_tool_use") or observation.get("tool_use_required"):
            return True
        
        # 如果启用了 multimath
        if observation.get("multimath_enabled"):
            return True
        
        # 几何问题关键词
        geometry_keywords = [
            'triangle', 'angle', 'congruent', 'similar', 'parallel',
            'perpendicular', 'circle', 'radius', 'diameter', 'area',
            'perimeter', 'pythagorean', 'theorem', 'proof', 'find',
            'calculate', 'prove', 'show that', 'verify'
        ]
        
        # 如果是几何问题
        if any(keyword in question for keyword in geometry_keywords):
            return True
        
        # 如果问题中包含数学符号
        math_symbols = ['∠', '△', '∼', '≅', '⊥', '∥', '°']
        if any(symbol in observation.get("question", "") for symbol in math_symbols):
            return True
        
        return False
    
    def _should_use_sam2(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用SAM2"""
        question = observation.get("question", "").lower()
        
        if self._is_pure_calculation(question):
            return False
        
        if not observation.get("has_image", False):
            return False
        
        # 医学图像分割相关关键词
        segmentation_keywords = [
            "region", "area", "location", "where", "identify",
            "segment", "boundary", "extent", "coverage",
            "affected", "normal", "abnormal", "lesion",
            "organ", "structure", "anatomy"
        ]
        
        # 特定器官关键词
        organ_keywords = [
            "brain", "lung", "heart", "liver", "kidney",
            "cerebral", "pulmonary", "cardiac", "hepatic", "renal"
        ]
        
        # 如果问题涉及分割或定位
        if any(keyword in question for keyword in segmentation_keywords):
            return True
        
        # 如果问题涉及特定器官且需要视觉分析
        if any(keyword in question for keyword in organ_keywords):
            if any(word in question for word in ["show", "where", "locate", "identify", "region"]):
                return True
        
        return False
    
    def _should_use_sympy_geometry(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用SymPy几何工具"""
        question = observation.get("question", "").lower()
        
        if self._is_pure_calculation(question):
            return False
        
        # 几何计算相关关键词
        geometry_calc_keywords = [
            'calculate', 'compute', 'find the angle', 'find the length',
            'what is the area', 'what is the perimeter', 'prove that',
            'show that', 'verify', 'pythagorean', 'triangle inequality',
            'inscribed angle', 'central angle', 'arc length', 'hypotenuse',
            'distance from point to line', 'triangle type'
        ]
        
        # 需要精确计算的几何问题
        if any(keyword in question for keyword in geometry_calc_keywords):
            return True
        
        # 如果问题包含具体的几何数值
        if re.search(r'\b\d+\s*(?:degrees?|°|cm|m|units?)\b', question):
            if any(word in question for word in ['angle', 'length', 'area', 'perimeter']):
                return True
        
        # 如果问题涉及几何定理
        theorem_keywords = ['theorem', 'proof', 'congruent', 'similar', 'bisector']
        if any(keyword in question for keyword in theorem_keywords):
            return True
        
        return False
    
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
            'congruent', 'similar', 'polygon', 'radius', 'diameter',
            'hypotenuse', 'pythagorean', 'inscribed', 'circumference'
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
        
        # 对于医学图像，如果需要定位后分割，可以使用
        if observation.get("is_medical_vqa", False) and self.enable_sam2:
            if any(word in question for word in ["locate", "find", "where", "identify"]):
                return True
        
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
        # 对于医学图像，使用更具体的提示
        if observation.get("is_medical_vqa", False):
            question = observation.get("question", "").lower()
            if "brain" in question:
                caption = "brain region"
            elif "lung" in question:
                caption = "lung area"
            elif "heart" in question:
                caption = "heart region"
            else:
                caption = "organ . anatomical structure"
        else:
            caption = "all objects"
        
        tool_call = {
            "tool": "grounding_dino",
            "parameters": {
                "caption": caption,
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