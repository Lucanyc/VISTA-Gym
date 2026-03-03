from typing import Dict, Any, Optional, List, Tuple
from .vlm_agent import VLMAgent
import json
import re
import os
from PIL import Image
import tempfile
import random
import hashlib  # 新增：用于稳定的哈希


class VLMAgentWithTools(VLMAgent):
    """
    VLM Agent with tool support for MathVista dataset
    Supports: DiagramFormalizer (geometry), GroundingDINO (object detection), ChartMoE (charts), DeepEyes (visual enhancement)
    Enhanced: Now supports Qwen-generated bbox for DeepEyes
    """
    
    # 定义工具索引映射
    TOOL_INDEX_MAP = {
        0: {
            "name": "deepeyes",  # 修改：统一使用 deepeyes 作为工具名
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
            "type": "text_recognition",
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
            "enable_deepeyes_tools",  # 添加 deepeyes
            "grounding_dino_config",
            "diagram_formalizer_config",
            "chartmoe_config",
            "deepeyes_config",  # 添加 deepeyes 配置
            "easycor_config",  # 添加 EasyOCR 配置
            "enable_tool_collaboration"  # 新增：多工具协同参数
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
        print(f"\n[VLMAgentWithTools.__init__] Initializing VLM Agent with Tools")
        print(f"  - Base config keys: {list(base_config.keys())}")
        print(f"  - Tool config keys: {list(tool_config.keys())}")
        
        # 工具使用相关配置（从tool_config中读取）
        self.enable_tools = tool_config.get("enable_tools", True)
        self.max_tool_calls = tool_config.get("max_tool_calls", 3)
        self.tool_selection_strategy = tool_config.get("tool_selection_strategy", "auto")
        self.tool_response_mode = tool_config.get("tool_response_mode", "auto")
        
        # 新增：多工具协同配置
        self.enable_tool_collaboration = tool_config.get("enable_tool_collaboration", False)
        self.enable_grounding_dino = tool_config.get("enable_grounding_dino", False)
        self.enable_chartmoe = tool_config.get("enable_chartmoe", False)
        self.enable_diagram_formalizer = tool_config.get("enable_diagram_formalizer", False)
        self.enable_deepeyes_tools = tool_config.get("enable_deepeyes_tools", False)
        self.easyocr_tool = tool_config.get("enable_easyocr", False)  # 添加 EasyOCR 配置
        
        self.enable_tool_collaboration = tool_config.get("enable_tool_collaboration", False)
        
    
        
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - max_tool_calls: {self.max_tool_calls}")
        print(f"  - enable_tool_collaboration: {self.enable_tool_collaboration}")
        
        # 跟踪工具使用历史
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}  # 记录每个问题的工具调用历史
        
        # 新增：工具上下文存储（用于多工具协同）
        self.tool_context = {}  # 存储每个工具的输出结果
        
        # ⭐ 新增：bbox相关状态
        self.current_bbox = None  # 存储当前的bbox
        self.bbox_history = []    # bbox历史记录
        
        # 新增：定义工具链规则（添加 DeepEyes 相关规则）
        self.tool_chain_rules = {
            "visual_detail": {
                "description": "Problems that need visual detail examination",
                "primary": "deepeyes",
                "secondary": "grounding_dino",
                "condition": r'\bsmall|tiny|detail|unclear|hard to see|zoom|closer\b',
                "examples": ["small text", "tiny objects", "unclear details", "zoom in"]
            },
            "geometry_counting": {
                "description": "Geometry problems that need counting",
                "primary": "diagram_formalizer",
                "secondary": "grounding_dino",
                "condition": r'\bhow many|count|number of|total\b',
                "examples": ["How many triangles", "Count the angles", "Total number of sides"]
            },
            "geometry_location": {
                "description": "Geometry problems that need location/position",
                "primary": "diagram_formalizer",
                "secondary": "grounding_dino",
                "condition": r'\bwhere|position|locate|which\s+(?:angle|vertex|side)\b',
                "examples": ["Where is angle ABC", "Which vertex", "Position of the center"]
            },
            "chart_identification": {
                "description": "Chart problems that need element identification",
                "primary": "chartmoe",
                "secondary": "grounding_dino",
                "condition": r'\bwhich\s+(?:bar|column|line|point)|where|position|locate\b',
                "examples": ["Which bar is highest", "Where is the peak", "Locate the minimum"]
            },
            "chart_counting": {
                "description": "Chart problems that need counting elements",
                "primary": "chartmoe",
                "secondary": "grounding_dino", 
                "condition": r'\bhow many\s+(?:bars|columns|lines|points)|count\b',
                "examples": ["How many bars", "Count the data points", "Number of columns"]
            }
        }
        
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
            "collaboration_chains": 0  # 新增：记录协同工具链使用次数
        }
        
        # 工具实例引用（将由环境或其他地方设置）
        self.grounding_dino_tool = None
        self.diagram_formalizer_tool = None
        self.chartmoe_tool = None
        self.deepeyes_tool = None  # 添加 DeepEyes 工具实例
        self.easyocr_tool = None  # 添加 EasyOCR 工具实例
        
        print(f"  - Tool instances initialized to None (will be set by environment)")
        
        # 打印工具链规则（如果启用协同）
        if self.enable_tool_collaboration:
            print(f"\n  - Tool Chain Rules enabled:")
            for rule_name, rule_config in self.tool_chain_rules.items():
                print(f"    * {rule_name}: {rule_config['primary']} → {rule_config['secondary']}")
                print(f"      Description: {rule_config['description']}")
    
    # ⭐ 新增方法：增强观察以包含bbox指令
    def _enhance_observation_with_bbox_instruction(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """在观察中添加bbox生成指令（完全参考独立脚本）"""
        # 只在有图像且可能需要bbox的情况下添加
        if not observation.get("has_image", False):
            return observation
        
        enhanced = observation.copy()
        
        # 使用独立脚本中的prompt格式
        question = observation.get("question", "")
        choices = observation.get("choices", [])
        
        if choices:
            labeled_choices = [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
            choice_text = "\n".join(labeled_choices)
            
            bbox_prompt = f"""Question: {question}

    Options:
    {choice_text}

    Please analyze this step by step:
    1. First, describe your reasoning in <think> tags
    2. If you need to focus on a specific region, provide coordinates in <box>[x1,y1,x2,y2]</box> format
    3. Finally, provide your answer (just the letter, e.g., "A") in <answer> tags

    Example format:
    <think>I need to analyze the image carefully...</think>
    <box>[100,200,300,400]</box>
    <answer>A</answer>"""
        else:
            # 非选择题格式
            bbox_prompt = f"""Question: {question}

    Please analyze this step by step:
    1. First, describe your reasoning in <think> tags
    2. If you need to focus on a specific region, provide coordinates in <box>[x1,y1,x2,y2]</box> format
    3. Finally, provide your answer in <answer> tags

    Example format:
    <think>I need to analyze the image carefully...</think>
    <box>[100,200,300,400]</box>
    <answer>Your answer here</answer>"""
        
        enhanced["output_format_instruction"] = bbox_prompt
        enhanced["bbox_instruction_added"] = True
        return enhanced
    
    # ⭐ 新增方法：解析响应中的bbox
    def _parse_bbox_from_response(self, response: str) -> Optional[List[int]]:
        """从响应中解析bbox（增强容错性）"""
        import re
        
        print(f"  - [DEBUG] Parsing bbox from response: {response[:200]}...")
        
        # 修复各种可能的bbox格式错误
        response = re.sub(r'<box\[(\d+),(\d+),(\d+),(\d+)\]></box>', r'<box>[\1,\2,\3,\4]</box>', response)
        # 修复缺少斜杠的情况
        response = re.sub(r'<box>\[(\d+),(\d+),(\d+),(\d+)\]box>', r'<box>[\1,\2,\3,\4]</box>', response)
        # 修复其他变体
        response = re.sub(r'<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*box>', r'<box>[\1,\2,\3,\4]</box>', response)
        
        # 标准解析
        bbox_pattern = r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</box>'
        bbox_match = re.search(bbox_pattern, response)
        
        if bbox_match:
            bbox = [int(x) for x in bbox_match.groups()]
            print(f"  - Found bbox: {bbox}")
            return bbox
        
        # 如果标准解析失败，尝试更宽松的模式
        # 匹配 <box>[数字,数字,数字,数字] 后面跟任意内容
        loose_pattern = r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        loose_match = re.search(loose_pattern, response)
        
        if loose_match:
            bbox = [int(x) for x in loose_match.groups()]
            print(f"  - Found bbox with loose pattern: {bbox}")
            return bbox
        
        print(f"  - No valid bbox found")
        return None
    
    # ⭐ 新增方法：使用指定的bbox构建DeepEyes调用
    def _build_deepeyes_prompt_with_bbox(self, bbox: List[int]) -> str:
        """使用指定的bbox构建DeepEyes调用"""
        tool_call = {
            "name": "image_zoom_in_tool",
            "arguments": {
                "bbox_2d": bbox
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        根据观察生成action - 支持多工具协同版本和bbox生成
        返回 (action_string, extra_info_dict)
        """
        # 确保模型已加载（使用父类的 load_model 方法）
        self.load_model()
        
        # 添加调试输出
        print(f"\n[VLMAgentWithTools.act] DEBUG START")
        print(f"  - Observation keys: {list(observation.keys())}")
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - enable_tool_collaboration: {self.enable_tool_collaboration}")
        print(f"  - Current tool calls: {self.current_tool_calls}")
        print(f"  - Tool context: {list(self.tool_context.keys())}")
        print(f"  - Current bbox: {self.current_bbox}")
        
        # 检查是否是新的episode
        current_question = observation.get("question", "")
        if observation.get("episode_start", False) or self._is_new_task(observation):
            self.current_tool_calls = 0
            self.conversation_history = []
            self.tool_call_history = {}  # 重置工具历史
            self.tool_context = {}  # 重置工具上下文（重要！）
            self.current_bbox = None  # ⭐ 重置bbox
            self.bbox_history = []
            print(f"  - Starting new task, reset tool history, context and bbox")
        
        # 记录当前交互
        self.conversation_history.append({"role": "observation", "content": observation})
        
        # ⭐ 关键：检查是否是错误答案后的重试
        if observation.get("previous_attempt_failed", False):
            print(f"  - Previous attempt failed")
            
            # 如果有bbox且还没使用DeepEyes，自动使用
            if self.current_bbox and self.enable_deepeyes_tools and "deepeyes" not in self.tool_context:
                print(f"  - Have bbox {self.current_bbox}, will use DeepEyes for zoom")
                
                # 直接生成DeepEyes调用
                deepeyes_call = self._build_deepeyes_prompt_with_bbox(self.current_bbox)
                
                self.current_tool_calls += 1
                self.tool_use_stats["total_calls"] += 1
                
                extra_info = {
                    "action_type": "tool_call",
                    "tool_call_count": self.current_tool_calls,
                    "tool_used": "deepeyes",
                    "reason": "Using DeepEyes after incorrect answer with bbox",
                    "bbox_used": self.current_bbox,
                    "log_tool_usage": True
                }
                
                print(f"  - Generated DeepEyes call with bbox: {self.current_bbox}")
                return deepeyes_call, extra_info
        
        # === 协同模式特殊处理 ===
        if self.enable_tool_collaboration and self.enable_tools:
            # 检查是否是环境返回的工具反馈
            if observation.get("requires_response") and "tool_feedback" in observation:
                print(f"  - Handling tool feedback in collaboration mode")
                
                # ⭐ 特殊处理：如果是DeepEyes反馈，准备重新分析
                tool_feedback = observation.get("tool_feedback", {})
                if tool_feedback.get("tool") == "deepeyes" and tool_feedback.get("success"):
                    print(f"  - DeepEyes feedback received, preparing final answer")
                    
                    # ⭐ 保存DeepEyes处理后的图像
                    if "image" in tool_feedback:
                        try:
                            import os
                            from PIL import Image
                            
                            save_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/scienceqa_deepeyes/"
                            os.makedirs(save_dir, exist_ok=True)
                            
                            task_id = observation.get("task_id", "unknown")
                            attempt = observation.get("attempt", 1)
                            tool_used = tool_feedback.get("tool_used", "unknown")
                            
                            filename = f"{task_id}_attempt{attempt}_deepeyes_{tool_used}.png"
                            save_path = os.path.join(save_dir, filename)
                            
                            processed_image = tool_feedback["image"]
                            if isinstance(processed_image, Image.Image):
                                processed_image.save(save_path)
                                print(f"  - Saved DeepEyes processed image to: {save_path}")
                        except Exception as e:
                            print(f"  - ERROR saving DeepEyes image: {str(e)}")
                    
                    # 为重新分析准备观察（移除bbox指令）
                    clean_obs = observation.copy()
                    clean_obs["output_format_instruction"] = """Based on this zoomed image, answer the question.

    You MUST output your answer inside <answer> tags.
    For multiple choice, just put the letter: <answer>A</answer>"""
                    
                    # 生成基于缩放图像的答案
                    response, base_info = super().act(clean_obs)
                    
                    # 确保格式正确
                    response = self._ensure_answer_format(response, observation)
                    
                    extra_info = base_info.copy()
                    extra_info.update({
                        "action_type": "final_answer_after_deepeyes",
                        "used_deepeyes": True,
                        "stage": "final_answer"
                    })
                    
                    return response, extra_info
                
                # 处理其他工具反馈
                response, extra_info = self._handle_tool_feedback(observation)
                
                # 检查是否需要继续工具收集
                if extra_info.get("ready_for_answer", False):
                    # 所有工具收集完毕，生成最终答案
                    print(f"  - Tool collection complete, building final answer")
                    return self._build_final_answer(observation)
                else:
                    # 继续工具收集阶段
                    return response, extra_info
            
            # 如果不是工具反馈，检查是否需要开始工具收集
            available_tools = self._get_available_tools(observation)
            if available_tools and self._analyze_tool_need(observation):
                # Stage A: 开始工具收集
                next_tool = self.decide_tool(observation)
                if next_tool:
                    print(f"  - Starting tool collection with first tool")
                    self.current_tool_calls += 1
                    self.tool_use_stats["total_calls"] += 1
                    
                    extra_info = {
                        "action_type": "tool_call",
                        "stage": "tool_collection_start",
                        "tool_call_count": self.current_tool_calls,
                        "tools_planned": self._get_planned_tools(observation),
                        "log_tool_usage": True
                    }
                    return next_tool, extra_info
        
        # === 非协同模式或不需要工具的处理 ===
        
        # 检查是否是环境返回的工具反馈（非协同模式）
        if observation.get("requires_response") and "tool_feedback" in observation:
            print(f"  - Handling tool feedback from environment (non-collaboration mode)")
            
            # ⭐ 特殊处理DeepEyes反馈
            tool_feedback = observation.get("tool_feedback", {})
            if tool_feedback.get("tool") == "deepeyes" and tool_feedback.get("success"):
                print(f"  - DeepEyes feedback, generating final answer based on zoomed image")
                
                # ⭐ 保存DeepEyes处理后的图像
                if "image" in tool_feedback:
                    try:
                        import os
                        from PIL import Image
                        
                        save_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/scienceqa_deepeyes/"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        task_id = observation.get("task_id", "unknown")
                        attempt = observation.get("attempt", 1)
                        tool_used = tool_feedback.get("tool_used", "unknown")
                        
                        filename = f"{task_id}_attempt{attempt}_deepeyes_{tool_used}.png"
                        save_path = os.path.join(save_dir, filename)
                        
                        processed_image = tool_feedback["image"]
                        if isinstance(processed_image, Image.Image):
                            processed_image.save(save_path)
                            print(f"  - Saved DeepEyes processed image to: {save_path}")
                    except Exception as e:
                        print(f"  - ERROR saving DeepEyes image: {str(e)}")
                
                # 准备干净的观察
                clean_obs = observation.copy()
                clean_obs["output_format_instruction"] = """Based on this zoomed image, answer the question.

    You MUST output your answer inside <answer> tags.
    For multiple choice, just put the letter: <answer>A</answer>"""
                
                # 生成答案
                response, base_info = super().act(clean_obs)
                response = self._ensure_answer_format(response, observation)
                
                extra_info = base_info.copy()
                extra_info.update({
                    "action_type": "final_answer_after_deepeyes",
                    "stage": "final"
                })
                
                return response, extra_info
            
            # 处理其他工具反馈
            response, extra_info = self._handle_tool_feedback(observation)
            response = self._ensure_answer_format(response, observation)
            return response, extra_info
        
        # ⭐ 对于初次尝试且有图像，添加bbox指令
        if self.current_tool_calls == 0 and observation.get("has_image", False) and not observation.get("bbox_instruction_added", False):
            observation = self._enhance_observation_with_bbox_instruction(observation)
            print(f"  - Enhanced observation with bbox instruction")
        
        # 获取可用工具列表
        available_tools = self._get_available_tools(observation)
        print(f"  - Available tools: {available_tools}")
        
        # 检查是否启用工具
        if not self.enable_tools or not available_tools:
            print(f"  - Tools disabled or no tools available, generating direct answer")
            response, extra_info = self._generate_direct_answer(observation)
            
            # ⭐ 解析响应中的bbox
            bbox = self._parse_bbox_from_response(response)
            if bbox:
                print(f"  - Found bbox in response: {bbox}")
                self.current_bbox = bbox
                self.bbox_history.append(bbox)
                extra_info["bbox_found"] = True
                extra_info["bbox"] = bbox
            
            # 确保格式正确
            response = self._ensure_answer_format(response, observation)
            return response, extra_info
        
        # 检查是否达到工具调用上限
        if self.current_tool_calls >= self.max_tool_calls:
            print(f"  - Reached max tool calls ({self.max_tool_calls}), forcing final answer")
            # 如果是协同模式且有工具上下文，使用 _build_final_answer
            if self.enable_tool_collaboration and self.tool_context:
                return self._build_final_answer(observation)
            else:
                response, extra_info = self._generate_forced_final_answer(observation)
                response = self._ensure_answer_format(response, observation)
                return response, extra_info
        
        # 分析是否需要使用工具
        tool_needed = self._analyze_tool_need(observation)
        
        if tool_needed:
            print(f"  - Tool needed for this task")
            response, extra_info = self._generate_tool_call(observation, available_tools)
        else:
            print(f"  - No tool needed, generating direct answer")
            response, extra_info = self._generate_direct_answer(observation)
            
            # ⭐ 解析bbox
            bbox = self._parse_bbox_from_response(response)
            if bbox:
                print(f"  - Found bbox in response: {bbox}")
                self.current_bbox = bbox
                self.bbox_history.append(bbox)
                extra_info["bbox_found"] = True
                extra_info["bbox"] = bbox
            
            # 确保格式正确（除非是工具调用）
            if not response.startswith("<tool_call>"):
                response = self._ensure_answer_format(response, observation)
        
        print(f"[VLMAgentWithTools.act] DEBUG END\n")
        return response, extra_info
    
    
    
    def _get_planned_tools(self, observation: Dict[str, Any]) -> List[str]:
        """预测可能需要的工具序列（用于调试）"""
        question = observation.get("question", "").lower()
        planned = []
        
        # 检查是否匹配工具链规则
        for rule_name, rule in self.tool_chain_rules.items():
            if re.search(rule["condition"], question):
                planned.append(rule["primary"])
                planned.append(rule["secondary"])
                break
        
        # 如果没有匹配的规则，基于单个工具判断
        if not planned:
            if self._is_geometry_problem(question):
                planned.append("diagram_formalizer")
            elif self._should_use_chartmoe(observation):
                planned.append("chartmoe")
            elif self._should_use_grounding_dino(observation):
                planned.append("grounding_dino")
            elif self._should_use_deepeyes(observation):
                planned.append("deepeyes")
        
        return planned
    
    
    def decide_tool(self, observation: Dict[str, Any]) -> Optional[str]:
        """决定下一个需要使用的工具 - 基于规则的工具链决策"""
        question = observation.get("question", "").lower()
        tc = self.tool_context  # 已经使用过的工具
        
        print(f"\n[decide_tool] Current tool context: {list(tc.keys())}")
        print(f"[decide_tool] Current tool calls: {self.current_tool_calls}/{self.max_tool_calls}")
        print(f"[decide_tool] Current bbox: {self.current_bbox}")
        
        # ⭐ 如果有bbox且还没使用过DeepEyes，优先使用DeepEyes
        if self.current_bbox and "deepeyes" not in tc:
            print(f"[decide_tool] Model generated bbox available, using DeepEyes")
            return self._build_deepeyes_prompt(observation)
        
        # 检查是否达到工具调用上限
        if self.current_tool_calls >= self.max_tool_calls:
            print(f"[decide_tool] Reached max tool calls limit")
            return None
        
        # 检查是否是纯计算问题
        if self._is_pure_calculation(question):
            print(f"[decide_tool] Pure calculation detected, no tools needed")
            return None
        
        # 1. 检查工具链规则
        for rule_name, rule in self.tool_chain_rules.items():
            if re.search(rule["condition"], question):
                print(f"[decide_tool] Matched rule: {rule_name}")
                
                # 检查主工具
                primary_tool = rule["primary"]
                if primary_tool not in tc:
                    print(f"[decide_tool] Primary tool '{primary_tool}' not used yet, selecting it")
                    # 根据工具名返回相应的构建函数
                    if primary_tool == "deepeyes":
                        return self._build_deepeyes_prompt(observation)
                    elif primary_tool == "diagram_formalizer":
                        return self._build_diagram_formalizer_prompt(observation)
                    elif primary_tool == "chartmoe":
                        return self._build_chartmoe_prompt(observation)
                    elif primary_tool == "grounding_dino":
                        return self._build_grounding_dino_prompt(observation)
                
                # 主工具已使用，检查次要工具
                secondary_tool = rule["secondary"]
                if secondary_tool not in tc:
                    # 额外检查：是否真的需要次要工具
                    if self._should_use_secondary_tool(primary_tool, tc[primary_tool], question):
                        print(f"[decide_tool] Secondary tool '{secondary_tool}' needed after {primary_tool}")
                        if secondary_tool == "grounding_dino":
                            return self._build_grounding_dino_prompt(observation)
                        elif secondary_tool == "diagram_formalizer":
                            return self._build_diagram_formalizer_prompt(observation)
                        elif secondary_tool == "chartmoe":
                            return self._build_chartmoe_prompt(observation)
                        elif secondary_tool == "deepeyes":
                            return self._build_deepeyes_prompt(observation)
                
                # 该规则的工具链已完成
                print(f"[decide_tool] Tool chain for rule '{rule_name}' completed")
                break
        
        # 2. 如果没有匹配的工具链规则，检查单个工具需求
        if not tc:  # 还没有使用任何工具
            print(f"[decide_tool] No tools used yet, checking single tool needs")
            
            # 视觉增强优先（如果需要查看细节）
            if self._should_use_deepeyes(observation) and "deepeyes" not in tc:
                print(f"[decide_tool] Visual enhancement needed, using DeepEyes")
                return self._build_deepeyes_prompt(observation)
            
            # 几何问题优先
            if self._is_geometry_problem(question) and "diagram_formalizer" not in tc:
                print(f"[decide_tool] Geometry problem detected, using DiagramFormalizer")
                return self._build_diagram_formalizer_prompt(observation)
            
            # 图表问题
            if self._should_use_chartmoe(observation) and "chartmoe" not in tc:
                print(f"[decide_tool] Chart problem detected, using ChartMoE")
                return self._build_chartmoe_prompt(observation)
            
            # 纯视觉检测
            if self._should_use_grounding_dino(observation) and "grounding_dino" not in tc:
                print(f"[decide_tool] Visual detection needed, using GroundingDINO")
                return self._build_grounding_dino_prompt(observation)
        
        # 3. 没有更多工具需要使用
        print(f"[decide_tool] No more tools needed, tool collection complete")
        return None

    
    
    def _ensure_answer_format(self, response: str, observation: Dict[str, Any]) -> str:
        """确保响应包含正确的格式（使用<think>标签）"""
        # 检查是否已有正确格式
        if "<answer>" in response and "</answer>" in response:
            return response
        
        # 如果是工具调用，不需要修改
        if "<tool_call>" in response:
            return response
        
        # 对于包含think的响应，保持原样
        if "<think>" in response and "</think>" in response:
            if "<answer>" not in response:
                answer = self._extract_answer_from_content(response, observation)
                return response + f"\n<answer>{answer}</answer>"
            else:
                return response
        
        # 构建带think和answer标签的响应
        answer = self._extract_answer_from_content(response, observation)
        return f"""<think>
    {response}
    </think>
    <answer>{answer}</answer>"""
    
    
    def _extract_answer_from_content(self, response: str, observation: Dict[str, Any]) -> str:
        """从响应内容中提取答案 - 宽松版本"""
        question = observation.get("question", "").lower()
        
        # 提取数字答案（更宽松的匹配）
        if any(kw in question for kw in ["how many", "count", "number", "total", "sum", "difference"]):
            # 匹配整数和小数
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
            if numbers:
                # 优先选择较大的数字（通常是最终答案而不是中间步骤）
                # 但如果问题是关于差值，可能需要较小的数字
                if "difference" in question:
                    return numbers[-1]  # 最后出现的数字
                else:
                    # 对于计数问题，通常最后的数字是答案
                    return numbers[-1]
        
        # 提取选择题答案
        if "choices" in observation:
            # 查找 A, B, C, D, E（更宽松，包括小写）
            choices = re.findall(r'\b[A-Ea-e]\b', response)
            if choices:
                # 返回最后出现的选项（转为大写）
                return choices[-1].upper()
        
        # 提取 Yes/No 答案
        if any(kw in question for kw in ["is", "are", "does", "do", "can", "will", "would", "should"]):
            response_lower = response.lower()
            # 更宽松的匹配
            if any(yes_word in response_lower for yes_word in ["yes", "correct", "true", "affirmative"]):
                return "Yes"
            elif any(no_word in response_lower for no_word in ["no", "incorrect", "false", "negative", "not"]):
                return "No"
        
        # 对于其他类型，尝试提取最后一句话或关键信息
        # 分句
        sentences = re.split(r'[.!?]+', response.strip())
        if sentences:
            # 返回最后一个非空句子
            for sentence in reversed(sentences):
                if sentence.strip():
                    return sentence.strip()
        
        # 默认：返回整个响应的核心部分
        return response.strip()[:100]  # 限制长度
    
    
    def _is_pure_calculation(self, question: str) -> bool:
        """判断是否是纯计算问题（不需要视觉分析）"""
        question_lower = question.lower()
        
        # 纯计算的特征 - 改进的正则表达式
        calculation_indicators = [
            r'(?:^|\s)\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?',  # 支持小数和前面有文字
            r'what is \d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?',  # What is 5.5 + 3.2
            r'calculate:?\s*\d+(?:\.\d+)?',  # Calculate: 123.45
            r'solve:?\s*[\d\.\+\-\*/\s]+',  # Solve: 12.5 * 34
            r'evaluate:?\s*[\d\.\+\-\*/\s]+',  # Evaluate: 56.7 / 7
            r'compute:?\s*[\d\.\+\-\*/\s]+',  # Compute: 78 - 23.5
            r'simplify:?\s*[\d\.\+\-\*/\s\(\)]+',  # Simplify: (12 + 8) * 3
        ]
        
        # 如果匹配纯计算模式
        if any(re.search(pattern, question_lower) for pattern in calculation_indicators):
            # 再检查是否有视觉相关词汇
            visual_words = ["image", "picture", "shown", "see", "diagram", "figure", "graph", "chart", "above", "below", "display", "illustrat"]
            if not any(word in question_lower for word in visual_words):
                return True
        
        # 检查是否是文字题但实际是纯计算
        word_problem_calc = [
            r'if .* has \d+(?:\.\d+)? .* and .* has \d+(?:\.\d+)?',  # If John has 5.5 apples and Mary has 3
            r'total of \d+(?:\.\d+)? and \d+(?:\.\d+)?',  # Total of 15.5 and 25
            r'sum of \d+(?:\.\d+)? and \d+(?:\.\d+)?',  # Sum of 10 and 20.5
            r'difference between \d+(?:\.\d+)? and \d+(?:\.\d+)?',  # Difference between 100.5 and 45
            r'product of \d+(?:\.\d+)? and \d+(?:\.\d+)?',  # Product of 12.5 and 15
            r'\d+(?:\.\d+)?\s*(?:plus|minus|times|divided by)\s*\d+(?:\.\d+)?',  # 5 plus 3, 10 times 4
        ]
        
        if any(re.search(pattern, question_lower) for pattern in word_problem_calc):
            # 这些通常是纯计算，除非明确提到图像
            if not any(word in question_lower for word in ["image", "picture", "diagram", "shown", "graph", "chart"]):
                return True
        
        return False
    
    
    def _should_use_grounding_dino(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用 GroundingDINO - 改进版本"""
        question = observation.get("question", "").lower()
        task = observation.get("task", "")
        
        # 第一步：快速检查是否是纯计算问题
        if self._is_pure_calculation(question):
            print(f"  - Pure calculation detected, skipping Grounding DINO")
            return False
        
        # 获取 MathVista 的具体任务类型（如果有）
        task_type = observation.get("task_type", "").lower()
        problem_type = observation.get("problem_type", "").lower()
        
        # 第二步：检查任务类型 - 明确不需要视觉检测的任务
        no_detection_tasks = [
            "arithmetic reasoning",  # 纯数学计算
            "algebraic reasoning",   # 代数方程
            "logical reasoning",     # 逻辑推理
        ]
        
        if any(no_task in task_type for no_task in no_detection_tasks):
            print(f"  - Task type '{task_type}' does not require object detection")
            return False
        
        # 第三步：积极匹配需要检测的任务类型
        detection_positive_tasks = [
            "numeric commonsense",    # 常识比较、计数
            "statistical reasoning",  # 图表分析
            "geometry reasoning",     # 几何问题（配合 DiagramFormalizer）
        ]
        
        if any(pos_task in task_type for pos_task in detection_positive_tasks):
            print(f"  - Task type '{task_type}' typically benefits from object detection")
            # 但仍需要进一步检查不是纯计算
        
        # 第四步：对 scientific reasoning 特殊处理
        if "scientific reasoning" in task_type:
            # 只有涉及装置、实验器材、位置关系时才使用 - 修复：全部小写
            science_visual_keywords = [
                "apparatus", "equipment", "device", "instrument",
                "position", "location", "where", "which part",
                "measure", "reading", "scale", "gauge",
                "setup", "experiment", "observation"
            ]
            if any(keyword in question for keyword in science_visual_keywords):
                print(f"  - Scientific reasoning with visual elements detected")
                return True
            else:
                print(f"  - Scientific reasoning without visual dependency")
                return False
        
        # 第五步：基于问题内容的精确匹配
        # 明确需要检测的模式
        detection_required_patterns = [
            # 计数类 - 但要排除纯数字计算
            (r'\bhow many\b(?!.*[\+\-\*/=])', "counting objects"),  # how many 但不是 how many is 2+3
            (r'\bcount\b(?:.*(?:in the|on the|within))', "counting in image"),
            (r'\bnumber of\b(?:.*(?:in the|shown|visible))', "number of visible items"),
            
            # 定位类
            (r'\bwhere\s+(?:is|are)\b', "location query"),
            (r'\blocate\b', "locate objects"),
            (r'\bfind\b(?:.*in the (?:image|picture|diagram))', "find in image"),
            (r'\bposition of\b', "position query"),
            
            # 比较类（视觉）
            (r'\bwhich\s+(?:bar|column|line|object)\b', "visual comparison"),
            (r'\b(?:largest|smallest|biggest|tallest|shortest)\b(?:.*(?:in the|shown))', "size comparison"),
            (r'\b(?:leftmost|rightmost|topmost|bottommost)\b', "spatial comparison"),
            
            # 图表特定
            (r'\b(?:bar|column|pie|line)\s*(?:chart|graph)\b.*(?:which|what|how)', "chart analysis"),
            (r'\bpeak\b|\bmaximum\b|\bminimum\b(?:.*graph)', "graph extrema"),
            
            # 几何图形检测
            (r'\b(?:triangle|circle|square|rectangle|polygon)\b(?!.*formula)', "geometric shapes"),
            (r'\b(?:angle|vertex|edge|side)\b.*(?:marked|shown|labeled)', "geometric elements"),
        ]
        
        for pattern, reason in detection_required_patterns:
            if re.search(pattern, question):  # 已经是 lowercase，不需要 IGNORECASE
                print(f"  - Detection needed: {reason}")
                return True
        
        # 第六步：检查是否有图像且问题确实涉及图像内容
        has_image = observation.get("has_image", False)
        if has_image:
            # 检查问题是否引用图像
            image_reference_patterns = [
                r'\bin the (?:image|picture|diagram|figure)\b',
                r'\bshown\b',
                r'\bsee\b',
                r'\bvisible\b',
                r'\babove\b(?!.*equation)',  # above 但不是 "solve the above equation"
                r'\bgiven\s+(?:image|picture|diagram)\b',
            ]
            
            if any(re.search(pattern, question) for pattern in image_reference_patterns):
                # 问题引用了图像，进一步检查是否需要检测
                # 排除纯 OCR 或符号识别任务
                if not any(word in question for word in ["equation", "formula", "expression", "solve for"]):
                    print(f"  - Question references image content")
                    return True
        
        print(f"  - No clear indication for object detection")
        return False
    
    
    def _should_use_deepeyes(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用 DeepEyes 工具"""
        question = observation.get("question", "").lower()
        
        # 第一步：检查是否是纯计算问题
        if self._is_pure_calculation(question):
            print(f"  - Pure calculation detected, skipping DeepEyes")
            return False
        
        # 检查是否有图像
        has_image = observation.get("has_image", False)
        if not has_image:
            return False
        
        # ⭐ 最重要：如果已经有bbox，应该使用DeepEyes
        if self.current_bbox:
            print(f"  - Have bbox, should use DeepEyes for zoom")
            return True
        
        # ⭐ 新增：如果是第一次尝试（没有bbox），只在有明确DeepEyes需求时使用
        # 避免过早使用DeepEyes，让模型先有机会生成bbox
        if self.current_tool_calls == 0 and not self.current_bbox:
            # 只检查明确需要视觉增强的关键词
            explicit_deepeyes_keywords = [
                "zoom", "zoom in", "zoom into", "zoomed",
                "detail", "detailed", "details", "closer look",
                "small", "tiny", "hard to see", "can't see",
                "unclear", "blurry", "fuzzy",
                "magnify", "enlarge", "enhance",
                "rotate", "rotated", "angle", "turn",
                "examine closely", "look closely", "clearer view",
                "fine print", "small text", "tiny text"
            ]
            
            # 检查是否有明确的DeepEyes需求
            has_explicit_need = any(keyword in question for keyword in explicit_deepeyes_keywords)
            
            if has_explicit_need:
                print(f"  - Explicit DeepEyes keywords detected in question")
                return True
            else:
                # 第一次尝试且没有明确需求，不使用DeepEyes
                # 让模型先尝试回答并可能生成bbox
                print(f"  - First attempt without explicit DeepEyes need, skip to allow bbox generation")
                return False
        
        # 如果不是第一次尝试，检查是否是重试后需要视觉增强的情况
        if observation.get("previous_attempt_failed", False):
            # 检查是否是视觉相关的失败（不是格式错误）
            if "format error" not in str(observation.get("error", "")):
                # 对于某些类型的问题，失败后可能需要视觉增强
                visual_problem_keywords = [
                    "map", "diagram", "chart", "graph",
                    "position", "location", "direction",
                    "identify", "recognize", "distinguish",
                    "compare", "difference", "similar"
                ]
                
                if any(keyword in question for keyword in visual_problem_keywords):
                    print(f"  - Visual problem failed, DeepEyes might help")
                    return True
        
        # 检查是否是需要仔细观察的任务类型
        task_type = observation.get("task_type", "").lower()
        if any(task in task_type for task in ["fine-grained", "detail", "microscopic"]):
            print(f"  - Task type suggests visual enhancement needed")
            return True
        
        # 默认不使用
        return False
    
    
    
    def _analyze_tool_need(self, observation: Dict[str, Any]) -> bool:
        """分析是否需要使用工具 - 改进版本"""
        question = observation.get("question", "").lower()
        task = observation.get("task", "")
        task_type = observation.get("task_type", "").lower()
        
        # 首先检查是否是纯计算（这是主要调用点）
        if self._is_pure_calculation(question):
            print(f"  - Pure calculation problem, no tools needed")
            return False
        
        # 视觉增强问题 - 需要 DeepEyes
        if self._should_use_deepeyes(observation):
            return True
        
        # 几何问题 - 通常需要 DiagramFormalizer
        if task == "geometry problem solving" or self._is_geometry_problem(question):
            # 但如果是纯几何计算（如已知边长求面积），可能不需要工具
            geometry_calc_patterns = [
                r'area.*=.*\d+.*\*.*\d+',  # area = 5 * 3
                r'perimeter.*=.*\d+.*\+',  # perimeter = 2 + 3 + 4
                r'volume.*=.*\d+',         # volume = length * width * height
            ]
            if any(re.search(pattern, question) for pattern in geometry_calc_patterns):
                print(f"  - Geometry calculation with formula, might not need tools")
                return False
            return True
        
        # 需要对象检测的问题
        if self._should_use_grounding_dino(observation):
            return True
        
        # 图表分析问题
        if self._should_use_chartmoe(observation):
            return True
        
        # 检查是否是重试且之前失败了
        if observation.get("previous_attempt_failed", False):
            # 但如果是格式错误，不需要工具
            if "format error" in str(observation.get("error", "")):
                return False
            return True
        
        # 对于某些任务类型，明确不需要工具
        no_tool_needed = [
            "arithmetic reasoning",
            "algebraic reasoning", 
            "logical reasoning",
        ]
        
        if any(task_name in task_type for task_name in no_tool_needed):
            print(f"  - Task type '{task_type}' typically doesn't need tools")
            return False
        
        return False
    
    
    def _generate_tool_call(self, observation: Dict[str, Any], available_tools: List[int]) -> Tuple[str, Dict[str, Any]]:
        """生成工具调用 - 带去重检查"""
        question = observation.get("question", "")
        task = observation.get("task", "")
        
        # 双重保险：即使前面的检查通过了，这里再检查一次
        if self._is_pure_calculation(question):
            print(f"  - Last check: pure calculation detected, falling back to direct answer")
            return self._generate_direct_answer(observation)
        
        # 使用稳定的哈希作为问题标识
        question_key = hashlib.md5(question.encode()).hexdigest()
        
        # 添加 DeepEyes 支持
        if self._should_use_deepeyes(observation) and 0 in available_tools:
            tool_name = "deepeyes"
            # 检查是否已调用过
            if question_key in self.tool_call_history and tool_name in self.tool_call_history[question_key]:
                print(f"  - Tool {tool_name} already called for this question, skipping")
                return self._generate_direct_answer(observation)
            
            tool_call = self._build_deepeyes_prompt(observation)
            
        elif task == "geometry problem solving" and 1 in available_tools:
            tool_name = "diagram_formalizer"
            # 检查是否已调用过
            if question_key in self.tool_call_history and tool_name in self.tool_call_history[question_key]:
                print(f"  - Tool {tool_name} already called for this question, skipping")
                return self._generate_direct_answer(observation)
            
            tool_call = self._build_diagram_formalizer_prompt(observation)
            
        elif self._should_use_grounding_dino(observation) and 2 in available_tools:
            tool_name = "grounding_dino"
            # 检查是否已调用过
            if question_key in self.tool_call_history and tool_name in self.tool_call_history[question_key]:
                print(f"  - Tool {tool_name} already called for this question")
                # 检查是否有结果可用
                if "grounding_dino_results" in observation:
                    print(f"  - Using cached results: {observation['grounding_dino_results'].get('num_detections', 0)} detections")
                    return self._handle_cached_grounding_dino_results(observation)
                return self._generate_direct_answer(observation)
            
            tool_call = self._build_grounding_dino_prompt(observation)
            
        elif self._should_use_chartmoe(observation) and 3 in available_tools:
            tool_name = "chartmoe"
            if question_key in self.tool_call_history and tool_name in self.tool_call_history[question_key]:
                print(f"  - Tool {tool_name} already called for this question, skipping")
                return self._generate_direct_answer(observation)
            
            tool_call = self._build_chartmoe_prompt(observation)
        else:
            # 如果没有合适的工具，生成直接答案
            return self._generate_direct_answer(observation)
        
        # 记录工具调用
        if question_key not in self.tool_call_history:
            self.tool_call_history[question_key] = set()
        self.tool_call_history[question_key].add(tool_name)
        
        # 更新工具调用计数
        self.current_tool_calls += 1
        self.tool_use_stats["total_calls"] += 1
        
        # 构建额外信息
        extra_info = {
            "action_type": "tool_call",
            "tool_call_count": self.current_tool_calls,
            "tool_used": tool_name,
            "reason": f"Using {tool_name} for {task or 'task'}",
            "first_call": True,  # 标记这是首次调用
            "log_tool_usage": True  # 提示环境记录工具使用
        }
        
        print(f"  - Tool call generated: {tool_name} (first call)")
        print(f"[VLMAgentWithTools.act] DEBUG END\n")
        
        return tool_call, extra_info
    
    
    def _handle_cached_grounding_dino_results(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理缓存的 Grounding DINO 结果"""
        results = observation.get("grounding_dino_results", {})
        num_detections = results.get("num_detections", 0)
        
        response = f"""<think>
Using previously detected results: {num_detections} objects found.
</think>
<answer>{num_detections}</answer>"""
        
        extra_info = {
            "action_type": "cached_tool_result",
            "tool_used": "grounding_dino",
            "used_cache": True
        }
        
        return response, extra_info
    
    
    def _get_available_tools(self, observation: Dict[str, Any]) -> List[int]:
        """获取当前可用的工具索引"""
        available = []
        
        # 方法1：检查 available_tools 列表
        available_tools_list = observation.get("available_tools", [])
        if "deepeyes" in available_tools_list:  # 修改：使用统一的 deepeyes 名称
            available.append(0)
        if "grounding_dino" in available_tools_list:
            available.append(2)
        if "diagram_formalizer" in available_tools_list:
            available.append(1)
        if "chartmoe" in available_tools_list:
            available.append(3)
        
        # 方法2：检查历史记录（备选）
        if not available and "grounding_dino_history" in observation:
            available.append(2)  # 如果有历史记录，说明工具可用
        
        print(f"[DEBUG _get_available_tools] available_tools_list: {available_tools_list}")
        print(f"[DEBUG _get_available_tools] returned indices: {available}")
        
        return available
    
    
    def _is_geometry_problem(self, question: str) -> bool:
        """判断是否是几何问题"""
        geometry_keywords = [
            'angle', 'degree', '°', 'triangle', 'circle', 'square', 'rectangle',
            'perimeter', 'area', 'volume', 'parallel', 'perpendicular',
            'congruent', 'similar', 'polygon', 'radius', 'diameter'
        ]
        return any(keyword in question for keyword in geometry_keywords)
    
    
    def _should_use_chartmoe(self, observation: Dict[str, Any]) -> bool:
        """判断是否应该使用 ChartMoE"""
        question = observation.get("question", "").lower()
        task = observation.get("task", "")
        
        # 图表相关关键词
        chart_keywords = [
            "chart", "graph", "plot", "bar", "line", "pie", "histogram",
            "axis", "trend", "data", "value", "maximum", "minimum",
            "increase", "decrease", "correlation"
        ]
        
        # 检查任务类型
        if "chart" in task.lower() or "graph" in task.lower():
            return True
        
        # 检查关键词
        return any(keyword in question for keyword in chart_keywords)
    
    
    def _extract_detection_target(self, question: str) -> str:
        """从问题中提取检测目标 - 改进版本"""
        question_lower = question.lower()
        
        # 1. 物理/科学相关对象
        physics_objects = {
            "block": ["block", "cube", "box", "mass"],
            "spring": ["spring", "coil", "elastic"],
            "ball": ["ball", "sphere", "circle"],
            "rod": ["rod", "bar", "stick", "pole"],
            "pendulum": ["pendulum", "bob", "weight"],
            "pulley": ["pulley", "wheel"],
            "incline": ["incline", "ramp", "slope"],
            "beam": ["beam", "plank", "lever"],
            "force": ["force", "arrow", "vector"],
            "angle": ["angle", "degree", "arc"]
        }
        
        # 检查物理对象
        for main_term, synonyms in physics_objects.items():
            for synonym in synonyms:
                if synonym in question_lower:
                    return f"{main_term}s"  # 返回复数形式
        
        # 2. 数学/几何对象
        geometry_objects = [
            "triangle", "circle", "rectangle", "square", "polygon",
            "line", "point", "angle", "vertex", "edge",
            "shape", "figure", "diagram"
        ]
        
        for obj in geometry_objects:
            if obj in question_lower:
                return f"all {obj}s"
        
        # 3. 图表相关对象
        chart_objects = [
            "bar", "column", "line", "point", "axis",
            "label", "legend", "title", "data point"
        ]
        
        for obj in chart_objects:
            if obj in question_lower and ("chart" in question_lower or "graph" in question_lower):
                return f"{obj}s in the chart"
        
        # 4. 通用计数模式
        count_patterns = [
            (r'how many (\w+)', 1),
            (r'count (?:the |all )?(\w+)', 1),
            (r'number of (\w+)', 1),
            (r'total (\w+)', 1),
            (r'(\w+) are there', 1),
            (r'(\w+) can you see', 1),
            (r'(\w+) in the (?:image|picture|diagram)', 1)
        ]
        
        for pattern, group_idx in count_patterns:
            match = re.search(pattern, question_lower)
            if match:
                target = match.group(group_idx)
                # 过滤掉无意义的词
                if target not in ["much", "many", "the", "a", "an", "this", "that"]:
                    return f"all {target}"
        
        # 5. 特定场景检测
        if "equation" in question_lower:
            return "equations and mathematical symbols"
        elif "measurement" in question_lower:
            return "rulers, scales, and measurements"
        elif "experiment" in question_lower:
            return "experimental apparatus and equipment"
        
        # 6. 基于问题上下文的智能检测
        # 如果问题包含比较
        if any(word in question_lower for word in ["larger", "smaller", "biggest", "smallest", "compare"]):
            # 尝试找到比较的对象
            nouns = re.findall(r'\b(?:the )?(\w+)s?\b', question_lower)
            meaningful_nouns = [n for n in nouns if len(n) > 2 and n not in 
                               ["the", "and", "are", "is", "which", "what", "how"]]
            if meaningful_nouns:
                return f"all {meaningful_nouns[0]}s"
        
        # 默认：返回更具体的描述而不是 "all objects"
        return "all visible objects and their labels"
    
    
    def _build_deepeyes_prompt(self, observation: Dict[str, Any]) -> str:
        """构建 DeepEyes 工具调用提示 - 优先使用模型生成的bbox"""
        question = observation.get("question", "")
        question_lower = question.lower()
        
        # ⭐ 优先使用模型生成的bbox
        if self.current_bbox:
            print(f"[DEBUG] Using model-generated bbox: {self.current_bbox}")
            return self._build_deepeyes_prompt_with_bbox(self.current_bbox)
        
        # 获取实际图像尺寸
        image_width = 800  # 默认值
        image_height = 600  # 默认值
        
        # 从observation中获取实际尺寸
        if "image" in observation and observation["image"]:
            try:
                if hasattr(observation["image"], "size"):
                    image_width, image_height = observation["image"].size
                elif hasattr(observation["image"], "width"):
                    image_width = observation["image"].width
                    image_height = observation["image"].height
            except:
                pass
        
        print(f"[DEBUG] No model bbox, using keyword-based approach with image size: {image_width}x{image_height}")
        
        # 根据问题内容决定使用哪个 DeepEyes 内部工具
        if any(kw in question_lower for kw in ["rotate", "angle", "turn", "orientation"]):
            # 旋转相关
            # 尝试提取角度
            angle_match = re.search(r'(\d+)\s*degrees?', question_lower)
            angle = int(angle_match.group(1)) if angle_match else 90
            
            tool_call = {
                "name": "image_rotate_tool",
                "arguments": {
                    "angle": angle
                }
            }
        else:
            # 缩放相关（默认）
            # 根据问题内容智能确定缩放区域
            if any(kw in question_lower for kw in ["top", "upper"]):
                bbox = [0, 0, image_width, image_height // 3]
            elif any(kw in question_lower for kw in ["bottom", "lower"]):
                bbox = [0, 2 * image_height // 3, image_width, image_height]
            elif any(kw in question_lower for kw in ["left"]):
                bbox = [0, 0, image_width // 3, image_height]
            elif any(kw in question_lower for kw in ["right"]):
                bbox = [2 * image_width // 3, 0, image_width, image_height]
            elif any(kw in question_lower for kw in ["center", "middle"]):
                # 中心区域
                margin_x = image_width // 4
                margin_y = image_height // 4
                bbox = [margin_x, margin_y, image_width - margin_x, image_height - margin_y]
            else:
                # 默认：略微缩小的全图
                margin = min(50, image_width // 20, image_height // 20)
                bbox = [margin, margin, image_width - margin, image_height - margin]
            
            # 确保bbox在图像范围内
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_width, bbox[2])
            bbox[3] = min(image_height, bbox[3])
            
            tool_call = {
                "name": "image_zoom_in_tool",
                "arguments": {
                    "bbox_2d": bbox
                }
            }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    
    def _build_diagram_formalizer_prompt(self, observation: Dict[str, Any]) -> str:
        """构建 DiagramFormalizer 工具调用提示"""
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
        """构建 GroundingDINO 工具调用提示 - 改进版本"""
        question = observation.get("question", "")
        
        # 提取智能化的检测目标
        caption = self._extract_detection_target(question)
        
        # 根据问题类型调整阈值
        question_lower = question.lower()
        if "small" in question_lower or "tiny" in question_lower:
            box_threshold = 0.25  # 降低阈值以检测小对象
            text_threshold = 0.20
        elif "all" in question_lower or "every" in question_lower:
            box_threshold = 0.30  # 中等阈值
            text_threshold = 0.20
        else:
            box_threshold = 0.35  # 默认阈值
            text_threshold = 0.25
        
        # 构建工具调用
        tool_call = {
            "tool": "grounding_dino",
            "parameters": {
                "caption": caption,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
        }
        
        # 添加调试信息
        print(f"  - GroundingDINO caption: '{caption}'")
        print(f"  - Thresholds: box={box_threshold}, text={text_threshold}")
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    
    def _extract_count_target(self, question: str) -> str:
        """从计数问题中提取目标对象"""
        question_lower = question.lower()
        
        # 匹配模式
        patterns = [
            r'how many (\w+)',
            r'count (?:the |all )?(\w+)',
            r'number of (\w+)',
            r'total (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)
        
        return "all objects"
    
    
    def _build_chartmoe_prompt(self, observation: Dict[str, Any]) -> str:
        """构建 ChartMoE 工具调用提示"""
        question = observation.get("question", "")
        
        tool_call = {
            "tool": "chartmoe",
            "parameters": {
                "query": question,
                "analysis_type": "comprehensive"
            }
        }
        
        json_str = json.dumps(tool_call, ensure_ascii=False)
        return f'<tool_call>{json_str}</tool_call>'
    
    
    def _should_use_secondary_tool(self, primary_tool: str, primary_result: Dict, question: str) -> bool:
        """判断是否需要使用次要工具"""
        # DeepEyes 后是否需要 GroundingDINO
        if primary_tool == "deepeyes":
            # 如果问题涉及计数或定位，可能需要 GroundingDINO
            if re.search(r'\bhow many|count|number of|where|position|locate\b', question.lower()):
                return True
        
        # DiagramFormalizer 后是否需要 GroundingDINO
        elif primary_tool == "diagram_formalizer":
            # 如果几何分析没有给出明确答案，且问题涉及计数/定位
            if not primary_result.get("solution") or primary_result.get("solution") == "":
                if re.search(r'\bhow many|count|number of|where|position|locate\b', question.lower()):
                    return True
            # 如果问题明确需要计数
            if re.search(r'\bhow many|count|number of\b', question.lower()):
                return True
        
        # ChartMoE 后是否需要 GroundingDINO
        elif primary_tool == "chartmoe":
            # 如果需要定位具体的图表元素
            if re.search(r'\bwhich\s+(?:bar|column|line|point)|where|position\b', question.lower()):
                return True
            # 如果ChartMoE没有完全回答位置相关问题
            output = primary_result.get("output", "")
            if "position" in question.lower() and "position" not in output.lower():
                return True
        
        return False
    
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """统一处理工具反馈 - 协同模式核心方法"""
        import os
        from PIL import Image
        
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        print(f"\n[_handle_tool_feedback] Processing {tool_name} feedback")
        print(f"[_handle_tool_feedback] Current tool context: {list(self.tool_context.keys())}")
        
        # 存储工具反馈到上下文
        self.tool_context[tool_name] = tool_feedback
        print(f"[_handle_tool_feedback] Stored {tool_name} results to context")
        
        # ⭐ 如果是DeepEyes且成功，保存图像
        if tool_name == "deepeyes" and tool_feedback.get("success") and "image" in tool_feedback:
            try:
                # 创建保存目录
                save_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/scienceqa_deepeyes/"
                os.makedirs(save_dir, exist_ok=True)
                
                # 获取任务ID
                task_id = observation.get("task_id", "unknown")
                
                # 获取处理后的图像
                processed_image = tool_feedback["image"]
                
                # 获取具体使用的工具类型
                tool_used = tool_feedback.get("tool_used", "unknown")
                
                # 构建文件名（包含尝试次数）
                attempt = observation.get("attempt", 1)
                filename = f"{task_id}_attempt{attempt}_deepeyes_{tool_used}.png"
                save_path = os.path.join(save_dir, filename)
                
                # 保存图像
                if isinstance(processed_image, Image.Image):
                    processed_image.save(save_path)
                    print(f"[_handle_tool_feedback] Saved DeepEyes image to: {save_path}")
                    
                    # 保存bbox信息
                    bbox_info_path = os.path.join(save_dir, f"{task_id}_attempt{attempt}_bbox_info.txt")
                    with open(bbox_info_path, 'w') as f:
                        f.write(f"Task ID: {task_id}\n")
                        f.write(f"Attempt: {attempt}\n")
                        f.write(f"Tool used: {tool_used}\n")
                        f.write(f"Original bbox: {self.current_bbox}\n")
                        if hasattr(processed_image, 'size'):
                            f.write(f"Processed image size: {processed_image.size}\n")
                        f.write(f"Success: {tool_feedback.get('success')}\n")
                        
                else:
                    print(f"[_handle_tool_feedback] WARNING: Image is not PIL Image")
                    
            except Exception as e:
                print(f"[_handle_tool_feedback] ERROR saving DeepEyes image: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 更新统计
        if tool_name in ["deepeyes", "diagram_formalizer", "grounding_dino", "chartmoe"]:
            self.tool_use_stats["successful_calls"] += 1
        
        # 构建中间响应
        response_parts = [
            f"<think>",
            f"{tool_name} execution complete.",
            f"Tools used so far: {', '.join(self.tool_context.keys())}",
        ]
        
        # 添加工具特定的总结
        if tool_name == "deepeyes":
            tool_used = tool_feedback.get("tool_used", "")
            response_parts.append(f"DeepEyes {tool_used} completed.")
        elif tool_name == "grounding_dino":
            num_detections = tool_feedback.get("num_detections", 0)
            response_parts.append(f"Detected {num_detections} objects.")
        elif tool_name == "diagram_formalizer":
            if tool_feedback.get("solution"):
                response_parts.append(f"Geometry solution found: {tool_feedback['solution']}")
        elif tool_name == "chartmoe":
            task_type = tool_feedback.get("task_type", "unknown")
            response_parts.append(f"Chart analysis ({task_type}) completed.")
        
        response_parts.append("Checking if more tools are needed...")
        response_parts.append("</think>")
        
        # 决定下一个工具
        next_tool = self.decide_tool(observation)
        
        if next_tool:
            # 需要更多工具
            print(f"[_handle_tool_feedback] Next tool needed, continuing collection")
            response = "\n".join(response_parts) + f"\n{next_tool}"
            
            # 更新工具调用计数
            self.current_tool_calls += 1
            self.tool_use_stats["total_calls"] += 1
            
            # 检查是否是工具链
            if len(self.tool_context) > 1:
                self.tool_use_stats["collaboration_chains"] += 1
            
            extra_info = {
                "action_type": "tool_chain_continue",
                "tool_processed": tool_name,
                "next_tool_call": True,
                "tools_used": list(self.tool_context.keys()),
                "tool_call_count": self.current_tool_calls,
                "stage": "tool_collection_continue"
            }
        else:
            # 工具收集完成
            print(f"[_handle_tool_feedback] All tools collected, ready for final answer")
            response = "\n".join(response_parts) + "\n<!-- All tools collected, ready for final answer -->"
            
            extra_info = {
                "action_type": "tool_collection_complete",
                "tools_used": list(self.tool_context.keys()),
                "ready_for_answer": True,
                "tool_call_count": self.current_tool_calls,
                "stage": "tool_collection_complete"
            }
        
        return response, extra_info
    
    

    def _build_final_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """基于所有工具信息构建最终答案 - 综合多工具结果"""
        print(f"\n[_build_final_answer] Building final answer with tools: {list(self.tool_context.keys())}")
        
        parts = []
        
        # 1. 汇总 DeepEyes 信息
        if "deepeyes" in self.tool_context:
            de_data = self.tool_context["deepeyes"]
            tool_used = de_data.get("tool_used", "")
            processed_output = de_data.get("processed_output", "")
            
            de_summary = [f"=== Visual Enhancement (DeepEyes - {tool_used}) ==="]
            de_summary.append(processed_output)
            if de_data.get("new_size"):
                de_summary.append(f"Processed image size: {de_data['new_size']}")
            
            parts.append("\n".join(de_summary))
        
        # 2. 汇总 DiagramFormalizer 信息
        if "diagram_formalizer" in self.tool_context:
            df_data = self.tool_context["diagram_formalizer"]
            solution = df_data.get("solution", "")
            formalized_output = df_data.get("formalized_output", "")
            steps = df_data.get("steps", [])
            
            df_summary = ["=== Geometry Analysis (DiagramFormalizer) ==="]
            if solution:
                df_summary.append(f"Solution found: {solution}")
            if formalized_output:
                df_summary.append(f"Formalized problem:\n{formalized_output[:300]}...")
            if steps:
                df_summary.append(f"Solution involves {len(steps)} steps")
            
            parts.append("\n".join(df_summary))
        
        # 3. 汇总 ChartMoE 信息
        if "chartmoe" in self.tool_context:
            cm_data = self.tool_context["chartmoe"]
            output = cm_data.get("output", "")
            task_type = cm_data.get("task_type", "")
            
            cm_summary = [f"=== Chart Analysis (ChartMoE - {task_type}) ==="]
            if output:
                # 根据任务类型提取关键信息
                if task_type == "to_table":
                    cm_summary.append("Extracted table data:")
                    cm_summary.append(output[:500] + "..." if len(output) > 500 else output)
                elif task_type == "analyze":
                    cm_summary.append("Analysis results:")
                    cm_summary.append(output[:400] + "..." if len(output) > 400 else output)
                else:
                    cm_summary.append(output[:300] + "..." if len(output) > 300 else output)
            
            parts.append("\n".join(cm_summary))
        
        # 4. 汇总 GroundingDINO 信息
        if "grounding_dino" in self.tool_context:
            gd_data = self.tool_context["grounding_dino"]
            num_detections = gd_data.get("num_detections", 0)
            phrases = gd_data.get("phrases", [])
            boxes = gd_data.get("boxes", [])
            
            gd_summary = ["=== Object Detection (GroundingDINO) ==="]
            gd_summary.append(f"Detected {num_detections} objects")
            
            if phrases:
                gd_summary.append("Objects found:")
                for i, phrase in enumerate(phrases[:10]):  # 最多显示10个
                    gd_summary.append(f"  {i+1}. {phrase}")
                if len(phrases) > 10:
                    gd_summary.append(f"  ... and {len(phrases) - 10} more")
            
            # 如果有特定的计数目标，提供更详细的信息
            query = gd_data.get("query", "")
            if query and query != "all objects":
                specific_count = sum(1 for phrase in phrases if query.lower() in phrase.lower())
                gd_summary.append(f"Specifically for '{query}': found {specific_count}")
            
            parts.append("\n".join(gd_summary))
        
        # 构建综合提示
        tool_summary = "\n\n".join(parts) if parts else "No tool information available."
        
        # 创建最终提示，强调综合使用多个工具的信息
        question = observation.get("question", "")
        prompt = f"""You have access to multiple tool analysis results. Please synthesize all the information to answer the question accurately.

    {tool_summary}

    === IMPORTANT INSTRUCTIONS ===
    1. Synthesize information from ALL tools used above
    2. If tools provide conflicting information, prioritize:
    - DiagramFormalizer for geometric calculations
    - ChartMoE for chart data extraction
    - GroundingDINO for counting and object location
    - DeepEyes for enhanced visual details
    3. Provide a clear, direct answer based on the combined analysis

    Original question: "{question}"

    You MUST output your final answer inside <answer> tags.
    For numerical answers, just put the number: <answer>42</answer>
    For multiple choice, just put the letter: <answer>A</answer>"""
        
        # 构建增强观察
        enhanced_observation = observation.copy()
        enhanced_observation["available_tools"] = []  # 禁用工具，防止递归
        enhanced_observation["output_format_instruction"] = prompt
        enhanced_observation["tool_synthesis_mode"] = True  # 标记这是工具综合模式
        
        # 调用父类生成答案
        print(f"[_build_final_answer] Calling parent VLM to synthesize answer")
        response, base_info = super().act(enhanced_observation)
        
        # 确保答案格式
        response = self._ensure_answer_format(response, observation)
        
        # 构建详细的信息
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "final_answer_with_tools",
            "tools_used": list(self.tool_context.keys()),
            "tool_collaboration": True,
            "collaboration_chain_length": len(self.tool_context),
            "stage": "answer_generation",
            "synthesis_mode": True
        })
        
        # 记录这是一个成功的工具链协作
        if len(self.tool_context) > 1:
            print(f"[_build_final_answer] Successful tool collaboration chain completed")
        
        return response, extra_info
    
    
    
    
    def _handle_grounding_dino_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理 GroundingDINO 工具反馈 - 兼容协同模式"""
        # 如果启用了工具协同，使用新的处理方式
        if self.enable_tool_collaboration:
            return self._handle_tool_feedback(observation)
        
        # 否则使用原有的立即生成答案的方式
        tool_feedback = observation.get("tool_feedback", {})
        
        # 获取检测结果
        num_detections = tool_feedback.get("num_detections", 0)
        boxes = tool_feedback.get("boxes", [])
        phrases = tool_feedback.get("phrases", [])
        logits = tool_feedback.get("logits", [])
        original_image_path = tool_feedback.get("original_image_path", "")
        
        print(f"[_handle_grounding_dino_feedback] Detected {num_detections} objects")
        print(f"[_handle_grounding_dino_feedback] Original image path: {original_image_path}")
        
        # 获取原始问题
        question = observation.get("question", "")
        question_lower = question.lower()
        
        # 根据问题类型生成答案
        if any(kw in question_lower for kw in ["how many", "count", "number of"]):
            # 计数问题 - 直接返回数字
            # 尝试识别具体要计数的对象
            target_object = self._extract_count_target(question)
            
            if target_object and target_object != "all objects":
                # 计算特定对象的数量
                count = sum(1 for phrase in phrases if target_object.lower() in phrase.lower())
                answer_text = str(count)
            else:
                # 计算所有对象
                answer_text = str(num_detections)
            
            # 构建带 <answer> 标签的响应
            response = f"""<think>
    Using GroundingDINO, I detected {num_detections} objects in the image.
    Detected objects: {', '.join(phrases[:5])}{'...' if len(phrases) > 5 else ''}
    The question asks about: {target_object if target_object else 'total objects'}
    Count: {answer_text}
    </think>
    <answer>{answer_text}</answer>"""
            
        elif any(kw in question_lower for kw in ["where", "locate", "position"]):
            # 位置问题
            if num_detections > 0:
                # 描述第一个检测到的对象位置
                location_desc = self._describe_location(boxes[0], phrases[0], tool_feedback.get("size", [100, 100]))
                response = f"""<think>
    GroundingDINO detected {num_detections} objects.
    Main object: {phrases[0]} is {location_desc}
    </think>
    <answer>{location_desc}</answer>"""
            else:
                response = "<answer>No objects detected in the image</answer>"
        
        else:
            # 其他类型问题 - 使用可视化分析
            if num_detections > 0 and original_image_path and os.path.exists(original_image_path):
                try:
                    # 生成可视化
                    vis_image_path = self._create_visualization(
                        original_image_path, boxes, phrases, logits
                    )
                    
                    # 创建增强观察，让VLM分析标注图像
                    enhanced_observation = observation.copy()
                    enhanced_observation["image_path"] = vis_image_path
                    enhanced_observation.pop("tool_feedback", None)
                    enhanced_observation.pop("requires_response", None)
                    enhanced_observation["available_tools"] = []  # 防止递归调用工具
                    
                    # 构建提示，确保输出 <answer> 标签
                    detection_summary = f"Detected {num_detections} objects: "
                    detection_summary += ", ".join([f"{p} ({l:.2f})" for p, l in zip(phrases[:5], logits[:5])])
                    
                    enhanced_observation["output_format_instruction"] = f"""Based on the annotated image showing {num_detections} detected objects:
    {detection_summary}

    Answer the question: "{question}"

    You MUST output your final answer inside <answer> tags. For example:
    - For numerical answers: <answer>42</answer>
    - For multiple choice: <answer>A</answer>
    - For yes/no questions: <answer>Yes</answer>

    Think step by step if needed, but always end with <answer>your_answer</answer>."""
                    
                    # 调用父类分析
                    response, base_info = super().act(enhanced_observation)
                    
                    # 验证响应包含 <answer> 标签
                    if "<answer>" not in response:
                        # 如果模型没有生成正确格式，尝试提取答案
                        response = self._ensure_answer_format(response, observation)
                    
                    extra_info = base_info.copy()
                    extra_info.update({
                        "action_type": "grounding_dino_visual_analysis",
                        "tool_used": "grounding_dino",
                        "detection_count": num_detections,
                        "visualized_image_path": vis_image_path,
                        "used_visualization": True
                    })
                    
                    return response, extra_info
                    
                except Exception as e:
                    print(f"[_handle_grounding_dino_feedback] Visualization error: {e}")
            
            # 备用方案：基于检测结果生成文本答案
            if num_detections > 0:
                objects_summary = ", ".join(phrases[:3]) + ("..." if len(phrases) > 3 else "")
                response = f"""<think>
    GroundingDINO detected {num_detections} objects: {objects_summary}
    Based on these detections, I'll answer the question.
    </think>
    <answer>The image contains {num_detections} objects including {objects_summary}</answer>"""
            else:
                response = "<answer>No objects detected in the image</answer>"
        
        # 构建返回信息
        extra_info = {
            "action_type": "grounding_dino_analysis",
            "tool_used": "grounding_dino", 
            "detection_count": num_detections,
            "used_visualization": False,
            "answered_with_format": True  # 标记已使用正确格式
        }
        
        return response, extra_info
        
    
    def _describe_location(self, box: List[float], phrase: str, image_size: List[int]) -> str:
        """描述对象位置"""
        x1, y1, x2, y2 = box
        h, w = image_size if len(image_size) == 2 else [100, 100]
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 判断位置
        if cx < w/3:
            h_pos = "left"
        elif cx > 2*w/3:
            h_pos = "right"
        else:
            h_pos = "center"
        
        if cy < h/3:
            v_pos = "top"
        elif cy > 2*h/3:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        return f"located at the {v_pos}-{h_pos} of the image"
    
    
    def _create_visualization(self, image_path: str, boxes: List, phrases: List, logits: List) -> str:
        """创建可视化图像"""
        from PIL import Image, ImageDraw, ImageFont
        
        # 加载原始图像
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 绘制检测框
        for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
            color = tuple(random.randint(100, 255) for _ in range(3))
            
            # 绘制边界框
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            label = f"{phrase} ({logit:.2f})"
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            label_y = y1 - text_height - 4 if y1 - text_height - 4 > 0 else y1
            draw.rectangle(
                [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
                fill=color
            )
            draw.text((x1 + 2, label_y + 2), label, fill="white", font=font)
        
        # 保存到临时文件
        temp_dir = tempfile.mkdtemp()
        vis_image_path = os.path.join(temp_dir, "grounding_dino_annotated.jpg")
        image.save(vis_image_path)
        
        return vis_image_path
    
    
    def _handle_deepeyes_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理 DeepEyes 工具反馈 - 兼容协同模式"""
        import os
        from PIL import Image
        
        # 如果启用了工具协同，使用新的处理方式
        if self.enable_tool_collaboration:
            return self._handle_tool_feedback(observation)
        
        # 否则使用原有的立即生成答案的方式
        tool_feedback = observation.get("tool_feedback", {})
        tool_used = tool_feedback.get("tool_used", "")
        processed_output = tool_feedback.get("processed_output", "")
        
        # ⭐ 保存DeepEyes处理后的图像
        if tool_feedback.get("success") and "image" in tool_feedback:
            try:
                # 创建保存目录
                save_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/scienceqa_deepeyes/"
                os.makedirs(save_dir, exist_ok=True)
                
                # 获取任务ID和尝试次数
                task_id = observation.get("task_id", "unknown")
                attempt = observation.get("attempt", 1)
                
                # 获取处理后的图像
                processed_image = tool_feedback["image"]
                
                # 构建文件名
                filename = f"{task_id}_attempt{attempt}_deepeyes_{tool_used}.png"
                save_path = os.path.join(save_dir, filename)
                
                # 保存图像
                if isinstance(processed_image, Image.Image):
                    processed_image.save(save_path)
                    print(f"  - Saved DeepEyes processed image to: {save_path}")
                    
                    # 保存bbox信息
                    bbox_info_path = os.path.join(save_dir, f"{task_id}_attempt{attempt}_bbox_info.txt")
                    with open(bbox_info_path, 'w') as f:
                        f.write(f"Task ID: {task_id}\n")
                        f.write(f"Attempt: {attempt}\n")
                        f.write(f"Tool used: {tool_used}\n")
                        f.write(f"Original bbox: {self.current_bbox}\n")
                        if hasattr(processed_image, 'size'):
                            f.write(f"Processed image size: {processed_image.size}\n")
                        f.write(f"Original image path: {observation.get('image_path', 'unknown')}\n")
                        f.write(f"Question: {observation.get('question', 'unknown')}\n")
                        
                else:
                    print(f"  - WARNING: Processed image is not a PIL Image object")
                    
            except Exception as e:
                print(f"  - ERROR saving DeepEyes image: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []  # 防止递归调用工具
        
        # 如果有处理后的图像，更新观察中的图像
        if tool_feedback.get("has_processed_image") and "image" in tool_feedback:
            enhanced_observation["image"] = tool_feedback["image"]
        
        # 构建提示
        tool_summary = f"DeepEyes {tool_used} Analysis:\n"
        tool_summary += processed_output
        
        enhanced_observation["output_format_instruction"] = f"""Based on the visual enhancement analysis:
    {tool_summary}

    Answer the question: "{observation.get('question', '')}"

    You MUST output your answer inside <answer> tags."""
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "deepeyes_analysis",
            "tool_used": "deepeyes",
            "sub_tool": tool_used
        })
        
        return response, extra_info
    
    
    def _handle_diagram_formalizer_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理 DiagramFormalizer 反馈 - 兼容协同模式"""
        # 如果启用了工具协同，使用新的处理方式
        if self.enable_tool_collaboration:
            return self._handle_tool_feedback(observation)
        
        # 否则使用原有的立即生成答案的方式
        tool_feedback = observation.get("tool_feedback", {})
        tool_solution = tool_feedback.get("solution", "")
        
        print(f"[_handle_diagram_formalizer_feedback] Solution: {tool_solution}")
        
        # 如果工具提供了答案
        if tool_solution:
            # 检查是否是多选题
            choices = observation.get("choices", [])
            if choices:
                # 尝试匹配选项
                answer_letter = self._match_solution_to_choice(tool_solution, choices)
                if answer_letter:
                    response = f"""<think>
    DiagramFormalizer solved the problem and got {tool_solution}.
    This matches option {answer_letter}.
    </think>
    <answer>{answer_letter}</answer>"""
                    
                    extra_info = {
                        "action_type": "direct_tool_answer",
                        "tool_used": "diagram_formalizer",
                        "tool_solution": tool_solution,
                        "matched_answer": answer_letter
                    }
                    
                    return response, extra_info
            else:
                # 数值题，直接使用答案
                response = f"""<think>
    DiagramFormalizer solved the problem.
    The answer is {tool_solution}.
    </think>
    <answer>{tool_solution}</answer>"""
                
                extra_info = {
                    "action_type": "direct_tool_answer",
                    "tool_used": "diagram_formalizer",
                    "tool_solution": tool_solution
                }
                
                return response, extra_info
        
        # 如果工具没有提供明确答案，基于分析让模型回答
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []  # 防止递归调用工具
        
        # 提取分析信息
        formalized_output = tool_feedback.get("formalized_output", "")
        steps = tool_feedback.get("steps", [])
        
        # 构建提示
        analysis_summary = "DiagramFormalizer Analysis:\n"
        if formalized_output:
            analysis_summary += f"Formalized problem: {formalized_output[:300]}...\n"
        if steps:
            analysis_summary += f"Solution steps: {len(steps)} steps provided\n"
        
        enhanced_observation["output_format_instruction"] = f"""Based on the geometric analysis:
    {analysis_summary}

    Answer the question: "{observation.get('question', '')}"

    You MUST output your answer inside <answer> tags."""
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "diagram_formalizer_analysis",
            "tool_used": "diagram_formalizer",
            "had_solution": bool(tool_solution)
        })
        
        return response, extra_info

    
    
    def _handle_chartmoe_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理 ChartMoE 反馈 - 兼容协同模式"""
        # 如果启用了工具协同，使用新的处理方式
        if self.enable_tool_collaboration:
            return self._handle_tool_feedback(observation)
        
        # 否则使用原有的立即生成答案的方式
        tool_feedback = observation.get("tool_feedback", {})
        task_type = tool_feedback.get("task_type", "unknown")
        output = tool_feedback.get("output", "")
        analysis = tool_feedback.get("analysis", {})
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []  # 防止递归调用工具
        
        # 构建基于图表分析的提示
        chart_summary = f"ChartMoE Analysis (Task: {task_type}):\n"
        
        # 根据任务类型添加特定信息
        if task_type == "to_table" and output:
            chart_summary += f"Extracted table data:\n{output[:500]}...\n"
        elif task_type == "describe" and output:
            chart_summary += f"Chart description:\n{output[:300]}...\n"
        elif task_type == "extract_data" and output:
            chart_summary += f"Extracted data:\n{output[:300]}...\n"
        elif task_type == "analyze" and output:
            chart_summary += f"Analysis results:\n{output[:400]}...\n"
        else:
            chart_summary += self._summarize_chart_analysis(analysis)
        
        enhanced_observation["output_format_instruction"] = f"""Based on the chart analysis:
    {chart_summary}

    Now answer the question: "{observation.get('question', '')}"

    You MUST output your answer inside <answer> tags."""
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "chartmoe_analysis",
            "tool_used": "chartmoe",
            "task_type": task_type
        })
        
        return response, extra_info
    
    def _handle_generic_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """处理通用工具反馈"""
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        
        # 创建增强的观察
        enhanced_observation = observation.copy()
        enhanced_observation.pop("tool_feedback", None)
        enhanced_observation.pop("requires_response", None)
        enhanced_observation["available_tools"] = []  # 防止递归调用工具
        
        # 添加工具输出信息
        tool_output = json.dumps(tool_feedback, indent=2, ensure_ascii=False)[:1000]
        enhanced_observation["output_format_instruction"] = f"""Based on the tool analysis:
{tool_output}

Answer the question: "{observation.get('question', '')}"

You MUST output your answer inside <answer> tags."""
        
        # 调用父类模型生成答案
        response, base_info = super().act(enhanced_observation)
        
        # 确保格式正确
        response = self._ensure_answer_format(response, observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "tool_feedback_response",
            "tool_used": tool_name
        })
        
        return response, extra_info
    
    
    def _match_solution_to_choice(self, solution: str, choices: List) -> Optional[str]:
        """匹配工具解决方案到选择题选项"""
        try:
            # 尝试数值匹配
            tool_value = float(solution)
            for i, choice in enumerate(choices):
                try:
                    choice_value = float(str(choice).strip())
                    # 使用容差匹配
                    if abs(choice_value - tool_value) < 0.01:
                        return chr(65 + i)  # A, B, C, D
                except ValueError:
                    pass
        except ValueError:
            # 字符串匹配
            for i, choice in enumerate(choices):
                if str(choice).strip() == str(solution).strip():
                    return chr(65 + i)
        
        return None
    
    
    def _summarize_chart_analysis(self, analysis: Dict) -> str:
        """总结图表分析结果"""
        summary = []
        
        if "data_points" in analysis:
            summary.append(f"Data points: {analysis['data_points']}")
        if "trends" in analysis:
            summary.append(f"Trends: {analysis['trends']}")
        if "max_value" in analysis:
            summary.append(f"Maximum value: {analysis['max_value']}")
        if "min_value" in analysis:
            summary.append(f"Minimum value: {analysis['min_value']}")
        
        return "\n".join(summary) if summary else "Chart analysis completed."
    
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """生成直接答案（不使用工具）"""
        print(f"  - Generating direct answer without tools")
        
        # 直接调用父类的 act 方法
        response, base_info = super().act(observation)
        
        # ⭐ 解析响应中的bbox
        bbox = self._parse_bbox_from_response(response)
        if bbox:
            print(f"  - Found bbox in response: {bbox}")
            self.current_bbox = bbox
            self.bbox_history.append(bbox)
        
        # 添加工具相关信息
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "direct_answer",
            "tool_call_count": 0,
            "tools_disabled": not self.enable_tools,
            "bbox_found": bbox is not None,
            "bbox": bbox
        })
        
        return response, extra_info
    
    
    def _generate_forced_final_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """当达到工具调用上限时，强制生成最终答案"""
        enhanced_observation = observation.copy()
        enhanced_observation["output_format_instruction"] = """You have used the maximum number of tools. 
Based on all the analysis so far, provide your final answer.
You MUST output your answer inside <answer> tags."""
        
        # 调用父类的 act 方法
        response, base_info = super().act(enhanced_observation)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": "forced_final_answer",
            "tool_call_count": self.current_tool_calls,
            "reason": "reached_tool_limit"
        })
        
        return response, extra_info
    
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """判断是否是新任务"""
        if not self.conversation_history:
            return True
        
        # 检查问题是否改变
        current_question = observation.get("question")
        for hist in reversed(self.conversation_history):
            if hist["role"] == "observation" and isinstance(hist["content"], dict):
                last_question = hist["content"].get("question")
                return current_question != last_question
        
        return True
    
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取工具使用性能报告"""
        report = {
            "overall_stats": self.tool_use_stats,
            "tool_performance": {},
            "problem_distribution": {}
        }
        
        # 工具性能统计
        for idx, perf in self.tool_performance.items():
            if perf["attempts"] > 0:
                tool_name = self.TOOL_INDEX_MAP[idx]["name"]
                report["tool_performance"][tool_name] = {
                    "attempts": perf["attempts"],
                    "success_rate": perf["successes"] / perf["attempts"] if perf["attempts"] > 0 else 0,
                    "failure_rate": perf["failures"] / perf["attempts"] if perf["attempts"] > 0 else 0,
                    "problem_types": perf["problem_types"]
                }
        
        return report
    
    
    def reset(self):
        """重置agent状态"""
        super().reset()
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        self.tool_call_history = {}
        self.tool_context = {}  # 重置工具上下文
        self.current_bbox = None  # ⭐ 重置bbox
        self.bbox_history = []