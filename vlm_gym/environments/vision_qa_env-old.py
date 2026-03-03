import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from .action import VLMActionSet
import copy
import os
import re   

# 导入新的DeepEyes工具
from vlm_gym.environments.tools.deepeyes_tool import DeepEyesTool

# 导入Grounding DINO工具
from vlm_gym.environments.tools.grounding_dino import GroundingDINOTool

logger = logging.getLogger(__name__)


class SimpleChatManager:
    """简单的聊天历史管理器"""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: Any):
        """添加消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.history.append(message)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self.history.copy()
    
    def clear(self):
        """清空历史"""
        self.history.clear()


class VisionQAEnv:
    """视觉问答环境 - 支持复杂动作系统、新DeepEyes工具、Grounding DINO和ChartMoE"""
    
    def __init__(
        self,
        dataset_path: str,
        task_entrypoint: Optional[callable] = None,
        task_kwargs: Dict[str, Any] = None,
        max_steps: int = 10,
        time_limit: Optional[float] = None,
        enable_actions: bool = True,
        custom_actions: Optional[Dict[str, callable]] = None,
        enable_deepeyes_tools: bool = False,
        deepeyes_config: Dict[str, Any] = None,  # 新DeepEyes配置
        enable_grounding_dino: bool = False,
        grounding_dino_config: Dict[str, Any] = None,
        enable_chartmoe: bool = False,
        chartmoe_config: Dict[str, Any] = None,
        enable_diagram_formalizer: bool = False,
        diagram_formalizer_config: Dict[str, Any] = None,
        **kwargs
    ):
        """初始化视觉问答环境"""
        self.dataset_path = Path(dataset_path)
        self.task_kwargs = task_kwargs or {}
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.enable_actions = enable_actions
        self.enable_deepeyes_tools = enable_deepeyes_tools
        self.enable_grounding_dino = enable_grounding_dino
        self.enable_chartmoe = enable_chartmoe
        self.enable_diagram_formalizer = enable_diagram_formalizer
        
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        # 初始化动作系统
        if self.enable_actions:
            self.action_set = VLMActionSet(custom_actions=custom_actions)
        else:
            self.action_set = None
        
        # 初始化新的DeepEyes工具
        self.deepeyes_tool = None
        if self.enable_deepeyes_tools:
            try:
                self.deepeyes_tool = DeepEyesTool(config=deepeyes_config)
                logger.info(f"New DeepEyes tool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DeepEyes tool: {e}")
                self.deepeyes_tool = None
        
        # 初始化Grounding DINO工具
        self.grounding_dino_tool = None
        if self.enable_grounding_dino:
            try:
                # 构建正确的配置
                grounding_dino_config = {
                    'model_path': '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/GroundingDINO/groundingdino_swint_ogc.pth',
                    'model_config': '/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                    'device': 'cpu',  # 使用CPU模式
                    'box_threshold': 0.35,
                    'text_threshold': 0.25,
                    'nms_threshold': 0.8
                }
                
                # 如果有传入的配置，合并它们
                if grounding_dino_config:
                    grounding_dino_config.update(grounding_dino_config)
                
                print(f"[DEBUG] Initializing GroundingDINO with config: {grounding_dino_config}")
                
                self.grounding_dino_tool = GroundingDINOTool(grounding_dino_config)
                logger.info("Grounding DINO tool initialized")
                print(f"[DEBUG] Grounding DINO tool initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Grounding DINO tool: {e}")
                print(f"[DEBUG] ❌ Grounding DINO initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.grounding_dino_tool = None
        
        # 初始化ChartMoE工具
        self.chartmoe_tool = None
        if self.enable_chartmoe:
            try:
                # ⭐ 修改：使用你的 FixedChartMoETool
                from chartmoe_vlmgym_tool import FixedChartMoETool
                config = chartmoe_config or {}
                self.chartmoe_tool = FixedChartMoETool(config)
                logger.info("ChartMoE tool initialized")
                print(f"[DEBUG] ChartMoE tool initialized: {self.chartmoe_tool}")
                print(f"[DEBUG] Tool type: {type(self.chartmoe_tool)}")
            except Exception as e:
                logger.error(f"Failed to initialize ChartMoE tool: {e}")
                print(f"[DEBUG] ❌ ChartMoE initialization failed:")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {str(e)}")
                import traceback
                traceback.print_exc()
                self.chartmoe_tool = None
        
        # 初始化 DiagramFormalizer 工具
        self.diagram_formalizer_tool = None
        if self.enable_diagram_formalizer:
            try:
                from vlm_gym.environments.tools.geometry_tools.diagram_formalizer import DiagramFormalizerTool
                config = diagram_formalizer_config or {}
                self.diagram_formalizer_tool = DiagramFormalizerTool(config)
                logger.info("DiagramFormalizer tool initialized")
                print(f"[DEBUG] DiagramFormalizer tool initialized: {self.diagram_formalizer_tool}")
                print(f"[DEBUG] Tool type: {type(self.diagram_formalizer_tool)}")
            except Exception as e:
                logger.error(f"Failed to initialize DiagramFormalizer tool: {e}")
                print(f"[DEBUG] ❌ DiagramFormalizer initialization failed:")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {str(e)}")
                import traceback
                traceback.print_exc()
                self.diagram_formalizer_tool = None
        
        # 工具管理器
        self.tool_manager = {}
        if self.deepeyes_tool:
            self.tool_manager['deepeyes'] = self.deepeyes_tool
        if self.grounding_dino_tool:
            self.tool_manager['grounding_dino'] = self.grounding_dino_tool
        if self.chartmoe_tool:
            self.tool_manager['chartmoe'] = self.chartmoe_tool
        if self.diagram_formalizer_tool:
            self.tool_manager['diagram_formalizer'] = self.diagram_formalizer_tool
        
        logger.info(f"Registered tools: {list(self.tool_manager.keys())}")
        
        # 初始化聊天管理器
        self.chat = SimpleChatManager()
        
        # 环境状态
        self.current_step = 0
        self.task = None
        self.start_time = None
        
        # 当前图像和问题（由环境管理）
        self.current_image = None
        self.current_question = ""
        self.task_goal = ""
        self.task_info = {}
        self.action_history = []  # 记录执行的动作
        
        # 任务入口点（由外部设置）
        self.task_entrypoint = task_entrypoint
        
        # ⭐ 新增：存储待处理的工具反馈
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取可用工具列表（供Agent查询）"""
        available_tools = {}
        
        # 新DeepEyes工具
        if self.deepeyes_tool and self.enable_deepeyes_tools:
            available_tools['deepeyes'] = self.deepeyes_tool.get_capabilities()
        
        # Grounding DINO工具
        if self.grounding_dino_tool and self.enable_grounding_dino:
            # ⭐ 修改：添加默认的能力信息，如果工具没有 get_capabilities 方法
            if hasattr(self.grounding_dino_tool, 'get_capabilities'):
                available_tools['grounding_dino'] = self.grounding_dino_tool.get_capabilities()
            else:
                available_tools['grounding_dino'] = {
                    "name": "grounding_dino",
                    "description": "Open-vocabulary object detection tool",
                    "capabilities": ["object_detection", "phrase_grounding"]
                }
        
        # ChartMoE工具
        if self.chartmoe_tool and self.enable_chartmoe:
            # ⭐ 修改：添加默认的能力信息
            if hasattr(self.chartmoe_tool, 'get_capabilities'):
                available_tools['chartmoe'] = self.chartmoe_tool.get_capabilities()
            else:
                available_tools['chartmoe'] = {
                    "name": "chartmoe",
                    "description": "Chart analysis and data extraction tool",
                    "capabilities": ["to_table", "describe", "extract_data", "summarize", "analyze", "compare", "trend"],
                    "tasks": {
                        "to_table": "Convert chart to table format",
                        "describe": "Describe chart in detail",
                        "extract_data": "Extract numerical data",
                        "summarize": "Summarize chart content",
                        "analyze": "Deep analysis with insights",
                        "compare": "Compare data series",
                        "trend": "Identify trends"
                    }
                }
        
        # DiagramFormalizer 工具
        if self.diagram_formalizer_tool and self.enable_diagram_formalizer:
            if hasattr(self.diagram_formalizer_tool, 'get_capabilities'):
                available_tools['diagram_formalizer'] = self.diagram_formalizer_tool.get_capabilities()
            else:
                available_tools['diagram_formalizer'] = {
                    "name": "diagram_formalizer",
                    "description": "Formalize geometric problems into mathematical expressions",
                    "capabilities": ["formalize", "solve", "analyze", "extract_constraints", "prove"],
                    "tasks": {
                        "formalize": "Convert geometric problems to formal notation",
                        "solve": "Solve geometric problems step by step",
                        "analyze": "Analyze geometric relationships",
                        "extract_constraints": "Extract geometric constraints",
                        "prove": "Generate formal proofs"
                    }
                }
        
        return available_tools
    
    def reset(self, task_id: str = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        # 重置状态
        self.current_step = 0
        self.start_time = time.time()
        self.chat.clear()
        self.action_history = []
        
        # ⭐ 重置工具反馈状态
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        # 创建任务实例
        if not self.task_entrypoint:
            raise ValueError("task_entrypoint not set. Please set env.task_entrypoint before reset.")
        
        try:
            # 准备任务参数，传递action_set给任务（如果启用）
            task_kwargs = self.task_kwargs.copy()
            if self.enable_actions:
                task_kwargs['action_set'] = self.action_set
            
            if callable(self.task_entrypoint):
                self.task = self.task_entrypoint(task_id=task_id, **task_kwargs)
            else:
                self.task = self.task_entrypoint(task_id=task_id, **task_kwargs)
                
        except Exception as e:
            logger.error(f"Failed to create task {task_id}: {e}")
            raise
        
        # 设置任务
        try:
            self.task_goal, self.task_info = self.task.setup()
            logger.debug(f"Task {task_id} setup complete")
        except Exception as e:
            logger.error(f"Failed to setup task {task_id}: {e}")
            raise
        
        # 加载图像
        self._load_current_image()
        
        # 设置当前问题
        task_data = self.task.task_data
        self.current_question = task_data.get("question", "")
        
        # 初始化新DeepEyes工具（如果启用）
        if self.deepeyes_tool and self.current_image:
            try:
                self.deepeyes_tool.reset(self.current_image)
                logger.debug("DeepEyes tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset DeepEyes tool: {e}")
        
        # 初始化Grounding DINO工具（如果启用）
        if self.grounding_dino_tool and self.current_image:
            try:
                self.grounding_dino_tool.reset(self.current_image)
                logger.debug("Grounding DINO tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset Grounding DINO tool: {e}")
        
        # 初始化ChartMoE工具（如果启用）
        if self.chartmoe_tool and self.current_image:
            try:
                self.chartmoe_tool.reset(self.current_image)
                logger.debug("ChartMoE tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset ChartMoE tool: {e}")
        
        # 初始化 DiagramFormalizer 工具（如果启用）
        if self.diagram_formalizer_tool:
            try:
                # DiagramFormalizer 主要处理文本，可以选择性传入图像
                self.diagram_formalizer_tool.reset(self.current_image)
                logger.debug("DiagramFormalizer tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset DiagramFormalizer tool: {e}")
        
        # 添加初始系统消息
        self.chat.add_message(
            role="system",
            content={
                "text": self.task_goal,
                "task_info": self.task_info
            }
        )
        
        # 构建观察
        observation = self._get_obs()
        
        # 构建信息（包含动作空间描述）
        info = {
            "task_id": task_id,
            "task_goal": self.task_goal,
            "task_info": self.task_info,
            "max_steps": self.max_steps,
            "time_limit": self.time_limit,
            "actions_enabled": self.enable_actions,
            "deepeyes_enabled": self.enable_deepeyes_tools and self.deepeyes_tool is not None,
            "grounding_dino_enabled": self.enable_grounding_dino and self.grounding_dino_tool is not None,
            "chartmoe_enabled": self.enable_chartmoe and self.chartmoe_tool is not None,
            "diagram_formalizer_enabled": self.enable_diagram_formalizer and self.diagram_formalizer_tool is not None,
            "available_tools": self.get_available_tools()
        }
        
        if self.enable_actions:
            info["available_actions"] = self.action_set.list_actions()
            info["action_space_description"] = self.action_set.describe(with_examples=False)
        
        return observation, info
    
    def validate_format(self, response: str) -> Tuple[bool, str]:
        """
        验证响应格式是否正确
        Args:
            response: 模型的响应文本
        Returns:
            (is_valid, error_message)
        """
        if not response or not isinstance(response, str):
            return False, "Empty or invalid response"
        
        response_lower = response.lower()
        
        # 检查 XML 格式
        has_xml_answer = '<answer>' in response_lower
        has_xml_reasoning = '<reasoning>' in response_lower
        
        # 检查 Markdown 格式
        has_md_answer = '## final_answer:' in response_lower or '##final_answer:' in response_lower
        has_md_reasoning = '## reasoning:' in response_lower or '##reasoning:' in response_lower
        
        # 如果是工具调用，不需要检查格式
        if '<tool_call>' in response:
            return True, "Tool call format"
        
        # 至少需要一种格式的答案标记
        if not has_xml_answer and not has_md_answer:
            if has_xml_reasoning:
                return False, "Found <reasoning> but missing <answer> tag. You must include <answer>your_answer</answer>"
            elif has_md_reasoning:
                return False, "Found ## REASONING: but missing ## FINAL_ANSWER: section. Both sections are required!"
            else:
                return False, "Invalid format. Use either <reasoning>...</reasoning><answer>...</answer> OR ## REASONING: ... ## FINAL_ANSWER: ..."
        
        return True, "Format is valid"
    
    def _clear_tool_feedback(self):
        """清空工具反馈 - 应该在Agent处理完反馈后调用"""
        if self.pending_tool_feedback is not None and isinstance(self.pending_tool_feedback, dict):
            print(f"[DEBUG] Clearing tool feedback for tool: {self.pending_tool_feedback.get('tool', 'unknown')}")
        self.pending_tool_feedback = None
        self.requires_tool_response = False
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行一步动作并返回观察、奖励、是否结束等信息"""
        
        if not self.task:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.current_step += 1
        
        # ===== 添加调试 =====
        print(f"\n[DEBUG VisionQAEnv.step] Step {self.current_step}")
        print(f"[DEBUG VisionQAEnv.step] Action type: {type(action)}")
        print(f"[DEBUG VisionQAEnv.step] Action length: {len(str(action))}")
        print(f"[DEBUG VisionQAEnv.step] Action preview: {str(action)[:200]}...")
        print(f"[DEBUG VisionQAEnv.step] Contains <tool_call>: {'<tool_call>' in str(action)}")
        print(f"[DEBUG VisionQAEnv.step] DeepEyes enabled: {self.enable_deepeyes_tools}")
        print(f"[DEBUG VisionQAEnv.step] Grounding DINO enabled: {self.enable_grounding_dino}")
        print(f"[DEBUG VisionQAEnv.step] ChartMoE enabled: {self.enable_chartmoe}")
        print(f"[DEBUG VisionQAEnv.step] DiagramFormalizer enabled: {self.enable_diagram_formalizer}")
        
        # 检查是否超时
        truncated = False
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            truncated = True
        
        # 检查是否超过最大步数
        if self.current_step > self.max_steps:
            truncated = True
        
        # 记录用户动作
        self.chat.add_message(
            role="user",
            content={
                "text": action,
                "step": self.current_step
            }
        )
        
        # 执行动作
        try:
            action_result = self._execute_action(action)
            logger.debug(f"Action executed: {action_result.get('type', 'unknown')}")
            
            # ===== 新增：格式验证 =====
            # 只对答案类型的响应进行格式验证
            if action_result.get("type") == "answer" or (
                action_result.get("type") == "direct_answer" and 
                hasattr(self.task, 'use_structured_output') and 
                self.task.use_structured_output
            ):
                format_valid, format_error = self.validate_format(action)
                
                if not format_valid and hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                    # 格式错误，触发反思
                    self.task.last_format_error = format_error  # 传递给task
                    
                    # 修改action_result以表示格式错误
                    action_result = {
                        "type": "format_error",
                        "status": "FAILED",
                        "content": format_error,
                        "error": format_error,
                        "original_action": action
                    }
            # ===== 格式验证结束 =====
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action_result = {
                "type": "error",
                "status": "FAILED",
                "content": str(e),
                "error": str(e)
            }
        
        # ⭐ 现在可以安全使用 action_result - 如果这是对工具反馈的响应，清空反馈
        if self.requires_tool_response and self.pending_tool_feedback is not None:
            is_new_tool_call = '<tool_call>' in str(action) or action_result.get("type") == "tool_call"
            if not is_new_tool_call:
                print(f"\n[DEBUG] Agent responded to tool feedback, clearing pending feedback")
                self._clear_tool_feedback()
        
        # 记录动作历史
        self.action_history.append({
            "step": self.current_step,
            "action": action,
            "result": action_result
        })
        
        # 记录助手回复
        self.chat.add_message(
            role="assistant", 
            content=action_result
        )
        
        # 处理任务验证
        reward, done, message, validation_info = self._handle_task_validation(
            action, action_result
        )
        
        # 如果截断了但还没完成，设置完成状态
        if truncated and not done:
            done = True
            if reward == 0:
                reward = -0.1  # 轻微惩罚
            message = f"Task truncated: {message}"
        
        # ⭐ 修改：在获取观察之前设置工具反馈
        if action_result.get("type") in ["deepeyes_feedback", "tool_result"]:
            # ⭐ 关键修改：只有在不同工具冲突时才清空
            if (self.pending_tool_feedback is not None 
                and self.pending_tool_feedback.get("tool") != action_result.get("tool")):
                print(f"[WARNING] Overwriting unprocessed tool feedback from {self.pending_tool_feedback.get('tool')} to {action_result.get('tool')}")
                self._clear_tool_feedback()
            
            self.requires_tool_response = True

            # 对DeepEyes特殊处理
            if action_result.get("tool") == "deepeyes":
                self.pending_tool_feedback = {
                    "tool": "deepeyes",
                    "tool_used": action_result.get("tool_used"),
                    "result": action_result.get("result", {}),
                    "processed_output": action_result.get("processed_output", ""),
                    "success": action_result.get("success", False)
                }
                if "image" in action_result:
                    self.pending_tool_feedback["has_processed_image"] = True
                print(f"\n[DEBUG VisionQAEnv.step] Set pending DeepEyes feedback")

            # 对Grounding DINO特殊处理
            elif action_result.get("tool") == "grounding_dino":
                # 优先从 detections 中获取（已经整理好的数据）
                detections = action_result.get("detections", {})
                # 如果 detections 为空，尝试从 result 中获取
                if not detections and action_result.get("result"):
                    result = action_result.get("result", {})
                    detections = {
                        "boxes": result.get("boxes", []),
                        "phrases": result.get("phrases", []),
                        "logits": result.get("logits", []),
                        "num_detections": result.get("num_detections", 0),
                        "size": result.get("size", [])
                    }
                
                # 获取图像路径（注意：vision_qa_env 中没有 current_image_path 属性）
                image_path = ""
                if self.task and self.task.task_data:
                    image_path = str(self.task.task_data.get("image_path", ""))
                    # 如果是相对路径，转换为绝对路径
                    if image_path and not os.path.isabs(image_path):
                        image_path = str(self.dataset_path / image_path)
                
                self.pending_tool_feedback = {
                    "tool": "grounding_dino",
                    "original_question": self.current_question,
                    "query": action_result.get("query", ""),
                    
                    # 原始图像路径（这是关键！）
                    "original_image_path": image_path,
                    
                    # 从 detections 中提取检测信息
                    "num_detections": detections.get("num_detections", 0),
                    "boxes": detections.get("boxes", []),
                    "phrases": detections.get("phrases", []),
                    "logits": detections.get("logits", []),
                    "size": detections.get("size", action_result.get("result", {}).get("size", [])),
                    
                    # 额外信息
                    "success": action_result.get("status") == "SUCCESS"
                }
                
                print(f"\n[DEBUG VisionQAEnv.step] Set pending Grounding DINO feedback:")
                print(f"  - num_detections: {self.pending_tool_feedback['num_detections']}")
                print(f"  - original_image_path: {self.pending_tool_feedback['original_image_path']}")
                print(f"  - image exists: {os.path.exists(image_path)}")
                print(f"  - requires_tool_response: {self.requires_tool_response}")
            
            # 对ChartMoE特殊处理
            elif action_result.get("tool") == "chartmoe":
                self.pending_tool_feedback = {
                    "tool": "chartmoe",
                    "task_type": action_result.get("task_type", "unknown"),
                    "output": action_result.get("output", {}),
                    "original_question": self.current_question,
                    "result": action_result.get("result", {})
                }
                print(f"\n[DEBUG VisionQAEnv.step] Set pending ChartMoE feedback:")
                print(f"  - task_type: {self.pending_tool_feedback['task_type']}")
                print(f"  - output: {self.pending_tool_feedback.get('output', 'N/A')}")
                print(f"  - requires_tool_response: {self.requires_tool_response}")
            
            # 对DiagramFormalizer特殊处理 - 修复：不要重新构建，保持原有的完整反馈
            elif action_result.get("tool") == "diagram_formalizer":
                # ⭐ 关键修改：检查是否已经在 _execute_diagram_formalizer 中设置了反馈
                if hasattr(self, 'pending_tool_feedback') and self.pending_tool_feedback and self.pending_tool_feedback.get('tool') == 'diagram_formalizer':
                    # 保持原有的完整反馈，它包含了solution等关键字段
                    print(f"\n[DEBUG VisionQAEnv.step] Keeping complete DiagramFormalizer feedback from _execute_diagram_formalizer")
                    print(f"  - Has solution: {'solution' in self.pending_tool_feedback}")
                    print(f"  - Solution value: '{self.pending_tool_feedback.get('solution', 'N/A')}'")
                    print(f"  - Task type: {self.pending_tool_feedback.get('task_type', 'N/A')}")
                    print(f"  - Formalized output preview: {str(self.pending_tool_feedback.get('formalized_output', ''))[:100]}")
                else:
                    # 如果没有预设的反馈，才构建新的（这种情况不应该发生）
                    print(f"[WARNING] No preset DiagramFormalizer feedback, building from action_result")
                    self.pending_tool_feedback = {
                        "tool": "diagram_formalizer",
                        "task_type": action_result.get("task_type", "unknown"),
                        "solution": action_result.get("solution", ""),  # 确保包含solution
                        "formalized_output": action_result.get("formalized_output", ""),
                        "steps": action_result.get("steps", []),
                        "original_question": self.current_question,
                        "result": action_result.get("result", {})
                    }
                print(f"  - requires_tool_response: {self.requires_tool_response}")
                
            else:
                self.pending_tool_feedback = action_result.get("tool_feedback", action_result.get("result", {}))
        
        # 构建观察（现在会包含工具反馈）
        observation = self._get_obs()
        
        # 构建信息
        info = {
            "step": self.current_step,
            "action_result": action_result,
            "validation": validation_info,
            "message": message,
            "truncated": truncated,
            "chat_history": self.chat.get_history(),
            "action_history": self.action_history
        }
        
        return observation, reward, done, truncated, info
    
    def _parse_tool_call(self, action: str) -> Tuple[str, Any]:
        """解析工具调用格式"""
        import re
        
        # 尝试匹配<tool_call>格式
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_match = re.search(tool_pattern, action, re.DOTALL)
        
        if tool_match:
            try:
                tool_content = json.loads(tool_match.group(1))
                return "tool_call", tool_content
            except json.JSONDecodeError as e:
                return "invalid_tool_call", {
                    "error": f"Invalid JSON in tool_call: {e}",
                    "raw_json": tool_match.group(1)
                }
        
        return None, None
    
    def _execute_action(self, action: str) -> Dict[str, Any]:
        """执行动作 - 支持新DeepEyes工具、Grounding DINO、ChartMoE、标准动作系统和直接答案"""
        if not action.strip():
            return {
                "type": "error",
                "status": "FAILED",
                "content": "Empty action provided",
                "error": "Empty action"
            }
        
        # ===== 添加调试输出 =====
        print(f"\n[DEBUG _execute_action] START")
        print(f"  - Action length: {len(action)}")
        print(f"  - Action preview (first 200 chars): {action[:200]}")
        print(f"  - Contains <tool_call>: {'<tool_call>' in action}")
        print(f"  - Contains <answer>: {'<answer>' in action}")
        print(f"  - DeepEyes enabled: {self.enable_deepeyes_tools}")
        print(f"  - Grounding DINO enabled: {self.enable_grounding_dino}")
        print(f"  - ChartMoE enabled: {self.enable_chartmoe}")
        print(f"  - DiagramFormalizer enabled: {self.enable_diagram_formalizer}")
        
        # 首先尝试解析通用工具调用格式
        tool_type, tool_content = self._parse_tool_call(action)
        
        # ⭐ 添加调试信息
        print(f"\n[DEBUG] Tool parsing result:")
        print(f"  - tool_type: {tool_type}")
        print(f"  - tool_content: {tool_content}")
       
        if tool_type == "tool_call":
            # 处理通用工具调用
            tool_name = tool_content.get("tool", "")
            if not tool_name:
                tool_name = tool_content.get("name", "")
            
            # ⭐ 添加更多调试信息
            print(f"\n[DEBUG] Tool call processing:")
            print(f"  - tool_name: '{tool_name}'")
            print(f"  - deepeyes_tool is None: {self.deepeyes_tool is None}")
            print(f"  - grounding_dino_tool is None: {self.grounding_dino_tool is None}")
            print(f"  - chartmoe_tool is None: {self.chartmoe_tool is None}")
            print(f"  - diagram_formalizer_tool is None: {self.diagram_formalizer_tool is None}")
            print(f"  - Available tools: {list(self.tool_manager.keys())}")
            
            # 处理新DeepEyes工具调用
            if tool_name == "deepeyes" and self.deepeyes_tool:
                print(f"[DEBUG] Executing DeepEyes tool call")
                return self._execute_deepeyes(tool_content)
            
            # 处理DeepEyes内部工具调用（image_zoom_in_tool等）
            elif tool_name in ["image_zoom_in_tool", "image_rotate_tool"] and self.deepeyes_tool:
                print(f"[DEBUG] Executing DeepEyes {tool_name}")
                return self._execute_deepeyes(tool_content)
            
            elif tool_name == "grounding_dino" and self.grounding_dino_tool:
                print(f"[DEBUG] Executing Grounding DINO tool call")
                return self._execute_grounding_dino(tool_content)
            
            elif tool_name == "chartmoe" and self.chartmoe_tool:
                print(f"[DEBUG] Executing ChartMoE tool call")
                return self._execute_chartmoe(tool_content)
            
            elif tool_name == "diagram_formalizer" and self.diagram_formalizer_tool:
                print(f"[DEBUG] Executing DiagramFormalizer tool call")
                return self._execute_diagram_formalizer(tool_content)
            
            else:
                # ⭐ 更详细的错误信息
                print(f"\n[DEBUG] ❌ Tool execution failed:")
                print(f"  - Requested tool: '{tool_name}'")
                print(f"  - Available tools: {list(self.tool_manager.keys())}")
                print(f"  - Tool manager content: {self.tool_manager}")
                
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": f"Unknown tool: {tool_name}",
                    "error": "Unknown tool",
                    "available_tools": list(self.tool_manager.keys())
                }
        
        elif tool_type == "invalid_tool_call":
            # JSON解析错误
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"Invalid tool_call format: {tool_content.get('error', 'Unknown error')}",
                "error": "Invalid tool_call format",
                "raw_json": tool_content.get('raw_json', '')
            }
        
        # 如果不是工具格式，检查标准动作系统
        if self.enable_actions and self.action_set:
            print(f"[DEBUG] Checking standard action system")
            
            # 首先尝试验证是否是有效的动作
            if self.action_set.validate_action(action):
                print(f"[DEBUG] Valid standard action detected")
                # 执行动作
                result = self.action_set.execute_action(action)
                
                # 标准化结果格式
                if result.get("status") == "SUCCESS":
                    return {
                        "type": result.get("type", "action_execution"),
                        "status": "SUCCESS",
                        "content": result.get("result"),
                        "action": result.get("action"),
                        "raw_action": action
                    }
                else:
                    return {
                        "type": "error",
                        "status": "FAILED",
                        "content": result.get("error", "Action failed"),
                        "error": result.get("error"),
                        "raw_action": action
                    }
            
            # 如果不是有效动作，检查是否是直接答案
            elif self._is_direct_answer(action):
                print(f"[DEBUG] Direct answer detected")
                return self._process_direct_answer(action)
            
            else:
                # 既不是有效动作也不是答案格式
                print(f"[DEBUG] Invalid action format")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": f"Invalid action format: {action}",
                    "error": "Invalid action format",
                    "available_actions": self.action_set.list_actions()
                }
        
        else:
            # 没有启用动作系统，使用原有的简单处理
            print(f"[DEBUG] No action system enabled, processing as direct answer")
            return self._process_direct_answer(action)
    
    def _execute_deepeyes(self, tool_content: Dict[str, Any]) -> Dict[str, Any]:
        """执行新DeepEyes工具调用"""
        
        print(f"\n[DEBUG _execute_deepeyes] START")
        print(f"  - Tool content: {tool_content}")
        print(f"  - Tool available: {self.deepeyes_tool is not None}")
        
        if not self.deepeyes_tool:
            return {
                "type": "error",
                "status": "FAILED",
                "content": "DeepEyes tool not initialized",
                "error": "Tool not initialized"
            }
        
        # 构造完整的action字符串（DeepEyes期望的格式）
        tool_name = tool_content.get("name", tool_content.get("tool", ""))
        
        # 如果是DeepEyes内部工具（image_zoom_in_tool等）
        if tool_name in ["image_zoom_in_tool", "image_rotate_tool"]:
            action_string = f"""<tool_call>
{json.dumps(tool_content)}
</tool_call>"""
        else:
            # 获取参数中的action
            parameters = tool_content.get("parameters", {})
            action_string = parameters.get("action", json.dumps(tool_content))
        
        print(f"  - Constructed action string: {action_string[:200]}...")
        
        try:
            print(f"\n[DEBUG] Calling deepeyes_tool.execute()...")
            
            # 执行DeepEyes
            result = self.deepeyes_tool.execute(action_string)
            
            print(f"\n[DEBUG] DeepEyes execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Success: {result.get('success', False) if isinstance(result, dict) else 'N/A'}")
            
            # 检查是否有错误
            if "error" in result:
                print(f"  - ERROR in result: {result['error']}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result["error"],
                    "error": result["error"],
                    "error_type": result.get("error_type", "Unknown")
                }
            
            # 成功情况
            if result.get('success'):
                # 检查是否是最终答案
                if result.get('final_answer'):
                    return {
                        "type": "answer",
                        "status": "SUCCESS",
                        "content": result['final_answer'],
                        "source": "deepeyes"
                    }
                
                # 工具执行成功，需要反馈
                tool_used = result.get('tool_used', tool_name)
                content = f"DeepEyes {tool_used} executed successfully"
                
                return_dict = {
                    "type": "tool_result",
                    "status": "SUCCESS",
                    "tool": "deepeyes",
                    "tool_used": tool_used,
                    "content": content,
                    "result": result,
                    "requires_tool_response": True,
                    "processed_output": result.get("processed_output", "")
                }
                
                # 如果有处理后的图像
                if "image" in result:
                    return_dict["image"] = result["image"]
                    return_dict["new_size"] = result.get("new_size", [])
                    
                    # ⭐ 添加保存图片功能
                    try:
                        import os
                        from PIL import Image
                        
                        save_dir = "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/vlm_gym/environments/tools/scienceqa_deepeyes/"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 获取任务ID
                        task_id = getattr(self.task, 'task_id', 'unknown') if hasattr(self, 'task') else 'unknown'
                        
                        # 获取当前尝试次数
                        attempt = 1
                        if hasattr(self, 'task') and hasattr(self.task, 'current_attempt'):
                            attempt = self.task.current_attempt + 1
                        
                        # 构建文件名
                        filename = f"{task_id}_attempt{attempt}_deepeyes_{tool_used}.png"
                        save_path = os.path.join(save_dir, filename)
                        
                        # 保存图像
                        processed_image = result['image']
                        if isinstance(processed_image, Image.Image):
                            processed_image.save(save_path)
                            print(f"  - Saved DeepEyes processed image to: {save_path}")
                            
                            # 保存bbox信息
                            bbox_info_path = os.path.join(save_dir, f"{task_id}_attempt{attempt}_bbox_info.txt")
                            with open(bbox_info_path, 'w') as f:
                                f.write(f"Task ID: {task_id}\n")
                                f.write(f"Attempt: {attempt}\n")
                                f.write(f"Tool used: {tool_used}\n")
                                f.write(f"Original bbox: {tool_content.get('arguments', {}).get('bbox_2d', 'unknown')}\n")
                                f.write(f"Processed image size: {processed_image.size}\n")
                                f.write(f"New size: {result.get('new_size', 'unknown')}\n")
                                f.write(f"Success: True\n")
                        else:
                            print(f"  - WARNING: Image is not PIL Image object")
                            
                    except Exception as e:
                        print(f"  - ERROR saving DeepEyes image: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    # ⭐ 保存图片功能结束

                print(f"\n[DEBUG] Returning tool_result:")
                print(f"  - Type: {return_dict['type']}")
                print(f"  - Status: {return_dict['status']}")
                print(f"  - Tool used: {return_dict['tool_used']}")
                print(f"  - requires_tool_response: {return_dict.get('requires_tool_response', False)}")
                
                return return_dict
            
            else:
                # 执行失败
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": "DeepEyes execution failed",
                    "error": "Execution failed",
                    "result": result
                }
                
        except Exception as e:
            print(f"\n[DEBUG] DeepEyes execution FAILED with exception:")
            print(f"  - Exception type: {type(e).__name__}")
            print(f"  - Exception message: {str(e)}")
            import traceback
            traceback.print_exc()
            
            logger.error(f"DeepEyes execution failed: {e}")
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"DeepEyes execution error: {str(e)}",
                "error": str(e)
            }
    
    def _execute_grounding_dino(self, tool_content: Dict[str, Any]) -> Dict[str, Any]:
        """执行Grounding DINO工具调用"""
        
        # ⭐ 1. 函数开始时打印
        print(f"\n[DEBUG _execute_grounding_dino] START")
        print(f"  - Tool content: {tool_content}")
        print(f"  - Tool available: {self.grounding_dino_tool is not None}")
        
        if not self.grounding_dino_tool:
            return {
                "type": "error",
                "status": "FAILED",
                "content": "Grounding DINO tool not initialized",
                "error": "Tool not initialized"
            }
        
        # 获取参数
        parameters = tool_content.get("parameters", {})
        
        # ⭐ 2. 打印参数信息
        print(f"  - Parameters extracted: {parameters}")
        print(f"  - Caption: {parameters.get('caption', 'N/A')}")
        
        try:
            # ⭐ 3. 执行前打印
            print(f"\n[DEBUG] Calling grounding_dino_tool.execute()...")
            
            # 执行检测
            result = self.grounding_dino_tool.execute(parameters)
            
            # ⭐ 4. 执行后立即打印结果
            print(f"\n[DEBUG] Grounding DINO execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Num detections: {result.get('num_detections', 0) if isinstance(result, dict) else 'N/A'}")
            
            # 检查是否有错误
            if "error" in result:
                # ⭐ 5. 错误情况打印
                print(f"  - ERROR in result: {result['error']}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result["error"],
                    "error": result["error"]
                }
            
            # ⭐ 6. 成功情况详细打印
            print(f"\n[DEBUG] Processing successful detection result:")
            print(f"  - Boxes: {result.get('boxes', [])[:3]}...")  # 只打印前3个
            print(f"  - Phrases: {result.get('phrases', [])}")
            print(f"  - Logits: {result.get('logits', [])[:3]}...")  # 只打印前3个
            print(f"  - Image size: {result.get('size', 'N/A')}")
            
            # ⭐ 新增：获取图像路径
            image_path = ""
            if self.task and self.task.task_data:
                image_path = str(self.task.task_data.get("image_path", ""))
                # 如果是相对路径，转换为绝对路径
                if image_path and not os.path.isabs(image_path):
                    from pathlib import Path
                    image_path = str(self.dataset_path / image_path)
            
            print(f"  - Image path: {image_path}")
            print(f"  - Image exists: {os.path.exists(image_path) if image_path else False}")
            
            # 成功返回
            return_dict = {
                "type": "tool_result",
                "status": "SUCCESS",
                "tool": "grounding_dino",
                "content": f"Detected {result.get('num_detections', 0)} objects",
                "result": result,
                "requires_tool_response": True,  # ⭐ 添加这个标志
                "detections": {
                    "boxes": result.get("boxes", []),
                    "phrases": result.get("phrases", []),
                    "logits": result.get("logits", []),
                    "num_detections": result.get("num_detections", 0),
                    "size": result.get("size", [])  # ⭐ 添加图像尺寸
                },
                "query": parameters.get("caption", ""),  # 保存查询词
                # ⭐ 新增：添加图像路径到返回结果
                "image_path": image_path,
                "image_info": {
                    "path": image_path,
                    "exists": os.path.exists(image_path) if image_path else False,
                    "size": result.get("size", [])  # [H, W]
                }
            }
            
            # ⭐ 7. 返回前打印
            print(f"\n[DEBUG] Returning tool_result:")
            print(f"  - Type: {return_dict['type']}")
            print(f"  - Status: {return_dict['status']}")
            print(f"  - requires_tool_response: {return_dict.get('requires_tool_response', False)}")
            print(f"  - Content: {return_dict['content']}")
            print(f"  - Image path included: {bool(return_dict.get('image_path'))}")
            
            return return_dict
            
        except Exception as e:
            # ⭐ 8. 异常情况打印
            print(f"\n[DEBUG] Grounding DINO execution FAILED with exception:")
            print(f"  - Exception type: {type(e).__name__}")
            print(f"  - Exception message: {str(e)}")
            import traceback
            traceback.print_exc()
            
            logger.error(f"Grounding DINO execution failed: {e}")
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"Grounding DINO execution error: {str(e)}",
                "error": str(e)
            }
    
    def _execute_chartmoe(self, tool_content: Dict[str, Any]) -> Dict[str, Any]:
        """执行ChartMoE工具调用"""
        
        print(f"\n[DEBUG _execute_chartmoe] START")
        print(f"  - Tool content: {tool_content}")
        print(f"  - Tool available: {self.chartmoe_tool is not None}")
        
        if not self.chartmoe_tool:
            return {
                "type": "error",
                "status": "FAILED",
                "content": "ChartMoE tool not initialized",
                "error": "Tool not initialized"
            }
        
        # ⭐ 修改：正确处理参数格式
        # ChartMoE期望的参数格式是 {"task": "xxx"} 或 {"prompt": "xxx"}
        # 但 tool_content 可能包含 {"tool": "chartmoe", "parameters": {...}}
        
        # 提取实际参数
        if "parameters" in tool_content:
            parameters = tool_content["parameters"]
        else:
            # 移除 "tool" 键，保留其他所有参数
            parameters = {k: v for k, v in tool_content.items() if k != "tool"}
        
        print(f"  - Parameters extracted: {parameters}")
        print(f"  - Task type: {parameters.get('task', 'N/A')}")
        print(f"  - Prompt: {parameters.get('prompt', 'N/A')[:50] if parameters.get('prompt') else 'N/A'}...")
        
        try:
            print(f"\n[DEBUG] Calling chartmoe_tool.execute()...")
            
            # 执行ChartMoE
            result = self.chartmoe_tool.execute(parameters)
            
            print(f"\n[DEBUG] ChartMoE execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Success: {result.get('success', False) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Task type: {result.get('task_type', 'N/A') if isinstance(result, dict) else 'N/A'}")
            
            # 检查是否有错误
            if not result.get('success', True) or "error" in result:
                print(f"  - ERROR in result: {result.get('error', 'Unknown error')}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result.get("error", "ChartMoE execution failed"),
                    "error": result.get("error", "Unknown error"),
                    "error_type": result.get("error_type", "Unknown")
                }
            
            # 成功情况
            processed_output = result.get("processed_output", "")
            task_type = result.get("task_type", "unknown")
            
            print(f"\n[DEBUG] Processing successful ChartMoE result:")
            print(f"  - Task type: {task_type}")
            print(f"  - Output length: {len(str(processed_output))}")
            print(f"  - Output preview: {str(processed_output)[:200]}...")
            
            # 构建返回内容
            if task_type == "to_table":
                # 计算表格行数（简单估算）
                lines = processed_output.strip().split('\n') if processed_output else []
                table_rows = len([l for l in lines if '|' in l and not l.strip().startswith('|--')])
                content = f"Extracted table with {table_rows} data rows"
            elif task_type == "describe":
                content = f"Generated chart description ({len(processed_output)} characters)"
            elif task_type == "extract_data":
                content = f"Extracted data from chart"
            elif task_type == "summarize":
                content = f"Generated chart summary"
            elif task_type == "analyze":
                content = f"Completed chart analysis"
            elif task_type == "compare":
                content = f"Completed data comparison"
            elif task_type == "trend":
                content = f"Identified chart trends"
            else:
                content = f"ChartMoE processed query: {parameters.get('prompt', task_type)[:50]}..."
            
            # 成功返回
            return_dict = {
                "type": "tool_result",
                "status": "SUCCESS",
                "tool": "chartmoe",
                "content": content,
                "result": result,
                "requires_tool_response": True,
                "task_type": task_type,
                "output": processed_output,  # 直接使用字符串输出
                "raw_output": processed_output,
                "prompt_used": parameters.get("prompt", "")
            }
            
            print(f"\n[DEBUG] Returning tool_result:")
            print(f"  - Type: {return_dict['type']}")
            print(f"  - Status: {return_dict['status']}")
            print(f"  - requires_tool_response: {return_dict.get('requires_tool_response', False)}")
            print(f"  - Content: {return_dict['content']}")
            
            return return_dict
            
        except Exception as e:
            print(f"\n[DEBUG] ChartMoE execution FAILED with exception:")
            print(f"  - Exception type: {type(e).__name__}")
            print(f"  - Exception message: {str(e)}")
            import traceback
            traceback.print_exc()
            
            logger.error(f"ChartMoE execution failed: {e}")
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"ChartMoE execution error: {str(e)}",
                "error": str(e)
            }
    
    def _execute_diagram_formalizer(self, tool_content: Dict[str, Any]) -> Dict[str, Any]:
        """执行 DiagramFormalizer 工具调用"""
        
        print(f"\n[DEBUG _execute_diagram_formalizer] START")
        print(f"  - Tool content: {tool_content}")
        print(f"  - Tool available: {self.diagram_formalizer_tool is not None}")
        
        if not self.diagram_formalizer_tool:
            return {
                "type": "error",
                "status": "FAILED",
                "content": "DiagramFormalizer tool not initialized",
                "error": "Tool not initialized"
            }
        
        # 提取实际参数
        if "parameters" in tool_content:
            parameters = tool_content["parameters"]
        else:
            # 移除 "tool" 键，保留其他所有参数
            parameters = {k: v for k, v in tool_content.items() if k != "tool"}
        
        print(f"  - Parameters extracted: {parameters}")
        print(f"  - Task type: {parameters.get('task', 'N/A')}")
        print(f"  - Problem: {parameters.get('problem', 'N/A')[:100] if parameters.get('problem') else 'N/A'}...")
        
        try:
            print(f"\n[DEBUG] Calling diagram_formalizer_tool.execute()...")
            
            # 执行 DiagramFormalizer
            result = self.diagram_formalizer_tool.execute(parameters)
            
            print(f"\n[DEBUG] DiagramFormalizer execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Success: {result.get('success', False) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Task type: {result.get('task_type', 'N/A') if isinstance(result, dict) else 'N/A'}")
            
            # 检查是否有错误
            if not result.get('success', True):
                print(f"  - ERROR in result: {result.get('error', 'Unknown error')}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result.get("error", "DiagramFormalizer execution failed"),
                    "error": result.get("error", "Unknown error"),
                    "error_type": result.get("error_type", "Unknown")
                }
            
            # ===== 关键修改：提取和确保 solution 存在 =====
            # 使用 _extract_solution_from_result 方法提取 solution
            solution = self._extract_solution_from_result(result)
            result["solution"] = solution  # 确保 solution 在 result 中
            
            print(f"\n[DEBUG] Solution extraction:")
            print(f"  - Original solution in result: '{result.get('solution', 'NOT FOUND')}'")
            print(f"  - Extracted solution: '{solution}'")
            print(f"  - Final answer in result: '{result.get('final_answer', 'NOT FOUND')}'")
            
            # 如果 solution 还是空的，尝试从 final_answer 提取
            if not solution and "final_answer" in result:
                solution = str(result["final_answer"])
                result["solution"] = solution
                print(f"  - Extracted solution from final_answer: '{solution}'")
            
            # 成功情况
            task_type = result.get("task_type", "unknown")
            formalized_output = result.get("formalized_output", "")
            
            print(f"\n[DEBUG] Processing successful DiagramFormalizer result:")
            print(f"  - Task type: {task_type}")
            print(f"  - Solution: '{solution}' (empty: {not solution})")
            print(f"  - Output length: {len(str(formalized_output))}")
            print(f"  - Output preview: {str(formalized_output)[:200]}...")
            
            # 构建工具反馈
            tool_feedback = {
                "tool": "diagram_formalizer",
                "task_type": task_type,
                "formalized_output": formalized_output,
                "solution": solution,  # 关键：确保包含 solution
                "steps": result.get("steps", []),
                "raw_response": result.get("raw_response", ""),
                "analysis": result.get("analysis", {})
            }
            
            # ===== 关键：设置 pending_tool_feedback 并立即验证 =====
            self.pending_tool_feedback = tool_feedback
            
            print(f"\n[DEBUG] Setting pending_tool_feedback:")
            print(f"  - pending_tool_feedback keys: {list(self.pending_tool_feedback.keys())}")
            print(f"  - pending_tool_feedback solution: '{self.pending_tool_feedback.get('solution', 'EMPTY')}'")
            print(f"  - Full pending_tool_feedback: {self.pending_tool_feedback}")
            
            # 构建返回内容的消息
            if task_type == "formalize":
                content = f"Formalized geometric problem into mathematical notation"
            elif task_type == "solve":
                content = f"Solved geometric problem step by step"
                if solution:
                    content += f" (solution: {solution})"
            elif task_type == "analyze":
                content = f"Analyzed geometric relationships"
            elif task_type == "extract_constraints":
                constraints = result.get("constraints", [])
                content = f"Extracted {len(constraints)} geometric constraints"
            elif task_type == "prove":
                proof_steps = result.get("proof_steps", [])
                content = f"Generated formal proof with {len(proof_steps)} steps"
            else:
                content = f"DiagramFormalizer processed query: {task_type}"
            
            # 成功返回
            return_dict = {
                "type": "tool_result",
                "status": "SUCCESS",
                "tool": "diagram_formalizer",  # ⭐ 添加这个字段，让step方法知道这是diagram_formalizer的结果
                "message": f"DiagramFormalizer {'found solution: ' + solution if solution else 'completed analysis'}",
                "requires_tool_response": True,
                "content": content
            }
            
            print(f"\n[DEBUG] Returning tool_result:")
            print(f"  - Type: {return_dict['type']}")
            print(f"  - Status: {return_dict['status']}")
            print(f"  - Tool: {return_dict.get('tool', 'N/A')}")  # ⭐ 打印tool字段
            print(f"  - Message: {return_dict['message']}")
            print(f"  - requires_tool_response: {return_dict.get('requires_tool_response', False)}")
            print(f"  - Content: {return_dict['content']}")
            
            return return_dict
            
        except Exception as e:
            print(f"\n[DEBUG] DiagramFormalizer execution FAILED with exception:")
            print(f"  - Exception type: {type(e).__name__}")
            print(f"  - Exception message: {str(e)}")
            import traceback
            traceback.print_exc()
            
            logger.error(f"DiagramFormalizer execution failed: {e}")
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"DiagramFormalizer execution error: {str(e)}",
                "error": str(e)
            }

    def _extract_solution_from_result(self, result: Dict[str, Any]) -> str:
        """从 DiagramFormalizer 结果中提取 solution - 更灵活的版本"""
        # 首先检查是否已有 solution
        solution = result.get("solution", "")
        if solution:
            return str(solution).strip()
        
        # 检查 final_answer 字段
        if "final_answer" in result and result["final_answer"]:
            return str(result["final_answer"]).strip()
        
        import re
        
        # 可能的答案模式（更宽松）
        answer_patterns = [
            r'(?:answer|solution)\s*(?:is|=|:)\s*(\d+\.?\d*)',  # answer is 15, solution = 15
            r'(?:x|y|z)\s*=\s*(\d+\.?\d*)',                     # x = 15, y = 15
            r'(?:equals?|is)\s+(\d+\.?\d*)',                    # equals 15, is 15
            r'∠\s*\w+\s*=\s*(\d+\.?\d*)',                       # ∠ABC = 15
            r'measure.*?(?:is|equals?)\s*(\d+\.?\d*)',          # measure is 15
            r'(?:therefore|thus|so).*?(\d+\.?\d*)',             # therefore 15
            r'final.*?(\d+\.?\d*)',                             # final answer: 15
            r'(?:The answer is|Answer:)\s*(\d+\.?\d*)',         # The answer is 15
        ]
        
        # 先尝试从 steps 的最后几步中提取
        if result.get("steps"):
            for step in reversed(result.get("steps", [])[-3:]):  # 检查最后3步
                for pattern in answer_patterns:
                    match = re.search(pattern, str(step), re.IGNORECASE)
                    if match:
                        solution = match.group(1)
                        print(f"[DEBUG] Extracted solution '{solution}' from step: {step[:50]}...")
                        return solution
        
        # 再尝试从 formalized_output 中提取
        if result.get("formalized_output"):
            output = result["formalized_output"]
            for pattern in answer_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    # 使用最后一个匹配（通常是最终答案）
                    solution = matches[-1]
                    print(f"[DEBUG] Extracted solution '{solution}' from formalized_output")
                    return solution
        
        # 尝试从 raw_response 中提取
        if result.get("raw_response"):
            raw = result["raw_response"]
            for pattern in answer_patterns:
                matches = re.findall(pattern, raw, re.IGNORECASE)
                if matches:
                    solution = matches[-1]
                    print(f"[DEBUG] Extracted solution '{solution}' from raw_response")
                    return solution
        
        # 最后尝试提取任何合理的数字
        text_to_search = str(result.get("formalized_output", "")) + " " + str(result.get("raw_response", ""))
        if text_to_search.strip():
            numbers = re.findall(r'\b\d+\.?\d*\b', text_to_search)
            # 过滤掉太大或太小的数字
            reasonable_numbers = []
            for num in numbers:
                try:
                    val = float(num)
                    if 0 < val < 1000:  # 合理的答案范围
                        reasonable_numbers.append(num)
                except:
                    pass
            
            if reasonable_numbers:
                solution = reasonable_numbers[-1]  # 使用最后一个
                print(f"[DEBUG] Extracted solution '{solution}' as last reasonable number")
                return solution
        
        print(f"[DEBUG] No solution found in result")
        return ""
    
    def _is_direct_answer(self, action: str) -> bool:
        """检查是否是直接答案格式"""
        cleaned = action.strip()
        # 检查各种答案格式
        return (
            cleaned.startswith("answer_question(") or
            cleaned.startswith("Answer:") or
            cleaned.startswith("answer:") or
            # 检查<answer>标签
            "<answer>" in cleaned or
            # 简单的文本答案（不包含函数调用格式）
            ("(" not in cleaned and "<" not in cleaned)
        )
    
    def _process_direct_answer(self, action: str) -> Dict[str, Any]:
        """处理直接答案（兼容原有逻辑）"""
        cleaned_action = action.strip()
        
        # 检查<answer>标签
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_action, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return {
                "type": "answer",
                "status": "SUCCESS",
                "content": answer,
                "raw_action": action,
                "source": "answer_tag"
            }
        
        # 提取答案内容
        if cleaned_action.startswith("answer_question(") and cleaned_action.endswith(")"):
            # 从格式化动作中提取答案
            try:
                match = re.search(r'answer="([^"]*)"', cleaned_action)
                if match:
                    answer = match.group(1)
                else:
                    content = cleaned_action[16:-1]
                    answer = content
            except Exception:
                answer = cleaned_action
        else:
            # 移除常见前缀
            for prefix in ["Answer:", "answer:", "A:", "a:"]:
                if cleaned_action.startswith(prefix):
                    answer = cleaned_action[len(prefix):].strip()
                    break
            else:
                answer = cleaned_action
        
        return {
            "type": "answer",
            "status": "SUCCESS",
            "content": answer,
            "raw_action": action
        }
    
    def _handle_task_validation(self, action: str, action_result: Dict[str, Any]) -> Tuple[float, bool, str, Dict]:
        """处理任务验证逻辑 - 支持反思机制"""
        # 如果动作执行失败，给予小惩罚
        if action_result.get("status") == "FAILED":
            return -0.1, False, action_result.get("error", "Action failed"), {"error": True}
        
        # ===== 新增：处理格式错误 =====
        if action_result.get("type") == "format_error":
            if hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                current_attempt = getattr(self.task, 'current_attempt', 0)
                max_attempts = getattr(self.task, 'max_attempts', 1)
                
                if current_attempt < max_attempts:
                    return -0.05, False, f"Format error: {action_result.get('error')}. Please use the correct format.", {
                        "format_error": True,
                        "error": action_result.get('error'),
                        "needs_retry": True
                    }
            
            return -0.1, False, f"Format error: {action_result.get('error')}", {"format_error": True}
        # ===== 格式错误处理结束 =====
        
        # 检查是否是答案类型的结果
        if action_result.get("type") == "answer":
            # ⭐ 新增：先检查任务是否有step方法（支持反思）
            if hasattr(self.task, 'step') and hasattr(self.task, 'enable_reflection'):
                # 使用任务的step方法来处理答案和反思
                try:
                    # 调用task的step方法
                    task_obs, task_reward, task_done, task_truncated, task_info = self.task.step(action)
                    
                    # 如果任务还没完成（需要反思）
                    if not task_done and self.task.enable_reflection:
                        # 更新环境的观察，包含反思相关信息
                        if isinstance(task_obs, dict):
                            # 合并task返回的观察到环境观察中
                            self.current_question = task_obs.get('question', self.current_question)
                            self.task_info.update(task_obs)
                        
                        # 返回中间结果，允许继续
                        return task_reward, False, task_info.get('message', 'Reflection needed'), task_info
                    
                    # 任务完成（正确或达到最大尝试）
                    return task_reward, task_done, task_info.get('message', 'Task completed'), task_info
                    
                except Exception as e:
                    logger.error(f"Error calling task.step: {e}")
                    # 如果task.step失败，回退到validate方法
            
            # 回退：使用任务的验证逻辑（旧方式）
            reward, done, message, validation_info = self.task.validate(
                chat_history=self.chat.get_history(),
                observation=action_result,
                full_history=self.action_history
            )
            
            # ⭐ 检查是否需要继续反思
            if hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                # 检查当前尝试次数
                current_attempt = getattr(self.task, 'current_attempt', 0)
                max_attempts = getattr(self.task, 'max_attempts', 1)
                
                # 如果答案错误且还有尝试机会
                if reward == 0 and current_attempt < max_attempts:
                    # 准备反思信息
                    reflection_info = {
                        "needs_reflection": True,
                        "current_attempt": current_attempt,
                        "max_attempts": max_attempts,
                        "previous_answer": action_result.get("content", ""),
                        "feedback": validation_info.get("feedback", "Your answer is incorrect. Please try again.")
                    }
                    
                    # 合并验证信息和反思信息
                    validation_info.update(reflection_info)
                    
                    # 返回非结束状态，允许继续尝试
                    return 0.0, False, f"Incorrect answer. Attempt {current_attempt}/{max_attempts}", validation_info
            
            # 正常返回（没有反思或反思结束）
            return reward, done, message, validation_info
        
        # DeepEyes工具反馈，给予中间奖励
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "deepeyes":
            # 根据工具使用给予奖励
            tool_used = action_result.get("tool_used", "")
            if tool_used in ["image_zoom_in_tool", "image_rotate_tool"]:
                reward = 0.15  # 成功使用工具
                message = f"Successfully used {tool_used}"
            else:
                reward = 0.1
                message = "DeepEyes tool executed"
                
            return reward, False, message, {"tool_executed": "deepeyes", "tool_used": tool_used}
        
        # Grounding DINO工具结果，给予中间奖励
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "grounding_dino":
            # 根据检测结果给予奖励
            result = action_result.get("result", {})
            num_detections = result.get("num_detections", 0)
            
            if num_detections > 0:
                reward = 0.2  # 成功检测到对象
                message = f"Detected {num_detections} objects successfully"
            else:
                reward = 0.05  # 执行成功但没有检测到对象
                message = "No objects detected"
                
            return reward, False, message, {"tool_executed": "grounding_dino", "detections": num_detections}
        
        # ChartMoE工具结果，给予中间奖励
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "chartmoe":
            # 根据任务类型给予奖励
            task_type = action_result.get("task_type", "unknown")
            output = action_result.get("output", "")
            
            if task_type == "to_table":
                # 简单计算表格行数
                lines = output.strip().split('\n') if output else []
                rows = len([l for l in lines if '|' in l and not l.strip().startswith('|--')])
                if rows > 0:
                    reward = 0.25  # 成功提取表格
                    message = f"Extracted table with {rows} data rows"
                else:
                    reward = 0.1  # 执行成功但表格为空
                    message = "Table extraction completed but no data found"
            
            elif task_type in ["describe", "analyze"]:
                reward = 0.3  # 描述或分析任务，较高奖励
                message = f"ChartMoE completed {task_type} task"
            
            elif task_type == "extract_data":
                reward = 0.25  # 数据提取
                message = "ChartMoE extracted chart data"
            
            else:
                reward = 0.15  # 其他任务类型
                message = f"ChartMoE completed task: {task_type}"
                
            return reward, False, message, {"tool_executed": "chartmoe", "task_type": task_type}
        
        # DiagramFormalizer工具结果，给予中间奖励
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "diagram_formalizer":
            # 根据任务类型给予奖励
            task_type = action_result.get("task_type", "unknown")
            
            if task_type == "formalize":
                reward = 0.3  # 形式化任务，较高奖励
                message = "Successfully formalized geometric problem"
            elif task_type == "solve":
                reward = 0.35  # 求解任务，最高奖励
                message = "Successfully solved geometric problem"
            elif task_type == "analyze":
                reward = 0.25  # 分析任务
                message = "Successfully analyzed geometric relationships"
            elif task_type == "extract_constraints":
                constraints = action_result.get("constraints", [])
                reward = 0.2 if constraints else 0.1
                message = f"Extracted {len(constraints)} geometric constraints"
            elif task_type == "prove":
                proof_steps = action_result.get("proof_steps", [])
                reward = 0.3 if proof_steps else 0.15
                message = f"Generated proof with {len(proof_steps)} steps"
            else:
                reward = 0.15  # 其他任务类型
                message = f"DiagramFormalizer completed task: {task_type}"
                
            return reward, False, message, {"tool_executed": "diagram_formalizer", "task_type": task_type}
        
        # ⭐ 新增：工具调用等待响应
        elif action_result.get("type") == "tool_call":
            # 工具调用已提交，等待执行
            return 0.0, False, "Tool call submitted", {"tool_call_pending": True}
        
        # ⭐ 新增：思考过程（如果启用了结构化输出）
        elif action_result.get("type") == "thinking":
            # 思考步骤，不给奖励但允许继续
            return 0.0, False, "Processing thoughts", {"thinking_step": True}
        
        # 对于其他动作，给予中间奖励
        elif action_result.get("status") == "SUCCESS":
            # 检查是否是有助于解决任务的动作
            helpful_actions = ["analyze_image", "extract_text", "detect_objects", "request_info"]
            action_name = action_result.get("action", "")
            
            if any(helpful in action_name for helpful in helpful_actions):
                reward = 0.1  # 有用的中间步骤
                message = f"Executed {action_name} successfully"
            else:
                reward = 0.05  # 中性动作
                message = f"Action {action_name} completed"
            
            return reward, False, message, {"intermediate_action": True}
        
        # ⭐ 新增：处理错误但不是失败的情况
        elif action_result.get("type") == "error" and action_result.get("error") != "Action failed":
            # 某些错误可能是可恢复的（如格式错误）
            error_msg = action_result.get("error", "Unknown error")
            
            # 检查是否是反思相关的错误
            if hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                current_attempt = getattr(self.task, 'current_attempt', 0)
                max_attempts = getattr(self.task, 'max_attempts', 1)
                
                if current_attempt < max_attempts:
                    # 给予机会重试
                    return -0.05, False, f"Error: {error_msg}. Please try again.", {
                        "error": error_msg,
                        "recoverable": True,
                        "attempt": current_attempt
                    }
            
            # 不可恢复的错误
            return -0.1, False, f"Error: {error_msg}", {"error": error_msg}
        
        else:
            # 其他未知情况
            return 0, False, "Action processed", {"unknown_action": True}
    
    def _load_current_image(self):
        """加载当前任务的图像"""
        if not self.task:
            return
        
        task_data = self.task.task_data
        image_path = task_data.get("image_path")
        
        if not image_path:
            logger.warning(f"No image path found for task {self.task.task_id}")
            self.current_image = None
            return
        
        image_path = Path(image_path)
        
        # 如果路径不是绝对路径，相对于数据集路径
        if not image_path.is_absolute():
            image_path = self.dataset_path / image_path
        
        try:
            if image_path.exists():
                self.current_image = Image.open(image_path)
                logger.debug(f"Loaded image: {image_path}")
            else:
                logger.error(f"Image not found: {image_path}")
                self.current_image = None
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            self.current_image = None
    
    def _get_obs(self) -> Dict[str, Any]:
        """获取当前观察 - 修复版本：不立即清空pending_tool_feedback"""
        obs = {
            "text": self.current_question,
            "question": self.current_question,  # 添加question字段，兼容agent
            "step": self.current_step,
            "max_steps": self.max_steps,
            "chat_history": self.chat.get_history(),
            "has_image": self.current_image is not None,
            "image": self.current_image if self.current_image else None,
            "image_path": str(self.task.task_data.get("image_path", "")) if self.task else "",
            "available_tools": list(self.tool_manager.keys())  # 添加可用工具列表
        }
        
        # ⭐ 修改：包含待处理的工具反馈，但不清空
        if self.pending_tool_feedback is not None:
            obs["tool_feedback"] = copy.deepcopy(self.pending_tool_feedback)  # 使用深拷贝
            obs["requires_response"] = self.requires_tool_response
            
            # 调试输出
            print(f"\n[DEBUG _get_obs] Including tool feedback in observation:")
            print(f"  - Tool feedback type: {type(self.pending_tool_feedback)}")
            if isinstance(self.pending_tool_feedback, dict):
                print(f"  - Tool feedback keys: {list(self.pending_tool_feedback.keys())}")
                print(f"  - Tool: {self.pending_tool_feedback.get('tool', 'N/A')}")
                
                # 特别关注 DiagramFormalizer 的 solution
                if self.pending_tool_feedback.get('tool') == 'diagram_formalizer':
                    print(f"  - Solution: '{self.pending_tool_feedback.get('solution', 'N/A')}'")
                    print(f"  - Task type: {self.pending_tool_feedback.get('task_type', 'N/A')}")
                    print(f"  - Formalized output preview: {str(self.pending_tool_feedback.get('formalized_output', ''))[:100]}")
                    print(f"  - Steps: {len(self.pending_tool_feedback.get('steps', []))} steps")
                    print(f"  - Analysis: {self.pending_tool_feedback.get('analysis', {})}")
                elif self.pending_tool_feedback.get('tool') == 'grounding_dino':
                    print(f"  - Num detections: {self.pending_tool_feedback.get('num_detections', 'N/A')}")
                elif self.pending_tool_feedback.get('tool') == 'chartmoe':
                    print(f"  - Task type: {self.pending_tool_feedback.get('task_type', 'N/A')}")
                    print(f"  - Output preview: {str(self.pending_tool_feedback.get('output', ''))[:100]}")
                elif self.pending_tool_feedback.get('tool') == 'deepeyes':
                    print(f"  - Tool used: {self.pending_tool_feedback.get('tool_used', 'N/A')}")
                    print(f"  - Has processed image: {self.pending_tool_feedback.get('has_processed_image', False)}")
                    
            print(f"  - requires_response: {self.requires_tool_response}")

            # ⚠️ 关键修改：删除这两行！不要在这里清空
            # self.pending_tool_feedback = None  # ❌ 删除
            # self.requires_tool_response = False  # ❌ 删除
        
        # 添加动作相关信息
        if self.enable_actions:
            obs["available_actions"] = self.action_set.list_actions()
            obs["action_history"] = self.action_history
        
        # 添加DeepEyes相关信息
        if self.enable_deepeyes_tools:
            obs["deepeyes_enabled"] = self.deepeyes_tool is not None
        
        # 添加Grounding DINO相关信息
        if self.enable_grounding_dino:
            obs["grounding_dino_enabled"] = self.grounding_dino_tool is not None
        
        # 添加ChartMoE相关信息
        if self.enable_chartmoe:
            obs["chartmoe_enabled"] = self.chartmoe_tool is not None
        
        # 添加DiagramFormalizer相关信息
        if self.enable_diagram_formalizer:
            obs["diagram_formalizer_enabled"] = self.diagram_formalizer_tool is not None
        
        # 添加任务信息
        if self.task:
            try:
                task_obs = self.task.get_observation()
                # 避免覆盖关键字段
                for key, value in task_obs.items():
                    if key not in ["text", "question", "image", "image_path", "tool_feedback"]:  # ⭐ 添加tool_feedback到保护列表
                        obs[key] = value
            except Exception as e:
                logger.debug(f"Failed to get task observation: {e}")
        
        # 添加时间信息
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            obs["elapsed_time"] = elapsed_time
            if self.time_limit:
                obs["remaining_time"] = max(0, self.time_limit - elapsed_time)
        
        return obs
    
    def close(self):
        """关闭环境并清理资源"""
        if self.task:
            try:
                self.task.teardown()
            except Exception as e:
                logger.debug(f"Task teardown failed: {e}")
        
        self.task = None
        self.current_image = None
        if self.chat:
            self.chat.clear()
        self.current_step = 0
        self.action_history = []
        self.deepeyes_tool = None
        self.grounding_dino_tool = None
        self.chartmoe_tool = None
        self.diagram_formalizer_tool = None
        
        # ⭐ 清理工具反馈状态
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        logger.debug("VisionQAEnv closed")