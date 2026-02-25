# vlm_gym/environments/vision_qa_env.py
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from .action import VLMActionSet

from vlm_gym.environments.tools.mm_process.visual_toolbox_v2 import VisualToolBoxV2
from vlm_gym.environments.tools.mm_process.visual_toolbox import VisualToolBoxV5
from vlm_gym.environments.action.parser import parse_deepeyes_action
from vlm_gym.environments.tools.chart import ChartMoETool


# Import Grounding DINO tool
from vlm_gym.environments.tools.grounding_dino import GroundingDINOTool

VisualToolBox = VisualToolBoxV2 

logger = logging.getLogger(__name__)


class SimpleChatManager:
    """Simple chat history manager"""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: Any):
        """Add a message"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.history.append(message)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get history"""
        return self.history.copy()
    
    def clear(self):
        """Clear history"""
        self.history.clear()


class VisionQAEnv:
    """Vision QA Environment - Supports complex action system, DeepEyes tools, Grounding DINO, and ChartMoE"""
    
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
        deepeyes_version: str = "v2",
        enable_grounding_dino: bool = False,
        grounding_dino_config: Dict[str, Any] = None,
        enable_chartmoe: bool = False,
        chartmoe_config: Dict[str, Any] = None,
        **kwargs
    ):
        """Initialize the Vision QA environment"""
        self.dataset_path = Path(dataset_path)
        self.task_kwargs = task_kwargs or {}
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.enable_actions = enable_actions
        self.enable_deepeyes_tools = enable_deepeyes_tools
        self.deepeyes_version = deepeyes_version
        self.enable_grounding_dino = enable_grounding_dino
        self.enable_chartmoe = enable_chartmoe
        
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        # Initialize action system
        if self.enable_actions:
            self.action_set = VLMActionSet(custom_actions=custom_actions)
        else:
            self.action_set = None
        
        # Initialize DeepEyes tool
        self.deepeyes_tool = None
        if self.enable_deepeyes_tools:
            try:
                if deepeyes_version == "v2":
                    self.deepeyes_tool = VisualToolBoxV2("visual_toolbox_v2", None, None)
                else:
                    self.deepeyes_tool = VisualToolBox("visual_toolbox", None, None)
                logger.info(f"DeepEyes tool initialized: visual_toolbox_{deepeyes_version}")
            except Exception as e:
                logger.error(f"Failed to initialize DeepEyes tool: {e}")
                self.deepeyes_tool = None
        
        # Initialize Grounding DINO tool
        self.grounding_dino_tool = None
        if self.enable_grounding_dino:
            try:
                config = grounding_dino_config or {}
                self.grounding_dino_tool = GroundingDINOTool(config)
                logger.info("Grounding DINO tool initialized")
                # ⭐ Added debug output
                print(f"[DEBUG] Grounding DINO tool initialized: {self.grounding_dino_tool}")
                print(f"[DEBUG] Tool type: {type(self.grounding_dino_tool)}")
            except Exception as e:
                logger.error(f"Failed to initialize Grounding DINO tool: {e}")
                # ⭐ Added detailed error info
                print(f"[DEBUG] ❌ Grounding DINO initialization failed:")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {str(e)}")
                import traceback
                traceback.print_exc()
                self.grounding_dino_tool = None
        
        # Initialize ChartMoE tool
        self.chartmoe_tool = None
        if self.enable_chartmoe:
            try:
                # ⭐ Modified: Use your FixedChartMoETool
                from chartmoe_vlmgym_tool import ChartMoETool
                config = chartmoe_config or {}
                self.chartmoe_tool = ChartMoETool(config)
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
        
        # Tool manager
        self.tool_manager = {}
        if self.deepeyes_tool:
            self.tool_manager['deepeyes'] = self.deepeyes_tool
        if self.grounding_dino_tool:
            self.tool_manager['grounding_dino'] = self.grounding_dino_tool
        if self.chartmoe_tool:
            self.tool_manager['chartmoe'] = self.chartmoe_tool
        
        logger.info(f"Registered tools: {list(self.tool_manager.keys())}")
        
        # Initialize chat manager
        self.chat = SimpleChatManager()
        
        # Environment state
        self.current_step = 0
        self.task = None
        self.start_time = None
        
        # Current image and question (managed by the environment)
        self.current_image = None
        self.current_question = ""
        self.task_goal = ""
        self.task_info = {}
        self.action_history = []  # Record executed actions
        
        # DeepEyes-related state
        self.deepeyes_initialized = False
        self.deepeyes_interaction_count = 0
        
        # Task entry point (set externally)
        self.task_entrypoint = None
        
        # ⭐ New: Store pending tool feedback
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available tools (for agent queries)"""
        available_tools = {}
        
        # DeepEyes tool
        if self.deepeyes_tool and self.enable_deepeyes_tools:
            available_tools['deepeyes'] = {
                "name": "deepeyes",
                "description": "DeepEyes visual processing tool",
                "initialized": self.deepeyes_initialized
            }
        
        # Grounding DINO tool
        if self.grounding_dino_tool and self.enable_grounding_dino:
            # ⭐ Modified: Add default capability info if tool doesn't have get_capabilities method
            if hasattr(self.grounding_dino_tool, 'get_capabilities'):
                available_tools['grounding_dino'] = self.grounding_dino_tool.get_capabilities()
            else:
                available_tools['grounding_dino'] = {
                    "name": "grounding_dino",
                    "description": "Open-vocabulary object detection tool",
                    "capabilities": ["object_detection", "phrase_grounding"]
                }
        
        # ChartMoE tool
        if self.chartmoe_tool and self.enable_chartmoe:
            # ⭐ Modified: Add default capability info
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
        
        return available_tools
    
    def reset(self, task_id: str = None, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment"""
        # Reset state
        self.current_step = 0
        self.start_time = time.time()
        self.chat.clear()
        self.action_history = []
        self.deepeyes_initialized = False
        self.deepeyes_interaction_count = 0
        
        # ⭐ Reset tool feedback state
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        # Create task instance
        if not self.task_entrypoint:
            raise ValueError("task_entrypoint not set. Please set env.task_entrypoint before reset.")
        
        try:
            # Prepare task arguments, pass action_set to task (if enabled)
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
        
        # Set up task
        try:
            self.task_goal, self.task_info = self.task.setup()
            logger.debug(f"Task {task_id} setup complete")
        except Exception as e:
            logger.error(f"Failed to setup task {task_id}: {e}")
            raise
        
        # Load image
        self._load_current_image()
        
        # Set current question
        task_data = self.task.task_data
        self.current_question = task_data.get("question", "")
        
        # Initialize DeepEyes tool (if enabled)
        if self.deepeyes_tool and self.current_image:
            try:
                raw_prompt = [{
                    "role": "user",
                    "content": self.current_question
                }]
                origin_multi_modal_data = {"image": [self.current_image]}
                
                self.deepeyes_tool.reset(raw_prompt, None, origin_multi_modal_data)
                self.deepeyes_initialized = True
                logger.debug("DeepEyes tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset DeepEyes tool: {e}")
                self.deepeyes_initialized = False
        
        # Initialize Grounding DINO tool (if enabled)
        if self.grounding_dino_tool and self.current_image:
            try:
                self.grounding_dino_tool.reset(self.current_image)
                logger.debug("Grounding DINO tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset Grounding DINO tool: {e}")
        
        # Initialize ChartMoE tool (if enabled)
        if self.chartmoe_tool and self.current_image:
            try:
                self.chartmoe_tool.reset(self.current_image)
                logger.debug("ChartMoE tool reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset ChartMoE tool: {e}")
        
        # Add initial system message
        self.chat.add_message(
            role="system",
            content={
                "text": self.task_goal,
                "task_info": self.task_info
            }
        )
        
        # Build observation
        observation = self._get_obs()
        
        # Build info (includes action space description)
        info = {
            "task_id": task_id,
            "task_goal": self.task_goal,
            "task_info": self.task_info,
            "max_steps": self.max_steps,
            "time_limit": self.time_limit,
            "actions_enabled": self.enable_actions,
            "deepeyes_enabled": self.enable_deepeyes_tools and self.deepeyes_initialized,
            "grounding_dino_enabled": self.enable_grounding_dino and self.grounding_dino_tool is not None,
            "chartmoe_enabled": self.enable_chartmoe and self.chartmoe_tool is not None,
            "available_tools": self.get_available_tools()
        }
        
        if self.enable_actions:
            info["available_actions"] = self.action_set.list_actions()
            info["action_space_description"] = self.action_set.describe(with_examples=False)
        
        return observation, info
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:

        """Execute one step and return observation, reward, done status, etc."""
        
        if not self.task:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.current_step += 1
        
        # ===== Added debug output =====
        print(f"\n[DEBUG VisionQAEnv.step] Step {self.current_step}")
        print(f"[DEBUG VisionQAEnv.step] Action type: {type(action)}")
        print(f"[DEBUG VisionQAEnv.step] Action length: {len(str(action))}")
        print(f"[DEBUG VisionQAEnv.step] Action preview: {str(action)[:200]}...")
        print(f"[DEBUG VisionQAEnv.step] Contains <tool_call>: {'<tool_call>' in str(action)}")
        print(f"[DEBUG VisionQAEnv.step] DeepEyes enabled: {self.enable_deepeyes_tools}")
        print(f"[DEBUG VisionQAEnv.step] DeepEyes initialized: {self.deepeyes_initialized}")
        print(f"[DEBUG VisionQAEnv.step] Grounding DINO enabled: {self.enable_grounding_dino}")
        print(f"[DEBUG VisionQAEnv.step] ChartMoE enabled: {self.enable_chartmoe}")
    
        
        # Check if timed out
        truncated = False
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            truncated = True
        
        # Check if maximum steps exceeded
        if self.current_step > self.max_steps:
            truncated = True
        
        # Record user action
        self.chat.add_message(
            role="user",
            content={
                "text": action,
                "step": self.current_step
            }
        )
        
        # Execute action
        try:
            action_result = self._execute_action(action)
            logger.debug(f"Action executed: {action_result.get('type', 'unknown')}")
            
            # Record action history
            self.action_history.append({
                "step": self.current_step,
                "action": action,
                "result": action_result
            })
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            action_result = {
                "type": "error",
                "status": "FAILED",
                "content": str(e),
                "error": str(e)
            }
        
        # Record assistant reply
        self.chat.add_message(
            role="assistant", 
            content=action_result
        )
        
        # Handle task validation
        reward, done, message, validation_info = self._handle_task_validation(
            action, action_result
        )
        
        # If truncated but not done, set done status
        if truncated and not done:
            done = True
            if reward == 0:
                reward = -0.1  # Slight penalty
            message = f"Task truncated: {message}"
        
        # ⭐ Modified: Set tool feedback before getting observation
        if action_result.get("type") in ["deepeyes_feedback", "tool_result"]:
            self.requires_tool_response = True

            # Special handling for Grounding DINO
            if action_result.get("tool") == "grounding_dino":
                self.pending_tool_feedback = {
                    "tool": "grounding_dino",
                    "detections": action_result.get("detections", {}),
                    "original_question": self.current_question,
                    "query": action_result.get("query", ""),  # Add query term
                    "num_detections": action_result.get("detections", {}).get("num_detections", 0)
                }
                print(f"\n[DEBUG VisionQAEnv.step] Set pending Grounding DINO feedback:")
                print(f"  - num_detections: {self.pending_tool_feedback['num_detections']}")
                print(f"  - requires_tool_response: {self.requires_tool_response}")
            
            # Special handling for ChartMoE
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
            
            # Special handling for DeepEyes
            elif action_result.get("type") == "deepeyes_feedback":
                self.pending_tool_feedback = action_result.get("tool_feedback", {})
                print(f"\n[DEBUG VisionQAEnv.step] Set pending DeepEyes feedback:")
                print(f"  - has_observation: {'observation' in self.pending_tool_feedback}")
                print(f"  - has_processed_images: {len(action_result.get('processed_images', []))}")
                print(f"  - requires_tool_response: {self.requires_tool_response}")
                
            else:
                self.pending_tool_feedback = action_result.get("tool_feedback", action_result.get("result", {}))
        
        # Build observation (now includes tool feedback)
        observation = self._get_obs()
        
        # Build info
        info = {
            "step": self.current_step,
            "action_result": action_result,
            "validation": validation_info,
            "message": message,
            "truncated": truncated,
            "chat_history": self.chat.get_history(),
            "action_history": self.action_history
        }
        
        if self.enable_deepeyes_tools:
            info["deepeyes_interactions"] = self.deepeyes_interaction_count
        
        return observation, reward, done, truncated, info
    
    def _parse_tool_call(self, action: str) -> Tuple[str, Any]:
        """Parse tool call format"""
        import re
        
        # Try to match <tool_call> format
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
        """Execute action - Supports DeepEyes tools, Grounding DINO, ChartMoE, standard action system, and direct answers"""
        if not action.strip():
            return {
                "type": "error",
                "status": "FAILED",
                "content": "Empty action provided",
                "error": "Empty action"
            }
        
        # ===== Added debug output =====
        print(f"\n[DEBUG _execute_action] START")
        print(f"  - Action length: {len(action)}")
        print(f"  - Action preview (first 200 chars): {action[:200]}")
        print(f"  - Contains <tool_call>: {'<tool_call>' in action}")
        print(f"  - Contains <answer>: {'<answer>' in action}")
        print(f"  - Contains <think>: {'<think>' in action}")
        print(f"  - DeepEyes enabled: {self.enable_deepeyes_tools}")
        print(f"  - DeepEyes initialized: {self.deepeyes_initialized}")
        print(f"  - Grounding DINO enabled: {self.enable_grounding_dino}")
        print(f"  - ChartMoE enabled: {self.enable_chartmoe}")
        
        # First try to parse generic tool call format
        tool_type, tool_content = self._parse_tool_call(action)
        
        # ⭐ Added debug info
        print(f"\n[DEBUG] Tool parsing result:")
        print(f"  - tool_type: {tool_type}")
        print(f"  - tool_content: {tool_content}")
       
        if tool_type == "tool_call":
            # Handle generic tool call
            tool_name = tool_content.get("tool", "")
            if not tool_name:
                tool_name = tool_content.get("name", "")
            
            # ⭐ Added more debug info
            print(f"\n[DEBUG] Tool call processing:")
            print(f"  - tool_name: '{tool_name}'")
            print(f"  - grounding_dino_tool is None: {self.grounding_dino_tool is None}")
            print(f"  - grounding_dino_tool type: {type(self.grounding_dino_tool)}")
            print(f"  - chartmoe_tool is None: {self.chartmoe_tool is None}")
            print(f"  - chartmoe_tool type: {type(self.chartmoe_tool)}")
            print(f"  - Available tools: {list(self.tool_manager.keys())}")
            
            if tool_name == "grounding_dino" and self.grounding_dino_tool:
                print(f"[DEBUG] Executing Grounding DINO tool call")
                return self._execute_grounding_dino(tool_content)
            
            elif tool_name == "chartmoe" and self.chartmoe_tool:
                print(f"[DEBUG] Executing ChartMoE tool call")
                return self._execute_chartmoe(tool_content)
            
            elif tool_name == "image_zoom_in_tool" and self.deepeyes_tool:
                print(f"[DEBUG] Forwarding to DeepEyes through tool_call")
                # Convert format and execute DeepEyes
                return self._execute_deepeyes_action(action, "tool_call", tool_content)
            
            elif tool_name == "deepeyes" and self.deepeyes_tool:
                print(f"[DEBUG] Forwarding to DeepEyes through tool_call")
                # Convert format and execute DeepEyes
                return self._execute_deepeyes_action(action, "tool_call", tool_content)
            
            else:
                # ⭐ More detailed error info
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
            # JSON parse error
            return {
                "type": "error",
                "status": "FAILED",
                "content": f"Invalid tool_call format: {tool_content.get('error', 'Unknown error')}",
                "error": "Invalid tool_call format",
                "raw_json": tool_content.get('raw_json', '')
            }
        
        # If not a generic tool call format, check if it's DeepEyes format
        if self.enable_deepeyes_tools and self.deepeyes_initialized:
            # Call parser
            action_type, content = parse_deepeyes_action(action)
            
            # ===== Key debug output =====
            print(f"\n[DEBUG] DeepEyes Parser Result:")
            print(f"  - Parsed action_type: {action_type}")
            print(f"  - Content type: {type(content)}")
            
            # ⭐ Added DeepEyes specific format detection
            if "<tool_call>" in action and "image_zoom_in_tool" in action:
                print(f"\n[DEBUG] DeepEyes tool call detected:")
                print(f"  - deepeyes_tool is None: {self.deepeyes_tool is None}")
                if self.deepeyes_tool is not None:
                    print(f"  - deepeyes_tool type: {type(self.deepeyes_tool)}")
                
                # Extract tool call content
                try:
                    tool_start = action.find("<tool_call>") + 11
                    tool_end = action.find("</tool_call>")
                    if tool_start > 10 and tool_end > tool_start:
                        tool_json = action[tool_start:tool_end]
                        tool_data = json.loads(tool_json)
                        print(f"  - Tool name: {tool_data.get('name')}")
                        print(f"  - Arguments: {tool_data.get('arguments', {})}")
                        
                        if self.deepeyes_tool is not None:
                            print(f"[DEBUG] Executing DeepEyes tool call")
                            
                            # Get arguments
                            bbox = tool_data.get("arguments", {}).get("bbox", [])
                            print(f"[DEBUG] Bbox coordinates: {bbox}")
                            
                            # Execute tool
                            if self.deepeyes_version == "v2":
                                print(f"[DEBUG] Calling visual_toolbox_v2...")
                                result = self.deepeyes_tool(self.current_image, bbox)
                            else:
                                print(f"[DEBUG] Calling DeepEyesV1.execute()...")
                                result = self.deepeyes_tool.execute(
                                    tool_name=tool_data["name"],
                                    image_path=self.current_image,
                                    **tool_data.get("arguments", {})
                                )
                            
                            print(f"[DEBUG] DeepEyes execution result type: {type(result)}")
                            if isinstance(result, dict):
                                print(f"[DEBUG] Result keys: {list(result.keys())}")
                            
                            # Return processed result
                            return {
                                "type": "tool_execution",
                                "status": "SUCCESS",
                                "tool": "deepeyes",
                                "tool_name": tool_data.get("name"),
                                "result": result,
                                "requires_response": True,
                                "source": "deepeyes_direct"
                            }
                except Exception as e:
                    print(f"[DEBUG] Error processing DeepEyes tool call: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process parse result
            if action_type in ["tool_call", "answer", "think"]:
                print(f"[DEBUG] Processing DeepEyes {action_type}")
                return self._execute_deepeyes_action(action, action_type, content)
                
            elif action_type == "error":
                print(f"[DEBUG] DeepEyes parse error detected")
                error_msg = content.get('error', 'Unknown error') if isinstance(content, dict) else str(content)
                
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": f"DeepEyes action parse error: {error_msg}",
                    "error": "DeepEyes parse error",
                    "raw_json": content.get('raw_json', '') if isinstance(content, dict) else None
                }
        
        # If not a tool format, check standard action system
        if self.enable_actions and self.action_set:
            print(f"[DEBUG] Checking standard action system")
            
            # First try to validate if it's a valid action
            if self.action_set.validate_action(action):
                print(f"[DEBUG] Valid standard action detected")
                # Execute action
                result = self.action_set.execute_action(action)
                
                # Standardize result format
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
            
            # If not a valid action, check if it's a direct answer
            elif self._is_direct_answer(action):
                print(f"[DEBUG] Direct answer detected")
                return self._process_direct_answer(action)
            
            else:
                # Neither a valid action nor an answer format
                print(f"[DEBUG] Invalid action format")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": f"Invalid action format: {action}",
                    "error": "Invalid action format",
                    "available_actions": self.action_set.list_actions()
                }
        
        else:
            # Action system not enabled, use original simple processing
            print(f"[DEBUG] No action system enabled, processing as direct answer")
            return self._process_direct_answer(action)
    
    
    def _execute_grounding_dino(self, tool_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Grounding DINO tool call"""
        
        # ⭐ 1. Print at function start
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
        
        # Get parameters
        parameters = tool_content.get("parameters", {})
        
        # ⭐ 2. Print parameter info
        print(f"  - Parameters extracted: {parameters}")
        print(f"  - Caption: {parameters.get('caption', 'N/A')}")
        
        try:
            # ⭐ 3. Print before execution
            print(f"\n[DEBUG] Calling grounding_dino_tool.execute()...")
            
            # Execute detection
            result = self.grounding_dino_tool.execute(parameters)
            
            # ⭐ 4. Print result immediately after execution
            print(f"\n[DEBUG] Grounding DINO execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Num detections: {result.get('num_detections', 0) if isinstance(result, dict) else 'N/A'}")
            
            # Check for errors
            if "error" in result:
                # ⭐ 5. Print on error
                print(f"  - ERROR in result: {result['error']}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result["error"],
                    "error": result["error"]
                }
            
            # ⭐ 6. Print detailed success info
            print(f"\n[DEBUG] Processing successful detection result:")
            print(f"  - Boxes: {result.get('boxes', [])[:3]}...")  # Only print first 3
            print(f"  - Phrases: {result.get('phrases', [])}")
            print(f"  - Logits: {result.get('logits', [])[:3]}...")  # Only print first 3
            print(f"  - Image size: {result.get('size', 'N/A')}")
            
            # Success return
            return_dict = {
                "type": "tool_result",
                "status": "SUCCESS",
                "tool": "grounding_dino",
                "content": f"Detected {result.get('num_detections', 0)} objects",
                "result": result,
                "requires_tool_response": True,  # ⭐ Added this flag
                "detections": {
                    "boxes": result.get("boxes", []),
                    "phrases": result.get("phrases", []),
                    "logits": result.get("logits", []),
                    "num_detections": result.get("num_detections", 0)
                },
                "query": parameters.get("caption", "")  # Save query term
            }
            
            # ⭐ 7. Print before returning
            print(f"\n[DEBUG] Returning tool_result:")
            print(f"  - Type: {return_dict['type']}")
            print(f"  - Status: {return_dict['status']}")
            print(f"  - requires_tool_response: {return_dict.get('requires_tool_response', False)}")
            print(f"  - Content: {return_dict['content']}")
            
            return return_dict
            
        except Exception as e:
            # ⭐ 8. Print on exception
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
        """Execute ChartMoE tool call"""
        
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
        
        # ⭐ Modified: Correctly handle parameter format
        # ChartMoE expects parameters in format {"task": "xxx"} or {"prompt": "xxx"}
        # But tool_content may contain {"tool": "chartmoe", "parameters": {...}}
        
        # Extract actual parameters
        if "parameters" in tool_content:
            parameters = tool_content["parameters"]
        else:
            # Remove "tool" key, keep all other parameters
            parameters = {k: v for k, v in tool_content.items() if k != "tool"}
        
        print(f"  - Parameters extracted: {parameters}")
        print(f"  - Task type: {parameters.get('task', 'N/A')}")
        print(f"  - Prompt: {parameters.get('prompt', 'N/A')[:50] if parameters.get('prompt') else 'N/A'}...")
        
        try:
            print(f"\n[DEBUG] Calling chartmoe_tool.execute()...")
            
            # Execute ChartMoE
            result = self.chartmoe_tool.execute(parameters)
            
            print(f"\n[DEBUG] ChartMoE execution complete")
            print(f"  - Result type: {type(result)}")
            print(f"  - Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Success: {result.get('success', False) if isinstance(result, dict) else 'N/A'}")
            print(f"  - Task type: {result.get('task_type', 'N/A') if isinstance(result, dict) else 'N/A'}")
            
            # Check for errors
            if not result.get('success', True) or "error" in result:
                print(f"  - ERROR in result: {result.get('error', 'Unknown error')}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": result.get("error", "ChartMoE execution failed"),
                    "error": result.get("error", "Unknown error"),
                    "error_type": result.get("error_type", "Unknown")
                }
            
            # Success case
            processed_output = result.get("processed_output", "")
            task_type = result.get("task_type", "unknown")
            
            print(f"\n[DEBUG] Processing successful ChartMoE result:")
            print(f"  - Task type: {task_type}")
            print(f"  - Output length: {len(str(processed_output))}")
            print(f"  - Output preview: {str(processed_output)[:200]}...")
            
            # Build return content
            if task_type == "to_table":
                # Estimate number of table rows
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
            
            # Success return
            return_dict = {
                "type": "tool_result",
                "status": "SUCCESS",
                "tool": "chartmoe",
                "content": content,
                "result": result,
                "requires_tool_response": True,
                "task_type": task_type,
                "output": processed_output,  # Use string output directly
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
        
    def _execute_deepeyes_action(self, action: str, action_type: str, content: Any) -> Dict[str, Any]:
        """Execute a DeepEyes-format action"""
        
        # ===== Debug output start =====
        print(f"\n{'='*60}")
        print(f"[DEBUG VisionQAEnv._execute_deepeyes_action] START")
        print(f"  - Action type: {action_type}")
        print(f"  - Content type: {type(content)}")
        print(f"  - Content: {content}")
        print(f"  - DeepEyes tool available: {self.deepeyes_tool is not None}")
        print(f"  - DeepEyes tool class: {self.deepeyes_tool.__class__.__name__ if self.deepeyes_tool else 'None'}")
        print(f"  - DeepEyes interaction count: {self.deepeyes_interaction_count}")
        print(f"  - Raw action (first 500 chars):")
        print(f"    {action[:500]}")
        print(f"{'='*60}")
        
        self.deepeyes_interaction_count += 1
        
        if action_type == "tool_call":
            # Execute DeepEyes tool call
            try:
                # ⭐ New: Initialize tool before calling execute
                if hasattr(self.deepeyes_tool, 'reset') and self.current_image:
                    try:
                        print(f"[DEBUG] Initializing DeepEyes tool with current image")
                        # Prepare data
                        raw_prompt = ""  # Can be empty
                        multi_modal_data = {"image": [self.current_image]}
                        
                        # Call reset to initialize
                        self.deepeyes_tool.reset(
                            raw_prompt=raw_prompt,
                            multi_modal_data=multi_modal_data,
                            origin_multi_modal_data=multi_modal_data
                        )
                        print(f"[DEBUG] DeepEyes tool initialized successfully")
                    except Exception as e:
                        print(f"[DEBUG] Failed to initialize DeepEyes tool: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"\n[DEBUG] Calling DeepEyes tool.execute()...")
                print(f"[DEBUG] Tool instance: {self.deepeyes_tool}")
                
                obs, reward, done, info = self.deepeyes_tool.execute(action)
                
                print(f"\n[DEBUG] DeepEyes execution result:")
                print(f"  - Done: {done}")
                print(f"  - Reward: {reward}")
                print(f"  - Info: {info}")
                print(f"  - Observation type: {type(obs)}")
                if isinstance(obs, dict):
                    print(f"  - Observation keys: {list(obs.keys())}")
                    if "multi_modal_data" in obs:
                        mmd = obs["multi_modal_data"]
                        print(f"  - Multi-modal data keys: {list(mmd.keys()) if isinstance(mmd, dict) else 'N/A'}")
                        if isinstance(mmd, dict) and "image" in mmd:
                            print(f"  - Number of processed images: {len(mmd['image']) if isinstance(mmd['image'], list) else 1}")
                
                if done:
                    print(f"\n[DEBUG] DeepEyes indicates done=True")
                    # DeepEyes found an answer (action contains <answer> tag)
                    import re
                    answer_match = re.search(r'<answer>(.*?)</answer>', action, re.DOTALL)
                    if answer_match:
                        answer = answer_match.group(1).strip()
                        print(f"[DEBUG] Found answer in action: {answer}")
                        return {
                            "type": "answer",
                            "status": "SUCCESS",
                            "content": answer,
                            "source": "deepeyes",
                            "tool_info": info
                        }
                    else:
                        print(f"[DEBUG] No answer tag found, returning deepeyes_done")
                        return {
                            "type": "deepeyes_done",
                            "status": "SUCCESS",
                            "content": "DeepEyes execution completed",
                            "tool_info": info
                        }
                else:
                    print(f"\n[DEBUG] DeepEyes needs more interaction (done=False)")
                    # Need to continue interaction, return tool observation
                    tool_feedback = {
                        "observation": obs,
                        "reward": reward,
                        "info": info
                    }
                    
                    # If obs contains processed images
                    processed_images = []
                    if isinstance(obs, dict) and "multi_modal_data" in obs:
                        if "image" in obs["multi_modal_data"]:
                            processed_images = obs["multi_modal_data"]["image"]
                            print(f"[DEBUG] Found {len(processed_images)} processed images")
                    
                    result = {
                        "type": "deepeyes_feedback",
                        "status": "SUCCESS",
                        "content": "Tool executed, awaiting response",
                        "deepeyes_feedback": True,
                        "tool_feedback": tool_feedback,
                        "processed_images": processed_images,
                        "intermediate_reward": reward
                    }
                    
                    print(f"[DEBUG] Returning deepeyes_feedback result")
                    return result
                    
            except Exception as e:
                print(f"\n[DEBUG] DeepEyes execution failed with error: {e}")
                import traceback
                traceback.print_exc()
                
                logger.error(f"DeepEyes execution failed: {e}")
                return {
                    "type": "error",
                    "status": "FAILED",
                    "content": f"DeepEyes execution error: {str(e)}",
                    "error": str(e)
                }
        
        elif action_type == "answer":
            print(f"\n[DEBUG] Processing final answer: {content}")
            # DeepEyes-format final answer
            return {
                "type": "answer",
                "status": "SUCCESS",
                "content": content,
                "source": "deepeyes_direct"
            }
        
        elif action_type == "think":
            print(f"\n[DEBUG] Processing thinking: {content[:100]}...")
            # Thinking only, no action execution
            return {
                "type": "thinking",
                "status": "SUCCESS",
                "content": content,
                "source": "deepeyes"
            }
        
        else:
            print(f"\n[DEBUG] Unknown action type, processing as direct answer")
            # Text format, treat as normal response
            return self._process_direct_answer(action)
    
    def _is_direct_answer(self, action: str) -> bool:
        """Check if the action is a direct answer format"""
        cleaned = action.strip()
        # Check various answer formats
        return (
            cleaned.startswith("answer_question(") or
            cleaned.startswith("Answer:") or
            cleaned.startswith("answer:") or
            # Check <answer> tag
            "<answer>" in cleaned or
            # Simple text answer (no function call format)
            ("(" not in cleaned and "<" not in cleaned)
        )
    
    def _process_direct_answer(self, action: str) -> Dict[str, Any]:
        """Process a direct answer (compatible with original logic)"""
        cleaned_action = action.strip()
        
        # Check <answer> tag
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
        
        # Extract answer content
        if cleaned_action.startswith("answer_question(") and cleaned_action.endswith(")"):
            # Extract answer from formatted action
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
            # Remove common prefixes
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
        """Handle task validation logic - supports reflection mechanism"""
        # If action execution failed, apply small penalty
        if action_result.get("status") == "FAILED":
            return -0.1, False, action_result.get("error", "Action failed"), {"error": True}
        
        # Check if the result is an answer type
        if action_result.get("type") == "answer":
            # ⭐ New: First check if the task has a step method (supports reflection)
            if hasattr(self.task, 'step') and hasattr(self.task, 'enable_reflection'):
                # Use the task's step method to handle answers and reflection
                try:
                    # Call task's step method
                    task_obs, task_reward, task_done, task_truncated, task_info = self.task.step(action)
                    
                    # If task is not done (needs reflection)
                    if not task_done and self.task.enable_reflection:
                        # Update environment observation with reflection-related info
                        if isinstance(task_obs, dict):
                            # Merge task-returned observation into environment observation
                            self.current_question = task_obs.get('question', self.current_question)
                            self.task_info.update(task_obs)
                        
                        # Return intermediate result, allow continuation
                        return task_reward, False, task_info.get('message', 'Reflection needed'), task_info
                    
                    # Task completed (correct or max attempts reached)
                    return task_reward, task_done, task_info.get('message', 'Task completed'), task_info
                    
                except Exception as e:
                    logger.error(f"Error calling task.step: {e}")
                    # If task.step fails, fall back to validate method
            
            # Fallback: Use task's validation logic (old way)
            reward, done, message, validation_info = self.task.validate(
                chat_history=self.chat.get_history(),
                observation=action_result,
                full_history=self.action_history
            )
            
            # ⭐ Check if reflection should continue
            if hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                # Check current attempt count
                current_attempt = getattr(self.task, 'current_attempt', 0)
                max_attempts = getattr(self.task, 'max_attempts', 1)
                
                # If answer is wrong and there are remaining attempts
                if reward == 0 and current_attempt < max_attempts:
                    # Prepare reflection info
                    reflection_info = {
                        "needs_reflection": True,
                        "current_attempt": current_attempt,
                        "max_attempts": max_attempts,
                        "previous_answer": action_result.get("content", ""),
                        "feedback": validation_info.get("feedback", "Your answer is incorrect. Please try again.")
                    }
                    
                    # Merge validation info and reflection info
                    validation_info.update(reflection_info)
                    
                    # Return non-terminated state, allow more attempts
                    return 0.0, False, f"Incorrect answer. Attempt {current_attempt}/{max_attempts}", validation_info
            
            # Normal return (no reflection or reflection ended)
            return reward, done, message, validation_info
        
        # DeepEyes intermediate feedback, grant intermediate reward
        elif action_result.get("type") == "deepeyes_feedback":
            intermediate_reward = action_result.get("intermediate_reward", 0.1)
            return intermediate_reward, False, "DeepEyes tool executed", {"deepeyes_intermediate": True}
        
        # Grounding DINO tool result, grant intermediate reward
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "grounding_dino":
            # Grant reward based on detection result
            result = action_result.get("result", {})
            num_detections = result.get("num_detections", 0)
            
            if num_detections > 0:
                reward = 0.2  # Successfully detected objects
                message = f"Detected {num_detections} objects successfully"
            else:
                reward = 0.05  # Execution succeeded but no objects detected
                message = "No objects detected"
                
            return reward, False, message, {"tool_executed": "grounding_dino", "detections": num_detections}
        
        # ChartMoE tool result, grant intermediate reward
        elif action_result.get("type") == "tool_result" and action_result.get("tool") == "chartmoe":
            # Grant reward based on task type
            task_type = action_result.get("task_type", "unknown")
            output = action_result.get("output", "")
            
            if task_type == "to_table":
                # Simple table row count
                lines = output.strip().split('\n') if output else []
                rows = len([l for l in lines if '|' in l and not l.strip().startswith('|--')])
                if rows > 0:
                    reward = 0.25  # Successfully extracted table
                    message = f"Extracted table with {rows} data rows"
                else:
                    reward = 0.1  # Execution succeeded but table is empty
                    message = "Table extraction completed but no data found"
            
            elif task_type in ["describe", "analyze"]:
                reward = 0.3  # Description or analysis task, higher reward
                message = f"ChartMoE completed {task_type} task"
            
            elif task_type == "extract_data":
                reward = 0.25  # Data extraction
                message = "ChartMoE extracted chart data"
            
            else:
                reward = 0.15  # Other task types
                message = f"ChartMoE completed task: {task_type}"
                
            return reward, False, message, {"tool_executed": "chartmoe", "task_type": task_type}
        
        # ⭐ New: Tool call awaiting response
        elif action_result.get("type") == "tool_call":
            # Tool call submitted, awaiting execution
            return 0.0, False, "Tool call submitted", {"tool_call_pending": True}
        
        # ⭐ New: Thinking process (if structured output is enabled)
        elif action_result.get("type") == "thinking":
            # Thinking step, no reward but allow continuation
            return 0.0, False, "Processing thoughts", {"thinking_step": True}
        
        # For other actions, grant intermediate reward
        elif action_result.get("status") == "SUCCESS":
            # Check if the action helps solve the task
            helpful_actions = ["analyze_image", "extract_text", "detect_objects", "request_info"]
            action_name = action_result.get("action", "")
            
            if any(helpful in action_name for helpful in helpful_actions):
                reward = 0.1  # Useful intermediate step
                message = f"Executed {action_name} successfully"
            else:
                reward = 0.05  # Neutral action
                message = f"Action {action_name} completed"
            
            return reward, False, message, {"intermediate_action": True}
        
        # ⭐ New: Handle errors that are not outright failures
        elif action_result.get("type") == "error" and action_result.get("error") != "Action failed":
            # Some errors may be recoverable (e.g., format errors)
            error_msg = action_result.get("error", "Unknown error")
            
            # Check if it's a reflection-related error
            if hasattr(self.task, 'enable_reflection') and self.task.enable_reflection:
                current_attempt = getattr(self.task, 'current_attempt', 0)
                max_attempts = getattr(self.task, 'max_attempts', 1)
                
                if current_attempt < max_attempts:
                    # Give opportunity to retry
                    return -0.05, False, f"Error: {error_msg}. Please try again.", {
                        "error": error_msg,
                        "recoverable": True,
                        "attempt": current_attempt
                    }
            
            # Unrecoverable error
            return -0.1, False, f"Error: {error_msg}", {"error": error_msg}
        
        else:
            # Other unknown cases
            return 0, False, "Action processed", {"unknown_action": True}
    
    def _load_current_image(self):
        """Load the current task's image"""
        if not self.task:
            return
        
        task_data = self.task.task_data
        image_path = task_data.get("image_path")
        
        if not image_path:
            logger.warning(f"No image path found for task {self.task.task_id}")
            self.current_image = None
            return
        
        image_path = Path(image_path)
        
        # If path is not absolute, make it relative to dataset path
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
        """Get the current observation"""
        obs = {
            "text": self.current_question,
            "question": self.current_question,  # Add question field for agent compatibility
            "step": self.current_step,
            "max_steps": self.max_steps,
            "chat_history": self.chat.get_history(),
            "has_image": self.current_image is not None,
            "image": self.current_image if self.current_image else None,
            "image_path": str(self.task.task_data.get("image_path", "")) if self.task else "",
            "available_tools": list(self.tool_manager.keys())  # Add available tools list
        }
        
        # ⭐ New: Include pending tool feedback
        if self.pending_tool_feedback is not None:
            obs["tool_feedback"] = self.pending_tool_feedback
            obs["requires_response"] = self.requires_tool_response
            
            # Debug output
            print(f"\n[DEBUG _get_obs] Including tool feedback in observation:")
            print(f"  - Tool feedback type: {type(self.pending_tool_feedback)}")
            if isinstance(self.pending_tool_feedback, dict):
                print(f"  - Tool: {self.pending_tool_feedback.get('tool', 'N/A')}")
                if self.pending_tool_feedback.get('tool') == 'grounding_dino':
                    print(f"  - Num detections: {self.pending_tool_feedback.get('num_detections', 'N/A')}")
                elif self.pending_tool_feedback.get('tool') == 'chartmoe':
                    print(f"  - Task type: {self.pending_tool_feedback.get('task_type', 'N/A')}")
                    print(f"  - Output preview: {str(self.pending_tool_feedback.get('output', ''))[:100]}")
                # DeepEyes debug info
                elif 'observation' in self.pending_tool_feedback:  # DeepEyes feedback
                    print(f"  - Has observation: True")
                    print(f"  - Has reward: {'reward' in self.pending_tool_feedback}")
            print(f"  - requires_response: {self.requires_tool_response}")

            # Clear feedback to avoid sending it repeatedly
            self.pending_tool_feedback = None
            self.requires_tool_response = False
        
        # Add action-related info
        if self.enable_actions:
            obs["available_actions"] = self.action_set.list_actions()
            obs["action_history"] = self.action_history
        
        # Add DeepEyes-related info
        # Initialize DeepEyes tool
        if self.enable_deepeyes_tools:
            try:
                if self.deepeyes_version == "v1":
                    from .tools.deepeyes import DeepEyesV1
                    self.deepeyes_tool = DeepEyesV1()
                    tool_name = "DeepEyesV1"
                else:  # v2
                    from .tools.mm_process.visual_toolbox_v2 import VisualToolBoxV2
                    self.deepeyes_tool = VisualToolBoxV2("visual_toolbox_v2", None, None)
                    tool_name = "visual_toolbox_v2"
                
                # Add detailed initialization log
                print(f"\n[DEBUG] DeepEyes tool initialization:")
                print(f"  - Tool initialized: {self.deepeyes_tool}")
                print(f"  - Tool type: {type(self.deepeyes_tool)}")
                print(f"  - Tool version: {self.deepeyes_version}")
                print(f"  - Tool name: {tool_name}")
                
                # Check tool attributes and methods
                if hasattr(self.deepeyes_tool, '__name__'):
                    print(f"  - Tool __name__: {self.deepeyes_tool.__name__}")
                if hasattr(self.deepeyes_tool, '__module__'):
                    print(f"  - Tool __module__: {self.deepeyes_tool.__module__}")

                self.deepeyes_initialized = True
                logger.info(f"DeepEyes tool initialized: {tool_name}")
                print(f"[DEBUG] ✓ DeepEyes initialization successful")
                
            except Exception as e:
                logger.error(f"Failed to initialize DeepEyes tools: {e}")
                print(f"\n[DEBUG] ❌ Failed to initialize DeepEyes: {e}")
                import traceback
                traceback.print_exc()
                self.deepeyes_tool = None
                self.deepeyes_initialized = False
        
        # Add Grounding DINO-related info
        if self.enable_grounding_dino:
            obs["grounding_dino_enabled"] = self.grounding_dino_tool is not None
        
        # Add ChartMoE-related info
        if self.enable_chartmoe:
            obs["chartmoe_enabled"] = self.chartmoe_tool is not None
        
        # Add task info
        if self.task:
            try:
                task_obs = self.task.get_observation()
                # Avoid overwriting key fields
                for key, value in task_obs.items():
                    if key not in ["text", "question", "image", "image_path"]:
                        obs[key] = value
            except Exception as e:
                logger.debug(f"Failed to get task observation: {e}")
        
        # Add time info
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            obs["elapsed_time"] = elapsed_time
            if self.time_limit:
                obs["remaining_time"] = max(0, self.time_limit - elapsed_time)
        
        return obs
    
    def close(self):
        """Close the environment and clean up resources"""
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
        self.deepeyes_initialized = False
        self.grounding_dino_tool = None
        self.chartmoe_tool = None
        
        # ⭐ Clean up tool feedback state
        self.pending_tool_feedback = None
        self.requires_tool_response = False
        
        logger.debug("VisionQAEnv closed")
