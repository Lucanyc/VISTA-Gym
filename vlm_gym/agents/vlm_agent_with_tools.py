
from typing import Dict, Any, Optional, List, Tuple
from .vlm_agent import VLMAgent
import json
import re
import os
from PIL import Image


class VLMAgentWithTools(VLMAgent):
    """
   
    """
    
    # 
    TOOL_INDEX_MAP = {
        0: {
            "name": "image_zoom_in_tool",
            "type": "visual_enhancement",
            "brief": "zoom into image regions",
            "requires_params": True
        },
        1: {
            "name": "grounding_dino",
            "type": "object_detection",
            "brief": "detect and count objects",
            "requires_params": True
        },
        2: {
            "name": "chartmoe",
            "type": "data_extraction",
            "brief": "extract chart data",
            "requires_params": True
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """"""
        # 
        tool_config_keys = ["enable_tools", "max_tool_calls", "tool_selection_strategy", 
                           "tool_response_mode", "deepeyes_prompt_style"]
        
        # 提取工具配置
        tool_config = {}
        base_config = {}
        
        for key, value in config.items():
            if key in tool_config_keys:
                tool_config[key] = value
            else:
                base_config[key] = value
        
        #
        super().__init__(base_config)
        
        #
        print(f"\n[VLMAgentWithTools.__init__] Initializing with config keys: {list(config.keys())}")
        
        #
        self.enable_tools = tool_config.get("enable_tools", True)
        self.max_tool_calls = tool_config.get("max_tool_calls", 5)
        self.tool_selection_strategy = tool_config.get("tool_selection_strategy", "adaptive")
        self.tool_response_mode = tool_config.get("tool_response_mode", "auto")
        
        print(f"  - enable_tools set to: {self.enable_tools}")
        print(f"  - max_tool_calls set to: {self.max_tool_calls}")
        print(f"  - tool_selection_strategy set to: {self.tool_selection_strategy}")
        print(f"  - tool_response_mode set to: {self.tool_response_mode}")
        
        #
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
        
        # 
        self.tool_performance = {
            idx: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "avg_improvement": 0.0,
                "contexts": {}  
            }
            for idx in self.TOOL_INDEX_MAP.keys()
        }
        
        # 
        self.tool_use_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        return (action_string, extra_info_dict)
        """
        # 添加调试输出
        print(f"\n[VLMAgentWithTools.act] DEBUG START")
        print(f"  - Observation keys: {list(observation.keys())}")
        print(f"  - enable_tools: {self.enable_tools}")
        print(f"  - Current tool calls: {self.current_tool_calls}")
        
        # 
        if observation.get("requires_response") and "tool_feedback" in observation:
            print(f"  - Handling tool feedback from environment")
            return self._handle_tool_feedback(observation)
        
        #
        if observation.get("episode_start", False) or self._is_new_task(observation):
            self.current_tool_calls = 0
            self.conversation_history = []
            print(f"  - Starting new episode, reset tool call count")
        
        # 
        self.conversation_history.append({"role": "observation", "content": observation})
        
        # 
        available_tools = self._get_available_tools(observation)
        print(f"  - Available tools: {available_tools}")
        
        # 
        if not self.enable_tools or not available_tools:
            print(f"  - Tools disabled or no tools available, generating direct answer")
            return self._generate_direct_answer(observation)
        
        # 
        if self.current_tool_calls >= self.max_tool_calls:
            print(f"  - Reached max tool calls ({self.max_tool_calls}), forcing final answer")
            return self._generate_forced_final_answer(observation)
        
        # 
        must_use_tool = observation.get("must_use_tool", False)
        previous_failed = observation.get("previous_attempt_failed", False)
        chartmoe_enabled = observation.get("chartmoe_enabled", False)
        
        if must_use_tool or (previous_failed and self.current_tool_calls == 0):
            print(f"  - Previous attempt failed, FORCING tool use")
            print(f"  - must_use_tool: {must_use_tool}, previous_failed: {previous_failed}")
            
            # 
            if chartmoe_enabled and 2 in available_tools:  #chartmoe with index 1
                print(f"  - Forcing ChartMoE tool call")
                action = '<tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>'
                tool_name = "chartmoe"
            elif 1 in available_tools:  # Grounding DINO with index 2 1
                print(f"  - Forcing Grounding DINO tool call")
                action = '<tool_call>{"tool": "grounding_dino", "parameters": {"caption": "chart data values"}}</tool_call>'
                tool_name = "grounding_dino"
            elif 0 in available_tools:  # DeepEyes with index 0
                print(f"  - Forcing DeepEyes tool call")
                action = '<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500]}}</tool_call>'
                tool_name = "deepeyes"
            else:
                print(f"  - No suitable tool available for forcing")
                
                response, extra_info = self._generate_vlm_action(observation)
                return response, extra_info
            
            #
            self.current_tool_calls += 1
            
            # 
            extra_info = {
                "action_type": "tool_call",
                "tool_call_count": self.current_tool_calls,
                "forced": True,
                "reason": "previous_attempt_failed",
                "tool_used": tool_name
            }
            
            # 
            self._analyze_response(action)
            
            print(f"  - Forced tool call generated: {tool_name}")
            print(f"[VLMAgentWithTools.act] DEBUG END\n")
            
            return action, extra_info
        
        print(f"  - Generating response with tool capability")
        
        # 
        response, extra_info = self._generate_vlm_action(observation)
        
        # 
        self._analyze_response(response)
        
        print(f"[VLMAgentWithTools.act] DEBUG END\n")
        
        return response, extra_info
    
    def _get_available_tools(self, observation: Dict[str, Any]) -> List[int]:
        """"""
        available = []
        
        # 
        if observation.get("deepeyes_enabled", False):
            available.append(0)  # image_zoom_in_tool
        
        if observation.get("grounding_dino_enabled", False):
            available.append(1)  # grounding_dino
            
        if observation.get("chartmoe_enabled", False):
            available.append(2)  # chartmoe
        
        return available
    
    def _generate_vlm_action(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """"""
        
        print(f"\n{'='*60}")
        print(f"[DEBUG VLMAgentWithTools._generate_vlm_action] START")
        print(f"  - Question: {observation.get('question', '')[:100]}...")
        print(f"  - Image path: {observation.get('image_path', 'N/A')}")
        print(f"  - Tool calls so far: {self.current_tool_calls}/{self.max_tool_calls}")
        print(f"{'='*60}")
        
        # 
        max_retries = 2
        retry_count = 0
        formatted_action = None
        
        while retry_count <= max_retries:
            #
            print(f"\n[DEBUG] Building tool-aware prompt (attempt {retry_count + 1})...")
            enhanced_prompt = self._build_tool_aware_prompt(observation, retry_count > 0)
            print(f"[DEBUG] Enhanced prompt (first 300 chars):\n{enhanced_prompt[:300]}...")
            
            # 
            enhanced_observation = observation.copy()
            existing_instruction = observation.get('output_format_instruction', '')
            enhanced_observation["output_format_instruction"] = enhanced_prompt + "\n\n" + existing_instruction
            
            #
            print(f"\n[DEBUG] Calling parent VLM to generate response...")
            vlm_response, base_info = super().act(enhanced_observation)
            
            print(f"\n[DEBUG] Raw VLM response (first 500 chars):")
            print(f"{vlm_response[:500]}...")
            
            # 
            formatted_action = self._validate_and_format_response(vlm_response, observation)
            
            # 
            if formatted_action == "NEED_REGENERATION" and retry_count < max_retries:
                print(f"[DEBUG] Response needs regeneration")
                retry_count += 1
                observation["retry_context"] = self._get_retry_context(vlm_response)
                continue
            
            break
        
        # 
        if formatted_action == "NEED_REGENERATION":
            print(f"[DEBUG] Failed to generate valid tool call, formatting as answer")
            formatted_action = self._format_as_answer(vlm_response)
        
        #
        self._update_action_stats(formatted_action)
        
        extra_info = base_info.copy()
        extra_info.update({
            "action_type": self._get_action_type(formatted_action),
            "tool_call_count": self.current_tool_calls,
            "retry_count": retry_count
        })
        
        print(f"\n[DEBUG] Summary:")
        print(f"  - Action type: {extra_info['action_type']}")
        print(f"  - Tool calls so far: {self.current_tool_calls}")
        print(f"{'='*60}")
        
        return formatted_action, extra_info
    
    def _build_tool_aware_prompt(self, observation: Dict[str, Any], is_retry: bool = False) -> str:
        """"""
        question = observation.get("question", "")
        available_tools = self._get_available_tools(observation)
        
        # 
        is_retry_after_failure = observation.get("previous_attempt_failed", False)
        
        # 
        is_first_attempt = observation.get("attempt", 1) == 1
        
        # 
        question_lower = question.lower()
        is_numerical_question = any(word in question_lower for word in [
            'how many', 'count', 'total', 'sum', 'average', 'percentage', 'percent',
            'difference', 'ratio', 'how much', 'what is the value', 'number'
        ])
        
        # 
        if is_first_attempt and 2 in available_tools and not is_retry:  # 
            prompt = f"""You are analyzing a VISUAL CHART/GRAPH to answer: "{question}"

    ⚠️ CRITICAL: This is your FIRST attempt at a visual chart question. You MUST use ChartMoE tool to ensure accuracy!

    DO NOT try to answer directly by looking at the chart. Charts can be misleading and values hard to read precisely.

    Your response MUST start with the following tool call:

    <tool_call>
    {{"tool": "chartmoe", "task": "to_table"}}
    </tool_call>

    This will extract all data from the chart in a structured table format. After receiving the results, you can analyze the data to answer the question accurately.

    Available tools:
    """
            #
            for idx in available_tools:
                tool_info = self.TOOL_INDEX_MAP[idx]
                prompt += f"- Tool {idx}: {tool_info['brief']} (type: {tool_info['type']})\n"
            
            # 
            prompt += """
    ChartMoE Tasks Guide:
    - "to_table": Extract all data as a structured table (RECOMMENDED for numerical questions)
    - "describe": Get detailed chart description
    - "extract_data": Extract specific numerical values
    - "summarize": Get chart summary
    - "analyze": Deep analysis with insights
    - "compare": Compare data series
    - "trend": Identify trends
    - Custom: {"tool": "chartmoe", "prompt": "your specific question"}

    Remember: Start with <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call> FIRST!
    """
        
        #
        elif is_retry_after_failure:
            prompt = f"""You are analyzing a VISUAL CHART/GRAPH to answer: "{question}"

    ⚠️ CRITICAL: Your previous answer was WRONG! You MUST use tools this time to extract accurate data!

    """
            
            if 2 in available_tools:  # ChartMoE
                prompt += """You MUST use ChartMoE tool because your previous visual reading was incorrect!

    Start with:
    <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>

    This will give you the exact data values from the chart.

    """
            elif 1 in available_tools:  # Grounding DINO
                prompt += """You MUST use Grounding DINO tool to detect and count objects accurately!

    Start with:
    <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "chart data values"}}</tool_call>

    """
            
            prompt += "Available tools:\n"
            for idx in available_tools:
                tool_info = self.TOOL_INDEX_MAP[idx]
                prompt += f"- Tool {idx}: {tool_info['brief']} (type: {tool_info['type']})\n"
        
        # 
        elif is_numerical_question and not is_retry:
            prompt = f"""You are analyzing a VISUAL CHART/GRAPH to answer: "{question}"

    📊 This is a NUMERICAL QUESTION that requires precise data extraction from the chart.
    Visual estimation often leads to errors. You SHOULD use tools for accuracy!

    Available tools:
    """
            for idx in available_tools:
                tool_info = self.TOOL_INDEX_MAP[idx]
                prompt += f"- Tool {idx}: {tool_info['brief']} (type: {tool_info['type']})\n"
            
            prompt += "\n"
            
            # 
            if 2 in available_tools:  # ChartMoE
                prompt += """STRONGLY RECOMMENDED for numerical questions:
    <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>

    This extracts all chart data into a table format, making calculations accurate and easy.

    Other ChartMoE tasks:
    - "extract_data": For specific values
    - "analyze": For detailed analysis
    - "compare": For comparisons
    - Custom prompt: {"tool": "chartmoe", "prompt": "your question"}
    """
            elif 1 in available_tools:  # Grounding DINO
                prompt += """For counting/detection:
    <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "bars" or "data points"}}</tool_call>
    """
        
        #
        else:
            prompt = f"""You are analyzing a VISUAL CHART/GRAPH to answer: "{question}"

    This is a VISUAL QUESTION that requires analyzing chart data.
    {"⚠️ Your previous answer was incorrect. Consider using tools for better accuracy!" if is_retry else "Tools are available if you need precise data extraction."}

    Available tools (select by index if needed):
    """
            # 
            for idx in available_tools:
                tool_info = self.TOOL_INDEX_MAP[idx]
                prompt += f"- Tool {idx}: {tool_info['brief']} (type: {tool_info['type']})\n"
            
            #
            performance_hint = self._get_performance_hint(available_tools, observation)
            if performance_hint:
                prompt += f"\n{performance_hint}\n"
        
        # 
        if is_retry and not is_retry_after_failure:
            retry_context = observation.get("retry_context", {})
            prompt += self._build_retry_prompt(retry_context)
        
        
        if not is_first_attempt and not is_retry_after_failure:
            prompt += """
    Tool Usage Examples:

    For ChartMoE (Tool 2):
    - Extract table: <tool_call>{"tool": "chartmoe", "task": "to_table"}</tool_call>
    - Describe chart: <tool_call>{"tool": "chartmoe", "task": "describe"}</tool_call>
    - Custom question: <tool_call>{"tool": "chartmoe", "prompt": "What is the trend?"}</tool_call>

    For Grounding DINO (Tool 1):
    - Detect objects: <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "bars"}}</tool_call>
    - Count items: <tool_call>{"tool": "grounding_dino", "parameters": {"caption": "data points"}}</tool_call>

    You can choose to use tools or answer directly based on your confidence.
    """
        
        return prompt
    
    
    def _build_retry_prompt(self, retry_context: Dict[str, Any]) -> str:
        
        missing_tool = retry_context.get("missing_tool", "")
        missing_params = retry_context.get("missing_params", [])
        
        prompt = f"""
Your previous response was incomplete.
"""
        
        if missing_tool == "image_zoom_in_tool" and "bbox_2d" in missing_params:
            prompt += """
You selected the zoom tool but didn't provide coordinates.
You MUST specify where to zoom by providing bbox_2d: [x1, y1, x2, y2]
Based on the chart content, calculate the actual coordinates.
"""
        elif missing_tool == "chartmoe" and missing_params:
            prompt += """
You selected ChartMoE but didn't provide required parameters.
You must provide either:
- "task": one of ["to_table", "describe", "extract_data", "summarize", "analyze", "compare", "trend"]
- OR "prompt": a specific question about the chart
"""
        elif missing_params:
            prompt += f"""
Missing required parameters: {', '.join(missing_params)}
Please provide a complete tool call with all required parameters.
"""
        
        prompt += """
Try again with a complete response:
"""
        
        return prompt
    
    def _validate_and_format_response(self, response: str, observation: Dict[str, Any]) -> str:
        
        
     
        if "<answer>" in response and "</answer>" in response:
            return response
        
   
        if "<tool_call>" in response:
        
            validation_result = self._validate_tool_call(response)
            if validation_result["valid"]:
                self.current_tool_calls += 1
                return response
            else:
                print(f"[DEBUG] Invalid tool call: {validation_result['reason']}")
                return "NEED_REGENERATION"
        
        
        if "<tool_selection>" in response:
            print(f"[DEBUG] Tool selection without complete call")
            return "NEED_REGENERATION"
        
        
        return self._format_as_answer(response)
    
    def _validate_tool_call(self, response: str) -> Dict[str, Any]:
     
        try:
            tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
            if not tool_match:
                return {"valid": False, "reason": "No tool_call tags found"}
            
            tool_json = json.loads(tool_match.group(1).strip())
            
     
            if tool_json.get("name") == "image_zoom_in_tool":
                bbox = tool_json.get("arguments", {}).get("bbox_2d")
                if not bbox or len(bbox) != 4:
                    return {"valid": False, "reason": "Missing or invalid bbox_2d"}
                    
            elif tool_json.get("tool") == "grounding_dino":
                caption = tool_json.get("parameters", {}).get("caption")
                if not caption:
                    return {"valid": False, "reason": "Missing caption"}
                    
            elif tool_json.get("tool") == "chartmoe":


                has_task = "task" in tool_json
                has_prompt = "prompt" in tool_json
                
                if not has_task and not has_prompt:
                    # 检查是否在 parameters 中
                    params = tool_json.get("parameters", {})
                    has_task = "task" in params
                    has_prompt = "prompt" in params
                    
                    if not has_task and not has_prompt:
                        return {"valid": False, "reason": "ChartMoE requires either 'task' or 'prompt'"}
                
   
                if has_task:
                    task = tool_json.get("task") or tool_json.get("parameters", {}).get("task")
                    valid_tasks = ["to_table", "describe", "extract_data", "summarize", "analyze", "compare", "trend"]
                    if task not in valid_tasks:
                        return {"valid": False, "reason": f"Invalid task '{task}'. Valid tasks: {valid_tasks}"}
            
            return {"valid": True}
            
        except json.JSONDecodeError:
            return {"valid": False, "reason": "Invalid JSON format"}
        except Exception as e:
            return {"valid": False, "reason": str(e)}
    
    def _get_retry_context(self, response: str) -> Dict[str, Any]:

        context = {}
        

        if "image_zoom_in_tool" in response and "bbox_2d" not in response:
            context["missing_tool"] = "image_zoom_in_tool"
            context["missing_params"] = ["bbox_2d"]
        elif "chartmoe" in response.lower():

            if "task" not in response and "prompt" not in response:
                context["missing_tool"] = "chartmoe"
                context["missing_params"] = ["task or prompt"]
        
        return context
    
    def _get_performance_hint(self, available_tools: List[int], observation: Dict[str, Any]) -> str:


        total_attempts = sum(self.tool_performance[idx]["attempts"] for idx in available_tools)
        if total_attempts < 10:
            return ""
        

        best_tool = None
        best_score = -1
        
        for idx in available_tools:
            perf = self.tool_performance[idx]
            if perf["attempts"] > 0:
                success_rate = perf["successes"] / perf["attempts"]
                if success_rate > best_score:
                    best_score = success_rate
                    best_tool = idx
        
        if best_tool is not None and best_score > 0.6:
            return f"Historical data suggests Tool {best_tool} has been effective ({best_score:.1%} success rate)"
        
        return ""
    
    def _update_action_stats(self, action: str):

        self.tool_use_stats["total_calls"] += 1
        
        if "<tool_call>" in action:

            pass
    
    def _get_action_type(self, action: str) -> str:

        if "<tool_call>" in action:
            return "tool_call"
        elif "<answer>" in action:
            return "final_answer"
        else:
            return "direct_response"
    
    def _analyze_response(self, response: str):

        if "<tool_call>" in response:
            try:
                tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
                if tool_match:
                    tool_data = json.loads(tool_match.group(1).strip())
                    print(f"  ✓ Tool call detected: {json.dumps(tool_data, indent=2)}")
            except:
                print(f"  ❌ Failed to parse tool call")
    
    def _handle_tool_feedback(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:

        print(f"\n[_handle_tool_feedback] Processing tool feedback")
        
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        original_question = observation.get("question", "")
        

        if tool_name == "chartmoe":
            feedback_prompt = self._build_chartmoe_feedback_prompt(tool_feedback, original_question)
        elif tool_name == "grounding_dino":
            feedback_prompt = self._build_grounding_feedback_prompt(tool_feedback, original_question)
        else:

            feedback_prompt = f"""Based on the tool results, answer the original question: "{original_question}"

Provide your answer in <answer>...</answer> tags."""
        

        enhanced_observation = observation.copy()
        enhanced_observation["output_format_instruction"] = feedback_prompt
        

        response, base_info = super().act(enhanced_observation)
        

        if "<answer>" not in response:
            formatted_response = self._format_as_answer(response)
        else:
            formatted_response = response
        

        self._record_tool_result(observation, formatted_response)
        
        extra_info = base_info.copy()
        extra_info["action_type"] = "tool_feedback_response"
        
        return formatted_response, extra_info
    
    def _build_chartmoe_feedback_prompt(self, tool_feedback: Dict[str, Any], original_question: str) -> str:

        task_type = tool_feedback.get("task_type", "unknown")
        output = tool_feedback.get("output", "")
        
        prompt = f"""ChartMoE has analyzed the chart with task '{task_type}'.

Result:
{output[:1000]}...

Based on this analysis, answer the original question: "{original_question}"

Provide your answer in <answer>...</answer> tags."""
        
        return prompt
    
    def _build_grounding_feedback_prompt(self, tool_feedback: Dict[str, Any], original_question: str) -> str:
        """"""
        num_detections = tool_feedback.get("num_detections", 0)
        query = tool_feedback.get("query", "")
        
        prompt = f"""Grounding DINO detected {num_detections} objects matching "{query}".

Based on this detection result, answer the original question: "{original_question}"

Provide your answer in <answer>...</answer> tags."""
        
        return prompt
    
    def _record_tool_result(self, observation: Dict[str, Any], response: str):
        """"""
        #
        tool_feedback = observation.get("tool_feedback", {})
        tool_name = tool_feedback.get("tool", "unknown")
        

        tool_idx = None
        for idx, tool_info in self.TOOL_INDEX_MAP.items():
            if tool_info["name"] == tool_name or tool_name in tool_info["name"]:
                tool_idx = idx
                break
        
        if tool_idx is not None:
            self.tool_performance[tool_idx]["attempts"] += 1

            if "<answer>" in response:
                self.tool_performance[tool_idx]["successes"] += 1
            else:
                self.tool_performance[tool_idx]["failures"] += 1
    
    def _generate_direct_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:

        answer, base_info = super().act(observation)
        formatted_action = self._format_as_answer(answer)
        
        extra_info = base_info.copy()
        extra_info["action_type"] = "direct_answer"
        extra_info["tool_call_count"] = 0
        
        return formatted_action, extra_info
    
    def _generate_forced_final_answer(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:

        enhanced_observation = observation.copy()
        prompt = "You have used the maximum number of tools. Provide your final answer in <answer>...</answer> tags."
        enhanced_observation["output_format_instruction"] = prompt
        
        answer, base_info = super().act(enhanced_observation)
        formatted_action = self._format_as_answer(answer)
        
        extra_info = base_info.copy()
        extra_info["action_type"] = "forced_final_answer"
        
        return formatted_action, extra_info
    
    def _format_as_answer(self, response: str) -> str:
        """"""
        response = response.strip()
        if "<answer>" in response:
            return response
        return f"<answer>{response}</answer>"
    
    def _is_new_task(self, observation: Dict[str, Any]) -> bool:
        """"""
        if not self.conversation_history:
            return True
        
        # 
        current_question = observation.get("question")
        for hist in reversed(self.conversation_history):
            if hist["role"] == "observation" and isinstance(hist["content"], dict):
                last_question = hist["content"].get("question")
                return current_question != last_question
        
        return True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """"""
        report = {
            "overall_stats": self.tool_use_stats,
            "tool_performance": {}
        }
        
        for idx, perf in self.tool_performance.items():
            if perf["attempts"] > 0:
                tool_name = self.TOOL_INDEX_MAP[idx]["name"]
                report["tool_performance"][tool_name] = {
                    "attempts": perf["attempts"],
                    "success_rate": perf["successes"] / perf["attempts"] if perf["attempts"] > 0 else 0,
                    "failure_rate": perf["failures"] / perf["attempts"] if perf["attempts"] > 0 else 0
                }
        
        return report
    
    def reset(self):
        """"""
        super().reset()
        self.tool_history = []
        self.current_tool_calls = 0
        self.conversation_history = []
       
